import argparse
import os
import time as _time

import numpy as np
import torch

from root.checkpoint_io import load_checkpoint
from root.data.ingest import download_tess_lightcurve, list_available_sectors
from root.data.synthetic_data import LightCurveSimulator
from root.preprocessing.clean import clean_and_flatten
from root.detection.bls_detector import run_bls
from root.detection.dual_stream import extract_dual_views
from root.classification.tri_branch_ensemble import TriBranchTESSNet
from root.classification.xai import compute_1d_gradcam
from root.classification.physical_verification import verify_astrophysics
from root.reporting.pdf_report import generate_vetting_report
from root.status import DEFAULT_MIN_BLS_SNR, derive_overall_status

# A light curve with fewer valid cadences than this cannot support a BLS
# search, a phase-folded view or any vetting; fail loudly instead of
# returning a meaningless score.
MIN_VALID_CADENCES = 100

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
DEFAULT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "tri_branch_tess_net.pt")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


class Pipeline:
    """
    Owns the (comparatively expensive to construct) TriBranchTESSNet so a
    caller processing multiple targets in one session (e.g. the
    dashboard) can build ONE Pipeline and call run()/run_from_arrays()
    repeatedly instead of re-loading weights on every call.

    If no trained weights are found at `weights_path`, the model runs
    with its randomly-initialized weights, and every result this
    Pipeline returns is tagged `weights_loaded=False`. Callers MUST
    surface that prominently -- an untrained network's output here is
    noise, not a classification. Run train_all.py to produce real
    weights.
    """

    def __init__(self, weights_path=DEFAULT_WEIGHTS_PATH, tabular_dim=6, min_bls_snr=DEFAULT_MIN_BLS_SNR):
        self.model = TriBranchTESSNet()
        self.min_bls_snr = float(min_bls_snr)
        self.weights_path = weights_path
        self.weights_loaded = False
        self.weights_present = bool(weights_path and os.path.exists(weights_path))
        self.checkpoint_temperature = 1.0
        if weights_path and os.path.exists(weights_path):
            checkpoint = load_checkpoint(weights_path)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
                format_version = int(checkpoint.get("format_version", 0))
                self.checkpoint_temperature = float(checkpoint.get("temperature", 1.0))
            else:
                state = checkpoint
                format_version = 0  # legacy raw state_dict

            load_result = self.model.load_state_dict(state, strict=False)
            if load_result.missing_keys:
                raise RuntimeError(f"checkpoint is incompatible; missing keys: {load_result.missing_keys}")

            if format_version >= 5:
                self.weights_loaded = True
                print(f"   Loaded v9 real-TESS morphology weights from '{weights_path}' (temperature={self.checkpoint_temperature:.4f}).")
            elif format_version >= 2:
                self.weights_loaded = True
                print(f"   Loaded validated-format TriBranchTESSNet weights from '{weights_path}'.")
            else:
                # The bundled checkpoint predates the normalized-view +
                # inference-safe BatchNorm preprocessing contract. Loading its
                # tensors is useful for inspection, but it must not be presented
                # as a validated classifier until train_all.py is rerun.
                self.weights_loaded = False
                print(f"   Loaded legacy weights from '{weights_path}', but marked them UNVALIDATED "
                      "because the preprocessing/model contract changed. Retrain with `python -m root.train_all`." )

            if load_result.unexpected_keys:
                print("   Ignored obsolete checkpoint keys: " + ", ".join(load_result.unexpected_keys))
        else:
            print(f"   WARNING: no trained weights found at '{weights_path}'. TriBranchTESSNet "
                  "is running with its randomly-initialized weights -- its output is NOT a "
                  "learned assessment of this candidate. Run `python -m root.train_all` first "
                  "to produce real weights.")
        self.model.eval()

    def _run_common(self, time, flux, flux_err, target_id, r_star, teff, logg, generate_report):
        t_clean, f_clean, err_clean = clean_and_flatten(time, flux, flux_err)
        if len(t_clean) < MIN_VALID_CADENCES:
            raise ValueError(
                f"{target_id}: only {len(t_clean)} valid cadences after cleaning "
                f"(need at least {MIN_VALID_CADENCES}); cannot run a transit search."
            )
        bls_res = run_bls(t_clean, f_clean, err_clean)
        g_view, l_view, view_meta = extract_dual_views(
            t_clean, f_clean, bls_res["period"], bls_res["t0"], bls_res["duration"],
            return_meta=True,
        )

        g_tensor = torch.tensor(g_view, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        l_tensor = torch.tensor(l_view, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        phys_res = verify_astrophysics(
            t_clean, f_clean, bls_res["period"], bls_res["t0"], bls_res["duration"], r_star,
        )
        # The v9+ model is morphology-only and ignores tabular input.
        tab_tensor = None

        with torch.no_grad():
            logits = self.model(g_tensor, l_tensor, tab_tensor)
            temperature = 1.0
            if getattr(self, "checkpoint_temperature", None) is not None:
                temperature = max(float(self.checkpoint_temperature), 1e-3)
            prob = torch.softmax(logits / temperature, dim=1)[0, 1].item()

        saliency = None
        try:
            saliency = compute_1d_gradcam(self.model, l_tensor, g_tensor, tab_tensor)
        except Exception as e:
            print(f"   Grad-CAM failed ({type(e).__name__}: {e}); continuing without a saliency map.")

        # A BLS peak below the noise floor, a peak at the search boundary and
        # a period supported by fewer than three distinct transit events are
        # each insufficient for a periodic detection. AI probability remains
        # available as a model score, but it is not promoted to a final
        # candidate status. (Logic lives in root/status.py so the PNG report
        # and the dashboard can never disagree with it.)
        overall_status = derive_overall_status(bls_res, phys_res, self.min_bls_snr)

        result = {
            "target_id": target_id,
            "weights_loaded": self.weights_loaded,
            "weights_present": self.weights_present,
            "light_curve": {"time_raw": time, "flux_raw": flux, "time_clean": t_clean, "flux_clean": f_clean},
            "bls": bls_res,
            "views": {"global_view": g_view, "local_view": l_view, "saliency": saliency, **view_meta},
            "ai": {"planet_probability": prob, "planet_score": prob},
            "verification": phys_res,
            "overall_status": overall_status,
            "stellar": {"r_star": r_star, "teff": teff, "logg": logg},
        }

        if generate_report:
            report_path = os.path.join(REPORTS_DIR, f"{target_id}_report.png")
            generate_vetting_report(
                target_id, bls_res, phys_res, prob,
                views=result["views"], weights_loaded=self.weights_loaded,
                output_path=report_path, overall_status=overall_status,
            )
            result["report_file"] = report_path

        return result

    def run(self, tic_id, sector=None, r_star=1.0, teff=5778.0, logg=4.44, generate_report=True):
        """Fetch a real TESS light curve for `tic_id` from MAST and run the full pipeline on it."""
        time, flux, flux_err = download_tess_lightcurve(tic_id, sector=sector)
        return self._run_common(time, flux, flux_err, f"TIC_{tic_id}", r_star, teff, logg, generate_report)

    def run_from_arrays(self, time, flux, target_id="synthetic", r_star=1.0, teff=5778.0, logg=4.44,
                         generate_report=False):
        """
        Run the full pipeline on an already-in-memory (time, flux) light
        curve -- e.g. a synthetic sample -- instead of fetching one from
        MAST. There's no real per-cadence flux_err for synthetic data, so
        clean_and_flatten falls back to its default placeholder.
        """
        return self._run_common(time, flux, None, target_id, r_star, teff, logg, generate_report)


def run_pipeline(tic_id, sector=None, r_star=1.0, teff=5778.0, logg=4.44, weights_path=DEFAULT_WEIGHTS_PATH,
                 min_bls_snr=DEFAULT_MIN_BLS_SNR):
    """CLI convenience entry point for a single, one-off run against a real TESS target."""
    print(f"\n=== TESS Planet Detection Pipeline: TIC {tic_id} ===")
    t_stage = _time.time()
    pipeline = Pipeline(weights_path=weights_path, min_bls_snr=min_bls_snr)
    result = pipeline.run(tic_id, sector=sector, r_star=r_star, teff=teff, logg=logg)
    bls_res = result["bls"]
    print(f"Candidate -> P: {bls_res['period']:.4f}d, SNR: {bls_res['snr']:.1f}, "
          f"period quality: {bls_res.get('period_quality', 'UNKNOWN')}")
    print(f"Transit events: {result['verification'].get('observed_transits', 0)} observed / "
          f"{result['verification'].get('expected_transits', 0)} expected; "
          f"period validation: {result['verification'].get('period_validation_status', 'UNKNOWN')}")
    print(f"Overall status: {result.get('overall_status', 'UNKNOWN')}")
    if result["weights_loaded"]:
        tag = ""
    elif result.get("weights_present"):
        tag = "  [LEGACY WEIGHTS -- UNVALIDATED; retrain before using confidence]"
    else:
        tag = "  [UNTRAINED WEIGHTS -- not a meaningful assessment]"
    if result["weights_loaded"]:
        print(f"AI Planet Score: {result['ai']['planet_probability'] * 100:.2f}%")
    else:
        print("AI Planet Confidence: N/A" + tag)
    print(f"Done in {_time.time() - t_stage:.1f}s. Report saved to: {result.get('report_file')}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TESS exoplanet detection pipeline on a real target.")
    parser.add_argument("--tic", type=int, default=383715654, help="Target TIC ID")
    parser.add_argument("--sector", type=int, default=None, help="TESS Sector (default: first available sector)")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH,
                         help="Path to trained TriBranchTESSNet weights (see train_all.py).")
    parser.add_argument("--min-snr", type=float, default=DEFAULT_MIN_BLS_SNR,
                         help="BLS SNR below which a target is reported as NO_SIGNIFICANT_DETECTION.")
    parser.add_argument("--list-sectors", action="store_true",
                         help="Just query MAST for which sectors/authors have data for --tic, "
                              "print them, and exit -- no download, no model run. Use this "
                              "before picking a --sector value.")
    args = parser.parse_args()

    if args.list_sectors:
        print(f"Querying MAST for available sectors on TIC {args.tic}...")
        rows = list_available_sectors(args.tic)
        if not rows:
            print(f"No TESS data found for TIC {args.tic} on MAST at all (any sector/author).")
        else:
            print(f"Found {len(rows)} product(s):")
            for r in rows:
                print(f"   sector={r['sector']}  author={r['author']}  exptime={r['exptime']}s")
            real_sectors = sorted({r["sector"] for r in rows if r["sector"] is not None})
            if real_sectors:
                print(f"\nTry: python -m root.main_pipeline --tic {args.tic} --sector {real_sectors[0]}")
        raise SystemExit(0)

    run_pipeline(args.tic, sector=args.sector, weights_path=args.weights, min_bls_snr=args.min_snr)
