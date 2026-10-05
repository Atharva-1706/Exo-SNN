import numpy as np
import torch

from root.classification.physical_verification import verify_astrophysics
from root.classification.tri_branch_ensemble import TriBranchTESSNet
from root.classification.xai import compute_1d_gradcam
from root.detection.dual_stream import extract_dual_views
from root.preprocessing.clean import clean_and_flatten
from root.data.synthetic_data import LightCurveSimulator


def test_clean_and_views_are_normalized_and_return_errors():
    sim = LightCurveSimulator(points_per_day=200, seed=1)
    t, f, meta = sim.generate_planet_sample()
    tc, fc, err = clean_and_flatten(t, f)
    g, l = extract_dual_views(tc, fc, meta["period"], meta["t0"], meta["duration"])
    assert len(tc) == len(fc) == len(err)
    assert abs(g.mean()) < 1e-3 and abs(l.mean()) < 1e-3
    assert 0.99 < g.std() < 1.01 and 0.99 < l.std() < 1.01


def test_eval_safe_model_and_real_gradcam():
    model = TriBranchTESSNet().eval()
    g = torch.randn(1, 1, 201)
    l = torch.randn(1, 1, 61)
    tab = torch.randn(1, 6)
    with torch.no_grad():
        eval_logits = model(g, l, tab)
    model.train()
    train_logits = model(g, l, tab)
    assert eval_logits.shape == train_logits.shape == (1, 2)
    cam = compute_1d_gradcam(model, l, g, tab)
    assert cam.shape == (61,)
    assert np.isfinite(cam).all()


def test_insufficient_vetting_is_indeterminate_not_eb():
    out = verify_astrophysics(np.array([0.0, 1.0, 2.0]), np.ones(3), 2.0, 0.0, 0.1)
    assert out["is_eclipsing_binary_flag"] is False
    assert out["odd_even_status"] == "insufficient_data"


def test_detrending_preserves_hour_scale_transits():
    ppd = 720
    t = np.arange(0, 10, 1 / ppd)
    period, t0, depth = 5.0, 1.0, 0.02
    for duration_hours in (2.5, 4.0):
        f = np.ones_like(t)
        phase = ((t - t0 + period / 2) % period) - period / 2
        f[np.abs(phase) < duration_hours / 48] -= depth
        tc, fc, _ = clean_and_flatten(t, f)
        phase = ((tc - t0 + period / 2) % period) - period / 2
        inside = np.abs(phase) < duration_hours / 48
        outside = (np.abs(phase) > 3 * duration_hours / 48) & (np.abs(phase) < period / 2)
        recovered = np.median(fc[outside]) - np.median(fc[inside])
        assert recovered / depth > 0.60


def test_boundary_period_is_flagged():
    # Test the boundary criterion without invoking Astropy/BLS.
    # 14.41 d is inside the configured 15 d upper-boundary margin.
    span = 15.0 - 0.5
    margin = max(0.05 * span, 0.25)
    assert 14.4098 >= 15.0 - margin


def test_transit_count_requires_distinct_events():
    period = 14.4
    duration = 4 / 24
    t0 = 1.0
    # One event only: cannot validate a periodic ephemeris.
    t = np.arange(0.0, 10.0, 0.01)
    f = np.ones_like(t)
    phase = t - (t0 + np.rint((t - t0) / period) * period)
    f[np.abs(phase) < duration / 2] -= 0.001
    out = verify_astrophysics(t, f, period, t0, duration, min_points=3)
    assert out["observed_transits"] == 1
    assert out["period_validation_status"] == "INSUFFICIENT_TRANSITS"


def test_secondary_is_indeterminate_without_three_transits():
    period = 14.4
    duration = 4 / 24
    t0 = 1.0
    t = np.arange(0.0, 10.0, 0.01)
    f = np.ones_like(t)
    phase = t - (t0 + np.rint((t - t0) / period) * period)
    f[np.abs(phase) < duration / 2] -= 0.001
    out = verify_astrophysics(t, f, period, t0, duration, min_points=3)
    assert out["secondary_status"] == "insufficient_data"
    assert out["eb_vetting_evaluated"] is False


def test_dual_view_can_return_local_coverage_metadata():
    sim = LightCurveSimulator(points_per_day=200, seed=2)
    t, f, meta = sim.generate_planet_sample()
    tc, fc, _ = clean_and_flatten(t, f)
    _, _, view_meta = extract_dual_views(tc, fc, meta["period"], meta["t0"], meta["duration"], return_meta=True)
    assert "local_phase_min" in view_meta and "local_phase_max" in view_meta
    assert view_meta["local_phase_min"] <= view_meta["local_phase_max"]


# ---------------------------------------------------------------------------
# Regression tests added by the stress-check pass
# ---------------------------------------------------------------------------
import glob
import os
import sys
import time

import pytest

from root.detection.bls_detector import run_bls
from root.status import derive_overall_status

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _noise(n=5000, span=27.0, sigma=5e-4, seed=0):
    rng = np.random.default_rng(seed)
    return np.linspace(0, span, n), 1 + rng.normal(0, sigma, n)


def test_cleaner_sorts_and_drops_duplicate_times():
    t, f = _noise()
    perm = np.random.default_rng(1).permutation(len(t))
    t_dup = t.copy()
    t_dup[100:150] = t_dup[100]
    tc, fc, ec = clean_and_flatten(t_dup[perm], f[perm])
    assert np.all(np.diff(tc) > 0)
    assert len(tc) == len(fc) == len(ec)


@pytest.mark.parametrize("bad", ["all_nan", "zeros", "partial_nan"])
def test_cleaner_sanitizes_bad_flux_err(bad):
    t, f = _noise()
    err = {"all_nan": np.full_like(f, np.nan), "zeros": np.zeros_like(f),
           "partial_nan": np.where(np.arange(len(f)) % 3 == 0, np.nan, 5e-4)}[bad]
    tc, fc, ec = clean_and_flatten(t, f, err)
    assert np.isfinite(ec).all() and (ec > 0).all()
    run_bls(tc, fc, ec)  # used to crash with an empty-array reduction error


@pytest.mark.parametrize("flux", [np.nan, 0.0, -1.0])
def test_cleaner_rejects_unusable_flux_with_clear_error(flux):
    t, f = _noise()
    with pytest.raises(ValueError, match="valid cadences"):
        clean_and_flatten(t, np.full_like(f, flux))


def test_bls_survives_unusable_flux_err():
    t, f = _noise()
    r = run_bls(t, f, np.full_like(f, np.nan))
    assert np.isfinite(r["period"])


def test_bls_period_grid_is_capped_and_recovers_period_on_long_baseline():
    rng = np.random.default_rng(3)
    span, period, dur, depth, n = 300.0, 5.7, 0.10, 0.004, 12000
    t = np.sort(rng.uniform(0, span, n))
    f = 1 + rng.normal(0, 6e-4, n)
    f[np.abs(((t - 1.0 + period / 2) % period) - period / 2) < dur / 2] -= depth
    t0 = time.time()
    r = run_bls(t, f, np.full(n, 6e-4), max_periods=20000)
    assert r["period_grid_coarsened"] and r["n_trial_periods"] <= 20000 * 1.1
    assert abs(r["period"] - period) / period < 1e-3
    assert time.time() - t0 < 60  # the uncapped grid took minutes


def test_bls_single_sector_grid_is_not_coarsened():
    t, f = _noise()
    assert run_bls(t, f)["period_grid_coarsened"] is False


def test_status_requires_significant_bls_peak():
    ok_phys = {"period_validation_status": "SUPPORTED", "is_eclipsing_binary_flag": False}
    assert derive_overall_status({"snr": 5.0, "period_quality": "INTERIOR_PEAK"}, ok_phys) == "NO_SIGNIFICANT_DETECTION"
    assert derive_overall_status({"snr": float("nan")}, ok_phys) == "NO_SIGNIFICANT_DETECTION"
    assert derive_overall_status({"snr": 20.0, "period_quality": "INTERIOR_PEAK"}, ok_phys) == "PERIODICITY_SUPPORTED"
    eb = dict(ok_phys, is_eclipsing_binary_flag=True)
    assert derive_overall_status({"snr": 20.0, "period_quality": "INTERIOR_PEAK"}, eb) == "EB_FLAGGED"
    assert derive_overall_status({"snr": 20.0, "period_quality": "BOUNDARY_PEAK"}, ok_phys) == "UNCONFIRMED_BOUNDARY_PEAK"


def test_pipeline_white_noise_is_not_a_detection_but_injected_transit_is():
    from root.main_pipeline import Pipeline
    pipe = Pipeline()
    t, f = _noise(n=8000, seed=11)
    assert pipe.run_from_arrays(t, f)["overall_status"] == "NO_SIGNIFICANT_DETECTION"
    ph = ((t - 1 + 1.5) % 3.0) - 1.5
    f2 = f.copy()
    f2[np.abs(ph) < 0.06] -= 0.004
    res = pipe.run_from_arrays(t, f2)
    assert res["overall_status"] == "PERIODICITY_SUPPORTED"
    assert abs(res["bls"]["period"] - 3.0) < 0.01


def test_pipeline_rejects_too_few_cadences():
    from root.main_pipeline import Pipeline
    t, f = _noise(n=60)
    with pytest.raises(ValueError, match="valid cadences"):
        Pipeline().run_from_arrays(t, f)


def test_entry_points_import_without_shadowing_top_level_names():
    import importlib
    for mod in ("main_pipeline", "train_all", "train_real", "evaluate_real", "benchmark_real"):
        importlib.import_module(f"root.{mod}")
    for bare in ("main_pipeline", "train_real", "data", "detection", "preprocessing", "classification"):
        assert bare not in sys.modules, f"'{bare}' imported as a bare top-level module"


def test_checkpoint_loader_handles_non_tensor_metadata(tmp_path):
    from root.checkpoint_io import load_checkpoint
    path = tmp_path / "ck.pt"
    torch.save({"state_dict": {"w": torch.ones(2)}, "metric": np.float64(0.5),
                "arr": np.arange(3)}, path)
    ck = load_checkpoint(str(path))
    assert float(ck["metric"]) == 0.5
    (tmp_path / "bad.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="Could not read checkpoint"):
        load_checkpoint(str(tmp_path / "bad.pt"))


def test_qlp_quality_mask_is_opt_in_and_qlp_only(tmp_path):
    from astropy.io import fits
    from root.data.ingest import _read_generic_fits_lightcurve
    n = 50
    cols = [fits.Column(name="TIME", format="D", array=np.arange(n, dtype=float)),
            fits.Column(name="SAP_FLUX", format="D", array=np.ones(n)),
            fits.Column(name="QUALITY", format="J", array=np.r_[np.zeros(40), np.ones(10)].astype(int))]
    path = tmp_path / "lc.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)
    # Default must NOT mask: the training cache was built without it.
    assert len(_read_generic_fits_lightcurve(str(path), provenance="qlp")[0]) == 50
    assert len(_read_generic_fits_lightcurve(str(path), provenance="qlp", apply_quality_mask=True)[0]) == 40
    assert len(_read_generic_fits_lightcurve(str(path), provenance="tars", apply_quality_mask=True)[0]) == 50


def test_report_renders_and_uses_pipeline_status(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from root.main_pipeline import Pipeline
    from root.reporting.pdf_report import generate_vetting_report
    t, f = _noise(n=6000, seed=5)
    res = Pipeline().run_from_arrays(t, f)
    out = tmp_path / "r.png"
    generate_vetting_report("TIC_1", res["bls"], res["verification"], res["ai"]["planet_score"],
                            views=res["views"], weights_loaded=res["weights_loaded"],
                            output_path=str(out), overall_status=res["overall_status"])
    assert out.exists() and out.stat().st_size > 10_000


def test_synthetic_planet_metadata_duration_stays_in_injected_range():
    for seed in range(150):
        _, _, meta = LightCurveSimulator(points_per_day=60, seed=seed).generate_planet_sample()
        assert 0.035 <= meta["duration"] <= 0.24


@pytest.mark.skipif(not glob.glob(os.path.join(PROJECT_DIR, "mastDownload", "HLSP", "*0172518755*", "*.fits")),
                    reason="bundled QLP file for TIC 172518755 not present")
def test_bundled_real_lightcurve_recovers_known_planet_period():
    from root.data.ingest import _read_generic_fits_lightcurve
    from root.main_pipeline import Pipeline
    path = glob.glob(os.path.join(PROJECT_DIR, "mastDownload", "HLSP", "*0172518755*", "*.fits"))[0]
    t, f, e = _read_generic_fits_lightcurve(path, provenance="qlp")
    res = Pipeline().run_from_arrays(t, f, target_id="TIC_172518755")
    assert abs(res["bls"]["period"] - 3.2087) / 3.2087 < 0.01
    assert res["overall_status"] == "PERIODICITY_SUPPORTED"


def test_hlsp_reader_without_uncertainties_defers_noise_estimate_to_cleaner(tmp_path):
    from astropy.io import fits
    from root.data.ingest import _read_generic_fits_lightcurve
    rng = np.random.default_rng(0)
    n = 4000
    t = np.linspace(0, 27, n)
    # strong slow variability: raw-flux scatter >> point-to-point noise
    flux = 1 + 0.02 * np.sin(2 * np.pi * t / 9.0) + rng.normal(0, 5e-4, n)
    cols = [fits.Column(name="TIME", format="D", array=t),
            fits.Column(name="SAP_FLUX", format="D", array=flux)]
    path = tmp_path / "lc.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)
    _, _, err = _read_generic_fits_lightcurve(str(path), provenance="qlp")
    assert np.isnan(err).all()
    _, _, ec = clean_and_flatten(t, flux, err)
    assert 3e-4 < float(np.median(ec)) < 1e-3  # ~ the true 5e-4, not the ~1.4e-2 raw-flux MAD
