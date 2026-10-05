"""
Exoplanet AI - Interactive Dashboard
Framework: Streamlit

Rewritten to match the current architecture: a single TriBranchTESSNet
(global-view CNN + local-view CNN + differentiable SNN branch + tabular
features, fused into one classifier), Grad-CAM saliency instead of a
separate SHAP-on-summary-features model, and odd-even / secondary-
eclipse vetting instead of a Bayesian MCMC fit + pixel-level centroid
check (the latter isn't available in this architecture -- see this
project's own notes on what was dropped in the rewrite).

Everything below is read from the Pipeline's result dict -- if a field
is missing, the UI shows "N/A" rather than a fabricated number.

Run with: streamlit run root/ux/dashboard.py  (from the astro/ directory)
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Make `root` importable when Streamlit launches this file directly.
_ROOT_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT_PARENT not in sys.path:
    sys.path.insert(0, _ROOT_PARENT)

from root.main_pipeline import Pipeline, DEFAULT_WEIGHTS_PATH  # noqa: E402
from root.data.synthetic_data import LightCurveSimulator  # noqa: E402
from root.data.ingest import list_available_sectors  # noqa: E402


st.set_page_config(page_title="Exoplanet AI Dashboard", layout="wide", page_icon="\U0001FA90")

st.title("EXOPLANET AI")
st.subheader("TESS Candidate Vetting Dashboard")


@st.cache_resource
def get_pipeline():
    return Pipeline(weights_path=DEFAULT_WEIGHTS_PATH)


with st.sidebar:
    st.header("Run Analysis")
    source = st.selectbox(
        "Data source",
        ["Real TESS target (MAST)", "Synthetic planet sample", "Synthetic false-positive sample"],
    )
    tic_id_input = None
    sector_input = None
    if source == "Real TESS target (MAST)":
        tic_id_input = st.number_input("TIC ID", value=383715654, step=1,
                                        help="e.g. 383715654. Downloads a real SPOC light curve from "
                                             "MAST via lightkurve -- requires network access.")
        sector_input = st.text_input("Sector (optional)", value="")
        if st.button("List available sectors", use_container_width=True):
            with st.spinner(f"Querying MAST for sectors covering TIC {int(tic_id_input)}..."):
                try:
                    st.session_state["sector_rows"] = list_available_sectors(int(tic_id_input))
                    st.session_state["sector_rows_tic"] = int(tic_id_input)
                except Exception as e:
                    st.session_state["sector_rows"] = None
                    st.session_state["sector_rows_tic"] = None
                    st.error(f"Sector lookup failed: {e}")

        if st.session_state.get("sector_rows_tic") == int(tic_id_input):
            rows = st.session_state.get("sector_rows")
            if rows is None:
                pass  # error already shown above
            elif len(rows) == 0:
                st.warning(f"No TESS data found for TIC {int(tic_id_input)} on MAST at all.")
            else:
                st.caption(f"{len(rows)} product(s) found -- pick one of these sector numbers:")
                st.dataframe(
                    [{"sector": r["sector"], "author": r["author"], "exptime (s)": r["exptime"]}
                     for r in rows],
                    use_container_width=True, hide_index=True,
                )
    r_star = st.number_input("Stellar radius (R_sun)", value=1.0, min_value=0.05, step=0.05)
    teff = st.number_input("Effective temperature (K)", value=5778.0, step=50.0)
    logg = st.number_input("log(g)", value=4.44, step=0.05)
    run_clicked = st.button("\u25B6 Run Pipeline", use_container_width=True, type="primary")
    st.divider()
    st.caption(
        "Calls the real pipeline (root.main_pipeline.Pipeline). No values below are hardcoded -- "
        "if a field is missing from the pipeline's output, it is shown as N/A."
    )

if "result" not in st.session_state:
    st.session_state["result"] = None
    st.session_state["ground_truth"] = None
if "sector_rows" not in st.session_state:
    st.session_state["sector_rows"] = None
    st.session_state["sector_rows_tic"] = None

if run_clicked:
    pipeline = get_pipeline()
    if source == "Real TESS target (MAST)":
        with st.spinner(f"Downloading real light curve for TIC {tic_id_input} from MAST, then running "
                         "the full pipeline... this can take a minute or more."):
            try:
                sector = int(sector_input) if sector_input else None
                st.session_state["result"] = pipeline.run(
                    int(tic_id_input), sector=sector, r_star=r_star, teff=teff, logg=logg,
                    generate_report=False,
                )
                st.session_state["ground_truth"] = None
            except Exception as e:
                st.session_state["result"] = None
                st.error(f"Could not fetch/process real data for TIC {tic_id_input}: {e}")
    else:
        with st.spinner("Running full pipeline (clean -> BLS -> dual-view AI model -> vetting)..."):
            sim = LightCurveSimulator(seed=int(np.random.randint(0, 1_000_000)))
            if source == "Synthetic planet sample":
                time_arr, flux_arr, meta = sim.generate_planet_sample()
            else:
                time_arr, flux_arr, meta = sim.generate_false_positive_sample()
            st.session_state["result"] = pipeline.run_from_arrays(
                time_arr, flux_arr, target_id="synthetic", r_star=r_star, teff=teff, logg=logg,
            )
            st.session_state["ground_truth"] = meta

result = st.session_state["result"]

if result is None:
    st.info("Click **Run Pipeline** in the sidebar to analyze a target. Nothing below is populated yet.")
    st.stop()

bls = result.get("bls", {})
views = result.get("views", {})
ai = result.get("ai", {})
verification = result.get("verification", {})
light_curve = result.get("light_curve", {})
weights_loaded = result.get("weights_loaded", False)

if not weights_loaded:
    if result.get("weights_present"):
        st.warning(
            "**LEGACY WEIGHTS — UNVALIDATED**: a pre-v2 checkpoint was loaded, but it was trained "
            "under the old preprocessing/model contract. Retrain with `python -m root.train_all` "
            "before treating AI confidence as a learned assessment."
        )
    else:
        st.error(
            "**UNTRAINED WEIGHTS**: no trained TriBranchTESSNet weights were found "
            f"at `{DEFAULT_WEIGHTS_PATH}`. The AI confidence is from a randomly-initialized "
            "network and is not a meaningful assessment. Run `python -m root.train_all` to produce "
            "real weights."
        )

# ----------------------------------------------------------------------
# Overall verdict (single source of truth: root/status.py via the Pipeline)
# ----------------------------------------------------------------------
from root.status import STATUS_DISPLAY  # noqa: E402

_status_code = result.get("overall_status")
if _status_code:
    _status_text = STATUS_DISPLAY.get(_status_code, _status_code.replace("_", " "))
    if _status_code == "PERIODICITY_SUPPORTED":
        st.success(f"**Overall status:** {_status_text}")
    elif _status_code == "EB_FLAGGED":
        st.error(f"**Overall status:** {_status_text}")
    else:
        st.warning(f"**Overall status:** {_status_text}")
    st.caption("The AI score is a model score, not confirmation of a planet. "
               "Status reflects BLS significance, transit count and odd/even + secondary vetting.")

# ----------------------------------------------------------------------
# 1. Quick overview
# ----------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Pipeline Status**")
    status_cols = st.columns(4)
    steps = [
        ("1. Detection (BLS)", "bls" in result),
        ("2. Dual Views", "views" in result),
        ("3. AI Classification", "ai" in result and weights_loaded),
        ("4. Physical Vetting", "verification" in result),
    ]
    for i, (label, done) in enumerate(steps):
        with status_cols[i]:
            (st.success if done else st.warning)(label)

with col2:
    st.markdown("**Quick Overview**")
    metric_cols = st.columns(2)
    metric_cols[0].metric("BLS SNR", f"{bls.get('snr', float('nan')):.3f}" if bls else "N/A")
    metric_cols[1].metric(
        "AI Confidence" + ("" if weights_loaded else " (untrained)"),
        f"{ai.get('planet_probability', 0):.0%}" if ai else "N/A",
    )

if result.get("ground_truth") is None and st.session_state.get("ground_truth") is not None:
    gt = st.session_state["ground_truth"]
    label = "planet" if gt.get("label") == 1 else "false positive"
    st.caption(f"Ground truth (synthetic sample): **{label}**" +
               (f", true period={gt.get('period'):.4f}d, true depth={gt.get('depth'):.4f}"
                if gt.get("label") == 1 else ""))

st.divider()

# ----------------------------------------------------------------------
# 2. Light curve + BLS periodogram
# ----------------------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.markdown("**Light Curve (raw vs. cleaned)**")
    if "time_raw" in light_curve:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(light_curve["time_raw"], light_curve["flux_raw"], color="#888888", alpha=0.4,
                linewidth=0.5, label="Raw")
        ax.plot(light_curve["time_clean"], light_curve["flux_clean"], color="#1f77b4",
                linewidth=0.6, label="Cleaned")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Normalized flux")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No light curve data in result.")

with col4:
    st.markdown("**BLS Periodogram**")
    if bls.get("periods") is not None:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(bls["periods"], bls["powers"], color="navy", lw=0.8)
        ax.axvline(bls["period"], color="red", linestyle="--", label=f"P = {bls['period']:.4f} d")
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Power / SNR")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No BLS periodogram in result.")

st.divider()

# ----------------------------------------------------------------------
# 3. Global/local views + Grad-CAM
# ----------------------------------------------------------------------
col5, col6 = st.columns(2)

with col5:
    st.markdown("**Global View (phase-folded, full period)**")
    global_view = views.get("global_view")
    if global_view is not None:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.plot(np.linspace(-0.5, 0.5, len(global_view)), global_view, color="#1f77b4", lw=1)
        ax.set_xlabel("Phase")
        ax.set_ylabel("Flux (baseline-subtracted)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No global view in result.")

with col6:
    st.markdown("**Local View + Grad-CAM Saliency**")
    local_view = views.get("local_view")
    saliency = views.get("saliency")
    if local_view is not None:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        x = np.linspace(-1, 1, len(local_view))
        if saliency is not None and len(saliency) == len(local_view):
            for i in range(len(x) - 1):
                ax.axvspan(x[i], x[i + 1], color=plt.cm.Reds(float(saliency[i])), alpha=0.6, lw=0)
        ax.plot(x, local_view, color="black", lw=1.2)
        ax.set_xlabel("Local view phase (normalized)")
        ax.set_ylabel("Flux (baseline-subtracted)")
        st.pyplot(fig)
        plt.close(fig)
        if saliency is None:
            st.caption("Grad-CAM saliency unavailable for this run (see console warning).")
    else:
        st.warning("No local view in result.")

st.divider()

# ----------------------------------------------------------------------
# 4. Physical vetting + estimated parameters
# ----------------------------------------------------------------------
col7, col8 = st.columns(2)

with col7:
    st.markdown("**Estimated Parameters (BLS)**")
    if bls:
        st.text(
            f"Period:   {bls.get('period', float('nan')):.5f} d\n"
            f"Duration: {bls.get('duration', float('nan')):.5f} d\n"
            f"Depth:    {bls.get('depth', float('nan')):.5f}\n"
            f"SNR:      {bls.get('snr', float('nan')):.3f}"
        )
    else:
        st.text("N/A")

with col8:
    st.markdown("**Physical Vetting (odd-even / secondary eclipse)**")
    if verification:
        st.text(
            f"Odd depth:    {verification.get('odd_depth', float('nan')):.6f}\n"
            f"Even depth:   {verification.get('even_depth', float('nan')):.6f}\n"
            f"Mismatch:     {verification.get('odd_even_mismatch_pct', float('nan')):.1f}%\n"
            f"Secondary:    {verification.get('secondary_depth', float('nan')):.6f}"
        )
        if verification.get("odd_even_status") == "insufficient_data" or verification.get("secondary_status") == "insufficient_data":
            st.warning("Eclipsing-binary vetting: INDETERMINATE (too few in-transit points)")
        elif verification.get("is_eclipsing_binary_flag"):
            st.error("Eclipsing-binary flag: POSITIVE")
        else:
            st.success("Eclipsing-binary flag: not flagged")
        st.caption(
            "Note: this architecture does not include a pixel-level Target Pixel File centroid "
            "check -- only light-curve-based odd-even and secondary-eclipse tests."
        )
    else:
        st.text("N/A")

st.divider()

# ----------------------------------------------------------------------
# 5. Report
# ----------------------------------------------------------------------
st.markdown("**Report & Output**")
if st.button("Generate Vetting Report Image"):
    from root.reporting.pdf_report import generate_vetting_report
    with st.spinner("Compiling report..."):
        report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", f"{result.get('target_id', 'target')}_report.png")
        generate_vetting_report(
            result.get("target_id", "target"), bls, verification, ai.get("planet_probability", 0.0),
            views=views, weights_loaded=weights_loaded, output_path=report_path,
            overall_status=result.get("overall_status"),
        )
    st.success(f"{os.path.basename(report_path)} generated.")
    st.image(report_path)
    with open(report_path, "rb") as f:
        st.download_button("Download Report", f.read(), file_name=os.path.basename(report_path))
