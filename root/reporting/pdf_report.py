import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from root.status import STATUS_DISPLAY, derive_overall_status


def generate_vetting_report(tic_id, bls_res, phys_res, confidence, views=None, weights_loaded=True,
                             output_path="report.png", overall_status=None):
    # Keep the report readable when the diagnostic block grows.  The original
    # 11x7 layout allowed the summary text to overflow into the lower panels.
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0],
                          left=0.065, right=0.975, bottom=0.075, top=0.90,
                          hspace=0.34, wspace=0.20)
    axs = np.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            axs[r, c] = fig.add_subplot(gs[r, c])

    display_id = str(tic_id)
    for prefix in ("TIC_", "TIC ", "TIC"):
        if display_id.upper().startswith(prefix.upper()):
            display_id = display_id[len(prefix):].lstrip("_ ")
            break
    title = f"TESS Candidate Vetting Sheet: TIC {display_id}"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # ------------------------------------------------------------------
    # BLS panel
    # ------------------------------------------------------------------
    axs[0, 0].plot(bls_res["periods"], bls_res["powers"], color="navy", lw=1)
    axs[0, 0].axvline(bls_res["period"], color="red", linestyle="--",
                      label=f"P = {bls_res['period']:.4f} d")
    axs[0, 0].set_title("BLS Periodogram Search", fontsize=12)
    axs[0, 0].set_xlabel("Period (days)")
    axs[0, 0].set_ylabel("Power / SNR")
    axs[0, 0].legend(loc="upper left")

    # ------------------------------------------------------------------
    # Physical/AI summary panel
    # ------------------------------------------------------------------
    period_quality = bls_res.get("period_quality", "UNKNOWN")
    period_validation = phys_res.get("period_validation_status", "UNKNOWN")
    observed_transits = phys_res.get("observed_transits", 0)
    expected_transits = phys_res.get("expected_transits", 0)

    # Use the pipeline's own status when supplied so the sheet can never
    # disagree with the returned result; otherwise derive it identically.
    status_code = overall_status or derive_overall_status(bls_res, phys_res)
    overall_status = STATUS_DISPLAY.get(status_code, str(status_code).replace("_", " "))

    if weights_loaded:
        ai_line = f"AI Planet Score: {confidence * 100:.1f}%"
        untrained_note = ""
    else:
        ai_line = "AI Planet Score: N/A"
        untrained_note = "AI status: UNTRAINED WEIGHTS (not a learned assessment)"

    eb_evaluated = phys_res.get("eb_vetting_evaluated", False)
    eb_line = ("POSITIVE" if phys_res.get("is_eclipsing_binary_flag") else "NOT FLAGGED") \
        if eb_evaluated else "NOT EVALUATED"

    def fmt_ppm(value):
        value = value * 1e6
        return f"{value:.1f}" if np.isfinite(value) else "N/A"

    def fmt_pct(value):
        return f"{value:.1f}%" if np.isfinite(value) else "N/A"

    summary_text = (
        f"Period: {bls_res['period']:.4f} days\n"
        f"Period quality: {period_quality}\n"
        f"Observed transit events: {observed_transits}\n"
        f"Expected events in baseline: {expected_transits}\n"
        f"Period validation: {period_validation}\n"
        f"Transit Epoch (T0): {bls_res['t0']:.3f} BTJD\n"
        f"Duration: {bls_res['duration'] * 24.0:.2f} hours\n"
        f"Transit Depth: {bls_res['depth'] * 1e6:.1f} ppm\n"
        f"BLS SNR: {bls_res['snr']:.1f}\n"
        f"──────────────────────────────────\n"
        f"Odd Depth: {fmt_ppm(phys_res.get('odd_depth', np.nan))} ppm\n"
        f"Even Depth: {fmt_ppm(phys_res.get('even_depth', np.nan))} ppm\n"
        f"Mismatch: {fmt_pct(phys_res.get('odd_even_mismatch_pct', np.nan))}\n"
        f"Odd/Even check: {phys_res.get('odd_even_status', 'unknown').upper()}\n"
        f"Secondary check: {phys_res.get('secondary_status', 'unknown').upper()}\n"
        f"EB Flag: {eb_line}\n"
        f"──────────────────────────────────\n"
        f"{ai_line}\n"
        f"{untrained_note}"
    )

    text_color = "red" if not weights_loaded else "black"
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Physical Vetting & AI Scores", fontsize=12, pad=4)
    axs[0, 1].text(0.02, 0.98, summary_text, fontsize=8.5, family="monospace",
                    color=text_color, va="top", ha="left",
                    transform=axs[0, 1].transAxes, linespacing=1.12)

    # Dedicated, wrapped status box so the status can never collide with the
    # lower row, even when the text becomes long.
    status_color = "red" if overall_status.startswith(("UNCONFIRMED", "NO SIGNIFICANT")) else "black"
    axs[0, 1].text(0.02, 0.03,
                    "OVERALL STATUS\n" + overall_status,
                    fontsize=9.0, family="monospace", fontweight="bold",
                    color=status_color, va="bottom", ha="left",
                    transform=axs[0, 1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.35", fill=False,
                              edgecolor=status_color, linewidth=0.9))

    # ------------------------------------------------------------------
    # Local folded view + saliency
    # ------------------------------------------------------------------
    local_view = views.get("local_view") if views else None
    saliency = views.get("saliency") if views else None
    if local_view is not None:
        x = np.linspace(-1, 1, len(local_view))
        plot_mask = np.ones(len(local_view), dtype=bool)
        phase_min = views.get("local_phase_min") if views else None
        phase_max = views.get("local_phase_max") if views else None
        half_phase = views.get("local_half_phase") if views else None
        if phase_min is not None and phase_max is not None and half_phase:
            xmin = max(-1.0, float(phase_min) / float(half_phase))
            xmax = min(1.0, float(phase_max) / float(half_phase))
            plot_mask = (x >= xmin) & (x <= xmax)
        if saliency is not None and len(saliency) == len(local_view):
            for i in range(len(x) - 1):
                if plot_mask[i] and plot_mask[i + 1]:
                    axs[1, 0].axvspan(x[i], x[i + 1],
                                      color=plt.cm.Reds(float(saliency[i])),
                                      alpha=0.6, lw=0)
        axs[1, 0].plot(np.where(plot_mask, x, np.nan),
                       np.where(plot_mask, local_view, np.nan),
                       color="black", lw=1.2)
        axs[1, 0].set_xlabel("Local view phase (normalized)")
        axs[1, 0].set_ylabel("Flux (baseline-subtracted)")
        axs[1, 0].set_title("Local View + Grad-CAM Saliency" if saliency is not None
                            else "Local (Folded) View", fontsize=12)
    else:
        axs[1, 0].text(0.5, 0.5, "No local view available", ha="center", va="center")
        axs[1, 0].axis("off")
        axs[1, 0].set_title("Local (Folded) View", fontsize=12)

    # ------------------------------------------------------------------
    # Odd/even panel: never display a fabricated zero bar.  If only one
    # parity is available, show the measured value but make it explicit that
    # no comparison was performed.
    # ------------------------------------------------------------------
    odd = phys_res.get("odd_depth", np.nan) * 1e6
    even = phys_res.get("even_depth", np.nan) * 1e6
    odd_ok = np.isfinite(odd)
    even_ok = np.isfinite(even)
    parity_status = str(phys_res.get("odd_even_status", "unknown")).upper()

    if odd_ok and even_ok:
        axs[1, 1].bar(["Odd Transits", "Even Transits"], [odd, even])
        axs[1, 1].set_ylabel("Depth (ppm)")
    elif odd_ok or even_ok:
        label = "Odd" if odd_ok else "Even"
        value = odd if odd_ok else even
        # Do not draw a lone bar: it visually resembles a completed
        # odd/even comparison.  Present the one measured parity as a
        # diagnostic card instead.
        axs[1, 1].axis("off")
        axs[1, 1].text(0.5, 0.63,
                        "INSUFFICIENT DATA",
                        transform=axs[1, 1].transAxes, ha="center", va="center",
                        fontsize=12, fontweight="bold")
        axs[1, 1].text(0.5, 0.48,
                        "Parity comparison not performed",
                        transform=axs[1, 1].transAxes, ha="center", va="center",
                        fontsize=10)
        axs[1, 1].text(0.5, 0.33,
                        f"{label} transit depth measured: {value:.1f} ppm",
                        transform=axs[1, 1].transAxes, ha="center", va="center",
                        fontsize=10)
    else:
        axs[1, 1].text(0.5, 0.5,
                        "Insufficient transit events\nParity comparison not performed",
                        transform=axs[1, 1].transAxes, ha="center", va="center",
                        fontsize=10.5)
        axs[1, 1].set_ylabel("Depth (ppm)")

    axs[1, 1].set_title("Odd-Even Parity Check", fontsize=12)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Report saved to: {output_path}")
