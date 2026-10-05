import numpy as np
from root.views import normalize_view

def extract_dual_views(time, flux, period, t0, duration, n_global=201, n_local=61, return_meta=False):
    """Extract phase-folded global/local views with the training normalization."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length")
    if period <= 0 or duration <= 0:
        raise ValueError("period and duration must be positive")

    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    sort_idx = np.argsort(phase)
    p_sorted, f_sorted = phase[sort_idx], flux[sort_idx]

    global_bins = np.linspace(-0.5, 0.5, n_global + 1)
    sums, _ = np.histogram(p_sorted, bins=global_bins, weights=f_sorted)
    counts, _ = np.histogram(p_sorted, bins=global_bins)
    global_view = np.where(counts > 0, sums / np.maximum(counts, 1), np.nanmedian(flux))

    half_phase = min(0.5, max((duration / period) * 2.0, 1.0 / n_local))
    local_mask = (p_sorted >= -half_phase) & (p_sorted <= half_phase)
    local_phase_min = float(np.min(p_sorted[local_mask])) if np.any(local_mask) else -half_phase
    local_phase_max = float(np.max(p_sorted[local_mask])) if np.any(local_mask) else half_phase
    if np.sum(local_mask) > 10:
        local_bins = np.linspace(-half_phase, half_phase, n_local + 1)
        sums, _ = np.histogram(p_sorted[local_mask], bins=local_bins, weights=f_sorted[local_mask])
        counts, _ = np.histogram(p_sorted[local_mask], bins=local_bins)
        fill = np.nanmedian(f_sorted[local_mask])
        local_view = np.where(counts > 0, sums / np.maximum(counts, 1), fill)
    else:
        local_view = np.full(n_local, np.nanmedian(flux), dtype=float)

    result = (normalize_view(global_view), normalize_view(local_view))
    if return_meta:
        meta = {
            "local_half_phase": float(half_phase),
            "local_phase_min": local_phase_min,
            "local_phase_max": local_phase_max,
            "local_has_data": bool(np.sum(local_mask) > 10),
        }
        return result[0], result[1], meta
    return result
