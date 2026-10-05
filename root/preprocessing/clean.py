import numpy as np
from scipy.signal import savgol_filter


def _odd_window(n, target):
    """Return an odd Savitzky-Golay window that fits ``n`` samples."""
    if n < 5:
        return None
    w = min(int(target), n if n % 2 else n - 1)
    if w < 5:
        return None
    return w if w % 2 else w - 1


def clean_and_flatten(time, flux, flux_err=None, window_length=None,
                      window_hours=48.0, sigma_upper=5.0, sigma_lower=15.0):
    """Clean and flatten a TESS light curve without fitting away transits.

    The detrending window is expressed in hours rather than a fixed number of
    samples. This keeps the physical timescale consistent between the training
    data (coarser cadence) and native 2-minute TESS data. The default 48-hour
    window is deliberately much longer than the 1--4 hour BLS transit durations.

    Input hygiene: non-finite / non-positive samples are dropped, the series is
    sorted by time, and duplicate timestamps are removed (first sample kept).
    Non-finite or non-positive uncertainties are replaced by the median valid
    uncertainty; if none are usable a robust estimate is used instead.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length")
    if len(time) < 10:
        raise ValueError("light curve is too short to clean")

    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        if len(flux_err) != len(time):
            raise ValueError("flux_err must have the same length as time and flux")

    valid = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    t, f = time[valid], flux[valid]
    err = flux_err[valid] if flux_err is not None else None

    # Real/stitched data can arrive unsorted or with repeated timestamps; both
    # would otherwise corrupt the cadence estimate and the Savitzky-Golay fit.
    order = np.argsort(t, kind="stable")
    t, f = t[order], f[order]
    if err is not None:
        err = err[order]
    unique = np.ones(len(t), dtype=bool)
    unique[1:] = np.diff(t) > 0
    t, f = t[unique], f[unique]
    if err is not None:
        err = err[unique]
    if len(t) < 10:
        raise ValueError(
            f"only {len(t)} valid cadences remain after removing NaN, non-positive "
            "and duplicate-time samples; need at least 10"
        )

    med = np.nanmedian(f)
    if not np.isfinite(med) or med <= 0:
        raise ValueError("light curve has no valid positive baseline")
    f = f / med
    if err is not None:
        err = err / med
        good_err = np.isfinite(err) & (err > 0)
        if not np.any(good_err):
            err = None  # fall back to the robust estimate below
        elif not np.all(good_err):
            err = np.where(good_err, err, np.median(err[good_err]))

    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time array must be strictly increasing")

    if window_length is None:
        target = max(5, int(round(window_hours / 24.0 / dt)))
    else:
        # Backward-compatible override, but callers should prefer window_hours.
        target = int(window_length)
    window = _odd_window(len(f), target)
    if window is None:
        return t, f.astype(np.float32), (err if err is not None else _estimate_flux_err(f)).astype(np.float32)

    for _ in range(2):
        smooth = savgol_filter(f, window_length=window, polyorder=2, mode="interp")
        resid = f - smooth
        std = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        if not np.isfinite(std) or std <= 1e-8:
            std = np.nanstd(resid)
        if not np.isfinite(std) or std <= 1e-8:
            break
        keep = (resid > -sigma_lower * std) & (resid < sigma_upper * std)
        t, f = t[keep], f[keep]
        if err is not None:
            err = err[keep]
        window = _odd_window(len(f), target)
        if window is None:
            break

    if window is None:
        flat = f
    else:
        baseline = savgol_filter(f, window_length=window, polyorder=2, mode="interp")
        baseline = np.where(np.abs(baseline) > 1e-8, baseline, 1.0)
        flat = f / baseline

    if err is None:
        err = _estimate_flux_err(flat)
    return t.astype(np.float64), flat.astype(np.float32), np.asarray(err, dtype=np.float32)


def _estimate_flux_err(flux):
    """Robust per-cadence uncertainty estimate for synthetic/no-error inputs."""
    diff = np.diff(np.asarray(flux, dtype=float))
    mad = np.nanmedian(np.abs(diff - np.nanmedian(diff)))
    sigma = mad / 0.6745 / np.sqrt(2.0) if np.isfinite(mad) else np.nanstd(flux)
    sigma = float(max(sigma, 1e-6))
    return np.full(len(flux), sigma, dtype=np.float32)
