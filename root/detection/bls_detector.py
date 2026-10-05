import numpy as np
from astropy.timeseries import BoxLeastSquares


def _downsample_for_bls(time, flux, flux_err=None, max_points=20000):
    """Reduce very long light curves before BLS without changing the cadence-scale signal.

    BLS only needs minute-scale resolution for the 1--4 hour durations searched here.
    Keeping at most ``max_points`` samples prevents runtime from growing catastrophically
    when several TESS sectors are stitched together.
    """
    if len(time) <= max_points:
        return time, flux, flux_err

    order = np.argsort(time)
    time = np.asarray(time)[order]
    flux = np.asarray(flux)[order]
    err = None if flux_err is None else np.asarray(flux_err)[order]

    edges = np.linspace(0, len(time), max_points + 1, dtype=int)
    t_out, f_out, e_out = [], [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        sl = slice(start, stop)
        t_out.append(np.mean(time[sl]))
        f_out.append(np.mean(flux[sl]))
        if err is not None:
            e_out.append(np.sqrt(np.sum(np.square(err[sl]))) / (stop - start))

    return (np.asarray(t_out), np.asarray(f_out),
            None if err is None else np.asarray(e_out))


def run_bls(time, flux, flux_err=None, min_period=0.5, max_period=15.0,
            duration_hours=(1.0, 2.5, 4.0), frequency_factor=2.0,
            oversample=5, max_points=20000, boundary_fraction=0.05,
            max_periods=60000):
    """Run BLS and return a physically meaningful transit SNR.

    ``flux_err`` is passed to Astropy's ``BoxLeastSquares`` so its ``snr``
    objective actually represents the uncertainty of the fitted depth. For
    very long multi-sector curves, the data are conservatively block-averaged
    before the search to keep runtime bounded.

    The trial-period grid grows roughly with baseline**2 (a single 27 d sector
    needs ~17k periods, 300 d needs ~2M and took minutes). When the grid would
    exceed ``max_periods`` it is coarsened to fit, and the best peak is then
    re-searched on a fine local grid so the reported period keeps full
    resolution. Single-sector runs (grid below the cap) are unchanged.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if len(time) != len(flux) or len(time) < 20:
        raise ValueError("time and flux must have the same length and contain at least 20 points")

    valid = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        if len(flux_err) != len(time):
            raise ValueError("flux_err must have the same length as time and flux")
        err_ok = np.isfinite(flux_err) & (flux_err > 0)
        if np.sum(valid & err_ok) >= 20:
            valid &= err_ok
        else:
            flux_err = None  # no usable uncertainties: search unweighted instead of crashing

    if np.sum(valid) < 20:
        raise ValueError("fewer than 20 finite samples are available for BLS")
    time, flux = time[valid], flux[valid]
    flux_err = flux_err[valid] if flux_err is not None else None
    time, flux, flux_err = _downsample_for_bls(time, flux, flux_err, max_points=max_points)

    durations = np.asarray(duration_hours, dtype=float) / 24.0
    model = BoxLeastSquares(time, flux, dy=flux_err) if flux_err is not None else BoxLeastSquares(time, flux)
    grid_kwargs = dict(minimum_period=min_period, maximum_period=max_period)
    periods = model.autoperiod(durations, frequency_factor=frequency_factor, **grid_kwargs)
    grid_coarsened = len(periods) > max_periods
    if grid_coarsened:
        scale = 1.05 * len(periods) / max_periods
        periods = model.autoperiod(durations, frequency_factor=frequency_factor * scale, **grid_kwargs)
    periodogram = model.power(periods, durations, objective="snr", oversample=oversample)

    best_idx = int(np.argmax(periodogram.power))
    best = periodogram
    best_pos = best_idx

    if grid_coarsened:
        # Re-search the winning peak at full resolution (grid spacing for
        # frequency_factor=1) so coarsening does not blur the reported period.
        baseline = float(time.max() - time.min())
        coarse_step = float(np.max(np.abs(np.diff(periods[max(0, best_idx - 1):best_idx + 2])))) \
            if len(periods) > 2 else 0.0
        fine_step = max(float(periodogram.period[best_idx]) ** 2 * float(durations.min()) / baseline ** 2, 1e-6)
        half_window = max(2.0 * coarse_step, 5.0 * fine_step)
        lo = max(min_period, float(periodogram.period[best_idx]) - half_window)
        hi = min(max_period, float(periodogram.period[best_idx]) + half_window)
        n_fine = int(np.clip((hi - lo) / fine_step + 1, 11, 20000))
        fine = model.power(np.linspace(lo, hi, n_fine), durations, objective="snr", oversample=oversample)
        fine_idx = int(np.argmax(fine.power))
        if fine.power[fine_idx] >= periodogram.power[best_idx]:
            best, best_pos = fine, fine_idx

    best_period = float(best.period[best_pos])
    best_t0 = float(best.transit_time[best_pos])
    best_duration = float(best.duration[best_pos])
    best_depth = float(best.depth[best_pos])
    best_snr = float(best.power[best_pos])
    depth_err = float(best.depth_err[best_pos]) if hasattr(best, "depth_err") else np.nan

    # A maximum at/near the configured search boundary is not a trustworthy
    # period detection. It commonly means the BLS statistic is still rising
    # with period because it is fitting a long-timescale trend or an isolated
    # event. Keep the measured period/SNR, but explicitly mark the solution.
    search_span = max_period - min_period
    boundary_margin = max(boundary_fraction * search_span, 0.25)
    near_lower_boundary = best_period <= min_period + boundary_margin
    near_upper_boundary = best_period >= max_period - boundary_margin
    period_quality = "BOUNDARY_PEAK" if (near_lower_boundary or near_upper_boundary) else "INTERIOR_PEAK"

    stats = model.compute_stats(best_period, best_duration, best_t0)

    return {
        "period": best_period,
        "t0": best_t0,
        "duration": best_duration,
        "depth": best_depth,
        "depth_err": depth_err,
        "snr": best_snr,
        "stats": stats,
        "periods": np.asarray(periodogram.period),
        "powers": np.asarray(periodogram.power),
        "min_period": float(min_period),
        "max_period": float(max_period),
        "period_quality": period_quality,
        "near_lower_boundary": bool(near_lower_boundary),
        "near_upper_boundary": bool(near_upper_boundary),
        "boundary_margin_days": float(boundary_margin),
        "period_grid_coarsened": bool(grid_coarsened),
        "n_trial_periods": int(len(periods)),
    }
