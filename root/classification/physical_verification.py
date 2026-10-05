import numpy as np


def _event_ids(time, period, t0, duration):
    """Return integer transit numbers and an in-transit mask."""
    transit_num = np.rint((time - t0) / period).astype(int)
    residual = time - (t0 + transit_num * period)
    in_transit = np.abs(residual) < 0.5 * duration
    return transit_num, residual, in_transit


def verify_astrophysics(time, flux, period, t0, duration, r_star=1.0, min_points=6,
                        min_transits_for_period=3):
    """Perform conservative transit-count, odd/even and secondary-eclipse checks.

    A vetting test is indeterminate, not zero-depth, when too few cadence points
    or distinct transit events are available. In particular, a one-sector light
    curve often cannot validate a ~14-day period because it contains only one or
    two transit events. Such a period is reported, but is never treated as a
    validated periodic detection.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]

    base = {
        "odd_depth": np.nan, "even_depth": np.nan,
        "odd_even_mismatch_pct": np.nan, "secondary_depth": np.nan,
        "is_eclipsing_binary_flag": False,
        "eb_vetting_evaluated": False,
        "odd_even_status": "invalid_ephemeris",
        "secondary_status": "invalid_ephemeris",
        "observed_transits": 0,
        "expected_transits": 0,
        "transit_numbers": [],
        "period_validation_status": "INVALID_EPHEMERIS",
    }
    if period <= 0 or duration <= 0 or len(time) == 0:
        return base

    transit_num, residual, in_transit = _event_ids(time, period, t0, duration)
    observed_ids = np.unique(transit_num[in_transit])
    observed_ids = observed_ids[np.isfinite(observed_ids)]
    observed_transits = int(len(observed_ids))

    # Number of events whose centers fall inside the observed baseline.
    first_n = int(np.ceil((np.nanmin(time) - t0) / period))
    last_n = int(np.floor((np.nanmax(time) - t0) / period))
    expected_transits = max(0, last_n - first_n + 1)

    base.update({
        "observed_transits": observed_transits,
        "expected_transits": int(expected_transits),
        "transit_numbers": [int(x) for x in observed_ids],
    })

    if observed_transits >= min_transits_for_period:
        base["period_validation_status"] = "SUPPORTED"
    elif observed_transits >= 1:
        base["period_validation_status"] = "INSUFFICIENT_TRANSITS"
    else:
        base["period_validation_status"] = "NO_TRANSITS"

    def measured_depth(mask):
        if np.sum(mask) < min_points:
            return np.nan, "insufficient_data"
        baseline_mask = ~in_transit
        baseline = (np.nanmedian(flux[baseline_mask])
                    if np.sum(baseline_mask) >= min_points else np.nanmedian(flux))
        depth = baseline - np.nanmedian(flux[mask])
        return float(max(depth, 0.0)), "measured"

    odd_mask = in_transit & ((transit_num % 2) != 0)
    even_mask = in_transit & ((transit_num % 2) == 0)
    odd_depth, odd_status = measured_depth(odd_mask)
    even_depth, even_status = measured_depth(even_mask)

    # Odd/even is only meaningful when both parity groups have usable points
    # and there are enough distinct transit events to compare.
    odd_events = len(np.unique(transit_num[odd_mask]))
    even_events = len(np.unique(transit_num[even_mask]))
    if (np.isfinite(odd_depth) and np.isfinite(even_depth)
            and odd_events >= 1 and even_events >= 1):
        avg_depth = max((odd_depth + even_depth) / 2.0, 1e-8)
        mismatch = abs(odd_depth - even_depth) / avg_depth * 100.0
        odd_even_status = "measured"
    else:
        mismatch = np.nan
        odd_even_status = "insufficient_data"

    # The residual is centered around zero, so secondary eclipse occurs near
    # +/- P/2 relative to the nearest primary transit.
    # A secondary-eclipse test is not meaningful when the candidate period
    # itself is supported by fewer than three distinct transit events. A
    # single event can make an arbitrary phase region look like a secondary.
    if observed_transits < min_transits_for_period:
        secondary_depth, secondary_status = np.nan, "insufficient_data"
    else:
        secondary_phase_distance = np.abs(np.abs(residual) - 0.5 * period)
        secondary_mask = secondary_phase_distance < 0.5 * duration
        secondary_depth, secondary_status = measured_depth(secondary_mask)

    eb_flag = False
    eb_evaluated = False
    if odd_even_status == "measured" and mismatch > 35.0:
        eb_flag = True
        eb_evaluated = True
    if (secondary_status == "measured" and np.isfinite(secondary_depth)
            and np.isfinite(odd_depth) and np.isfinite(even_depth)):
        avg_depth = max((odd_depth + even_depth) / 2.0, 1e-8)
        eb_flag = eb_flag or (secondary_depth > 0.4 * avg_depth and secondary_depth > 0.002)
        eb_evaluated = True

    base.update({
        "odd_depth": float(odd_depth) if np.isfinite(odd_depth) else np.nan,
        "even_depth": float(even_depth) if np.isfinite(even_depth) else np.nan,
        "odd_even_mismatch_pct": float(mismatch) if np.isfinite(mismatch) else np.nan,
        "secondary_depth": float(secondary_depth) if np.isfinite(secondary_depth) else np.nan,
        "is_eclipsing_binary_flag": bool(eb_flag),
        "eb_vetting_evaluated": bool(eb_evaluated),
        "odd_even_status": odd_even_status,
        "secondary_status": secondary_status,
    })
    return base
