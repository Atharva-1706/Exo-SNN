"""Single source of truth for the pipeline's overall candidate status."""

DEFAULT_MIN_BLS_SNR = 7.0

# Human-readable text used by the PNG vetting report.
STATUS_DISPLAY = {
    "NO_SIGNIFICANT_DETECTION": "NO SIGNIFICANT DETECTION: BLS SNR BELOW THRESHOLD",
    "UNCONFIRMED_BOUNDARY_PEAK_AND_INSUFFICIENT_TRANSITS":
        "UNCONFIRMED: BLS BOUNDARY PEAK + INSUFFICIENT TRANSITS",
    "UNCONFIRMED_BOUNDARY_PEAK": "UNCONFIRMED: BLS PEAK AT SEARCH BOUNDARY",
    "UNCONFIRMED_INSUFFICIENT_TRANSITS": "UNCONFIRMED: INSUFFICIENT TRANSIT EVENTS",
    "EB_FLAGGED": "EB FLAGGED",
    "PERIODICITY_SUPPORTED": "PERIODICITY SUPPORTED",
}


def derive_overall_status(bls_res, phys_res, min_snr=DEFAULT_MIN_BLS_SNR):
    """Combine BLS quality and physical vetting into one status code.

    A BLS maximum below ``min_snr`` is indistinguishable from the best peak of
    pure noise (white-noise light curves reach SNR ~5 under this BLS setup), so
    it must never be promoted to 'periodicity supported' or an EB flag.
    """
    snr = bls_res.get("snr")
    try:
        snr = float(snr)
    except (TypeError, ValueError):
        snr = float("nan")
    if not (snr >= min_snr):  # also catches NaN
        return "NO_SIGNIFICANT_DETECTION"

    boundary = bls_res.get("period_quality") == "BOUNDARY_PEAK"
    supported = phys_res.get("period_validation_status") == "SUPPORTED"
    if boundary and not supported:
        return "UNCONFIRMED_BOUNDARY_PEAK_AND_INSUFFICIENT_TRANSITS"
    if boundary:
        return "UNCONFIRMED_BOUNDARY_PEAK"
    if not supported:
        return "UNCONFIRMED_INSUFFICIENT_TRANSITS"
    if phys_res.get("is_eclipsing_binary_flag"):
        return "EB_FLAGGED"
    return "PERIODICITY_SUPPORTED"
