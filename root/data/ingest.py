import time as _time

import numpy as np
import lightkurve as lk
from astroquery.mast import Observations
from astropy.io import fits

# astroquery's MAST request timeout defaults to 600s and is NOT controlled
# by setting Observations.TIMEOUT (that's a silent no-op) -- it lives on
# Observations._portal_api_connection.TIMEOUT, which is undocumented and
# has shifted across astroquery versions. We set it defensively via
# _set_mast_timeout() below. A normal MAST query returns in a few seconds;
# taking anywhere near this timeout means something is actually wrong
# (blocked/filtered connection to mast.stsci.edu, corporate proxy, etc.),
# not that the query just needs more time.
MAST_TIMEOUT_SECONDS = 45
MAST_RETRY_ATTEMPTS = 3
MAST_RETRY_BASE_DELAY = 2.0


def _with_retries(fn, attempts=MAST_RETRY_ATTEMPTS, base_delay=MAST_RETRY_BASE_DELAY):
    """Call ``fn`` and retry transient failures (MAST routinely drops requests)."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # network, HTTP 5xx, partial reads, ...
            last = exc
            if i < attempts - 1:
                _time.sleep(base_delay * (2 ** i))
    raise last


def _set_mast_timeout(seconds: int):
    Observations.TIMEOUT = seconds  # documented attribute; kept for forward-compat
    try:
        Observations._portal_api_connection.TIMEOUT = seconds  # the one actually used
    except AttributeError:
        pass  # older/newer astroquery layout -- fall back to the 600s default rather than crash


def list_available_sectors(tic_id: int, author: str = "SPOC"):
    """
    Search MAST (no download) and return the sectors/cadences actually
    available for this target, so a caller can pick a real --sector
    value instead of guessing. TESS observes a different strip of sky
    each ~27-day sector, so most targets only have data in a handful of
    the 90+ sectors flown so far -- "No TESS data found" for a specific
    --sector almost always means that sector just didn't cover this
    part of the sky, not a bug in the pipeline.
    """
    target_str = f"TIC {tic_id}"
    _set_mast_timeout(MAST_TIMEOUT_SECONDS)
    t0 = _time.time()
    try:
        search = _with_retries(lambda: lk.search_lightcurve(target_str, mission="TESS", author=author))
        if len(search) == 0:
            search = _with_retries(lambda: lk.search_lightcurve(target_str, mission="TESS"))
    except Exception as e:
        raise ConnectionError(
            f"MAST search for {target_str} failed after {_time.time() - t0:.0f}s "
            f"(timeout set to {MAST_TIMEOUT_SECONDS}s). A normal query returns in a few "
            "seconds, so this means the connection to mast.stsci.edu is being blocked, "
            "filtered, or is extremely slow -- check firewall/VPN/proxy settings, or try "
            "a different network. Original error: "
            f"{type(e).__name__}: {e}"
        ) from e

    if len(search) == 0:
        return []

    rows = []
    cols = search.table.colnames
    for i in range(len(search)):
        row = search.table[i]
        entry = {
            "sector": int(row["sequence_number"]) if "sequence_number" in cols else None,
            "author": str(row["author"]) if "author" in cols else None,
            "exptime": float(row["exptime"]) if "exptime" in cols else None,
        }
        # MAST search results normally expose the observation window as
        # t_min/t_max. Keep it so the real-data trainer can select a sector
        # that actually contains a catalog-predicted transit.
        for key in ("t_min", "t_max"):
            if key in cols:
                try:
                    entry[key] = float(row[key])
                except Exception:
                    entry[key] = None
            else:
                entry[key] = None
        rows.append(entry)
    return rows


def download_tess_lightcurve(tic_id: int, sector: int = None, author: str = "SPOC", all_sectors: bool = False,
                             qlp_quality_mask: bool = False):
    """
    Queries MAST for real TESS PDCSAP light curves.
    """
    target_str = f"TIC {tic_id}"
    print(f"Querying MAST for {target_str}...")
    _set_mast_timeout(MAST_TIMEOUT_SECONDS)

    t0 = _time.time()
    try:
        search = _with_retries(
            lambda: lk.search_lightcurve(target_str, mission="TESS", author=author, sector=sector))
        if len(search) == 0:
            print(f"No {author} data found. Searching all authors...")
            search = _with_retries(lambda: lk.search_lightcurve(target_str, mission="TESS", sector=sector))
    except Exception as e:
        raise ConnectionError(
            f"MAST search for {target_str} failed after {_time.time() - t0:.0f}s "
            f"(timeout set to {MAST_TIMEOUT_SECONDS}s). A normal query returns in a few "
            "seconds, so this means the connection to mast.stsci.edu is being blocked, "
            "filtered, or is extremely slow -- check firewall/VPN/proxy settings, or try "
            "a different network. Original error: "
            f"{type(e).__name__}: {e}"
        ) from e

    if len(search) == 0:
        raise ValueError(f"No TESS data found for TIC {tic_id} on MAST.")

    sectors_found = None
    try:
        if "sequence_number" in search.table.colnames:
            sectors_found = sorted(set(int(s) for s in search.table["sequence_number"]))
    except Exception:
        pass  # purely informational -- never let this block the actual download

    print(f"   Found {len(search)} data product(s)"
          + (f" across sectors {sectors_found}" if sectors_found else "") + ".")
    if sector is None and not all_sectors and sectors_found:
        # A single sector is the safe default for interactive vetting.
        # Multi-sector stitching remains available explicitly via all_sectors=True.
        first_sector = sectors_found[0]
        search = _with_retries(
            lambda: lk.search_lightcurve(target_str, mission="TESS", author=author, sector=first_sector))
        if len(search) == 0:
            search = _with_retries(
                lambda: lk.search_lightcurve(target_str, mission="TESS", sector=first_sector))
        print(f"   No sector selected; using first available sector {first_sector}. "
              "Pass all_sectors=True to stitch every available sector.")

    print(f"   Downloading ({_time.time() - t0:.0f}s elapsed so far)...")

    # Lightkurve's SearchResult.download_all() always forwards
    # quality_bitmask into lk.read().  That is correct for SPOC/TESS
    # readers, but it breaks on generic HLSP products such as the 2026
    # TARS light curves because read_generic_lightcurve() does not accept
    # that keyword.  Prefer the normal Lightkurve path when the selected
    # products have a supported author; otherwise download the FITS file
    # directly through MAST and parse its table ourselves.
    authors = []
    try:
        if "author" in search.table.colnames:
            authors = [str(a).upper() for a in search.table["author"]]
    except Exception:
        pass

    generic_provenance = {
        "TARS": "tars",
        "QLP": "qlp",
        "TASOC": "tasoc",
        "PATHOS": "pathos",
        "CDIPS": "cdips",
        "TGLC": "tglc",
        "TESS-YSO": "tess-yso",
    }
    provenance = next((generic_provenance[a] for a in authors if a in generic_provenance), None)
    is_generic_hlsp = provenance is not None

    if not is_generic_hlsp:
        lc_collection = _with_retries(lambda: search.download_all())
        if lc_collection is None or len(lc_collection) == 0:
            raise RuntimeError(
                f"MAST returned products for {target_str} but none could be downloaded/read "
                "(lightkurve returned no light curves)."
            )
        print(f"   Download complete in {_time.time() - t0:.0f}s total. Stitching light curve...")
        lc = lc_collection.stitch().remove_nans()

        time = np.ascontiguousarray(lc.time.value, dtype=np.float64)
        flux = np.ascontiguousarray(lc.flux.value, dtype=np.float64)
        if lc.flux_err is not None:
            flux_err = np.ascontiguousarray(lc.flux_err.value, dtype=np.float64)
        else:
            flux_err = np.full_like(flux, np.nan)  # clean_and_flatten substitutes a robust estimate
        if len(time) == 0:
            raise ValueError(f"Light curve for {target_str} is empty after removing NaN cadences.")
        print(f"   Stitched light curve has {len(time)} points.")
        return time, flux, flux_err

    print("   Generic HLSP product detected; bypassing Lightkurve quality_bitmask reader.")
    time, flux, flux_err = _download_and_read_generic_hlsp(
        tic_id, sector, t0, provenance=provenance, apply_quality_mask=qlp_quality_mask)
    print(f"   Download complete in {_time.time() - t0:.0f}s total. Loaded {len(time)} points.")
    return time, flux, flux_err


def _download_and_read_generic_hlsp(tic_id: int, sector: int, t0: float, provenance: str = "tars",
                                    apply_quality_mask: bool = False):
    """Download a generic TESS HLSP FITS light curve and read it directly.

    This avoids Lightkurve's ``quality_bitmask`` argument, which is not
    accepted by its generic FITS reader.  TARS is currently the important
    case: MAST documents its ``_lc.fits`` products as tabular FITS files.
    """
    target_str = f"TIC {tic_id}"
    _set_mast_timeout(MAST_TIMEOUT_SECONDS)

    obs = Observations.query_criteria(
        provenance_name=provenance,
        target_name=target_str,
        sequence_number=sector,
    )
    if len(obs) == 0:
        # Some MAST tables store the TIC identifier without the ``TIC`` prefix.
        obs = Observations.query_criteria(
            provenance_name=provenance,
            target_name=str(tic_id),
            sequence_number=sector,
        )
    if len(obs) == 0:
        raise ValueError(f"No compatible generic HLSP light curve found for {target_str} sector {sector}.")

    products = Observations.get_product_list(obs)
    if len(products) == 0:
        raise ValueError(f"MAST returned no products for {target_str} sector {sector}.")

    # Prefer the actual light-curve FITS product and never the preview image.
    mask = []
    names = [str(x).lower() for x in products["productFilename"]]
    for i, name in enumerate(names):
        if name.endswith("_lc.fits") or name.endswith("_llc.fits"):
            mask.append(i)
    if not mask:
        raise ValueError(f"MAST returned no FITS light-curve product for {target_str} sector {sector}.")
    products = products[mask[:1]]

    manifest = _with_retries(lambda: Observations.download_products(products))
    paths = []
    if manifest is not None and "Local Path" in manifest.colnames:
        paths = [str(x) for x in manifest["Local Path"] if str(x) and str(x) != "nan"]
    if not paths:
        raise RuntimeError("MAST downloaded the HLSP product but did not return a local file path.")

    path = paths[0]
    return _read_generic_fits_lightcurve(path, provenance=provenance, apply_quality_mask=apply_quality_mask)


def _read_generic_fits_lightcurve(path: str, provenance: str = None, apply_quality_mask: bool = False):
    """Read common TESS/HLSP tabular FITS light-curve layouts."""
    time_names = ("TIME", "BTJD", "BJD", "BJD_TDB")
    flux_names = ("PDCSAP_FLUX", "FLUX", "FLUX_CORR", "SAP_FLUX", "FLUX_RAW")
    err_names = ("PDCSAP_FLUX_ERR", "FLUX_ERR", "FLUX_CORR_ERR", "SAP_FLUX_ERR", "FLUX_RAW_ERR")

    with fits.open(path, memmap=False) as hdul:
        table = None
        for hdu in hdul:
            if not hasattr(hdu, "columns") or hdu.data is None:
                continue
            names = {str(n).upper(): str(n) for n in hdu.columns.names}
            if any(n in names for n in time_names) and any(n in names for n in flux_names):
                table = (hdu.data, names)
                break
        if table is None:
            raise ValueError(f"Could not find a TIME + flux table in {path}")

        data, names = table
        qcol = (names.get("QUALITY")
                if apply_quality_mask and str(provenance or "").lower() == "qlp" else None)
        quality = np.asarray(data[qcol]) if qcol is not None else None
        tcol = next(names[n] for n in time_names if n in names)
        fcol = next(names[n] for n in flux_names if n in names)
        ecol = next((names[n] for n in err_names if n in names), None)

        time = np.asarray(data[tcol], dtype=np.float64)
        flux = np.asarray(data[fcol], dtype=np.float64)
        if ecol is not None:
            flux_err = np.asarray(data[ecol], dtype=np.float64)
        else:
            flux_err = np.full_like(flux, np.nan, dtype=np.float64)

    # Keep only finite samples.  Generic HLSP quality flags have different
    # semantics, so do not blindly apply a SPOC bitmask to them.
    good = np.isfinite(time) & np.isfinite(flux)
    if quality is not None:
        # Opt-in only. QLP documents QUALITY == 0 as the good-cadence set, but
        # on the bundled files this removes 35-45% of cadences and changes
        # vetting outcomes, and the real-TESS training cache was built WITHOUT
        # it -- enabling it at inference alone would be a train/inference
        # mismatch. Retrain with it enabled before turning it on in production.
        good &= (quality == 0)
    time, flux, flux_err = time[good], flux[good], flux_err[good]
    if len(time) == 0:
        raise ValueError(f"No usable cadences in {path} after quality/NaN filtering")

    # If the HLSP has no usable uncertainties, return NaN and let
    # clean_and_flatten estimate the noise from the DETRENDED flux. (The old
    # fallback used the MAD of the raw flux, which includes stellar
    # variability and instrumental trends the cleaner removes; on the bundled
    # QLP files that overstated the noise 1.1-6x and understated BLS SNR by the
    # same factor, e.g. 38.7 vs 244. Constant errors do not change which BLS
    # peak wins, only the SNR scale.)
    valid_err = np.isfinite(flux_err) & (flux_err > 0)
    if not np.any(valid_err):
        flux_err = np.full_like(flux, np.nan, dtype=np.float64)
    else:
        fallback = np.nanmedian(flux_err[valid_err])
        flux_err[~valid_err] = fallback

    order = np.argsort(time)
    return (
        np.ascontiguousarray(time[order], dtype=np.float64),
        np.ascontiguousarray(flux[order], dtype=np.float64),
        np.ascontiguousarray(flux_err[order], dtype=np.float64),
    )
