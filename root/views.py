"""
Phase-folded "local view" construction for the CNN/Transformer branches.

Root-cause fix (found while training, not called out explicitly in the
original roadmap doc, but the doc anticipated the symptom: "If the
Transformer doesn't improve performance, we don't need it"): feeding the
CNN/Transformer the raw global light curve, resampled down to a fixed
1440-point view of the ENTIRE observation baseline, was diluting a
typical transit down to ~5 points out of 1440 (measured: a 0.09-day
transit in a 27-day baseline is 0.34% of the baseline). No CNN is going
to learn a transit's shape from 5 points buried in 1440, and empirically
neither the CNN nor the Transformer branch's training loss moved off the
chance-level baseline (~ln(2)) when trained this way.

This is the same problem real transit-classification pipelines solve by
NOT feeding a raw global view: Shallue & Vanderburg's AstroNet (the
reference architecture for this exact task) phase-folds the light curve
on the BLS candidate period and zooms into a window around the transit
("local view"), typically a few transit durations wide, alongside a
coarser global view. We implement the local view here -- it's the
higher-leverage of the two for a light, single-branch CNN/Transformer,
and is what actually gives the branches a transit shape to learn from.
"""
import numpy as np


def phase_fold_local_view(time, flux, period, t0, duration, n_points=1440, view_span_in_durations=8.0):
    """
    Phase-fold `time`/`flux` on the given ephemeris and extract a fixed-
    length local view centered on phase 0 (the transit), spanning
    `view_span_in_durations` x the transit duration on each side.

    Falls back gracefully (returns a plain resample of the whole curve)
    if period/duration are missing or degenerate, so this is always safe
    to call even on a non-detection.
    """
    flux = np.asarray(flux, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)

    if not period or period <= 0 or not duration or duration <= 0 or len(time) < 2:
        return normalize_view(_resample(flux, n_points))

    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period  # in [-period/2, period/2)

    half_window = max(view_span_in_durations * duration, 3 * duration)
    in_window = np.abs(phase) <= half_window
    if in_window.sum() < 8:
        # Window too narrow / too few points captured -- widen once, then give up to global view.
        half_window *= 3
        in_window = np.abs(phase) <= half_window
        if in_window.sum() < 8:
            return normalize_view(_resample(flux, n_points))

    local_phase = phase[in_window]
    local_flux = flux[in_window]

    order = np.argsort(local_phase)
    local_phase = local_phase[order]
    local_flux = local_flux[order]

    # Resample onto a fixed uniform phase grid spanning [-half_window, half_window].
    target_phase = np.linspace(-half_window, half_window, n_points)
    # np.interp needs a strictly increasing x; duplicate phase values (rare,
    # e.g. two samples folding to ~identical phase) are harmless for interp.
    resampled = np.interp(target_phase, local_phase, local_flux, left=local_flux[0], right=local_flux[-1])
    return normalize_view(resampled)


def _resample(flux, n_points):
    if len(flux) == n_points:
        return flux.astype(np.float32)
    original_x = np.linspace(0, 1, len(flux))
    target_x = np.linspace(0, 1, n_points)
    return np.interp(target_x, original_x, flux).astype(np.float32)


def normalize_view(flux):
    """
    Per-sample zero-mean, unit-std normalization.

    This turned out to matter a lot more than the phase-folding fix
    above: flux arrays sit around a baseline of ~1.0 with a transit
    signal of amplitude ~0.005-0.03 on top. Fed raw into the CNN/
    Transformer, that huge constant DC offset relative to the tiny
    class-discriminating signal dominated the first-layer activations
    and the networks never moved off a chance-level loss (~ln(2)) no
    matter how long they trained -- confirmed empirically: identical
    architecture, identical (correctly phase-folded) local-view data,
    only difference is this normalization, val accuracy went from
    ~50% (chance) to ~93% (see tests/test_ensemble.py). Must be applied
    identically at training time and at inference time (TriBranchEnsemble
    .aggregate() calls this too) or it's yet another train/inference
    mismatch.
    """
    flux = np.asarray(flux, dtype=np.float32)
    mean = flux.mean()
    std = flux.std() + 1e-8
    return ((flux - mean) / std).astype(np.float32)