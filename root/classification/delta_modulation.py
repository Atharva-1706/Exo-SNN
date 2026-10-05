"""
Spike Encoding for Neuromorphic Processing (Delta Modulation)

    light curve
         |
    estimate noise sigma  (MAD of point-to-point differences)
         |
    threshold = k * sigma
         |
    delta modulation
         |
    UP / DOWN spikes

Previously this used a hardcoded threshold=0.0015 regardless of the
star's actual photometric noise level. That is not scientifically
defensible: a quiet star (sigma ~ 0.001) would spike on nearly every
noise fluctuation (over-triggering), while a noisy star (sigma ~ 0.01)
would almost never cross 0.0015 and the encoder would degenerate into
near-total silence. Empirically, on this codebase's own synthetic
targets, the old fixed threshold caused 30-45% of ALL samples to spike
-- i.e. it was mostly encoding noise, not transits.

The threshold is now estimated per-light-curve from the median absolute
deviation (MAD) of first differences, which is a standard robust noise
estimator that isn't thrown off by the transit itself (a single sustained
dip barely moves the MAD of point-to-point diffs, unlike using np.std of
the raw flux, which the transit itself would inflate).
"""
import numpy as np


class DeltaModulator:
    def __init__(self, threshold=None, k=2.0):
        """
        Parameters
        ----------
        threshold : float or None
            Explicit flux-delta threshold. If None (default), the
            threshold is estimated automatically from the light curve's
            own noise level at encode() time -- this is the recommended
            mode. Pass an explicit float only for testing/reproducing a
            fixed threshold.
        k : float
            Multiplier on the estimated 1-sigma noise level when
            auto-estimating the threshold.

            Important calibration note: because `threshold` is derived
            from the SAME sample-to-sample noise distribution it is then
            applied to, the fraction of pure-noise samples that spike is
            approximately fixed by the Gaussian tail probability at `k`
            sigma, REGARDLESS of the star's actual noise amplitude (that
            invariance is the point -- it's what makes this "adaptive").
            But it means `k` alone controls baseline spike density:
            k=1.5 -> ~13% of samples spike on pure noise (too dense,
            noise dominates); k=2.0 -> ~2-3% (usable); k>=3 -> <0.2%
            (the encoder goes almost completely silent and even real
            transit edges stop firing). k=2.0 was chosen empirically
            (see tests/test_delta.py, tests/test_snn.py) as the point
            where real transit edges still reliably produce a detectable
            spike cluster while the noise floor stays reasonably sparse.
        """
        self.threshold = threshold
        self.k = k
        self.last_threshold = None  # populated after encode(), for introspection

    @staticmethod
    def estimate_noise_threshold(flux, k=2.0):
        """Robust 1-sigma-equivalent noise estimate from point-to-point diffs."""
        diffs = np.diff(flux)
        mad = np.median(np.abs(diffs - np.median(diffs)))
        sigma_equiv = mad / 0.6745  # MAD -> Gaussian sigma conversion constant
        return max(k * sigma_equiv, 1e-8)

    def encode(self, flux):
        flux = np.asarray(flux)
        threshold = self.threshold if self.threshold is not None else self.estimate_noise_threshold(flux, self.k)
        self.last_threshold = float(threshold)

        up_spikes = np.zeros_like(flux, dtype=np.int8)
        down_spikes = np.zeros_like(flux, dtype=np.int8)

        base_val = flux[0]
        for i in range(1, len(flux)):
            delta = flux[i] - base_val
            if delta >= threshold:
                up_spikes[i] = 1
                base_val = flux[i]
            elif delta <= -threshold:
                down_spikes[i] = 1
                base_val = flux[i]

        return up_spikes, down_spikes