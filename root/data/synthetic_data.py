"""
TESS-like synthetic light-curve generator for classifier training.

The classifier must distinguish planetary transits from *transit-shaped
false positives*, not merely from white noise.  This generator therefore
creates a mixture of:
  - trapezoidal/diluted planetary transits;
  - eclipsing-binary-like periodic signals with primary/secondary eclipses;
  - quasi-periodic stellar variability;
  - instrumental/systematic events and isolated dips.

The generator deliberately varies cadence, noise, stellar variability,
dilution, gaps and outliers so the model cannot memorize one idealized
box-transit morphology.
"""
import numpy as np


class LightCurveSimulator:
    def __init__(self, days=27, points_per_day=360, seed=None):
        self.days = float(days)
        self.points_per_day = int(points_per_day)
        self.rng = np.random.default_rng(seed)
        n = max(10, int(round(self.days * self.points_per_day)))
        self.time = np.linspace(0.0, self.days, n, endpoint=False)

    def _trapezoid(self, time, period, duration, depth, t0,
                   ingress_fraction=0.18, secondary=False, secondary_phase=0.5):
        """Return a smooth trapezoidal eclipse train."""
        p = float(period)
        d = max(float(duration), 2.0 / self.points_per_day)
        ingress = np.clip(float(ingress_fraction), 0.05, 0.45) * d
        flat = max(d - 2.0 * ingress, 0.0)
        phase0 = secondary_phase * p if secondary else 0.0
        phase = ((time - t0 - phase0 + 0.5 * p) % p) - 0.5 * p
        a = np.abs(phase)
        half = d / 2.0
        flat_half = flat / 2.0
        out = np.zeros_like(time, dtype=float)
        in_full = a <= flat_half
        out[in_full] = depth
        ramp = (a > flat_half) & (a < half)
        if np.any(ramp):
            out[ramp] = depth * (half - a[ramp]) / max(half - flat_half, 1e-12)
        return out

    def _stellar_variability(self, amplitude):
        t = self.time
        p1 = self.rng.uniform(6.0, 18.0)
        p2 = self.rng.uniform(1.5, 7.0)
        phase1 = self.rng.uniform(0, 2*np.pi)
        phase2 = self.rng.uniform(0, 2*np.pi)
        return (
            amplitude * np.sin(2*np.pi*t/p1 + phase1)
            + 0.45 * amplitude * np.sin(2*np.pi*t/p2 + phase2)
        )

    def _correlated_noise(self, sigma, rho=0.92):
        eps = self.rng.normal(0.0, sigma, len(self.time))
        x = np.empty_like(eps)
        x[0] = eps[0]
        scale = np.sqrt(max(1.0 - rho*rho, 1e-6))
        for i in range(1, len(eps)):
            x[i] = rho * x[i-1] + scale * eps[i]
        return x

    def _instrument_effects(self, flux, strength=0.003):
        out = flux.copy()
        t = self.time
        # One or two smooth local ramps/jumps.
        for _ in range(self.rng.integers(0, 3)):
            center = self.rng.uniform(0, self.days)
            width = self.rng.uniform(0.15, 1.2)
            amp = self.rng.uniform(-strength, strength)
            out += amp * np.tanh((t - center) / max(width, 1e-3))
        # Sparse outliers, mostly positive as in cosmic-ray contamination.
        n_out = max(1, int(len(t) * self.rng.uniform(0.0003, 0.0015)))
        idx = self.rng.choice(len(t), size=n_out, replace=False)
        out[idx] += self.rng.choice([-1.0, 1.0], size=n_out) * self.rng.uniform(
            0.01, 0.08, size=n_out
        )
        return out

    def _observe(self, intrinsic, noise_sigma=None, variability=None):
        if variability is None:
            variability = self.rng.uniform(0.0015, 0.012)
        if noise_sigma is None:
            noise_sigma = self.rng.uniform(0.0008, 0.006)
        flux = intrinsic + self._stellar_variability(variability)
        flux += self.rng.normal(0.0, noise_sigma, len(flux))
        flux += self._correlated_noise(noise_sigma * self.rng.uniform(0.25, 0.8))
        flux = self._instrument_effects(flux, strength=self.rng.uniform(0.001, 0.006))

        # Random missing cadences.  The production cleaner naturally removes them.
        keep = np.ones(len(flux), dtype=bool)
        for _ in range(self.rng.integers(0, 4)):
            start = self.rng.integers(0, max(1, len(flux) - 5))
            width = self.rng.integers(1, max(2, int(self.points_per_day * 0.08)))
            keep[start:min(len(keep), start + width)] = False

        # Return the same time axis with NaNs for gaps; clean_and_flatten handles them.
        observed = flux.copy()
        observed[~keep] = np.nan
        return observed

    def generate_planet_sample(self):
        period = self.rng.uniform(1.4, min(14.0, max(2.0, self.days * 0.65)))
        duration = self.rng.uniform(0.035, 0.24)  # ~0.8 h to 5.8 h
        depth = 10 ** self.rng.uniform(np.log10(0.00035), np.log10(0.03))
        t0 = self.rng.uniform(0, period)
        ingress = self.rng.uniform(0.08, 0.38)
        dilution = self.rng.uniform(0.65, 1.0)

        transit = self._trapezoid(
            self.time, period, duration, depth * dilution, t0,
            ingress_fraction=ingress
        )
        intrinsic = 1.0 - transit

        # NOTE: this branch used to rescale `duration` AFTER the transit was
        # already injected, so the recorded metadata duration disagreed with the
        # injected signal for ~18% of planets. The draws are kept (so seeded
        # datasets/checkpoints stay reproducible) but no longer touch metadata.
        # A genuine grazing/V-shaped population is not generated; adding one
        # changes the training distribution and requires retraining.
        if self.rng.random() < 0.18:
            self.rng.uniform(0.75, 1.15)

        flux = self._observe(
            intrinsic,
            noise_sigma=self.rng.uniform(0.0007, 0.0055),
            variability=self.rng.uniform(0.001, 0.010),
        )
        metadata = {
            "label": 1,
            "kind": "planet",
            "period": period,
            "duration": duration,
            "depth": depth * dilution,
            "t0": t0,
        }
        return self.time.copy(), flux, metadata

    def generate_eb_sample(self):
        """Hard negative: periodic eclipses with a primary + secondary."""
        period = self.rng.uniform(1.4, min(14.0, max(2.0, self.days * 0.65)))
        duration = self.rng.uniform(0.04, 0.28)
        primary = 10 ** self.rng.uniform(np.log10(0.0015), np.log10(0.035))
        secondary = primary * self.rng.uniform(0.12, 0.85)
        t0 = self.rng.uniform(0, period)
        secondary_phase = self.rng.uniform(0.42, 0.58)

        signal = self._trapezoid(
            self.time, period, duration, primary, t0,
            ingress_fraction=self.rng.uniform(0.12, 0.42)
        )
        signal += self._trapezoid(
            self.time, period, duration * self.rng.uniform(0.8, 1.2), secondary, t0,
            ingress_fraction=self.rng.uniform(0.12, 0.42),
            secondary=True, secondary_phase=secondary_phase
        )

        # Occasionally make alternating primary eclipses slightly different.
        phase = ((self.time - t0 + 0.5 * period) % period) - 0.5 * period
        cycle = np.floor((self.time - t0) / period).astype(int)
        odd = (cycle % 2) != 0
        alt = self.rng.uniform(0.78, 1.22)
        signal = np.where(odd & (np.abs(phase) < duration/2), signal * alt, signal)

        flux = self._observe(
            1.0 - signal,
            noise_sigma=self.rng.uniform(0.0008, 0.005),
            variability=self.rng.uniform(0.001, 0.009),
        )
        return self.time.copy(), flux, {
            "label": 0, "kind": "eclipsing_binary", "period": period,
            "duration": duration, "depth": primary, "t0": t0,
            "secondary_depth": secondary,
        }

    def generate_variable_sample(self):
        """Periodic/quasi-periodic stellar variability with transit-like minima."""
        t = self.time
        p = self.rng.uniform(1.8, 14.0)
        amp = self.rng.uniform(0.003, 0.025)
        harmonic = self.rng.uniform(0.15, 0.55)
        phase = self.rng.uniform(0, 2*np.pi)
        x = 2*np.pi*t/p + phase
        intrinsic = 1.0 + amp*np.sin(x) + harmonic*amp*np.sin(2*x + self.rng.uniform(0, 2*np.pi))
        # A smooth localized feature makes some negatives deceptively transit-like.
        if self.rng.random() < 0.7:
            centers = self.rng.uniform(0, self.days, self.rng.integers(1, 4))
            for c in centers:
                width = self.rng.uniform(0.03, 0.25)
                intrinsic -= self.rng.uniform(0.002, 0.012) * np.exp(
                    -0.5*((t-c)/width)**2
                )
        flux = self._observe(
            intrinsic,
            noise_sigma=self.rng.uniform(0.001, 0.006),
            variability=self.rng.uniform(0.001, 0.008),
        )
        return self.time.copy(), flux, {
            "label": 0, "kind": "stellar_variability",
            "period": p, "duration": None, "depth": None, "t0": None
        }

    def generate_artifact_sample(self):
        """Instrumental/systematic false positives, including repeated dips."""
        flux = np.ones_like(self.time)
        t = self.time
        # Repeated events at an unrelated cadence can produce a strong BLS peak.
        if self.rng.random() < 0.65:
            p = self.rng.uniform(2.0, 12.0)
            t0 = self.rng.uniform(0, p)
            duration = self.rng.uniform(0.03, 0.18)
            depth = self.rng.uniform(0.002, 0.02)
            signal = self._trapezoid(t, p, duration, depth, t0,
                                     ingress_fraction=self.rng.uniform(0.05, 0.45))
            flux -= signal
        # Add a sector-like ramp and local discontinuity.
        center = self.rng.uniform(5, max(6, self.days-5))
        flux += self.rng.uniform(-0.01, 0.01) * np.tanh((t-center)/self.rng.uniform(0.05, 0.5))
        flux = self._observe(
            flux,
            noise_sigma=self.rng.uniform(0.001, 0.007),
            variability=self.rng.uniform(0.001, 0.012),
        )
        return self.time.copy(), flux, {
            "label": 0, "kind": "instrumental",
            "period": None, "duration": None, "depth": None, "t0": None
        }

    def generate_false_positive_sample(self):
        # Balanced mixture of hard negatives; EB is intentionally common.
        r = self.rng.random()
        if r < 0.45:
            return self.generate_eb_sample()
        if r < 0.70:
            return self.generate_variable_sample()
        if r < 0.90:
            return self.generate_artifact_sample()

        # Isolated dip / non-periodic noise.
        flux = np.ones_like(self.time)
        for _ in range(self.rng.integers(2, 7)):
            center = self.rng.uniform(0, self.days)
            width = self.rng.uniform(0.015, 0.10)
            depth = self.rng.uniform(0.002, 0.018)
            flux -= depth * np.exp(-0.5*((self.time-center)/width)**2)
        flux = self._observe(
            flux,
            noise_sigma=self.rng.uniform(0.001, 0.007),
            variability=self.rng.uniform(0.001, 0.012),
        )
        return self.time.copy(), flux, {
            "label": 0, "kind": "aperiodic_dips",
            "period": None, "duration": None, "depth": None, "t0": None
        }

    def generate_dataset(self, n_samples=100):
        n_planets = n_samples // 2
        samples = [self.generate_planet_sample() for _ in range(n_planets)]
        samples += [self.generate_false_positive_sample() for _ in range(n_samples - n_planets)]
        self.rng.shuffle(samples)
        return samples

    def get_mock_data(self):
        t, f, _ = self.generate_planet_sample()
        return t, f


if __name__ == "__main__":
    sim = LightCurveSimulator(seed=42)
    samples = sim.generate_dataset(10)
    print(f"Generated {len(samples)} samples; planets={sum(m['label'] for _,_,m in samples)}")
