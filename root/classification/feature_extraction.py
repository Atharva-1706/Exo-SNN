"""Compatibility feature extraction for the current detection stack.

The old RF-era classes (ClassicalBLS, SpikingTransitDetector, TriBranchEnsemble)
were removed when the active pipeline was consolidated. This module now exposes
the same public feature names using the active BLS + delta-modulation code.
"""
import numpy as np

from root.classification.delta_modulation import DeltaModulator
from root.detection.bls_detector import run_bls

FEATURE_NAMES = [
    "bls_snr", "bls_depth", "bls_period",
    "snn_spike_density", "snn_burst_duration", "snn_peak_membrane",
    "cnn_probability", "transformer_probability",
]


def _snn_metrics(flux):
    mod = DeltaModulator()
    up, down = mod.encode(flux)
    spikes = np.asarray(up, dtype=bool) | np.asarray(down, dtype=bool)
    idx = np.flatnonzero(spikes)
    bursts = np.diff(idx) if len(idx) > 1 else np.array([], dtype=int)
    return {
        "spike_density": float(spikes.mean()),
        "mean_burst_duration": float(np.mean(bursts)) if len(bursts) else 0.0,
        "peak_membrane_potential": float(np.sum(down)),
    }


def features_from_metrics(bls_metrics, snn_metrics, ensemble_result=None):
    ensemble_result = ensemble_result or {}
    return [
        float(bls_metrics.get("snr", 0.0)),
        float(bls_metrics.get("depth", 0.0)),
        float(bls_metrics.get("period", 0.0)),
        float(snn_metrics.get("spike_density", 0.0)),
        float(snn_metrics.get("mean_burst_duration", 0.0)),
        float(snn_metrics.get("peak_membrane_potential", 0.0)),
        float(ensemble_result.get("cnn_confidence", 0.0)),
        float(ensemble_result.get("transformer_confidence", 0.0)),
    ]


def extract_detection_features(time, flux, ensemble=None, delta_modulator=None, snn_detector=None,
                                bls_detector=None, verbose=False, precomputed_bls_metrics=None):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if len(time) != len(flux) or len(time) == 0:
        raise ValueError("time and flux must have the same non-zero length")

    snn_metrics = _snn_metrics(flux) if snn_detector is None else snn_detector.extract_features(time, *DeltaModulator().encode(flux))
    bls_metrics = precomputed_bls_metrics if precomputed_bls_metrics is not None else run_bls(time, flux)
    ensemble_result = {}
    if ensemble is not None and hasattr(ensemble, "aggregate"):
        ensemble_result = ensemble.aggregate(flux, bls_metrics.get("snr", 0.0), snn_metrics,
                                             bls_metrics=bls_metrics, time=time, verbose=verbose)
    features = features_from_metrics(bls_metrics, snn_metrics, ensemble_result)
    return features, snn_metrics, bls_metrics, ensemble_result


def build_training_dataset(n_samples=100, seed=42, ensemble=None, points_per_day=360, verbose=True):
    from root.data.dataset_builder import build_clean_dataset
    samples = build_clean_dataset(n_samples=n_samples, seed=seed, points_per_day=points_per_day)
    X, y = [], []
    for i, sample in enumerate(samples):
        if verbose and i % 50 == 0:
            print(f"      Extracting features {i}/{len(samples)}...")
        features, *_ = extract_detection_features(
            sample.time, sample.flux, ensemble, precomputed_bls_metrics=sample.bls_metrics
        )
        X.append(features); y.append(sample.label)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), samples
