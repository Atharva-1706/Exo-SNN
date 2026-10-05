"""Compatibility dataset builder using the current active pipeline API."""
import os
import pickle
import numpy as np

from root.data.synthetic_data import LightCurveSimulator
from root.preprocessing.clean import clean_and_flatten
from root.detection.bls_detector import run_bls

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "dataset_cache")


class CleanedSample:
    __slots__ = ("time", "flux", "label", "metadata", "n_removed", "bls_metrics")

    def __init__(self, time, flux, label, metadata, n_removed, bls_metrics=None):
        self.time = time
        self.flux = flux
        self.label = label
        self.metadata = metadata
        self.n_removed = n_removed
        self.bls_metrics = bls_metrics


def build_clean_dataset(n_samples=600, seed=42, points_per_day=360, cache=True, cache_name=None, run_bls=True):
    cache_name = cache_name or f"clean_n{n_samples}_seed{seed}_ppd{points_per_day}_bls{run_bls}.pkl"
    cache_path = os.path.join(CACHE_DIR, cache_name)
    if cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    simulator = LightCurveSimulator(points_per_day=points_per_day, seed=seed)
    raw_samples = simulator.generate_dataset(n_samples=n_samples)
    cleaned = []
    for time, flux, metadata in raw_samples:
        t_clean, flat, err = clean_and_flatten(time, flux)
        bls_metrics = run_bls(t_clean, flat, err) if run_bls else None
        cleaned.append(CleanedSample(
            time=t_clean, flux=flat, label=metadata["label"], metadata=metadata,
            n_removed=len(time) - len(t_clean), bls_metrics=bls_metrics,
        ))

    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cleaned, f)
    return cleaned


def split_dataset(samples, train_frac=0.7, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_train = int(len(samples) * train_frac)
    n_val = int(len(samples) * val_frac)
    return ([samples[i] for i in idx[:n_train]],
            [samples[i] for i in idx[n_train:n_train+n_val]],
            [samples[i] for i in idx[n_train+n_val:]])


def resample_flux(flux, target_length=1440):
    if len(flux) == target_length:
        return np.asarray(flux, dtype=np.float32)
    return np.interp(np.linspace(0, 1, target_length), np.linspace(0, 1, len(flux)), flux).astype(np.float32)


def to_cnn_arrays(samples, target_length=1440):
    from root.views import normalize_view
    X = np.stack([normalize_view(resample_flux(s.flux, target_length)) for s in samples])
    y = np.array([s.label for s in samples], dtype=np.float32)
    return X, y


def to_local_view_arrays(samples, target_length=1440, view_span_in_durations=8.0):
    from root.views import phase_fold_local_view
    X = []
    for s in samples:
        b = s.bls_metrics or {}
        X.append(phase_fold_local_view(
            s.time, s.flux, b.get("period"), b.get("t0"), b.get("duration"),
            n_points=target_length, view_span_in_durations=view_span_in_durations,
        ))
    return np.stack(X), np.array([s.label for s in samples], dtype=np.float32)
