"""
Train the production TriBranchTESSNet classifier.

v8 training changes:
  * harder negatives plus physical-feature fusion: eclipsing binaries, periodic stellar variability,
    repeated instrumental dips and aperiodic artifacts;
  * richer planet morphology: trapezoidal ingress/egress, dilution,
    shallow/deep transits, variable duration/period and realistic noise;
  * GroupNorm + per-sample view normalization for train/inference parity;
  * the SNN branch now sees the complete 61-point temporal sequence;
  * deterministic stratified train/validation split;
  * checkpoint selection uses validation ROC-AUC rather than the last epoch.

This is still a synthetic pretraining stage. A real-TESS holdout remains
necessary before interpreting the network score as a calibrated probability.
"""
import argparse
import copy
import os
import time as _time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from root.data.synthetic_data import LightCurveSimulator
from root.preprocessing.clean import clean_and_flatten
from root.detection.bls_detector import run_bls
from root.detection.dual_stream import extract_dual_views
from root.classification.tri_branch_ensemble import TriBranchTESSNet
from root.classification.physical_verification import verify_astrophysics

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
DEFAULT_PRETRAINED_PATH = os.path.join(WEIGHTS_DIR, "tri_branch_tess_net_pretrained.pt")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_physical_features(time, flux, bls_res):
    """Return the v8 physical-measurement vector used by both training and inference.

    The final rule-based EB flag is intentionally excluded to prevent label leakage.
    Missing odd/even or secondary measurements become zero; observed transit count
    tells the network whether those measurements were actually supported.
    """
    phys = verify_astrophysics(
        time, flux, bls_res["period"], bls_res["t0"], bls_res["duration"]
    )
    finite = lambda x: float(x) if np.isfinite(x) else 0.0
    baseline = np.nanmedian(flux)
    outside = np.abs(flux - baseline)
    # Robust baseline variability estimate in normalized-flux units.
    baseline_var = float(np.nanmedian(np.abs(flux - np.nanmedian(flux))) * 1.4826)
    return [
        1.0, 5778.0 / 5000.0, 4.44 / 4.0,
        bls_res["period"] / 10.0,
        bls_res["snr"] / 20.0,
        bls_res["depth"] * 100.0,
        bls_res["duration"] / 0.2,
        finite(phys.get("odd_depth")) * 100.0,
        finite(phys.get("even_depth")) * 100.0,
        finite(phys.get("odd_even_mismatch_pct")) / 100.0,
        finite(phys.get("secondary_depth")) * 100.0,
        phys.get("observed_transits", 0) / 10.0,
        baseline_var / 0.01,
    ]


def _process_one_sample(sample):
    idx, time, flux, meta, seed = sample
    rng = np.random.default_rng(seed)
    try:
        t_clean, f_clean, err_clean = clean_and_flatten(time, flux)
        bls_res = run_bls(t_clean, f_clean, err_clean)
        g_view, l_view = extract_dual_views(
            t_clean, f_clean, bls_res["period"], bls_res["t0"], bls_res["duration"]
        )
        # Stellar parameters remain nuisance context, while the remaining
        # features are measured from the candidate itself. The physical EB
        # verdict is deliberately not passed to the model.
        r_star = rng.uniform(0.7, 1.8)
        teff = rng.uniform(4500, 7000)
        logg = rng.uniform(3.8, 4.6)
        phys = verify_astrophysics(
            t_clean, f_clean, bls_res["period"], bls_res["t0"], bls_res["duration"]
        )
        finite = lambda x: float(x) if np.isfinite(x) else 0.0
        baseline_var = float(np.nanmedian(np.abs(f_clean - np.nanmedian(f_clean))) * 1.4826)
        tab = [
            r_star, teff / 5000.0, logg / 4.0,
            bls_res["period"] / 10.0,
            bls_res["snr"] / 20.0,
            bls_res["depth"] * 100.0,
            bls_res["duration"] / 0.2,
            finite(phys.get("odd_depth")) * 100.0,
            finite(phys.get("even_depth")) * 100.0,
            finite(phys.get("odd_even_mismatch_pct")) / 100.0,
            finite(phys.get("secondary_depth")) * 100.0,
            phys.get("observed_transits", 0) / 10.0,
            baseline_var / 0.01,
        ]
        return idx, g_view, l_view, tab, int(meta["label"]), None
    except Exception as e:
        return idx, None, None, None, None, f"{type(e).__name__}: {e}"


def build_dataset(n_samples, seed, points_per_day, days, verbose=True,
                  workers=None, progress_every=50):
    if workers is None:
        workers = max(1, cpu_count() - 1)

    sim = LightCurveSimulator(days=days, points_per_day=points_per_day, seed=seed)
    raw_samples = sim.generate_dataset(n_samples=n_samples)
    jobs = [(i, t, f, m, seed + i + 1) for i, (t, f, m) in enumerate(raw_samples)]

    results = [None] * len(jobs)
    n_failed = 0
    n_done = 0
    t_start = _time.time()

    if verbose:
        print(f"   Running {len(jobs)} samples through clean -> BLS -> dual-view "
              f"using {workers} worker process(es)...")

    if workers <= 1:
        iterator = (_process_one_sample(job) for job in jobs)
        pool = None
    else:
        pool = Pool(processes=workers)
        iterator = pool.imap_unordered(_process_one_sample, jobs, chunksize=4)

    try:
        for idx, g_view, l_view, tab, label, err in iterator:
            n_done += 1
            if err is not None:
                n_failed += 1
                if verbose:
                    print(f"      (skipped sample {idx} -- {err})")
            else:
                results[idx] = (g_view, l_view, tab, label)

            if verbose and progress_every and n_done % progress_every == 0:
                elapsed = _time.time() - t_start
                rate = n_done / max(elapsed, 1e-6)
                eta = (len(jobs) - n_done) / rate
                print(f"      ...{n_done}/{len(jobs)} samples done "
                      f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    usable = [r for r in results if r is not None]
    G = np.asarray([r[0] for r in usable], dtype=np.float32)
    L = np.asarray([r[1] for r in usable], dtype=np.float32)
    T = np.asarray([r[2] for r in usable], dtype=np.float32)
    y = np.asarray([r[3] for r in usable], dtype=np.int64)

    if verbose:
        print(f"   Built {len(y)} usable samples ({n_failed} skipped) "
              f"in {_time.time() - t_start:.1f}s.")
        print(f"   Class balance: planet={int(y.sum())}, false-positive={int((y == 0).sum())}")

    return G, L, T, y


def stratified_split(y, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def evaluate(model, G, L, T, y):
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(G).unsqueeze(1),
            torch.from_numpy(L).unsqueeze(1),
            torch.from_numpy(T)
        )
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"),
        "average_precision": average_precision_score(y, probs) if len(np.unique(y)) > 1 else float("nan"),
        "mean_planet_score": float(np.mean(probs[y == 1])) if np.any(y == 1) else float("nan"),
        "mean_fp_score": float(np.mean(probs[y == 0])) if np.any(y == 0) else float("nan"),
    }
    return metrics


def train(model, G_train, L_train, T_train, y_train,
          G_val, L_val, T_val, y_val,
          epochs=30, batch_size=32, lr=5e-4, weight_decay=1e-4, verbose=True):
    g_t = torch.from_numpy(G_train).unsqueeze(1)
    l_t = torch.from_numpy(L_train).unsqueeze(1)
    t_t = torch.from_numpy(T_train)
    y_t = torch.from_numpy(y_train)

    g_v = torch.from_numpy(G_val).unsqueeze(1)
    l_v = torch.from_numpy(L_val).unsqueeze(1)
    t_v = torch.from_numpy(T_val)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    n = len(y_train)
    best_auc = -np.inf
    best_state = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(g_t[idx], l_t[idx], t_t[idx])
            loss = criterion(logits, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)

        metrics = evaluate(model, G_val, L_val, T_val, y_val)
        scheduler.step(metrics["roc_auc"])

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if verbose:
            print(
                f"      epoch {epoch + 1:02d}/{epochs} "
                f"loss={total_loss/n:.4f} "
                f"val_acc={metrics['accuracy']:.3f} "
                f"val_auc={metrics['roc_auc']:.3f} "
                f"val_f1={metrics['f1']:.3f} "
                f"planet={metrics['mean_planet_score']:.3f} "
                f"fp={metrics['mean_fp_score']:.3f}"
            )

        if stale >= 7 and epoch >= 12:
            if verbose:
                print("      early stopping: validation AUC has stopped improving.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def main():
    parser = argparse.ArgumentParser(
        description="Train TriBranchTESSNet on realistic synthetic TESS-like light curves."
    )
    parser.add_argument("--n-samples", type=int, default=1200,
                        help="Total synthetic samples; balanced planet/false-positive.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points-per-day", type=int, default=360,
                        help="Synthetic cadence. Use 360 for a good speed/realism compromise; "
                             "720 approximates native 2-minute TESS cadence.")
    parser.add_argument("--days", type=int, default=27)
    parser.add_argument("--weights-out", type=str, default=DEFAULT_PRETRAINED_PATH)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    seed_everything(args.seed)

    print("=== Step 1/3: Building realistic synthetic training dataset ===")
    t0 = _time.time()
    G, L, T, y = build_dataset(
        args.n_samples, args.seed, args.points_per_day, args.days,
        workers=args.workers, progress_every=args.progress_every,
    )
    print(f"   Done in {_time.time() - t0:.1f}s.")
    print(f"   Global={G.shape} Local={L.shape} Tabular={T.shape} Labels={y.shape}")

    train_idx, val_idx = stratified_split(y, args.val_frac, args.seed)
    print(f"   Train={len(train_idx)}  Validation={len(val_idx)}")

    print("=== Step 2/3: Training TriBranchTESSNet ===")
    model = TriBranchTESSNet(tabular_dim=T.shape[1])
    t0 = _time.time()
    model, best_auc = train(
        model,
        G[train_idx], L[train_idx], T[train_idx], y[train_idx],
        G[val_idx], L[val_idx], T[val_idx], y[val_idx],
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
    )
    print(f"   Training done in {_time.time() - t0:.1f}s. Best validation AUC={best_auc:.4f}")

    print("=== Step 3/3: Final validation + saving weights ===")
    metrics = evaluate(model, G[val_idx], L[val_idx], T[val_idx], y[val_idx])
    for k, v in metrics.items():
        print(f"   val {k}: {v:.4f}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    checkpoint = {
        "format_version": 3,
        "model_contract": "groupnorm + per_sample_zscore + realistic_hard_negative_pretraining",
        "view_normalization": "per_sample_zscore",
        "batchnorm_running_stats": False,
        "normalization_layer": "GroupNorm",
        "snn_temporal_sequence": 61,
        "synthetic_negative_mix": {
            "eclipsing_binary": 0.45,
            "stellar_variability": 0.25,
            "instrumental": 0.20,
            "aperiodic_dips": 0.10,
        },
        "training_metrics": metrics,
        "seed": args.seed,
        "n_samples": int(len(y)),
        "points_per_day": args.points_per_day,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, args.weights_out)
    print(f"Saved validated-format v3 PRETRAINED checkpoint to: {args.weights_out}")
    print("NOTE: synthetic validation is a pretraining metric; run the real-TESS holdout before "
          "interpreting AI scores as calibrated probabilities.")


if __name__ == "__main__":
    main()
