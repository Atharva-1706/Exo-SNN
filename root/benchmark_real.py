"""Build and run a leakage-safe real-TESS benchmark for the v9 model.

The benchmark is deliberately separated from training/validation. It reads the
saved real-TESS checkpoint metadata and excludes every TIC listed as train,
validation, or explicit holdout. It then samples a fresh, deterministic set of
confirmed planets (CP/KP) and false positives/alarms (FP/FA) from the NASA
Exoplanet Archive TOI table, chooses a sector using the same timing-aware logic
used by real-data training, runs the full inference pipeline, and writes CSV +
JSON results and aggregate metrics.

Typical use:
  python -m root.benchmark_real --weights root/weights/tri_branch_tess_net.pt --n-per-class 20

Selection only (no MAST light-curve downloads):
  python -m root.benchmark_real --weights root/weights/tri_branch_tess_net.pt --n-per-class 20 --select-only

The benchmark TICs are never used to tune the model. Do not change the
checkpoint after running this benchmark if you want an honest test result.
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from root.checkpoint_io import load_checkpoint
from root.main_pipeline import Pipeline
from root.train_real import archive_query, _finite, _row_meta, choose_training_sector

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "tri_branch_tess_net.pt")
DEFAULT_OUT = os.path.join(ROOT, "benchmark_results")


def load_checkpoint_meta(weights_path):
    """Load only checkpoint metadata; fail closed if train/val provenance is absent."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    ckpt = load_checkpoint(weights_path)
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            "Benchmark requires a v9 checkpoint dictionary containing train_tics/validation_tics. "
            "The supplied weights look like a legacy raw state_dict."
        )
    train_tics = {int(x) for x in ckpt.get("train_tics", [])}
    val_tics = {int(x) for x in ckpt.get("validation_tics", [])}
    holdout = {int(x) for x in ckpt.get("excluded_holdout_tics", [])}
    if not train_tics or not val_tics:
        raise RuntimeError(
            "Checkpoint does not contain train_tics and validation_tics. "
            "Cannot guarantee a leakage-safe benchmark. Retrain with the updated train_real.py."
        )
    excluded = train_tics | val_tics | holdout
    return ckpt, excluded


def fetch_catalog_by_tic():
    sql = (
        "select toi,tid,tfopwg_disp,pl_orbper,pl_tranmid,"
        "pl_trandurh,pl_trandep,st_teff,st_logg,st_rad "
        "from toi where tid is not null and tfopwg_disp is not null"
    )
    rows = archive_query(sql)
    by_tic = defaultdict(list)
    for r in rows:
        try:
            tic = int(float(r[1]))
        except Exception:
            continue
        by_tic[tic].append(r)
    return by_tic


def make_candidate_pools(by_tic, excluded):
    """Mirror train_real.py label logic, but leave all excluded TICs untouched."""
    positive = []
    negative = []
    for tic, rows in by_tic.items():
        if tic in excluded:
            continue
        pos = [r for r in rows if str(r[2]).upper().strip() in {"CP", "KP"}]
        neg = [r for r in rows if str(r[2]).upper().strip() in {"FP", "FA"}]
        if pos:
            r = next((x for x in pos if _finite(x[3]) and float(x[3]) > 0 and _finite(x[4])), None)
            if r is not None:
                positive.append(_row_meta(tic, r, 1))
        elif neg:
            r = next((x for x in neg if _finite(x[3]) and float(x[3]) > 0), neg[0])
            negative.append(_row_meta(tic, r, 0))
    return positive, negative


def choose_cases(positive, negative, n_per_class, seed, max_sector_checks):
    """Pick candidates whose selected TESS sector can actually be resolved."""
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)

    chosen = []
    checked = {0: 0, 1: 0}
    for label, pool in ((1, positive), (0, negative)):
        for meta in pool:
            if len([x for x in chosen if x["label"] == label]) >= n_per_class:
                break
            if checked[label] >= max_sector_checks:
                break
            checked[label] += 1
            try:
                sector = choose_training_sector(meta)
            except Exception as exc:
                print(f"   SKIP TIC {meta['tic']} label={label}: sector selection failed: {exc}")
                continue
            item = dict(meta)
            item["sector"] = int(sector)
            chosen.append(item)
            print(f"   SELECT TIC {item['tic']} label={'PLANET' if label else 'FP'} sector={sector}")
    got_p = sum(x["label"] == 1 for x in chosen)
    got_n = sum(x["label"] == 0 for x in chosen)
    if got_p < n_per_class or got_n < n_per_class:
        raise RuntimeError(
            f"Could only select {got_p} planets and {got_n} false positives; "
            f"requested {n_per_class} each. Increase --max-sector-checks or use a different seed."
        )
    chosen.sort(key=lambda x: (x["label"], x["tic"]))
    return chosen


def write_manifest(cases, path, excluded, weights_path):
    payload = {
        "weights": os.path.abspath(weights_path),
        "excluded_tic_count": len(excluded),
        "excluded_tics": sorted(excluded),
        "cases": cases,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def evaluate_cases(cases, weights_path):
    pipeline = Pipeline(weights_path=weights_path)
    rows = []
    total = len(cases)
    for i, case in enumerate(cases, 1):
        tic = int(case["tic"])
        sector = int(case["sector"])
        label = int(case["label"])
        print(f"\n[{i}/{total}] TIC {tic} / sector {sector} / {'PLANET' if label else 'FALSE POSITIVE'}")
        try:
            result = pipeline.run(tic, sector=sector, generate_report=False)
            b = result["bls"]
            v = result["verification"]
            score = float(result["ai"]["planet_score"])
            row = {
                "tic": tic,
                "sector": sector,
                "label": label,
                "known": "PLANET" if label else "FALSE_POSITIVE",
                "ai_score": score,
                "period": float(b.get("period", np.nan)),
                "snr": float(b.get("snr", np.nan)),
                "period_quality": b.get("period_quality"),
                "observed_transits": int(v.get("observed_transits", 0)),
                "expected_transits": int(v.get("expected_transits", 0)),
                "odd_even_mismatch_pct": float(v.get("odd_even_mismatch_pct", np.nan)),
                "eb_flag": bool(v.get("is_eclipsing_binary_flag", False)),
                "period_validation": v.get("period_validation_status"),
                "overall_status": result.get("overall_status"),
                "error": "",
            }
        except Exception as exc:
            row = {
                "tic": tic, "sector": sector, "label": label,
                "known": "PLANET" if label else "FALSE_POSITIVE",
                "ai_score": np.nan, "period": np.nan, "snr": np.nan,
                "period_quality": "ERROR", "observed_transits": 0,
                "expected_transits": 0, "odd_even_mismatch_pct": np.nan,
                "eb_flag": False, "period_validation": "ERROR",
                "overall_status": "ERROR", "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"   ERROR: {row['error']}")
        rows.append(row)
    return rows


def aggregate(rows):
    valid = [r for r in rows if np.isfinite(r["ai_score"])]
    y = np.asarray([r["label"] for r in valid], dtype=int)
    p = np.asarray([r["ai_score"] for r in valid], dtype=float)
    pred = (p >= 0.5).astype(int)
    out = {
        "n_total": len(rows),
        "n_valid": len(valid),
        "n_errors": len(rows) - len(valid),
        "n_planets": int(np.sum(y == 1)),
        "n_false_positives": int(np.sum(y == 0)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "recall": float(recall_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "mean_planet_score": float(p[y == 1].mean()) if np.any(y == 1) else np.nan,
        "mean_fp_score": float(p[y == 0].mean()) if np.any(y == 0) else np.nan,
        "median_planet_score": float(np.median(p[y == 1])) if np.any(y == 1) else np.nan,
        "median_fp_score": float(np.median(p[y == 0])) if np.any(y == 0) else np.nan,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist() if len(y) else [[0, 0], [0, 0]],
    }
    return out


def save_results(rows, metrics, output_dir, manifest):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    json_path = os.path.join(output_dir, "benchmark_metrics.json")
    manifest_path = os.path.join(output_dir, "benchmark_manifest.json")
    if rows:
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return csv_path, json_path, manifest_path


def main():
    ap = argparse.ArgumentParser(description="Leakage-safe 20+20 real-TESS benchmark")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--max-sector-checks", type=int, default=80)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()

    ckpt, excluded = load_checkpoint_meta(args.weights)
    print("=== LEAKAGE-SAFE REAL-TESS BENCHMARK ===")
    print(f"Checkpoint: {args.weights}")
    print(f"Excluded TICs: {len(excluded)} (train + validation + explicit holdouts)")
    print(f"Selecting {args.n_per_class} planets + {args.n_per_class} false positives")

    by_tic = fetch_catalog_by_tic()
    positive, negative = make_candidate_pools(by_tic, excluded)
    print(f"Eligible pools after exclusion: {len(positive)} planets, {len(negative)} false positives")

    cases = choose_cases(positive, negative, args.n_per_class, args.seed, args.max_sector_checks)
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "benchmark_manifest.json")
    manifest = {
        "weights": os.path.abspath(args.weights),
        "checkpoint_best_validation_auc": ckpt.get("best_validation_auc"),
        "checkpoint_best_validation_epoch": ckpt.get("best_validation_epoch"),
        "excluded_tics": sorted(excluded),
        "selection_seed": args.seed,
        "cases": cases,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    if args.select_only:
        return

    rows = evaluate_cases(cases, args.weights)
    metrics = aggregate(rows)
    csv_path, json_path, _ = save_results(rows, metrics, args.output_dir, manifest)

    print("\n" + "=" * 72)
    print("REAL-TESS BENCHMARK RESULTS")
    print("=" * 72)
    for key in ("n_valid", "n_planets", "n_false_positives", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "mean_planet_score", "mean_fp_score", "median_planet_score", "median_fp_score"):
        print(f"{key:>24}: {metrics[key]}")
    print("\nConfusion matrix [rows=true FP/Planet, cols=pred FP/Planet]:")
    print(np.asarray(metrics["confusion_matrix"]))
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("AI score is a model score, not a calibrated probability.")
    print("Do not tune the model on this benchmark after seeing these results.")


if __name__ == "__main__":
    main()
