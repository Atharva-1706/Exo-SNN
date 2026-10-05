"""Leakage-safe evaluation on a larger unseen real-TESS sample.

This is intentionally separate from benchmark_real.py so the frozen 40-object
benchmark is never overwritten. It excludes:
  1. the v9 checkpoint train/validation TICs,
  2. checkpoint explicit holdouts,
  3. every TIC in benchmark_results/benchmark_manifest.json.

It then selects a deterministic 100 planets + 100 false positives/alarms from
the remaining NASA Exoplanet Archive TOI pool, resolves a TESS sector using
the same timing-aware logic as training, freezes the manifest, and evaluates
with the existing v9 checkpoint.

IMPORTANT: Do not retrain or tune the v9 model after this evaluation if you
intend to report these results as a blind/generalization test.

Commands:
  python -m root.evaluate_unseen --n-per-class 100 --select-only
  python -m root.evaluate_unseen --n-per-class 100
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from root.checkpoint_io import load_checkpoint
from root.main_pipeline import Pipeline
from root.train_real import archive_query, _finite, _row_meta, choose_training_sector

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "tri_branch_tess_net.pt")
DEFAULT_OUT = os.path.join(ROOT, "unseen_evaluation_200")
FROZEN_BENCHMARK = os.path.join(ROOT, "benchmark_results", "benchmark_manifest.json")


def load_excluded(weights_path):
    ckpt = load_checkpoint(weights_path)
    if not isinstance(ckpt, dict):
        raise RuntimeError("Expected a checkpoint dictionary with provenance metadata.")
    train = {int(x) for x in ckpt.get("train_tics", [])}
    val = {int(x) for x in ckpt.get("validation_tics", [])}
    holdout = {int(x) for x in ckpt.get("excluded_holdout_tics", [])}
    if not train or not val:
        raise RuntimeError("Checkpoint lacks train_tics/validation_tics; refusing blind evaluation.")
    excluded = train | val | holdout
    if not os.path.exists(FROZEN_BENCHMARK):
        raise FileNotFoundError(f"Frozen benchmark manifest not found: {FROZEN_BENCHMARK}")
    with open(FROZEN_BENCHMARK, "r", encoding="utf-8") as f:
        bench = json.load(f)
    bench_cases = bench.get("cases", [])
    frozen = {int(x["tic"]) for x in bench_cases if "tic" in x}
    excluded |= frozen
    return ckpt, excluded, frozen


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


def make_pools(by_tic, excluded):
    positive, negative = [], []
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
            r = next((x for x in neg if _finite(x[3]) and float(x[3]) > 0 and _finite(x[4])), None)
            if r is not None:
                negative.append(_row_meta(tic, r, 0))
    return positive, negative


def select_cases(positive, negative, n_per_class, seed, max_sector_checks):
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    selected = []
    for label, pool in ((1, positive), (0, negative)):
        got = 0
        checked = 0
        for meta in pool:
            if got >= n_per_class or checked >= max_sector_checks:
                break
            checked += 1
            try:
                sector = int(choose_training_sector(meta))
            except Exception as exc:
                print(f"SKIP TIC {meta['tic']} label={label}: {exc}")
                continue
            item = dict(meta)
            item["sector"] = sector
            selected.append(item)
            got += 1
        if got < n_per_class:
            raise RuntimeError(
                f"Only selected {got}/{n_per_class} for label {label}. "
                f"Increase --max-sector-checks or use another seed."
            )
    selected.sort(key=lambda x: (x["label"], x["tic"]))
    return selected


def evaluate(cases, weights):
    pipeline = Pipeline(weights_path=weights)
    rows = []
    for i, c in enumerate(cases, 1):
        tic, sector, label = int(c["tic"]), int(c["sector"]), int(c["label"])
        print(f"[{i}/{len(cases)}] TIC {tic} / S{sector} / {'PLANET' if label else 'FP'}")
        try:
            result = pipeline.run(tic, sector=sector, generate_report=False)
            b, v = result["bls"], result["verification"]
            rows.append({
                "tic": tic, "sector": sector, "label": label,
                "known": "PLANET" if label else "FALSE_POSITIVE",
                "ai_score": float(result["ai"]["planet_score"]),
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
            })
        except Exception as exc:
            rows.append({
                "tic": tic, "sector": sector, "label": label,
                "known": "PLANET" if label else "FALSE_POSITIVE",
                "ai_score": np.nan, "period": np.nan, "snr": np.nan,
                "period_quality": "ERROR", "observed_transits": 0,
                "expected_transits": 0, "odd_even_mismatch_pct": np.nan,
                "eb_flag": False, "period_validation": "ERROR",
                "overall_status": "ERROR", "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"    ERROR: {rows[-1]['error']}")
    return rows


def metrics(rows):
    valid = [r for r in rows if np.isfinite(r["ai_score"])]
    y = np.asarray([r["label"] for r in valid], dtype=int)
    p = np.asarray([r["ai_score"] for r in valid], dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "n_total": len(rows), "n_valid": len(valid), "n_errors": len(rows)-len(valid),
        "n_planets": int(np.sum(y == 1)), "n_false_positives": int(np.sum(y == 0)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "mean_planet_score": float(p[y == 1].mean()),
        "mean_fp_score": float(p[y == 0].mean()),
        "median_planet_score": float(np.median(p[y == 1])),
        "median_fp_score": float(np.median(p[y == 0])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def save(output_dir, manifest, rows, m):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "unseen_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(output_dir, "unseen_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(output_dir, "unseen_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, allow_nan=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--max-sector-checks", type=int, default=500)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()

    ckpt, excluded, frozen = load_excluded(args.weights)
    print("=== UNSEEN REAL-TESS EVALUATION ===")
    print(f"Checkpoint: {args.weights}")
    print(f"Excluded train/val/holdout + frozen benchmark TICs: {len(excluded)}")
    print(f"Frozen benchmark TICs excluded: {len(frozen)}")

    by_tic = fetch_catalog_by_tic()
    positive, negative = make_pools(by_tic, excluded)
    print(f"Eligible pool: {len(positive)} planets, {len(negative)} false positives")
    cases = select_cases(positive, negative, args.n_per_class, args.seed, args.max_sector_checks)

    manifest = {
        "evaluation": "unseen_real_tess_200",
        "weights": os.path.abspath(args.weights),
        "checkpoint_best_validation_auc": ckpt.get("best_validation_auc"),
        "checkpoint_best_validation_epoch": ckpt.get("best_validation_epoch"),
        "selection_seed": args.seed,
        "n_per_class": args.n_per_class,
        "excluded_tic_count": len(excluded),
        "frozen_benchmark_excluded_count": len(frozen),
        "excluded_tics": sorted(excluded),
        "cases": cases,
    }
    save(args.output_dir, manifest, [], {}) if False else os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "unseen_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {os.path.join(args.output_dir, 'unseen_manifest.json')}")
    if args.select_only:
        return

    rows = evaluate(cases, args.weights)
    m = metrics(rows)
    save(args.output_dir, manifest, rows, m)
    print("\n=== UNSEEN RESULTS ===")
    for k in ("n_total","n_valid","n_errors","accuracy","precision","recall","f1","roc_auc","average_precision","mean_planet_score","mean_fp_score"):
        print(f"{k:>24}: {m[k]}")
    print("Confusion matrix [rows=true FP/Planet, cols=pred FP/Planet]:")
    print(np.asarray(m["confusion_matrix"]))
    print(f"Results: {args.output_dir}")
    print("AI score is a model score, not a calibrated probability.")

if __name__ == "__main__":
    main()
