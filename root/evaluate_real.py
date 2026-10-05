"""
Evaluate the trained AI against a user-supplied real-TESS benchmark.

Example:
  python -m root.evaluate_real \
    --case 172518755:21:1:3.2087 \
    --case 224245334:29:1:3.7853 \
    --case 331484419:55:0:6.2566

The final field is an optional reference period used only to report period
error. Labels are supplied by the user rather than hidden in the model.
"""
import argparse
import math
from root.main_pipeline import Pipeline


def parse_case(value):
    parts = value.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "case must be TIC:SECTOR:LABEL[:REFERENCE_PERIOD]"
        )
    tic = int(parts[0])
    sector = int(parts[1])
    label = int(parts[2])
    if label not in (0, 1):
        raise argparse.ArgumentTypeError("LABEL must be 0 or 1")
    ref = float(parts[3]) if len(parts) == 4 else None
    return tic, sector, label, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", required=True, type=parse_case,
                    help="TIC:SECTOR:LABEL[:REFERENCE_PERIOD], repeatable")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    pipeline = Pipeline(weights_path=args.weights) if args.weights else Pipeline()
    rows = []

    for tic, sector, label, ref_period in args.case:
        print(f"\n=== TIC {tic} / Sector {sector} ===")
        result = pipeline.run(tic, sector=sector, generate_report=True)
        b = result["bls"]
        v = result["verification"]
        score = result["ai"]["planet_score"]

        period_error = None
        if ref_period and ref_period > 0:
            period_error = abs(b["period"] - ref_period) / ref_period * 100.0

        rows.append({
            "tic": tic,
            "sector": sector,
            "known": "PLANET" if label else "FALSE_POSITIVE",
            "score": score * 100.0,
            "period": b["period"],
            "period_error_pct": period_error,
            "period_quality": b.get("period_quality"),
            "events": v.get("observed_transits", 0),
            "physical": v.get("period_validation_status"),
            "eb": v.get("is_eclipsing_binary_flag"),
            "overall": result.get("overall_status"),
        })

    print("\n" + "=" * 110)
    print("REAL-TESS AI BENCHMARK")
    print("=" * 110)
    print(f"{'TIC':>12} {'KNOWN':>15} {'AI%':>8} {'PERIOD':>10} {'ERR%':>8} "
          f"{'EVENTS':>8} {'EB':>8} {'STATUS':>32}")
    print("-" * 110)
    for r in rows:
        err = f"{r['period_error_pct']:.3f}" if r["period_error_pct"] is not None else "-"
        eb = str(r["eb"])
        print(f"{r['tic']:>12} {r['known']:>15} {r['score']:>7.1f} "
              f"{r['period']:>10.4f} {err:>8} {r['events']:>8} {eb:>8} "
              f"{r['overall']:>32}")
    print("=" * 110)
    print("AI% is a model score, not a calibrated probability. The benchmark is "
          "most useful when the TICs were not used during training.")


if __name__ == "__main__":
    main()
