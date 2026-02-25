"""
cln_analyze.py — Analyze cln_sweep_results.csv
Run after cln_sweep.py finishes (or while it's still running).
"""
import csv
import sys
import os
from collections import defaultdict

INFILE = "cln_sweep_results.csv"

def load(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in r:
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
            rows.append(r)
    return rows

def analyze(rows: list) -> None:
    by_depth = defaultdict(list)
    for r in rows:
        by_depth[int(r["and_depth"])].append(r)

    print(f"Total rows: {len(rows)}")
    print()

    # ── Summary table ──
    header = (f"{'depth':>5} {'trials':>6} {'hits':>5} {'hit%':>6} "
              f"{'mean_gap':>10} {'max_gap':>10} {'mean_ratio':>10} {'max_ratio':>10} "
              f"{'mean_peak_M':>11} {'mean_peak_C':>11} "
              f"{'mean_fin_M':>11} {'mean_fin_C':>11} "
              f"{'neg_gaps':>8}")
    print(header)
    print("-" * len(header))

    for depth in sorted(by_depth.keys()):
        rs = by_depth[depth]
        n = len(rs)
        hits = [r for r in rs if r["has_gap"] > 0.5]
        nh = len(hits)

        gaps = [r["peak_gap"] for r in rs]
        ratios = [r["peak_ratio"] for r in rs]
        peak_m = [r["peak_mul"] for r in rs]
        peak_c = [r["peak_conv"] for r in rs]
        fin_m = [r["final_mul"] for r in rs]
        fin_c = [r["final_conv"] for r in rs]

        neg_gaps = sum(1 for g in gaps if g < -1e-10)

        print(f"{depth:5d} {n:6d} {nh:5d} {100*nh/n:5.1f}% "
              f"{sum(gaps)/n:10.4f} {max(gaps):10.4f} "
              f"{sum(ratios)/n:10.4f} {max(ratios):10.4f} "
              f"{sum(peak_m)/n:11.4f} {sum(peak_c)/n:11.4f} "
              f"{sum(fin_m)/n:11.4f} {sum(fin_c)/n:11.4f} "
              f"{neg_gaps:8d}")

    print()

    # ── Gap distribution per depth ──
    print("Gap distribution (peak_gap) per depth:")
    print(f"{'depth':>5} {'min':>10} {'p25':>10} {'median':>10} {'p75':>10} {'max':>10} {'std':>10}")
    print("-" * 75)
    for depth in sorted(by_depth.keys()):
        gaps = sorted(r["peak_gap"] for r in by_depth[depth])
        n = len(gaps)
        if n == 0:
            continue
        import math
        mean = sum(gaps) / n
        var = sum((g - mean) ** 2 for g in gaps) / n
        std = math.sqrt(var)
        p25 = gaps[n // 4]
        med = gaps[n // 2]
        p75 = gaps[3 * n // 4]
        print(f"{depth:5d} {gaps[0]:10.4f} {p25:10.4f} {med:10.4f} {p75:10.4f} {gaps[-1]:10.4f} {std:10.4f}")

    print()

    # ── Ratio scaling: does peak_ratio grow exponentially? ──
    print("Peak ratio scaling (is it exponential in depth?):")
    print(f"{'depth':>5} {'mean_ratio':>10} {'log2(mean)':>10}")
    print("-" * 30)
    for depth in sorted(by_depth.keys()):
        ratios = [r["peak_ratio"] for r in by_depth[depth]]
        mr = sum(ratios) / len(ratios)
        lr = math.log2(mr) if mr > 0 else 0
        print(f"{depth:5d} {mr:10.4f} {lr:10.4f}")

    print()

    # ── Final gap: does Mul also distort the final value? ──
    print("Final value comparison:")
    print(f"{'depth':>5} {'mean_final_gap':>14} {'max_final_gap':>14} {'finals_equal%':>13}")
    print("-" * 50)
    for depth in sorted(by_depth.keys()):
        rs = by_depth[depth]
        fgaps = [r["final_gap"] for r in rs]
        eq = sum(1 for fg in fgaps if abs(fg) < 1e-10)
        print(f"{depth:5d} {sum(fgaps)/len(fgaps):14.6f} {max(fgaps):14.6f} {100*eq/len(rs):12.1f}%")

    print()

    # ── Top 10 biggest gaps overall ──
    print("Top 10 biggest peak gaps:")
    print(f"{'depth':>5} {'trial':>5} {'peak_mul':>10} {'peak_conv':>10} {'gap':>10} {'ratio':>8} {'ands':>5} {'ors':>5}")
    print("-" * 65)
    top = sorted(rows, key=lambda r: r["peak_gap"], reverse=True)[:10]
    for r in top:
        print(f"{int(r['and_depth']):5d} {int(r['trial']):5d} "
              f"{r['peak_mul']:10.4f} {r['peak_conv']:10.4f} "
              f"{r['peak_gap']:10.4f} {r['peak_ratio']:8.3f} "
              f"{int(r['n_ands']):5d} {int(r['n_ors']):5d}")

    # ── Any negative gaps? ──
    negs = [r for r in rows if r["peak_gap"] < -1e-10]
    if negs:
        print(f"\n*** {len(negs)} NEGATIVE GAPS FOUND (Mul beats Conv) ***")
        for r in sorted(negs, key=lambda r: r["peak_gap"])[:5]:
            print(f"  depth={int(r['and_depth'])} trial={int(r['trial'])} gap={r['peak_gap']:.6f}")
    else:
        print("\nNo negative gaps found. Mul NEVER beats Conv on peak excursion.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        INFILE = sys.argv[1]
    if not os.path.exists(INFILE):
        print(f"File not found: {INFILE}")
        sys.exit(1)
    rows = load(INFILE)
    analyze(rows)