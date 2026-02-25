"""
cln_sat3_analyze.py — Analyze 3-SAT excursion results.
Primary metric: dist_to_target (not heuristic booldist).
"""
import csv
import math
import os
import sys
from collections import defaultdict

SUMMARY_FILE = "sat3_excursion_summary.csv"
TRACE_FILE   = "sat3_excursion_traces.csv"

def load_csv(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for r in csv.DictReader(f):
            for k in r:
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
            rows.append(r)
    return rows

def stats(vals: list) -> dict:
    if not vals:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "med": 0, "max": 0,
                "p25": 0, "p75": 0}
    vals = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {
        "n": n, "mean": mean, "std": math.sqrt(var),
        "min": vals[0], "p25": vals[n // 4], "med": vals[n // 2],
        "p75": vals[3 * n // 4], "max": vals[-1],
    }

def fmt(x, w=10, d=4):
    return f"{x:>{w}.{d}f}"

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx < 1e-15 or sy < 1e-15:
        return 0.0
    return cov / (sx * sy)

def section(title):
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")

def analyze_summary(rows: list):
    by_ratio = defaultdict(list)
    for r in rows:
        by_ratio[r["clause_ratio"]].append(r)

    print(f"\nTotal instances: {len(rows)}")

    # ── 1. Main excursion table (combinatorial nodes only) ──
    section("COMBINATORIAL EXCURSION BY CLAUSE RATIO")
    print(f"{'ratio':>7} {'trials':>6} {'unsat%':>6} "
          f"{'pk_dist':>9} {'pk_l1':>9} {'pk_gr':>6} "
          f"{'exc_dl':>9} {'exc_dlg':>10} "
          f"{'frac_sat':>9}")
    print("-" * 85)

    for ratio in sorted(by_ratio.keys()):
        rs = by_ratio[ratio]
        n = len(rs)
        n_unsat = sum(1 for r in rs if r["is_unsat"] > 0.5)
        print(f"{ratio:7.3f} {n:6d} {100*n_unsat/n:5.0f}% "
              f"{sum(r['comb_peak_dist'] for r in rs)/n:9.4f} "
              f"{sum(r['comb_peak_l1'] for r in rs)/n:9.4f} "
              f"{sum(r['comb_peak_grade'] for r in rs)/n:6.1f} "
              f"{sum(r['comb_exc_dl'] for r in rs)/n:9.4f} "
              f"{sum(r['comb_exc_dlg'] for r in rs)/n:10.4f} "
              f"{sum(r['frac_sat'] for r in rs)/n:9.3f}")

    # ── 2. dist_to_target distribution per ratio ──
    section("DIST-TO-TARGET DISTRIBUTION (combinatorial peak)")
    print(f"{'ratio':>7} {'min':>9} {'p25':>9} {'med':>9} {'p75':>9} {'max':>9} {'std':>9}")
    print("-" * 65)
    for ratio in sorted(by_ratio.keys()):
        s = stats([r["comb_peak_dist"] for r in by_ratio[ratio]])
        print(f"{ratio:7.3f} {s['min']:9.4f} {s['p25']:9.4f} {s['med']:9.4f} "
              f"{s['p75']:9.4f} {s['max']:9.4f} {s['std']:9.4f}")

    # ── 3. SAT vs UNSAT ──
    sat_rows = [r for r in rows if r["is_unsat"] < 0.5]
    unsat_rows = [r for r in rows if r["is_unsat"] > 0.5]

    section("SAT vs UNSAT (combinatorial metrics)")
    for label, subset in [("SAT", sat_rows), ("UNSAT", unsat_rows)]:
        if not subset:
            print(f"\n  {label}: no instances")
            continue
        n = len(subset)
        print(f"\n  {label} ({n} instances):")
        for metric in ["comb_peak_dist", "comb_peak_l1", "comb_peak_grade",
                        "comb_exc_dl", "comb_exc_dlg",
                        "final_dist", "target_l1"]:
            s = stats([r[metric] for r in subset])
            print(f"    {metric:25s}  mean={s['mean']:8.4f}  std={s['std']:8.4f}  "
                  f"med={s['med']:8.4f}  max={s['max']:8.4f}")

    # ── 4. Excursion vs frac_sat ──
    section("EXCURSION vs FRACTION SATISFIABLE")
    buckets = defaultdict(list)
    for r in rows:
        bucket = round(r["frac_sat"] * 20) / 20  # 0.05 resolution
        buckets[bucket].append(r)

    print(f"{'frac_sat':>9} {'count':>6} {'pk_dist':>9} {'pk_l1':>9} "
          f"{'exc_dl':>9} {'exc_dlg':>10} {'target_l1':>10}")
    print("-" * 75)
    for bucket in sorted(buckets.keys()):
        rs = buckets[bucket]
        n = len(rs)
        if n < 2:
            continue
        print(f"{bucket:9.3f} {n:6d} "
              f"{sum(r['comb_peak_dist'] for r in rs)/n:9.4f} "
              f"{sum(r['comb_peak_l1'] for r in rs)/n:9.4f} "
              f"{sum(r['comb_exc_dl'] for r in rs)/n:9.4f} "
              f"{sum(r['comb_exc_dlg'] for r in rs)/n:10.4f} "
              f"{sum(r['target_l1'] for r in rs)/n:10.4f}")

    # ── 5. Where peaks occur ──
    section("PEAK POSITIONS (fraction through trace)")
    print(f"{'ratio':>7} {'l1_pos':>8} {'dist_pos':>9} {'grade_pos':>10}")
    print("-" * 40)
    for ratio in sorted(by_ratio.keys()):
        rs = by_ratio[ratio]
        n = len(rs)
        print(f"{ratio:7.3f} "
              f"{sum(r['comb_peak_l1_pos'] for r in rs)/n:8.3f} "
              f"{sum(r['comb_peak_dist_pos'] for r in rs)/n:9.3f} "
              f"{sum(r['peak_grade_pos'] for r in rs)/n:10.3f}")

    # ── 6. Correlations ──
    section("CORRELATIONS")
    pairs = [
        ("comb_peak_dist", "comb_peak_l1"),
        ("comb_peak_dist", "comb_peak_grade"),
        ("comb_peak_l1", "comb_peak_grade"),
        ("comb_exc_dl", "frac_sat"),
        ("comb_exc_dlg", "frac_sat"),
        ("comb_peak_dist", "frac_sat"),
        ("comb_peak_l1", "frac_sat"),
        ("n_clauses", "comb_exc_dl"),
        ("n_clauses", "comb_peak_dist"),
        ("n_clauses", "comb_peak_l1"),
        ("target_l1", "comb_peak_dist"),
        ("target_l1", "comb_exc_dl"),
        ("target_l1", "frac_sat"),
    ]
    for a, b in pairs:
        r = pearson([row[a] for row in rows], [row[b] for row in rows])
        print(f"  corr({a}, {b}) = {r:+.4f}")

    # ── 7. Top excursion instances ──
    section("TOP 15 HIGHEST comb_exc_dl")
    print(f"{'ratio':>7} {'trial':>5} {'exc_dl':>9} {'pk_dist':>9} "
          f"{'pk_l1':>9} {'pk_gr':>6} {'frac_sat':>9} {'unsat':>5} {'tgt_l1':>8}")
    print("-" * 80)
    top = sorted(rows, key=lambda r: r["comb_exc_dl"], reverse=True)[:15]
    for r in top:
        print(f"{r['clause_ratio']:7.3f} {int(r['trial']):5d} "
              f"{r['comb_exc_dl']:9.4f} {r['comb_peak_dist']:9.4f} "
              f"{r['comb_peak_l1']:9.4f} {int(r['comb_peak_grade']):6d} "
              f"{r['frac_sat']:9.3f} {'Y' if r['is_unsat']>0.5 else 'N':>5s} "
              f"{r['target_l1']:8.4f}")

    # ── 8. Final dist check ──
    section("FINAL DISTANCE (should be ~0 for correct translation)")
    for ratio in sorted(by_ratio.keys()):
        rs = by_ratio[ratio]
        fds = [r["final_dist"] for r in rs]
        print(f"  ratio={ratio:.3f}  mean_final_dist={sum(fds)/len(fds):.8f}  "
              f"max={max(fds):.8f}")

def analyze_traces(rows: list):
    section("TRACE ANALYSIS: COMBINATORIAL NODES ONLY")

    by_instance = defaultdict(list)
    for r in rows:
        key = (r["clause_ratio"], int(r["trial"]))
        by_instance[key].append(r)

    by_ratio = defaultdict(list)
    for (ratio, trial), all_nodes in by_instance.items():
        comb = [nd for nd in all_nodes if nd.get("kind") == "combinatorial"]
        if len(comb) < 3:
            continue

        comb_sorted = sorted(comb, key=lambda r: r["node_id"])
        dists = [nd["dist_to_target"] for nd in comb_sorted]
        l1s = [nd["l1"] for nd in comb_sorted]
        nn = len(comb_sorted)

        peak_dist_idx = max(range(nn), key=lambda i: dists[i])
        peak_frac = peak_dist_idx / nn

        # monotonicity after peak: does dist decrease steadily?
        if peak_dist_idx < nn - 1:
            post = dists[peak_dist_idx:]
            decreases = sum(1 for i in range(1, len(post)) if post[i] < post[i-1])
            mono_score = decreases / (len(post) - 1) if len(post) > 1 else 0
        else:
            mono_score = 0

        # correlation between dist and l1 within this trace
        r_dl = pearson(dists, l1s)

        by_ratio[ratio].append({
            "peak_frac": peak_frac,
            "mono_score": mono_score,
            "corr_dist_l1": r_dl,
            "n_comb": nn,
        })

    print(f"{'ratio':>7} {'count':>6} {'pk_pos':>8} {'mono':>8} {'corr_d_l1':>10} {'n_comb':>7}")
    print("-" * 55)
    for ratio in sorted(by_ratio.keys()):
        rs = by_ratio[ratio]
        n = len(rs)
        print(f"{ratio:7.3f} {n:6d} "
              f"{sum(r['peak_frac'] for r in rs)/n:8.3f} "
              f"{sum(r['mono_score'] for r in rs)/n:8.3f} "
              f"{sum(r['corr_dist_l1'] for r in rs)/n:10.4f} "
              f"{sum(r['n_comb'] for r in rs)/n:7.1f}")

if __name__ == "__main__":
    summary_path = sys.argv[1] if len(sys.argv) > 1 else SUMMARY_FILE
    trace_path = sys.argv[2] if len(sys.argv) > 2 else TRACE_FILE

    if not os.path.exists(summary_path):
        print(f"File not found: {summary_path}")
        sys.exit(1)

    rows = load_csv(summary_path)
    analyze_summary(rows)

    if os.path.exists(trace_path):
        trace_rows = load_csv(trace_path)
        analyze_traces(trace_rows)
    else:
        print(f"\nNo trace file found at {trace_path}; skipping trace analysis.")