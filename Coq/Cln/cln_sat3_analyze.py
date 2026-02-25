"""
cln_sat3_analyze.py — UPDATED for dist_rel + overlap_norm keys
(FIX: is_finite now accepts int as well, so n_comb no longer prints nan)

Reads:
  sat3_excursion_summary.csv
  sat3_excursion_traces.csv (optional)

Handles NaNs (from dist_rel when target_l1==0) by filtering them out in stats/corr.
"""

import csv
import math
import os
import sys
from collections import defaultdict

SUMMARY_FILE = "sat3_excursion_summary.csv"
TRACE_FILE = "sat3_excursion_traces.csv"


def is_finite(x) -> bool:
    """True for finite ints/floats; False for NaN/inf and non-numbers."""
    if isinstance(x, bool):  # avoid treating booleans as numbers here
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return (x == x) and (abs(x) != float("inf"))
    return False


def load_csv(path: str) -> list:
    rows = []
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            for k in list(r.keys()):
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def stats(vals: list) -> dict:
    vals = [v for v in vals if is_finite(v)]
    if not vals:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "p25": float("nan"), "med": float("nan"),
                "p75": float("nan"), "max": float("nan")}
    vals = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    if n >= 2:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    else:
        var = 0.0
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": vals[0],
        "p25": vals[n // 4],
        "med": vals[n // 2],
        "p75": vals[(3 * n) // 4],
        "max": vals[-1],
    }


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if is_finite(x) and is_finite(y)]
    n = len(pairs)
    if n < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sxx = sum(d * d for d in dx)
    syy = sum(d * d for d in dy)
    if sxx <= 1e-18 or syy <= 1e-18:
        return float("nan")
    sxy = sum(a * b for a, b in zip(dx, dy))
    return sxy / math.sqrt(sxx * syy)


def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def analyze_summary(rows: list):
    by_ratio = defaultdict(list)
    for r in rows:
        by_ratio[r["clause_ratio"]].append(r)

    print(f"\nTotal instances: {len(rows)}")

    section("COMBINATORIAL SUMMARY BY CLAUSE RATIO")
    print(
        f"{'ratio':>7} {'trials':>6} {'unsat%':>6} "
        f"{'pk_dist':>9} {'pk_rel':>9} {'pk_ov':>7} {'pk_l1':>9} {'pk_gr':>6} "
        f"{'exc_dl':>9} {'exc_dlr':>9} {'exc_dlg':>10} "
        f"{'frac_sat':>9} {'tgt_l1':>9}"
    )
    print("-" * 120)

    for ratio in sorted(by_ratio.keys()):
        rs = by_ratio[ratio]
        n = len(rs)
        n_unsat = sum(1 for r in rs if r["is_unsat"] > 0.5)

        def mean(key):
            vals = [r[key] for r in rs if is_finite(r.get(key, float("nan")))]
            return (sum(vals) / len(vals)) if vals else float("nan")

        print(
            f"{ratio:7.3f} {n:6d} {100*n_unsat/n:5.0f}% "
            f"{mean('comb_peak_dist'):9.4f} {mean('comb_peak_dist_rel'):9.4f} {mean('comb_peak_overlap'):7.4f} "
            f"{mean('comb_peak_l1'):9.4f} {mean('comb_peak_grade'):6.1f} "
            f"{mean('comb_exc_dl'):9.4f} {mean('comb_exc_dlr'):9.4f} {mean('comb_exc_dlg'):10.4f} "
            f"{mean('frac_sat'):9.3f} {mean('target_l1'):9.3f}"
        )

    section("DISTRIBUTION: comb_peak_dist")
    print(f"{'ratio':>7} {'min':>9} {'p25':>9} {'med':>9} {'p75':>9} {'max':>9} {'std':>9}")
    print("-" * 65)
    for ratio in sorted(by_ratio.keys()):
        s = stats([r["comb_peak_dist"] for r in by_ratio[ratio]])
        print(f"{ratio:7.3f} {s['min']:9.4f} {s['p25']:9.4f} {s['med']:9.4f} {s['p75']:9.4f} {s['max']:9.4f} {s['std']:9.4f}")

    section("DISTRIBUTION: comb_peak_dist_rel (finite only)")
    print(f"{'ratio':>7} {'min':>9} {'p25':>9} {'med':>9} {'p75':>9} {'max':>9} {'std':>9} {'n':>5}")
    print("-" * 72)
    for ratio in sorted(by_ratio.keys()):
        s = stats([r["comb_peak_dist_rel"] for r in by_ratio[ratio]])
        print(f"{ratio:7.3f} {s['min']:9.4f} {s['p25']:9.4f} {s['med']:9.4f} {s['p75']:9.4f} {s['max']:9.4f} {s['std']:9.4f} {s['n']:5d}")

    section("DISTRIBUTION: comb_peak_overlap (0..1 similarity-ish)")
    print(f"{'ratio':>7} {'min':>9} {'p25':>9} {'med':>9} {'p75':>9} {'max':>9} {'std':>9}")
    print("-" * 65)
    for ratio in sorted(by_ratio.keys()):
        s = stats([r["comb_peak_overlap"] for r in by_ratio[ratio]])
        print(f"{ratio:7.3f} {s['min']:9.4f} {s['p25']:9.4f} {s['med']:9.4f} {s['p75']:9.4f} {s['max']:9.4f} {s['std']:9.4f}")

    sat_rows = [r for r in rows if r["is_unsat"] < 0.5]
    unsat_rows = [r for r in rows if r["is_unsat"] > 0.5]

    section("SAT vs UNSAT (key metrics)")
    for label, subset in [("SAT", sat_rows), ("UNSAT", unsat_rows)]:
        if not subset:
            print(f"\n  {label}: no instances")
            continue
        print(f"\n  {label} ({len(subset)} instances):")
        for metric in [
            "comb_peak_dist", "comb_peak_dist_rel", "comb_peak_overlap",
            "comb_peak_l1", "comb_peak_grade",
            "comb_exc_dl", "comb_exc_dlr", "comb_exc_dlg",
            "final_dist", "final_dist_rel", "final_overlap",
            "target_l1",
        ]:
            s = stats([r.get(metric, float("nan")) for r in subset])
            print(f"    {metric:20s}  mean={s['mean']:10.4f}  std={s['std']:10.4f}  med={s['med']:10.4f}  max={s['max']:10.4f}  n={s['n']:4d}")

    section("CORRELATIONS (sample Pearson, finite pairs only)")
    pairs = [
        ("comb_peak_dist", "comb_peak_l1"),
        ("comb_peak_dist_rel", "comb_peak_l1"),
        ("comb_peak_overlap", "frac_sat"),
        ("comb_peak_dist", "frac_sat"),
        ("comb_peak_dist_rel", "frac_sat"),
        ("target_l1", "frac_sat"),
        ("target_l1", "comb_peak_dist"),
        ("target_l1", "comb_peak_dist_rel"),
        ("n_clauses", "comb_peak_dist"),
        ("n_clauses", "comb_peak_dist_rel"),
    ]
    for a, b in pairs:
        r = pearson([row.get(a, float("nan")) for row in rows], [row.get(b, float("nan")) for row in rows])
        print(f"  corr({a}, {b}) = {r:+.4f}")

    section("FINAL DISTANCE (translation sanity signal)")
    for ratio in sorted(by_ratio.keys()):
        rs = by_ratio[ratio]
        fds = [r["final_dist"] for r in rs]
        frs = [r["final_dist_rel"] for r in rs]
        print(
            f"  ratio={ratio:.3f}  "
            f"mean_final_dist={stats(fds)['mean']:.8f}  "
            f"mean_final_dist_rel={stats(frs)['mean']:.8f}"
        )

    section("TOP 15 HIGHEST comb_exc_dl")
    top = sorted(
        rows,
        key=lambda r: (r.get("comb_exc_dl", float("nan")) if is_finite(r.get("comb_exc_dl", float("nan"))) else -1e300),
        reverse=True
    )[:15]
    print(f"{'ratio':>7} {'trial':>6} {'exc_dl':>10} {'pk_dist':>9} {'pk_rel':>9} {'pk_ov':>7} {'pk_l1':>9} {'pk_gr':>6} {'frac_sat':>9} {'unsat':>6} {'tgt_l1':>9}")
    print("-" * 105)
    for r in top:
        print(
            f"{r['clause_ratio']:7.3f} {int(r['trial']):6d} {r['comb_exc_dl']:10.4f} "
            f"{r['comb_peak_dist']:9.4f} {r['comb_peak_dist_rel']:9.4f} {r['comb_peak_overlap']:7.4f} "
            f"{r['comb_peak_l1']:9.4f} {int(r['comb_peak_grade']):6d} {r['frac_sat']:9.3f} "
            f"{'Y' if r['is_unsat'] > 0.5 else 'N':>6} {r['target_l1']:9.4f}"
        )


def analyze_traces(trace_rows: list):
    section("TRACE ANALYSIS: COMBINATORIAL NODES ONLY")
    by_instance = defaultdict(list)
    for r in trace_rows:
        key = (r["clause_ratio"], int(r["trial"]))
        by_instance[key].append(r)

    by_ratio = defaultdict(list)

    for (ratio, trial), nodes in by_instance.items():
        comb = [nd for nd in nodes if nd.get("kind") == "combinatorial"]
        if len(comb) < 3:
            continue

        comb_sorted = sorted(comb, key=lambda r: r["node_id"])
        d = [nd["dist"] for nd in comb_sorted]
        dr = [nd["dist_rel"] for nd in comb_sorted]
        ov = [nd["overlap_norm"] for nd in comb_sorted]
        l1 = [nd["l1"] for nd in comb_sorted]
        nn = len(comb_sorted)

        peak_i = max(range(nn), key=lambda i: d[i])
        peak_frac = 0.0 if nn <= 1 else peak_i / (nn - 1)

        post = d[peak_i:]
        if len(post) >= 2:
            dec = sum(1 for i in range(1, len(post)) if post[i] < post[i - 1])
            mono = dec / (len(post) - 1)
        else:
            mono = float("nan")

        by_ratio[ratio].append({
            "pk_pos": peak_frac,
            "mono": mono,
            "corr_d_l1": pearson(d, l1),
            "corr_dr_l1": pearson(dr, l1),
            "corr_ov_l1": pearson(ov, l1),
            "n_comb": nn,  # int; now is_finite accepts ints
        })

    print(f"{'ratio':>7} {'count':>6} {'pk_pos':>8} {'mono':>8} {'corr_d_l1':>10} {'corr_dr_l1':>11} {'corr_ov_l1':>11} {'n_comb':>7}")
    print("-" * 82)
    for ratio in sorted(by_ratio.keys()):
        rs = by_ratio[ratio]
        n = len(rs)

        def m(key):
            vals = [r[key] for r in rs if is_finite(r.get(key, float("nan")))]
            return sum(vals) / len(vals) if vals else float("nan")

        print(
            f"{ratio:7.3f} {n:6d} {m('pk_pos'):8.3f} {m('mono'):8.3f} "
            f"{m('corr_d_l1'):10.4f} {m('corr_dr_l1'):11.4f} {m('corr_ov_l1'):11.4f} "
            f"{m('n_comb'):7.1f}"
        )


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