"""
cln_sat3.py — 3-SAT excursion profiler (UPDATED: dist_rel + overlap_norm, 0/1 target)

Changes (as requested):
  - Revert to 0/1 embedding target (False->0, True->1), consistent with:
      x_i  -> 0.5*(1 + e_i)
      NOT  -> 1 - p
      TRUE -> 1
      FALSE-> 0
      AND  -> Conv
      OR   -> p + q - (p AND q)
  - Remove the problematic symmetric normalization (it saturates ~1).
  - Add:
      dist_rel      = dist / target_l1   (NaN if target_l1 == 0)
      overlap_norm  = overlap / min(l1_v, target_l1)  (0..1-ish), where
         overlap = 0.5*(||F||_1 + ||T||_1 - ||F-T||_1)

Outputs updated keys in both summary + trace CSVs.
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================
# Config
# ============================================================
N_VARS = 10
CLAUSE_RATIOS = [2.0, 3.0, 3.5, 4.0, 4.267, 4.5, 5.0, 6.0, 8.0]
TRIALS_PER_RATIO = 50
SEED = 42

OUTFILE_SUMMARY = "sat3_excursion_summary.csv"
OUTFILE_TRACES = "sat3_excursion_traces.csv"
WRITE_TRACES = True

EPS = 1e-15
SANITY_CHECKS = True

# ============================================================
# MV = Dict[int, float]
# ============================================================
MV = Dict[int, float]


def mv_add(u: MV, v: MV) -> MV:
    out = dict(u)
    for k, c in v.items():
        out[k] = out.get(k, 0.0) + c
    return {k: c for k, c in out.items() if abs(c) > EPS}


def mv_scale(a: float, u: MV) -> MV:
    if abs(a) <= EPS:
        return {}
    return {k: a * c for k, c in u.items() if abs(a * c) > EPS}


def mv_sub(u: MV, v: MV) -> MV:
    return mv_add(u, mv_scale(-1.0, v))


def mv_l1(u: MV) -> float:
    return sum(abs(c) for c in u.values())


def mv_support(u: MV) -> int:
    return len(u)


def popcount(x: int) -> int:
    return x.bit_count()


def mv_max_grade(u: MV) -> int:
    if not u:
        return 0
    return max(popcount(k) for k in u.keys())


def mv_grade_profile(u: MV, n: int) -> List[float]:
    profile = [0.0] * (n + 1)
    for k, c in u.items():
        profile[popcount(k)] += abs(c)
    return profile


# ============================================================
# FWHT
# ============================================================
def fwht(a: List[float]) -> None:
    h = 1
    n = len(a)
    while h < n:
        step = 2 * h
        for i in range(0, n, step):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h = step


def mv_conv_fwht(F: MV, G: MV, n: int) -> MV:
    N = 1 << n
    A = [0.0] * N
    B = [0.0] * N
    for k, v in F.items():
        A[k] = v
    for k, v in G.items():
        B[k] = v
    fwht(A)
    fwht(B)
    C = [A[i] * B[i] for i in range(N)]
    fwht(C)
    invN = 1.0 / N
    return {i: c * invN for i, c in enumerate(C) if abs(c * invN) > EPS}


# ============================================================
# Clifford geometric product
# ============================================================
def clifford_swap_sign(a: int, b: int, n: int) -> int:
    s = 0
    for i in range(n):
        if (a >> i) & 1:
            s += (b & ((1 << i) - 1)).bit_count()
    return -1 if (s & 1) else 1


def mv_mul(u: MV, v: MV, n: int) -> MV:
    out: MV = {}
    for a, ca in u.items():
        for b, cb in v.items():
            res = a ^ b
            sign = clifford_swap_sign(a, b, n)
            val = ca * cb * sign
            out[res] = out.get(res, 0.0) + val
    return {k: c for k, c in out.items() if abs(c) > EPS}


# ============================================================
# 0/1 embedding via FWHT (TARGET)
# ============================================================
def embed_from_truth_table(tt: List[bool], n: int) -> MV:
    """Embed Boolean function as 0/1 values: True->1, False->0."""
    N = 1 << n
    a = [1.0 if tt[s] else 0.0 for s in range(N)]
    fwht(a)
    invN = 1.0 / N
    return {m: a[m] * invN for m in range(N) if abs(a[m] * invN) > EPS}


def dist_to_target(F: MV, target: MV) -> float:
    return mv_l1(mv_sub(F, target))


# ============================================================
# GA_expr AST
# ============================================================
@dataclass(frozen=True)
class GAExpr:
    pass


@dataclass(frozen=True)
class Scalar(GAExpr):
    q: float


@dataclass(frozen=True)
class Basis(GAExpr):
    i: int


@dataclass(frozen=True)
class Add(GAExpr):
    e1: GAExpr
    e2: GAExpr


@dataclass(frozen=True)
class Mul(GAExpr):
    e1: GAExpr
    e2: GAExpr


@dataclass(frozen=True)
class Conv(GAExpr):
    e1: GAExpr
    e2: GAExpr


# ============================================================
# BoolFormula + 3-SAT
# ============================================================
@dataclass(frozen=True)
class BoolFormula:
    pass


@dataclass(frozen=True)
class BVar(BoolFormula):
    i: int


@dataclass(frozen=True)
class BConst(BoolFormula):
    b: bool


@dataclass(frozen=True)
class BAnd(BoolFormula):
    p: BoolFormula
    q: BoolFormula


@dataclass(frozen=True)
class BNot(BoolFormula):
    p: BoolFormula


@dataclass(frozen=True)
class BOr(BoolFormula):
    p: BoolFormula
    q: BoolFormula


def translate(phi: BoolFormula, n: int) -> GAExpr:
    """
    0/1-world translation (consistent with embed_from_truth_table):
      var x_i ↦ 0.5*(1 + e_i)
      AND     ↦ Conv
      NOT p   ↦ 1 - p
      OR      ↦ p + q - (p AND q)
      TRUE    ↦ 1
      FALSE   ↦ 0
    """
    if isinstance(phi, BVar):
        return Add(Scalar(0.5), Mul(Scalar(-0.5), Basis(phi.i)))
    if isinstance(phi, BConst):
        return Scalar(1.0 if phi.b else 0.0)
    if isinstance(phi, BAnd):
        return Conv(translate(phi.p, n), translate(phi.q, n))
    if isinstance(phi, BNot):
        p = translate(phi.p, n)
        return Add(Scalar(1.0), Mul(Scalar(-1.0), p))  # 1 - p
    if isinstance(phi, BOr):
        p = translate(phi.p, n)
        q = translate(phi.q, n)
        return Add(Add(p, q), Mul(Scalar(-1.0), Conv(p, q)))  # p+q - p∧q
    raise TypeError(f"Unknown BoolFormula: {type(phi)}")


# ============================================================
# 3-SAT generation / evaluation
# ============================================================
@dataclass
class Clause:
    lits: List[Tuple[int, bool]]  # (var_index, is_positive)


def rand_3sat(n_vars: int, n_clauses: int) -> List[Clause]:
    clauses: List[Clause] = []
    for _ in range(n_clauses):
        vars_chosen = random.sample(range(n_vars), 3)
        lits = [(v, random.random() < 0.5) for v in vars_chosen]
        clauses.append(Clause(lits))
    return clauses


def sat3_to_formula(clauses: List[Clause]) -> BoolFormula:
    def lit_to_bf(var: int, pos: bool) -> BoolFormula:
        v = BVar(var)
        return v if pos else BNot(v)

    def clause_to_bf(c: Clause) -> BoolFormula:
        a = lit_to_bf(*c.lits[0])
        b = lit_to_bf(*c.lits[1])
        d = lit_to_bf(*c.lits[2])
        return BOr(BOr(a, b), d)

    if not clauses:
        return BConst(True)

    cur: BoolFormula = clause_to_bf(clauses[0])
    for c in clauses[1:]:
        cur = BAnd(cur, clause_to_bf(c))
    return cur


def eval_3sat(clauses: List[Clause], n_vars: int) -> List[bool]:
    N = 1 << n_vars
    tt: List[bool] = []
    for assignment in range(N):
        sat = True
        for c in clauses:
            clause_sat = False
            for var, pos in c.lits:
                val = bool((assignment >> var) & 1)
                if val == pos:
                    clause_sat = True
                    break
            if not clause_sat:
                sat = False
                break
        tt.append(sat)
    return tt


# ============================================================
# Node classification
# ============================================================
def classify_nodes(e: GAExpr) -> Dict[int, str]:
    tags: Dict[int, str] = {}

    def mark_all(x: GAExpr, kind: str):
        xid = id(x)
        if xid in tags:
            return
        tags[xid] = kind
        if isinstance(x, (Scalar, Basis)):
            return
        if isinstance(x, (Add, Mul, Conv)):
            mark_all(x.e1, kind)
            mark_all(x.e2, kind)

    def is_var_gadget(x: GAExpr) -> bool:
        if not isinstance(x, Mul):
            return False
        if not (isinstance(x.e1, Scalar) and abs(x.e1.q - 0.5) <= EPS):
            return False
        if not isinstance(x.e2, Add):
            return False
        a, b = x.e2.e1, x.e2.e2
        return (isinstance(a, Scalar) and abs(a.q - 1.0) <= EPS and isinstance(b, Basis)) or \
               (isinstance(b, Scalar) and abs(b.q - 1.0) <= EPS and isinstance(a, Basis))

    def walk(x: GAExpr):
        xid = id(x)
        if xid in tags:
            return

        if is_var_gadget(x):
            mark_all(x, "structural")
            return

        if isinstance(x, (Scalar, Basis)):
            tags[xid] = "structural"
            return

        if isinstance(x, Conv):
            tags[xid] = "combinatorial"
            walk(x.e1)
            walk(x.e2)
            return

        if isinstance(x, (Add, Mul)):
            walk(x.e1)
            walk(x.e2)
            c1 = tags.get(id(x.e1), "structural")
            c2 = tags.get(id(x.e2), "structural")
            tags[xid] = "combinatorial" if (c1 == "combinatorial" or c2 == "combinatorial") else "structural"
            return

        raise TypeError(f"Unknown GAExpr node: {type(x)}")

    walk(e)
    return tags


# ============================================================
# Evaluator with dist_rel + overlap_norm
# ============================================================
@dataclass
class NodeStats:
    node_id: int
    label: str
    kind: str  # structural | combinatorial
    l1: float
    max_grade: int
    dist: float
    dist_rel: float          # NaN if target_l1==0
    overlap_norm: float      # similarity-ish (0..1)
    support: int
    grade_energy: List[float]


def eval_with_excursion(e: GAExpr, n: int, target_embed: MV) -> Tuple[MV, List[NodeStats]]:
    tags = classify_nodes(e)
    l1_t = mv_l1(target_embed)

    memo_val: Dict[int, MV] = {}
    nodes: List[NodeStats] = []

    def go(x: GAExpr) -> MV:
        xid = id(x)
        if xid in memo_val:
            return memo_val[xid]

        if isinstance(x, Scalar):
            v = {0: x.q} if abs(x.q) > EPS else {}
            label = f"Scalar({x.q:.2g})"
        elif isinstance(x, Basis):
            v = {1 << x.i: 1.0}
            label = f"e{x.i}"
        elif isinstance(x, Add):
            v = mv_add(go(x.e1), go(x.e2))
            label = "Add"
        elif isinstance(x, Mul):
            v = mv_mul(go(x.e1), go(x.e2), n)
            label = "Mul"
        elif isinstance(x, Conv):
            v = mv_conv_fwht(go(x.e1), go(x.e2), n)
            label = "Conv"
        else:
            raise TypeError(f"Unknown GAExpr: {type(x)}")

        l1_v = mv_l1(v)
        mg = mv_max_grade(v)
        dt = dist_to_target(v, target_embed)

        # Relative-to-target distance (undefined when target_l1==0)
        if l1_t > 1e-12:
            drel = dt / l1_t
        else:
            drel = float("nan")

        # Overlap-based similarity (robust to disjoint-support saturation)
        overlap = 0.5 * (l1_v + l1_t - dt)
        if overlap < 0.0:
            overlap = 0.0
        overlap_norm = overlap / (min(l1_v, l1_t) + 1e-12)

        nodes.append(NodeStats(
            node_id=len(nodes),
            label=label,
            kind=tags.get(xid, "structural"),
            l1=l1_v,
            max_grade=mg,
            dist=dt,
            dist_rel=drel,
            overlap_norm=overlap_norm,
            support=mv_support(v),
            grade_energy=mv_grade_profile(v, n),
        ))

        memo_val[xid] = v
        return v

    final = go(e)
    return final, nodes


# ============================================================
# CSV
# ============================================================
SUMMARY_FIELDS = [
    "n_vars", "n_clauses", "clause_ratio", "trial",
    "n_sat", "frac_sat", "is_unsat",
    "target_l1",

    "peak_l1", "peak_dist", "peak_dist_rel", "peak_overlap", "peak_max_grade",
    "peak_l1_pos", "peak_dist_pos", "peak_grade_pos",

    "comb_peak_l1", "comb_peak_dist", "comb_peak_dist_rel", "comb_peak_overlap", "comb_peak_grade",
    "comb_peak_l1_pos", "comb_peak_dist_pos",

    "comb_exc_dl",     # max(dist * l1)
    "comb_exc_dlr",    # max(dist_rel * l1) over finite dist_rel
    "comb_exc_dlg",    # max(dist * l1 * grade)

    "final_l1", "final_dist", "final_dist_rel", "final_overlap", "final_max_grade",

    "n_nodes", "n_comb_nodes",
]

TRACE_FIELDS = [
    "n_vars", "n_clauses", "clause_ratio", "trial",
    "node_id", "label", "kind",
    "l1", "max_grade",
    "dist", "dist_rel", "overlap_norm",
    "support",
]


def init_csvs():
    with open(OUTFILE_SUMMARY, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()
    if WRITE_TRACES:
        with open(OUTFILE_TRACES, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRACE_FIELDS).writeheader()


def append_summary(row: dict):
    with open(OUTFILE_SUMMARY, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writerow(row)


def append_trace_rows(rows: List[dict]):
    if not WRITE_TRACES:
        return
    with open(OUTFILE_TRACES, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        for r in rows:
            w.writerow(r)


# ============================================================
# Sanity checks
# ============================================================
def _truth_table_of_formula(phi: BoolFormula, n: int) -> List[bool]:
    N = 1 << n

    def eval_phi(p: BoolFormula, a: int) -> bool:
        if isinstance(p, BVar):
            return bool((a >> p.i) & 1)
        if isinstance(p, BConst):
            return p.b
        if isinstance(p, BNot):
            return not eval_phi(p.p, a)
        if isinstance(p, BAnd):
            return eval_phi(p.p, a) and eval_phi(p.q, a)
        if isinstance(p, BOr):
            return eval_phi(p.p, a) or eval_phi(p.q, a)
        raise TypeError(type(p))

    return [eval_phi(phi, a) for a in range(N)]


def run_sanity_checks():
    print("\n" + "=" * 60)
    print("SANITY CHECKS (n=1) — 0/1 embedding target")
    print("=" * 60)

    n = 1
    tests: List[Tuple[str, BoolFormula]] = [
        ("x0", BVar(0)),
        ("~x0", BNot(BVar(0))),
        ("x0 OR ~x0", BOr(BVar(0), BNot(BVar(0)))),
        ("x0 AND ~x0", BAnd(BVar(0), BNot(BVar(0)))),
        ("TRUE", BConst(True)),
        ("FALSE", BConst(False)),
    ]

    for name, phi in tests:
        tt = _truth_table_of_formula(phi, n)
        target = embed_from_truth_table(tt, n)
        ga = translate(phi, n)
        _, nodes = eval_with_excursion(ga, n, target)
        fd = nodes[-1].dist
        print(
            f"  {name:12s}  sat_frac={sum(tt)/len(tt):.2f}  "
            f"final_dist={fd:.6f}  target_l1={mv_l1(target):.6f}"
        )


# ============================================================
# Main sweep
# ============================================================
def _pos(i: int, n: int) -> float:
    return 0.0 if n <= 1 else (i / (n - 1))


def run_sweep():
    random.seed(SEED)
    init_csvs()

    if SANITY_CHECKS:
        run_sanity_checks()

    total = len(CLAUSE_RATIOS) * TRIALS_PER_RATIO
    done = 0
    t0 = time.time()

    for ratio in CLAUSE_RATIOS:
        n_clauses = max(1, int(round(ratio * N_VARS)))
        print(f"\n{'=' * 60}")
        print(f"clause_ratio={ratio:.3f}  n_clauses={n_clauses}  n_vars={N_VARS}")
        print(f"{'=' * 60}")

        for trial in range(TRIALS_PER_RATIO):
            clauses = rand_3sat(N_VARS, n_clauses)
            phi = sat3_to_formula(clauses)

            tt = eval_3sat(clauses, N_VARS)
            n_sat = sum(tt)
            frac_sat = n_sat / (1 << N_VARS)
            is_unsat = (n_sat == 0)

            target = embed_from_truth_table(tt, N_VARS)
            target_l1 = mv_l1(target)

            ga_expr = translate(phi, N_VARS)
            _, nodes = eval_with_excursion(ga_expr, N_VARS, target)

            nn = len(nodes)
            comb = [nd for nd in nodes if nd.kind == "combinatorial"]

            # all-node peaks
            peak_l1 = max(nd.l1 for nd in nodes)
            peak_dist = max(nd.dist for nd in nodes)
            peak_mg = max(nd.max_grade for nd in nodes)

            i_pl1 = max(range(nn), key=lambda i: nodes[i].l1)
            i_pd = max(range(nn), key=lambda i: nodes[i].dist)
            i_pg = max(range(nn), key=lambda i: nodes[i].max_grade)

            peak_dist_rel = nodes[i_pd].dist_rel
            peak_overlap = nodes[i_pd].overlap_norm

            # combinatorial peaks
            if comb:
                c_peak_l1 = max(nd.l1 for nd in comb)
                c_peak_dist = max(nd.dist for nd in comb)
                c_peak_grade = max(nd.max_grade for nd in comb)

                c_ids = [nd.node_id for nd in comb]
                i_cpl1 = max(c_ids, key=lambda i: nodes[i].l1)
                i_cpd = max(c_ids, key=lambda i: nodes[i].dist)

                c_peak_dist_rel = nodes[i_cpd].dist_rel
                c_peak_overlap = nodes[i_cpd].overlap_norm

                c_exc_dl = max(nd.dist * nd.l1 for nd in comb)
                # dist_rel can be NaN when target_l1==0; ignore non-finite
                finite_dlr = [nd.dist_rel * nd.l1 for nd in comb if nd.dist_rel == nd.dist_rel]
                c_exc_dlr = max(finite_dlr) if finite_dlr else float("nan")
                c_exc_dlg = max(nd.dist * nd.l1 * nd.max_grade for nd in comb)
            else:
                c_peak_l1 = c_peak_dist = 0.0
                c_peak_grade = 0
                i_cpl1 = i_cpd = 0
                c_peak_dist_rel = float("nan")
                c_peak_overlap = 0.0
                c_exc_dl = float("nan")
                c_exc_dlr = float("nan")
                c_exc_dlg = float("nan")

            fn = nodes[-1]

            row = {
                "n_vars": N_VARS,
                "n_clauses": n_clauses,
                "clause_ratio": ratio,
                "trial": trial,
                "n_sat": n_sat,
                "frac_sat": round(frac_sat, 6),
                "is_unsat": int(is_unsat),
                "target_l1": round(target_l1, 8),

                "peak_l1": round(peak_l1, 8),
                "peak_dist": round(peak_dist, 8),
                "peak_dist_rel": peak_dist_rel,
                "peak_overlap": round(peak_overlap, 8),
                "peak_max_grade": peak_mg,
                "peak_l1_pos": round(_pos(i_pl1, nn), 4),
                "peak_dist_pos": round(_pos(i_pd, nn), 4),
                "peak_grade_pos": round(_pos(i_pg, nn), 4),

                "comb_peak_l1": round(c_peak_l1, 8),
                "comb_peak_dist": round(c_peak_dist, 8),
                "comb_peak_dist_rel": c_peak_dist_rel,
                "comb_peak_overlap": round(c_peak_overlap, 8),
                "comb_peak_grade": c_peak_grade,
                "comb_peak_l1_pos": round(_pos(i_cpl1, nn), 4),
                "comb_peak_dist_pos": round(_pos(i_cpd, nn), 4),

                "comb_exc_dl": round(c_exc_dl, 8) if c_exc_dl == c_exc_dl else c_exc_dl,
                "comb_exc_dlr": round(c_exc_dlr, 8) if c_exc_dlr == c_exc_dlr else c_exc_dlr,
                "comb_exc_dlg": round(c_exc_dlg, 8) if c_exc_dlg == c_exc_dlg else c_exc_dlg,

                "final_l1": round(fn.l1, 8),
                "final_dist": round(fn.dist, 8),
                "final_dist_rel": fn.dist_rel,
                "final_overlap": round(fn.overlap_norm, 8),
                "final_max_grade": fn.max_grade,

                "n_nodes": nn,
                "n_comb_nodes": len(comb),
            }
            append_summary(row)

            if WRITE_TRACES:
                trace_rows: List[dict] = []
                for nd in nodes:
                    trace_rows.append({
                        "n_vars": N_VARS,
                        "n_clauses": n_clauses,
                        "clause_ratio": ratio,
                        "trial": trial,
                        "node_id": nd.node_id,
                        "label": nd.label,
                        "kind": nd.kind,
                        "l1": round(nd.l1, 8),
                        "max_grade": nd.max_grade,
                        "dist": round(nd.dist, 8),
                        "dist_rel": nd.dist_rel,
                        "overlap_norm": round(nd.overlap_norm, 8),
                        "support": nd.support,
                    })
                append_trace_rows(trace_rows)

            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                sat_tag = "UNSAT" if is_unsat else f"{frac_sat:.1%}sat"
                print(
                    f"  r={ratio:.3f} t={trial:3d}  "
                    f"pk_l1={c_peak_l1:.3f} pk_dist={c_peak_dist:.4f} "
                    f"pk_gr={c_peak_grade} exc_dl={c_exc_dl:.4f}  "
                    f"{sat_tag}  [{done}/{total} {rate:.1f}/s]"
                )

    elapsed = time.time() - t0
    print(f"\nDone. {total} trials in {elapsed:.1f}s")
    print(f"Summary: {OUTFILE_SUMMARY}")
    if WRITE_TRACES:
        print(f"Traces:  {OUTFILE_TRACES}")


if __name__ == "__main__":
    run_sweep()