"""
cln_sat3.py — 3-SAT excursion profiler.
Tracks (dist_to_target, l1, max_grade) at every node.
"""
from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# ============================================================
# Config
# ============================================================
N_VARS = 10
CLAUSE_RATIOS = [2.0, 3.0, 3.5, 4.0, 4.267, 4.5, 5.0, 6.0, 8.0]
TRIALS_PER_RATIO = 50
SEED = 42
OUTFILE_SUMMARY = "sat3_excursion_summary.csv"
OUTFILE_TRACES  = "sat3_excursion_traces.csv"
WRITE_TRACES = True

EPS = 1e-15

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
    if a == 0.0:
        return {}
    return {k: a * c for k, c in u.items() if abs(a * c) > EPS}

def mv_sub(u: MV, v: MV) -> MV:
    return mv_add(u, mv_scale(-1.0, v))

def mv_l1(u: MV) -> float:
    return sum(abs(c) for c in u.values())

def mv_support(u: MV) -> int:
    return len(u)

def popcount(x: int) -> int:
    return bin(x).count("1")

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
    for k, v in F.items(): A[k] = v
    for k, v in G.items(): B[k] = v
    fwht(A); fwht(B)
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
            s += bin(b & ((1 << i) - 1)).count("1")
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
# Embed via FWHT
# ============================================================
def embed_from_truth_table(tt: List[bool], n: int) -> MV:
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
class GAExpr: ...

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
class BoolFormula: ...

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
    if isinstance(phi, BVar):
        return Mul(Scalar(0.5), Add(Scalar(1.0), Basis(phi.i)))
    if isinstance(phi, BConst):
        return Scalar(1.0 if phi.b else 0.0)
    if isinstance(phi, BAnd):
        p = translate(phi.p, n)
        q = translate(phi.q, n)
        return Conv(p, q)
    if isinstance(phi, BNot):
        p = translate(phi.p, n)
        return Add(Scalar(1.0), Mul(Scalar(-1.0), p))
    if isinstance(phi, BOr):
        p = translate(phi.p, n)
        q = translate(phi.q, n)
        return Add(Add(p, q), Mul(Scalar(-1.0), Conv(p, q)))
    raise TypeError

# ============================================================
# 3-SAT generation
# ============================================================
@dataclass
class Clause:
    lits: List[Tuple[int, bool]]

def rand_3sat(n_vars: int, n_clauses: int) -> List[Clause]:
    clauses = []
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
        l0 = lit_to_bf(*c.lits[0])
        l1 = lit_to_bf(*c.lits[1])
        l2 = lit_to_bf(*c.lits[2])
        return BOr(BOr(l0, l1), l2)

    if not clauses:
        return BConst(True)
    cur = clause_to_bf(clauses[0])
    for c in clauses[1:]:
        cur = BAnd(cur, clause_to_bf(c))
    return cur

def eval_3sat(clauses: List[Clause], n_vars: int) -> List[bool]:
    N = 1 << n_vars
    tt = []
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
# "structural" = inside the mechanical variable translation
# "combinatorial" = combining clauses (Conv/Add at clause level)

def classify_nodes(e: GAExpr) -> Dict[int, str]:
    """
    Tag every node by id(e) -> 'structural' | 'combinatorial'.
    The variable translation Mul(0.5, Add(1, e_i)) and literal NOT
    wrappers are structural. Everything above clause level is combinatorial.
    """
    tags: Dict[int, str] = {}

    def mark_structural(x: GAExpr):
        tags[id(x)] = "structural"
        if isinstance(x, (Scalar, Basis)):
            pass
        elif isinstance(x, Add):
            mark_structural(x.e1)
            mark_structural(x.e2)
        elif isinstance(x, Mul):
            mark_structural(x.e1)
            mark_structural(x.e2)
        elif isinstance(x, Conv):
            mark_structural(x.e1)
            mark_structural(x.e2)

    def walk(x: GAExpr, inside_literal: bool):
        xid = id(x)
        if xid in tags:
            return

        if isinstance(x, BVar.__class__):
            return

        if isinstance(x, Scalar):
            tags[xid] = "structural"
        elif isinstance(x, Basis):
            tags[xid] = "structural"
        elif isinstance(x, Mul):
            # Mul(0.5, Add(1, e_i)) is variable translation
            if isinstance(x.e1, Scalar) and isinstance(x.e2, Add):
                # this is a variable node — mark entire subtree structural
                mark_structural(x)
            elif isinstance(x.e1, Scalar) and x.e1.q == -1.0:
                # Mul(-1, ...) is part of NOT or OR correction
                tags[xid] = "structural"
                walk(x.e2, inside_literal)
            else:
                tags[xid] = "combinatorial"
                walk(x.e1, False)
                walk(x.e2, False)
        elif isinstance(x, Add):
            # Add(1, Mul(-1, p)) is NOT
            # Add(Add(p,q), Mul(-1, Conv(p,q))) is OR
            # In both cases the Add itself is combinatorial only if at clause-combine level
            # Simpler: if any child is combinatorial, we are combinatorial
            walk(x.e1, False)
            walk(x.e2, False)
            c1 = tags.get(id(x.e1), "structural")
            c2 = tags.get(id(x.e2), "structural")
            if c1 == "combinatorial" or c2 == "combinatorial":
                tags[xid] = "combinatorial"
            else:
                tags[xid] = "structural"
        elif isinstance(x, Conv):
            # Conv is always combinatorial (clause combining)
            tags[xid] = "combinatorial"
            walk(x.e1, False)
            walk(x.e2, False)

    walk(e, False)
    return tags

# ============================================================
# DAG evaluator with excursion tracking
# ============================================================
@dataclass
class NodeStats:
    node_id: int
    label: str
    kind: str           # "structural" or "combinatorial"
    l1: float
    max_grade: int
    dist_to_target: float
    support: int
    grade_energy: List[float]

def eval_with_excursion(
    e: GAExpr, n: int,
    target_embed: MV,
) -> Tuple[MV, List[NodeStats]]:
    tags = classify_nodes(e)

    memo_val: Dict[int, MV] = {}
    nodes: List[NodeStats] = []

    def go(x: GAExpr) -> MV:
        xid = id(x)
        if xid in memo_val:
            return memo_val[xid]

        if isinstance(x, Scalar):
            v = {0: x.q} if x.q != 0.0 else {}
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
            raise TypeError

        l1 = mv_l1(v)
        mg = mv_max_grade(v)
        dt = dist_to_target(v, target_embed)
        gp = mv_grade_profile(v, n)
        kind = tags.get(xid, "structural")

        nid = len(nodes)
        nodes.append(NodeStats(
            node_id=nid, label=label, kind=kind,
            l1=l1, max_grade=mg,
            dist_to_target=dt,
            support=mv_support(v),
            grade_energy=gp,
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
    # ALL nodes
    "peak_l1", "peak_dist", "peak_max_grade",
    "peak_l1_pos", "peak_dist_pos", "peak_grade_pos",
    # COMBINATORIAL nodes only
    "comb_peak_l1", "comb_peak_dist", "comb_peak_grade",
    "comb_peak_l1_pos", "comb_peak_dist_pos",
    # composite excursion (combinatorial only)
    "comb_exc_dl",      # max(dist * l1) over combinatorial nodes
    "comb_exc_dlg",     # max(dist * l1 * grade)
    "comb_exc_d",       # max(dist) over combinatorial nodes
    # final
    "final_l1", "final_dist", "final_max_grade",
    # means (combinatorial)
    "comb_mean_l1", "comb_mean_dist", "comb_mean_grade",
    # counts
    "n_nodes", "n_comb_nodes",
    # grade spread
    "comb_max_grade_in_trace",
    # target l1 (l1 norm of embed(f) itself)
    "target_l1",
]

TRACE_FIELDS = [
    "n_vars", "n_clauses", "clause_ratio", "trial",
    "node_id", "label", "kind",
    "l1", "max_grade", "dist_to_target", "support",
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
# Main sweep
# ============================================================
def run_sweep():
    random.seed(SEED)
    init_csvs()

    total = len(CLAUSE_RATIOS) * TRIALS_PER_RATIO
    done = 0
    t0 = time.time()

    for ratio in CLAUSE_RATIOS:
        n_clauses = max(1, int(round(ratio * N_VARS)))
        print(f"\n{'='*60}")
        print(f"clause_ratio={ratio:.3f}  n_clauses={n_clauses}  n_vars={N_VARS}")
        print(f"{'='*60}")

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
            final_mv, nodes = eval_with_excursion(ga_expr, N_VARS, target)

            nn = len(nodes)
            comb = [nd for nd in nodes if nd.kind == "combinatorial"]
            nc = len(comb) if comb else 1  # avoid div by zero

            # all nodes
            peak_l1 = max(nd.l1 for nd in nodes)
            peak_dist = max(nd.dist_to_target for nd in nodes)
            peak_mg = max(nd.max_grade for nd in nodes)

            i_pl1 = max(range(nn), key=lambda i: nodes[i].l1)
            i_pd  = max(range(nn), key=lambda i: nodes[i].dist_to_target)
            i_pg  = max(range(nn), key=lambda i: nodes[i].max_grade)

            # combinatorial nodes
            if comb:
                c_peak_l1 = max(nd.l1 for nd in comb)
                c_peak_dist = max(nd.dist_to_target for nd in comb)
                c_peak_grade = max(nd.max_grade for nd in comb)

                c_ids = [nd.node_id for nd in comb]
                i_cpl1 = max(c_ids, key=lambda i: nodes[i].l1)
                i_cpd  = max(c_ids, key=lambda i: nodes[i].dist_to_target)

                c_exc_dl  = max(nd.dist_to_target * nd.l1 for nd in comb)
                c_exc_dlg = max(nd.dist_to_target * nd.l1 * nd.max_grade for nd in comb)
                c_exc_d   = c_peak_dist

                c_mean_l1 = sum(nd.l1 for nd in comb) / nc
                c_mean_dist = sum(nd.dist_to_target for nd in comb) / nc
                c_mean_grade = sum(nd.max_grade for nd in comb) / nc
                c_max_grade = max(nd.max_grade for nd in comb)
            else:
                c_peak_l1 = c_peak_dist = c_peak_grade = 0
                i_cpl1 = i_cpd = 0
                c_exc_dl = c_exc_dlg = c_exc_d = 0
                c_mean_l1 = c_mean_dist = c_mean_grade = c_max_grade = 0

            fn = nodes[-1]

            row = {
                "n_vars": N_VARS,
                "n_clauses": n_clauses,
                "clause_ratio": ratio,
                "trial": trial,
                "n_sat": n_sat,
                "frac_sat": round(frac_sat, 6),
                "is_unsat": int(is_unsat),

                "peak_l1": round(peak_l1, 8),
                "peak_dist": round(peak_dist, 8),
                "peak_max_grade": peak_mg,
                "peak_l1_pos": round(i_pl1 / nn, 4),
                "peak_dist_pos": round(i_pd / nn, 4),
                "peak_grade_pos": round(i_pg / nn, 4),

                "comb_peak_l1": round(c_peak_l1, 8),
                "comb_peak_dist": round(c_peak_dist, 8),
                "comb_peak_grade": c_peak_grade,
                "comb_peak_l1_pos": round(i_cpl1 / nn, 4),
                "comb_peak_dist_pos": round(i_cpd / nn, 4),

                "comb_exc_dl": round(c_exc_dl, 8),
                "comb_exc_dlg": round(c_exc_dlg, 8),
                "comb_exc_d": round(c_exc_d, 8),

                "final_l1": round(fn.l1, 8),
                "final_dist": round(fn.dist_to_target, 8),
                "final_max_grade": fn.max_grade,

                "comb_mean_l1": round(c_mean_l1, 6),
                "comb_mean_dist": round(c_mean_dist, 6),
                "comb_mean_grade": round(c_mean_grade, 4),

                "n_nodes": nn,
                "n_comb_nodes": len(comb),
                "comb_max_grade_in_trace": c_max_grade,
                "target_l1": round(target_l1, 8),
            }
            append_summary(row)

            if WRITE_TRACES:
                trace_rows = []
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
                        "dist_to_target": round(nd.dist_to_target, 8),
                        "support": nd.support,
                    })
                append_trace_rows(trace_rows)

            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                sat_tag = "UNSAT" if is_unsat else f"{frac_sat:.1%}sat"
                print(f"  r={ratio:.3f} t={trial:3d}  "
                      f"pk_l1={c_peak_l1:.3f} pk_dist={c_peak_dist:.4f} "
                      f"pk_gr={c_peak_grade} exc_dl={c_exc_dl:.4f}  "
                      f"{sat_tag}  [{done}/{total} {rate:.1f}/s]")

    elapsed = time.time() - t0
    print(f"\nDone. {total} trials in {elapsed:.1f}s")
    print(f"Summary: {OUTFILE_SUMMARY}")
    if WRITE_TRACES:
        print(f"Traces:  {OUTFILE_TRACES}")

if __name__ == "__main__":
    run_sweep()