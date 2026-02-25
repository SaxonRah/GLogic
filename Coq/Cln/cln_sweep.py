"""
cln_sweep.py — Mul-vs-Conv excursion sweep over AND depth.
Writes results incrementally to CSV.  Uses floats for speed.
"""
from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

# ============================================================
# Config
# ============================================================
N = 10
TRIALS_PER_DEPTH = 200
AND_DEPTHS = [1, 2, 3, 4, 5, 6]
OR_CHAIN_K_RANGE = (4, 14)       # k for each OR chain
SEED = 42
OUTFILE = "cln_sweep_results.csv"

# ============================================================
# MV = Dict[int, float]  (sparse)
# ============================================================
MV = Dict[int, float]
EPS = 1e-15

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

def parity(x: int) -> int:
    return bin(x).count("1") & 1

# ============================================================
# FWHT convolution (floats)
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

def gp_blade(a: int, b: int, n: int) -> Tuple[float, int]:
    # metric = +1 for all generators (Euclidean signature)
    res = a ^ b
    sign = clifford_swap_sign(a, b, n)
    return float(sign), res

def mv_mul(u: MV, v: MV, n: int) -> MV:
    out: MV = {}
    for a, ca in u.items():
        for b, cb in v.items():
            coeff, res = gp_blade(a, b, n)
            val = ca * cb * coeff
            out[res] = out.get(res, 0.0) + val
    return {k: c for k, c in out.items() if abs(c) > EPS}

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
# BoolFormula AST + translate
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

def translate(phi: BoolFormula, n: int, and_uses_mul: bool) -> GAExpr:
    if isinstance(phi, BVar):
        return Mul(Scalar(0.5), Add(Scalar(1.0), Basis(phi.i)))
    if isinstance(phi, BConst):
        return Scalar(1.0 if phi.b else 0.0)
    if isinstance(phi, BAnd):
        p = translate(phi.p, n, and_uses_mul)
        q = translate(phi.q, n, and_uses_mul)
        return Mul(p, q) if and_uses_mul else Conv(p, q)
    if isinstance(phi, BNot):
        p = translate(phi.p, n, and_uses_mul)
        return Add(Scalar(1.0), Mul(Scalar(-1.0), p))
    if isinstance(phi, BOr):
        p = translate(phi.p, n, and_uses_mul)
        q = translate(phi.q, n, and_uses_mul)
        return Add(Add(p, q), Mul(Scalar(-1.0), Conv(p, q)))
    raise TypeError

# ============================================================
# DAG evaluator — returns peak l1 and final l1
# ============================================================
def eval_stats(e: GAExpr, n: int) -> Tuple[float, float, int, int]:
    """
    Returns (peak_l1, final_l1, peak_support, num_nodes).
    DAG-memoized.
    """
    memo: Dict[int, MV] = {}
    peak = 0.0
    peak_supp = 0
    node_count = 0

    def go(x: GAExpr) -> MV:
        nonlocal peak, peak_supp, node_count
        xid = id(x)
        if xid in memo:
            return memo[xid]

        if isinstance(x, Scalar):
            v = {0: x.q} if x.q != 0.0 else {}
        elif isinstance(x, Basis):
            v = {1 << x.i: 1.0}
        elif isinstance(x, Add):
            v = mv_add(go(x.e1), go(x.e2))
        elif isinstance(x, Mul):
            v = mv_mul(go(x.e1), go(x.e2), n)
        elif isinstance(x, Conv):
            v = mv_conv_fwht(go(x.e1), go(x.e2), n)
        else:
            raise TypeError

        l1 = mv_l1(v)
        supp = mv_support(v)
        if l1 > peak:
            peak = l1
            peak_supp = supp
        node_count += 1
        memo[xid] = v
        return v

    final = go(e)
    return peak, mv_l1(final), peak_supp, node_count

# ============================================================
# Formula generators
# ============================================================
def rand_lit(n: int, p_neg: float = 0.5) -> BoolFormula:
    v = BVar(random.randrange(n))
    return BNot(v) if random.random() < p_neg else v

def rand_or_chain(n: int, k: int) -> BoolFormula:
    lits = [rand_lit(n) for _ in range(k)]
    cur = lits[0]
    for t in lits[1:]:
        cur = BOr(cur, t)
    return cur

def rand_nested_and(n: int, and_depth: int, k_range: Tuple[int, int]) -> BoolFormula:
    """
    Build a formula with controlled AND depth:
      depth 1: S & S
      depth 2: (S&S) & (S&S)
      depth 3: ((S&S)&(S&S)) & ((S&S)&(S&S))
      etc.
    Each leaf S is an independent random OR chain.
    """
    if and_depth <= 0:
        k = random.randint(*k_range)
        return rand_or_chain(n, k)
    left = rand_nested_and(n, and_depth - 1, k_range)
    right = rand_nested_and(n, and_depth - 1, k_range)
    return BAnd(left, right)

def count_ops(phi: BoolFormula) -> Tuple[int, int, int]:
    """(#ands, #ors, #nots)"""
    if isinstance(phi, (BVar, BConst)):
        return (0, 0, 0)
    if isinstance(phi, BAnd):
        a1, o1, n1 = count_ops(phi.p)
        a2, o2, n2 = count_ops(phi.q)
        return (a1 + a2 + 1, o1 + o2, n1 + n2)
    if isinstance(phi, BOr):
        a1, o1, n1 = count_ops(phi.p)
        a2, o2, n2 = count_ops(phi.q)
        return (a1 + a2, o1 + o2 + 1, n1 + n2)
    if isinstance(phi, BNot):
        a, o, nn = count_ops(phi.p)
        return (a, o, nn + 1)
    raise TypeError

def formula_str(phi: BoolFormula) -> str:
    if isinstance(phi, BVar): return f"x{phi.i}"
    if isinstance(phi, BConst): return "T" if phi.b else "F"
    if isinstance(phi, BNot): return f"(~{formula_str(phi.p)})"
    if isinstance(phi, BAnd): return f"({formula_str(phi.p)}&{formula_str(phi.q)})"
    if isinstance(phi, BOr):  return f"({formula_str(phi.p)}|{formula_str(phi.q)})"
    return "?"

# ============================================================
# CSV writer (incremental)
# ============================================================
FIELDNAMES = [
    "and_depth", "trial", "n_vars",
    "n_ands", "n_ors", "n_nots",
    "peak_mul", "peak_conv", "peak_gap", "peak_ratio",
    "final_mul", "final_conv", "final_gap", "final_ratio",
    "mul_nodes", "conv_nodes",
    "mul_peak_supp", "conv_peak_supp",
    "has_gap",
    "formula_len",
]

def init_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()

def append_row(path: str, row: dict) -> None:
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writerow(row)

# ============================================================
# Main sweep
# ============================================================
def run_sweep():
    random.seed(SEED)
    init_csv(OUTFILE)

    total = len(AND_DEPTHS) * TRIALS_PER_DEPTH
    done = 0
    t0 = time.time()

    for depth in AND_DEPTHS:
        print(f"\n{'='*60}")
        print(f"AND depth = {depth}  ({2**depth} leaf OR-chains)")
        print(f"{'='*60}")

        for trial in range(TRIALS_PER_DEPTH):
            phi = rand_nested_and(N, depth, OR_CHAIN_K_RANGE)
            n_ands, n_ors, n_nots = count_ops(phi)

            e_mul = translate(phi, N, and_uses_mul=True)
            e_con = translate(phi, N, and_uses_mul=False)

            pm, fm, pm_supp, mn = eval_stats(e_mul, N)
            pc, fc, pc_supp, cn = eval_stats(e_con, N)

            gap = pm - pc
            fgap = fm - fc
            has_gap = abs(gap) > 1e-10 or abs(fgap) > 1e-10

            row = {
                "and_depth": depth,
                "trial": trial,
                "n_vars": N,
                "n_ands": n_ands,
                "n_ors": n_ors,
                "n_nots": n_nots,
                "peak_mul": round(pm, 10),
                "peak_conv": round(pc, 10),
                "peak_gap": round(gap, 10),
                "peak_ratio": round(pm / pc, 6) if pc > 1e-15 else float("inf"),
                "final_mul": round(fm, 10),
                "final_conv": round(fc, 10),
                "final_gap": round(fgap, 10),
                "final_ratio": round(fm / fc, 6) if fc > 1e-15 else float("inf"),
                "mul_nodes": mn,
                "conv_nodes": cn,
                "mul_peak_supp": pm_supp,
                "conv_peak_supp": pc_supp,
                "has_gap": int(has_gap),
                "formula_len": len(formula_str(phi)),
            }
            append_row(OUTFILE, row)

            done += 1
            if done % 50 == 0 or has_gap:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                marker = " *** GAP" if has_gap else ""
                print(f"  depth={depth} trial={trial:4d}  "
                      f"peak_mul={pm:.4f} peak_conv={pc:.4f} gap={gap:+.4f} "
                      f"ratio={pm/pc if pc>0 else 0:.3f}"
                      f"  [{done}/{total} {rate:.1f}/s]{marker}")

    elapsed = time.time() - t0
    print(f"\nDone. {total} trials in {elapsed:.1f}s. Results in {OUTFILE}")

if __name__ == "__main__":
    run_sweep()