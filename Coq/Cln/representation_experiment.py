#!/usr/bin/env python3
"""
representation_experiment.py

DECISIVE EXPERIMENT (two-track):

Track A (Expanded): expand the compiled GAProg into monomials and measure support
         -> tells you what "normal form / coefficient table" does (often exponential).

Track B (Factored): measure GAProg AST/DAG size & depth under *gate-by-gate compilation*
         -> this is the real “does representation compose like computation?” test.

What this script answers:
1) Do common boolean computations admit small *factored* representations? (usually yes)
2) Does gate-by-gate compilation keep those representations polynomial WITHOUT requiring
   global re-factorization? (this is what you care about)
3) Where is the blow-up coming from: expansion/normalization, or the program itself?

Run:
  python representation_experiment.py
  python representation_experiment.py --max_n 40 --expand_n 12
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple
import argparse
import math
import random
import sys

# =============================================================================
# GAProg (really: an arithmetic/GA "program AST" used as a representation language)
# =============================================================================

@dataclass(frozen=True)
class Node:
    op: str
    args: Tuple[Any, ...]  # Node or scalar or var-index
    # Because this is frozen, it's hashable -> easy DAG unique-counting


def Const(c: float) -> Node:
    return Node("const", (float(c),))


def Var(i: int) -> Node:
    # "blade i" corresponds to variable x_i in {±1} (grade-1 vector)
    return Node("var", (int(i),))


def Add(a: Node, b: Node) -> Node:
    return Node("add", (a, b))


def Mul(a: Node, b: Node) -> Node:
    return Node("mul", (a, b))


def Scale(k: float, a: Node) -> Node:
    return Node("scale", (float(k), a))


# -----------------------------------------------------------------------------
# Structural metrics on representations
# -----------------------------------------------------------------------------

def tree_size(n: Node) -> int:
    """Counts nodes with duplication (pure AST tree size)."""
    op = n.op
    if op in ("const", "var"):
        return 1
    if op == "scale":
        return 1 + tree_size(n.args[1])
    if op in ("add", "mul"):
        return 1 + tree_size(n.args[0]) + tree_size(n.args[1])
    raise ValueError(f"Unknown op: {op}")


def tree_depth(n: Node) -> int:
    """Depth with duplication (tree). Leaves have depth 0."""
    op = n.op
    if op in ("const", "var"):
        return 0
    if op == "scale":
        return 1 + tree_depth(n.args[1])
    if op in ("add", "mul"):
        return 1 + max(tree_depth(n.args[0]), tree_depth(n.args[1]))
    raise ValueError(f"Unknown op: {op}")


def vars_used(n: Node) -> Set[int]:
    """Set of variable indices used anywhere in the program (tree semantics)."""
    out: Set[int] = set()

    def rec(x: Node):
        if x.op == "var":
            out.add(x.args[0])
            return
        for a in x.args:
            if isinstance(a, Node):
                rec(a)

    rec(n)
    return out


def dag_unique_nodes(n: Node) -> int:
    """Counts unique nodes (DAG size), exploiting hash-consing by Node hashing."""
    seen: Set[Node] = set()

    def rec(x: Node):
        if x in seen:
            return
        seen.add(x)
        for a in x.args:
            if isinstance(a, Node):
                rec(a)

    rec(n)
    return len(seen)


def max_grade_upper_bound(n: Node) -> int:
    """
    Conservative upper bound on max grade if you interpret Var(i) as grade-1.

    Note: In real Clifford/GA multiplication, max grade is not just "sum of max grades"
    for arbitrary multivectors; this is an upper bound. For these boolean-arithmetic
    encodings, it’s still a useful structural proxy.
    """
    if n.op == "const":
        return 0
    if n.op == "var":
        return 1
    if n.op == "scale":
        return max_grade_upper_bound(n.args[1])
    if n.op == "add":
        return max(max_grade_upper_bound(n.args[0]), max_grade_upper_bound(n.args[1]))
    if n.op == "mul":
        return max_grade_upper_bound(n.args[0]) + max_grade_upper_bound(n.args[1])
    raise ValueError(f"Unknown op: {n.op}")


# =============================================================================
# Gate library (compilation templates)
# =============================================================================
# Convention:
# - Variables x_i take values in {±1}
# - Outputs are in {0,1} (reals), exact for these gates.

def AND_gate(p: Node, q: Node) -> Node:
    # AND(p,q) in {0,1} when p,q in {±1}? Not directly.
    # We instead compile AND of *literals* x,y in ±1:
    #
    # AND(x,y) = (1/4)(1+x)(1+y)  (outputs 1 iff both +1 else 0)
    #
    # For general p,q that are already {0,1}, the boolean AND is p*q.
    # But in many compiled pipelines we keep {0,1} signals after first layer.
    #
    # Here we assume p,q are {0,1} signals, so AND = p*q.
    return Mul(p, q)


def OR_gate(p: Node, q: Node) -> Node:
    # For {0,1} signals: OR = p + q - p*q
    return Add(Add(p, q), Scale(-1.0, Mul(p, q)))


def NOT_gate(p: Node) -> Node:
    # For {0,1} signals: NOT = 1 - p
    return Add(Const(1.0), Scale(-1.0, p))


def XOR_gate(p: Node, q: Node) -> Node:
    # For {0,1} signals: XOR = p + q - 2pq
    return Add(Add(p, q), Scale(-2.0, Mul(p, q)))


# Literal encodings from ±1 vars into {0,1}:
def LIT_pos(i: int) -> Node:
    # (1+x_i)/2
    return Scale(0.5, Add(Const(1.0), Var(i)))


def LIT_neg(i: int) -> Node:
    # (1-x_i)/2
    return Scale(0.5, Add(Const(1.0), Scale(-1.0, Var(i))))


# =============================================================================
# Circuit builders (gate-by-gate compilation)
# =============================================================================

def chain_AND_of_literals(n: int) -> Node:
    """Build AND(x1, x2, ..., xn) using literals -> {0,1} then AND gates."""
    assert n >= 2
    p = LIT_pos(0)
    for i in range(1, n):
        p = AND_gate(p, LIT_pos(i))
    return p


def balanced_AND_of_literals(n: int) -> Node:
    """Balanced tree AND of literals."""
    assert n >= 2
    nodes = [LIT_pos(i) for i in range(n)]
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                nxt.append(AND_gate(nodes[i], nodes[i + 1]))
            else:
                nxt.append(nodes[i])
        nodes = nxt
    return nodes[0]


def chain_OR_of_literals(n: int) -> Node:
    assert n >= 2
    p = LIT_pos(0)
    for i in range(1, n):
        p = OR_gate(p, LIT_pos(i))
    return p


def parity_XOR_of_literals(n: int) -> Node:
    """Parity via XOR chain on {0,1} literals."""
    assert n >= 2
    p = LIT_pos(0)
    for i in range(1, n):
        p = XOR_gate(p, LIT_pos(i))
    return p


def duplicate_subcircuit_blowup_demo(n: int) -> Node:
    """
    Deliberately creates tree blow-up by duplicating a subcircuit:
      p_1 = LIT_pos(0)
      p_{k+1} = OR(p_k, p_k)  (syntactic duplication)
    This shows why DAG-size matters vs tree-size.
    """
    assert n >= 2
    p = LIT_pos(0)
    for _ in range(1, n):
        p = OR_gate(p, p)  # duplicates p in the AST
    return p


# =============================================================================
# Expansion to monomials (normal form)
# =============================================================================
# We expand into a polynomial in variables x_i where monomials are represented by
# bitmasks over vars (since x_i^2 = 1 for ±1 variables, we can reduce exponents mod 2).
#
# This expansion is what explodes for things like AND_n in coefficient-table form.

Poly = Dict[int, float]  # monomial bitmask -> coefficient


def poly_add(a: Poly, b: Poly) -> Poly:
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, 0.0) + c
        if abs(out[m]) < 1e-12:
            del out[m]
    return out


def poly_scale(k: float, a: Poly) -> Poly:
    if abs(k) < 1e-12:
        return {}
    return {m: k * c for m, c in a.items() if abs(k * c) >= 1e-12}


def poly_mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            # multiply monomials: x^m1 * x^m2 = x^(m1 xor m2) under x_i^2 = 1
            m = m1 ^ m2
            out[m] = out.get(m, 0.0) + c1 * c2
    out = {m: c for m, c in out.items() if abs(c) >= 1e-12}
    return out


def expand_to_poly(n: Node) -> Poly:
    op = n.op
    if op == "const":
        return {0: float(n.args[0])}
    if op == "var":
        i = int(n.args[0])
        return {1 << i: 1.0}
    if op == "scale":
        k = float(n.args[0])
        return poly_scale(k, expand_to_poly(n.args[1]))
    if op == "add":
        return poly_add(expand_to_poly(n.args[0]), expand_to_poly(n.args[1]))
    if op == "mul":
        return poly_mul(expand_to_poly(n.args[0]), expand_to_poly(n.args[1]))
    raise ValueError(f"Unknown op: {op}")


def poly_support(p: Poly) -> int:
    return len(p)


def poly_max_degree(p: Poly) -> int:
    if not p:
        return 0
    return max(int.bit_count(m) for m in p.keys())


# =============================================================================
# Reporting / decision logic
# =============================================================================

@dataclass
class Row:
    n: int
    name: str
    tree_sz: int
    dag_sz: int
    depth: int
    grade_ub: int
    vars: int
    poly_support: Optional[int] = None
    poly_deg: Optional[int] = None


def measure(name: str, n: int, prog: Node, do_expand: bool) -> Row:
    ts = tree_size(prog)
    ds = dag_unique_nodes(prog)
    d = tree_depth(prog)
    g = max_grade_upper_bound(prog)
    v = len(vars_used(prog))
    if do_expand:
        poly = expand_to_poly(prog)
        return Row(n, name, ts, ds, d, g, v, poly_support(poly), poly_max_degree(poly))
    return Row(n, name, ts, ds, d, g, v)


def print_table(rows: List[Row], title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    hdr = (
        f"{'n':>3} | {'prog':<22} | {'tree':>6} | {'dag':>6} | {'depth':>5} | "
        f"{'grade_ub':>8} | {'vars':>4} | {'poly_supp':>9} | {'poly_deg':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ps = "-" if r.poly_support is None else str(r.poly_support)
        pd = "-" if r.poly_deg is None else str(r.poly_deg)
        print(
            f"{r.n:3d} | {r.name:<22} | {r.tree_sz:6d} | {r.dag_sz:6d} | {r.depth:5d} | "
            f"{r.grade_ub:8d} | {r.vars:4d} | {ps:>9} | {pd:>8}"
        )


def simple_growth_fit(ns: List[int], ys: List[int]) -> str:
    """
    Quick-and-dirty growth read:
    - If y ~ c*n or c*n^2 etc, we call it poly-ish.
    - If y doubles consistently, exponential-ish.
    """
    if len(ns) < 4:
        return "insufficient points"
    # ratios between successive ys (avoid 0)
    ratios = []
    for i in range(1, len(ys)):
        if ys[i - 1] > 0:
            ratios.append(ys[i] / ys[i - 1])
    avg_ratio = sum(ratios) / len(ratios)

    # Compare to doubling
    if avg_ratio > 1.6:
        return f"looks exponential-ish (avg ratio ~ {avg_ratio:.2f})"
    # Compare to polynomial using log-log slope
    xs = [math.log(n) for n in ns]
    ls = [math.log(max(1, y)) for y in ys]
    # slope via least squares
    xbar = sum(xs) / len(xs)
    ybar = sum(ls) / len(ls)
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ls))
    den = sum((x - xbar) ** 2 for x in xs)
    slope = num / den if den > 1e-12 else 0.0
    return f"looks poly-ish (log-log slope ~ {slope:.2f})"


def verdict(rows: List[Row], label: str):
    # Focus on DAG size and depth, because those are what a reasonable compiler/IR preserves.
    ns = [r.n for r in rows]
    dag = [r.dag_sz for r in rows]
    depth = [r.depth for r in rows]
    print("\n" + "-" * 90)
    print(f"Growth readout for: {label}")
    print(f"  DAG size:   {simple_growth_fit(ns, dag)}")
    print(f"  Depth:      {simple_growth_fit(ns, depth)}")
    # If expanded poly_support is available, report it too
    if rows[0].poly_support is not None:
        supp = [r.poly_support or 0 for r in rows]
        print(f"  Poly supp:  {simple_growth_fit(ns, supp)}")
    print("-" * 90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_n", type=int, default=40, help="max n for program-structure tests")
    ap.add_argument("--expand_n", type=int, default=12, help="max n for polynomial expansion (normal form)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    # ----------------------------
    # Program structure experiments (the real hinge)
    # ----------------------------
    prog_builders = [
        ("AND_chain_literals", chain_AND_of_literals),
        ("AND_balanced_literals", balanced_AND_of_literals),
        ("OR_chain_literals", chain_OR_of_literals),
        ("XOR_parity_chain", parity_XOR_of_literals),
        ("DUPLICATE_demo_OR(p,p)", duplicate_subcircuit_blowup_demo),
    ]

    structure_rows: List[Row] = []
    for n in range(2, args.max_n + 1):
        for name, builder in prog_builders:
            prog = builder(n)
            structure_rows.append(measure(name, n, prog, do_expand=False))

    print_table(structure_rows, "TRACK B: FACTORED REPRESENTATION (AST/DAG) — gate-by-gate compilation")

    # Verdict by family
    for name, _ in prog_builders:
        family = [r for r in structure_rows if r.name == name]
        verdict(family, name)

    # ----------------------------
    # Expansion experiments (shows where exponential lives)
    # ----------------------------
    expand_rows: List[Row] = []
    for n in range(2, min(args.expand_n, args.max_n) + 1):
        for name, builder in prog_builders:
            # Expansion for the duplication demo gets huge fast; keep it modest
            if name.startswith("DUPLICATE") and n > max(8, args.expand_n // 2):
                continue
            prog = builder(n)
            expand_rows.append(measure(name, n, prog, do_expand=True))

    print_table(expand_rows, "TRACK A: EXPANDED NORMAL FORM (monomial support) — where blow-ups live")

    for name, _ in prog_builders:
        family = [r for r in expand_rows if r.name == name]
        if family:
            verdict(family, name + " (expanded)")

    # ----------------------------
    # Practical interpretation
    # ----------------------------
    print("\n" + "=" * 90)
    print("HOW TO INTERPRET THIS")
    print("=" * 90)
    print(
        "1) If DAG size and depth stay polynomial in n for gate-by-gate compilation,\n"
        "   that supports the *bridge feasibility* (representation can compose like computation).\n\n"
        "2) If expanded poly_support explodes but DAG size stays tame, that's GOOD:\n"
        "   it means the exponential is in *normalization/expansion*, not in the representation.\n\n"
        "3) If DAG size itself explodes under realistic compilation patterns (not just the DUPLICATE demo),\n"
        "   the bridge is in trouble.\n\n"
        "4) The DUPLICATE demo is there to remind you: tree-size can blow up from syntactic duplication,\n"
        "   but DAG-size does not. Real compilers use DAG/IR sharing.\n"
    )

    print("\nNEXT STEP (if DAG stays tame):")
    print(
        "- Add an 'eval' semantics and verify correctness on random inputs for small n.\n"
        "- Then test compilation of *random circuits* (not just n-ary gates).\n"
        "- Then design your real G_k on programs (AST/DAG), not on expanded supports.\n"
    )


if __name__ == "__main__":
    main()
