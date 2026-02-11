#!/usr/bin/env python3
"""
representation_experiment.py

DECISIVE EXPERIMENT (two-track) + TWO NEW HINGES:

Track A (Expanded):
  expand the compiled program into monomials and measure support
  -> tells you what "normal form / coefficient table" does (often exponential).

Track B (Factored):
  measure program DAG size & depth under gate-by-gate compilation
  -> this is the real “does representation compose like computation?” test.

NEW:
  (1) Fan-out reuse tests:
      - shared (compiler-like): p computed once and reused many times
      - duplicated (bad IR): p recomputed each time
  (2) Random circuits generator:
      - controllable sharing rate to stress fan-out behavior
      - reports growth vs circuit size

Key outcome:
  If dag_shared stays ~O(circuit_size) under high fan-out, the bridge is plausible.
  If it explodes, you need explicit sharing / let-binding in the RepLang.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import argparse
import math
import random

# =============================================================================
# Node language (AST) + explicit sharing nodes (Ref/Let)
# =============================================================================

@dataclass(frozen=True)
class Node:
    op: str
    args: Tuple[Any, ...]  # Node | scalar | var-index | ref-index

def Const(c: float) -> Node:
    return Node("const", (float(c),))

def Var(i: int) -> Node:
    return Node("var", (int(i),))

def Add(a: Node, b: Node) -> Node:
    return Node("add", (a, b))

def Mul(a: Node, b: Node) -> Node:
    return Node("mul", (a, b))

def Scale(k: float, a: Node) -> Node:
    return Node("scale", (float(k), a))

# Explicit sharing: a tiny "let/DAG" encoding.
# Program = (defs: list[Node], root: Node) where defs may be referenced by Ref(idx).
def Ref(i: int) -> Node:
    return Node("ref", (int(i),))

@dataclass
class SharedProg:
    defs: List[Node]
    root: Node

def Let(def_node: Node, body_builder, prog: SharedProg) -> SharedProg:
    """
    Add def_node to prog.defs, pass a Ref to body_builder, return new SharedProg.
    body_builder: (ref_node: Node) -> Node
    """
    idx = len(prog.defs)
    prog.defs.append(def_node)
    prog.root = body_builder(Ref(idx))
    return prog

# =============================================================================
# Structural metrics
# =============================================================================

def tree_size(n: Node) -> int:
    if n.op in ("const", "var", "ref"):
        return 1
    if n.op == "scale":
        return 1 + tree_size(n.args[1])
    if n.op in ("add", "mul"):
        return 1 + tree_size(n.args[0]) + tree_size(n.args[1])
    raise ValueError(f"Unknown op: {n.op}")

def tree_depth(n: Node) -> int:
    if n.op in ("const", "var", "ref"):
        return 0
    if n.op == "scale":
        return 1 + tree_depth(n.args[1])
    if n.op in ("add", "mul"):
        return 1 + max(tree_depth(n.args[0]), tree_depth(n.args[1]))
    raise ValueError(f"Unknown op: {n.op}")

def vars_used(n: Node) -> Set[int]:
    out: Set[int] = set()
    def rec(x: Node):
        if x.op == "var":
            out.add(x.args[0]); return
        for a in x.args:
            if isinstance(a, Node):
                rec(a)
    rec(n)
    return out

def dag_unique_nodes_structural(n: Node) -> int:
    """
    Counts unique nodes by structural hashing.
    NOTE: This is NOT "compiler sharing" unless identical subtrees repeat literally.
    """
    seen: Set[Node] = set()
    def rec(x: Node):
        if x in seen: return
        seen.add(x)
        for a in x.args:
            if isinstance(a, Node):
                rec(a)
    rec(n)
    return len(seen)

def dag_size_shared_old(prog: SharedProg) -> int:
    """
    True shared-DAG size: #defs + unique nodes reachable from root after accounting for refs.
    Here we count each def once, and count structural uniques inside defs/root excluding refs.
    """
    # Count defs as nodes (one each) + all structural nodes within defs/root.
    seen: Set[Node] = set()

    def rec(x: Node):
        if x.op == "ref":
            # ref itself counts as a node structurally (optional),
            # but we mainly care about referenced def counted once.
            # We'll still add it to 'seen' for consistency.
            if x in seen: return
            seen.add(x)
            ref_idx = x.args[0]
            rec(prog.defs[ref_idx])
            return
        if x in seen: return
        seen.add(x)
        for a in x.args:
            if isinstance(a, Node):
                rec(a)

    rec(prog.root)
    # Add defs themselves (each stored once)
    return len(seen) + len(prog.defs)

def dag_size_shared(prog: SharedProg) -> int:
    seen_nodes: Set[Node] = set()
    seen_defs: Set[int] = set()

    def rec(x: Node):
        if x in seen_nodes:
            return
        seen_nodes.add(x)

        if x.op == "ref":
            idx = x.args[0]
            if idx in seen_defs:
                return
            seen_defs.add(idx)
            rec(prog.defs[idx])
            return

        for a in x.args:
            if isinstance(a, Node):
                rec(a)

    rec(prog.root)
    return len(seen_nodes) + len(seen_defs)


def max_grade_upper_bound(n: Node) -> int:
    if n.op == "const": return 0
    if n.op == "var": return 1
    if n.op == "ref": return 1  # conservative
    if n.op == "scale": return max_grade_upper_bound(n.args[1])
    if n.op == "add": return max(max_grade_upper_bound(n.args[0]), max_grade_upper_bound(n.args[1]))
    if n.op == "mul": return max_grade_upper_bound(n.args[0]) + max_grade_upper_bound(n.args[1])
    raise ValueError(f"Unknown op: {n.op}")

# =============================================================================
# Gate library (templates)
# Convention: vars are ±1; literals map to {0,1}; internal signals are {0,1}.
# =============================================================================

def AND_gate(p: Node, q: Node) -> Node:
    return Mul(p, q)  # for {0,1}

def OR_gate(p: Node, q: Node) -> Node:
    return Add(Add(p, q), Scale(-1.0, Mul(p, q)))  # p+q-pq

def XOR_gate(p: Node, q: Node) -> Node:
    return Add(Add(p, q), Scale(-2.0, Mul(p, q)))  # p+q-2pq

def NOT_gate(p: Node) -> Node:
    return Add(Const(1.0), Scale(-1.0, p))

def LIT_pos(i: int) -> Node:
    return Scale(0.5, Add(Const(1.0), Var(i)))

def LIT_neg(i: int) -> Node:
    return Scale(0.5, Add(Const(1.0), Scale(-1.0, Var(i))))

# =============================================================================
# Canonical builders (your originals)
# =============================================================================

def chain_AND_of_literals(n: int) -> Node:
    assert n >= 2
    p = LIT_pos(0)
    for i in range(1, n):
        p = AND_gate(p, LIT_pos(i))
    return p

def balanced_AND_of_literals(n: int) -> Node:
    assert n >= 2
    nodes = [LIT_pos(i) for i in range(n)]
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes): nxt.append(AND_gate(nodes[i], nodes[i+1]))
            else: nxt.append(nodes[i])
        nodes = nxt
    return nodes[0]

def chain_OR_of_literals(n: int) -> Node:
    assert n >= 2
    p = LIT_pos(0)
    for i in range(1, n):
        p = OR_gate(p, LIT_pos(i))
    return p

def parity_XOR_of_literals(n: int) -> Node:
    assert n >= 2
    p = LIT_pos(0)
    for i in range(1, n):
        p = XOR_gate(p, LIT_pos(i))
    return p

# =============================================================================
# NEW 1: Fan-out / reuse tests
# =============================================================================

def build_subcircuit_OR_of_literals(n: int) -> Node:
    """p = OR(x1..xn) as a chain (factored)."""
    return chain_OR_of_literals(n)

def fanout_reuse_shared(n: int, k: int) -> SharedProg:
    """
    Compiler-like: compute p once, then build a tree combining p with itself k times.
    Uses explicit Let/Ref sharing.
    """
    assert n >= 2 and k >= 1
    prog = SharedProg(defs=[], root=Const(0.0))

    p = build_subcircuit_OR_of_literals(n)

    # let p = OR(x1..xn) in combine(p,p,...) k times
    def body(p_ref: Node) -> Node:
        # Build a balanced OR-tree with k leaves, all the same reference.
        leaves = [p_ref for _ in range(k)]
        while len(leaves) > 1:
            nxt = []
            for i in range(0, len(leaves), 2):
                if i+1 < len(leaves): nxt.append(OR_gate(leaves[i], leaves[i+1]))
                else: nxt.append(leaves[i])
            leaves = nxt
        return leaves[0]

    prog = Let(p, body, prog)
    return prog

def fanout_reuse_duplicated(n: int, k: int) -> Node:
    """
    Bad IR: recompute p each time (no sharing), then OR them in a balanced tree.
    This should blow up roughly proportional to k * size(p).
    """
    assert n >= 2 and k >= 1
    leaves = [build_subcircuit_OR_of_literals(n) for _ in range(k)]
    while len(leaves) > 1:
        nxt = []
        for i in range(0, len(leaves), 2):
            if i+1 < len(leaves): nxt.append(OR_gate(leaves[i], leaves[i+1]))
            else: nxt.append(leaves[i])
        leaves = nxt
    return leaves[0]

# =============================================================================
# NEW 2: Random circuit generator
# =============================================================================

Gate = str  # "AND"|"OR"|"XOR"|"NOT"

def random_gate(rng: random.Random, p_not: float = 0.15) -> Gate:
    if rng.random() < p_not:
        return "NOT"
    return rng.choice(["AND", "OR", "XOR"])

def apply_gate(g: Gate, a: Node, b: Optional[Node] = None) -> Node:
    if g == "NOT":
        return NOT_gate(a)
    assert b is not None
    if g == "AND": return AND_gate(a, b)
    if g == "OR":  return OR_gate(a, b)
    if g == "XOR": return XOR_gate(a, b)
    raise ValueError(g)

def random_circuit_no_sharing(n_vars: int, size: int, rng: random.Random) -> Node:
    """
    Builds an AST for a random circuit of given 'size' (number of gate applications).
    No explicit sharing beyond structural coincidence.
    """
    assert n_vars >= 2 and size >= 1
    pool: List[Node] = [LIT_pos(i) for i in range(n_vars)]
    for _ in range(size):
        g = random_gate(rng)
        if g == "NOT":
            a = rng.choice(pool)
            pool.append(apply_gate(g, a))
        else:
            a = rng.choice(pool); b = rng.choice(pool)
            pool.append(apply_gate(g, a, b))
    return pool[-1]

def random_circuit_with_sharing(n_vars: int, size: int, share_prob: float, rng: random.Random) -> SharedProg:
    """
    Builds a random circuit in an explicit shared form:
    with probability share_prob, picks a previously defined subcircuit as an input again (fan-out).
    This models compiler IR / DAG reuse.
    """
    assert n_vars >= 2 and size >= 1
    prog = SharedProg(defs=[], root=Const(0.0))

    # available refs (including literals as "defs" for uniformity)
    refs: List[Node] = []
    for i in range(n_vars):
        prog.defs.append(LIT_pos(i))
        refs.append(Ref(len(prog.defs)-1))

    def pick_input() -> Node:
        if rng.random() < share_prob and refs:
            return rng.choice(refs)
        # else make a fresh literal (counts as new def)
        i = rng.randrange(n_vars)
        prog.defs.append(LIT_pos(i))
        r = Ref(len(prog.defs)-1)
        refs.append(r)
        return r

    for _ in range(size):
        g = random_gate(rng)
        if g == "NOT":
            a = pick_input()
            prog.defs.append(apply_gate(g, a))
        else:
            a = pick_input()
            b = pick_input()
            prog.defs.append(apply_gate(g, a, b))
        refs.append(Ref(len(prog.defs)-1))

    prog.root = refs[-1]
    return prog

# =============================================================================
# Expansion to monomials (Track A)
# =============================================================================

Poly = Dict[int, float]

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
    return {m: k*c for m, c in a.items() if abs(k*c) >= 1e-12}

def poly_mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            m = m1 ^ m2  # since x_i^2=1 for ±1 vars
            out[m] = out.get(m, 0.0) + c1*c2
    return {m: c for m, c in out.items() if abs(c) >= 1e-12}

def expand_to_poly(node: Node, defs: Optional[List[Node]] = None) -> Poly:
    """
    Expand Node (and optionally resolve refs against defs) into polynomial in x_i (bitmask).
    Warning: exponential in support for many functions.
    """
    if node.op == "ref":
        assert defs is not None
        return expand_to_poly(defs[node.args[0]], defs)
    if node.op == "const":
        return {0: float(node.args[0])}
    if node.op == "var":
        i = int(node.args[0])
        return {1 << i: 1.0}
    if node.op == "scale":
        k = float(node.args[0])
        return poly_scale(k, expand_to_poly(node.args[1], defs))
    if node.op == "add":
        return poly_add(expand_to_poly(node.args[0], defs), expand_to_poly(node.args[1], defs))
    if node.op == "mul":
        return poly_mul(expand_to_poly(node.args[0], defs), expand_to_poly(node.args[1], defs))
    raise ValueError(f"Unknown op: {node.op}")

def poly_support(p: Poly) -> int:
    return len(p)

def poly_max_degree(p: Poly) -> int:
    return 0 if not p else max(int.bit_count(m) for m in p.keys())

# =============================================================================
# Reporting
# =============================================================================

@dataclass
class Row:
    tag: str           # family label
    n: int             # primary parameter (vars, or fanout leaves, or circuit size)
    tree_sz: int
    dag_struct: int
    dag_shared: int
    depth: int
    grade_ub: int
    vars: int
    poly_supp: Optional[int] = None
    poly_deg: Optional[int] = None

def measure_node(tag: str, n: int, root: Node, do_expand: bool, expand_defs: Optional[List[Node]] = None) -> Row:
    ts = tree_size(root)
    ds = dag_unique_nodes_structural(root)
    d  = tree_depth(root)
    g  = max_grade_upper_bound(root)
    v  = len(vars_used(root))

    # shared metric doesn't apply unless we have SharedProg; approximate: same as structural
    dshared = ds

    ps = pd = None
    if do_expand:
        poly = expand_to_poly(root, expand_defs)
        ps = poly_support(poly)
        pd = poly_max_degree(poly)

    return Row(tag, n, ts, ds, dshared, d, g, v, ps, pd)

def measure_shared(tag: str, n: int, prog: SharedProg, do_expand: bool) -> Row:
    # For shared programs, tree_size/structural dag on the *root with refs* is misleading,
    # so we report:
    #  - tree_sz: tree_size(prog.root) (small)
    #  - dag_struct: structural uniques starting at root resolving refs (rough)
    #  - dag_shared: true shared size via defs+refs traversal
    ts = tree_size(prog.root)
    # compute a structural dag count by traversing with ref resolution:
    seen: Set[Node] = set()
    def rec(x: Node):
        if x in seen: return
        seen.add(x)
        if x.op == "ref":
            rec(prog.defs[x.args[0]])
            return
        for a in x.args:
            if isinstance(a, Node):
                rec(a)
    rec(prog.root)
    ds = len(seen)

    dshared = dag_size_shared(prog)

    # depth with ref expansion:
    def depth_ref(x: Node) -> int:
        if x.op == "ref":
            return depth_ref(prog.defs[x.args[0]])
        return tree_depth(x)
    d = depth_ref(prog.root)

    g = max_grade_upper_bound(prog.root)
    # vars used via ref expansion:
    used: Set[int] = set()
    def vars_rec(x: Node):
        if x.op == "ref":
            vars_rec(prog.defs[x.args[0]]); return
        if x.op == "var":
            used.add(x.args[0]); return
        for a in x.args:
            if isinstance(a, Node):
                vars_rec(a)
    vars_rec(prog.root)
    v = len(used)

    ps = pd = None
    if do_expand:
        poly = expand_to_poly(prog.root, prog.defs)
        ps = poly_support(poly)
        pd = poly_max_degree(poly)

    return Row(tag, n, ts, ds, dshared, d, g, v, ps, pd)

def print_table(rows: List[Row], title: str):
    print("\n" + "="*110)
    print(title)
    print("="*110)
    hdr = (
        f"{'tag':<28} | {'n':>6} | {'tree':>10} | {'dag_struct':>10} | {'dag_shared':>10} | "
        f"{'depth':>6} | {'grade_ub':>8} | {'vars':>4} | {'poly_supp':>9} | {'poly_deg':>8}"
    )
    print(hdr)
    print("-"*len(hdr))
    for r in rows:
        ps = "-" if r.poly_supp is None else str(r.poly_supp)
        pd = "-" if r.poly_deg is None else str(r.poly_deg)
        print(
            f"{r.tag:<28} | {r.n:6d} | {r.tree_sz:10d} | {r.dag_struct:10d} | {r.dag_shared:10d} | "
            f"{r.depth:6d} | {r.grade_ub:8d} | {r.vars:4d} | {ps:>9} | {pd:>8}"
        )

def simple_growth_fit(xs: List[int], ys: List[int]) -> str:
    if len(xs) < 4: return "insufficient points"
    ratios = []
    for i in range(1, len(ys)):
        if ys[i-1] > 0:
            ratios.append(ys[i]/ys[i-1])
    avg_ratio = sum(ratios)/len(ratios)

    if avg_ratio > 1.6:
        return f"exp-ish (avg ratio ~ {avg_ratio:.2f})"

    lx = [math.log(max(2, x)) for x in xs]
    ly = [math.log(max(1, y)) for y in ys]
    xbar = sum(lx)/len(lx)
    ybar = sum(ly)/len(ly)
    num = sum((a-xbar)*(b-ybar) for a, b in zip(lx, ly))
    den = sum((a-xbar)**2 for a in lx)
    slope = num/den if den > 1e-12 else 0.0
    return f"poly-ish (log-log slope ~ {slope:.2f})"

def verdict_family(rows: List[Row], label: str):
    xs = [r.n for r in rows]
    dstruct = [r.dag_struct for r in rows]
    dshared = [r.dag_shared for r in rows]
    depth = [r.depth for r in rows]

    print("\n" + "-"*110)
    print(f"Growth readout: {label}")
    print(f"  dag_struct: {simple_growth_fit(xs, dstruct)}")
    print(f"  dag_shared: {simple_growth_fit(xs, dshared)}   <-- THIS is the fan-out truth")
    print(f"  depth:      {simple_growth_fit(xs, depth)}")
    if rows and rows[0].poly_supp is not None:
        supp = [r.poly_supp or 0 for r in rows]
        print(f"  poly_supp:  {simple_growth_fit(xs, supp)}")
    print("-"*110)

# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--max_n", type=int, default=30, help="max n for canonical families")
    # ap.add_argument("--expand_n", type=int, default=10, help="max n to attempt Track A expansion")
    ap.add_argument("--max_n", type=int, default=20, help="max n for canonical families")
    ap.add_argument("--expand_n", type=int, default=8, help="max n to attempt Track A expansion")
    ap.add_argument("--fanout_n", type=int, default=20, help="n vars inside the reused subcircuit p")
    ap.add_argument("--fanout_k", type=int, default=200, help="max fanout leaves (reuse count)")
    ap.add_argument("--fanout_steps", type=int, default=8, help="number of k points (log-spaced)")
    ap.add_argument("--rand_vars", type=int, default=30, help="random circuit variable count")
    ap.add_argument("--rand_size", type=int, default=200, help="random circuit gate count")
    ap.add_argument("--rand_trials", type=int, default=5, help="random circuit trials")
    # ap.add_argument("--fanout_k", type=int, default=200, help="max fanout leaves (reuse count)")
    # ap.add_argument("--fanout_steps", type=int, default=8, help="number of k points (log-spaced)")
    # ap.add_argument("--rand_vars", type=int, default=50, help="random circuit variable count")
    # ap.add_argument("--rand_size", type=int, default=500, help="random circuit gate count")
    # ap.add_argument("--rand_trials", type=int, default=20, help="random circuit trials")
    ap.add_argument("--rand_share", type=float, default=0.3, help="sharing probability for shared random circuits")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ----------------------------
    # Canonical families (Track B + small Track A)
    # ----------------------------
    canonical = [
        ("AND_chain_literals", chain_AND_of_literals),
        ("AND_balanced_literals", balanced_AND_of_literals),
        ("OR_chain_literals", chain_OR_of_literals),
        ("XOR_parity_chain", parity_XOR_of_literals),
    ]

    rowsB: List[Row] = []
    for n in range(2, args.max_n + 1):
        for name, builder in canonical:
            prog = builder(n)
            rowsB.append(measure_node(name, n, prog, do_expand=False))

    print_table(rowsB, "TRACK B: Canonical families (factored)")

    for name, _ in canonical:
        fam = [r for r in rowsB if r.tag == name]
        verdict_family(fam, name)

    rowsA: List[Row] = []
    for n in range(2, min(args.expand_n, args.max_n) + 1):
        for name, builder in canonical:
            prog = builder(n)
            rowsA.append(measure_node(name, n, prog, do_expand=True))

    print_table(rowsA, "TRACK A: Canonical families (expanded normal form)")
    for name, _ in canonical:
        fam = [r for r in rowsA if r.tag == name]
        verdict_family(fam, name + " (expanded)")

    # ----------------------------
    # NEW: Fan-out reuse stress test
    # ----------------------------
    # log-spaced k values
    ks: List[int] = []
    if args.fanout_steps <= 1:
        ks = [args.fanout_k]
    else:
        for t in range(args.fanout_steps):
            frac = t / (args.fanout_steps - 1)
            k = int(round((args.fanout_k ** frac)))  # from 1 to fanout_k
            ks.append(max(1, k))
        ks = sorted(set(ks))

    fan_rows: List[Row] = []
    for k in ks:
        shared = fanout_reuse_shared(args.fanout_n, k)
        dup = fanout_reuse_duplicated(args.fanout_n, k)
        fan_rows.append(measure_shared("FANOUT_shared(p reused)", k, shared, do_expand=False))
        fan_rows.append(measure_node("FANOUT_dup(recompute p)", k, dup, do_expand=False))

    print_table(
        fan_rows,
        f"FAN-OUT STRESS: p = OR_n (n={args.fanout_n}) reused k times (shared vs duplicated)"
    )

    verdict_family([r for r in fan_rows if r.tag.startswith("FANOUT_shared")], "FANOUT shared")
    verdict_family([r for r in fan_rows if r.tag.startswith("FANOUT_dup")], "FANOUT duplicated")

    # ----------------------------
    # NEW: Random circuits
    # ----------------------------
    # We report averages across trials for:
    # - AST (no explicit sharing)
    # - shared DAG (explicit sharing)
    rand_rows: List[Row] = []
    for trial in range(args.rand_trials):
        ast = random_circuit_no_sharing(args.rand_vars, args.rand_size, rng)
        rand_rows.append(measure_node("RAND_ast(no_share)", trial, ast, do_expand=False))

        shared = random_circuit_with_sharing(args.rand_vars, args.rand_size, args.rand_share, rng)
        rand_rows.append(measure_shared("RAND_shared(share)", trial, shared, do_expand=False))

    print_table(
        rand_rows,
        f"RANDOM CIRCUITS: vars={args.rand_vars}, size={args.rand_size}, trials={args.rand_trials}, share_prob={args.rand_share}"
    )

    # summarize random circuit averages
    def avg(rows: List[Row], field: str) -> float:
        return sum(getattr(r, field) for r in rows) / max(1, len(rows))

    ast_rows = [r for r in rand_rows if r.tag.startswith("RAND_ast")]
    sh_rows = [r for r in rand_rows if r.tag.startswith("RAND_shared")]

    print("\n" + "="*110)
    print("RANDOM CIRCUIT AVERAGES (what to look at)")
    print("="*110)
    print(f"AST (no_share):   avg tree={avg(ast_rows,'tree_sz'):.1f}, avg dag_struct={avg(ast_rows,'dag_struct'):.1f}, avg depth={avg(ast_rows,'depth'):.1f}")
    print(f"SHARED (share):   avg dag_shared={avg(sh_rows,'dag_shared'):.1f}, avg depth={avg(sh_rows,'depth'):.1f}")
    print("\nInterpretation:")
    print("  - If avg dag_shared is ~O(rand_size), great: compilation-like sharing survives fan-out.")
    print("  - If avg dag_shared balloons superlinearly as rand_size grows (try rand_size=1k, 2k), you need a better RepLang/IR.")
    print("="*110)

if __name__ == "__main__":
    main()
