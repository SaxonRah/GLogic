from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple, Any

# ============================================================
# Settings
# ============================================================
DO_RANDOM_SEARCH = True
RANDOM_MODE = "square"      # "square" or "swap"
N = 10
TRIALS = 100
SEED = 1

# During search we keep "any hit" (value/final/peak differs), then rank hits.
MAX_ORS = 10_000            # keep high unless you intentionally want to skip OR-heavy formulas
PEAK_RADIUS = 6             # neighborhood size around peak for printing
CONV_CALLS = 0

# Sanity checks (optional; keep small)
DO_SANITY_CONV_CHECK = False   # FWHT conv == naive conv (n<=6)
DO_SANITY_GP_CHECK = False     # check e2e1 = - e1e2 swap sign

# Constants
ZERO = Fraction(0, 1)
ONE  = Fraction(1, 1)

# ============================================================
# MV n = Mask -> Q  (Mask is int 0..2^n-1)
# ============================================================
MV = Dict[int, Fraction]

def mv_add(u: MV, v: MV) -> MV:
    out = dict(u)
    for k, c in v.items():
        out[k] = out.get(k, Fraction(0)) + c
        if out[k] == 0:
            del out[k]
    return out

def mv_scale(a: Fraction, u: MV) -> MV:
    if a == 0:
        return {}
    return {k: a * c for k, c in u.items() if a * c}

def mv_sub(u: MV, v: MV) -> MV:
    return mv_add(u, mv_scale(Fraction(-1), v))

def mv_l1(u: MV) -> Fraction:
    return sum(abs(c) for c in u.values())

def parity(x: int) -> int:
    return bin(x).count("1") & 1

def mask_bits(mask: int, n: int) -> str:
    return ''.join('1' if (mask >> i) & 1 else '0' for i in range(n))

def mv_pretty(v: MV, n: int) -> str:
    if not v:
        return "0"
    parts = []
    for m in sorted(v.keys()):
        parts.append(f"{v[m]}·m{mask_bits(m,n)}")
    return " + ".join(parts)

# ============================================================
# XOR convolution via FWHT (exact over Fractions)
# ============================================================
def fwht(a: List[Fraction]) -> None:
    """In-place Walsh-Hadamard transform (unnormalized)."""
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

def mv_to_dense(F: MV, n: int) -> List[Fraction]:
    N = 1 << n
    a = [Fraction(0, 1)] * N
    for k, v in F.items():
        a[k] = v
    return a

def dense_to_mv(a: List[Fraction]) -> MV:
    return {i: v for i, v in enumerate(a) if v}

def mv_conv_fwht(F: MV, G: MV, n: int) -> MV:
    """XOR convolution via FWHT: O(n 2^n). Exact over Fractions."""
    A = mv_to_dense(F, n)
    B = mv_to_dense(G, n)
    fwht(A); fwht(B)
    C = [A[i] * B[i] for i in range(1 << n)]
    fwht(C)
    invN = Fraction(1, 1 << n)
    C = [c * invN for c in C]
    return dense_to_mv(C)

def mv_conv_naive(F: MV, G: MV, n: int) -> MV:
    out: MV = {}
    for U in range(1 << n):
        s = Fraction(0, 1)
        for A in range(1 << n):
            s += F.get(A, Fraction(0)) * G.get(A ^ U, Fraction(0))
        if s:
            out[U] = s
    return out

# ============================================================
# Clifford geometric product (gp) on masks
# ============================================================
def clifford_swap_sign(a: int, b: int, n: int) -> int:
    s = 0
    for i in range(n):
        if (a >> i) & 1:
            lower = b & ((1 << i) - 1)
            s += bin(lower).count("1")
    return -1 if (s & 1) else 1

def gp_blade(a: int, b: int, sq: List[Fraction], n: int) -> Tuple[Fraction, int]:
    res = a ^ b
    inter = a & b
    sign = clifford_swap_sign(a, b, n)
    metric = Fraction(1, 1)
    for i in range(n):
        if (inter >> i) & 1:
            metric *= sq[i]
    return Fraction(sign, 1) * metric, res

def make_gp_mul(sq: List[Fraction]):
    n = len(sq)
    def blade_mul(a: int, b: int) -> Tuple[Fraction, int]:
        return gp_blade(a, b, sq, n)
    return blade_mul

def mv_mul(u: MV, v: MV, blade_mul: Callable[[int, int], Tuple[Fraction, int]]) -> MV:
    out: MV = {}
    for a, ca in u.items():
        for b, cb in v.items():
            coeff, res = blade_mul(a, b)
            out[res] = out.get(res, Fraction(0)) + ca * cb * coeff
    return {k: c for k, c in out.items() if c}

# ============================================================
# bQ / chi / embed (Coq-faithful)
# ============================================================
def chi(m: int, a: int, n: int) -> Fraction:
    return Fraction(-1, 1) if parity(m & a) else Fraction(1, 1)

def embed(f: Callable[[int], bool], n: int) -> MV:
    scale = Fraction(1, 1 << n)
    out: MV = {}
    for m in range(1 << n):
        s = Fraction(0, 1)
        for a in range(1 << n):
            if f(a):
                s += scale * chi(m, a, n)
        if s:
            out[m] = s
    return out

def all_boolean_functions(n: int):
    N = 1 << n
    for tt in range(1 << N):
        def f(a: int, tt=tt):
            return bool((tt >> a) & 1)
        yield tt, f

def tt_str(tt: int, n: int) -> str:
    return ''.join('1' if (tt >> a) & 1 else '0' for a in range(1 << n))

def min_l1_to_any_embed(v: MV, n: int) -> Tuple[int, Fraction]:
    best_tt = 0
    best: Optional[Fraction] = None
    for tt, f in all_boolean_functions(n):
        e = embed(f, n)
        dist = mv_l1(mv_sub(v, e))
        if best is None or dist < best:
            best = dist
            best_tt = tt
    return best_tt, best if best is not None else Fraction(0, 1)

# ============================================================
# GA_expr + BoolFormula + translate
# ============================================================
@dataclass(frozen=True)
class GAExpr: ...

@dataclass(frozen=True)
class Scalar(GAExpr):
    q: Fraction

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
        i = phi.i
        return Mul(Scalar(Fraction(1, 2)), Add(Scalar(Fraction(1, 1)), Basis(i)))
    if isinstance(phi, BConst):
        return Scalar(Fraction(1, 1) if phi.b else Fraction(0, 1))
    if isinstance(phi, BAnd):
        p = translate(phi.p, n, and_uses_mul)
        q = translate(phi.q, n, and_uses_mul)
        return Mul(p, q) if and_uses_mul else Conv(p, q)
    if isinstance(phi, BNot):
        p = translate(phi.p, n, and_uses_mul)
        return Add(Scalar(Fraction(1, 1)), Mul(Scalar(Fraction(-1, 1)), p))
    if isinstance(phi, BOr):
        p = translate(phi.p, n, and_uses_mul)
        q = translate(phi.q, n, and_uses_mul)
        return Add(Add(p, q), Mul(Scalar(Fraction(-1, 1)), Conv(p, q)))
    raise TypeError("unknown BoolFormula")

# ============================================================
# eval + trace (memoized VALUES; trace can still repeat by structure)
# ============================================================

def eval_with_trace(e: Any, n: int, sq: List[Any]) -> Tuple[Dict[int, Any], List[Tuple[str, Dict[int, Any]]], int]:
    """
    DAG-trace evaluator (fixes the giant indices / trace-concatenation blowup):

    - memoizes by expression node (structural hash)
    - records each unique subexpression exactly once (postorder)
    - still increments CONV_CALLS exactly once per *computed* Conv node

    Returns: (value, nodes, root_id)
      nodes[i] = (label, MV_value)
    """
    gp_mul = make_gp_mul(sq)

    # expr -> node id
    memo_id: Dict[Any, int] = {}
    # expr -> computed MV value
    memo_val: Dict[Any, Dict[int, Any]] = {}
    # node list (postorder)
    nodes: List[Tuple[str, Dict[int, Any]]] = []

    def add_node(expr: Any, label: str, v: Dict[int, Any]) -> int:
        node_id = len(nodes)
        nodes.append((label, v))
        memo_id[expr] = node_id
        memo_val[expr] = v
        return node_id

    def go(x: Any) -> int:
        global CONV_CALLS
        if x in memo_id:
            return memo_id[x]

        # IMPORTANT: keep these isinstance checks matching your GAExpr classes.
        if isinstance(x, Scalar):
            v = {0: x.q} if x.q else {}
            return add_node(x, f"Scalar({x.q})", v)

        if isinstance(x, Basis):
            v = {1 << x.i: ONE}  # use your existing ONE = Fraction(1,1)
            return add_node(x, f"Basis({x.i})", v)

        if isinstance(x, Add):
            go(x.e1); go(x.e2)
            v = mv_add(memo_val[x.e1], memo_val[x.e2])
            return add_node(x, "Add", v)

        if isinstance(x, Mul):
            go(x.e1); go(x.e2)
            v = mv_mul(memo_val[x.e1], memo_val[x.e2], gp_mul)
            return add_node(x, "Mul(gp)", v)

        if isinstance(x, Conv):
            go(x.e1); go(x.e2)
            CONV_CALLS += 1
            v = mv_conv_fwht(memo_val[x.e1], memo_val[x.e2], n)
            return add_node(x, "Conv(mv_conv)", v)

        raise TypeError(f"unknown GAExpr node: {type(x)}")

    root_id = go(e)
    return memo_val[e], nodes, root_id


def l1_profile(e: Any, n: int, sq: List[Any]) -> List[Tuple[int, str, Any]]:
    """
    Profile of (index, label, l1(value)) over the DAG trace.
    """
    _, nodes, _ = eval_with_trace(e, n, sq)
    return [(i, lab, mv_l1(v)) for i, (lab, v) in enumerate(nodes)]

# ============================================================
# Random formula generator
# ============================================================
def rand_lit(n: int, p_neg: float = 0.5) -> BoolFormula:
    v = BVar(random.randrange(n))
    return BNot(v) if random.random() < p_neg else v

def rand_or_chain(n: int, k: int) -> BoolFormula:
    lits = [rand_lit(n) for _ in range(k)]
    cur = lits[0] if lits else BConst(False)
    for t in lits[1:]:
        cur = BOr(cur, t)
    return cur

def rand_and_chain(n: int, k: int) -> BoolFormula:
    lits = [rand_lit(n) for _ in range(k)]
    random.shuffle(lits)
    cur = lits[0] if lits else BConst(True)
    for t in lits[1:]:
        cur = BAnd(cur, t)
    return cur

def rand_swap_formula(n: int, depth: int) -> BoolFormula:
    phi: BoolFormula = rand_lit(n)
    for _ in range(depth):
        r = random.random()
        if r < 0.45:
            phi = BAnd(phi, rand_and_chain(n, k=random.randrange(2, 6)))
        elif r < 0.75:
            phi = BOr(phi, rand_or_chain(n, k=random.randrange(2, 6)))
        else:
            phi = BNot(phi)
    return phi

def rand_double_square(n: int, k: int) -> BoolFormula:
    # phi = (S&S) & (S&S)
    S = rand_or_chain(n, k)
    Q = BAnd(S, S)
    return BAnd(Q, Q)

def count_nodes(phi: BoolFormula) -> Tuple[int, int, int]:
    """returns (#And, #Or, #Not). Counts with sharing multiplicity (DAG counted as tree)."""
    if isinstance(phi, (BVar, BConst)):
        return (0, 0, 0)
    if isinstance(phi, BAnd):
        a1, o1, n1 = count_nodes(phi.p)
        a2, o2, n2 = count_nodes(phi.q)
        return (a1 + a2 + 1, o1 + o2, n1 + n2)
    if isinstance(phi, BOr):
        a1, o1, n1 = count_nodes(phi.p)
        a2, o2, n2 = count_nodes(phi.q)
        return (a1 + a2, o1 + o2 + 1, n1 + n2)
    if isinstance(phi, BNot):
        a, o, nn = count_nodes(phi.p)
        return (a, o, nn + 1)
    raise TypeError

def show(phi: BoolFormula) -> str:
    if isinstance(phi, BVar): return f"x{phi.i}"
    if isinstance(phi, BConst): return "T" if phi.b else "F"
    if isinstance(phi, BNot): return f"(~{show(phi.p)})"
    if isinstance(phi, BAnd): return f"({show(phi.p)} & {show(phi.q)})"
    if isinstance(phi, BOr):  return f"({show(phi.p)} | {show(phi.q)})"
    return "<?>"


# ============================================================
# Comparison + reporting
# ============================================================
def compare_mul_conv(phi: Any, n: int, sq: List[Any]) -> dict:
    """
    Computes mul-translation vs conv-translation in one pass each,
    stores DAG traces so reporting never re-evaluates (prevents regressions).
    """
    e_mul = translate(phi, n, and_uses_mul=True)
    e_con = translate(phi, n, and_uses_mul=False)

    val_mul, nodes_mul, root_mul = eval_with_trace(e_mul, n, sq)
    val_con, nodes_con, root_con = eval_with_trace(e_con, n, sq)

    peak_mul = max(mv_l1(v) for _, v in nodes_mul) if nodes_mul else ZERO
    peak_con = max(mv_l1(v) for _, v in nodes_con) if nodes_con else ZERO

    final_l1_mul = mv_l1(val_mul)
    final_l1_con = mv_l1(val_con)

    return {
        "val_equal": (val_mul == val_con),
        "peak_mul": peak_mul,
        "peak_conv": peak_con,
        "peak_gap": peak_mul - peak_con,
        "final_l1_mul": final_l1_mul,
        "final_l1_conv": final_l1_con,
        "final_gap": final_l1_mul - final_l1_con,
        "val_mul": val_mul,
        "val_conv": val_con,
        "e_mul": e_mul,
        "e_con": e_con,
        "nodes_mul": nodes_mul,
        "nodes_con": nodes_con,
        "root_mul": root_mul,
        "root_con": root_con,
    }


def print_peak_neighborhood(prof: List[Tuple[int, str, Fraction]], label: str, radius: int) -> None:
    i_peak, lab_peak, v_peak = max(prof, key=lambda t: t[2])
    lo = max(0, i_peak - radius)
    hi = min(len(prof), i_peak + radius + 1)
    print(f"{label} peak at index {i_peak}: {lab_peak}  l1={v_peak}")
    for i in range(lo, hi):
        i0, lab0, v0 = prof[i]
        mark = "<<<" if i == i_peak else "   "
        print(f"  {label}[{i0:5d}] {lab0:14s} l1={v0} {mark}")

def report_hit(t: int, phi: Any, res: dict, n: int, sq: List[Any]) -> None:
    """
    Uses stored traces from compare_mul_conv (no re-eval).
    Output format preserved, but indices now stay small (DAG trace).
    """
    print("trial:", t)
    print("phi:", show(phi))
    print(f"peak_mul={res['peak_mul']}  peak_conv={res['peak_conv']}  peak_gap={res['peak_gap']}")
    print(f"final_l1_mul={res['final_l1_mul']}  final_l1_conv={res['final_l1_conv']}  final_gap={res['final_gap']}")
    print("mul MV:", res["val_mul"])
    print("con MV:", res["val_conv"])
    print()

    nodes_mul: List[Tuple[str, Dict[int, Any]]] = res["nodes_mul"]
    nodes_con: List[Tuple[str, Dict[int, Any]]] = res["nodes_con"]
    root_mul: int = res["root_mul"]
    root_con: int = res["root_con"]

    # Root summary (keeps your “Mul last / Con last” vibe, but now truly the root)
    print("Mul last:", (root_mul, nodes_mul[root_mul][0], mv_l1(nodes_mul[root_mul][1])))
    print("Con last:", (root_con, nodes_con[root_con][0], mv_l1(nodes_con[root_con][1])))
    print()

    prof_mul = [(i, lab, mv_l1(v)) for i, (lab, v) in enumerate(nodes_mul)]
    prof_con = [(i, lab, mv_l1(v)) for i, (lab, v) in enumerate(nodes_con)]

    print_peak_neighborhood(prof_mul, "M", radius=PEAK_RADIUS)
    print()
    print_peak_neighborhood(prof_con, "C", radius=PEAK_RADIUS)

def random_search(n: int, trials: int, mode: str, seed: int, max_ors: int) -> None:
    global CONV_CALLS
    CONV_CALLS = 0

    random.seed(seed)
    sq = [Fraction(1, 1)] * n

    hits = []
    hits_pos = 0
    hits_neg = 0

    for t in range(trials):
        if mode == "square":
            # tweak k-range as desired
            phi = rand_double_square(n, k=random.randrange(6, 14))
        else:
            phi = rand_swap_formula(n, depth=random.randrange(3, 10))

        _, ors, _ = count_nodes(phi)
        if ors > max_ors:
            continue

        res = compare_mul_conv(phi, n, sq)

        # keep ANY interesting hit (don’t miss final-only diffs)
        if (not res["val_equal"]) or (res["final_l1_mul"] != res["final_l1_conv"]) or (res["peak_mul"] != res["peak_conv"]):
            hits.append((t, phi, res))
            if res["peak_gap"] > 0:
                hits_pos += 1
            elif res["peak_gap"] < 0:
                hits_neg += 1

    print(f"n={n}, trials={trials}, mode={mode}: found {len(hits)} hits.  CONV_CALLS={CONV_CALLS}")
    if hits:
        print(f"peak_gap>0: {hits_pos}   peak_gap<0: {hits_neg}   peak_gap=0: {len(hits)-hits_pos-hits_neg}")

    if not hits:
        return

    # Best overall by |peak gap| then |final gap|
    # hits_sorted = sorted(hits, key=lambda x: (abs(x[2]["peak_gap"]), abs(x[2]["final_gap"])), reverse=True)
    # t_best, phi_best, res_best = hits_sorted[0]

    # Best overall by |peak gap| then |final gap|
    # hits_sorted = sorted(hits, key=lambda x: score_result(x[2]), reverse=True)
    # t_best, phi_best, res_best = hits_sorted[0]

    # Best overall by |peak gap| then |final gap|
    t_best, phi_best, res_best = max(hits, key=lambda x: score_result(x[2]))
    print("\n=== BEST (by |peak_gap| then |final_gap|) ===")
    report_hit(t_best, phi_best, res_best, n, sq)

    # Best NEGATIVE peak gap (Mul better than Conv), if any
    neg = [h for h in hits if h[2]["peak_gap"] < 0]
    if neg:
        neg_sorted = sorted(neg, key=lambda x: x[2]["peak_gap"])  # most negative first
        tN, phiN, resN = neg_sorted[0]
        print("\n=== BEST NEGATIVE PEAK GAP (Mul < Conv) ===")
        report_hit(tN, phiN, resN, n, sq)

def score_result(res: dict) -> Tuple[Any, Any]:
    # “BEST (by |peak_gap| then |final_gap|)” – keep exactly as your banner says
    return (abs(res["peak_gap"]), abs(res["final_gap"]))

# ============================================================
# Sanity checks
# ============================================================
def sanity_conv_check():
    n = 6
    random.seed(0)
    F = {i: Fraction(random.randint(-2, 2), 4) for i in range(1 << n) if random.random() < 0.2}
    G = {i: Fraction(random.randint(-2, 2), 4) for i in range(1 << n) if random.random() < 0.2}
    fw = mv_conv_fwht(F, G, n)
    nv = mv_conv_naive(F, G, n)
    assert fw == nv, "FWHT conv != naive conv (bug)"
    print("sanity_conv_check: OK")

def sanity_gp_check():
    n = 5
    sq = [Fraction(1, 1)] * n
    gp = make_gp_mul(sq)
    c_e2e1, _ = gp(1 << 1, 1 << 0)
    c_e1e2, _ = gp(1 << 0, 1 << 1)
    print("gp sign check: e2e1 coeff =", c_e2e1, " ; e1e2 coeff =", c_e1e2)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if DO_SANITY_CONV_CHECK:
        sanity_conv_check()
    if DO_SANITY_GP_CHECK:
        sanity_gp_check()

    if DO_RANDOM_SEARCH:
        random_search(n=N, trials=TRIALS, mode=RANDOM_MODE, seed=SEED, max_ors=MAX_ORS)