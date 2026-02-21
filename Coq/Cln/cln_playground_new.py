# ============================================================
# FIXES ONLY (nothing removed): make Python match Cln Coq 1-to-1
#   - mv_gp sq: add metric_factor (sq vector) + exact swaps_parity sign
#   - Pi / embed / eval exactly as in Cln_Full.v
#   - BoolDist exactly as in Cln_BoolDist.v (exists g : Corner->bool)
#     * we provide: bool_dist_le_check(F,d,g)
#     * and exact/bruteforce minimization only for small n (n<=3 by default)
#   - composite_excursion now reports BOTH:
#       booldist_maskwise_proxy  (your old one, unchanged)
#       booldist_coq_*          (the Coq one)
# ============================================================

from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Any, List, Optional, Iterable, Union
from fractions import Fraction
import math
import random

Coeff = Fraction
Mask = int
Corner = int  # we encode Corner n (Vector.t Sign n) as n-bit int: bit=1 means Neg, bit=0 means Pos.

# ------------------ unchanged helpers ------------------

def popcount(x: int) -> int:
    return x.bit_count()

def format_mask(u: int, n: int, base: str = "bin") -> str:
    if base == "bin":
        return "0b" + format(u, f"0{n}b")
    if base == "hex":
        width = (n + 3) // 4
        return "0x" + format(u, f"0{width}x")
    if base == "dec":
        return str(u)
    raise ValueError("base must be one of: 'bin', 'hex', 'dec'")

@dataclass(frozen=True)
class GA:
    """Sparse MV/GA element: mask(int) -> rational coefficient."""
    n: int
    terms: Dict[Mask, Coeff]

    def __post_init__(self):
        cleaned = {m: c for m, c in self.terms.items() if c != 0}
        object.__setattr__(self, "terms", cleaned)

    @staticmethod
    def zero(n: int) -> "GA":
        return GA(n, {})

    @staticmethod
    def basis(n: int, mask: int, coeff: Coeff = Fraction(1, 1)) -> "GA":
        return GA(n, {mask: coeff})

    def __add__(self, other: "GA") -> "GA":
        assert self.n == other.n
        out = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = out.get(m, Fraction(0, 1)) + c
        return GA(self.n, out)

    def __sub__(self, other: "GA") -> "GA":
        assert self.n == other.n
        out = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = out.get(m, Fraction(0, 1)) - c
        return GA(self.n, out)

    def scale(self, a: Coeff) -> "GA":
        if a == 0:
            return GA.zero(self.n)
        return GA(self.n, {m: a * c for m, c in self.terms.items()})

    def l1(self) -> Coeff:
        # equals Coq l1_norm because absent terms are 0
        return sum((abs(c) for c in self.terms.values()), Fraction(0, 1))

    def support(self) -> List[int]:
        return sorted(self.terms.keys())

    def pretty(self, base: str = "bin", max_terms: int = 80) -> str:
        items = sorted(self.terms.items(), key=lambda kv: kv[0])
        if not items:
            return "0"
        parts = []
        for i, (m, c) in enumerate(items):
            if i >= max_terms:
                parts.append("…")
                break
            if c == 1:
                coef = ""
            elif c == -1:
                coef = "-"
            else:
                coef = f"{c}*"
            parts.append(f"{coef}{format_mask(m, self.n, base)}")
        return " + ".join(parts).replace("+ -", "- ")

# ------------------ all_masks / all_corners (Coq: Vector -> list) ------------------

def all_masks(n: int) -> List[int]:
    return list(range(1 << n))

def all_corners(n: int) -> List[int]:
    # Corner n is Vector.t Sign n; we encode Sign as bit: Pos=0, Neg=1
    return list(range(1 << n))

# ------------------ Coq: sQ / chi / pow2 / Pi / eval / embed ------------------

def pow2(n: int) -> Fraction:
    # Coq pow2 : nat -> Q
    return Fraction(1 << n, 1)

def sQ(sign_bit: int) -> Fraction:
    # sign_bit 0 => Pos => +1, 1 => Neg => -1
    return Fraction(-1, 1) if sign_bit else Fraction(1, 1)

def chi(mask: int, corner: int, n: int) -> Fraction:
    """
    Coq chi' m a = Π_i (if m_i then sQ(a_i) else 1)
    With our encoding, this is (-1)^(popcount(mask & corner)).
    """
    return Fraction(-1, 1) if (popcount(mask & corner) & 1) else Fraction(1, 1)

def Pi_of_corner(n: int, a: Corner) -> GA:
    """
    Coq:
      Pi a : MV n := fun m => (1/pow2 n) * chi m a
    This is dense (all masks get ±1/2^n).
    """
    scale = Fraction(1, 1) / pow2(n)
    out = {m: scale * chi(m, a, n) for m in all_masks(n)}
    return GA(n, out)

def eval_mv(F: GA, s: Corner) -> Fraction:
    """
    Coq:
      eval F s := Σ_m F m * chi m s
    Our GA is sparse but we must sum over all masks; missing terms are 0,
    so summing only support is equivalent.
    """
    n = F.n
    total = Fraction(0, 1)
    for m, c in F.terms.items():
        total += c * chi(m, s, n)
    return total

# ============================================================
# Coq-style corner truth table (eval over Corner n)
# ============================================================

def print_corner_table(
    F: GA,
    *,
    base: str = "bin",
    sort_by_mask: bool = True,
    show_zero: bool = True
) -> None:
    """
    Print the full table:
        s (corner)  |  eval F s

    This mirrors how you'd inspect eval F in Coq proofs
    over all Corner n.
    """

    n = F.n
    corners = all_corners(n)

    if sort_by_mask:
        corners = sorted(corners)

    print(f"\n--- Corner table for F (n={n}) ---")
    print("F =", F.pretty(base))
    print("-" * 40)

    for s in corners:
        val = eval_mv(F, s)
        if not show_zero and val == 0:
            continue
        print(f"s={format_mask(s,n,base)}   eval={val}   ({float(val)})")

    print("-" * 40)

def bQ(b: bool) -> Fraction:
    return Fraction(1, 1) if b else Fraction(0, 1)

def embed_coq(n: int, f: Callable[[Corner], bool]) -> GA:
    """
    Coq:
      embed f m := Σ_a bQ(f a) * Pi a m, over all corners a
    Since Pi a m = (1/2^n) * chi(m,a), this is:
      embed f m = (1/2^n) * Σ_{a: f a = true} chi(m,a)
    We implement it directly (1-to-1 with Coq definition).
    """
    scale = Fraction(1, 1) / pow2(n)
    corners = all_corners(n)

    # Dense build (matches Coq semantics exactly)
    out: Dict[int, Fraction] = {}
    for m in all_masks(n):
        s = Fraction(0, 1)
        for a in corners:
            if f(a):
                s += chi(m, a, n)
        val = scale * s
        if val != 0:
            out[m] = val
    return GA(n, out)

def print_embed_corner_table(
    n: int,
    g: Callable[[Corner], bool],
    *,
    base: str = "bin"
) -> None:
    """
    Prints:
        s | eval (embed g) s   vs   bQ(g s)

    Directly checks Coq lemma:
        eval (embed g) s = bQ (g s)
    """

    E = embed_coq(n, g)

    print(f"\n--- Corner table for embed(g) (n={n}) ---")
    print("-" * 40)

    for s in all_corners(n):
        lhs = eval_mv(E, s)
        rhs = bQ(g(s))
        print(
            f"s={format_mask(s,n,base)}   "
            f"eval(embed g)={lhs}   "
            f"bQ(g s)={rhs}"
        )

    print("-" * 40)

# ------------------ mv_conv (already correct) ------------------

def conv(F: GA, G: GA) -> GA:
    """Coq mv_conv: (F*G)(U) = Σ_A Σ_B [A xor B = U] F(A)G(B)."""
    assert F.n == G.n
    n = F.n
    out: Dict[int, Coeff] = {}
    for a, ca in F.terms.items():
        for b, cb in G.terms.items():
            u = a ^ b
            out[u] = out.get(u, Fraction(0, 1)) + ca * cb
    return GA(n, out)

# ------------------ mv_gp sq: swaps_parity + metric_factor ------------------

def swaps_parity(a: int, b: int, n: int) -> int:
    """
    Coq swaps_parity returns bool; sgnQ maps it to ±1.
    This parity is # {(i in A, j in B) | j < i} mod 2.
    Return 1 if odd, 0 if even (as an int bit).
    """
    parity = 0
    for i in range(n):
        if (a >> i) & 1:
            parity ^= (popcount(b & ((1 << i) - 1)) & 1)
    return parity  # 0/1

def sgnQ(parity_bit: int) -> Fraction:
    # Coq: if parity then -1 else +1
    return Fraction(-1, 1) if parity_bit else Fraction(1, 1)

def metric_factor(sq: List[Fraction], a: int, b: int, n: int) -> Fraction:
    """
    Coq:
      metric_factor sq A B = ∏_{i where A_i && B_i} sq_i
    """
    if len(sq) != n:
        raise ValueError(f"sq length {len(sq)} must equal n={n}")
    both = a & b
    prod = Fraction(1, 1)
    i = 0
    while both:
        if both & 1:
            prod *= sq[i]
        both >>= 1
        i += 1
    # if both becomes 0 before i reaches n, remaining bits are 0 anyway
    return prod

def basis_mul_coeff(sq: List[Fraction], A: int, B: int, n: int) -> Fraction:
    # Coq: sgnQ(swaps_parity A B) * metric_factor sq A B
    return sgnQ(swaps_parity(A, B, n)) * metric_factor(sq, A, B, n)

def mv_gp(n: int, sq: List[Fraction], F: GA, G: GA) -> GA:
    """
    Coq mv_gp:
      (F ⋆ G)(U) = Σ_A Σ_B F(A)G(B) * basis_mul_coeff sq A B * [A xor B = U]
    Sparse implementation via supports (equivalent).
    """
    assert F.n == n and G.n == n
    out: Dict[int, Fraction] = {}
    for A, cA in F.terms.items():
        for B, cB in G.terms.items():
            U = A ^ B
            c = basis_mul_coeff(sq, A, B, n)
            out[U] = out.get(U, Fraction(0, 1)) + cA * cB * c
    return GA(n, out)

def basis(n: int, m: int) -> GA:
    return GA(n, {m: Fraction(1,1)})

def scalar_one(n: int) -> GA:
    return GA(n, {0: Fraction(1,1)})

def gp_basis(n: int, sq: List[Fraction], a: int, b: int) -> GA:
    return mv_gp(n, sq, basis(n,a), basis(n,b))

def check_clifford_laws(n: int, sq: List[Fraction]) -> None:
    one = scalar_one(n)

    # squares
    for i in range(n):
        ei = 1 << i
        lhs = gp_basis(n, sq, ei, ei)
        rhs = one.scale(sq[i])
        assert lhs.terms == rhs.terms, f"e{i}*e{i} mismatch: {lhs.pretty()} vs {rhs.pretty()}"

    # anticommutation
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ei, ej = 1 << i, 1 << j
            lhs = gp_basis(n, sq, ei, ej)
            rhs = gp_basis(n, sq, ej, ei).scale(Fraction(-1,1))
            assert lhs.terms == rhs.terms, f"anticomm mismatch i={i} j={j}: {lhs.pretty()} vs {rhs.pretty()}"

    print("Clifford laws OK (squares + anticommutation).")

def gp(F: GA, G: GA) -> GA:
    """
    Keep your old gp name, but now it matches Coq mv_gp with sq all +1.
    """
    n = F.n
    sq = [Fraction(1, 1)] * n
    return mv_gp(n, sq, F, G)

# ------------------ Coq BoolDist: bool_dist_le ------------------

def mv_sub(F: GA, G: GA) -> GA:
    return F - G

def l1_norm(F: GA) -> Fraction:
    # Coq l1_norm is Σ_m |F m| over all masks.
    # Sparse sum is equivalent (abs(0)=0).
    return F.l1()

def bool_dist_le_check(n: int, F: GA, d: Fraction, g: Callable[[Corner], bool]) -> bool:
    """
    Coq:
      bool_dist_le F d := exists g, l1_norm(F - embed g) <= d
    This checks the inequality for a provided witness g.
    """
    E = embed_coq(n, g)
    dist = l1_norm(mv_sub(F, E))
    return dist <= d

def bool_dist_value_for_g(n: int, F: GA, g: Callable[[Corner], bool]) -> Fraction:
    E = embed_coq(n, g)
    return l1_norm(mv_sub(F, E))

def bool_dist_min_bruteforce(n: int, F: GA, *, max_n: int = 3) -> Tuple[Fraction, int]:
    """
    EXACT minimization of Coq BoolDist by brute force over all g : Corner->bool
    only feasible for small n.

    Returns (best_distance, best_g_bits) where best_g_bits encodes g as a
    bitmask over corners: bit a = 1 iff g(a)=True.
    """
    if n > max_n:
        raise ValueError(f"Coq BoolDist brute force explodes; asked n={n} but max_n={max_n}")

    corners = all_corners(n)
    num_corners = len(corners)  # = 2^n
    best_d: Optional[Fraction] = None
    best_bits: int = 0

    # enumerate all boolean functions on corners: 2^(2^n)
    for bits in range(1 << num_corners):
        def g(a: int, bits=bits) -> bool:
            return ((bits >> a) & 1) == 1

        d = bool_dist_value_for_g(n, F, g)
        if best_d is None or d < best_d:
            best_d = d
            best_bits = bits

    return (best_d if best_d is not None else Fraction(0, 1), best_bits)

def g_from_bits(n: int, bits: int) -> Callable[[Corner], bool]:
    """
    Decode a g witness (Corner->bool) from its truth table bitmask.
    bit a = 1 iff g(a)=True.
    """
    def g(a: int) -> bool:
        return ((bits >> a) & 1) == 1
    return g

# ------------------ KEEP your old maskwise proxy BoolDist (unchanged) ------------------

def booldist_exact_maskwise_proxy(F: GA) -> Tuple[Coeff, Dict[int, bool]]:
    """
    Your old 'exact' (pointwise to {0,1} on masks). Kept as-is.
    NOT the Coq BoolDist.
    """
    n = F.n
    size = 1 << n
    dist = Fraction(0, 1)
    g: Dict[int, bool] = {}
    for m in range(size):
        x = F.terms.get(m, Fraction(0, 1))
        d0 = abs(x)
        d1 = abs(x - 1)
        if d1 <= d0:
            g[m] = True
            dist += d1
        else:
            g[m] = False
            dist += d0
    return dist, g

def embed_g_maskwise_proxy(n: int, g: Dict[int, bool]) -> GA:
    return GA(n, {m: Fraction(1, 1) for m, v in g.items() if v})

def l1_dist_dense(F: GA, G: GA) -> Coeff:
    assert F.n == G.n
    n = F.n
    s = Fraction(0, 1)
    for m in all_masks(n):
        s += abs(F.terms.get(m, Fraction(0, 1)) - G.terms.get(m, Fraction(0, 1)))
    return s

# ------------------ Grade / excursion (extended, nothing removed) ------------------

def grade_of_mask(mask: int) -> int:
    return popcount(mask)

def grade_hist(F: GA) -> Dict[int, Coeff]:
    hist: Dict[int, Coeff] = {}
    for m, c in F.terms.items():
        g = grade_of_mask(m)
        hist[g] = hist.get(g, Fraction(0, 1)) + abs(c)
    return dict(sorted(hist.items()))

def composite_excursion(F: GA, *, coq_booldist_bruteforce_max_n: int = 3) -> Dict[str, Any]:
    """
    Now reports both:
      - booldist_maskwise_proxy: your old proxy (exact for that proxy)
      - booldist_coq_exact      : exact Coq BoolDist only when n <= max_n, else None
    """
    d_proxy, _ = booldist_exact_maskwise_proxy(F)

    d_coq = None
    g_bits = None
    if F.n <= coq_booldist_bruteforce_max_n:
        d_coq, g_bits = bool_dist_min_bruteforce(F.n, F, max_n=coq_booldist_bruteforce_max_n)

    return {
        "n": F.n,
        "support": len(F.terms),
        "l1": F.l1(),
        "grade_hist_abs": grade_hist(F),

        # old (proxy) metric:
        "booldist_maskwise_proxy": d_proxy,

        # Coq metric (exact only for small n):
        "booldist_coq_exact": d_coq,
        "booldist_coq_witness_bits": g_bits,
    }

def print_witness_bits_table(n: int, bits: int, *, base: str = "bin") -> None:
    """
    Pretty-print g : Corner n -> bool given witness_bits encoding:
      bit a = 1 iff g(a)=True.
    """
    print(f"\n--- Witness g table (n={n}) bits={bits} ---")
    for a in all_corners(n):
        v = ((bits >> a) & 1) == 1
        print(f"a={format_mask(a,n,base)}  g(a)={v}")

def print_mask_diff_table(F: GA, E: GA, *, base: str="bin") -> None:
    n = F.n
    print(f"\n--- Mask diff table (n={n}) ---")
    print("mask | F(m) | E(m) | F(m)-E(m) | abs")
    print("-" * 60)
    total = Fraction(0,1)
    for m in all_masks(n):
        fm = F.terms.get(m, Fraction(0,1))
        em = E.terms.get(m, Fraction(0,1))
        dm = fm - em
        am = abs(dm)
        total += am
        print(f"{format_mask(m,n,base)}  {str(fm):>6}  {str(em):>6}  {str(dm):>8}  {str(am):>6}")
    print("-" * 60)
    print("mask_L1 total =", total)

def print_booldist_diagnostics(
    F: GA,
    bits: int,
    *,
    base: str = "bin"
) -> None:
    """
    Diagnostics for Coq BoolDist:

      - Coq distance is mask L1:  Σ_m |(F - embed g)(m)|
      - We ALSO show corner eval differences (not the Coq metric),
        because it's useful for intuition/debugging.

    Prints:
        s | eval(F,s) | g(s) | eval(embed g,s) | abs diff
    and then prints BOTH:
        corner_L1 = Σ_s abs diff
        mask_L1   = Σ_m |(F-embed g)(m)|   (this equals Coq BoolDist value)
    """
    n = F.n
    g = g_from_bits(n, bits)
    E = embed_coq(n, g)

    print(f"\n--- BoolDist diagnostic table (n={n}) ---")
    print("Corner | eval(F,s) | g(s) | eval(embed g,s) | abs diff")
    print("-" * 70)

    corner_total = Fraction(0, 1)
    for s in all_corners(n):
        eval_F = eval_mv(F, s)
        eval_E = eval_mv(E, s)
        diff = abs(eval_F - eval_E)
        corner_total += diff

        print(
            f"{format_mask(s,n,base)}  "
            f"{str(eval_F):>6}     "
            f"{str(g(s)):>5}     "
            f"{str(eval_E):>6}          "
            f"{str(diff):>6}"
        )

    print("-" * 70)
    print("corner_L1 (NOT Coq metric) =", corner_total)

    # This is the actual Coq BoolDist quantity:
    mask_total = l1_norm(F - E)
    print("mask_L1 (Coq booldist)     =", mask_total)
    # show mask-domain breakdown (this is the real Coq metric)
    print_mask_diff_table(F, E, base=base)

def print_excursion(name: str, F: GA, base: str = "bin", *, coq_max_n: int = 3) -> None:
    ex = composite_excursion(F, coq_booldist_bruteforce_max_n=coq_max_n)
    print(f"{name}: {F.pretty(base)}")
    print(f"  support={ex['support']}  l1={float(ex['l1'])} ({ex['l1']})")
    print(f"  booldist_maskwise_proxy={float(ex['booldist_maskwise_proxy'])} ({ex['booldist_maskwise_proxy']})")
    if ex["booldist_coq_exact"] is None:
        print(f"  booldist_coq_exact=None (n>{coq_max_n}; brute-force too big)")
    else:
        d = ex["booldist_coq_exact"]
        bits = ex["booldist_coq_witness_bits"]

        print(f"  booldist_coq_exact={float(d)} ({d})  witness_bits={bits}")

        if bits is not None:
            print_witness_bits_table(F.n, bits, base=base)
            print_booldist_diagnostics(F, bits, base=base)

    print(f"  grade_hist_abs: { {k: float(v) for k, v in ex['grade_hist_abs'].items()} }")

# ============================================================
# WHT block unchanged (keep if you want it; it’s about conv)
# ============================================================

def wht(vals: List[complex]) -> List[complex]:
    a = vals[:]
    h = 1
    n = len(a)
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

def inv_wht(vals: List[complex]) -> List[complex]:
    n = int(math.log2(len(vals)))
    return [z / (1 << n) for z in wht(vals)]

def to_dense(F: GA) -> List[complex]:
    size = 1 << F.n
    arr = [0j] * size
    for m, c in F.terms.items():
        arr[m] = complex(c)
    return arr

def wht_conv_identity_holds(F: GA, G: GA, tol: float = 1e-9) -> bool:
    n = F.n
    f = wht(to_dense(F))
    g = wht(to_dense(G))
    h = wht(to_dense(conv(F, G)))
    for i in range(1 << n):
        if abs(h[i] - f[i] * g[i]) > tol:
            return False
    return True

def wht_involution_check(n: int, trials: int = 3, rng: Optional[random.Random] = None) -> None:
    rng = rng or random.Random()
    size = 1 << n
    for t in range(trials):
        arr = [complex(rng.randint(-3, 3)) for _ in range(size)]
        back = inv_wht(wht(arr))
        err = max(abs(back[i] - arr[i]) for i in range(size))
        print(f"involution check n={n} trial={t}: max_err={err:.3e}")

# ============================================================
# Case runner: only minimal edits so it keeps working
# ============================================================

@dataclass
class Case:
    name: str
    n: int
    F: GA
    G: GA
    # NEW (optional): sq for mv_gp; if None, uses all +1
    sq: Optional[List[Fraction]] = None

@dataclass
class RunConfig:
    base: str = "bin"
    show_elements: bool = True
    show_excursions: bool = True

    check_wht_involution: bool = True
    involution_trials: int = 2
    check_wht_conv_identity: bool = True

    # NEW: show mv_gp sq explicitly
    show_products: bool = True
    show_gp_with_sq: bool = True

    # Coq BoolDist exact brute force cutoff
    coq_booldist_bruteforce_max_n: int = 3

def run_case(case: Case, cfg: RunConfig) -> None:
    n, F, G = case.n, case.F, case.G
    sq = case.sq if case.sq is not None else [Fraction(1, 1)] * n

    print(f"\n================ CASE: {case.name} (n={n}) ================")

    print_corner_table(F, base=cfg.base)
    print_corner_table(G, base=cfg.base)

    if cfg.show_elements:
        print("F =", F.pretty(cfg.base))
        print("G =", G.pretty(cfg.base))

    if cfg.show_excursions:
        print_excursion("F", F, cfg.base, coq_max_n=cfg.coq_booldist_bruteforce_max_n)
        print_excursion("G", G, cfg.base, coq_max_n=cfg.coq_booldist_bruteforce_max_n)

    if cfg.check_wht_conv_identity:
        ok = wht_conv_identity_holds(F, G)
        print("WHT identity holds for conv(F,G)?", ok)

    C = conv(F, G)
    P_default = gp(F, G)             # mv_gp with sq all +1
    P_sq = mv_gp(n, sq, F, G)        # mv_gp with provided sq

    print_corner_table(C, base=cfg.base)
    print_corner_table(P_default, base=cfg.base)

    if cfg.show_products:
        print("conv(F,G) =", C.pretty(cfg.base))
        print("gp(F,G)   =", P_default.pretty(cfg.base))
        if cfg.show_gp_with_sq:
            print("mv_gp sq(F,G) =", P_sq.pretty(cfg.base), "  (sq=" + str(sq) + ")")

    if cfg.show_excursions:
        print_excursion("conv(F,G)", C, cfg.base, coq_max_n=cfg.coq_booldist_bruteforce_max_n)
        print_excursion("gp(F,G)", P_default, cfg.base, coq_max_n=cfg.coq_booldist_bruteforce_max_n)
        if cfg.show_gp_with_sq:
            print_excursion("mv_gp sq(F,G)", P_sq, cfg.base, coq_max_n=cfg.coq_booldist_bruteforce_max_n)

def run_suite(cases: List[Case], cfg: Optional[RunConfig] = None, *, seed: Optional[int] = None) -> None:
    cfg = cfg or RunConfig()
    rng = random.Random(seed)

    print("=== Suite start ===")
    if cfg.check_wht_involution:
        print("\n--- WHT involution sanity ---")
        n0 = cases[0].n if cases else 4
        wht_involution_check(n0, trials=cfg.involution_trials, rng=rng)

    # OPTIONAL: Coq embed correctness quick check (matches Cln_Full.embed_correct)
    print("\n--- Coq embed_correct spot check (random small n) ---")
    n0 = min(4, cases[0].n if cases else 4)
    corners0 = all_corners(n0)
    # random boolean f on corners
    bits = rng.getrandbits(1 << n0)
    f = g_from_bits(n0, bits)  # reuse the same encoding: bit a indicates f(a)=True
    E = embed_coq(n0, f)
    # test on a few corners
    for s in corners0[: min(6, len(corners0))]:
        lhs = eval_mv(E, s)
        rhs = bQ(f(s))
        if lhs != rhs:
            print("embed_correct FAILED at s=", format_mask(s, n0, "bin"), "lhs=", lhs, "rhs=", rhs)
            break
    else:
        print("embed_correct OK on sample (n=", n0, ")")

    for case in cases:
        run_case(case, cfg)

    print("\n=== Suite end ===")

# ============================================================
# Example usage (your existing main still works; nothing lost)
# ============================================================

if __name__ == "__main__":
    n = 4
    A = GA(n, {0b0001: Fraction(1,1), 0b0010: Fraction(1,1), 0b0100: Fraction(1,1)})
    B = GA(n, {0b0001: Fraction(1,1), 0b0011: Fraction(1,1), 0b1111: Fraction(1,1)})

    # Example: nontrivial signature vector sq (Cl(p,q) style)
    # e0^2=+1, e1^2=+1, e2^2=-1, e3^2=-1 for n=4
    sq = [Fraction(1,1), Fraction(1,1), Fraction(-1,1), Fraction(-1,1)]

    check_clifford_laws(n, sq)

    case_fixed = Case("fixed_A_B", n, A, B, sq=sq)

    cfg = RunConfig(
        base="bin",
        show_elements=True,
        show_excursions=True,
        check_wht_involution=True,
        involution_trials=2,
        check_wht_conv_identity=True,
        show_products=True,
        show_gp_with_sq=True,
        coq_booldist_bruteforce_max_n=4,  # exact Coq BoolDist only for n<=3
    )

    run_suite([case_fixed], cfg, seed=12345)