# ============================================================
# File: cln_counterexample_search.py
#
# Uses your existing Coq-faithful playground (cln_playground_new.py)
# to search for “interesting” counterexamples / growth patterns for
# Coq BoolDist under mv_gp, and to compute empirical constants/bounds.
#
# Drop this file next to cln_playground_new.py and run it.
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple, Dict, Any, Callable
import random
import math

# ---- import your core implementation (the one we built up) ----
# Expect this file to sit next to cln_playground_new.py
try:
    import cln_playground_new as cln
except ImportError as e:
    raise SystemExit(
        "Could not import cln_playground_new.py.\n"
        "Make sure cln_counterexample_search.py is in the same folder, "
        "or rename the import near the top.\n"
        f"Original error: {e}"
    )

# ============================================================
# Utilities
# ============================================================

_BOOL_DIST_CACHE: Dict[Tuple[int, Tuple[Tuple[int, Fraction], ...]], Tuple[Fraction, int]] = {}

def _frac(x: Fraction) -> float:
    return float(x)

def _sign_choice(rng: random.Random) -> Fraction:
    return Fraction(1, 1) if rng.getrandbits(1) == 0 else Fraction(-1, 1)

def random_bool_corner_function_bits(n: int, rng: random.Random, p_true: float = 0.5) -> int:
    """
    Encode g : Corner n -> bool as a bitmask over corners:
      bit a = 1 iff g(a)=True.
    """
    bits = 0
    for a in cln.all_corners(n):
        if rng.random() < p_true:
            bits |= (1 << a)
    return bits

def embed_from_bits(n: int, bits: int) -> cln.GA:
    g = cln.g_from_bits(n, bits)
    return cln.embed_coq(n, g)

def random_embed(n: int, rng: random.Random, p_true: float = 0.5) -> Tuple[cln.GA, int]:
    bits = random_bool_corner_function_bits(n, rng, p_true=p_true)
    return embed_from_bits(n, bits), bits

def random_sparse_noise(
    n: int,
    rng: random.Random,
    support: int,
    coeff_kind: str = "pm1",
    num_range: Tuple[int, int] = (-3, 3),
    den_range: Tuple[int, int] = (1, 6),
) -> cln.GA:
    """
    A small sparse perturbation in the MASK domain.
    This is not a Coq object, but it helps probe stability empirically.
    """
    size = 1 << n
    if support <= 0:
        return cln.GA.zero(n)

    masks = rng.sample(range(size), min(support, size))
    terms: Dict[int, Fraction] = {}
    for m in masks:
        if coeff_kind == "pm1":
            c = _sign_choice(rng)
        elif coeff_kind == "int":
            while True:
                k = rng.randint(num_range[0], num_range[1])
                if k != 0:
                    break
            c = Fraction(k, 1)
        elif coeff_kind == "rat":
            while True:
                num = rng.randint(num_range[0], num_range[1])
                den = rng.randint(den_range[0], den_range[1])
                if den != 0 and num != 0:
                    break
            c = Fraction(num, den)
        else:
            raise ValueError("coeff_kind must be one of: pm1, int, rat")
        terms[m] = terms.get(m, Fraction(0, 1)) + c
    return cln.GA(n, terms)

# def coq_booldist_exact(n: int, F: cln.GA, *, brute_max_n: int) -> Tuple[Fraction, int]:
#     """
#     Exact Coq BoolDist and a witness bits, by brute force (small n only).
#     """
#     d, bits = cln.bool_dist_min_bruteforce(n, F, max_n=brute_max_n)
#     return d, bits

def coq_booldist_exact(n: int, F: cln.GA, *, brute_max_n: int) -> Tuple[Fraction, int]:
    key = (n, tuple(sorted(F.terms.items())))
    if key in _BOOL_DIST_CACHE:
        return _BOOL_DIST_CACHE[key]
    d, bits = cln.bool_dist_min_bruteforce(n, F, max_n=brute_max_n)
    _BOOL_DIST_CACHE[key] = (d, bits)
    return d, bits

def normalize_sq(n: int, sq: Optional[List[Fraction]]) -> List[Fraction]:
    if sq is None:
        return [Fraction(1, 1)] * n
    if len(sq) != n:
        raise ValueError(f"sq must have length n={n}")
    return sq

# ============================================================
# “Interesting counterexample” search
# ============================================================

@dataclass
class Interesting:
    n: int
    kind: str
    sq: List[Fraction]
    F: cln.GA
    G: cln.GA
    dF: Fraction
    dG: Fraction
    P: cln.GA
    dP: Fraction
    score: Fraction  # larger is “more interesting”
    meta: Dict[str, Any]

def find_interesting_growth(
    *,
    n: int = 4,
    trials: int = 200,
    seed: int = 0,
    brute_max_n: int = 4,
    sq: Optional[List[Fraction]] = None,
    # generators:
    mode: str = "embed+noise",   # "embed+noise" | "embed-only" | "sparse-mask"
    p_true: float = 0.5,
    noise_support: int = 2,
    noise_scale: Fraction = Fraction(1, 2),
    sparse_support: int = 6,
    sparse_coeff_kind: str = "pm1",
    # what counts as “interesting”:
    objective: str = "dP_minus_sum",  # "dP_minus_sum" | "ratio_over_sum" | "ratio_over_max"
    top_k: int = 10,
) -> List[Interesting]:
    """
    Searches random pairs (F,G) and ranks by a chosen objective measuring
    growth of Coq BoolDist under mv_gp.

    objective:
      - dP_minus_sum:     score = dP - (dF + dG)
      - ratio_over_sum:   score = dP / max(1, dF + dG)
      - ratio_over_max:   score = dP / max(1, max(dF,dG))
    """
    rng = random.Random(seed)
    sqN = normalize_sq(n, sq)
    hits: List[Interesting] = []

    def gen_pair() -> Tuple[cln.GA, cln.GA, Dict[str, Any]]:
        if mode == "embed-only":
            F, bitsF = random_embed(n, rng, p_true=p_true)
            G, bitsG = random_embed(n, rng, p_true=p_true)
            return F, G, {"bitsF": bitsF, "bitsG": bitsG}
        if mode == "embed+noise":
            F0, bitsF = random_embed(n, rng, p_true=p_true)
            G0, bitsG = random_embed(n, rng, p_true=p_true)
            NF = random_sparse_noise(n, rng, support=noise_support, coeff_kind="pm1").scale(noise_scale)
            NG = random_sparse_noise(n, rng, support=noise_support, coeff_kind="pm1").scale(noise_scale)
            return (F0 + NF), (G0 + NG), {"bitsF": bitsF, "bitsG": bitsG, "NF": NF, "NG": NG}
        if mode == "sparse-mask":
            # purely random sparse in mask domain (not Coq-generated, but good for stress tests)
            F = cln.random_sparse_ga(n, sparse_support, rng=rng, coeffs=sparse_coeff_kind)
            G = cln.random_sparse_ga(n, sparse_support, rng=rng, coeffs=sparse_coeff_kind)
            return F, G, {}
        raise ValueError("mode must be one of: embed-only, embed+noise, sparse-mask")

    for t in range(trials):
        F, G, meta = gen_pair()

        # exact Coq BoolDist (small n)
        dF, _ = coq_booldist_exact(n, F, brute_max_n=brute_max_n)
        dG, _ = coq_booldist_exact(n, G, brute_max_n=brute_max_n)

        P = cln.mv_gp(n, sqN, F, G)
        dP, _ = coq_booldist_exact(n, P, brute_max_n=brute_max_n)

        if objective == "dP_minus_sum":
            score = dP - (dF + dG)
        elif objective == "ratio_over_sum":
            denom = dF + dG
            score = dP / (denom if denom != 0 else Fraction(1, 1))
        elif objective == "ratio_over_max":
            denom = max(dF, dG)
            score = dP / (denom if denom != 0 else Fraction(1, 1))
        else:
            raise ValueError("objective must be one of: dP_minus_sum, ratio_over_sum, ratio_over_max")

        hits.append(
            Interesting(
                n=n,
                kind=mode,
                sq=sqN,
                F=F,
                G=G,
                dF=dF,
                dG=dG,
                P=P,
                dP=dP,
                score=score,
                meta={"trial": t, **meta},
            )
        )

    hits.sort(key=lambda h: (h.score, h.dP), reverse=True)
    return hits[:top_k]

def print_interesting(hit: Interesting, *, base: str = "bin", show_tables: bool = False) -> None:
    print("\n============================================================")
    print(f"INTERESTING ({hit.kind}) n={hit.n} score={hit.score}  dF={hit.dF} dG={hit.dG} dP={hit.dP}")
    print("sq =", hit.sq)
    print("F =", hit.F.pretty(base))
    print("G =", hit.G.pretty(base))
    print("P = mv_gp(F,G) =", hit.P.pretty(base))
    print("meta:", hit.meta)

    if show_tables:
        # show witness + mask diff table for P only (it’s usually the focus)
        dP, bitsP = coq_booldist_exact(hit.n, hit.P, brute_max_n=hit.n)
        print("\n--- P BoolDist witness ---")
        print("dP =", dP, "bitsP =", bitsP)
        cln.print_witness_bits_table(hit.n, bitsP, base=base)
        cln.print_booldist_diagnostics(hit.P, bitsP, base=base)

# ============================================================
# Empirical bound estimation
# ============================================================

@dataclass
class BoundReport:
    n: int
    sq: List[Fraction]
    trials: int
    seed: int
    brute_max_n: int

    # candidate inequality (empirical):
    # dP <= C0 + C1*(dF+dG) + C2*(l1F*dG + l1G*dF) + C3*(l1F*l1G)
    C0: Fraction
    C1: Fraction
    C2: Fraction
    C3: Fraction

    # worst case seen
    worst_value: Fraction
    worst_ratio: float
    worst_example: Dict[str, Any]

def fit_linear_upper_bound(
    samples: List[Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]],
) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
    """
    Very simple “fit” for an upper bound of the form:
      dP <= C0 + C1*S + C2*T + C3*U
    where each sample provides:
      (dP, S, T, U, 1)

    We do a conservative heuristic:
      - start with C0=C1=C2=C3=0
      - for each sample, compute required slack; if violated, increase one coefficient
        in a greedy way (prefer increasing C2, then C1, then C3, then C0).
    This is not an optimizer; it’s a quick way to get a usable empirical bound.
    """
    C0 = Fraction(0, 1)
    C1 = Fraction(0, 1)
    C2 = Fraction(0, 1)
    C3 = Fraction(0, 1)

    for (dP, S, T, U, one) in samples:
        rhs = C0 + C1*S + C2*T + C3*U
        if dP <= rhs:
            continue
        gap = dP - rhs

        # Greedy: try to assign gap to the largest-feature term available
        # (this tends to stabilize across scales).
        candidates = []
        if T != 0: candidates.append(("C2", T))
        if S != 0: candidates.append(("C1", S))
        if U != 0: candidates.append(("C3", U))
        candidates.append(("C0", Fraction(1, 1)))

        # pick the max feature
        name, feat = max(candidates, key=lambda kv: kv[1])

        inc = gap / feat  # exact Fraction
        if name == "C2":
            C2 += inc
        elif name == "C1":
            C1 += inc
        elif name == "C3":
            C3 += inc
        else:
            C0 += inc

    return C0, C1, C2, C3

def empirical_bounds_mv_gp(
    *,
    n: int = 4,
    trials: int = 200,
    seed: int = 0,
    brute_max_n: int = 4,
    sq: Optional[List[Fraction]] = None,
    mode: str = "embed+noise",
    p_true: float = 0.5,
    noise_support: int = 2,
    noise_scale: Fraction = Fraction(1, 2),
) -> BoundReport:
    """
    Computes a *usable* empirical inequality of the form:

      dP <= C0 + C1*(dF+dG) + C2*(l1F*dG + l1G*dF) + C3*(l1F*l1G)

    where dX is Coq BoolDist exact (small n), and P = mv_gp(F,G).

    It also reports the worst observed ratio:
      dP / RHS
    """
    rng = random.Random(seed)
    sqN = normalize_sq(n, sq)

    samples_for_fit: List[Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]] = []
    raw_samples: List[Dict[str, Any]] = []

    for t in range(trials):
        # generate F,G
        if mode == "embed-only":
            F0, _ = random_embed(n, rng, p_true=p_true)
            G0, _ = random_embed(n, rng, p_true=p_true)
            F, G = F0, G0
        elif mode == "embed+noise":
            F0, _ = random_embed(n, rng, p_true=p_true)
            G0, _ = random_embed(n, rng, p_true=p_true)
            NF = random_sparse_noise(n, rng, support=noise_support, coeff_kind="pm1").scale(noise_scale)
            NG = random_sparse_noise(n, rng, support=noise_support, coeff_kind="pm1").scale(noise_scale)
            F, G = (F0 + NF), (G0 + NG)
        else:
            raise ValueError("mode must be embed-only or embed+noise for this bound routine")

        dF, _ = coq_booldist_exact(n, F, brute_max_n=brute_max_n)
        dG, _ = coq_booldist_exact(n, G, brute_max_n=brute_max_n)

        l1F = cln.l1_norm(F)
        l1G = cln.l1_norm(G)

        P = cln.mv_gp(n, sqN, F, G)
        dP, _ = coq_booldist_exact(n, P, brute_max_n=brute_max_n)

        S = dF + dG
        T = l1F*dG + l1G*dF
        U = l1F*l1G
        samples_for_fit.append((dP, S, T, U, Fraction(1, 1)))

        raw_samples.append({
            "trial": t,
            "F": F,
            "G": G,
            "P": P,
            "dF": dF, "dG": dG, "dP": dP,
            "l1F": l1F, "l1G": l1G,
            "S": S, "T": T, "U": U,
        })

    C0, C1, C2, C3 = fit_linear_upper_bound(samples_for_fit)

    # Find worst ratio dP / RHS
    worst_ratio = -1.0
    worst_value = Fraction(0, 1)
    worst_example: Dict[str, Any] = {}

    for samp in raw_samples:
        rhs = C0 + C1*samp["S"] + C2*samp["T"] + C3*samp["U"]
        if rhs == 0:
            continue
        ratio = float(samp["dP"] / rhs)
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_value = samp["dP"] - rhs
            worst_example = {
                "trial": samp["trial"],
                "dF": samp["dF"], "dG": samp["dG"], "dP": samp["dP"],
                "l1F": samp["l1F"], "l1G": samp["l1G"],
                "rhs": rhs,
                "F": samp["F"], "G": samp["G"], "P": samp["P"],
            }

    return BoundReport(
        n=n, sq=sqN, trials=trials, seed=seed, brute_max_n=brute_max_n,
        C0=C0, C1=C1, C2=C2, C3=C3,
        worst_value=worst_value,
        worst_ratio=worst_ratio,
        worst_example=worst_example
    )

def print_bound_report(rep: BoundReport, *, base: str = "bin") -> None:
    print("\n============================================================")
    print(f"EMPIRICAL BOUND REPORT n={rep.n} trials={rep.trials} seed={rep.seed} brute_max_n={rep.brute_max_n}")
    print("sq =", rep.sq)
    print("Proposed empirical inequality:")
    print("  dP <= C0 + C1*(dF+dG) + C2*(l1F*dG + l1G*dF) + C3*(l1F*l1G)")
    print("  C0 =", rep.C0)
    print("  C1 =", rep.C1)
    print("  C2 =", rep.C2)
    print("  C3 =", rep.C3)
    print("Worst observed ratio dP/RHS =", rep.worst_ratio)
    if rep.worst_example:
        ex = rep.worst_example
        print("\nWorst example:")
        print("  trial =", ex["trial"])
        print("  dF,dG,dP =", ex["dF"], ex["dG"], ex["dP"])
        print("  l1F,l1G  =", ex["l1F"], ex["l1G"])
        print("  RHS      =", ex["rhs"])
        print("  F =", ex["F"].pretty(base))
        print("  G =", ex["G"].pretty(base))
        print("  P =", ex["P"].pretty(base))

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Choose n=4 if you want exact Coq BoolDist by brute force (2^(2^4)=65536 functions).
    # n=5 is too big for exact brute force; keep brute_max_n=4.
    n = 4
    brute_max_n = 4

    # Example signature (same as your earlier demo): + + - -
    sq = [Fraction(1, 1), Fraction(1, 1), Fraction(-1, 1), Fraction(-1, 1)]

    print("=== Searching for interesting BoolDist growth under mv_gp ===")
    hits = find_interesting_growth(
        n=n,
        trials=200,
        # trials=10,
        seed=12345,
        brute_max_n=brute_max_n,
        sq=sq,
        mode="embed+noise",        # try "embed-only" for pure boolean-corner embeds
        p_true=0.5,
        noise_support=2,
        noise_scale=Fraction(1, 2),
        objective="dP_minus_sum",  # try also: ratio_over_sum, ratio_over_max
        top_k=5,
    )
    for h in hits:
        print_interesting(h, base="bin", show_tables=False)

    print("\n=== Empirical bound fitting (heuristic upper bound) ===")
    rep = empirical_bounds_mv_gp(
        n=n,
        trials=200,
        # trials=10,
        seed=999,
        brute_max_n=brute_max_n,
        sq=sq,
        mode="embed+noise",
        p_true=0.5,
        noise_support=2,
        noise_scale=Fraction(1, 2),
    )
    print_bound_report(rep, base="bin")

"""
C:\Users\Jupiter\AppData\Local\Programs\Python\Python311\python.exe E:\GLogic\Coq\Cln\cln_counterexample_search.py 
=== Searching for interesting BoolDist growth under mv_gp ===

============================================================
INTERESTING (embed+noise) n=4 score=15/8  dF=1 dG=1 dP=31/8
sq = [Fraction(1, 1), Fraction(1, 1), Fraction(-1, 1), Fraction(-1, 1)]
F = 15/16*0b0000 + 3/16*0b0001 - 1/16*0b0010 - 1/16*0b0011 - 1/16*0b0100 + 3/16*0b0101 - 1/16*0b0110 - 1/16*0b0111 - 1/16*0b1000 + 3/16*0b1001 - 9/16*0b1010 - 1/16*0b1011 - 1/16*0b1100 - 5/16*0b1101 - 1/16*0b1110 - 1/16*0b1111
G = 19/16*0b0000 + 3/16*0b0001 + 1/16*0b0010 + 1/16*0b0011 + 1/16*0b0100 - 3/16*0b0101 + 3/16*0b0110 - 1/16*0b0111 + 3/16*0b1000 - 1/16*0b1001 - 11/16*0b1010 + 1/16*0b1011 + 1/16*0b1100 + 1/16*0b1101 - 1/16*0b1110 - 1/16*0b1111
P = mv_gp(F,G) = 97/64*0b0000 + 25/64*0b0001 + 7/64*0b0010 - 9/64*0b0011 - 13/64*0b0100 + 3/64*0b0101 + 1/64*0b0110 - 23/64*0b0111 + 13/64*0b1000 + 17/64*0b1001 - 83/64*0b1010 - 23/64*0b1011 - 15/64*0b1100 - 19/64*0b1101 - 11/64*0b1110 - 7/64*0b1111
meta: {'trial': 153, 'bitsF': 42324, 'bitsG': 21471, 'NF': GA(n=4, terms={0: Fraction(1, 2), 10: Fraction(-1, 2)}), 'NG': GA(n=4, terms={10: Fraction(-1, 2), 0: Fraction(1, 2)})}

============================================================
INTERESTING (embed+noise) n=4 score=3/2  dF=1 dG=1 dP=7/2
sq = [Fraction(1, 1), Fraction(1, 1), Fraction(-1, 1), Fraction(-1, 1)]
F = 9/16*0b0000 + 1/16*0b0001 + 1/16*0b0010 + 1/16*0b0011 + 3/16*0b0100 - 1/16*0b0101 + 3/16*0b0110 - 9/16*0b0111 - 1/16*0b1000 - 1/16*0b1001 - 1/16*0b1010 - 1/16*0b1011 + 5/16*0b1100 + 1/16*0b1101 - 11/16*0b1110 + 1/16*0b1111
G = 15/16*0b0000 - 1/16*0b0001 - 1/16*0b0010 - 1/16*0b0011 - 1/16*0b0100 - 1/16*0b0101 + 3/16*0b0110 + 3/16*0b0111 + 3/16*0b1000 + 7/16*0b1001 - 1/16*0b1010 + 3/16*0b1011 - 1/16*0b1100 + 3/16*0b1101 - 1/16*0b1110 + 3/16*0b1111
P = mv_gp(F,G) = 27/64*0b0000 - 17/64*0b0001 + 5/64*0b0010 - 11/64*0b0011 + 1/64*0b0100 - 23/64*0b0101 + 31/64*0b0110 - 53/64*0b0111 - 15/64*0b1000 + 25/64*0b1001 + 9/64*0b1010 - 3/64*0b1011 + 29/64*0b1100 - 7/64*0b1101 - 59/64*0b1110 + 5/64*0b1111
meta: {'trial': 151, 'bitsF': 54031, 'bitsG': 17129, 'NF': GA(n=4, terms={7: Fraction(-1, 2), 14: Fraction(-1, 2)}), 'NG': GA(n=4, terms={0: Fraction(1, 2), 9: Fraction(1, 2)})}

============================================================
INTERESTING (embed+noise) n=4 score=45/32  dF=1 dG=1 dP=109/32
sq = [Fraction(1, 1), Fraction(1, 1), Fraction(-1, 1), Fraction(-1, 1)]
F = 9/16*0b0000 + 1/16*0b0001 + 1/16*0b0010 + 1/16*0b0011 + 3/16*0b0100 + 3/16*0b0101 - 1/16*0b0110 - 1/16*0b0111 - 1/16*0b1000 + 3/16*0b1001 + 3/16*0b1010 - 9/16*0b1011 - 11/16*0b1100 + 1/16*0b1101 - 3/16*0b1110 + 1/16*0b1111
G = 11/16*0b0000 + 3/16*0b0001 + 3/16*0b0010 - 1/16*0b0011 + 1/16*0b0100 + 1/16*0b0101 - 3/16*0b0110 + 9/16*0b0111 - 1/16*0b1000 + 3/16*0b1001 - 1/16*0b1010 - 9/16*0b1011 - 3/16*0b1100 + 1/16*0b1101 + 1/16*0b1110 + 1/16*0b1111
P = mv_gp(F,G) = 39/64*0b0000 + 7/64*0b0001 + 15/64*0b0010 - 13/64*0b0011 + 1/64*0b0100 + 21/64*0b0101 - 15/64*0b0110 - 7/64*0b0111 - 9/64*0b1000 + 31/64*0b1001 + 5/64*0b1010 - 71/64*0b1011 - 65/64*0b1100 + 3/64*0b1101 - 27/64*0b1110 + 5/64*0b1111
meta: {'trial': 125, 'bitsF': 36661, 'bitsG': 16245, 'NF': GA(n=4, terms={12: Fraction(-1, 2), 11: Fraction(-1, 2)}), 'NG': GA(n=4, terms={11: Fraction(-1, 2), 7: Fraction(1, 2)})}

============================================================
INTERESTING (embed+noise) n=4 score=5/4  dF=1 dG=1 dP=13/4
sq = [Fraction(1, 1), Fraction(1, 1), Fraction(-1, 1), Fraction(-1, 1)]
F = 11/16*0b0000 + 1/16*0b0001 - 9/16*0b0010 + 1/16*0b0011 - 1/16*0b0100 + 1/16*0b0101 + 3/16*0b0110 + 1/16*0b0111 - 9/16*0b1000 - 3/16*0b1001 - 1/16*0b1010 + 1/16*0b1011 - 1/16*0b1100 + 1/16*0b1101 - 1/16*0b1110 + 5/16*0b1111
G = 11/16*0b0000 + 3/16*0b0001 + 1/16*0b0010 + 1/16*0b0011 - 7/16*0b0100 - 3/16*0b0101 + 3/16*0b0110 - 1/16*0b0111 - 9/16*0b1000 - 1/16*0b1001 + 1/16*0b1010 + 1/16*0b1011 + 1/16*0b1100 - 3/16*0b1101 - 1/16*0b1110 + 3/16*0b1111
P = mv_gp(F,G) = 13/64*0b0000 + 11/64*0b0001 - 25/64*0b0010 + 21/64*0b0011 - 37/64*0b0100 - 15/64*0b0101 + 25/64*0b0110 - 1/64*0b0111 - 49/64*0b1000 + 1/64*0b1001 + 27/64*0b1010 - 7/64*0b1011 - 13/64*0b1100 + 1/64*0b1101 - 25/64*0b1110 + 9/64*0b1111
meta: {'trial': 11, 'bitsF': 55273, 'bitsG': 55131, 'NF': GA(n=4, terms={8: Fraction(-1, 2), 2: Fraction(-1, 2)}), 'NG': GA(n=4, terms={4: Fraction(-1, 2), 8: Fraction(-1, 2)})}

============================================================
INTERESTING (embed+noise) n=4 score=5/4  dF=1 dG=1 dP=13/4
sq = [Fraction(1, 1), Fraction(1, 1), Fraction(-1, 1), Fraction(-1, 1)]
F = 9/8*0b0000 + 1/8*0b0001 + 1/8*0b0010 - 1/8*0b0011 + 1/8*0b0100 - 1/8*0b0101 - 1/8*0b0110 - 5/8*0b0111 + 1/8*0b1000 + 1/8*0b1001 - 1/8*0b1010 + 1/8*0b1011 + 1/8*0b1100 - 1/8*0b1101 + 1/8*0b1110 + 1/8*0b1111
G = 1/2*0b0000 - 1/8*0b0001 + 1/4*0b0010 + 1/8*0b0011 - 1/8*0b0100 - 1/8*0b0110 - 1/2*0b0111 - 1/8*0b1000 - 1/8*0b1010 - 3/4*0b1100 + 1/8*0b1101 - 1/8*0b1111
P = mv_gp(F,G) = 17/16*0b0000 - 1/8*0b0001 + 5/16*0b0010 + 1/4*0b0011 - 1/8*0b0100 + 1/16*0b0101 - 1/8*0b0110 - 15/16*0b0111 + 1/8*0b1000 + 1/16*0b1001 - 1/4*0b1010 - 7/16*0b1011 - 13/16*0b1100 + 1/16*0b1110 + 1/8*0b1111
meta: {'trial': 145, 'bitsF': 13919, 'bitsG': 15280, 'NF': GA(n=4, terms={0: Fraction(1, 2), 7: Fraction(-1, 2)}), 'NG': GA(n=4, terms={12: Fraction(-1, 2), 7: Fraction(-1, 2)})}

=== Empirical bound fitting (heuristic upper bound) ===

============================================================
EMPIRICAL BOUND REPORT n=4 trials=200 seed=999 brute_max_n=4
sq = [Fraction(1, 1), Fraction(1, 1), Fraction(-1, 1), Fraction(-1, 1)]
Proposed empirical inequality:
  dP <= C0 + C1*(dF+dG) + C2*(l1F*dG + l1G*dF) + C3*(l1F*l1G)
  C0 = 0
  C1 = 0
  C2 = 0
  C3 = 21/44
Worst observed ratio dP/RHS = 1.0

Worst example:
  trial = 159
  dF,dG,dP = 1 1 21/8
  l1F,l1G  = 2 11/4
  RHS      = 21/8
  F = 1/4*0b0000 - 1/4*0b0011 + 1/4*0b0101 - 1/4*0b0110 - 1/2*0b1010 + 1/2*0b1101
  G = 7/16*0b0000 + 5/16*0b0001 - 1/16*0b0010 - 3/16*0b0011 - 3/16*0b0100 - 1/16*0b0101 - 3/16*0b0110 - 1/16*0b0111 - 1/16*0b1000 - 7/16*0b1001 - 1/16*0b1010 + 1/16*0b1011 + 1/16*0b1100 - 1/16*0b1101 + 1/16*0b1110 + 7/16*0b1111
  P = 5/32*0b0000 + 3/32*0b0001 - 7/32*0b0010 - 15/32*0b0011 + 3/32*0b0100 + 11/32*0b0101 - 9/32*0b0110 + 1/32*0b0111 - 3/32*0b1000 - 13/32*0b1001 - 15/32*0b1010 - 7/32*0b1011 + 15/32*0b1100 + 7/32*0b1101 - 5/32*0b1110 + 5/32*0b1111

Process finished with exit code 0

---

BoolDist ≤ l1
    Lemma booldist_le_l1 :
      forall n (F : MV n),
        bool_dist F <= l1_norm F.

Projector idempotence for square(+1) bivectors
(phrase “bivector squares to 1” in your basis_mul_coeff language.)
    Lemma projector_idempotent :
      forall n sq (B : MV n),
        (* hypotheses encoding: B is a blade and B ⋆ B = 1 *)
        mv_gp sq B B = one ->
        let p := (1/2) • (one - B) in
        mv_gp sq p p = p.

Multiply-by-projector identities
    Lemma gp_mul_projector_right :
      forall n sq (X B : MV n),
        mv_gp sq B B = one ->
        let p := (1/2) • (one - B) in
        mv_gp sq X p = (1/2) • (X - mv_gp sq X B).
    Lemma gp_mul_projector_left :
      forall n sq (X B : MV n),
        mv_gp sq B B = one ->
        let p := (1/2) • (one - B) in
        mv_gp sq p X = (1/2) • (X - mv_gp sq B X).

Commutator extraction
    Lemma gp_projector_commutator :
      forall n sq (X B : MV n),
        mv_gp sq B B = one ->
        let p := (1/2) • (one - B) in
        mv_gp sq X p - mv_gp sq p X = (1/2) • (mv_gp sq B X - mv_gp sq X B).

Excursion consequence
Use excursion definitions, but the key bridge is:
    Lemma booldist_lower_bound_implies_excursion_l1 :
      forall tr d,
        (* if at some step BoolDist >= d0 *)
        (* then exc_l1 >= d0 because BoolDist <= l1 pointwise *)


---

Lemma B1: triangle upper bound (fixed witness)
Math: ∥𝐹−𝐸∥_1 ≤ ∥𝐹∥_1 + ∥𝐸∥_1

    Lemma bool_dist_wrt_le_l1_plus :
      forall n (F : MV n) (g : Corner n -> bool),
        bool_dist_wrt F g <= l1_norm F + l1_norm (embed g).
    Proof.
      (* unfolds + use l1 triangle: ||F - E|| <= ||F|| + ||E|| *)
    Admitted.

Lemma B2: reverse triangle lower bound
Math: ∥𝐹∥_1 ≥ ∥𝐹−𝐸∥_1 − ∥𝐸∥_1

    Lemma l1_ge_bool_dist_wrt_minus_embed :
      forall n (F : MV n) (g : Corner n -> bool),
        l1_norm F >= bool_dist_wrt F g - l1_norm (embed g).
    Proof.
    Admitted.

Lemma B3: boolish_k_le gives an ℓ₁ upper bound in terms of coefficients
From triangle:
∥𝐹∥_1 ≤ ∥𝐹−𝐿∥_1 + ∥𝐿∥_1 ≤ 𝑑 + ∥𝐿∥_1
where 𝐿 =lincomb_embed 𝑐𝑠 𝑔

    Lemma l1_le_of_boolish_k_le :
      forall n (F : MV n) k d,
        boolish_k_le F k d ->
        exists cs gs,
          wf_lincomb cs gs /\
          (length gs <= k)%nat /\
          l1_norm F <= d + l1_norm (lincomb_embed cs gs).
    Proof.
      intros n F k d [cs [gs [Hwf [Hlen Hd]]]].
      exists cs, gs; repeat split; try assumption.
      (* triangle: ||F|| <= ||F-L|| + ||L|| *)
    Admitted.


Lemma B4: ℓ₁ of a Boolean lincomb is bounded by sum |cᵢ| (assuming embed has ℓ₁ ≤ 1)

This is the “coefficient growth → ℓ₁ growth” step.

We want (and it’s true for your standard Walsh/projector-style embed) a lemma:
    Lemma l1_embed_le_1 :
      forall n (g : Corner n -> bool),
        l1_norm (embed g) <= 1.
    Admitted.

Then by subadditivity + homogeneity:
    Lemma l1_lincomb_embed_le_sum_abs :
      forall n cs gs,
        wf_lincomb cs gs ->
        l1_norm (lincomb_embed cs gs) <= sumQ (map Qabs cs).
    Proof.
      (* uses l1_embed_le_1 and l1 of scalar multiples + triangle repeatedly *)
    Admitted.

Combine B3+B4:
    Lemma l1_le_of_boolish_k_le_sum_abs :
      forall n (F : MV n) k d,
        boolish_k_le F k d ->
        exists cs gs,
          wf_lincomb cs gs /\
          (length gs <= k)%nat /\
          l1_norm F <= d + sumQ (map Qabs cs).
    Admitted.

This is a workhorse lemma: staying 
𝑘-boolish with small 𝑑 constrains the trace to have a lincomb with controlled coefficient mass;
conversely, if GP forces you to need huge coefficients to approximate, you get huge ℓ₁.

---

C. Excursion lemmas: turning a pointwise ℓ₁ bound into exc_l1

You defined:
    Definition exc_of {n} (sq : Vector.t Q n) (e : GA_expr n) : ExcNum :=
      {| exc_grade := max_grade_during sq e;
         exc_l1    := max_l1_during sq e |}.

So you want the standard “max dominates every element in the trace”.

You presumably already have a semantics/trace function under the hood (e.g. trace_eval or similar). Whatever name it has, you need the lemma:

Lemma C1: max_l1_during is an upper bound for every intermediate
    Lemma l1_le_max_l1_during :
      forall n sq e (F : MV n),
        In F (trace_of sq e) ->
        l1_norm F <= max_l1_during sq e.
    Admitted.

Then immediately:
    Lemma l1_le_exc_l1 :
      forall n sq e (F : MV n),
        In F (trace_of sq e) ->
        l1_norm F <= exc_l1 (exc_of sq e).
    Proof.
      intros; unfold exc_of; simpl.
      eapply l1_le_max_l1_during; eauto.
    Qed.
    
Lemma C2: BoolDist lower bound at a step forces exc_l1 lower bound

Using Lemma B2 plus l1_le_exc_l1:

If at some step you have 
∥𝐹−embed(𝑔)∥_1 ≥ 𝑡, then
    exc_l1 ≥ ∥𝐹∥_1 ≥ 𝑡 − ∥embed(𝑔)∥_1

If you have l1_embed_le_1, that becomes >= t - 1.
    Lemma exc_l1_ge_bool_dist_wrt_minus_1 :
      forall n sq e (F : MV n) (g : Corner n -> bool),
        In F (trace_of sq e) ->
        exc_l1 (exc_of sq e) >= bool_dist_wrt F g - 1.
    Proof.
      intros n sq e F g Hin.
      (* exc_l1 >= ||F|| >= ||F-embed g|| - ||embed g|| >= bool_dist_wrt - 1 *)
    Admitted.

This is the direct “counterexample distance ⇒ excursion” bridge.

D. GP/projector lemmas that explain your counterexamples and force growth

These are the algebraic lemmas that encode “structure causes blow-up”.
They’ll sit above your existing l1_gp_submultiplicative.

Assume you have:

bilinearity / distributivity of mv_gp,
scalar compatibility,
one : MV n and mv_add, mv_sub, scalar smul, etc.

Lemma D1: projector idempotence
    Lemma gp_projector_idempotent :
      forall n sq (B : MV n),
        mv_gp sq B B = one ->
        let p := (1#2) • (mv_sub one B) in
        mv_gp sq p p = p.
    Admitted.
    
Lemma D2: “multiply by projector” identities
    Lemma gp_mul_projector_right :
      forall n sq (X B : MV n),
        mv_gp sq B B = one ->
        let p := (1#2) • (mv_sub one B) in
        mv_gp sq X p = (1#2) • (mv_sub X (mv_gp sq X B)).
    Admitted.

    Lemma gp_mul_projector_left :
      forall n sq (X B : MV n),
        mv_gp sq B B = one ->
        let p := (1#2) • (mv_sub one B) in
        mv_gp sq p X = (1#2) • (mv_sub X (mv_gp sq B X)).
    Admitted.
    
Lemma D3: commutator leakage (difference of left/right projection)
    Definition gp_comm {n} (sq : Vector.t Q n) (X Y : MV n) : MV n :=
      mv_sub (mv_gp sq X Y) (mv_gp sq Y X).

    Lemma gp_projector_commutator :
      forall n sq (X B : MV n),
        mv_gp sq B B = one ->
        let p := (1#2) • (mv_sub one B) in
        mv_sub (mv_gp sq X p) (mv_gp sq p X)
        = (1#2) • (gp_comm sq B X).
    Admitted.

Lemma D4: ℓ₁ lower bound from commutator
By triangle: ∥𝐴∥_1 + ∥𝐶∥_1 ≥ ∥𝐴−𝐶∥_1
Apply with
    𝐴 = 𝑋 ⋆ 𝑝
    𝐶 = 𝑝 ⋆ 𝑋

    Lemma l1_comm_lower_bound_from_projector :
      forall n sq (X B : MV n),
        mv_gp sq B B = one ->
        let p := (1#2) • (mv_sub one B) in
        l1_norm (mv_gp sq X p) + l1_norm (mv_gp sq p X)
          >= (1#2) * l1_norm (gp_comm sq B X).
    Admitted.

Meaning: if the commutator is large, then at least one of the “mixed” terms
has large ℓ₁ — so some intermediate must have large ℓ₁, hence large exc_l1.
That is your “structure → excursion” lever in lemma form.

E. How to turn this into “excursion growth” (the reusable reasoning pattern)

Here’s the exact logical template you’ll reuse in the separation proof:

Your trace invariant is something like: every intermediate 
𝐹_𝑡 is boolish_k_le F_t k d.

But GP steps create terms of the form 
𝑋 ⋆ 𝑝 or 𝑝 ⋆ 𝑋
(or appear as subterms when you expand 
(embed(𝑔) + 𝑝) ⋆ (embed(ℎ) + 𝑝)

If 𝑋 is close to a Boolean lincomb,
    you reduce analysis of 𝑔𝑝_𝑐𝑜𝑚𝑚(𝐵,𝑋) to analysis of 𝑔𝑝_𝑐𝑜𝑚𝑚(𝐵,embed(𝑔𝑖))
        plus error terms (using bilinearity and your ℓ₁ submultiplicativity).

If you can prove a lemma of the form: ∥𝑔𝑝_𝑐𝑜𝑚𝑚(𝐵,embed(𝑔))∥_1 ≥ 𝛼

for “many” 𝑔 (or for the particular 𝑔 that must appear when computing your hard family),
    then Lemma D4 forces ℓ₁ to be at least 𝛼/4 α/4 at some step, so by Lemma C2 you get an exc_l1 lower bound.

Iterating across depth / many subexpressions can turn a constant lower bound into growth via:
    either (a) repeated independent commutators, or (b) repeated need for cancellation inside a bounded 
    𝑘-term lincomb (forcing coefficient sum to grow),
    and then Lemma B4 turns that coefficient growth into ℓ₁ growth.

So the single most “direct” excursion-growth statement you should aim to prove next is:

Target lemma (one-step growth dichotomy)

For a GP multiplication node producing 
𝑃=𝑋⋆𝑌: Either 
    𝑃 is not 𝑘-boolish within 𝑑,
    or ∥𝑃∥_1 is forced large (in terms of commutator/projection mass).

In Coq shape (schematic):
    Lemma gp_step_forces_boolish_break_or_l1 :
      forall n sq X Y k d B,
        mv_gp sq B B = one ->
        (* some hypothesis that Y has nontrivial p-component, e.g. BoolDist(Y, p) small *)
        (* and X is close to some Boolean lincomb with bounded k,d *)
        boolish_k_le X k d ->
        (* conclusion: either result not boolish, or l1 large *)
        (boolish_k_le (mv_gp sq X Y) k d -> l1_norm (mv_gp sq X Y) >= LB).

The concrete counterexamples are exactly of the form
    “take 𝑌≈𝑝” and “𝑋≈embed(𝑔)” where the commutator lower bound is visible.

The mathematical mechanism is now pinned down:
    Blow-ups come from square-+1 blades creating idempotent projectors 𝑝=(1−𝐵)/2
    that persist under GP and force commutator leakage against embedded Booleans;
    commutator leakage forces ℓ₁ mass, and max ℓ₁ during evaluation is exactly exc_l1.
"""

