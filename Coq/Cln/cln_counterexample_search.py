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