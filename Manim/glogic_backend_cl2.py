from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Union

# Adjust the import path to wherever rn00.py lives in your repo.
# If it's in GeneralGLogic/rn00.py, either:
#  - put __init__.py files and import as a package, OR
#  - add that folder to PYTHONPATH in PyCharm run config, OR
#  - do a relative sys.path hack (not recommended long-term)
from rn00 import BooleanEmbedder  # expects rn00.py in import path


# Corner order used by the Manim component: (TT, TF, FT, FF)
CORNER_ORDER_SIGNS: List[Tuple[int, int]] = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]


def mask_from_bool_func(func: Callable[[bool, bool], bool]) -> int:
    """
    Convert a boolean function (p1,p2)->bool into a 4-bit mask.
    bit0=TT, bit1=TF, bit2=FT, bit3=FF
    """
    mask = 0
    for i, (s1, s2) in enumerate(CORNER_ORDER_SIGNS):
        p1, p2 = (s1 == +1), (s2 == +1)
        if func(p1, p2):
            mask |= (1 << i)
    return mask


def satisfying_assignments_from_mask(mask: int) -> List[Tuple[bool, bool]]:
    """
    Convert mask (TT,TF,FT,FF) into list of satisfying (bool,bool) assignments.
    """
    sats = []
    for i, (s1, s2) in enumerate(CORNER_ORDER_SIGNS):
        if (mask >> i) & 1:
            sats.append((s1 == +1, s2 == +1))
    return sats


def embed_cl2_from_mask(mask: int):
    """
    Build the Cl(2) multivector F for a 2-variable boolean op given by mask.
    """
    embedder = BooleanEmbedder(2)
    sats = satisfying_assignments_from_mask(mask)
    return embedder.embed_truth_table(sats)


def eval_mask_from_mv(F, threshold: float = None) -> int:
    """
    Evaluate the multivector F on all 4 corners and return the implied mask.
    Uses BooleanEmbedder.evaluate().
    """
    embedder = BooleanEmbedder(2)
    mask = 0
    for i, (s1, s2) in enumerate(CORNER_ORDER_SIGNS):
        val, truth = embedder.evaluate(F, [s1, s2], threshold=threshold)
        if truth:
            mask |= (1 << i)
    return mask


def cl2_coeffs_from_mv(F) -> Dict[str, float]:
    """
    Extract Cl(2) coefficients in the basis {1, e1, e2, e12}.
    In rn00, blades are indexed by bits:
      0 -> 1
      1 -> e1
      2 -> e2
      3 -> e12 (1|2)
    """
    # F.coeffs length is 2^n = 4 for n=2
    return {
        "1": float(F.coeffs[0]),
        "e1": float(F.coeffs[1]),
        "e2": float(F.coeffs[2]),
        "e12": float(F.coeffs[3]),
    }
