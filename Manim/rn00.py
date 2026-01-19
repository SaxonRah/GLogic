"""
Rn00.py — Generic Clifford Algebra Cl(n,0) Playground

EXTENDED VERSION

Adds:
- BooleanEmbedder (truth-table → multivector)
- Hypercube evaluation rule
- Sparse grade inspection utilities
- Simple rotation / inference experiments
- Visualization helpers for low n

Exploratory, educational, and faithful to Cl(n,0).
"""

import itertools
import csv
import math
from functools import lru_cache
from collections import defaultdict

# ------------------------------------------------------------
# Basis utilities
# ------------------------------------------------------------

def blade_bits_to_name(bits):
    if bits == 0:
        return "1"
    idxs = [str(i+1) for i in range(bits.bit_length()) if (bits >> i) & 1]
    return "e" + "".join(idxs)


def count_bits(x):
    return bin(x).count("1")


# ------------------------------------------------------------
# Clifford sign rules for Cl(n,0)
# ------------------------------------------------------------

@lru_cache(None)
def blade_mul(a, b):
    sign = 1
    res = a ^ b

    common = a & b
    swaps = 0
    while common:
        i = common & -common
        swaps += count_bits(a & (i - 1))
        common ^= i

    if swaps % 2:
        sign = -sign

    return sign, res


# ------------------------------------------------------------
# Multivector class
# ------------------------------------------------------------

class Rn00:
    def __init__(self, n, coeffs=None):
        self.n = n
        self.dim = 1 << n
        if coeffs is None:
            self.coeffs = [0.0] * self.dim
        else:
            if len(coeffs) != self.dim:
                raise ValueError("Coefficient length mismatch")
            self.coeffs = list(coeffs)

    @staticmethod
    def basis(n, blade_bits, value=1.0):
        mv = Rn00(n)
        mv.coeffs[blade_bits] = value
        return mv

    def copy(self):
        return Rn00(self.n, self.coeffs)

    def __add__(self, other):
        assert self.n == other.n
        return Rn00(self.n, [a+b for a,b in zip(self.coeffs, other.coeffs)])

    def __mul__(self, other):
        # multivector * multivector
        if isinstance(other, Rn00):
            assert self.n == other.n
            result = [0.0] * self.dim
            for i,a in enumerate(self.coeffs):
                if a == 0: continue
                for j,b in enumerate(other.coeffs):
                    if b == 0: continue
                    sign, k = blade_mul(i, j)
                    result[k] += sign * a * b
            return Rn00(self.n, result)
        # multivector * scalar
        elif isinstance(other, (int, float)):
            return Rn00(self.n, [c * other for c in self.coeffs])
        else:
            return NotImplemented
        assert self.n == other.n
        result = [0.0] * self.dim
        for i,a in enumerate(self.coeffs):
            if a == 0: continue
            for j,b in enumerate(other.coeffs):
                if b == 0: continue
                sign, k = blade_mul(i, j)
                result[k] += sign * a * b
        return Rn00(self.n, result)

    def grades(self):
        out = defaultdict(list)
        for i,c in enumerate(self.coeffs):
            if abs(c) > 1e-9:
                out[count_bits(i)].append((i,c))
        return out

    def __str__(self):
        terms = []
        for i,c in enumerate(self.coeffs):
            if abs(c) > 1e-9:
                terms.append(f"{c:+g}{blade_bits_to_name(i)}")
        return " ".join(terms) if terms else "0"


# ------------------------------------------------------------
# Boolean embedding & evaluation
# ------------------------------------------------------------

class BooleanEmbedder:
    def __init__(self, n):
        self.n = n

    def embed_truth_table(self, satisfying_assignments):
        mv = Rn00(self.n)
        for assignment in satisfying_assignments:
            signs = [1 if v else -1 for v in assignment]
            proj = Rn00(self.n)
            proj.coeffs[0] = 1.0
            for i,s in enumerate(signs):
                ei = Rn00.basis(self.n, 1 << i)
                proj = proj * (Rn00.basis(self.n, 0, 1.0) + ei * s) * 0.5
            mv = mv + proj
        return mv

    def evaluate(self, mv, signs, threshold=None):
        if threshold is None:
            threshold = 1.0 / (1 << self.n)
        total = 0.0
        for i,c in enumerate(mv.coeffs):
            if c == 0: continue
            prod = 1
            for bit in range(self.n):
                if (i >> bit) & 1:
                    prod *= signs[bit]
            total += c * prod
        return total, total >= threshold


# ------------------------------------------------------------
# Inference / rotation experiments
# ------------------------------------------------------------

def rotate(mv, rotor):
    return rotor * mv * rotor


def simple_bias_rotor(n, index, theta):
    ei = Rn00.basis(n, 1 << index)
    return Rn00.basis(n, 0, math.cos(theta)) + ei * math.sin(theta)


# ------------------------------------------------------------
# Visualization helpers (low n only)
# ------------------------------------------------------------

def print_grade_summary(mv):
    grades = mv.grades()
    for g in sorted(grades):
        print(f"Grade {g}:")
        for i,c in grades[g]:
            print(f"  {c:+.3f} {blade_bits_to_name(i)}")


def ascii_hypercube(mv):
    n = mv.n
    print("Hypercube evaluation (signs → value):")
    for bits in itertools.product([1,-1], repeat=n):
        val,_ = BooleanEmbedder(n).evaluate(mv, list(bits))
        print(bits, f"→ {val:.3f}")


# ------------------------------------------------------------
# Cayley table generation
# ------------------------------------------------------------

def generate_cayley_table(n, csv_path):
    dim = 1 << n
    names = [blade_bits_to_name(i) for i in range(dim)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["×"] + names)
        for i in range(dim):
            row = [names[i]]
            for j in range(dim):
                sign, k = blade_mul(i, j)
                entry = ("-" if sign < 0 else "") + names[k]
                row.append(entry)
            writer.writerow(row)


# ------------------------------------------------------------
# Demo
# ------------------------------------------------------------

if __name__ == "__main__":
    n = 10
    embedder = BooleanEmbedder(n)

    # XOR example
    xor_tt = [(True, False, False), (False, True, False), (False, False, True)]
    F = embedder.embed_truth_table(xor_tt)

    print("Embedded XOR:")
    print(F)
    print("\nGrade structure:")
    print_grade_summary(F)

    print("\nEvaluation:")
    ascii_hypercube(F)

    print("\nRotation experiment:")
    rotor = simple_bias_rotor(n, 0, 0.3)
    G = rotate(F, rotor)
    print_grade_summary(G)

    print("\nGenerating Cayley table...")
    generate_cayley_table(n, f"cayley_cl{n}.csv")
    print("Done.")
