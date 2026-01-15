"""
Demo 17: Interactive Exercises - Build Geometric Intuition
"""

from r200 import R200
import numpy as np


def get_components(mv):
    return np.array([mv[0], mv[1], mv[2], mv[3]])


def embed_formula(truth_table):
    result = R200(0, 0)
    for p1, p2 in truth_table:
        s1 = 1 if p1 else -1
        s2 = 1 if p2 else -1
        factor1 = (R200(1, 0) + R200(s1, 1)) * 0.5
        factor2 = (R200(1, 0) + R200(s2, 2)) * 0.5
        result = result + (factor1 * factor2)
    return result


print("=" * 80)
print("INTERACTIVE EXERCISES: BUILD GEOMETRIC INTUITION")
print("=" * 80)

# =============================================================================
# EXERCISE 1: Predict the Bivector
# =============================================================================
print("\n" + "=" * 80)
print("EXERCISE 1: PREDICT THE BIVECTOR")
print("=" * 80)
print("\nLook at the truth table. Can you predict if e12 is positive or negative?")
print("Hint: Count agreements vs disagreements")
print("-" * 80)

exercises_1 = [
    ("P1 ∧ P2", [(True, True)]),
    ("P1 → P2", [(True, True), (False, True), (False, False)]),
    ("P1 ⊕ P2", [(True, False), (False, True)]),
]

for name, truth_table in exercises_1:
    print(f"\n{name}:")
    print(f"  Satisfies: {truth_table}")

    # User thinks...
    agreements = sum(1 for p1, p2 in truth_table if p1 == p2)
    disagreements = len(truth_table) - agreements

    print(f"  Agreements: {agreements}, Disagreements: {disagreements}")
    print(f"  Your prediction: ", end="")

    if agreements > disagreements:
        print("POSITIVE (more agreements)")
    elif disagreements > agreements:
        print("NEGATIVE (more disagreements)")
    else:
        print("ZERO (balanced)")

    # Check answer
    mv = embed_formula(truth_table)
    actual = get_components(mv)[3]

    print(f"  Actual e12: {actual:+.3f}")

    # Check if prediction was right
    if (agreements > disagreements and actual > 0) or \
            (disagreements > agreements and actual < 0) or \
            (agreements == disagreements and abs(actual) < 0.01):
        print("  ✅ CORRECT! You're thinking geometrically!")
    else:
        print("  ❌ Not quite. Keep practicing!")

# =============================================================================
# EXERCISE 2: Estimate Distance
# =============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: ESTIMATE DISTANCE")
print("=" * 80)
print("\nWithout computing, estimate which pairs are CLOSEST in geometric space")
print("-" * 80)

formulas = {
    'AND': [(True, True)],
    'OR': [(True, True), (True, False), (False, True)],
    'XOR': [(True, False), (False, True)],
    'NAND': [(True, False), (False, True), (False, False)],
}

embedded = {name: embed_formula(sat) for name, sat in formulas.items()}

pairs = [
    ('AND', 'OR'),
    ('AND', 'NAND'),
    ('XOR', 'NAND'),
]

print("\nPairs to compare:")
for i, (name1, name2) in enumerate(pairs, 1):
    mv1 = embedded[name1]
    mv2 = embedded[name2]
    comps1 = get_components(mv1)
    comps2 = get_components(mv2)

    print(f"\n{i}. {name1} vs {name2}")
    print(f"   {name1}: truth={comps1[0]:.2f}, corr={comps1[3]:+.2f}")
    print(f"   {name2}: truth={comps2[0]:.2f}, corr={comps2[3]:+.2f}")

print("\n❓ Which pair is CLOSEST? (Enter 1, 2, or 3)")
print("\nThink about:")
print("  • Truth probability difference")
print("  • Correlation difference")
print("  • Both contribute to distance!")

# Compute actual distances
distances = []
for name1, name2 in pairs:
    mv1 = embedded[name1]
    mv2 = embedded[name2]
    comps1 = get_components(mv1)
    comps2 = get_components(mv2)
    dist = np.linalg.norm(comps1 - comps2)
    distances.append((name1, name2, dist))

print("\n💡 ANSWER:")
for i, (name1, name2, dist) in enumerate(distances, 1):
    print(f"{i}. {name1} ↔ {name2}: distance = {dist:.3f}")

closest_idx = np.argmin([d[2] for d in distances])
print(f"\nClosest pair: {distances[closest_idx][0]} ↔ {distances[closest_idx][1]}")

# =============================================================================
# EXERCISE 3: Visualize Transformations
# =============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: IMAGINE THE PATH")
print("=" * 80)
print("\nImagine smoothly transforming AND into OR")
print("What happens to the components?")
print("-" * 80)

AND = embed_formula([(True, True)])
OR = embed_formula([(True, True), (True, False), (False, True)])

and_comps = get_components(AND)
or_comps = get_components(OR)

print(f"\nStart (AND): truth={and_comps[0]:.2f}, corr={and_comps[3]:+.2f}")
print(f"End (OR):    truth={or_comps[0]:.2f}, corr={or_comps[3]:+.2f}")

print("\n❓ As we move from AND to OR:")
print("1. Does truth increase or decrease?")
print("2. Does correlation become more positive or negative?")

print("\n💡 ANSWER:")
print(f"1. Truth: {and_comps[0]:.2f} → {or_comps[0]:.2f} = INCREASES")
print(f"2. Correlation: {and_comps[3]:+.2f} → {or_comps[3]:+.2f} = BECOMES MORE NEGATIVE")

print("\nLet's trace the path:")
for t in [0, 0.25, 0.5, 0.75, 1.0]:
    t = float(t)
    interp = AND * (1 - t) + OR * t
    comps = get_components(interp)
    print(f"  t={t:.2f}: truth={comps[0]:.2f}, corr={comps[3]:+.2f}")

print("\n✨ Can you 'see' the smooth transition?")
print("   This is geometric thinking!")

# =============================================================================
# FINAL INSIGHT
# =============================================================================
print("\n" + "=" * 80)
print("🎯 KEY MENTAL SHIFT")
print("=" * 80)

insight = """
Boolean thinker asks: "What's the output?"
Geometric thinker asks: "What's the structure?"

Boolean: Formula → Evaluation → Answer (DONE)
Geometric: Formula → Position → Relationships → Transformations

The moment you start thinking:
  "Where is this formula?"
  "How far is it from that one?"
  "What's between them?"
  "How does it transform?"

...you're thinking geometrically! 🚀

Practice until this becomes NATURAL.
The 16 Boolean operations are just LANDMARKS
in an infinite geometric landscape.
"""

print(insight)