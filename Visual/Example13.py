"""
Demo 13: Geometric Operations - Boolean Logic Interpretation
Making sense of geometric operations for Boolean logicians
"""

from r200 import R200
import numpy as np
import matplotlib.pyplot as plt

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

def interpret_components(comps, name):
    """Provide human-readable interpretation"""
    print(f"\n{name}:")
    print(f"  Multivector: {comps[0]:.3f}·1 + {comps[1]:+.3f}·e1 + {comps[2]:+.3f}·e2 + {comps[3]:+.3f}·e12")
    print(f"  📊 Interpretation:")
    print(f"     Truth probability: {comps[0]:.1%}")
    print(f"     P1 bias: {comps[1]:+.3f} {'(tends true)' if comps[1] > 0.1 else '(tends false)' if comps[1] < -0.1 else '(neutral)'}")
    print(f"     P2 bias: {comps[2]:+.3f} {'(tends true)' if comps[2] > 0.1 else '(tends false)' if comps[2] < -0.1 else '(neutral)'}")
    print(f"     Correlation: {comps[3]:+.3f} {'(variables agree)' if comps[3] > 0.1 else '(variables disagree)' if comps[3] < -0.1 else '(independent)'}")

print("="*80)
print("GEOMETRIC OPERATIONS: BOOLEAN LOGIC INTERPRETATION")
print("="*80)
print("\nWhat do these operations MEAN for Boolean formulas?")
print("="*80)

# Create meaningful Boolean formulas
AND = embed_formula([(True, True)])
OR = embed_formula([(True, True), (True, False), (False, True)])
XOR = embed_formula([(True, False), (False, True)])
IFF = embed_formula([(True, True), (False, False)])

and_comps = get_components(AND)
or_comps = get_components(OR)

print("\n📐 Test Formulas:")
interpret_components(and_comps, "AND (P1 ∧ P2)")
interpret_components(or_comps, "OR (P1 ∨ P2)")

# =============================================================================
# 1. GEOMETRIC PRODUCT (*)
# =============================================================================
print("\n" + "="*80)
print("1️⃣  GEOMETRIC PRODUCT: A * B")
print("="*80)
print("\n🎯 MEANING: \"How do these formulas combine?\"")
print("   - Computes JOINT probability distribution")
print("   - Includes all correlation effects")
print("   - For independent formulas: same as Boolean AND")

result = AND * OR
result_comps = get_components(result)

print("\nExample: AND * OR")
interpret_components(result_comps, "Result")

print("\n💡 USE CASE:")
print("   Given: 'user is authenticated' AND 'user has permission'")
print("   Q: What's the correlation when BOTH are required?")
print("   A: Geometric product tells you instantly!")

# =============================================================================
# 2. OUTER PRODUCT (^)
# =============================================================================
print("\n" + "="*80)
print("2️⃣  OUTER PRODUCT (WEDGE): A ^ B")
print("="*80)
print("\n🎯 MEANING: \"Pure combination ignoring interactions\"")
print("   - Extracts the INDEPENDENT parts")
print("   - Grade-increasing: scalar∧vector → bivector")
print("   - Antisymmetric: A^B = -(B^A)")

result = AND ^ OR
result_comps = get_components(result)

print("\nExample: AND ^ OR")
interpret_components(result_comps, "Result")

print("\n💡 USE CASE:")
print("   Testing if two conditions are LOGICALLY INDEPENDENT")
print("   If A^B ≠ A*B, there are hidden correlations!")

# Compare
gp_result = AND * OR
op_result = AND ^ OR
gp_comps = get_components(gp_result)
op_comps = get_components(op_result)

print(f"\n   Geometric Product e12: {gp_comps[3]:+.3f}")
print(f"   Outer Product e12:     {op_comps[3]:+.3f}")
print(f"   Difference:             {abs(gp_comps[3] - op_comps[3]):.3f}")
if abs(gp_comps[3] - op_comps[3]) > 0.01:
    print("   ⚠️  They differ! There ARE interaction effects!")
else:
    print("   ✓ Same! Variables are independent!")

# =============================================================================
# 3. INNER PRODUCT (|)
# =============================================================================
print("\n" + "="*80)
print("3️⃣  INNER PRODUCT (DOT): A | B")
print("="*80)
print("\n🎯 MEANING: \"How aligned/similar are these formulas?\"")
print("   - Measures OVERLAP between formula patterns")
print("   - Grade-decreasing: bivector|vector → scalar")
print("   - Returns a similarity score")

result = AND | OR
result_comps = get_components(result)

print("\nExample: AND | OR")
interpret_components(result_comps, "Result")

# Test similarity between different pairs
pairs = [
    ("AND", "OR", AND, OR),
    ("AND", "XOR", AND, XOR),
    ("XOR", "IFF", XOR, IFF),
]

print("\n💡 SIMILARITY SCORES:")
for name1, name2, mv1, mv2 in pairs:
    inner = mv1 | mv2
    similarity = get_components(inner)[0]  # Scalar part
    print(f"   {name1:6} | {name2:6} = {similarity:+.3f}")

print("\n   Interpretation:")
print("   - Positive: Formulas are aligned (often true together)")
print("   - Negative: Formulas are opposed (rarely true together)")
print("   - Zero: Formulas are orthogonal (independent)")

# =============================================================================
# 4. REVERSE (~)
# =============================================================================
print("\n" + "="*80)
print("4️⃣  REVERSE: ~A")
print("="*80)
print("\n🎯 MEANING: \"Flip the orientation of correlations\"")
print("   - Reverses order of basis blades")
print("   - Bivector changes sign: e12 → -e12")
print("   - Scalar unchanged")

result = ~AND
result_comps = get_components(result)

print("\nExample: ~AND")
interpret_components(result_comps, "Result")

print("\n💡 INTERPRETATION:")
print(f"   Original AND: e12 = {and_comps[3]:+.3f} (positive correlation)")
print(f"   Reversed ~AND: e12 = {result_comps[3]:+.3f} (flipped!)")
print("\n   USE CASE:")
print("   - Changes the DIRECTION of correlation")
print("   - Used in sandwich products (rotations)")
print("   - Like transposing a matrix: (~A) * B * A")

# =============================================================================
# 5. DUAL (!)
# =============================================================================
print("\n" + "="*80)
print("5️⃣  DUAL: !A")
print("="*80)
print("\n🎯 MEANING: \"Complement in the full geometric space\"")
print("   - Related to logical NOT, but in ALL dimensions")
print("   - Swaps: scalar ↔ pseudoscalar (e12)")
print("   - Swaps: e1 ↔ e2")

result = AND.Dual()
result_comps = get_components(result)

print("\nExample: !AND")
interpret_components(result_comps, "Result")

print("\n💡 COMPARISON with NOT:")
NOT_AND = R200(1, 0) - AND  # Boolean NOT
not_and_comps = get_components(NOT_AND)

print(f"\n   Boolean NOT (1 - AND):")
print(f"     scalar: {not_and_comps[0]:.3f}, e12: {not_and_comps[3]:+.3f}")
print(f"   Geometric DUAL (!AND):")
print(f"     scalar: {result_comps[0]:.3f}, e12: {result_comps[3]:+.3f}")
print("\n   They're DIFFERENT! Dual operates on full geometry, not just truth!")

# =============================================================================
# 6. CONJUGATE
# =============================================================================
print("\n" + "="*80)
print("6️⃣  CONJUGATE: A.Conjugate()")
print("="*80)
print("\n🎯 MEANING: \"Mirror reflection in geometric space\"")
print("   - Negates ALL vector and bivector components")
print("   - Scalar unchanged")
print("   - Like complex conjugate, but for multivectors")

result = AND.Conjugate()
result_comps = get_components(result)

print("\nExample: AND.Conjugate()")
interpret_components(result_comps, "Result")

print("\n💡 INTERPRETATION:")
print(f"   Original:   e1={and_comps[1]:+.3f}, e2={and_comps[2]:+.3f}, e12={and_comps[3]:+.3f}")
print(f"   Conjugate:  e1={result_comps[1]:+.3f}, e2={result_comps[2]:+.3f}, e12={result_comps[3]:+.3f}")
print("\n   USE CASE:")
print("   - Computing norm: |A|² = A * A.Conjugate()")
print("   - Finding 'mirror image' formulas")

# Compute norm
norm_squared = AND * AND.Conjugate()
norm_sq_comps = get_components(norm_squared)
print(f"\n   Norm² = A * A.Conjugate() = {norm_sq_comps[0]:.3f}")
print(f"   Norm  = √{norm_sq_comps[0]:.3f} = {np.sqrt(norm_sq_comps[0]):.3f}")

# =============================================================================
# 7. SANDWICH PRODUCT
# =============================================================================
print("\n" + "="*80)
print("7️⃣  SANDWICH PRODUCT: A * B * ~A")
print("="*80)
print("\n🎯 MEANING: \"Transform B through the lens of A\"")
print("   - Rotates/reflects B using A as operator")
print("   - Preserves geometric properties (angles, norms)")
print("   - Core operation for geometric transformations")

result = AND * OR * ~AND
result_comps = get_components(result)

print("\nExample: AND * OR * ~AND")
interpret_components(result_comps, "Result")

print("\n💡 INTERPRETATION:")
print("   'View OR through the perspective of AND'")
print("   - Transforms correlation structure")
print("   - Preserves essential properties")
print("\n   Original OR:")
print(f"     scalar: {or_comps[0]:.3f}, e12: {or_comps[3]:+.3f}")
print("   Transformed:")
print(f"     scalar: {result_comps[0]:.3f}, e12: {result_comps[3]:+.3f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("📚 SUMMARY: GEOMETRIC OPERATIONS FOR BOOLEAN LOGICIANS")
print("="*80)

summary = """
┌────────────────┬───────────────────────────────────────────────────────┐
│ Operation      │ Boolean/Correlation Meaning                           │
├────────────────┼───────────────────────────────────────────────────────┤
│ A * B          │ Joint distribution (with correlations)                │
│ (Geometric)    │ "What happens when both formulas matter?"             │
├────────────────┼───────────────────────────────────────────────────────┤
│ A ^ B          │ Independent combination (no interaction)              │
│ (Outer/Wedge)  │ "What if variables DON'T influence each other?"       │
├────────────────┼───────────────────────────────────────────────────────┤
│ A | B          │ Similarity/overlap score                              │
│ (Inner/Dot)    │ "How aligned are these formulas?"                     │
├────────────────┼───────────────────────────────────────────────────────┤
│ ~A             │ Flip correlation orientation                          │
│ (Reverse)      │ "Reverse the direction of variable relationships"     │
├────────────────┼───────────────────────────────────────────────────────┤
│ !A             │ Full geometric complement                             │
│ (Dual)         │ "NOT in all dimensions (not just truth)"              │
├────────────────┼───────────────────────────────────────────────────────┤
│ A.Conjugate()  │ Mirror reflection in variable space                   │
│ (Conjugate)    │ "Flip all variable biases and correlations"           │
├────────────────┼───────────────────────────────────────────────────────┤
│ A * B * ~A     │ Transform B through perspective of A                  │
│ (Sandwich)     │ "View one formula through the lens of another"        │
└────────────────┴───────────────────────────────────────────────────────┘
"""

print(summary)

print("\n💡 KEY INSIGHTS:")
print("   1. These operations work on FULL structure (truth + correlations)")
print("   2. Boolean logic only has: AND, OR, NOT (scalar operations)")
print("   3. Geometric logic adds: rotations, reflections, projections")
print("   4. Result: Infinite operations vs Boolean's 16!")

print("\n🎯 PRACTICAL USES:")
print("   • Geometric Product → Combining dependent conditions")
print("   • Inner Product → Measuring formula similarity")
print("   • Outer Product → Detecting hidden correlations")
print("   • Sandwich Product → Context-dependent transformations")
print("   • Reverse → Symmetry analysis")
print("   • Dual → Multi-dimensional complementation")

print("\n" + "="*80)