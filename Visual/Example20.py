"""
CORRECTED: Demo 20 - Evaluation Functions
"""

from r200 import R200
import numpy as np

def get_components(mv):
    """Extract components as numpy array"""
    return np.array([mv[0], mv[1], mv[2], mv[3]])

def embed_formula(truth_table):
    """Embed a Boolean formula given its truth table."""
    result = R200(0, 0)
    for p1, p2 in truth_table:
        s1 = 1 if p1 else -1
        s2 = 1 if p2 else -1
        factor1 = (R200(1, 0) + R200(s1, 1)) * 0.5
        factor2 = (R200(1, 0) + R200(s2, 2)) * 0.5
        quasi_proj = factor1 * factor2
        result = result + quasi_proj
    return result

def evaluate_geometric_CORRECT(formula_mv, p1_value, p2_value):
    """
    CORRECTED: Evaluate a geometric formula on specific Boolean inputs.

    The key insight: A multivector in Cl(2,0) can be evaluated on sign values!

    If F = a·1 + b·e1 + c·e2 + d·e12, then evaluating on (s1, s2) gives:
    F(s1, s2) = a + b·s1 + c·s2 + d·(s1·s2)

    where s1, s2 ∈ {+1, -1}
    """
    # Extract components
    comps = get_components(formula_mv)
    scalar = comps[0]
    e1_comp = comps[1]
    e2_comp = comps[2]
    e12_comp = comps[3]

    # Convert Boolean values to signs
    s1 = 1 if p1_value else -1
    s2 = 1 if p2_value else -1

    # Evaluate the multivector at this point
    # This is the key: treat the multivector as a function on {-1,+1}²
    result = scalar + e1_comp * s1 + e2_comp * s2 + e12_comp * (s1 * s2)

    # If result ≥ 0.2, the formula is satisfied
    # (Each satisfying assignment contributes 0.25 to the result)
    return result >= 0.2

# Test the corrected evaluation
print("="*80)
print("CORRECTED EVALUATION TESTS")
print("="*80)

# Test OR
or_formula = embed_formula([(True, True), (True, False), (False, True)])
print("\nTesting OR formula:")
print(f"Embedded: {or_formula}")
print("-"*80)

for p1 in [True, False]:
    for p2 in [True, False]:
        geometric_result = evaluate_geometric_CORRECT(or_formula, p1, p2)
        boolean_result = p1 or p2
        match = "✓" if geometric_result == boolean_result else "✗"

        # Show the evaluation
        comps = get_components(or_formula)
        s1 = 1 if p1 else -1
        s2 = 1 if p2 else -1
        eval_value = comps[0] + comps[1]*s1 + comps[2]*s2 + comps[3]*(s1*s2)

        print(f"P1={p1:5}, P2={p2:5}: "
              f"eval={eval_value:.3f}, "
              f"Geometric={geometric_result:5}, "
              f"Boolean={boolean_result:5} {match}")

# Test IFF
print("\n" + "="*80)
print("Testing IFF formula:")
iff_formula = embed_formula([(True, True), (False, False)])
print(f"Embedded: {iff_formula}")
print("-"*80)

for p1 in [True, False]:
    for p2 in [True, False]:
        geometric_result = evaluate_geometric_CORRECT(iff_formula, p1, p2)
        boolean_result = (p1 and p2) or (not p1 and not p2)
        match = "✓" if geometric_result == boolean_result else "✗"

        # Show the evaluation
        comps = get_components(iff_formula)
        s1 = 1 if p1 else -1
        s2 = 1 if p2 else -1
        eval_value = comps[0] + comps[1]*s1 + comps[2]*s2 + comps[3]*(s1*s2)

        print(f"P1={p1:5}, P2={p2:5}: "
              f"eval={eval_value:.3f}, "
              f"Geometric={geometric_result:5}, "
              f"Boolean={boolean_result:5} {match}")

# Test all 16 operations
print("\n" + "="*80)
print("COMPLETE TEST: ALL 16 OPERATIONS")
print("="*80)

all_operations = {
    'Contradiction': [],
    'NOR': [(False, False)],
    'P1∧¬P2': [(True, False)],
    '¬P1': [(False, True), (False, False)],
    '¬P1∧P2': [(False, True)],
    '¬P2': [(True, False), (False, False)],
    'XOR': [(True, False), (False, True)],
    'NAND': [(True, False), (False, True), (False, False)],
    'AND': [(True, True)],
    'IFF': [(True, True), (False, False)],
    'P2': [(True, True), (False, True)],
    'P1→P2': [(True, True), (False, True), (False, False)],
    'P1': [(True, True), (True, False)],
    'P2→P1': [(True, True), (True, False), (False, False)],
    'OR': [(True, True), (True, False), (False, True)],
    'Tautology': [(True, True), (True, False), (False, True), (False, False)],
}

errors = 0
total_tests = 0

for name, truth_table in all_operations.items():
    formula_mv = embed_formula(truth_table)

    for p1 in [True, False]:
        for p2 in [True, False]:
            expected = (p1, p2) in truth_table
            result = evaluate_geometric_CORRECT(formula_mv, p1, p2)
            total_tests += 1

            if result != expected:
                errors += 1
                print(f"ERROR in {name}: ({p1}, {p2}) expected {expected}, got {result}")

print(f"\nTotal tests: {total_tests}")
print(f"Errors: {errors}")
print(f"Success rate: {100*(total_tests-errors)/total_tests:.1f}%")

if errors == 0:
    print("\n✅ ALL TESTS PASSED!")
else:
    print(f"\n❌ {errors} tests failed")

# Explanation
print("\n" + "="*80)
print("WHY THIS WORKS: THE GEOMETRIC EVALUATION PRINCIPLE")
print("="*80)

explanation = """
A multivector in Cl(2,0) is:
  F = a·1 + b·e1 + c·e2 + d·e12

We can EVALUATE this on sign values (s1, s2) ∈ {-1, +1}²:
  F(s1, s2) = a + b·s1 + c·s2 + d·(s1·s2)

This works because:
1. e1 acts as "multiplication by s1" when evaluated
2. e2 acts as "multiplication by s2" when evaluated  
3. e12 acts as "multiplication by s1·s2" when evaluated
4. The scalar is constant

Example: OR = 0.75 + 0.25·e1 + 0.25·e2 - 0.25·e12

Evaluate at (True, True) → (s1=+1, s2=+1):
  OR(+1, +1) = 0.75 + 0.25·(+1) + 0.25·(+1) - 0.25·(+1·+1)
             = 0.75 + 0.25 + 0.25 - 0.25
             = 1.00 ✓ TRUE!

Evaluate at (True, False) → (s1=+1, s2=-1):
  OR(+1, -1) = 0.75 + 0.25·(+1) + 0.25·(-1) - 0.25·(+1·-1)
             = 0.75 + 0.25 - 0.25 - 0.25·(-1)
             = 0.75 + 0.25
             = 1.00 ✓ TRUE!

Evaluate at (False, False) → (s1=-1, s2=-1):
  OR(-1, -1) = 0.75 + 0.25·(-1) + 0.25·(-1) - 0.25·(-1·-1)
             = 0.75 - 0.25 - 0.25 - 0.25
             = 0.00 ✓ FALSE!

The threshold is 0.2 because each satisfying assignment
contributes 0.25 to the result when evaluated at that point.

This is the BEAUTIFUL connection between:
  • Boolean logic (truth values)
  • Geometric algebra (multivectors)
  • Polynomial evaluation (on {-1,+1}²)
"""

print(explanation)

print("\n" + "="*80)
print("CORRECTED GeometricBoolean CLASS")
print("="*80)

class GeometricBoolean:
    """CORRECTED Helper class for geometric Boolean operations"""

    @staticmethod
    def from_truth_table(truth_table):
        """Create from truth table"""
        return embed_formula(truth_table)

    @staticmethod
    def from_function(bool_func):
        """Create from a Python Boolean function"""
        truth_table = []
        for p1 in [True, False]:
            for p2 in [True, False]:
                if bool_func(p1, p2):
                    truth_table.append((p1, p2))
        return embed_formula(truth_table)

    @staticmethod
    def evaluate(formula_mv, p1_val, p2_val):
        """Evaluate formula on inputs - CORRECTED"""
        return evaluate_geometric_CORRECT(formula_mv, p1_val, p2_val)

    @staticmethod
    def probability(formula_mv):
        """Get truth probability"""
        return get_components(formula_mv)[0]

    @staticmethod
    def correlation(formula_mv):
        """Get variable correlation"""
        return get_components(formula_mv)[3]

# Test the corrected class
print("\nTesting corrected GeometricBoolean class:")

def my_formula(p1, p2):
    return (p1 and p2) or (not p1 and not p2)

formula = GeometricBoolean.from_function(my_formula)
print(f"Formula: (P1 ∧ P2) ∨ (¬P1 ∧ ¬P2) [IFF]")
print(f"Embedded: {formula}")
print(f"Probability: {GeometricBoolean.probability(formula):.2%}")
print(f"Correlation: {GeometricBoolean.correlation(formula):+.3f}")

print("\nEvaluation tests:")
for p1 in [True, False]:
    for p2 in [True, False]:
        result = GeometricBoolean.evaluate(formula, p1, p2)
        expected = my_formula(p1, p2)
        match = "✓" if result == expected else "✗"
        print(f"  ({p1:5}, {p2:5}): {result:5} (expected {expected:5}) {match}")

print("\n🎉 EVALUATION FIXED!")
