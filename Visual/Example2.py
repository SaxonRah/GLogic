"""
Demo 2: Same Probability, Different Structure
Two formulas with 25% probability but OPPOSITE geometry
"""

from r200 import R200, e1, e2, e12
import numpy as np
import matplotlib.pyplot as plt


def get_components(mv):
    return np.array([mv[0], mv[1], mv[2], mv[3]])


def embed_formula(truth_table):
    """
    Embed Boolean formula into Cl(2,0).
    truth_table: list of (p1, p2) tuples that satisfy the formula
    """
    result = R200(0, 0)  # Zero multivector

    for p1, p2 in truth_table:
        s1 = 1 if p1 else -1
        s2 = 1 if p2 else -1

        # Create quasi-projector Π(α) = [(1 + s1*e1)/2] * [(1 + s2*e2)/2]
        factor1 = (R200(1, 0) + R200(s1, 1)) * 0.5
        factor2 = (R200(1, 0) + R200(s2, 2)) * 0.5
        quasi_proj = factor1 * factor2

        result = result + quasi_proj

    return result


print("=" * 70)
print("DEMO 2: SAME PROBABILITY, DIFFERENT GEOMETRY")
print("=" * 70)

# Two formulas with SAME 25% probability
formula1_sat = [(True, True)]  # P1 AND P2
formula2_sat = [(True, False)]  # P1 AND NOT P2

f1 = embed_formula(formula1_sat)
f2 = embed_formula(formula2_sat)

comps1 = get_components(f1)
comps2 = get_components(f2)

print("\n📐 Formula 1: P1 ∧ P2")
print(f"  Satisfies: {formula1_sat}")
print(f"  Multivector: {f1}")
print(f"  Probability: {comps1[0]:.2%}")
print(f"  Bivector e12: {comps1[3]:+.3f}")
print(f"  → Variables AGREE ✓")

print("\n📐 Formula 2: P1 ∧ ¬P2")
print(f"  Satisfies: {formula2_sat}")
print(f"  Multivector: {f2}")
print(f"  Probability: {comps2[0]:.2%}")
print(f"  Bivector e12: {comps2[3]:+.3f}")
print(f"  → Variables DISAGREE ✗")

print("\n" + "=" * 70)
print("💡 KEY INSIGHT")
print("=" * 70)
print("Both formulas are true 25% of the time...")
print("But their geometric structure is OPPOSITE!")
print(f"  AND:     e12 = {comps1[3]:+.3f} (positive)")
print(f"  AND NOT: e12 = {comps2[3]:+.3f} (negative)")
print("\nBoolean logic can't tell them apart.")
print("Geometric logic sees them as fundamentally different.")
print("=" * 70)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

components = ['scalar', 'e1', 'e2', 'e12']
colors = ['blue', 'green', 'green', 'red']

# Formula 1
bars1 = ax1.bar(components, comps1, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_title('P1 ∧ P2 (AND)\n25% probability', fontsize=12, fontweight='bold')
ax1.set_ylabel('Component Value')
ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_ylim(-0.4, 0.4)
bars1[3].set_edgecolor('red')
bars1[3].set_linewidth(3)

# Add annotation
ax1.text(3, comps1[3] + 0.03, f'{comps1[3]:+.2f}\nAGREE',
         ha='center', fontweight='bold', color='green')

# Formula 2
bars2 = ax2.bar(components, comps2, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_title('P1 ∧ ¬P2 (AND NOT)\n25% probability', fontsize=12, fontweight='bold')
ax2.set_ylabel('Component Value')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax2.set_ylim(-0.4, 0.4)
bars2[3].set_edgecolor('red')
bars2[3].set_linewidth(3)

# Add annotation
ax2.text(3, comps2[3] - 0.03, f'{comps2[3]:+.2f}\nDISAGREE',
         ha='center', va='top', fontweight='bold', color='red')

plt.suptitle('Demo 2: Same Probability, Opposite Structure',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('demo2_same_probability.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: demo2_same_probability.png")
plt.show()