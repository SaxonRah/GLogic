"""
Demo 3: The AHA Moment - IFF vs XOR
Geometric opposites with same probability
"""

from r200 import R200, e1, e2, e12
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

print("="*70)
print("DEMO 3: THE AHA MOMENT - IFF vs XOR")
print("="*70)

# IFF: P1 ↔ P2 (equivalence)
iff_sat = [(True, True), (False, False)]
iff = embed_formula(iff_sat)
iff_comps = get_components(iff)

# XOR: P1 ⊕ P2 (exclusive or)
xor_sat = [(True, False), (False, True)]
xor = embed_formula(xor_sat)
xor_comps = get_components(xor)

print("\n🔵 IFF (P1 ↔ P2): 'Variables must AGREE'")
print(f"  Satisfying: {iff_sat}")
print(f"  Multivector: {iff}")
print(f"  Scalar: {iff_comps[0]:.3f} (50% true)")
print(f"  e12 BIVECTOR: {iff_comps[3]:+.3f} ← STRONGLY POSITIVE!")
print("  → Variables are CORRELATED (tend to be same)")

print("\n🔴 XOR (P1 ⊕ P2): 'Variables must DISAGREE'")
print(f"  Satisfying: {xor_sat}")
print(f"  Multivector: {xor}")
print(f"  Scalar: {xor_comps[0]:.3f} (50% true)")
print(f"  e12 BIVECTOR: {xor_comps[3]:+.3f} ← STRONGLY NEGATIVE!")
print("  → Variables are ANTI-CORRELATED (tend to differ)")

print("\n" + "="*70)
print("💡 THE AHA MOMENT")
print("="*70)
print("SAME PROBABILITY (50%), OPPOSITE GEOMETRIC STRUCTURE!")
print("Boolean logic: 'Both are true half the time.'")
print("Geometric logic: 'These are GEOMETRIC OPPOSITES!'")
print(f"  IFF bivector: {iff_comps[3]:+.3f}")
print(f"  XOR bivector: {xor_comps[3]:+.3f}")
print(f"  They sum to: {iff_comps[3] + xor_comps[3]:.3f} (exactly zero!)")
print("="*70)

# Visualization
fig = plt.figure(figsize=(15, 5))

# Plot 1: Component comparison
ax1 = fig.add_subplot(131)
components = ['scalar', 'e1', 'e2', 'e12']
x = np.arange(len(components))
width = 0.35

bars1 = ax1.bar(x - width/2, iff_comps, width, label='IFF',
               color='blue', alpha=0.7, edgecolor='darkblue', linewidth=2)
bars2 = ax1.bar(x + width/2, xor_comps, width, label='XOR',
               color='red', alpha=0.7, edgecolor='darkred', linewidth=2)
ax1.set_ylabel('Component Value')
ax1.set_xticks(x)
ax1.set_xticklabels(components)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.legend()
ax1.set_title('Component Comparison', fontweight='bold')
ax1.set_ylim(-0.6, 0.6)

# Plot 2: Bivector focus
ax2 = fig.add_subplot(132)
formulas = ['IFF', 'XOR']
bivectors = [iff_comps[3], xor_comps[3]]
colors_biv = ['blue', 'red']
bars = ax2.bar(formulas, bivectors, color=colors_biv, alpha=0.7,
              edgecolor='black', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=2)
ax2.set_ylabel('e12 Bivector Component')
ax2.set_title('The Correlation Story', fontweight='bold')
ax2.set_ylim(-0.6, 0.6)

for bar, val in zip(bars, bivectors):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.05 if val > 0 else -0.05),
            f'{val:+.2f}\n{"AGREE" if val > 0 else "DISAGREE"}',
            ha='center', va='bottom' if val > 0 else 'top',
            fontweight='bold', fontsize=11)

# Plot 3: Truth assignments in sign space
ax3 = fig.add_subplot(133)

# IFF points
iff_points = [(1, 1), (-1, -1)]
for point in iff_points:
    ax3.scatter(*point, s=300, c='blue', alpha=0.7, marker='o',
               edgecolors='darkblue', linewidths=2, label='IFF' if point == iff_points[0] else "")

# XOR points
xor_points = [(1, -1), (-1, 1)]
for point in xor_points:
    ax3.scatter(*point, s=300, c='red', alpha=0.7, marker='s',
               edgecolors='darkred', linewidths=2, label='XOR' if point == xor_points[0] else "")

ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax3.axvline(x=0, color='k', linestyle='-', linewidth=1)
ax3.set_xlabel('P1 (sign)', fontsize=11)
ax3.set_ylabel('P2 (sign)', fontsize=11)
ax3.set_title('Satisfying Assignments\nin Sign Space', fontweight='bold')
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.suptitle('Demo 3: IFF vs XOR - The AHA Moment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('demo3_aha_moment.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: demo3_aha_moment.png")
plt.show()