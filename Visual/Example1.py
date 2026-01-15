"""
Demo 1: The Boolean Circuit Problem
Shows what Boolean logic loses vs Geometric logic
"""

from r200 import R200, e1, e2, e12
import numpy as np
import matplotlib.pyplot as plt


def get_components(mv):
    """Extract components from R200 multivector"""
    return np.array([mv[0], mv[1], mv[2], mv[3]])


def boolean_and_traditional(p1, p2):
    """Traditional Boolean AND - returns single scalar"""
    return p1 and p2


def boolean_and_geometric(p1, p2):
    """Geometric AND - returns full multivector"""
    # Embed Boolean values as multivectors
    # True → (+1), False → (-1) in sign
    p1_sign = 1 if p1 else -1
    p2_sign = 1 if p2 else -1

    # P1: 0.5 + 0.5*e1
    p1_mv = R200(0.5, 0) + R200(0.5 * p1_sign, 1)

    # P2: 0.5 + 0.5*e2
    p2_mv = R200(0.5, 0) + R200(0.5 * p2_sign, 2)

    # Geometric product
    result = p1_mv * p2_mv

    return result


# Test all combinations
print("=" * 60)
print("DEMO 1: TRADITIONAL BOOLEAN AND vs GEOMETRIC AND")
print("=" * 60)

results = []
for p1, p2 in [(True, True), (True, False), (False, True), (False, False)]:
    bool_result = boolean_and_traditional(p1, p2)
    geo_result = boolean_and_geometric(p1, p2)
    comps = get_components(geo_result)

    results.append((p1, p2, bool_result, comps))

    print(f"\nP1={p1}, P2={p2}")
    print(f"  Boolean: {bool_result}")
    print(f"  Geometric: {geo_result}")
    print(f"    Scalar: {comps[0]:.3f}")
    print(f"    e1: {comps[1]:.3f}")
    print(f"    e2: {comps[2]:.3f}")
    print(f"    e12: {comps[3]:.3f} ← The bivector (CORRELATION)!")

print("\n" + "=" * 60)
print("💡 KEY INSIGHT")
print("=" * 60)
print("Boolean logic only gives you the scalar (true/false).")
print("Geometric logic gives you the FULL structure:")
print("  • Scalar: Truth value")
print("  • e1, e2: Variable biases")
print("  • e12: CORRELATION between variables")
print("=" * 60)

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, (p1, p2, bool_res, comps) in enumerate(results):
    ax = axes[idx]
    labels = ['scalar', 'e1', 'e2', 'e12']
    colors = ['blue', 'green', 'green', 'red']

    bars = ax.bar(labels, comps, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_title(f'({p1}, {p2})\nBoolean: {bool_res}', fontweight='bold')
    ax.set_ylabel('Component Value')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.set_ylim(-0.1, 0.6)

    # Highlight e12
    bars[3].set_edgecolor('red')
    bars[3].set_linewidth(3)

plt.suptitle('Demo 1: Boolean Logic Only Sees Scalar, GLogic Sees Everything',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('demo1_opening.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: demo1_opening.png")
plt.show()