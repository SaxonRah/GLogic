"""
Demo 11: Geometric Mean of Boolean Formulas
Combine formulas in ways Boolean logic cannot!
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


def normalize(mv):
    """Normalize multivector (make it unit magnitude)"""
    comps = get_components(mv)
    mag = np.linalg.norm(comps)
    if mag > 1e-10:
        return R200.fromarray(comps / mag)
    return mv


def geometric_mean(mv1, mv2):
    """
    Geometric mean of two multivectors.
    This creates a formula "between" them in a geometric sense.
    """
    # Normalize both
    n1 = normalize(mv1)
    n2 = normalize(mv2)

    # Geometric mean: sqrt(n1 * n2)
    # Approximate: (n1 + n2) / |n1 + n2|
    sum_mv = n1 + n2
    return normalize(sum_mv)


print("=" * 70)
print("DEMO 11: GEOMETRIC MEAN OF FORMULAS")
print("=" * 70)

# Two very different formulas
AND = embed_formula([(True, True)])
XOR = embed_formula([(True, False), (False, True)])

and_comps = get_components(AND)
xor_comps = get_components(XOR)

print("\n📐 Input Formulas:")
print(f"AND: scalar={and_comps[0]:.3f}, e12={and_comps[3]:+.3f}")
print(f"XOR: scalar={xor_comps[0]:.3f}, e12={xor_comps[3]:+.3f}")

# Arithmetic mean (simple average)
arith_mean = (AND + XOR) * 0.5
arith_comps = get_components(arith_mean)

print(f"\n📊 Arithmetic Mean: (AND + XOR) / 2")
print(f"    scalar={arith_comps[0]:.3f}, e12={arith_comps[3]:+.3f}")
print(f"    This is just: 'true 37.5% of the time'")

# Geometric mean
geom_mean = geometric_mean(AND, XOR)
geom_comps = get_components(geom_mean)

print(f"\n🎯 Geometric Mean: sqrt(AND * XOR)")
print(f"    scalar={geom_comps[0]:.3f}, e12={geom_comps[3]:+.3f}")
print(f"    This balances BOTH probability AND correlation!")

print("\n💡 KEY INSIGHT:")
print("Boolean logic has only arithmetic combinations:")
print("  - AND")
print("  - OR")
print("  - NOT")
print("\nGeometric logic has GEOMETRIC combinations:")
print("  - Interpolation")
print("  - Rotation")
print("  - Geometric mean")
print("  - Projection")
print("\nThese create NEW logical operations not in Boolean logic!")

# Visualize the space
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Component comparison
formulas_comp = ['AND', 'XOR', 'Arith Mean', 'Geom Mean']
components_data = np.array([and_comps, xor_comps, arith_comps, geom_comps])

x = np.arange(4)  # scalar, e1, e2, e12
width = 0.2

for i, name in enumerate(formulas_comp):
    ax1.bar(x + i * width, components_data[i], width, label=name, alpha=0.7)

ax1.set_ylabel('Component Value')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(['scalar', 'e1', 'e2', 'e12'])
ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.legend()
ax1.set_title('Component Comparison', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Position in (scalar, e12) space
ax2.scatter([and_comps[0]], [and_comps[3]], s=300, c='blue',
            marker='o', edgecolors='darkblue', linewidths=3, label='AND')
ax2.scatter([xor_comps[0]], [xor_comps[3]], s=300, c='red',
            marker='s', edgecolors='darkred', linewidths=3, label='XOR')
ax2.scatter([arith_comps[0]], [arith_comps[3]], s=250, c='green',
            marker='^', edgecolors='darkgreen', linewidths=2, label='Arithmetic Mean')
ax2.scatter([geom_comps[0]], [geom_comps[3]], s=250, c='purple',
            marker='D', edgecolors='purple', linewidths=2, label='Geometric Mean')

# Draw lines
ax2.plot([and_comps[0], arith_comps[0]], [and_comps[3], arith_comps[3]],
         'g--', alpha=0.5, linewidth=2)
ax2.plot([xor_comps[0], arith_comps[0]], [xor_comps[3], arith_comps[3]],
         'g--', alpha=0.5, linewidth=2)

ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax2.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax2.set_xlabel('Scalar (Truth Probability)', fontsize=11)
ax2.set_ylabel('e₁₂ (Correlation)', fontsize=11)
ax2.set_title('Geometric Combinations in Formula Space', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Demo 11: Geometric Mean - Beyond Boolean Combinations',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('demo11_geometric_mean.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: demo11_geometric_mean.png")
plt.show()