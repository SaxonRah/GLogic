"""
Demo 5: The Geometric Product
Shows that P1 · P2 = P1 ∧ P2 for independent variables
"""

from r200 import R200, e1, e2, e12
import numpy as np
import matplotlib.pyplot as plt

def get_components(mv):
    return np.array([mv[0], mv[1], mv[2], mv[3]])

def embed_variable(var_index, is_true=True):
    """Embed single variable. var_index: 0 for P1, 1 for P2"""
    sign = 1 if is_true else -1
    if var_index == 0:
        return R200(0.5, 0) + R200(0.5 * sign, 1)  # 0.5 + 0.5*e1
    else:
        return R200(0.5, 0) + R200(0.5 * sign, 2)  # 0.5 + 0.5*e2

def embed_formula(satisfying_assignments):
    result = R200(0, 0)
    for p1, p2 in satisfying_assignments:
        s1 = 1 if p1 else -1
        s2 = 1 if p2 else -1
        factor1 = (R200(1, 0) + R200(s1, 1)) * 0.5
        factor2 = (R200(1, 0) + R200(s2, 2)) * 0.5
        result = result + (factor1 * factor2)
    return result

print("="*80)
print("DEMO 5: THE GEOMETRIC PRODUCT")
print("="*80)
print("Computing AND via Multiplication")
print("="*80)

# Embed P1 and P2 as independent variables
P1 = embed_variable(0, is_true=True)
P2 = embed_variable(1, is_true=True)

p1_comps = get_components(P1)
p2_comps = get_components(P2)

print("\n📦 Embedded Variables:")
print(f"ι(P1) = {P1}")
print(f"      = {p1_comps[0]:.3f}·1 + {p1_comps[1]:.3f}·e1 + {p1_comps[2]:.3f}·e2 + {p1_comps[3]:.3f}·e12")
print(f"ι(P2) = {P2}")
print(f"      = {p2_comps[0]:.3f}·1 + {p2_comps[1]:.3f}·e1 + {p2_comps[2]:.3f}·e2 + {p2_comps[3]:.3f}·e12")

# Compute geometric product
P1_gp_P2 = P1 * P2
gp_comps = get_components(P1_gp_P2)

print("\n⚙️  Geometric Product: ι(P1) · ι(P2)")
print(f"Result = {P1_gp_P2}")
print(f"       = {gp_comps[0]:.3f}·1 + {gp_comps[1]:.3f}·e1 + {gp_comps[2]:.3f}·e2 + {gp_comps[3]:.3f}·e12")

# Compare to Boolean AND embedding
P1_and_P2 = embed_formula([(True, True)])
and_comps = get_components(P1_and_P2)

print("\n📐 Boolean AND: ι(P1 ∧ P2)")
print(f"Result = {P1_and_P2}")
print(f"       = {and_comps[0]:.3f}·1 + {and_comps[1]:.3f}·e1 + {and_comps[2]:.3f}·e2 + {and_comps[3]:.3f}·e12")

# Check if they match
matches = np.allclose(gp_comps, and_comps)

print("\n" + "="*80)
if matches:
    print("✅ THEY MATCH PERFECTLY!")
    print("The geometric product NATURALLY computes Boolean AND")
    print("for independent variables!")
else:
    print("❌ They don't match")
    print(f"Difference: {gp_comps - and_comps}")
print("="*80)

# Test non-commutativity
P2_gp_P1 = P2 * P1
p2_gp_p1_comps = get_components(P2_gp_P1)

print("\n🔄 NON-COMMUTATIVITY")
print(f"P1 · P2 bivector: {gp_comps[3]:+.3f}")
print(f"P2 · P1 bivector: {p2_gp_p1_comps[3]:+.3f}")

if abs(gp_comps[3] + p2_gp_p1_comps[3]) < 1e-10:
    print("\n✅ The bivectors have OPPOSITE SIGNS!")
    print("This is the geometric product's non-commutativity.")
    print("The bivector encodes ORIENTED correlation.")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

components = ['scalar', 'e1', 'e2', 'e12']

# Plot 1: Input P1
ax = axes[0, 0]
bars = ax.bar(components, p1_comps, color=['blue', 'orange', 'gray', 'gray'],
             alpha=0.7, edgecolor='black', linewidth=2)
ax.set_title('Input: ι(P1)', fontweight='bold')
ax.set_ylabel('Component Value')
ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.set_ylim(-0.1, 0.6)

# Plot 2: Input P2
ax = axes[0, 1]
bars = ax.bar(components, p2_comps, color=['blue', 'gray', 'orange', 'gray'],
             alpha=0.7, edgecolor='black', linewidth=2)
ax.set_title('Input: ι(P2)', fontweight='bold')
ax.set_ylabel('Component Value')
ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.set_ylim(-0.1, 0.6)

# Plot 3: Geometric Product
ax = axes[1, 0]
bars = ax.bar(components, gp_comps, color=['purple', 'purple', 'purple', 'red'],
             alpha=0.7, edgecolor='black', linewidth=2)
ax.set_title('Output: ι(P1) · ι(P2) [Geometric Product]', fontweight='bold')
ax.set_ylabel('Component Value')
ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.set_ylim(-0.1, 0.3)
bars[3].set_edgecolor('red')
bars[3].set_linewidth(3)
ax.text(3, gp_comps[3] + 0.02, 'Bivector!', ha='center', fontweight='bold', color='red')

# Plot 4: Comparison
ax = axes[1, 1]
x = np.arange(len(components))
width = 0.35
bars1 = ax.bar(x - width/2, gp_comps, width, label='Geometric Product',
              color='red', alpha=0.7, edgecolor='darkred', linewidth=2)
bars2 = ax.bar(x + width/2, and_comps, width, label='Boolean AND',
              color='blue', alpha=0.7, edgecolor='darkblue', linewidth=2)
ax.set_ylabel('Component Value')
ax.set_xticks(x)
ax.set_xticklabels(components)
ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.legend()
ax.set_title('Comparison: PERFECT MATCH!', fontweight='bold')
ax.set_ylim(-0.1, 0.3)

for i in range(len(components)):
    if abs(gp_comps[i] - and_comps[i]) < 1e-10:
        ax.text(i, max(gp_comps[i], and_comps[i]) + 0.01, '✓',
                ha='center', fontsize=20, color='green', fontweight='bold')

plt.suptitle('Demo 5: The Geometric Product Computes AND', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('demo5_geometric_product.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: demo5_geometric_product.png")
plt.show()