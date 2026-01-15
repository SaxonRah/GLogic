"""
Demo 8: Fuzzy Logic via Geometric Interpolation
Boolean logic: discrete {T, F}
Geometric logic: continuous [0, 1] with correlation structure
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

print("="*70)
print("DEMO 8: FUZZY LOGIC - BEYOND BOOLEAN")
print("="*70)

# Boolean endpoints
AND = embed_formula([(True, True)])
OR = embed_formula([(True, True), (True, False), (False, True)])

and_comps = get_components(AND)
or_comps = get_components(OR)

print("\n📐 Boolean Endpoints:")
print(f"AND: {AND}")
print(f"  Scalar: {and_comps[0]:.3f}, e12: {and_comps[3]:+.3f}")
print(f"OR:  {OR}")
print(f"  Scalar: {or_comps[0]:.3f}, e12: {or_comps[3]:+.3f}")

# Create CONTINUOUS interpolation
print("\n🌈 Continuous Interpolation (NOT in Boolean logic!):")
print("-"*70)

interpolated = []
for t in np.linspace(0, 1, 11):
    # FIX: Convert numpy scalars to Python float
    t = float(t)

    # Geometric interpolation: (1-t)*AND + t*OR
    # Need to use scalar multiplication properly
    interp_mv = AND * (1 - t) + OR * t
    interp_comps = get_components(interp_mv)
    interpolated.append(interp_comps)

    # Check if in Boolean cone
    # Only endpoints are actual Boolean operations
    in_boolean = (abs(t) < 1e-10 or abs(t - 1) < 1e-10)

    print(f"t={t:.1f}: scalar={interp_comps[0]:.3f}, e12={interp_comps[3]:+.3f}  "
          f"{'[Boolean]' if in_boolean else '[FUZZY - not Boolean!]'}")

print("\n💡 KEY INSIGHT:")
print("Boolean logic has only 16 discrete operations.")
print("Geometric logic has INFINITE operations via interpolation!")
print("These intermediate states have NO Boolean equivalent.")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Interpolation path
interpolated = np.array(interpolated)
t_values = np.linspace(0, 1, 11)

for i, label in enumerate(['scalar', 'e1', 'e2', 'e12']):
    ax1.plot(t_values, interpolated[:, i], 'o-', linewidth=2,
            markersize=8, label=label, alpha=0.7)

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlabel('Interpolation parameter t', fontsize=11)
ax1.set_ylabel('Component Value', fontsize=11)
ax1.set_title('Continuous Path from AND to OR\n(Most points NOT in Boolean logic!)',
             fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Highlight Boolean points
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax1.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax1.text(0, 0.8, 'AND\n(Boolean)', ha='center', fontweight='bold', color='red')
ax1.text(1, 0.8, 'OR\n(Boolean)', ha='center', fontweight='bold', color='red')

# Plot 2: Scalar vs e12 trajectory
ax2.plot(interpolated[:, 0], interpolated[:, 3], 'o-', linewidth=3,
        markersize=10, color='purple', alpha=0.7)

# Mark endpoints
ax2.scatter([and_comps[0]], [and_comps[3]], s=300, c='red',
           marker='*', edgecolors='darkred', linewidths=3,
           label='AND (Boolean)', zorder=5)
ax2.scatter([or_comps[0]], [or_comps[3]], s=300, c='blue',
           marker='*', edgecolors='darkblue', linewidths=3,
           label='OR (Boolean)', zorder=5)

# Mark some intermediate points
for i in [3, 5, 7]:  # t=0.3, 0.5, 0.7
    ax2.scatter([interpolated[i, 0]], [interpolated[i, 3]],
               s=200, c='orange', marker='o', edgecolors='black',
               linewidths=2, zorder=4)
    ax2.annotate(f't={i/10:.1f}\n(Fuzzy)',
                (interpolated[i, 0], interpolated[i, 3]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, fontweight='bold')

ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax2.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax2.set_xlabel('Scalar (Truth Probability)', fontsize=11)
ax2.set_ylabel('e₁₂ (Correlation)', fontsize=11)
ax2.set_title('Truth-Correlation Space\n(Continuous, not Discrete!)',
             fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Demo 8: Fuzzy Logic via Geometric Interpolation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('demo8_fuzzy_logic.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: demo8_fuzzy_logic.png")
plt.show()