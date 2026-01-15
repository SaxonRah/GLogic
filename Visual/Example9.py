"""
Demo 9: Bivector Rotations
Rotate between formulas using exponential map
This has NO Boolean equivalent!
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


def rotor_exp(bivector_angle):
    """
    Exponential of bivector: e^(θ·e12)
    In Cl(2,0): e12² = -1, so this gives rotation
    e^(θ·e12) = cos(θ) + sin(θ)·e12
    """
    theta = bivector_angle
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    return R200(cos_t, 0) + R200(sin_t, 3)  # cos(θ) + sin(θ)·e12


def sandwich_product(rotor, mv):
    """
    Rotate multivector: R * mv * ~R
    where ~R is reverse of R
    """
    rotated = rotor * mv * ~rotor
    return rotated


print("=" * 70)
print("DEMO 9: BIVECTOR ROTATIONS")
print("=" * 70)
print("Continuously transform one formula into another")
print("via geometric rotation!")
print("=" * 70)

# Start with AND
start_formula = embed_formula([(True, True)])
print(f"\n🔵 Starting formula: AND")
print(f"   {start_formula}")

# Apply rotations
angles = np.linspace(0, np.pi, 21)
trajectory = []

print("\n🔄 Rotating in (scalar, e12) plane:")
print("-" * 70)

for i, angle in enumerate(angles):
    # Create rotor
    rotor = rotor_exp(angle)

    # Apply rotation
    rotated = sandwich_product(rotor, start_formula)
    rotated_comps = get_components(rotated)
    trajectory.append(rotated_comps)

    if i % 5 == 0:  # Print every 5th
        print(f"θ={angle:5.2f}: scalar={rotated_comps[0]:+.3f}, "
              f"e12={rotated_comps[3]:+.3f}")

print("\n💡 This creates a CONTINUOUS PATH through formula space")
print("   that is IMPOSSIBLE in Boolean logic!")
print("   Boolean logic: discrete jumps between 16 operations")
print("   Geometric logic: smooth rotations through infinite states")

# Visualize
fig = plt.figure(figsize=(16, 5))

trajectory = np.array(trajectory)

# Plot 1: All components over angle
ax1 = fig.add_subplot(131)
for i, label in enumerate(['scalar', 'e1', 'e2', 'e12']):
    ax1.plot(angles, trajectory[:, i], 'o-', linewidth=2,
             label=label, markersize=4)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlabel('Rotation Angle θ', fontsize=11)
ax1.set_ylabel('Component Value', fontsize=11)
ax1.set_title('Component Evolution Under Rotation', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Circular trajectory in (scalar, e12) plane
ax2 = fig.add_subplot(132)
ax2.plot(trajectory[:, 0], trajectory[:, 3], 'o-',
         linewidth=2, markersize=6, color='purple')

# Mark start and end
ax2.scatter([trajectory[0, 0]], [trajectory[0, 3]],
            s=300, c='green', marker='o', edgecolors='darkgreen',
            linewidths=3, label='Start (AND)', zorder=5)
ax2.scatter([trajectory[-1, 0]], [trajectory[-1, 3]],
            s=300, c='red', marker='s', edgecolors='darkred',
            linewidths=3, label='End', zorder=5)

# Draw axes
ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)

ax2.set_xlabel('Scalar', fontsize=11)
ax2.set_ylabel('e₁₂', fontsize=11)
ax2.set_title('Rotation Path in Truth-Correlation Space', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# Plot 3: Distance from Boolean cone
ax3 = fig.add_subplot(133)

# For demonstration: compute "distance" from nearest Boolean operation
# (In reality, we'd need to project onto cone, but let's approximate)
boolean_formulas = [
    embed_formula([]),  # Contradiction
    embed_formula([(True, True)]),  # AND
    embed_formula([(True, True), (False, False)]),  # IFF
    embed_formula([(True, True), (True, False), (False, True)]),  # OR
    embed_formula([(True, True), (True, False), (False, True), (False, False)]),  # Tautology
]

min_distances = []
for rot_comps in trajectory:
    distances = [np.linalg.norm(rot_comps - get_components(bf))
                 for bf in boolean_formulas]
    min_distances.append(min(distances))

ax3.plot(angles, min_distances, 'o-', linewidth=2, color='red', markersize=6)
ax3.fill_between(angles, 0, min_distances, alpha=0.3, color='red')
ax3.set_xlabel('Rotation Angle θ', fontsize=11)
ax3.set_ylabel('Min Distance from Boolean Operations', fontsize=11)
ax3.set_title('How Far from Boolean Logic?', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Highlight regions outside Boolean logic
ax3.axhline(y=0.1, color='orange', linestyle='--', linewidth=2,
            label='Threshold')
ax3.legend()

plt.suptitle('Demo 9: Bivector Rotations - Beyond Boolean Logic',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('demo9_rotations.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: demo9_rotations.png")
plt.show()