"""
Demo 4: The Bivector Formula
e12(ι(F)) = (1/4) · Σ sign(p1) · sign(p2)
"""

from r200 import R200, e1, e2, e12
import numpy as np
import matplotlib.pyplot as plt


def get_components(mv):
    return np.array([mv[0], mv[1], mv[2], mv[3]])


def compute_bivector_directly(satisfying_assignments):
    """
    Compute bivector using the formula:
    e12 = (1/4) · Σ sign(p1) · sign(p2)
    """
    total = 0
    contributions = []

    for p1, p2 in satisfying_assignments:
        s1 = 1 if p1 else -1
        s2 = 1 if p2 else -1
        contribution = s1 * s2
        total += contribution
        contributions.append((p1, p2, s1, s2, contribution))

    bivector = total / 4
    return bivector, contributions


def embed_formula(satisfying_assignments):
    result = R200(0, 0)
    for p1, p2 in satisfying_assignments:
        s1 = 1 if p1 else -1
        s2 = 1 if p2 else -1
        factor1 = (R200(1, 0) + R200(s1, 1)) * 0.5
        factor2 = (R200(1, 0) + R200(s2, 2)) * 0.5
        result = result + (factor1 * factor2)
    return result


# Test cases
formulas = {
    'P1 ∧ P2 (AND)': [(True, True)],
    'P1 ∨ P2 (OR)': [(True, True), (True, False), (False, True)],
    'P1 ⊕ P2 (XOR)': [(True, False), (False, True)],
    'P1 ↔ P2 (IFF)': [(True, True), (False, False)],
}

print("=" * 80)
print("DEMO 4: THE BIVECTOR FORMULA IN ACTION")
print("=" * 80)
print("\nFormula: e12(ι(F)) = (1/4) · Σ sign(p1) · sign(p2)")
print("                              (p1,p2)⊨F")
print("=" * 80)

all_bivectors = []
for name, sat_assignments in formulas.items():
    print(f"\n{'=' * 80}")
    print(f"📐 {name}")
    print(f"{'=' * 80}")
    print(f"Satisfying assignments: {sat_assignments}")
    print(f"\nStep-by-step calculation:")

    bivector_direct, contributions = compute_bivector_directly(sat_assignments)

    for p1, p2, s1, s2, contrib in contributions:
        agreement = "AGREE ✓" if contrib > 0 else "DISAGREE ✗"
        print(f"  ({str(p1):5}, {str(p2):5}) → ({s1:+2}, {s2:+2}) → "
              f"{contrib:+2}/4 = {contrib / 4:+.2f}  [{agreement}]")

    total = sum(c[4] for c in contributions)
    print(f"\n  Sum: {' + '.join(f'({c[4]:+d})' for c in contributions)} = {total:+d}")
    print(f"  Bivector e12: {total}/4 = {bivector_direct:+.3f}")

    # Verify with geometric embedding
    embedded = embed_formula(sat_assignments)
    bivector_geometric = get_components(embedded)[3]
    print(f"  Verification (geometric): {bivector_geometric:+.3f}")

    match = "✓ MATCH" if abs(bivector_direct - bivector_geometric) < 1e-10 else "✗ ERROR"
    print(f"  {match}")

    all_bivectors.append((name, bivector_direct, contributions))

    # Interpretation
    if abs(bivector_direct) > 0.4:
        if bivector_direct > 0:
            print(f"  💡 Strong POSITIVE correlation (variables agree often)")
        else:
            print(f"  💡 Strong NEGATIVE correlation (variables disagree often)")
    elif abs(bivector_direct) > 0.1:
        print(f"  💡 Moderate {'positive' if bivector_direct > 0 else 'negative'} correlation")
    else:
        print(f"  💡 No correlation (variables independent)")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Bivector values
formula_names = list(formulas.keys())
bivector_values = [bv for _, bv, _ in all_bivectors]

colors = ['green' if bv > 0.3 else 'red' if bv < -0.3 else 'gray'
          for bv in bivector_values]

bars = ax1.barh(formula_names, bivector_values, color=colors, alpha=0.7,
                edgecolor='black', linewidth=2)
ax1.axvline(x=0, color='k', linestyle='-', linewidth=2)
ax1.set_xlabel('e12 Bivector Component', fontsize=11)
ax1.set_title('Correlation Spectrum', fontsize=12, fontweight='bold')
ax1.set_xlim(-0.6, 0.6)

for bar, val in zip(bars, bivector_values):
    width = bar.get_width()
    ax1.text(width + (0.03 if val > 0 else -0.03), bar.get_y() + bar.get_height() / 2,
             f'{val:+.2f}',
             ha='left' if val > 0 else 'right', va='center',
             fontweight='bold')

# Plot 2: OR breakdown
or_name, or_bv, or_contribs = all_bivectors[1]  # OR is second
assignment_labels = [f"({c[2]:+d},{c[3]:+d})" for c in or_contribs]
contribution_values = [c[4] for c in or_contribs]

colors_contrib = ['green' if v > 0 else 'red' for v in contribution_values]
bars2 = ax2.bar(assignment_labels, contribution_values, color=colors_contrib,
                alpha=0.7, edgecolor='black', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=2)
ax2.set_ylabel('Contribution to Sum', fontsize=11)
ax2.set_title('OR Formula Breakdown', fontsize=12, fontweight='bold')
ax2.set_ylim(-2, 2)

total_contrib = sum(contribution_values)
ax2.axhline(y=total_contrib, color='blue', linestyle='--', linewidth=2,
            label=f'Sum = {total_contrib:+d}')
ax2.text(1, total_contrib + 0.15, f'÷ 4 = {total_contrib / 4:+.2f}',
         ha='center', fontweight='bold', color='blue', fontsize=11)
ax2.legend()

plt.suptitle('Demo 4: The Bivector Formula', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('demo4_bivector_formula.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: demo4_bivector_formula.png")
plt.show()