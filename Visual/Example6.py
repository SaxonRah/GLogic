"""
Demo 6: ALL 16 Binary Boolean Operations
Shows the complete geometric spectrum
"""

from r200 import R200, e1, e2, e12
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


# ALL 16 binary Boolean operations
operations = {
    # Index: (Name, Symbol, Satisfying assignments, Description)
    0: ('Contradiction', '⊥', [], 'Always false'),
    1: ('NOR', '↓', [(False, False)], '¬(P1 ∨ P2)'),
    2: ('P1 AND NOT P2', '>', [(True, False)], 'P1 ∧ ¬P2'),
    3: ('NOT P1', '¬P1', [(False, True), (False, False)], '¬P1'),
    4: ('NOT P1 AND P2', '<', [(False, True)], '¬P1 ∧ P2'),
    5: ('NOT P2', '¬P2', [(True, False), (False, False)], '¬P2'),
    6: ('XOR', '⊕', [(True, False), (False, True)], 'P1 ⊕ P2'),
    7: ('NAND', '↑', [(True, False), (False, True), (False, False)], '¬(P1 ∧ P2)'),
    8: ('AND', '∧', [(True, True)], 'P1 ∧ P2'),
    9: ('IFF', '↔', [(True, True), (False, False)], 'P1 ↔ P2'),
    10: ('P2', 'P2', [(True, True), (False, True)], 'P2'),
    11: ('P1 IMPLIES P2', '→', [(True, True), (False, True), (False, False)], 'P1 → P2'),
    12: ('P1', 'P1', [(True, True), (True, False)], 'P1'),
    13: ('P2 IMPLIES P1', '←', [(True, True), (True, False), (False, False)], 'P2 → P1'),
    14: ('OR', '∨', [(True, True), (True, False), (False, True)], 'P1 ∨ P2'),
    15: ('Tautology', '⊤', [(True, True), (True, False), (False, True), (False, False)], 'Always true'),
}

print("=" * 80)
print("DEMO 6: ALL 16 BINARY BOOLEAN OPERATIONS")
print("=" * 80)
print("Each has a UNIQUE geometric signature in Cl(2,0)")
print("=" * 80)

# Compute all embeddings
results = []
for idx in range(16):
    name, symbol, sat, desc = operations[idx]
    mv = embed_formula(sat)
    comps = get_components(mv)

    results.append({
        'index': idx,
        'name': name,
        'symbol': symbol,
        'sat': sat,
        'desc': desc,
        'mv': mv,
        'scalar': comps[0],
        'e1': comps[1],
        'e2': comps[2],
        'e12': comps[3],
        'probability': comps[0],
        'num_sat': len(sat)
    })

    print(f"\n[{idx:2d}] {name:20} ({symbol:3})")
    print(f"     {desc:30}")
    print(f"     Satisfies: {sat}")
    print(f"     Probability: {comps[0]:.2%} ({len(sat)}/4)")
    print(f"     e12 = {comps[3]:+.3f} ", end="")
    if abs(comps[3]) > 0.4:
        print(f"← {'Strong AGREE' if comps[3] > 0 else 'Strong DISAGREE'}")
    elif abs(comps[3]) > 0.1:
        print(f"← {'Moderate agree' if comps[3] > 0 else 'Moderate disagree'}")
    else:
        print("← Independent")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# Plot 1: All operations - full component breakdown
ax1 = plt.subplot(2, 3, 1)
data = np.array([[r['scalar'], r['e1'], r['e2'], r['e12']] for r in results])
im = ax1.imshow(data, aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
ax1.set_yticks(range(16))
ax1.set_yticklabels([f"{r['index']:2d}: {r['symbol']}" for r in results], fontsize=8)
ax1.set_xticks(range(4))
ax1.set_xticklabels(['scalar', 'e₁', 'e₂', 'e₁₂'], fontsize=10)
ax1.set_title('All 16 Operations - Complete Components', fontweight='bold')
plt.colorbar(im, ax=ax1)

# Plot 2: Bivector spectrum
ax2 = plt.subplot(2, 3, 2)
bivectors = [r['e12'] for r in results]
names = [f"{r['index']:2d}:{r['symbol']}" for r in results]
colors = ['green' if b > 0.3 else 'red' if b < -0.3 else 'gray' for b in bivectors]
bars = ax2.barh(range(16), bivectors, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax2.axvline(x=0, color='k', linestyle='-', linewidth=2)
ax2.set_yticks(range(16))
ax2.set_yticklabels(names, fontsize=8)
ax2.set_xlabel('e₁₂ Bivector (Correlation)', fontsize=10)
ax2.set_title('The Correlation Spectrum', fontweight='bold')
ax2.set_xlim(-0.6, 0.6)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Truth probability vs Correlation
ax3 = plt.subplot(2, 3, 3)
probs = [r['probability'] for r in results]
bivs = [r['e12'] for r in results]
scatter = ax3.scatter(probs, bivs, s=200, c=range(16), cmap='tab20',
                      edgecolors='black', linewidths=2, alpha=0.8)
for r in results:
    ax3.annotate(r['symbol'], (r['probability'], r['e12']),
                 fontsize=8, ha='center', va='center', fontweight='bold')
ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax3.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax3.set_xlabel('Truth Probability (scalar)', fontsize=10)
ax3.set_ylabel('Correlation (e₁₂)', fontsize=10)
ax3.set_title('Truth vs Correlation Space', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Grouped by probability
ax4 = plt.subplot(2, 3, 4)
prob_groups = {}
for r in results:
    p = r['num_sat']
    if p not in prob_groups:
        prob_groups[p] = []
    prob_groups[p].append(r)

x_pos = 0
colors_group = plt.cm.Set3(np.linspace(0, 1, 5))
for num_sat in sorted(prob_groups.keys()):
    group = prob_groups[num_sat]
    bivs = [r['e12'] for r in group]
    labels = [r['symbol'] for r in group]

    x_positions = np.arange(len(group)) + x_pos
    bars = ax4.bar(x_positions, bivs, color=colors_group[num_sat],
                   alpha=0.7, edgecolor='black', linewidth=1,
                   label=f'{num_sat}/4 true')

    for i, (x, label) in enumerate(zip(x_positions, labels)):
        ax4.text(x, -0.55, label, ha='center', fontsize=8, rotation=0)

    x_pos += len(group) + 0.5

ax4.axhline(y=0, color='k', linestyle='-', linewidth=2)
ax4.set_ylabel('e₁₂ Bivector', fontsize=10)
ax4.set_title('Grouped by Truth Probability', fontweight='bold')
ax4.set_ylim(-0.6, 0.6)
ax4.set_xticks([])
ax4.legend(fontsize=8)
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Functional relationships
ax5 = plt.subplot(2, 3, 5)
relationships = [
    ('AND', 'NAND', 'Negation'),
    ('OR', 'NOR', 'Negation'),
    ('XOR', 'IFF', 'Negation'),
    ('AND', 'OR', 'De Morgan'),
    ('P1', 'NOT P1', 'Negation'),
    ('P2', 'NOT P2', 'Negation'),
]

y_pos = 0
for op1_name, op2_name, rel_type in relationships:
    # Find operations
    op1 = next(r for r in results if r['name'] == op1_name)
    op2 = next(r for r in results if r['name'] == op2_name)

    # Plot pair
    ax5.scatter([op1['e12'], op2['e12']], [y_pos, y_pos],
                s=100, c=['blue', 'red'], edgecolors='black', linewidths=2)
    ax5.plot([op1['e12'], op2['e12']], [y_pos, y_pos],
             'k--', alpha=0.3, linewidth=1)

    ax5.text(-0.55, y_pos, f"{op1['symbol']}", ha='right', va='center', fontsize=9)
    ax5.text(0.55, y_pos, f"{op2['symbol']}", ha='left', va='center', fontsize=9)
    ax5.text(0, y_pos + 0.15, rel_type, ha='center', fontsize=7, style='italic')

    y_pos += 1

ax5.axvline(x=0, color='k', linestyle='-', linewidth=1)
ax5.set_xlim(-0.6, 0.6)
ax5.set_xlabel('e₁₂ Bivector', fontsize=10)
ax5.set_ylim(-0.5, y_pos)
ax5.set_yticks([])
ax5.set_title('Boolean Relationships in Geometry', fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# Plot 6: Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Count by bivector sign
positive = sum(1 for r in results if r['e12'] > 0.1)
negative = sum(1 for r in results if r['e12'] < -0.1)
neutral = sum(1 for r in results if abs(r['e12']) <= 0.1)

summary = f"""
ALL 16 BINARY BOOLEAN OPERATIONS
═════════════════════════════════

Total operations: 16

By Correlation:
  Positive (agree):     {positive:2d}
  Negative (disagree):  {negative:2d}
  Neutral (independent): {neutral:2d}

By Probability:
  0/4 (0%):    {len(prob_groups.get(0, [])):2d}
  1/4 (25%):   {len(prob_groups.get(1, [])):2d}
  2/4 (50%):   {len(prob_groups.get(2, [])):2d}
  3/4 (75%):   {len(prob_groups.get(3, [])):2d}
  4/4 (100%):  {len(prob_groups.get(4, [])):2d}

Key Insight:
  Each operation has a UNIQUE
  geometric signature!

  Boolean logic sees only
  probability (scalar).

  Geometric logic sees the
  full correlation structure.

Strongest Correlation:
  IFF: e₁₂ = +0.50 (agree)
  XOR: e₁₂ = -0.50 (disagree)
"""

ax6.text(0.1, 0.5, summary, fontsize=9, family='monospace',
         verticalalignment='center')

plt.suptitle('Demo 6: Complete Boolean Operation Spectrum in Cl(2,0)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('demo6_all_16_operations.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: demo6_all_16_operations.png")
plt.show()

# Print key observations
print("\n" + "=" * 80)
print("KEY OBSERVATIONS")
print("=" * 80)
print("\n1. UNIQUE GEOMETRIC SIGNATURES")
print("   Every operation has a distinct (scalar, e1, e2, e12) tuple.")
print("   No two operations have the same geometric fingerprint!")

print("\n2. CORRELATION SPECTRUM")
print(f"   Most positive:  IFF (↔)  = {results[9]['e12']:+.3f}")
print(f"   Most negative:  XOR (⊕)  = {results[6]['e12']:+.3f}")
print(f"   Exactly zero:   ⊤, ⊥, projections")

print("\n3. BOOLEAN DUALITY")
print("   Negated pairs have same |e12| but opposite scalar:")
for op1_name, op2_name in [('AND', 'NAND'), ('OR', 'NOR'), ('XOR', 'IFF')]:
    op1 = next(r for r in results if r['name'] == op1_name)
    op2 = next(r for r in results if r['name'] == op2_name)
    print(f"   {op1['symbol']:3} vs {op2['symbol']:3}: "
          f"e12 = {op1['e12']:+.2f} vs {op2['e12']:+.2f}, "
          f"scalar = {op1['scalar']:.2f} vs {op2['scalar']:.2f}")

print("\n4. FUNCTIONAL COMPLETENESS vs GEOMETRIC COMPLETENESS")
print("   Boolean: {AND, OR, NOT} is functionally complete")
print("   Geometric: ALL 16 operations have unique geometric meaning")
print("   → Derivation in Boolean logic ≠ geometric identity!")

print("\n" + "=" * 80)