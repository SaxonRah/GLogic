"""
Demo 14: Visual Comparison - Boolean vs Geometric Operations
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


# Create test formulas
AND = embed_formula([(True, True)])
OR = embed_formula([(True, True), (True, False), (False, True)])

# Compute all operations
operations = {
    'Original AND': get_components(AND),
    'Original OR': get_components(OR),
    'A * B (Geometric)': get_components(AND * OR),
    'A ^ B (Outer)': get_components(AND ^ OR),
    'A | B (Inner)': get_components(AND | OR),
    '~A (Reverse)': get_components(~AND),
    '!A (Dual)': get_components(AND.Dual()),
    'A.Conjugate()': get_components(AND.Conjugate()),
    'A*B*~A (Sandwich)': get_components(AND * OR * ~AND),
}

# Visualize
fig = plt.figure(figsize=(16, 10))

# Plot 1: Component heatmap
ax1 = plt.subplot(2, 2, 1)
data = np.array([v for v in operations.values()])
im = ax1.imshow(data, aspect='auto', cmap='RdBu', vmin=-0.8, vmax=0.8)
ax1.set_yticks(range(len(operations)))
ax1.set_yticklabels(operations.keys(), fontsize=9)
ax1.set_xticks(range(4))
ax1.set_xticklabels(['scalar', 'e₁', 'e₂', 'e₁₂'], fontsize=10)
ax1.set_title('All Operations - Component View', fontweight='bold')
plt.colorbar(im, ax=ax1)

# Add values
for i, (name, comps) in enumerate(operations.items()):
    for j in range(4):
        color = 'white' if abs(comps[j]) > 0.4 else 'black'
        ax1.text(j, i, f'{comps[j]:.2f}', ha='center', va='center',
                 color=color, fontsize=8, fontweight='bold')

# Plot 2: Truth probability comparison
ax2 = plt.subplot(2, 2, 2)
names = list(operations.keys())
scalars = [v[0] for v in operations.values()]
colors = ['blue' if 'Original' in n else 'green' for n in names]
bars = ax2.barh(range(len(names)), scalars, color=colors, alpha=0.7,
                edgecolor='black', linewidth=1)
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels(names, fontsize=9)
ax2.set_xlabel('Truth Probability (scalar)', fontsize=10)
ax2.set_title('Scalar Component: "How True?"', fontweight='bold')
ax2.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Correlation comparison
ax3 = plt.subplot(2, 2, 3)
bivectors = [v[3] for v in operations.values()]
colors = ['red' if b < -0.1 else 'green' if b > 0.1 else 'gray'
          for b in bivectors]
bars = ax3.barh(range(len(names)), bivectors, color=colors, alpha=0.7,
                edgecolor='black', linewidth=1)
ax3.set_yticks(range(len(names)))
ax3.set_yticklabels(names, fontsize=9)
ax3.set_xlabel('Correlation (e₁₂)', fontsize=10)
ax3.set_title('Bivector Component: "How Correlated?"', fontweight='bold')
ax3.axvline(x=0, color='k', linestyle='-', linewidth=2)
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Scatter plot in (scalar, e12) space
ax4 = plt.subplot(2, 2, 4)
for i, (name, comps) in enumerate(operations.items()):
    if 'Original' in name:
        marker = '*'
        size = 300
        color = 'red' if 'AND' in name else 'blue'
    else:
        marker = 'o'
        size = 150
        color = plt.cm.tab10(i / len(operations))

    ax4.scatter(comps[0], comps[3], s=size, marker=marker,
                color=color, edgecolors='black', linewidths=2,
                label=name, alpha=0.8)

ax4.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax4.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax4.set_xlabel('Scalar (Truth)', fontsize=11)
ax4.set_ylabel('e₁₂ (Correlation)', fontsize=11)
ax4.set_title('Formula Space Visualization', fontweight='bold')
ax4.legend(fontsize=7, loc='best')
ax4.grid(True, alpha=0.3)

plt.suptitle('Geometric Operations: Component Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('demo14_operation_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: demo14_operation_comparison.png")
plt.show()