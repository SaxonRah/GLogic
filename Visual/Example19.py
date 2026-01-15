"""
Demo 19: The Grand Finale - Animated Boolean Cone Journey
A complete visualization of all 16 operations and their transformations
"""

from r200 import R200
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import matplotlib.patches as mpatches

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

print("="*80)
print("THE GRAND FINALE: ANIMATING THE BOOLEAN CONE")
print("="*80)
print("\nGenerating comprehensive animation...")
print("This will take a moment...")
print("="*80)

# =============================================================================
# Generate all data
# =============================================================================

# All 16 Boolean operations
all_operations = {
    0:  ('⊥', [], 'Contradiction'),
    1:  ('NOR', [(False, False)], 'Neither'),
    2:  ('P1∧¬P2', [(True, False)], 'P1 not P2'),
    3:  ('¬P1', [(False, True), (False, False)], 'Not P1'),
    4:  ('¬P1∧P2', [(False, True)], 'P2 not P1'),
    5:  ('¬P2', [(True, False), (False, False)], 'Not P2'),
    6:  ('XOR', [(True, False), (False, True)], 'Exclusive'),
    7:  ('NAND', [(True, False), (False, True), (False, False)], 'Not both'),
    8:  ('AND', [(True, True)], 'Both'),
    9:  ('IFF', [(True, True), (False, False)], 'Equivalent'),
    10: ('P2', [(True, True), (False, True)], 'P2'),
    11: ('P1→P2', [(True, True), (False, True), (False, False)], 'If P1 then P2'),
    12: ('P1', [(True, True), (True, False)], 'P1'),
    13: ('P2→P1', [(True, True), (True, False), (False, False)], 'If P2 then P1'),
    14: ('OR', [(True, True), (True, False), (False, True)], 'Either'),
    15: ('⊤', [(True, True), (True, False), (False, True), (False, False)], 'Tautology'),
}

# Embed all operations
operations_data = {}
for idx, (symbol, truth_table, desc) in all_operations.items():
    mv = embed_formula(truth_table)
    comps = get_components(mv)
    operations_data[idx] = {
        'symbol': symbol,
        'desc': desc,
        'truth_table': truth_table,
        'components': comps,
        'mv': mv
    }

# Cone generators (4 quasi-projectors)
generators = {}
for p1, p2 in [(True, True), (True, False), (False, True), (False, False)]:
    s1 = 1 if p1 else -1
    s2 = 1 if p2 else -1
    factor1 = (R200(1, 0) + R200(s1, 1)) * 0.5
    factor2 = (R200(1, 0) + R200(s2, 2)) * 0.5
    gen = factor1 * factor2
    generators[(p1, p2)] = get_components(gen)

# Interesting transformation paths
transformation_paths = [
    ('Journey 1: Contradiction → Tautology', [0, 1, 7, 14, 15]),
    ('Journey 2: AND → OR (via XOR)', [8, 6, 14]),
    ('Journey 3: XOR → IFF (opposites)', [6, 9]),
    ('Journey 4: The Correlation Spectrum', [9, 8, 14, 6]),  # IFF → AND → OR → XOR
    ('Journey 5: Single Variable Projections', [3, 5, 12, 10]),
]

print(f"\n✅ Embedded all {len(operations_data)} Boolean operations")
print(f"✅ Generated {len(generators)} cone generators")
print(f"✅ Prepared {len(transformation_paths)} transformation journeys")

# =============================================================================
# Animation 1: Static Overview - The Full Cone
# =============================================================================

print("\n" + "="*80)
print("CREATING ANIMATION 1: The Boolean Cone Overview")
print("="*80)

fig = plt.figure(figsize=(18, 10))

# Main 3D view
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.set_title('Boolean Cone in 3D\n(scalar, e₁, e₂)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Scalar (Truth)', fontsize=10)
ax1.set_ylabel('e₁ (P₁ bias)', fontsize=10)
ax1.set_zlabel('e₂ (P₂ bias)', fontsize=10)

# Draw cone generators
gen_array = np.array(list(generators.values()))
ax1.scatter(gen_array[:, 0], gen_array[:, 1], gen_array[:, 2],
           s=300, c='red', marker='*', edgecolors='darkred',
           linewidths=3, label='Cone Generators', zorder=10)

# Draw cone edges
for i, j in combinations(range(4), 2):
    pts = gen_array[[i, j], :3]
    ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            'r-', alpha=0.3, linewidth=2)

# Plot all 16 operations
for idx, data in operations_data.items():
    comps = data['components']

    # Color by correlation
    if comps[3] > 0.3:
        color = 'green'
    elif comps[3] < -0.3:
        color = 'blue'
    else:
        color = 'gray'

    ax1.scatter(comps[0], comps[1], comps[2],
               s=150, c=color, marker='o',
               edgecolors='black', linewidths=1.5, alpha=0.8)

    # Label key operations
    if idx in [0, 8, 14, 15, 6, 9]:
        ax1.text(comps[0], comps[1], comps[2],
                f'  {data["symbol"]}', fontsize=8)

# 2D view: Truth vs Correlation
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_title('Truth-Correlation Space\n(scalar, e₁₂)', fontweight='bold', fontsize=12)
ax2.set_xlabel('Scalar (Truth Probability)', fontsize=10)
ax2.set_ylabel('e₁₂ (Correlation)', fontsize=10)

for idx, data in operations_data.items():
    comps = data['components']

    if comps[3] > 0.3:
        color = 'green'
    elif comps[3] < -0.3:
        color = 'blue'
    else:
        color = 'gray'

    ax2.scatter(comps[0], comps[3], s=150, c=color,
               marker='o', edgecolors='black', linewidths=1.5, alpha=0.8)

    # Label all operations
    ax2.annotate(data['symbol'], (comps[0], comps[3]),
                fontsize=7, ha='center', va='bottom')

ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax2.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax2.grid(True, alpha=0.3)

# Legend for colors
green_patch = mpatches.Patch(color='green', label='Positive Correlation')
blue_patch = mpatches.Patch(color='blue', label='Negative Correlation')
gray_patch = mpatches.Patch(color='gray', label='Independent')
ax2.legend(handles=[green_patch, blue_patch, gray_patch], fontsize=8)

# Component heatmap
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_title('All Operations - Components', fontweight='bold', fontsize=12)

data_matrix = np.array([data['components'] for data in operations_data.values()])
im = ax3.imshow(data_matrix, aspect='auto', cmap='RdBu', vmin=-0.6, vmax=0.6)
ax3.set_yticks(range(16))
ax3.set_yticklabels([operations_data[i]['symbol'] for i in range(16)], fontsize=8)
ax3.set_xticks(range(4))
ax3.set_xticklabels(['scalar', 'e₁', 'e₂', 'e₁₂'], fontsize=10)
plt.colorbar(im, ax=ax3)

# Correlation spectrum
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_title('Correlation Spectrum', fontweight='bold', fontsize=12)

sorted_ops = sorted(operations_data.items(), key=lambda x: x[1]['components'][3])
bivectors = [data['components'][3] for _, data in sorted_ops]
symbols = [data['symbol'] for _, data in sorted_ops]

colors = ['green' if b > 0.1 else 'blue' if b < -0.1 else 'gray' for b in bivectors]
# FIX: edgecolors -> edgecolor
ax4.barh(range(16), bivectors, color=colors, alpha=0.7, edgecolor='black')
ax4.set_yticks(range(16))
ax4.set_yticklabels(symbols, fontsize=8)
ax4.set_xlabel('e₁₂ (Correlation)', fontsize=10)
ax4.axvline(x=0, color='k', linestyle='-', linewidth=2)
ax4.grid(axis='x', alpha=0.3)

# Probability distribution
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_title('Truth Probability Distribution', fontweight='bold', fontsize=12)

prob_groups = {}
for idx, data in operations_data.items():
    num_sat = len(data['truth_table'])
    if num_sat not in prob_groups:
        prob_groups[num_sat] = []
    prob_groups[num_sat].append(data['symbol'])

x_pos = 0
colors_dist = plt.cm.Set3(np.linspace(0, 1, 5))
for num_sat in sorted(prob_groups.keys()):
    symbols = prob_groups[num_sat]
    height = len(symbols)

    ax5.barh(x_pos, 1, height=0.8, color=colors_dist[num_sat],
            alpha=0.7, edgecolor='black', label=f'{num_sat}/4 true')

    # Add symbols
    for i, sym in enumerate(symbols):
        ax5.text(0.5, x_pos + i/(height+1), sym,
                ha='center', va='center', fontsize=9, fontweight='bold')

    x_pos += 1

ax5.set_yticks([])
ax5.set_xlabel('Operations', fontsize=10)
ax5.legend(fontsize=8, loc='upper right')

# Summary info
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

summary = f"""
THE BOOLEAN CONE
════════════════

Total Operations: 16
Cone Generators: 4
Dimension: 4 (scalar, e₁, e₂, e₁₂)

By Correlation:
  Positive (agree): {sum(1 for d in operations_data.values() if d['components'][3] > 0.1)}
  Negative (disagree): {sum(1 for d in operations_data.values() if d['components'][3] < -0.1)}
  Independent: {sum(1 for d in operations_data.values() if abs(d['components'][3]) <= 0.1)}

Extreme Points:
  Most positive: IFF (e₁₂ = +0.50)
  Most negative: XOR (e₁₂ = -0.50)
  Always true: ⊤ (scalar = 1.00)
  Never true: ⊥ (scalar = 0.00)

Key Insight:
  These 16 points are just
  SAMPLES of an infinite
  continuous space!
"""

ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
        verticalalignment='center')

plt.suptitle('The Boolean Cone: All 16 Binary Operations',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('demo19_boolean_cone_overview.png', dpi=150, bbox_inches='tight')
print("✅ Saved: demo19_boolean_cone_overview.png")

# =============================================================================
# Animation 2: Transformation Journeys
# =============================================================================

print("\n" + "="*80)
print("CREATING ANIMATION 2: Transformation Journeys")
print("="*80)

fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for journey_idx, (title, path) in enumerate(transformation_paths):
    if journey_idx >= 6:
        break

    ax = axes[journey_idx]
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('Scalar (Truth)', fontsize=9)
    ax.set_ylabel('e₁₂ (Correlation)', fontsize=9)

    # Draw all operations faintly
    for idx, data in operations_data.items():
        comps = data['components']
        ax.scatter(comps[0], comps[3], s=50, c='lightgray',
                  marker='o', alpha=0.3)

    # Highlight path
    path_points = []
    for op_idx in path:
        comps = operations_data[op_idx]['components']
        path_points.append([comps[0], comps[3]])

    path_points = np.array(path_points)

    # Draw path
    ax.plot(path_points[:, 0], path_points[:, 1],
           'b-', linewidth=3, alpha=0.5, zorder=1)

    # Draw points on path
    for i, op_idx in enumerate(path):
        data = operations_data[op_idx]
        comps = data['components']

        # Color by position in path
        color = plt.cm.viridis(i / (len(path) - 1))

        ax.scatter(comps[0], comps[3], s=300, c=[color],
                  marker='o', edgecolors='black', linewidths=2,
                  zorder=5)

        # Label
        ax.annotate(f"{i+1}. {data['symbol']}",
                   (comps[0], comps[3]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=color, alpha=0.7))

    # Generate smooth interpolation
    n_frames = 50
    smooth_path = []

    for i in range(len(path) - 1):
        start_idx = path[i]
        end_idx = path[i + 1]

        start_mv = operations_data[start_idx]['mv']
        end_mv = operations_data[end_idx]['mv']

        for t in np.linspace(0, 1, n_frames // (len(path) - 1)):
            t = float(t)
            interp = start_mv * (1 - t) + end_mv * t
            comps = get_components(interp)
            smooth_path.append([comps[0], comps[3]])

    smooth_path = np.array(smooth_path)

    # Draw smooth path
    ax.plot(smooth_path[:, 0], smooth_path[:, 1],
           'r--', linewidth=1, alpha=0.3, zorder=2,
           label='Smooth interpolation')

    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

# Hide unused subplots
for i in range(len(transformation_paths), 6):
    axes[i].axis('off')

plt.suptitle('Transformation Journeys: Smooth Paths Between Operations',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('demo19_transformation_journeys.png', dpi=150, bbox_inches='tight')
print("✅ Saved: demo19_transformation_journeys.png")

# =============================================================================
# Animation 3: Animated Tour (GIF)
# =============================================================================

print("\n" + "="*80)
print("CREATING ANIMATION 3: Animated Tour (this will take a minute...)")
print("="*80)

fig3 = plt.figure(figsize=(16, 8))

def animate_tour(frame):
    """Animation function for the tour"""
    fig3.clear()

    # Determine which journey and position
    total_frames = 300
    journey_idx = (frame // 60) % len(transformation_paths)
    journey_title, path = transformation_paths[journey_idx]

    # Position within journey (0 to 1)
    journey_frame = (frame % 60) / 59.0

    # Determine which segment of the path
    n_segments = len(path) - 1
    segment_idx = min(int(journey_frame * n_segments), n_segments - 1)
    segment_t = (journey_frame * n_segments) - segment_idx

    # Get start and end of current segment
    start_idx = path[segment_idx]
    end_idx = path[segment_idx + 1] if segment_idx + 1 < len(path) else start_idx

    # Interpolate
    start_mv = operations_data[start_idx]['mv']
    end_mv = operations_data[end_idx]['mv']

    segment_t = float(segment_t)  # FIX: Ensure float
    current_mv = start_mv * (1 - segment_t) + end_mv * segment_t
    current_comps = get_components(current_mv)

    # Create two subplots
    ax1 = fig3.add_subplot(1, 2, 1)
    ax2 = fig3.add_subplot(1, 2, 2, projection='3d')

    # Plot 1: 2D Truth-Correlation view
    ax1.set_title(f'{journey_title}\nFrame {frame+1}/{total_frames}',
                 fontweight='bold', fontsize=12)
    ax1.set_xlabel('Scalar (Truth Probability)', fontsize=11)
    ax1.set_ylabel('e₁₂ (Correlation)', fontsize=11)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.6, 0.6)

    # Draw all operations faintly
    for idx, data in operations_data.items():
        comps = data['components']
        ax1.scatter(comps[0], comps[3], s=80, c='lightgray',
                   marker='o', alpha=0.3)

    # Draw the path
    for i, op_idx in enumerate(path):
        data = operations_data[op_idx]
        comps = data['components']

        if op_idx == start_idx:
            # Current starting point
            ax1.scatter(comps[0], comps[3], s=400, c='green',
                       marker='o', edgecolors='darkgreen', linewidths=3,
                       label='Start', zorder=10)
        elif op_idx == end_idx:
            # Current target
            ax1.scatter(comps[0], comps[3], s=400, c='red',
                       marker='s', edgecolors='darkred', linewidths=3,
                       label='Target', zorder=10)
        else:
            # Other waypoints
            ax1.scatter(comps[0], comps[3], s=200, c='blue',
                       marker='o', edgecolors='darkblue', linewidths=2,
                       alpha=0.5, zorder=5)

        ax1.annotate(data['symbol'], (comps[0], comps[3]),
                    fontsize=9, ha='center', va='bottom')

    # Draw lines between waypoints
    for i in range(len(path) - 1):
        start = operations_data[path[i]]['components']
        end = operations_data[path[i + 1]]['components']
        ax1.plot([start[0], end[0]], [start[3], end[3]],
                'b-', linewidth=2, alpha=0.3, zorder=1)

    # Draw current position
    ax1.scatter(current_comps[0], current_comps[3], s=500, c='yellow',
               marker='*', edgecolors='black', linewidths=3,
               label='Current', zorder=15)

    # Add component values
    ax1.text(0.02, 0.98,
            f'Current Position:\n'
            f'  scalar = {current_comps[0]:.3f}\n'
            f'  e₁ = {current_comps[1]:+.3f}\n'
            f'  e₂ = {current_comps[2]:+.3f}\n'
            f'  e₁₂ = {current_comps[3]:+.3f}',
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax1.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Plot 2: 3D view
    ax2.set_title('3D View (scalar, e₁, e₂)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Scalar', fontsize=10)
    ax2.set_ylabel('e₁', fontsize=10)
    ax2.set_zlabel('e₂', fontsize=10)

    # Draw cone generators
    gen_array = np.array(list(generators.values()))
    ax2.scatter(gen_array[:, 0], gen_array[:, 1], gen_array[:, 2],
               s=200, c='red', marker='*', edgecolors='darkred',
               linewidths=2, alpha=0.5)

    # Draw cone edges
    for i, j in combinations(range(4), 2):
        pts = gen_array[[i, j], :3]
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                'r-', alpha=0.2, linewidth=1)

    # Draw all operations
    for idx, data in operations_data.items():
        comps = data['components']
        ax2.scatter(comps[0], comps[1], comps[2],
                   s=50, c='lightgray', marker='o', alpha=0.3)

    # Draw path in 3D
    for i in range(len(path) - 1):
        start = operations_data[path[i]]['components']
        end = operations_data[path[i + 1]]['components']
        ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                'b-', linewidth=2, alpha=0.5)

    # Draw current position
    ax2.scatter(current_comps[0], current_comps[1], current_comps[2],
               s=500, c='yellow', marker='*', edgecolors='black',
               linewidths=3, zorder=15)

    # Rotate view
    ax2.view_init(elev=20, azim=frame * 2)  # Slowly rotating view

    plt.tight_layout()

# Create animation
print("Generating frames...")
anim = FuncAnimation(fig3, animate_tour, frames=300, interval=50, repeat=True)

# Save as GIF
writer = PillowWriter(fps=20)
anim.save('demo19_animated_tour.gif', writer=writer)
print("✅ Saved: demo19_animated_tour.gif")

print("\n" + "="*80)
print("ANIMATION COMPLETE!")
print("="*80)

summary_text = """
📊 Generated Files:
   1. demo19_boolean_cone_overview.png
      → Static overview of all 16 operations in the cone
   
   2. demo19_transformation_journeys.png
      → 5 different transformation paths visualized
   
   3. demo19_animated_tour.gif
      → Animated journey through the Boolean cone
      → Watch operations smoothly transform into each other!

🎯 Key Insights from the Visualization:

1. THE CONE STRUCTURE
   • 4 generators (quasi-projectors) at vertices
   • All 16 operations lie within this cone
   • Cone is convex: any blend of operations is valid

2. THE CORRELATION AXIS
   • IFF (+0.50) and XOR (-0.50) are extremes
   • Most operations cluster near zero (independent)
   • Correlation is CONTINUOUS, not discrete

3. TRANSFORMATION PATHS
   • Operations smoothly interpolate
   • Infinite intermediate states exist
   • Boolean logic sees 16 points
   • Geometric logic sees infinite space

4. THE GEOMETRIC STRUCTURE
   • Distance = dissimilarity
   • Angle = relationship type
   • Operations are MOVEMENTS
   • Cone preserves logical validity

💡 The Animation Shows:
   • Boolean operations are just SAMPLES
   • Smooth transitions between all operations
   • The cone as a continuous manifold
   • How correlation evolves during transformations
   • The beauty of geometric logic! ✨
"""

print(summary_text)

plt.show()

print("\n🎉 THE GRAND FINALE IS COMPLETE!")
print("="*80)