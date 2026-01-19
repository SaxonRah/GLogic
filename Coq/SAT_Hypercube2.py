import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Tuple, Set


# ==============================================================================
# Clause Representation
# ==============================================================================

class Clause:
    """A 3-clause in {±1} encoding (True=+1, False=-1)"""

    def __init__(self, literals: List[Tuple[int, bool]]):
        """
        literals: list of (var_index, polarity) where:
            - var_index in {0, 1, 2} for x1, x2, x3
            - polarity: True means positive literal, False means negated
        """
        self.literals = literals

    def violates(self, assignment: Tuple[int, int, int]) -> bool:
        """Check if assignment violates this clause (makes it false)"""
        # In {±1} encoding: literal x_i is satisfied if assignment[i] matches polarity
        # Clause is violated if ALL literals are false
        for var_idx, polarity in self.literals:
            var_value = assignment[var_idx]
            # Positive literal (x_i): satisfied if var_value = +1
            # Negative literal (¬x_i): satisfied if var_value = -1
            literal_satisfied = (var_value == 1 and polarity) or (var_value == -1 and not polarity)
            if literal_satisfied:
                return False  # At least one literal is true, clause is satisfied
        return True  # All literals false, clause violated


# ==============================================================================
# Geometry Utilities
# ==============================================================================

def cube_vertices():
    """Generate all 8 vertices of {±1}³"""
    vs = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vs.append((x, y, z))
    return np.array(vs, dtype=float)


def cube_edges(vs):
    """Generate edges connecting vertices differing in exactly one coordinate"""
    edges = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if np.sum(vs[i] != vs[j]) == 1:
                edges.append((vs[i], vs[j]))
    return edges


# ==============================================================================
# Visualization Functions
# ==============================================================================

def visualize_single_clause(clause: Clause, offset: float = 3.0,
                            title: str = "Single 3-Clause", ax=None):
    """Visualize one clause on two parallel cubes (x4 = ±1 slices)"""
    V = cube_vertices()
    E = cube_edges(V)

    V_pos = V + np.array([0.0, 0.0, 0.0])
    V_neg = V + np.array([offset, 0.0, 0.0])

    # Find forbidden vertices
    forbidden_indices = [i for i, v in enumerate(V) if clause.violates(tuple(v))]

    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')

    # Draw edges
    for a, b in E:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'gray', alpha=0.3, linewidth=0.5)
        a2, b2 = a + np.array([offset, 0.0, 0.0]), b + np.array([offset, 0.0, 0.0])
        ax.plot([a2[0], b2[0]], [a2[1], b2[1]], [a2[2], b2[2]], 'gray', alpha=0.3, linewidth=0.5)

    # Plot vertices
    allowed_pos = np.delete(V_pos, forbidden_indices, axis=0)
    allowed_neg = np.delete(V_neg, forbidden_indices, axis=0)

    if len(allowed_pos) > 0:
        ax.scatter(allowed_pos[:, 0], allowed_pos[:, 1], allowed_pos[:, 2],
                   s=60, c='blue', marker='o', alpha=0.7, label='satisfies')
    if len(allowed_neg) > 0:
        ax.scatter(allowed_neg[:, 0], allowed_neg[:, 1], allowed_neg[:, 2],
                   s=60, c='blue', marker='o', alpha=0.7)

    # Plot forbidden vertices
    for idx in forbidden_indices:
        fp = V_pos[idx]
        fn = V_neg[idx]
        ax.scatter([fp[0]], [fp[1]], [fp[2]], s=150, c='red', marker='x',
                   linewidths=3, label='violates' if idx == forbidden_indices[0] else '')
        ax.scatter([fn[0]], [fn[1]], [fn[2]], s=150, c='red', marker='x', linewidths=3)

    ax.set_xlabel("x1 (and offset)")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title(title)
    ax.set_box_aspect((2.0, 1.0, 1.0))
    ax.legend(loc='upper left')

    return ax


def visualize_two_clauses(clause1: Clause, clause2: Clause, offset: float = 3.0):
    """Show two clauses and their intersection"""
    V = cube_vertices()
    E = cube_edges(V)

    V_pos = V + np.array([0.0, 0.0, 0.0])
    V_neg = V + np.array([offset, 0.0, 0.0])

    # Find violations for each clause
    forbidden1 = {i for i, v in enumerate(V) if clause1.violates(tuple(v))}
    forbidden2 = {i for i, v in enumerate(V) if clause2.violates(tuple(v))}
    forbidden_both = forbidden1 | forbidden2  # Union: violated by either clause

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw edges
    for a, b in E:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'gray', alpha=0.3, linewidth=0.5)
        a2, b2 = a + np.array([offset, 0.0, 0.0]), b + np.array([offset, 0.0, 0.0])
        ax.plot([a2[0], b2[0]], [a2[1], b2[1]], [a2[2], b2[2]], 'gray', alpha=0.3, linewidth=0.5)

    # Categorize vertices
    only_clause1 = forbidden1 - forbidden2
    only_clause2 = forbidden2 - forbidden1
    both_clauses = forbidden1 & forbidden2
    satisfies_both = set(range(8)) - forbidden_both

    # Plot vertices by category
    for idx in satisfies_both:
        ax.scatter([V_pos[idx, 0]], [V_pos[idx, 1]], [V_pos[idx, 2]],
                   s=80, c='green', marker='o', alpha=0.8)
        ax.scatter([V_neg[idx, 0]], [V_neg[idx, 1]], [V_neg[idx, 2]],
                   s=80, c='green', marker='o', alpha=0.8)

    for idx in only_clause1:
        ax.scatter([V_pos[idx, 0]], [V_pos[idx, 1]], [V_pos[idx, 2]],
                   s=120, c='orange', marker='x', linewidths=2)
        ax.scatter([V_neg[idx, 0]], [V_neg[idx, 1]], [V_neg[idx, 2]],
                   s=120, c='orange', marker='x', linewidths=2)

    for idx in only_clause2:
        ax.scatter([V_pos[idx, 0]], [V_pos[idx, 1]], [V_pos[idx, 2]],
                   s=120, c='purple', marker='x', linewidths=2)
        ax.scatter([V_neg[idx, 0]], [V_neg[idx, 1]], [V_neg[idx, 2]],
                   s=120, c='purple', marker='x', linewidths=2)

    for idx in both_clauses:
        ax.scatter([V_pos[idx, 0]], [V_pos[idx, 1]], [V_pos[idx, 2]],
                   s=150, c='red', marker='X', linewidths=3)
        ax.scatter([V_neg[idx, 0]], [V_neg[idx, 1]], [V_neg[idx, 2]],
                   s=150, c='red', marker='X', linewidths=3)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label=f'Satisfies both ({len(satisfies_both)} vertices)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='orange',
               markersize=10, label=f'Violates clause 1 only'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='purple',
               markersize=10, label=f'Violates clause 2 only'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
               markersize=12, label=f'Violates both'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.set_xlabel("x1 (and offset)")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title(f"Two Clauses: {len(satisfies_both)}/8 vertices satisfy both (SAT)")
    ax.set_box_aspect((2.0, 1.0, 1.0))

    plt.tight_layout()
    return fig


def visualize_unsat(clauses: List[Clause], offset: float = 3.0):
    """Show UNSAT case where all vertices are eliminated"""
    V = cube_vertices()
    E = cube_edges(V)

    V_pos = V + np.array([0.0, 0.0, 0.0])
    V_neg = V + np.array([offset, 0.0, 0.0])

    # Find all forbidden vertices (union over all clauses)
    all_forbidden = set()
    for clause in clauses:
        all_forbidden.update(i for i, v in enumerate(V) if clause.violates(tuple(v)))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw edges more prominently
    for a, b in E:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'black', alpha=0.2, linewidth=1)
        a2, b2 = a + np.array([offset, 0.0, 0.0]), b + np.array([offset, 0.0, 0.0])
        ax.plot([a2[0], b2[0]], [a2[1], b2[1]], [a2[2], b2[2]], 'black', alpha=0.2, linewidth=1)

    # All vertices are forbidden
    for idx in range(8):
        ax.scatter([V_pos[idx, 0]], [V_pos[idx, 1]], [V_pos[idx, 2]],
                   s=200, c='darkred', marker='X', linewidths=4, alpha=0.9)
        ax.scatter([V_neg[idx, 0]], [V_neg[idx, 1]], [V_neg[idx, 2]],
                   s=200, c='darkred', marker='X', linewidths=4, alpha=0.9)

    ax.set_xlabel("x1 (and offset)")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title("UNSAT: All vertices eliminated (no solution exists)",
                 fontsize=14, fontweight='bold', color='darkred')
    ax.set_box_aspect((2.0, 1.0, 1.0))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='X', color='w', markerfacecolor='darkred',
               markersize=12, label='All 8 vertices violate constraints'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

    plt.tight_layout()
    return fig


def create_clause_animation(clauses: List[Clause], filename: str = "sat_animation.gif",
                            offset: float = 3.0, duration_per_clause: int = 60):
    """Create animation showing clauses being added sequentially"""
    V = cube_vertices()
    E = cube_edges(V)

    V_pos = V + np.array([0.0, 0.0, 0.0])
    V_neg = V + np.array([offset, 0.0, 0.0])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Track which vertices are still valid at each step
    valid_vertices = set(range(8))
    forbidden_history = []  # Track cumulative forbidden sets

    for clause in clauses:
        forbidden = {i for i, v in enumerate(V) if clause.violates(tuple(v))}
        valid_vertices -= forbidden
        forbidden_history.append(valid_vertices.copy())

    total_frames = len(clauses) * duration_per_clause + duration_per_clause  # Extra frames at end

    def update(frame):
        ax.clear()

        # Determine current clause index
        clause_idx = min(frame // duration_per_clause, len(clauses) - 1)
        progress = (frame % duration_per_clause) / duration_per_clause

        # Draw edges
        for a, b in E:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'gray', alpha=0.3, linewidth=0.5)
            a2, b2 = a + np.array([offset, 0.0, 0.0]), b + np.array([offset, 0.0, 0.0])
            ax.plot([a2[0], b2[0]], [a2[1], b2[1]], [a2[2], b2[2]], 'gray', alpha=0.3, linewidth=0.5)

        # Current state of vertices
        if clause_idx < len(forbidden_history):
            current_valid = forbidden_history[clause_idx]
        else:
            current_valid = forbidden_history[-1]

        current_forbidden = set(range(8)) - current_valid

        # Plot valid vertices
        for idx in current_valid:
            ax.scatter([V_pos[idx, 0]], [V_pos[idx, 1]], [V_pos[idx, 2]],
                       s=100, c='green', marker='o', alpha=0.8)
            ax.scatter([V_neg[idx, 0]], [V_neg[idx, 1]], [V_neg[idx, 2]],
                       s=100, c='green', marker='o', alpha=0.8)

        # Plot forbidden vertices with animation
        for idx in current_forbidden:
            alpha = min(1.0, 0.5 + 0.5 * progress)  # Fade in
            ax.scatter([V_pos[idx, 0]], [V_pos[idx, 1]], [V_pos[idx, 2]],
                       s=150, c='red', marker='X', linewidths=3, alpha=alpha)
            ax.scatter([V_neg[idx, 0]], [V_neg[idx, 1]], [V_neg[idx, 2]],
                       s=150, c='red', marker='X', linewidths=3, alpha=alpha)

        ax.set_xlabel("x1 (and offset)")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")

        if len(current_valid) == 0:
            ax.set_title(f"UNSAT: All vertices eliminated after {clause_idx + 1} clauses",
                         fontsize=13, fontweight='bold', color='darkred')
        else:
            ax.set_title(f"After clause {clause_idx + 1}/{len(clauses)}: "
                         f"{len(current_valid)}/8 vertices remain", fontsize=13)

        ax.set_box_aspect((2.0, 1.0, 1.0))
        ax.set_xlim(-1.5, offset + 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

    anim = FuncAnimation(fig, update, frames=total_frames, interval=50, repeat=True)

    # Save animation
    writer = PillowWriter(fps=20)
    anim.save(filename, writer=writer)
    print(f"Animation saved to {filename}")

    plt.close()
    return anim


# ==============================================================================
# Example Scenarios
# ==============================================================================

def example_single_clause():
    """Original example: (x1 OR x2 OR x3)"""
    clause = Clause([(0, True), (1, True), (2, True)])
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    visualize_single_clause(clause, title="Single Clause: (x1 ∨ x2 ∨ x3)", ax=ax)
    plt.tight_layout()
    plt.savefig("single_clause.png", dpi=200)
    print("Saved: single_clause.png")
    plt.close()


def example_two_clauses_sat():
    """Two clauses that leave some vertices (SAT case)"""
    # (x1 OR x2 OR x3) AND (¬x1 OR ¬x2 OR x3)
    clause1 = Clause([(0, True), (1, True), (2, True)])  # x1 ∨ x2 ∨ x3
    clause2 = Clause([(0, False), (1, False), (2, True)])  # ¬x1 ∨ ¬x2 ∨ x3

    fig = visualize_two_clauses(clause1, clause2)
    plt.savefig("two_clauses_sat.png", dpi=200)
    print("Saved: two_clauses_sat.png")
    plt.close()


def example_unsat():
    """Carefully crafted UNSAT example covering all 8 vertices"""
    # We need clauses that together eliminate all vertices
    # Each vertex is a 3-bit pattern; we'll create clauses that each forbid certain patterns

    clauses = [
        Clause([(0, True), (1, True), (2, True)]),  # Forbids (-1,-1,-1)
        Clause([(0, False), (1, True), (2, True)]),  # Forbids (+1,-1,-1)
        Clause([(0, True), (1, False), (2, True)]),  # Forbids (-1,+1,-1)
        Clause([(0, True), (1, True), (2, False)]),  # Forbids (-1,-1,+1)
        Clause([(0, False), (1, False), (2, True)]),  # Forbids (+1,+1,-1)
        Clause([(0, False), (1, True), (2, False)]),  # Forbids (+1,-1,+1)
        Clause([(0, True), (1, False), (2, False)]),  # Forbids (-1,+1,+1)
        Clause([(0, False), (1, False), (2, False)]),  # Forbids (+1,+1,+1)
    ]

    fig = visualize_unsat(clauses)
    plt.savefig("unsat_all_eliminated.png", dpi=200)
    print("Saved: unsat_all_eliminated.png")
    plt.close()


def example_animation():
    """Progressive elimination animation"""
    clauses = [
        Clause([(0, True), (1, True), (2, True)]),  # Eliminates 1 vertex
        Clause([(0, False), (1, False), (2, True)]),  # Eliminates more
        Clause([(0, True), (1, False), (2, False)]),  # Eliminates more
        Clause([(0, False), (1, True), (2, False)]),  # Even more
    ]

    create_clause_animation(clauses, "sat_progressive.gif", duration_per_clause=40)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("Generating SAT Hypercube Visualizations...")
    print("=" * 60)

    print("\n1. Single clause...")
    example_single_clause()

    print("\n2. Two clauses (SAT)...")
    example_two_clauses_sat()

    print("\n3. UNSAT case...")
    example_unsat()

    print("\n4. Animation (this takes a moment)...")
    example_animation()

    print("\n" + "=" * 60)
    print("✅ All visualizations generated!")
    print("\nGenerated files:")
    print("  - single_clause.png")
    print("  - two_clauses_sat.png")
    print("  - unsat_all_eliminated.png")
    print("  - sat_progressive.gif")