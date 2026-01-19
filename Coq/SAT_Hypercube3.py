import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
from typing import List, Tuple, Set, Dict
import textwrap


# ==============================================================================
# Enhanced Clause and Visualization with Solving Steps
# ==============================================================================

class Clause:
    """A 3-clause in {±1} encoding (True=+1, False=-1)"""

    def __init__(self, literals: List[Tuple[int, bool]], name: str = ""):
        self.literals = literals
        self.name = name

    def violates(self, assignment: Tuple[int, int, int]) -> bool:
        """Check if assignment violates this clause"""
        for var_idx, polarity in self.literals:
            var_value = assignment[var_idx]
            literal_satisfied = (var_value == 1 and polarity) or (var_value == -1 and not polarity)
            if literal_satisfied:
                return False
        return True

    def to_string(self) -> str:
        """Convert to readable formula string"""
        terms = []
        var_names = ['x₁', 'x₂', 'x₃']
        for var_idx, polarity in self.literals:
            if polarity:
                terms.append(var_names[var_idx])
            else:
                terms.append(f'¬{var_names[var_idx]}')
        return '(' + ' ∨ '.join(terms) + ')'


class SolvingStep:
    """Represents one step in the solving process"""

    def __init__(self,
                 step_num: int,
                 action: str,
                 reasoning: str,
                 highlight_vertices: Set[int] = None,
                 eliminated_vertices: Set[int] = None,
                 solution_vertices: Set[int] = None):
        self.step_num = step_num
        self.action = action
        self.reasoning = reasoning
        self.highlight_vertices = highlight_vertices or set()
        self.eliminated_vertices = eliminated_vertices or set()
        self.solution_vertices = solution_vertices or set()


def cube_vertices():
    """Generate all 8 vertices of {±1}³"""
    vs = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vs.append((x, y, z))
    return np.array(vs, dtype=float)


def cube_edges(vs):
    """Generate edges"""
    edges = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if np.sum(vs[i] != vs[j]) == 1:
                edges.append((vs[i], vs[j]))
    return edges


def vertex_to_assignment_str(vertex: Tuple[int, int, int]) -> str:
    """Convert vertex to readable assignment"""

    def sign_to_val(s):
        return 'T' if s == 1 else 'F'

    return f"({sign_to_val(vertex[0])},{sign_to_val(vertex[1])},{sign_to_val(vertex[2])})"


def create_solving_animation(
        clauses: List[Clause],
        solving_steps: List[SolvingStep],
        filename: str = "sat_solving.gif",
        fps: int = 24,
        frames_per_step: int = 60,
):
    """
    Create an educational animation showing step-by-step SAT solving

    Args:
        clauses: List of clauses in the formula
        solving_steps: List of solving steps with reasoning
        filename: Output filename
        fps: Frames per second (lower = slower, more readable)
        frames_per_step: How many frames to show each step
    """
    V = cube_vertices()
    E = cube_edges(V)

    # Setup for two parallel cubes (x4 = ±1 slices)
    offset = 3.5
    V_pos = V + np.array([0.0, 0.0, 0.0])
    V_neg = V + np.array([offset, 0.0, 0.0])

    fig = plt.figure(figsize=(14, 9))

    # Create grid: left side for 3D cube, right side for text
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 3])

    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_formula = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 1])

    ax_formula.axis('off')
    ax_text.axis('off')

    total_frames = len(solving_steps) * frames_per_step

    def update(frame):
        # Clear axes
        ax_3d.clear()
        ax_formula.clear()
        ax_text.clear()
        ax_formula.axis('off')
        ax_text.axis('off')

        # Determine current step
        step_idx = min(frame // frames_per_step, len(solving_steps) - 1)
        step = solving_steps[step_idx]

        # Progress within this step (for animations)
        progress = (frame % frames_per_step) / frames_per_step

        # ===== Draw Formula (top right) =====
        formula_text = "Formula:\n" + " ∧\n".join(c.to_string() for c in clauses)
        ax_formula.text(0.05, 0.95, formula_text,
                        transform=ax_formula.transAxes,
                        fontsize=13, verticalalignment='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # ===== Draw Current Step Info (bottom right) =====
        step_text = f"Step {step.step_num}: {step.action}\n\n"
        step_text += f"Reasoning:\n{textwrap.fill(step.reasoning, width=35)}\n\n"

        # Count current state
        valid_count = 8 - len(step.eliminated_vertices)
        if len(step.solution_vertices) > 0:
            step_text += f"✓ Found {len(step.solution_vertices)} solution(s)!"
        else:
            step_text += f"Remaining vertices: {valid_count}/8"

        ax_text.text(0.05, 0.95, step_text,
                     transform=ax_text.transAxes,
                     fontsize=11, verticalalignment='top',
                     fontfamily='sans-serif',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

        # ===== Draw 3D Visualization =====

        # Draw edges (faint)
        for a, b in E:
            ax_3d.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                       'gray', alpha=0.2, linewidth=0.5)
            a2 = a + np.array([offset, 0.0, 0.0])
            b2 = b + np.array([offset, 0.0, 0.0])
            ax_3d.plot([a2[0], b2[0]], [a2[1], b2[1]], [a2[2], b2[2]],
                       'gray', alpha=0.2, linewidth=0.5)

        # Categorize vertices
        eliminated = step.eliminated_vertices
        highlighted = step.highlight_vertices
        solutions = step.solution_vertices
        remaining = set(range(8)) - eliminated - solutions

        # Draw vertices with appropriate styling

        # 1. Eliminated vertices (red X)
        for idx in eliminated:
            alpha = min(1.0, 0.3 + 0.7 * progress)
            for v_set in [V_pos, V_neg]:
                ax_3d.scatter([v_set[idx, 0]], [v_set[idx, 1]], [v_set[idx, 2]],
                              s=120, c='red', marker='X', linewidths=2,
                              alpha=alpha, edgecolors='darkred')

        # 2. Highlighted vertices (yellow, pulsing)
        for idx in highlighted:
            pulse_size = 140 + 40 * np.sin(progress * 2 * np.pi)
            for v_set in [V_pos, V_neg]:
                ax_3d.scatter([v_set[idx, 0]], [v_set[idx, 1]], [v_set[idx, 2]],
                              s=pulse_size, c='yellow', marker='o',
                              linewidths=2, edgecolors='orange', alpha=0.9)

        # 3. Solution vertices (bright green, larger)
        for idx in solutions:
            pulse_size = 180 + 30 * np.sin(progress * 4 * np.pi)
            for v_set in [V_pos, V_neg]:
                ax_3d.scatter([v_set[idx, 0]], [v_set[idx, 1]], [v_set[idx, 2]],
                              s=pulse_size, c='lime', marker='*',
                              linewidths=3, edgecolors='darkgreen', alpha=1.0)

        # 4. Remaining valid vertices (blue)
        for idx in remaining:
            for v_set in [V_pos, V_neg]:
                ax_3d.scatter([v_set[idx, 0]], [v_set[idx, 1]], [v_set[idx, 2]],
                              s=80, c='cornflowerblue', marker='o',
                              alpha=0.7, edgecolors='navy')

        # Label some key vertices (on first cube only)
        if step_idx <= 2:  # Only label early on to avoid clutter
            for idx in [0, 7]:  # Label opposite corners
                v = V_pos[idx]
                label = vertex_to_assignment_str(tuple(V[idx]))
                ax_3d.text(v[0], v[1], v[2] + 0.3, label, fontsize=8)

        ax_3d.set_xlabel("x₁", fontsize=11)
        ax_3d.set_ylabel("x₂", fontsize=11)
        ax_3d.set_zlabel("x₃", fontsize=11)
        ax_3d.set_title(f"3-SAT Hypercube Solving (Step {step.step_num}/{len(solving_steps)})",
                        fontsize=13, fontweight='bold')
        ax_3d.set_box_aspect((2.0, 1.0, 1.0))

        # Fixed viewing angle for consistency
        ax_3d.view_init(elev=20, azim=45)
        ax_3d.set_xlim(-1.5, offset + 1.5)
        ax_3d.set_ylim(-1.5, 1.5)
        ax_3d.set_zlim(-1.5, 1.5)

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 // fps, repeat=True)

    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    print(f"✅ Animation saved to {filename}")

    plt.close()
    return anim


# ==============================================================================
# Example: Solve a 3-SAT Problem by Hand
# ==============================================================================

def example_educational_solving():
    """
    Create an educational GIF showing step-by-step solving of:

    (x₁ ∨ x₂ ∨ x₃) ∧ (¬x₁ ∨ x₂ ∨ ¬x₃) ∧ (x₁ ∨ ¬x₂ ∨ x₃)

    We'll solve this using reasoning steps that a human would use.
    """

    # Define the formula
    clauses = [
        Clause([(0, True), (1, True), (2, True)], "C1"),  # x₁ ∨ x₂ ∨ x₃
        Clause([(0, False), (1, True), (2, False)], "C2"),  # ¬x₁ ∨ x₂ ∨ ¬x₃
        Clause([(0, True), (1, False), (2, True)], "C3"),  # x₁ ∨ ¬x₂ ∨ x₃
    ]

    # Define solving steps with reasoning
    V = cube_vertices()

    # Helper: find vertices that violate clauses
    def find_violations(*clause_indices):
        violated = set()
        for idx in range(8):
            v = tuple(V[idx])
            for c_idx in clause_indices:
                if clauses[c_idx].violates(v):
                    violated.add(idx)
                    break
        return violated

    # Step 0: Initial state
    steps = [
        SolvingStep(
            step_num=0,
            action="Initial state",
            reasoning="We have 8 possible assignments for (x₁,x₂,x₃). Each vertex represents one assignment. We need to find which satisfy all three clauses.",
            highlight_vertices=set(range(8)),
            eliminated_vertices=set(),
        )
    ]

    # Step 1: Add first clause
    elim1 = find_violations(0)
    steps.append(SolvingStep(
        step_num=1,
        action="Apply clause C1: (x₁ ∨ x₂ ∨ x₃)",
        reasoning="This clause is FALSE only when x₁=F, x₂=F, x₃=F (all negative). That's vertex (-1,-1,-1). Eliminate it.",
        highlight_vertices={0},  # Highlight the vertex being eliminated
        eliminated_vertices=elim1,
    ))

    # Step 2: Add second clause
    elim2 = find_violations(0, 1)
    new_elim = elim2 - elim1
    steps.append(SolvingStep(
        step_num=2,
        action="Apply clause C2: (¬x₁ ∨ x₂ ∨ ¬x₃)",
        reasoning="This is FALSE when x₁=T, x₂=F, x₃=T. That's vertex (+1,-1,+1). Eliminate it. We now have 6 possible solutions left.",
        highlight_vertices=new_elim,
        eliminated_vertices=elim2,
    ))

    # Step 3: Add third clause
    elim3 = find_violations(0, 1, 2)
    new_elim = elim3 - elim2
    steps.append(SolvingStep(
        step_num=3,
        action="Apply clause C3: (x₁ ∨ ¬x₂ ∨ x₃)",
        reasoning="This is FALSE when x₁=F, x₂=T, x₃=F. That's vertex (-1,+1,-1). Eliminate it. Now 5 vertices remain.",
        highlight_vertices=new_elim,
        eliminated_vertices=elim3,
    ))

    # Step 4: Reasoning - notice x₂ pattern
    remaining = set(range(8)) - elim3
    steps.append(SolvingStep(
        step_num=4,
        action="Analyze remaining vertices",
        reasoning="Let's check: what do surviving vertices have in common? Looking at x₂ (the middle variable)...",
        highlight_vertices=remaining,
        eliminated_vertices=elim3,
    ))

    # Step 5: Observation about x₂
    # Find vertices where x₂=+1
    x2_positive = {idx for idx in remaining if V[idx][1] == 1}
    steps.append(SolvingStep(
        step_num=5,
        action="Key insight: x₂ = True works!",
        reasoning="Notice: if we set x₂=True, all three clauses are automatically satisfied (x₂ appears positive in C1 and C2, ¬x₂ in C3 is fine). Let's verify these solutions.",
        highlight_vertices=x2_positive,
        eliminated_vertices=elim3,
    ))

    # Step 6: Show solutions
    steps.append(SolvingStep(
        step_num=6,
        action="Solutions found!",
        reasoning="Assignments with x₂=True that weren't eliminated: (T,T,T), (T,T,F), (F,T,T). These are our 3 solutions! Note: (F,T,F) was eliminated by C3.",
        eliminated_vertices=elim3,
        solution_vertices=x2_positive,
    ))

    # Step 7: Final note
    steps.append(SolvingStep(
        step_num=7,
        action="Complete",
        reasoning="SAT SATISFIED! We found 3 solutions. While x₂=True helps, C3 still eliminated one such vertex. The geometric view shows solutions cluster on the x₂=+1 face, with one corner removed.",
        eliminated_vertices=elim3,
        solution_vertices=x2_positive,
    ))

    # Create the animation
    create_solving_animation(
        clauses=clauses,
        solving_steps=steps,
        filename="sat_solving_educational.gif",
        fps=12,
        frames_per_step=50,
        # 50 frames per step at 1 fps = 50 seconds per step (too long for real use, but good for demo)
    )


def example_quick_demo():
    """Faster version for quick demonstration (3 seconds per step)"""
    clauses = [
        Clause([(0, True), (1, True), (2, True)], "C1"),
        Clause([(0, False), (1, True), (2, False)], "C2"),
        Clause([(0, True), (1, False), (2, True)], "C3"),
    ]

    V = cube_vertices()

    def find_violations(*clause_indices):
        violated = set()
        for idx in range(8):
            v = tuple(V[idx])
            for c_idx in clause_indices:
                if clauses[c_idx].violates(v):
                    violated.add(idx)
                    break
        return violated

    elim1 = find_violations(0)
    elim2 = find_violations(0, 1)
    elim3 = find_violations(0, 1, 2)
    remaining = set(range(8)) - elim3
    x2_positive = {idx for idx in remaining if V[idx][1] == 1}

    steps = [
        SolvingStep(0, "Start", "8 possible assignments", highlight_vertices=set(range(8))),
        SolvingStep(1, "Clause 1", "Eliminates (-1,-1,-1)", eliminated_vertices=elim1),
        SolvingStep(2, "Clause 2", "Eliminates (+1,-1,+1)", eliminated_vertices=elim2),
        SolvingStep(3, "Clause 3", "Eliminates (-1,+1,-1)", eliminated_vertices=elim3),
        SolvingStep(4, "Insight", "x₂=True satisfies all!", highlight_vertices=x2_positive, eliminated_vertices=elim3),
        SolvingStep(5, "Solution", "4 satisfying assignments found!", eliminated_vertices=elim3,
                    solution_vertices=x2_positive),
    ]

    create_solving_animation(
        clauses=clauses,
        solving_steps=steps,
        filename="sat_solving_demo.gif",
        fps=2,  # 2 fps
        frames_per_step=6,  # 3 seconds per step
    )


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("Creating Educational SAT Solving Animation...")
    print("=" * 70)

    print("\nGenerating two versions:")
    print("  1. Educational (slow, detailed) - sat_solving_educational.gif")
    print("  2. Demo (fast, concise) - sat_solving_demo.gif")
    print()

    print("Creating demo version...")
    example_quick_demo()

    print("\nCreating educational version (this will be slower)...")
    example_educational_solving()

    print("\n" + "=" * 70)
    print("✅ Animations complete!")
    print("\nFiles created:")
    print("  - sat_solving_demo.gif (fast version, ~18 seconds)")
    print("  - sat_solving_educational.gif (slow version, detailed)")