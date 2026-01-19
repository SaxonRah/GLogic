"""
Enhanced Hybrid Neural-Geometric AI with Full Clifford Algebra
===============================================================

Integrates:
1. Traditional NN (pattern learning)
2. Full Clifford Algebra Cl(n,0) (geometric reasoning)
3. Visualization of geometric correlations
4. Participatory refinement with visual feedback
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import seaborn as sns


# ============================================================================
# Part 1: Full Clifford Algebra Implementation (from Boolean_GLogic.py)
# ============================================================================

class CliffordAlgebra:
    """
    Complete Clifford Algebra Cl(n,0) implementation.

    This is the REAL geometric algebra, not just scalar+bivector.
    """

    def __init__(self, n: int):
        self.n = n
        self.dim = 2 ** n

        # Build basis blade structure
        self.blades = []
        self.blade_names = []

        for i in range(self.dim):
            blade = frozenset(j for j in range(self.n) if i & (1 << j))
            self.blades.append(blade)

            if len(blade) == 0:
                name = "1"
            else:
                name = "e" + "".join(str(j+1) for j in sorted(blade))
            self.blade_names.append(name)

        self._build_multiplication_table()

    def _multiply_blades(self, blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, float]:
        """Geometric product of basis blades."""
        list_a = sorted(blade_a)
        list_b = sorted(blade_b)

        result = list_a.copy()
        sign = 1.0

        for b_elem in list_b:
            swaps_needed = sum(1 for r in result if r > b_elem)
            sign *= (-1) ** swaps_needed

            if b_elem in result:
                result.remove(b_elem)
            else:
                result.append(b_elem)
                result.sort()

        return frozenset(result), sign

    def _build_multiplication_table(self):
        """Build Cayley table for geometric product."""
        self.mult_table = np.zeros((self.dim, self.dim, 2), dtype=float)

        for i in range(self.dim):
            for j in range(self.dim):
                result_blade, sign = self._multiply_blades(self.blades[i], self.blades[j])
                k = self.blades.index(result_blade)
                self.mult_table[i, j, 0] = sign
                self.mult_table[i, j, 1] = k

    def multivector(self, *args) -> np.ndarray:
        """Create a multivector."""
        if len(args) == 1 and isinstance(args[0], (int, float)):
            mv = np.zeros(self.dim)
            mv[0] = float(args[0])
            return mv
        return np.array(args[0] if args else np.zeros(self.dim), dtype=float)

    def basis_vector(self, i: int) -> np.ndarray:
        """Create basis vector e_i."""
        mv = np.zeros(self.dim)
        mv[1 << i] = 1.0
        return mv

    def gp(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Geometric product - THE fundamental operation."""
        result = np.zeros(self.dim)

        for i in range(self.dim):
            if abs(a[i]) < 1e-15:
                continue
            for j in range(self.dim):
                if abs(b[j]) < 1e-15:
                    continue
                sign = self.mult_table[i, j, 0]
                k = int(self.mult_table[i, j, 1])
                result[k] += sign * a[i] * b[j]

        return result

    def grade(self, mv: np.ndarray, k: int) -> np.ndarray:
        """Extract grade-k component."""
        result = np.zeros(self.dim)
        for i, blade in enumerate(self.blades):
            if len(blade) == k:
                result[i] = mv[i]
        return result

    def scalar_part(self, mv: np.ndarray) -> float:
        """Extract scalar (grade 0)."""
        return float(mv[0])

    def bivector_part(self, mv: np.ndarray) -> np.ndarray:
        """Extract bivector (grade 2) components."""
        return self.grade(mv, 2)

    def magnitude(self, mv: np.ndarray) -> float:
        """Compute magnitude."""
        return float(np.sqrt(np.sum(mv * mv)))

    def print_mv(self, mv: np.ndarray, name: str = "", threshold: float = 1e-10):
        """Pretty print multivector."""
        terms = []
        for i, coeff in enumerate(mv):
            if abs(coeff) > threshold:
                if self.blade_names[i] == "1":
                    terms.append(f"{coeff:.3f}")
                else:
                    terms.append(f"{coeff:.3f}·{self.blade_names[i]}")

        if terms:
            result = ' + '.join(terms).replace('+ -', '- ')
            if name:
                print(f"{name}: {result}")
            return result
        else:
            if name:
                print(f"{name}: 0")
            return "0"


# ============================================================================
# Part 2: Enhanced Geometric Correlation Layer
# ============================================================================

class LogicalOperator(Enum):
    """Logical operators with their symbolic representations."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"
    IMPLIES = "IMPLIES"
    IFF = "IFF"
    NAND = "NAND"
    NOR = "NOR"


@dataclass
class FullGeometricCorrelation:
    """
    Complete geometric correlation using full multivector.

    Instead of just scalar+bivector, we now store the FULL multivector
    encoding all geometric information.
    """
    op1: LogicalOperator
    op2: LogicalOperator
    multivector: np.ndarray  # Full Clifford algebra element
    evidence: List[str]
    confidence: float

    def scalar(self) -> float:
        """Extract scalar component."""
        return float(self.multivector[0])

    def bivector_strength(self, alg: CliffordAlgebra) -> float:
        """Get total bivector magnitude."""
        biv = alg.bivector_part(self.multivector)
        return float(np.linalg.norm(biv))

    def primary_bivector(self) -> float:
        """Get primary bivector component (e12 for n=2)."""
        # For n=2, e12 is at index 3
        if len(self.multivector) >= 4:
            return float(self.multivector[3])
        return 0.0


class EnhancedGeometricLayer:
    """
    Full Clifford algebra geometric correlation layer.

    Uses actual geometric product for composing correlations.
    """

    def __init__(self, n: int = 2):
        self.alg = CliffordAlgebra(n)
        self.correlations: Dict[Tuple[str, str], FullGeometricCorrelation] = {}
        self._build_logical_correlations()

    def _build_logical_correlations(self):
        """Build full geometric correlations for Boolean operators."""

        # AND-OR relationship (De Morgan's laws)
        and_or_mv = self.alg.multivector([0.75, 0.1, 0.1, -0.35])
        # [scalar, e1, e2, e12]
        self.add_correlation(
            LogicalOperator.AND,
            LogicalOperator.OR,
            and_or_mv,
            [
                "De Morgan's law: ¬(A ∧ B) = (¬A) ∨ (¬B)",
                "Dual operators in Boolean algebra",
                "Complementary in CNF/DNF forms",
                "Transform via negation"
            ],
            0.95
        )

        # XOR-IFF (exact opposites)
        xor_iff_mv = self.alg.multivector([0.5, 0.0, 0.0, -0.9])
        self.add_correlation(
            LogicalOperator.XOR,
            LogicalOperator.IFF,
            xor_iff_mv,
            [
                "XOR = ¬IFF (exact negation)",
                "Opposite truth tables",
                "Same structure, inverted output",
                "Maximum anti-correlation"
            ],
            1.0
        )

        # IMPLIES-OR (material implication)
        implies_or_mv = self.alg.multivector([0.85, 0.2, 0.2, 0.7])
        self.add_correlation(
            LogicalOperator.IMPLIES,
            LogicalOperator.OR,
            implies_or_mv,
            [
                "Material implication: P → Q ≡ ¬P ∨ Q",
                "Direct logical equivalence",
                "Can convert between them",
                "Strong structural similarity"
            ],
            1.0
        )

        # AND-NOT (common pattern)
        and_not_mv = self.alg.multivector([0.6, 0.15, 0.0, 0.25])
        self.add_correlation(
            LogicalOperator.AND,
            LogicalOperator.NOT,
            and_not_mv,
            [
                "NAND is universal gate",
                "Negated literals common in clauses",
                "Pattern: A ∧ ¬B",
                "Moderate positive correlation"
            ],
            0.9
        )

        # AND-XOR (weak relationship)
        and_xor_mv = self.alg.multivector([0.3, 0.05, 0.05, 0.1])
        self.add_correlation(
            LogicalOperator.AND,
            LogicalOperator.XOR,
            and_xor_mv,
            [
                "Both binary operators",
                "No direct logical relationship",
                "Can appear in same formulas",
                "Weak structural connection"
            ],
            0.7
        )

        # OR-NOT (common pattern)
        or_not_mv = self.alg.multivector([0.65, 0.1, 0.1, 0.15])
        self.add_correlation(
            LogicalOperator.OR,
            LogicalOperator.NOT,
            or_not_mv,
            [
                "NOR is universal gate",
                "Pattern: A ∨ ¬B",
                "Common in implications",
                "Moderate positive correlation"
            ],
            0.85
        )

        # NAND-NOR (dual universal gates)
        nand_nor_mv = self.alg.multivector([0.8, 0.0, 0.0, -0.4])
        self.add_correlation(
            LogicalOperator.NAND,
            LogicalOperator.NOR,
            nand_nor_mv,
            [
                "Both are universal gates",
                "Dual relationship via De Morgan",
                "NAND = ¬(A ∧ B), NOR = ¬(A ∨ B)",
                "Negated counterparts"
            ],
            0.95
        )

    def add_correlation(self, op1, op2, multivector, evidence, confidence):
        """Add bidirectional correlation."""
        corr = FullGeometricCorrelation(op1, op2, multivector, evidence, confidence)

        self.correlations[(op1.value, op2.value)] = corr
        self.correlations[(op2.value, op1.value)] = corr

    def get_correlation(self, op1: str, op2: str) -> Optional[FullGeometricCorrelation]:
        """Retrieve correlation."""
        return self.correlations.get((op1, op2), None)

    def compose_correlations(self, op1: str, op2: str, op3: str) -> np.ndarray:
        """
        Compose correlations using geometric product.

        This is the KEY insight: correlations compose via GP!
        If we know A-B and B-C, we can infer A-C.
        """
        corr_ab = self.get_correlation(op1, op2)
        corr_bc = self.get_correlation(op2, op3)

        if corr_ab is None or corr_bc is None:
            return None

        # Geometric product gives transitive correlation
        composed = self.alg.gp(corr_ab.multivector, corr_bc.multivector)

        return composed

    def explain_relationship(self, op1: str, op2: str, verbose: bool = True) -> str:
        """Generate detailed explanation."""
        corr = self.get_correlation(op1, op2)

        if corr is None:
            return f"No known relationship between {op1} and {op2}"

        # Extract components
        scalar = corr.scalar()
        bivector = corr.primary_bivector()

        # Interpret bivector
        if abs(bivector) > 0.7:
            relationship = "strongly"
        elif abs(bivector) > 0.3:
            relationship = "moderately"
        else:
            relationship = "weakly"

        direction = "correlated" if bivector > 0 else "anti-correlated"

        explanation = f"\n{'='*60}\n"
        explanation += f"{op1} ↔ {op2}\n"
        explanation += f"{'='*60}\n"
        explanation += f"Relationship: {relationship} {direction}\n"
        explanation += f"Scalar: {scalar:.3f} | Bivector (e12): {bivector:+.3f}\n"
        explanation += f"Confidence: {corr.confidence:.1%}\n"

        if verbose:
            explanation += f"\nFull multivector:\n"
            explanation += "  " + self.alg.print_mv(corr.multivector)
            explanation += f"\n\nEvidence:\n"
            for i, ev in enumerate(corr.evidence, 1):
                explanation += f"  {i}. {ev}\n"

        return explanation

    def get_all_operators(self) -> List[str]:
        """Get list of all operators we have correlations for."""
        ops = set()
        for (op1, op2) in self.correlations.keys():
            ops.add(op1)
            ops.add(op2)
        return sorted(list(ops))


# ============================================================================
# Part 3: Visualization System
# ============================================================================

class GeometricVisualizer:
    """
    Visualize geometric correlations in multiple ways.
    """

    def __init__(self, geometric_layer: EnhancedGeometricLayer):
        self.geo = geometric_layer
        self.alg = geometric_layer.alg
        sns.set_style("whitegrid")

    def visualize_correlation_network(self, filename: str = "correlation_network.png"):
        """
        Visualize correlations as a network graph.

        Nodes = operators
        Edges = correlations (thickness = strength, color = sign)
        """
        G = nx.Graph()

        # Add all operators as nodes
        operators = self.geo.get_all_operators()
        G.add_nodes_from(operators)

        # Add correlations as edges
        edge_colors = []
        edge_widths = []
        edge_labels = {}

        processed = set()
        for (op1, op2), corr in self.geo.correlations.items():
            if (op1, op2) in processed or (op2, op1) in processed:
                continue
            processed.add((op1, op2))

            biv = corr.primary_bivector()

            G.add_edge(op1, op2, weight=abs(biv))

            # Color: red for negative, blue for positive
            color = 'red' if biv < 0 else 'blue'
            edge_colors.append(color)

            # Width proportional to strength
            edge_widths.append(1 + 5 * abs(biv))

            # Label with bivector value
            edge_labels[(op1, op2)] = f"{biv:+.2f}"

        # Layout
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                              node_color='lightgreen',
                              node_size=3000,
                              alpha=0.9)

        # Draw node labels
        nx.draw_networkx_labels(G, pos,
                               font_size=12,
                               font_weight='bold')

        # Draw edges
        nx.draw_networkx_edges(G, pos,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.6)

        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                     font_size=9)

        plt.title("Geometric Correlation Network\n" +
                 "Edge color: Blue=positive, Red=negative | " +
                 "Edge width: correlation strength",
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved network visualization to {filename}")
        plt.close()

    def visualize_correlation_matrix(self, filename: str = "correlation_matrix.png"):
        """
        Heatmap of all pairwise correlations.
        """
        operators = self.geo.get_all_operators()
        n_ops = len(operators)

        # Build correlation matrix
        matrix = np.zeros((n_ops, n_ops))

        for i, op1 in enumerate(operators):
            for j, op2 in enumerate(operators):
                if i == j:
                    matrix[i, j] = 1.0  # Self-correlation
                else:
                    corr = self.geo.get_correlation(op1, op2)
                    if corr:
                        matrix[i, j] = corr.primary_bivector()

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix,
                   xticklabels=operators,
                   yticklabels=operators,
                   cmap='RdBu_r',
                   center=0,
                   vmin=-1, vmax=1,
                   annot=True,
                   fmt='.2f',
                   square=True,
                   cbar_kws={'label': 'Bivector Correlation'})

        plt.title("Operator Correlation Matrix\n" +
                 "Values show primary bivector component (e12)",
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved correlation matrix to {filename}")
        plt.close()

    def visualize_correlation_space_3d(self, filename: str = "correlation_space_3d.png"):
        """
        3D visualization of operators in correlation space.

        For n=2, we have [scalar, e1, e2, e12].
        Plot in 3D using (scalar, e1, e12) as axes.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Collect operator embeddings
        operator_positions = {}

        # For each operator, compute its "position" in correlation space
        # by averaging all its correlation multivectors
        operators = self.geo.get_all_operators()

        for op in operators:
            # Get all correlations involving this operator
            correlations = []
            for (op1, op2), corr in self.geo.correlations.items():
                if op1 == op:
                    correlations.append(corr.multivector)

            if correlations:
                # Average position
                avg_mv = np.mean(correlations, axis=0)
                operator_positions[op] = avg_mv

        # Plot operators
        for op, mv in operator_positions.items():
            # Use scalar, e1, e12 as coordinates
            x = mv[0]  # scalar
            y = mv[1] if len(mv) > 1 else 0  # e1
            z = mv[3] if len(mv) > 3 else 0  # e12

            ax.scatter(x, y, z, s=300, alpha=0.6)
            ax.text(x, y, z, op, fontsize=12, fontweight='bold')

        # Plot correlation vectors
        processed = set()
        for (op1, op2), corr in self.geo.correlations.items():
            if (op1, op2) in processed or (op2, op1) in processed:
                continue
            processed.add((op1, op2))

            if op1 in operator_positions and op2 in operator_positions:
                mv1 = operator_positions[op1]
                mv2 = operator_positions[op2]

                x1, y1, z1 = mv1[0], mv1[1] if len(mv1) > 1 else 0, mv1[3] if len(mv1) > 3 else 0
                x2, y2, z2 = mv2[0], mv2[1] if len(mv2) > 1 else 0, mv2[3] if len(mv2) > 3 else 0

                biv = corr.primary_bivector()
                color = 'red' if biv < 0 else 'blue'
                alpha = min(abs(biv), 0.8)

                ax.plot([x1, x2], [y1, y2], [z1, z2],
                       color=color, alpha=alpha, linewidth=2)

        ax.set_xlabel('Scalar', fontweight='bold')
        ax.set_ylabel('e₁ (vector)', fontweight='bold')
        ax.set_zlabel('e₁₂ (bivector)', fontweight='bold')
        ax.set_title('Operators in 3D Correlation Space\n' +
                    'Blue lines = positive correlation, Red = negative',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved 3D correlation space to {filename}")
        plt.close()

    def visualize_refinement_comparison(self, op1: str, op2: str,
                                       old_mv: np.ndarray, new_mv: np.ndarray,
                                       filename: str = "refinement_comparison.png"):
        """
        Show before/after comparison when human refines a correlation.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Extract components
        components = ['Scalar', 'e₁', 'e₂', 'e₁₂']
        old_vals = old_mv[:4] if len(old_mv) >= 4 else list(old_mv) + [0]*(4-len(old_mv))
        new_vals = new_mv[:4] if len(new_mv) >= 4 else list(new_mv) + [0]*(4-len(new_mv))

        x = np.arange(len(components))
        width = 0.35

        # Before
        axes[0].bar(x, old_vals, width, label='Old', alpha=0.8, color='orange')
        axes[0].set_ylabel('Coefficient Value', fontweight='bold')
        axes[0].set_title(f'Before Refinement\n{op1} ↔ {op2}', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(components)
        axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0].grid(True, alpha=0.3)

        # After
        axes[1].bar(x, new_vals, width, label='New', alpha=0.8, color='green')
        axes[1].set_ylabel('Coefficient Value', fontweight='bold')
        axes[1].set_title(f'After Refinement\n{op1} ↔ {op2}', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(components)
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].grid(True, alpha=0.3)

        # Highlight changes
        for i, (old_val, new_val) in enumerate(zip(old_vals, new_vals)):
            if abs(old_val - new_val) > 0.01:
                change = new_val - old_val
                axes[1].annotate(f'Δ{change:+.2f}',
                               xy=(i, new_val),
                               xytext=(0, 10 if change > 0 else -20),
                               textcoords='offset points',
                               ha='center',
                               fontsize=10,
                               fontweight='bold',
                               color='red')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved refinement comparison to {filename}")
        plt.close()

    def visualize_composition(self, op1: str, op2: str, op3: str,
                            filename: str = "correlation_composition.png"):
        """
        Visualize transitive correlation via geometric product.

        Shows: A-B + B-C → A-C (composed)
        """
        corr_ab = self.geo.get_correlation(op1, op2)
        corr_bc = self.geo.get_correlation(op2, op3)

        if corr_ab is None or corr_bc is None:
            print(f"Cannot compose: missing correlation")
            return

        # Compose via geometric product
        composed = self.alg.gp(corr_ab.multivector, corr_bc.multivector)

        # Get actual correlation if it exists
        corr_ac = self.geo.get_correlation(op1, op3)

        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        components = ['Scalar', 'e₁', 'e₂', 'e₁₂']

        # A-B
        mv_ab = corr_ab.multivector[:4]
        axes[0, 0].bar(components, mv_ab, alpha=0.8, color='blue')
        axes[0, 0].set_title(f'{op1} ↔ {op2}', fontweight='bold')
        axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 0].grid(True, alpha=0.3)

        # B-C
        mv_bc = corr_bc.multivector[:4]
        axes[0, 1].bar(components, mv_bc, alpha=0.8, color='green')
        axes[0, 1].set_title(f'{op2} ↔ {op3}', fontweight='bold')
        axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 1].grid(True, alpha=0.3)

        # Composed A-C
        mv_composed = composed[:4]
        axes[1, 0].bar(components, mv_composed, alpha=0.8, color='purple')
        axes[1, 0].set_title(f'{op1} ↔ {op3} (via GP composition)', fontweight='bold')
        axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 0].grid(True, alpha=0.3)

        # Actual A-C (if exists)
        if corr_ac:
            mv_actual = corr_ac.multivector[:4]
            axes[1, 1].bar(components, mv_actual, alpha=0.8, color='orange')
            axes[1, 1].set_title(f'{op1} ↔ {op3} (actual)', fontweight='bold')
            axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            axes[1, 1].grid(True, alpha=0.3)

            # Show error
            error = np.linalg.norm(mv_composed - mv_actual)
            axes[1, 1].text(0.5, 0.95, f'Error: {error:.3f}',
                          transform=axes[1, 1].transAxes,
                          ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[1, 1].text(0.5, 0.5, 'No direct correlation exists',
                          transform=axes[1, 1].transAxes,
                          ha='center', va='center',
                          fontsize=14)
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])

        fig.suptitle(f'Correlation Composition via Geometric Product\n' +
                    f'{op1} → {op2} → {op3}',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved composition visualization to {filename}")
        plt.close()


# ============================================================================
# Part 4: Enhanced Participatory Layer
# ============================================================================

class EnhancedParticipatoryRefinement:
    """
    Participatory layer with full geometric reasoning and visualization.
    """

    def __init__(self, hybrid, visualizer: GeometricVisualizer):
        self.hybrid = hybrid
        self.viz = visualizer
        self.refinement_history = []

    def analyze_with_composition(self, formula: str):
        """
        Analyze using geometric product composition.

        If we have A-B and B-C, we can INFER A-C.
        """
        ops = self.hybrid._extract_operators(formula)

        print(f"\nAnalyzing: {formula}")
        print("=" * 60)
        print(f"Detected operators: {ops}")

        # Show direct correlations
        print("\nDirect correlations:")
        for i, op1 in enumerate(ops):
            for op2 in ops[i+1:]:
                corr = self.hybrid.geometric.get_correlation(op1, op2)
                if corr:
                    biv = corr.primary_bivector()
                    print(f"  {op1} ↔ {op2}: bivector = {biv:+.3f}")

        # Show composed correlations
        print("\nComposed correlations (via geometric product):")
        if len(ops) >= 3:
            for i, op1 in enumerate(ops):
                for j, op2 in enumerate(ops):
                    for k, op3 in enumerate(ops):
                        if i < j < k:
                            composed = self.hybrid.geometric.compose_correlations(op1, op2, op3)
                            if composed is not None:
                                biv = composed[3] if len(composed) > 3 else 0
                                print(f"  {op1} → {op2} → {op3}: composed bivector = {biv:+.3f}")

                                # Visualize this composition
                                self.viz.visualize_composition(op1, op2, op3,
                                    filename=f"composition_{op1}_{op2}_{op3}.png")

    def refine_with_visualization(self, op1: str, op2: str,
                                 new_multivector: np.ndarray,
                                 evidence: str):
        """
        Refine correlation with before/after visualization.
        """
        # Get old correlation
        old_corr = self.hybrid.geometric.get_correlation(op1, op2)
        old_mv = old_corr.multivector if old_corr else np.zeros(4)

        # Update
        self.hybrid.geometric.add_correlation(
            LogicalOperator[op1],
            LogicalOperator[op2],
            new_multivector,
            [evidence] + (old_corr.evidence if old_corr else []),
            0.95
        )

        # Visualize change
        self.viz.visualize_refinement_comparison(
            op1, op2, old_mv, new_multivector,
            filename=f"refinement_{op1}_{op2}.png"
        )

        # Record
        self.refinement_history.append({
            "operators": (op1, op2),
            "old_mv": old_mv,
            "new_mv": new_multivector,
            "evidence": evidence
        })

        print(f"✓ Refined {op1} ↔ {op2} correlation")
        print(f"  Old bivector: {old_mv[3]:+.3f}")
        print(f"  New bivector: {new_multivector[3]:+.3f}")
        print(f"  Evidence: {evidence}")


# ============================================================================
# Part 5: Enhanced Demo
# ============================================================================

def enhanced_demo():
    """
    Full demonstration with geometric algebra and visualizations.
    """
    print("=" * 60)
    print("ENHANCED GEOMETRIC AI DEMO")
    print("=" * 60)

    # Create enhanced geometric layer
    print("\nBuilding enhanced geometric layer with Clifford algebra...")
    geo_layer = EnhancedGeometricLayer(n=2)
    print(f"✓ Created Cl(2,0) algebra with {geo_layer.alg.dim} dimensions")
    print(f"✓ Encoded {len(geo_layer.correlations) // 2} operator correlations")

    # Create visualizer
    print("\nCreating visualizations...")
    viz = GeometricVisualizer(geo_layer)

    # Generate all visualizations
    viz.visualize_correlation_network()
    viz.visualize_correlation_matrix()
    viz.visualize_correlation_space_3d()

    # Show detailed correlations
    print("\n" + "=" * 60)
    print("DETAILED CORRELATION ANALYSIS")
    print("=" * 60)

    print(geo_layer.explain_relationship("AND", "OR"))
    print(geo_layer.explain_relationship("XOR", "IFF"))
    print(geo_layer.explain_relationship("IMPLIES", "OR"))

    # Demonstrate composition
    print("\n" + "=" * 60)
    print("GEOMETRIC PRODUCT COMPOSITION")
    print("=" * 60)

    print("\nComposing: AND → OR → IMPLIES")
    composed = geo_layer.compose_correlations("AND", "OR", "IMPLIES")
    if composed is not None:
        print("Composed multivector:")
        geo_layer.alg.print_mv(composed, "  AND ⊗ OR ⊗ IMPLIES")

        # Visualize
        viz.visualize_composition("AND", "OR", "IMPLIES")

    # Demonstrate participatory refinement
    print("\n" + "=" * 60)
    print("PARTICIPATORY REFINEMENT")
    print("=" * 60)

    # Mock hybrid system (without NN for this demo)
    class MockHybrid:
        def __init__(self, geo):
            self.geometric = geo
        def _extract_operators(self, formula):
            ops = []
            for op in LogicalOperator:
                if op.value in formula.upper():
                    ops.append(op.value)
            return ops

    hybrid = MockHybrid(geo_layer)
    participatory = EnhancedParticipatoryRefinement(hybrid, viz)

    # Analyze complex formula
    participatory.analyze_with_composition("A AND (B OR C)")

    # Human refinement
    print("\n" + "-" * 60)
    print("Human expert refines OR-XOR correlation...")
    new_mv = geo_layer.alg.multivector([0.4, 0.05, 0.05, -0.35])
    participatory.refine_with_visualization(
        "OR", "XOR", new_mv,
        "OR is inclusive (one or both), XOR is exclusive (exactly one) - moderately opposing"
    )

    # Re-generate visualizations to show changes
    print("\nRegenerating visualizations with refined correlations...")
    viz.visualize_correlation_network(filename="correlation_network_refined.png")
    viz.visualize_correlation_matrix(filename="correlation_matrix_refined.png")

    print("\n" + "=" * 60)
    print("✓ DEMO COMPLETE")
    print("=" * 60)
    print("\nGenerated visualizations:")
    print("  • correlation_network.png - Network graph of correlations")
    print("  • correlation_matrix.png - Heatmap of all pairwise correlations")
    print("  • correlation_space_3d.png - 3D embedding space")
    print("  • composition_*.png - Transitive correlation inference")
    print("  • refinement_*.png - Before/after human corrections")
    print("\nThe system combines:")
    print("  ✓ Full Clifford algebra (geometric product)")
    print("  ✓ Explicit correlation encoding (bivectors)")
    print("  ✓ Compositional reasoning (GP composition)")
    print("  ✓ Visual transparency (all visualizations)")
    print("  ✓ Human collaboration (surgical refinement)")


if __name__ == "__main__":
    enhanced_demo()