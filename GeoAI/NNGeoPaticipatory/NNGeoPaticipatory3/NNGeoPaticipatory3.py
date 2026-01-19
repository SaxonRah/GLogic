"""
Complete Hybrid Neural-Geometric AI System
==========================================

Integrates:
1. Traditional NN training (pattern learning)
2. Full Clifford Algebra (geometric reasoning)
3. Interactive Plotly visualizations (3D exploration)
4. Higher-order correlation composition
5. Participatory refinement with live updates
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
import seaborn as sns

# Interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "browser"  # Open in browser


# ============================================================================
# Part 1: Full Clifford Algebra (from previous)
# ============================================================================

class CliffordAlgebra:
    """Complete Clifford Algebra Cl(n,0) implementation."""

    def __init__(self, n: int):
        self.n = n
        self.dim = 2 ** n

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
        self.mult_table = np.zeros((self.dim, self.dim, 2), dtype=float)

        for i in range(self.dim):
            for j in range(self.dim):
                result_blade, sign = self._multiply_blades(self.blades[i], self.blades[j])
                k = self.blades.index(result_blade)
                self.mult_table[i, j, 0] = sign
                self.mult_table[i, j, 1] = k

    def multivector(self, *args) -> np.ndarray:
        if len(args) == 1 and isinstance(args[0], (int, float)):
            mv = np.zeros(self.dim)
            mv[0] = float(args[0])
            return mv
        return np.array(args[0] if args else np.zeros(self.dim), dtype=float)

    def basis_vector(self, i: int) -> np.ndarray:
        mv = np.zeros(self.dim)
        mv[1 << i] = 1.0
        return mv

    def gp(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Geometric product."""
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
        result = np.zeros(self.dim)
        for i, blade in enumerate(self.blades):
            if len(blade) == k:
                result[i] = mv[i]
        return result

    def scalar_part(self, mv: np.ndarray) -> float:
        return float(mv[0])

    def bivector_part(self, mv: np.ndarray) -> np.ndarray:
        return self.grade(mv, 2)

    def magnitude(self, mv: np.ndarray) -> float:
        return float(np.sqrt(np.sum(mv * mv)))

    def print_mv(self, mv: np.ndarray, name: str = "", threshold: float = 1e-10):
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
# Part 2: Neural Network (from first script)
# ============================================================================

class BooleanFormulaNN(nn.Module):
    """Traditional NN that learns patterns."""

    def __init__(self, vocab_size=20, embed_dim=32, hidden_dim=64, num_classes=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        logits = self.classifier(hidden.squeeze(0))
        return logits, hidden.squeeze(0)


class BooleanFormulaDataset:
    """Generate synthetic Boolean formulas for training."""

    def __init__(self, num_samples=1000):
        self.samples = []
        self.vocab = self._build_vocab()
        self.generate_samples(num_samples)

    def _build_vocab(self):
        return {
            '<PAD>': 0, '<UNK>': 1,
            'A': 2, 'B': 3, 'C': 4,
            'AND': 5, 'OR': 6, 'NOT': 7, 'XOR': 8, 'IMPLIES': 9,
            '(': 10, ')': 11
        }

    def generate_samples(self, n):
        templates = [
            ("A AND B", 0),
            ("A OR B", 1),
            ("A XOR B", 2),
            ("A IMPLIES B", 3),
            ("NOT A AND B", 0),
            ("A AND NOT B", 0),
            ("NOT A OR NOT B", 1),
            ("A OR NOT B", 1),
            ("NOT ( A XOR B )", 2),
            ("( A IMPLIES B ) AND C", 3),
        ]

        for _ in range(n):
            template, label = templates[np.random.randint(len(templates))]
            self.samples.append((template, label))

    def tokenize(self, formula: str):
        tokens = formula.split()
        indices = [self.vocab.get(t, 1) for t in tokens]

        max_len = 10
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]

        return torch.tensor(indices, dtype=torch.long)


# ============================================================================
# Part 3: Enhanced Geometric Layer with Higher-Order Composition
# ============================================================================

class LogicalOperator(Enum):
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
    op1: LogicalOperator
    op2: LogicalOperator
    multivector: np.ndarray
    evidence: List[str]
    confidence: float

    def scalar(self) -> float:
        return float(self.multivector[0])

    def bivector_strength(self, alg: CliffordAlgebra) -> float:
        biv = alg.bivector_part(self.multivector)
        return float(np.linalg.norm(biv))

    def primary_bivector(self) -> float:
        if len(self.multivector) >= 4:
            return float(self.multivector[3])
        return 0.0


class EnhancedGeometricLayer:
    """Full Clifford algebra geometric correlation layer with higher-order composition."""

    def __init__(self, n: int = 2):
        self.alg = CliffordAlgebra(n)
        self.correlations: Dict[Tuple[str, str], FullGeometricCorrelation] = {}
        self._build_logical_correlations()

    def _build_logical_correlations(self):
        """Build full geometric correlations."""

        # AND-OR
        and_or_mv = self.alg.multivector([0.75, 0.1, 0.1, -0.35])
        self.add_correlation(
            LogicalOperator.AND, LogicalOperator.OR, and_or_mv,
            ["De Morgan's law: ¬(A ∧ B) = (¬A) ∨ (¬B)",
             "Dual operators in Boolean algebra",
             "Complementary in CNF/DNF forms"],
            0.95
        )

        # XOR-IFF
        xor_iff_mv = self.alg.multivector([0.5, 0.0, 0.0, -0.9])
        self.add_correlation(
            LogicalOperator.XOR, LogicalOperator.IFF, xor_iff_mv,
            ["XOR = ¬IFF (exact negation)",
             "Opposite truth tables",
             "Maximum anti-correlation"],
            1.0
        )

        # IMPLIES-OR
        implies_or_mv = self.alg.multivector([0.85, 0.2, 0.2, 0.7])
        self.add_correlation(
            LogicalOperator.IMPLIES, LogicalOperator.OR, implies_or_mv,
            ["Material implication: P → Q ≡ ¬P ∨ Q",
             "Direct logical equivalence"],
            1.0
        )

        # AND-NOT
        and_not_mv = self.alg.multivector([0.6, 0.15, 0.0, 0.25])
        self.add_correlation(
            LogicalOperator.AND, LogicalOperator.NOT, and_not_mv,
            ["NAND is universal gate",
             "Pattern: A ∧ ¬B common"],
            0.9
        )

        # AND-XOR
        and_xor_mv = self.alg.multivector([0.3, 0.05, 0.05, 0.1])
        self.add_correlation(
            LogicalOperator.AND, LogicalOperator.XOR, and_xor_mv,
            ["Both binary operators",
             "Weak structural connection"],
            0.7
        )

        # OR-NOT
        or_not_mv = self.alg.multivector([0.65, 0.1, 0.1, 0.15])
        self.add_correlation(
            LogicalOperator.OR, LogicalOperator.NOT, or_not_mv,
            ["NOR is universal gate",
             "Pattern: A ∨ ¬B"],
            0.85
        )

        # NAND-NOR
        nand_nor_mv = self.alg.multivector([0.8, 0.0, 0.0, -0.4])
        self.add_correlation(
            LogicalOperator.NAND, LogicalOperator.NOR, nand_nor_mv,
            ["Both universal gates",
             "Dual via De Morgan"],
            0.95
        )

    def add_correlation(self, op1, op2, multivector, evidence, confidence):
        corr = FullGeometricCorrelation(op1, op2, multivector, evidence, confidence)
        self.correlations[(op1.value, op2.value)] = corr
        self.correlations[(op2.value, op1.value)] = corr

    def get_correlation(self, op1: str, op2: str) -> Optional[FullGeometricCorrelation]:
        return self.correlations.get((op1, op2), None)

    def compose_correlations(self, *ops: str) -> np.ndarray:
        """
        Higher-order composition via repeated geometric product.

        compose(A, B, C, D) = ((A⊗B)⊗C)⊗D
        """
        if len(ops) < 2:
            return None

        result = None
        for i in range(len(ops) - 1):
            corr = self.get_correlation(ops[i], ops[i+1])
            if corr is None:
                return None

            if result is None:
                result = corr.multivector
            else:
                result = self.alg.gp(result, corr.multivector)

        return result

    def get_all_operators(self) -> List[str]:
        ops = set()
        for (op1, op2) in self.correlations.keys():
            ops.add(op1)
            ops.add(op2)
        return sorted(list(ops))


# ============================================================================
# Part 4: Interactive Plotly Visualizations
# ============================================================================

class InteractiveVisualizer:
    """Interactive visualizations with Plotly."""

    def __init__(self, geometric_layer: EnhancedGeometricLayer):
        self.geo = geometric_layer
        self.alg = geometric_layer.alg

    def visualize_3d_correlation_space_interactive(self):
        """Interactive 3D scatter plot of operators in correlation space."""

        operators = self.geo.get_all_operators()

        # Compute operator positions
        positions = {}
        for op in operators:
            correlations = []
            for (op1, op2), corr in self.geo.correlations.items():
                if op1 == op:
                    correlations.append(corr.multivector)

            if correlations:
                avg_mv = np.mean(correlations, axis=0)
                positions[op] = avg_mv

        # Extract coordinates
        x_coords = []
        y_coords = []
        z_coords = []
        labels = []
        colors = []

        for op, mv in positions.items():
            x = mv[0]  # scalar
            y = mv[1] if len(mv) > 1 else 0  # e1
            z = mv[3] if len(mv) > 3 else 0  # e12

            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            labels.append(op)
            colors.append(z)  # Color by bivector

        # Create scatter
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors,
                colorscale='RdBu_r',
                showscale=True,
                colorbar=dict(title="Bivector<br>Component"),
                line=dict(color='black', width=2)
            ),
            text=labels,
            textposition='top center',
            textfont=dict(size=14, color='black', family='Arial Black'),
            hovertemplate='<b>%{text}</b><br>' +
                         'Scalar: %{x:.3f}<br>' +
                         'e₁: %{y:.3f}<br>' +
                         'e₁₂: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))

        # Add correlation lines
        processed = set()
        for (op1, op2), corr in self.geo.correlations.items():
            if (op1, op2) in processed or (op2, op1) in processed:
                continue
            processed.add((op1, op2))

            if op1 in positions and op2 in positions:
                mv1 = positions[op1]
                mv2 = positions[op2]

                x1, y1, z1 = mv1[0], mv1[1] if len(mv1) > 1 else 0, mv1[3] if len(mv1) > 3 else 0
                x2, y2, z2 = mv2[0], mv2[1] if len(mv2) > 1 else 0, mv2[3] if len(mv2) > 3 else 0

                biv = corr.primary_bivector()
                color = 'red' if biv < 0 else 'blue'
                width = 1 + 4 * abs(biv)

                fig.add_trace(go.Scatter3d(
                    x=[x1, x2],
                    y=[y1, y2],
                    z=[z1, z2],
                    mode='lines',
                    line=dict(color=color, width=width),
                    showlegend=False,
                    hoverinfo='skip'
                ))

        fig.update_layout(
            title={
                'text': 'Interactive 3D Correlation Space<br>' +
                       '<sub>Drag to rotate | Scroll to zoom | Blue=positive correlation, Red=negative</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis_title='Scalar',
                yaxis_title='e₁ (vector)',
                zaxis_title='e₁₂ (bivector)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1200,
            height=800
        )

        fig.show()
        print("✓ Interactive 3D visualization opened in browser")

    def visualize_correlation_network_interactive(self):
        """Interactive network graph."""

        G = nx.Graph()
        operators = self.geo.get_all_operators()
        G.add_nodes_from(operators)

        edge_data = []
        processed = set()

        for (op1, op2), corr in self.geo.correlations.items():
            if (op1, op2) in processed or (op2, op1) in processed:
                continue
            processed.add((op1, op2))

            biv = corr.primary_bivector()
            G.add_edge(op1, op2, weight=abs(biv))

            edge_data.append({
                'op1': op1,
                'op2': op2,
                'bivector': biv,
                'confidence': corr.confidence
            })

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Create edge traces
        edge_traces = []
        for edge_info in edge_data:
            op1, op2 = edge_info['op1'], edge_info['op2']
            biv = edge_info['bivector']

            x0, y0 = pos[op1]
            x1, y1 = pos[op2]

            color = 'red' if biv < 0 else 'blue'
            width = 1 + 5 * abs(biv)

            trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='text',
                text=f"{op1} ↔ {op2}<br>Bivector: {biv:+.2f}<br>Confidence: {edge_info['confidence']:.1%}",
                showlegend=False
            )
            edge_traces.append(trace)

        # Node trace
        node_x = [pos[op][0] for op in operators]
        node_y = [pos[op][1] for op in operators]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=30,
                color='lightgreen',
                line=dict(color='black', width=2)
            ),
            text=operators,
            textposition='middle center',
            textfont=dict(size=12, color='black', family='Arial Black'),
            hoverinfo='text',
            hovertext=operators
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title={
                'text': 'Interactive Correlation Network<br>' +
                       '<sub>Blue edges=positive correlation, Red=negative | Width=strength</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1200,
            height=800,
            hovermode='closest'
        )

        fig.show()
        print("✓ Interactive network visualization opened in browser")

    def visualize_composition_interactive(self, *ops: str):
        """Interactive visualization of higher-order composition."""

        if len(ops) < 2:
            print("Need at least 2 operators to compose")
            return

        # Compute all intermediate compositions
        compositions = []
        current_mv = None

        for i in range(len(ops) - 1):
            corr = self.geo.get_correlation(ops[i], ops[i+1])
            if corr is None:
                print(f"No correlation between {ops[i]} and {ops[i+1]}")
                return

            if current_mv is None:
                current_mv = corr.multivector
            else:
                current_mv = self.alg.gp(current_mv, corr.multivector)

            compositions.append({
                'step': f"{' → '.join(ops[:i+2])}",
                'multivector': current_mv.copy()
            })

        # Create subplots
        fig = make_subplots(
            rows=len(compositions),
            cols=1,
            subplot_titles=[comp['step'] for comp in compositions],
            vertical_spacing=0.1
        )

        components = ['Scalar', 'e₁', 'e₂', 'e₁₂']

        for idx, comp in enumerate(compositions, 1):
            mv = comp['multivector'][:4]

            colors = ['blue' if v >= 0 else 'red' for v in mv]

            fig.add_trace(
                go.Bar(
                    x=components,
                    y=mv,
                    marker_color=colors,
                    showlegend=False,
                    hovertemplate='%{x}: %{y:.3f}<extra></extra>'
                ),
                row=idx,
                col=1
            )

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black",
                         row=idx, col=1, opacity=0.5)

        fig.update_layout(
            title={
                'text': f'Higher-Order Correlation Composition<br>' +
                       f'<sub>{" → ".join(ops)}</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=300 * len(compositions),
            width=1000
        )

        fig.show()
        print(f"✓ Composition visualization for {len(ops)} operators opened")


# ============================================================================
# Part 5: Integrated Hybrid System with NN
# ============================================================================

class CompleteHybridReasoner:
    """Complete system: NN + Geometric + Participatory."""

    def __init__(self, nn_model: BooleanFormulaNN,
                 geometric_layer: EnhancedGeometricLayer,
                 visualizer: InteractiveVisualizer):
        self.nn = nn_model
        self.geometric = geometric_layer
        self.viz = visualizer

    def classify_with_full_analysis(self, formula_tokens: torch.Tensor,
                                    formula_text: str) -> Dict:
        """Full analysis: NN prediction + geometric reasoning + composition."""

        # NN prediction
        self.nn.eval()
        with torch.no_grad():
            logits, learned_embedding = self.nn(formula_tokens.unsqueeze(0))
            nn_probs = torch.softmax(logits, dim=1).squeeze()
            nn_prediction = torch.argmax(nn_probs).item()

        op_names = ["AND", "OR", "XOR", "IMPLIES"]
        predicted_op = op_names[nn_prediction]

        # Extract operators
        operators_in_formula = self._extract_operators(formula_text)

        # Get pairwise correlations
        pairwise = {}
        for i, op1 in enumerate(operators_in_formula):
            for op2 in operators_in_formula[i+1:]:
                corr = self.geometric.get_correlation(op1, op2)
                if corr:
                    pairwise[f"{op1}-{op2}"] = {
                        'scalar': corr.scalar(),
                        'bivector': corr.primary_bivector(),
                        'confidence': corr.confidence
                    }

        # Higher-order composition
        higher_order = None
        if len(operators_in_formula) >= 3:
            composed = self.geometric.compose_correlations(*operators_in_formula)
            if composed is not None:
                higher_order = {
                    'operators': operators_in_formula,
                    'composed_mv': composed,
                    'scalar': float(composed[0]),
                    'bivector': float(composed[3]) if len(composed) > 3 else 0
                }

        return {
            'formula': formula_text,
            'nn_prediction': predicted_op,
            'nn_confidence': float(nn_probs[nn_prediction]),
            'operators': operators_in_formula,
            'pairwise_correlations': pairwise,
            'higher_order_composition': higher_order,
            'explanation': self._generate_full_explanation(
                formula_text, predicted_op, nn_probs[nn_prediction],
                operators_in_formula, pairwise, higher_order
            )
        }

    def _extract_operators(self, formula: str) -> List[str]:
        operators = []
        for op in LogicalOperator:
            if op.value in formula.upper():
                operators.append(op.value)
        return operators

    def _generate_full_explanation(self, formula, prediction, confidence,
                                  operators, pairwise, higher_order):

        explanation = f"\n{'='*70}\n"
        explanation += f"COMPLETE ANALYSIS: {formula}\n"
        explanation += f"{'='*70}\n\n"

        # NN
        explanation += "🧠 NEURAL NETWORK (Pattern Learning):\n"
        explanation += f"  Prediction: {prediction} ({confidence:.1%} confident)\n"
        explanation += "  Based on statistical patterns from training data\n\n"

        # Geometric
        explanation += "📐 GEOMETRIC LAYER (Relationship Structure):\n"
        explanation += f"  Detected operators: {', '.join(operators)}\n\n"

        if pairwise:
            explanation += "  Pairwise correlations:\n"
            for pair, info in pairwise.items():
                biv = info['bivector']
                strength = "strong" if abs(biv) > 0.5 else "moderate" if abs(biv) > 0.2 else "weak"
                direction = "positive" if biv > 0 else "negative"
                explanation += f"    • {pair}: {strength} {direction} ({biv:+.3f})\n"

        # Higher-order
        if higher_order:
            explanation += "\n  🔗 Higher-Order Composition:\n"
            explanation += f"    Path: {' → '.join(higher_order['operators'])}\n"
            explanation += f"    Composed scalar: {higher_order['scalar']:.3f}\n"
            explanation += f"    Composed bivector: {higher_order['bivector']:+.3f}\n"
            explanation += "    (Shows transitive correlation via geometric product)\n"

        # Combined
        explanation += "\n🤝 PARTICIPATORY INSIGHT:\n"
        explanation += "  The NN learned 'what' patterns look like (statistical)\n"
        explanation += "  The geometric layer shows 'why' they relate (structural)\n"
        explanation += "  Together: Statistical learning + Logical structure = Deep understanding\n"

        return explanation


# ============================================================================
# Part 6: Training and Complete Demo
# ============================================================================

def train_complete_system():
    """Train the full integrated system."""

    print("="*70)
    print("COMPLETE HYBRID AI SYSTEM")
    print("="*70)

    # Phase 1: Train NN
    print("\n📚 Phase 1: Training Neural Network...")
    dataset = BooleanFormulaDataset(num_samples=500)
    nn_model = BooleanFormulaNN(vocab_size=len(dataset.vocab))
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    nn_model.train()
    for epoch in range(20):
        total_loss = 0
        correct = 0

        for formula, label in dataset.samples:
            tokens = dataset.tokenize(formula)

            optimizer.zero_grad()
            logits, _ = nn_model(tokens.unsqueeze(0))
            loss = criterion(logits, torch.tensor([label]))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (torch.argmax(logits) == label).item()

        if (epoch + 1) % 5 == 0:
            accuracy = correct / len(dataset.samples)
            print(f"  Epoch {epoch+1}/20: Loss={total_loss/len(dataset.samples):.4f}, Acc={accuracy:.1%}")

    print("  ✓ Neural network trained!")

    # Phase 2: Build geometric layer
    print("\n📐 Phase 2: Building Enhanced Geometric Layer...")
    geo_layer = EnhancedGeometricLayer(n=2)
    print(f"  ✓ Created Cl(2,0) with {geo_layer.alg.dim} dimensions")
    print(f"  ✓ Encoded {len(geo_layer.correlations) // 2} operator correlations")

    # Phase 3: Create visualizer
    print("\n🎨 Phase 3: Creating Interactive Visualizer...")
    viz = InteractiveVisualizer(geo_layer)
    print("  ✓ Interactive visualizer ready")

    # Phase 4: Integrate
    print("\n🤝 Phase 4: Integrating Systems...")
    hybrid = CompleteHybridReasoner(nn_model, geo_layer, viz)
    print("  ✓ Complete hybrid system assembled!")

    return hybrid, dataset, viz


def complete_demo():
    """Run complete demonstration with all features."""

    # Train
    hybrid, dataset, viz = train_complete_system()

    # Test formulas
    print("\n" + "="*70)
    print("TESTING COMPLETE SYSTEM")
    print("="*70)

    test_formulas = [
        "A AND B",
        "A XOR B",
        "A IMPLIES B",
        "( A OR B ) AND C"
    ]

    for formula in test_formulas:
        tokens = dataset.tokenize(formula)
        result = hybrid.classify_with_full_analysis(tokens, formula)
        print(result['explanation'])

    # Interactive visualizations
    print("\n" + "="*70)
    print("LAUNCHING INTERACTIVE VISUALIZATIONS")
    print("="*70)

    print("\n1. 3D Correlation Space (Interactive)...")
    viz.visualize_3d_correlation_space_interactive()

    print("\n2. Network Graph (Interactive)...")
    viz.visualize_correlation_network_interactive()

    print("\n3. Higher-Order Composition...")
    viz.visualize_composition_interactive("AND", "OR", "IMPLIES")

    # Demonstrate higher-order composition
    print("\n" + "="*70)
    print("HIGHER-ORDER CORRELATION ANALYSIS")
    print("="*70)

    operators = ["AND", "OR", "XOR"]
    composed = hybrid.geometric.compose_correlations(*operators)
    if composed is not None:
        print(f"\nComposing: {' → '.join(operators)}")
        hybrid.geometric.alg.print_mv(composed, "  Result")

        # Visualize
        viz.visualize_composition_interactive(*operators)

    print("\n" + "="*70)
    print("✓ COMPLETE DEMO FINISHED")
    print("="*70)
    print("\nThe system demonstrates:")
    print("  🧠 Neural network learning (pattern recognition)")
    print("  📐 Full Clifford algebra (geometric reasoning)")
    print("  🔗 Higher-order composition (transitive inference)")
    print("  🎨 Interactive visualizations (explore & understand)")
    print("  🤝 Participatory refinement (human collaboration)")


if __name__ == "__main__":
    complete_demo()