"""
Hybrid Neural-Geometric AI Layer
================================

Demonstrates combining:
1. Traditional NN (pattern learning from data)
2. Geometric layer (correlation/relationship encoding)
3. Participatory reasoning (explaining decisions)

Test case: Boolean formula classification
- Small, verifiable, demonstrates key concepts
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum


# ============================================================================
# Part 1: Traditional Neural Network (learns "WHAT")
# ============================================================================

class BooleanFormulaNN(nn.Module):
    """Traditional NN that learns to classify Boolean formulas."""

    def __init__(self, vocab_size=20, embed_dim=32, hidden_dim=64, num_classes=4):
        super().__init__()

        # Token embedding (converts formula to learned representation)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM to process sequential formula
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len] token indices
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # Process sequence
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Use final hidden state for classification
        logits = self.classifier(hidden.squeeze(0))

        return logits, hidden.squeeze(0)  # Return both logits and learned embedding


# ============================================================================
# Part 2: Geometric Correlation Layer (encodes "WHY")
# ============================================================================

class LogicalOperator(Enum):
    """Logical operators we'll encode correlations for."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"
    IMPLIES = "IMPLIES"
    IFF = "IFF"


@dataclass
class GeometricCorrelation:
    """Represents a correlation in the geometric space."""
    op1: LogicalOperator
    op2: LogicalOperator
    scalar: float  # Truth probability
    bivector: float  # Correlation strength
    evidence: List[str]  # Supporting evidence
    confidence: float  # How confident are we?


class GeometricLayer:
    """
    Encodes logical relationships as geometric correlations.

    This is manually built or LLM-assisted to capture domain knowledge.
    """

    def __init__(self):
        self.correlations: Dict[Tuple[str, str], GeometricCorrelation] = {}
        self._build_logical_correlations()

    def _build_logical_correlations(self):
        """Build correlation map for Boolean operators."""

        # AND-OR relationship (De Morgan's laws)
        self.add_correlation(
            op1=LogicalOperator.AND,
            op2=LogicalOperator.OR,
            scalar=0.75,  # Often related
            bivector=-0.35,  # Somewhat opposing (via negation)
            evidence=[
                "De Morgan's law: NOT(A AND B) = (NOT A) OR (NOT B)",
                "Dual operators in Boolean algebra",
                "Complementary in CNF/DNF forms"
            ],
            confidence=0.95
        )

        # AND-NOT relationship
        self.add_correlation(
            op1=LogicalOperator.AND,
            op2=LogicalOperator.NOT,
            scalar=0.6,
            bivector=+0.25,  # Often appear together (negated clauses)
            evidence=[
                "NAND is universal gate",
                "Negated literals in clauses common",
                "A AND NOT B is common pattern"
            ],
            confidence=0.9
        )

        # XOR-IFF relationship (exact opposites)
        self.add_correlation(
            op1=LogicalOperator.XOR,
            op2=LogicalOperator.IFF,
            scalar=0.5,
            bivector=-0.9,  # Strong anti-correlation
            evidence=[
                "XOR = NOT IFF",
                "Exactly opposite truth tables",
                "Same structure, negated output"
            ],
            confidence=1.0
        )

        # IMPLIES-OR relationship (P → Q = ¬P ∨ Q)
        self.add_correlation(
            op1=LogicalOperator.IMPLIES,
            op2=LogicalOperator.OR,
            scalar=0.85,
            bivector=+0.7,  # Strong positive correlation
            evidence=[
                "Material implication: P → Q ≡ ¬P ∨ Q",
                "Can always convert between them",
                "Logically equivalent modulo negation"
            ],
            confidence=1.0
        )

        # AND-XOR relationship (no strong connection)
        self.add_correlation(
            op1=LogicalOperator.AND,
            op2=LogicalOperator.XOR,
            scalar=0.3,
            bivector=+0.1,  # Weak, slightly positive
            evidence=[
                "Both binary operators",
                "No direct logical relationship",
                "Can appear in same formulas"
            ],
            confidence=0.7
        )

    def add_correlation(self, op1, op2, scalar, bivector, evidence, confidence):
        """Add a bidirectional correlation."""
        corr = GeometricCorrelation(op1, op2, scalar, bivector, evidence, confidence)

        # Store both directions
        self.correlations[(op1.value, op2.value)] = corr
        self.correlations[(op2.value, op1.value)] = corr

    def get_correlation(self, op1: str, op2: str) -> GeometricCorrelation:
        """Retrieve correlation between operators."""
        return self.correlations.get((op1, op2), None)

    def explain_relationship(self, op1: str, op2: str) -> str:
        """Generate human-readable explanation of relationship."""
        corr = self.get_correlation(op1, op2)

        if corr is None:
            return f"No known relationship between {op1} and {op2}"

        # Interpret bivector
        if corr.bivector > 0.5:
            relationship = "strongly correlated"
        elif corr.bivector > 0.2:
            relationship = "moderately correlated"
        elif corr.bivector < -0.5:
            relationship = "strongly anti-correlated"
        elif corr.bivector < -0.2:
            relationship = "moderately anti-correlated"
        else:
            relationship = "weakly related"

        explanation = f"{op1} and {op2} are {relationship} (bivector: {corr.bivector:.2f})\n"
        explanation += f"Confidence: {corr.confidence:.1%}\n"
        explanation += f"\nEvidence:\n"
        for ev in corr.evidence:
            explanation += f"  • {ev}\n"

        return explanation


# ============================================================================
# Part 3: Hybrid Reasoning Layer (combines WHAT + WHY)
# ============================================================================

class HybridReasoner:
    """
    Combines NN learned patterns with geometric correlations.

    Can explain decisions using both statistical learning and logical structure.
    """

    def __init__(self, nn_model: BooleanFormulaNN, geometric_layer: GeometricLayer):
        self.nn = nn_model
        self.geometric = geometric_layer

    def classify_with_explanation(self, formula_tokens: torch.Tensor,
                                  formula_text: str) -> Dict:
        """
        Classify formula and explain using both learning and reasoning.
        """
        # Get NN prediction
        self.nn.eval()
        with torch.no_grad():
            logits, learned_embedding = self.nn(formula_tokens.unsqueeze(0))
            nn_probs = torch.softmax(logits, dim=1).squeeze()
            nn_prediction = torch.argmax(nn_probs).item()

        # Map to operator names
        op_names = ["AND", "OR", "XOR", "IMPLIES"]
        predicted_op = op_names[nn_prediction]

        # Extract operators from formula
        operators_in_formula = self._extract_operators(formula_text)

        # Get geometric reasoning
        geometric_reasoning = {}
        for op1 in operators_in_formula:
            for op2 in operators_in_formula:
                if op1 != op2:
                    corr = self.geometric.get_correlation(op1, op2)
                    if corr:
                        geometric_reasoning[f"{op1}-{op2}"] = {
                            "bivector": corr.bivector,
                            "confidence": corr.confidence,
                            "evidence": corr.evidence
                        }

        # Combine insights
        return {
            "formula": formula_text,
            "nn_prediction": predicted_op,
            "nn_confidence": float(nn_probs[nn_prediction]),
            "nn_learned_embedding": learned_embedding.numpy(),
            "operators_detected": operators_in_formula,
            "geometric_correlations": geometric_reasoning,
            "combined_explanation": self._generate_explanation(
                formula_text, predicted_op, nn_probs[nn_prediction],
                operators_in_formula, geometric_reasoning
            )
        }

    def _extract_operators(self, formula: str) -> List[str]:
        """Extract logical operators from formula string."""
        operators = []
        for op in LogicalOperator:
            if op.value in formula.upper():
                operators.append(op.value)
        return operators

    def _generate_explanation(self, formula, prediction, confidence,
                              operators, geometric_reasoning):
        """Generate combined explanation using NN + geometric insights."""

        explanation = f"Analysis of: {formula}\n"
        explanation += "=" * 60 + "\n\n"

        # NN insight
        explanation += f"LEARNED PATTERN (Neural Network):\n"
        explanation += f"  Predicts this is {prediction}-like with {confidence:.1%} confidence\n"
        explanation += f"  Based on statistical patterns in training data\n\n"

        # Geometric insight
        explanation += f"LOGICAL STRUCTURE (Geometric Layer):\n"
        explanation += f"  Detected operators: {', '.join(operators)}\n\n"

        if geometric_reasoning:
            explanation += "  Operator relationships:\n"
            for pair, info in geometric_reasoning.items():
                op1, op2 = pair.split('-')
                biv = info['bivector']

                if abs(biv) > 0.5:
                    strength = "strongly"
                elif abs(biv) > 0.2:
                    strength = "moderately"
                else:
                    strength = "weakly"

                direction = "correlated" if biv > 0 else "anti-correlated"

                explanation += f"    • {op1} and {op2} are {strength} {direction}\n"
                explanation += f"      (bivector: {biv:+.2f}, confidence: {info['confidence']:.1%})\n"

        # Combined insight
        explanation += "\nCOMBINED REASONING:\n"
        explanation += "  The NN learned 'what' this pattern looks like from examples.\n"
        explanation += "  The geometric layer explains 'why' operators relate this way.\n"
        explanation += "  Together: statistical learning + logical structure = deeper understanding\n"

        return explanation


# ============================================================================
# Part 4: Training and Demo
# ============================================================================

class BooleanFormulaDataset:
    """Generate synthetic Boolean formulas for training."""

    def __init__(self, num_samples=1000):
        self.samples = []
        self.vocab = self._build_vocab()
        self.generate_samples(num_samples)

    def _build_vocab(self):
        """Build vocabulary for tokenization."""
        return {
            '<PAD>': 0, '<UNK>': 1,
            'A': 2, 'B': 3, 'C': 4,
            'AND': 5, 'OR': 6, 'NOT': 7, 'XOR': 8, 'IMPLIES': 9,
            '(': 10, ')': 11
        }

    def generate_samples(self, n):
        """Generate diverse Boolean formulas."""
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
        """Convert formula to token indices."""
        tokens = formula.split()
        indices = [self.vocab.get(t, 1) for t in tokens]

        # Pad to fixed length
        max_len = 10
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]

        return torch.tensor(indices, dtype=torch.long)


def train_hybrid_system():
    """Train the NN component, then add geometric layer."""

    print("=" * 60)
    print("TRAINING HYBRID NEURAL-GEOMETRIC AI SYSTEM")
    print("=" * 60)

    # Create dataset
    dataset = BooleanFormulaDataset(num_samples=500)

    # Initialize NN
    nn_model = BooleanFormulaNN(vocab_size=len(dataset.vocab))
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train NN (quick training for demo)
    print("\nPhase 1: Training Neural Network (learns patterns)...")
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
            print(f"  Epoch {epoch + 1}/20: Loss={total_loss / len(dataset.samples):.4f}, Acc={accuracy:.1%}")

    # Create geometric layer
    print("\nPhase 2: Building Geometric Correlation Layer (encodes relationships)...")
    geometric_layer = GeometricLayer()
    print(f"  Encoded {len(geometric_layer.correlations)} operator correlations")

    # Create hybrid reasoner
    print("\nPhase 3: Combining into Hybrid Reasoner...")
    hybrid = HybridReasoner(nn_model, geometric_layer)
    print("  ✓ Hybrid system ready!")

    return hybrid, dataset


def demo_participatory_reasoning():
    """Demonstrate the participatory aspect."""

    print("\n" + "=" * 60)
    print("PARTICIPATORY REASONING DEMO")
    print("=" * 60)

    hybrid, dataset = train_hybrid_system()

    # Test formulas
    test_formulas = [
        "A AND B",
        "A XOR B",
        "A IMPLIES B",
        "NOT A OR B"
    ]

    for formula in test_formulas:
        tokens = dataset.tokenize(formula)
        result = hybrid.classify_with_explanation(tokens, formula)

        print("\n" + "=" * 60)
        print(result["combined_explanation"])

        # Show geometric relationship details
        if result["geometric_correlations"]:
            print("\nDETAILED GEOMETRIC REASONING:")
            for pair, info in result["geometric_correlations"].items():
                print(f"\n{pair}:")
                for evidence in info["evidence"][:2]:  # Show first 2 pieces
                    print(f"  • {evidence}")


# ============================================================================
# Part 5: Interactive Refinement (Participatory Layer)
# ============================================================================

class ParticipatoryRefinement:
    """
    Allows human to refine both NN and geometric understanding.

    This is the "participating" layer that makes it collaborative.
    """

    def __init__(self, hybrid: HybridReasoner):
        self.hybrid = hybrid
        self.refinement_history = []

    def analyze_uncertainty(self, formula: str):
        """Identify areas of uncertainty for human review."""

        tokens = torch.tensor([self.hybrid.nn.embedding.num_embeddings - 1] * 10)
        result = self.hybrid.classify_with_explanation(tokens, formula)

        uncertainties = []

        # Check NN confidence
        if result["nn_confidence"] < 0.7:
            uncertainties.append({
                "type": "nn_uncertainty",
                "description": f"NN only {result['nn_confidence']:.1%} confident",
                "suggestion": "More training examples might help"
            })

        # Check for unknown correlations
        ops = result["operators_detected"]
        for i, op1 in enumerate(ops):
            for op2 in ops[i + 1:]:
                corr = self.hybrid.geometric.get_correlation(op1, op2)
                if corr is None:
                    uncertainties.append({
                        "type": "missing_correlation",
                        "description": f"No correlation data for {op1}-{op2}",
                        "suggestion": "Human expert should define this relationship"
                    })
                elif corr.confidence < 0.8:
                    uncertainties.append({
                        "type": "low_confidence_correlation",
                        "description": f"{op1}-{op2} correlation only {corr.confidence:.1%} confident",
                        "suggestion": "Review and strengthen evidence"
                    })

        return uncertainties

    def human_corrects_correlation(self, op1: str, op2: str,
                                   new_bivector: float, evidence: str):
        """Human provides correction to geometric correlation."""

        old_corr = self.hybrid.geometric.get_correlation(op1, op2)

        # Update correlation
        self.hybrid.geometric.add_correlation(
            op1=LogicalOperator[op1],
            op2=LogicalOperator[op2],
            scalar=old_corr.scalar if old_corr else 0.5,
            bivector=new_bivector,
            evidence=[evidence] + (old_corr.evidence if old_corr else []),
            confidence=0.95  # Human input is high confidence
        )

        # Record refinement
        self.refinement_history.append({
            "type": "correlation_update",
            "operators": (op1, op2),
            "old_bivector": old_corr.bivector if old_corr else None,
            "new_bivector": new_bivector,
            "evidence": evidence
        })

        print(f"✓ Updated {op1}-{op2} correlation to {new_bivector:+.2f}")
        print(f"  Evidence: {evidence}")


# ============================================================================
# Main Demo
# ============================================================================

if __name__ == "__main__":
    # Run the demo
    demo_participatory_reasoning()

    print("\n" + "=" * 60)
    print("PARTICIPATORY REFINEMENT EXAMPLE")
    print("=" * 60)

    # Quick setup for refinement demo
    hybrid, dataset = train_hybrid_system()
    participatory = ParticipatoryRefinement(hybrid)

    # Show uncertainty analysis
    print("\nAnalyzing: 'A AND (B XOR C)'")
    uncertainties = participatory.analyze_uncertainty("A AND ( B XOR C )")

    if uncertainties:
        print("\nAreas needing human input:")
        for u in uncertainties:
            print(f"  • {u['description']}")
            print(f"    → {u['suggestion']}")

    # Human provides refinement
    print("\n" + "-" * 60)
    print("Human expert refines AND-XOR correlation...")
    participatory.human_corrects_correlation(
        "AND", "XOR",
        new_bivector=-0.15,
        evidence="AND is conjunctive (both must be true), XOR is exclusive (exactly one true) - somewhat opposing"
    )

    print("\n✓ System refined through human collaboration!")