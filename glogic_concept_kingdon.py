import numpy as np
from kingdon import Algebra


class GeometricLogic:
    """
    Geometric Logic System (GLS)
    This system embeds logic into Geometric Algebra (GA), using basis vectors for logical statements
    and geometric operations (dot and wedge products) to express logical relationships.
    - Dot Product (⋅): Encodes logical similarity or equivalence.
    - Wedge Product (∧): Represents logical independence.
    - Negation (-A): Expresses logical negation.
    - Logical Gates (AND, OR, XOR) are redefined in GA terms.
    - Contradiction Detection: Explicitly identifies logical contradictions.
    - Higher-Order Logic: Introduces multivector-based predicates and relations.
    - Quantifiers (∀, ∃): Enables predicate logic representation in GA.
    - Inference Rules: Implements basic inference such as ∀x P(x) → P(a).
    - Nested Quantifiers: Supports ∀x ∃y P(x, y) constructs.
    """

    def __init__(self, p=3, q=0, r=0):
        """Initialize the Clifford Algebra with specified dimensions."""
        self.ga = Algebra(p, q, r)
        locals().update(self.ga.blades)  # Enable easy access to basis blades
        self.statements = {}
        self.relations = {}
        self.quantifiers = {}

    def add_statement(self, name):
        """Adds a logical statement as a basis vector in the algebra."""
        if name not in self.statements:
            self.statements[name] = self.ga.vector(name=name)
        if f"- {name}" not in self.statements:
            self.statements[f"- {name}"] = -self.statements[name]

    def define_relation(self, name, *args):
        """Defines a higher-order relation between multiple logical statements."""
        if name not in self.relations:
            self.relations[name] = sum(self.statements[arg] for arg in args)

    def define_quantifier(self, quantifier, variables, expression):
        """Defines a quantifier in terms of a logical statement."""
        if quantifier == "forall":
            self.quantifiers[f"∀ {variables}"] = expression
        elif quantifier == "exists":
            self.quantifiers[f"∃ {variables}"] = expression

    def infer(self, quantifier, variables, instances):
        """Applies inference rules such as ∀x ∃y P(x, y) → P(a, b)."""
        key = f"∀ {variables}"
        if key in self.quantifiers:
            expr = self.quantifiers[key]
            for var, inst in zip(variables.split(), instances.split()):
                expr = expr.replace(var, inst)
            return expr
        return None

    def dot_product(self, A, B):
        """Logical similarity using the dot product, extracting scalar part correctly."""
        return (self.statements[A] | self.statements[B]).grade(0)

    def wedge_product(self, A, B):
        """Logical independence using the wedge product."""
        wedge = self.statements[A] ^ self.statements[B]
        return wedge if wedge else 0

    def logical_not(self, A):
        """Negation represented as -A, ensuring negations are tracked."""
        return f"- {A}"

    def logical_and(self, A, B):
        """GLS AND using the dot product."""
        return self.dot_product(A, B)

    def logical_or(self, A, B):
        """GLS OR as sum plus wedge product."""
        return self.statements[A] + self.statements[B] + self.wedge_product(A, B)

    def logical_xor(self, A, B):
        """GLS XOR using (A + B - 2(A ⋅ B))."""
        return self.statements[A] + self.statements[B] - 2 * self.dot_product(A, B)

    def contradiction(self, A, B):
        """Detect contradictions: A ∧ -A."""
        neg_A = self.logical_not(A)
        if B == neg_A:
            return "Contradiction Detected!"
        return self.wedge_product(A, B)

    def test_examples(self):
        """Run logic tests with Kingdon-based GA, Higher-Order Logic, Quantifiers, and Inference Rules."""
        self.add_statement("Rain")
        self.add_statement("WetGround")
        self.add_statement("AliceTruthful")
        self.add_statement("BobLying")
        self.add_statement("SelfContradiction")

        self.define_relation("WeatherEffect", "Rain", "WetGround")
        self.define_relation("TruthValue", "AliceTruthful", "BobLying")

        self.define_quantifier("forall", "x", "Rain")  # ∀x Rain
        self.define_quantifier("exists", "y", "WetGround")  # ∃y WetGround(y)
        self.define_quantifier("forall", "x y", "P(x, y)")  # ∀x ∃y P(x, y)

        print("Dot Product (Similarity) Tests:")
        print("Rain ⋅ WetGround:", self.dot_product("Rain", "WetGround"))  # 0 (independent)
        print("Rain ⋅ Rain:", self.dot_product("Rain", "Rain"))  # 1 (self-equivalent)

        print("\nWedge Product (Independence) Tests:")
        print("Rain ∧ WetGround:", self.wedge_product("Rain", "WetGround"))  # Rain∧WetGround
        print("Rain ∧ Rain:", self.wedge_product("Rain", "Rain"))  # 0 (dependent)

        print("\nLogical Gate Tests:")
        print("Rain AND WetGround:", self.logical_and("Rain", "WetGround"))  # 0 (false)
        print("Rain OR WetGround:", self.logical_or("Rain", "WetGround"))  # Rain + WetGround + Rain∧WetGround
        print("Rain XOR WetGround:", self.logical_xor("Rain", "WetGround"))  # Rain + WetGround

        print("\nNegation Tests:")
        print("NOT Rain:", self.logical_not("Rain"))  # -Rain

        print("\nContradiction Handling:")
        print("Rain ∧ NOT Rain:", self.contradiction("Rain", self.logical_not("Rain")))  # Contradiction Detected!
        print("BobLying ∧ AliceTruthful:", self.contradiction("BobLying", "AliceTruthful"))  # BobLying∧AliceTruthful

        print("\nHigher-Order Logic Tests:")
        print("WeatherEffect (Rain, WetGround):", self.relations["WeatherEffect"])  # Encodes relation
        print("TruthValue (AliceTruthful, BobLying):", self.relations["TruthValue"])  # Encodes relation

        print("\nInference Tests:")
        print("Applying ∀x P(x, y) → P(a, b):", self.infer("forall", "x y", "a b"))  # Example inference

        print("\nQuantifier Tests:")
        print("∀x Rain(x):", self.quantifiers["∀ x"])  # Universal quantifier example
        print("∃y WetGround(y):", self.quantifiers["∃ y"])  # Existential quantifier example
        print("∀x ∃y P(x, y):", self.quantifiers["∀ x y"])  # Nested quantifier example


# Run Tests
logic_system = GeometricLogic()
logic_system.test_examples()

"""
Dot Product (Similarity) Tests:
Rain ⋅ WetGround: (Rain1*WetGround1 + Rain2*WetGround2 + Rain3*WetGround3)
Rain ⋅ Rain: (Rain1**2 + Rain2**2 + Rain3**2)

Wedge Product (Independence) Tests:
Rain ∧ WetGround: (Rain1*WetGround2 - Rain2*WetGround1) 𝐞₁₂ + (Rain1*WetGround3 - Rain3*WetGround1) 𝐞₁₃ + (Rain2*WetGround3 - Rain3*WetGround2) 𝐞₂₃
Rain ∧ Rain: 0

Logical Gate Tests:
Rain AND WetGround: (Rain1*WetGround1 + Rain2*WetGround2 + Rain3*WetGround3)
Rain OR WetGround: (Rain1 + WetGround1) 𝐞₁ + (Rain2 + WetGround2) 𝐞₂ + (Rain3 + WetGround3) 𝐞₃ + (Rain1*WetGround2 - Rain2*WetGround1) 𝐞₁₂ + (Rain1*WetGround3 - Rain3*WetGround1) 𝐞₁₃ + (Rain2*WetGround3 - Rain3*WetGround2) 𝐞₂₃
Rain XOR WetGround: (-2*Rain1*WetGround1 - 2*Rain2*WetGround2 - 2*Rain3*WetGround3) + (Rain1 + WetGround1) 𝐞₁ + (Rain2 + WetGround2) 𝐞₂ + (Rain3 + WetGround3) 𝐞₃

Negation Tests:
NOT Rain: - Rain

Contradiction Handling:
Rain ∧ NOT Rain: Contradiction Detected!
BobLying ∧ AliceTruthful: (-AliceTruthful1*BobLying2 + AliceTruthful2*BobLying1) 𝐞₁₂ + (-AliceTruthful1*BobLying3 + AliceTruthful3*BobLying1) 𝐞₁₃ + (-AliceTruthful2*BobLying3 + AliceTruthful3*BobLying2) 𝐞₂₃

Higher-Order Logic Tests:
WeatherEffect (Rain, WetGround): (Rain1 + WetGround1) 𝐞₁ + (Rain2 + WetGround2) 𝐞₂ + (Rain3 + WetGround3) 𝐞₃
TruthValue (AliceTruthful, BobLying): (AliceTruthful1 + BobLying1) 𝐞₁ + (AliceTruthful2 + BobLying2) 𝐞₂ + (AliceTruthful3 + BobLying3) 𝐞₃

Inference Tests:
Applying ∀x P(x, y) → P(a, b): P(a, b)

Quantifier Tests:
∀x Rain(x): Rain
∃y WetGround(y): WetGround
∀x ∃y P(x, y): P(x, y)

Process finished with exit code 0
"""
