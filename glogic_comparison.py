from glogic_concept_kingdon import GeometricLogic
from glogic_optimized import OptimizedGeometricLogic
import sys
import io


def capture_output(func):
    """Helper function to capture printed output of a function."""
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        func()
    finally:
        sys.stdout = old_stdout

    return buffer.getvalue()


def compare_outputs():
    """Compares output of OptimizedGeometricLogic and GeometricLogic."""

    # Instantiate both logic systems
    original_logic = GeometricLogic()
    optimized_logic = OptimizedGeometricLogic()

    # Capture outputs
    original_output = capture_output(original_logic.test_examples)
    optimized_output = capture_output(optimized_logic.test_optimizations)

    # Print comparison
    print("\n--- Comparison of Outputs ---\n")
    print("\nOriginal GeometricLogic Output:\n")
    print(original_output)

    print("\nOptimized GeometricLogic Output:\n")
    print(optimized_output)

    # Compare line by line
    original_lines = original_output.strip().split("\n")
    optimized_lines = optimized_output.strip().split("\n")

    print("\n--- Differences ---\n")
    for i, (orig, opt) in enumerate(zip(original_lines, optimized_lines)):
        if orig != opt:
            print(f"Line {i + 1}:")
            print(f"Original: {orig}")
            print(f"Optimized: {opt}")
            print("-")


if __name__ == "__main__":
    compare_outputs()


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

--- Comparison of Outputs ---


Original GeometricLogic Output:

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


Optimized GeometricLogic Output:


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
NOT Rain: (-Rain1) 𝐞₁ + (-Rain2) 𝐞₂ + (-Rain3) 𝐞₃

Sparse Wedge Product:
Rain ∧ WetGround: (Rain1*WetGround2 - Rain2*WetGround1) 𝐞₁₂ + (Rain1*WetGround3 - Rain3*WetGround1) 𝐞₁₃ + (Rain2*WetGround3 - Rain3*WetGround2) 𝐞₂₃

Cached OR Operation:
Rain OR WetGround: (Rain1 + WetGround1) 𝐞₁ + (Rain2 + WetGround2) 𝐞₂ + (Rain3 + WetGround3) 𝐞₃ + (Rain1*WetGround2 - Rain2*WetGround1) 𝐞₁₂ + (Rain1*WetGround3 - Rain3*WetGround1) 𝐞₁₃ + (Rain2*WetGround3 - Rain3*WetGround2) 𝐞₂₃


--- Differences ---

Line 15:
Original: NOT Rain: - Rain
Optimized: NOT Rain: (-Rain1) 𝐞₁ + (-Rain2) 𝐞₂ + (-Rain3) 𝐞₃
-
Line 17:
Original: Contradiction Handling:
Optimized: Sparse Wedge Product:
-
Line 18:
Original: Rain ∧ NOT Rain: Contradiction Detected!
Optimized: Rain ∧ WetGround: (Rain1*WetGround2 - Rain2*WetGround1) 𝐞₁₂ + (Rain1*WetGround3 - Rain3*WetGround1) 𝐞₁₃ + (Rain2*WetGround3 - Rain3*WetGround2) 𝐞₂₃
-
Line 19:
Original: BobLying ∧ AliceTruthful: (-AliceTruthful1*BobLying2 + AliceTruthful2*BobLying1) 𝐞₁₂ + (-AliceTruthful1*BobLying3 + AliceTruthful3*BobLying1) 𝐞₁₃ + (-AliceTruthful2*BobLying3 + AliceTruthful3*BobLying2) 𝐞₂₃
Optimized: 
-
Line 20:
Original: 
Optimized: Cached OR Operation:
-
Line 21:
Original: Higher-Order Logic Tests:
Optimized: Rain OR WetGround: (Rain1 + WetGround1) 𝐞₁ + (Rain2 + WetGround2) 𝐞₂ + (Rain3 + WetGround3) 𝐞₃ + (Rain1*WetGround2 - Rain2*WetGround1) 𝐞₁₂ + (Rain1*WetGround3 - Rain3*WetGround1) 𝐞₁₃ + (Rain2*WetGround3 - Rain3*WetGround2) 𝐞₂₃
-

"""