"""
Demo 12: Full Geometric Algebra Playground
Explore ALL geometric operations!
"""

from r200 import R200, e1, e2, e12
import numpy as np


class GeometricPlayground:
    def __init__(self):
        self.examples = {}

    def demo_operations(self):
        print("=" * 70)
        print("GEOMETRIC ALGEBRA OPERATIONS PLAYGROUND")
        print("=" * 70)

        # Create some test multivectors
        A = R200(0.5, 0) + R200(0.3, 1) + R200(0.2, 3)  # Mixed
        B = R200(0.7, 0) + R200(-0.4, 2)  # Different mix

        print("\nTest multivectors:")
        print(f"A = {A}")
        print(f"B = {B}")

        # Geometric Product (*)
        print("\n1️⃣  GEOMETRIC PRODUCT: A * B")
        result = A * B
        print(f"   {result}")
        print("   Multiplies including correlation (bivector generation)")

        # Outer Product (^)
        print("\n2️⃣  OUTER PRODUCT: A ^ B")
        result = A ^ B
        print(f"   {result}")
        print("   Pure grade increase (wedge product)")

        # Inner Product (|)
        print("\n3️⃣  INNER PRODUCT: A | B")
        result = A | B
        print(f"   {result}")
        print("   Grade reduction (contraction)")

        # Reverse (~)
        print("\n4️⃣  REVERSE: ~A")
        result = ~A
        print(f"   {result}")
        print("   Reverses basis blade order (flips bivector sign)")

        # Dual (!)
        print("\n5️⃣  DUAL: !A")
        result = A.Dual()
        print(f"   {result}")
        print("   Poincaré duality operator")

        # Conjugate
        print("\n6️⃣  CONJUGATE: A.Conjugate()")
        result = A.Conjugate()
        print(f"   {result}")
        print("   Clifford conjugation")

        # Sandwich Product (>>>)
        # Note: In R200, we can use the sandwich operation
        print("\n7️⃣  SANDWICH PRODUCT: A * B * ~A")
        result = A * B * ~A
        print(f"   {result}")
        print("   Rotation/transformation of B by A")

        print("\n" + "=" * 70)
        print("💡 ALL OF THESE operate on the FULL geometric structure")
        print("   including correlation (bivector) information!")
        print("   None of these exist in Boolean logic!")
        print("=" * 70)


# Run playground
playground = GeometricPlayground()
playground.demo_operations()