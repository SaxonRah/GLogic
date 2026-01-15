"""
Demo 7: Interactive Boolean Operation Explorer
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


def truth_table_to_index(truth_table):
    """Convert truth table to binary index 0-15"""
    index = 0
    for p1, p2 in [(True, True), (True, False), (False, True), (False, False)]:
        if (p1, p2) in truth_table:
            index += 2 ** (3 - [(True, True), (True, False), (False, True), (False, False)].index((p1, p2)))
    return index


class BooleanExplorer:
    def __init__(self):
        self.all_ops = {}

        # Generate all 16
        for i in range(16):
            truth_table = []
            for j, (p1, p2) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
                if i & (1 << (3 - j)):
                    truth_table.append((p1, p2))

            mv = embed_formula(truth_table)
            self.all_ops[i] = {
                'truth_table': truth_table,
                'components': get_components(mv)
            }

    def explore(self, operation_index):
        """Explore a specific operation"""
        op = self.all_ops[operation_index]
        comps = op['components']

        print(f"\n{'=' * 60}")
        print(f"Operation Index: {operation_index} (binary: {operation_index:04b})")
        print(f"{'=' * 60}")
        print(f"Truth table: {op['truth_table']}")
        print(f"Probability: {comps[0]:.2%}")
        print(f"Components: scalar={comps[0]:.3f}, e1={comps[1]:+.3f}, e2={comps[2]:+.3f}, e12={comps[3]:+.3f}")
        print(f"Correlation: ", end="")

        if abs(comps[3]) > 0.4:
            print(f"STRONG {'POSITIVE' if comps[3] > 0 else 'NEGATIVE'}")
        elif abs(comps[3]) > 0.1:
            print(f"Moderate {'positive' if comps[3] > 0 else 'negative'}")
        else:
            print("None (independent)")

        # Find related operations
        print(f"\nRelated operations:")

        # Negation
        neg_idx = 15 - operation_index
        print(f"  Negation: {neg_idx} (scalar={self.all_ops[neg_idx]['components'][0]:.2f})")

        # Same |e12|
        for i, other_op in self.all_ops.items():
            if i != operation_index and abs(abs(other_op['components'][3]) - abs(comps[3])) < 0.01:
                print(f"  Same |e12|: {i}")


# Run explorer
explorer = BooleanExplorer()

print("=" * 60)
print("BOOLEAN OPERATION EXPLORER")
print("=" * 60)
print("16 operations indexed 0-15")
print("Bit pattern (TT)(TF)(FT)(FF)")
print("=" * 60)

# Explore some interesting ones
for idx in range(16):  # AND, Tautology, XOR, IFF
    explorer.explore(idx)
