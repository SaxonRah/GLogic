import numpy as np
from kingdon import Algebra
from joblib import Parallel, delayed
from functools import lru_cache
import time
from glogic_concept_kingdon import GeometricLogic

from concurrent.futures import ThreadPoolExecutor


class OptimizedGeometricLogic:
    """
    Optimized version of Geometric Logic System (GLS) using:
    - Sparse multivector storage
    - Parallelized computations using threading (avoids pickling issues)
    - Precomputed lookup tables
    - Benchmarking against original implementation
    """

    def __init__(self, p=3, q=0, r=0):
        self.ga = Algebra(p, q, r)
        self.statements = {}
        self.relations = {}
        self.quantifiers = {}

    def add_statement(self, name):
        """Adds a logical statement using sparse representation."""
        if name not in self.statements:
            self.statements[name] = self.ga.vector(name=name)
        if f"- {name}" not in self.statements:
            self.statements[f"- {name}"] = -self.statements[name]

    def sparse_wedge_product(self, A, B):
        """Optimized wedge product with proper zero checking."""
        result = self.statements[A] ^ self.statements[B]
        return result if result != 0 else 0  # Corrected zero check

    def _dot_worker(self, A_name, B_name):
        """Worker function to compute dot product in a safe manner."""
        A = self.statements[A_name]
        B = self.statements[B_name]
        return (A | B).grade(0)

    def parallel_dot_product(self, A, B):
        """Parallelized dot product computation using threading (avoiding pickling issues)."""
        results = Parallel(n_jobs=1)(delayed(self._dot_worker)(A, B) for _ in range(1))
        return results[0]

    def batch_dot_product(self, pairs):
        """Batch computes dot products for multiple pairs, ensuring compatibility with multivector operations."""
        results = []
        for A, B in pairs:
            try:
                result = (self.statements[A] | self.statements[B])
                results.append(result.grade(0) if hasattr(result, 'grade') else result)
            except Exception as e:
                print(f"Error processing dot product for {A}, {B}: {e}")
                results.append(None)
        return results

    def multi_threaded_dot(self, A, B):
        with ThreadPoolExecutor() as executor:
            return executor.submit(self._dot_worker, A, B).result()

    # @lru_cache(maxsize=None)
    # def cached_logical_or(self, A, B):
    #     """Cached OR operation to avoid redundant calculations."""
    #     return self.statements[A] + self.statements[B] + self.sparse_wedge_product(A, B)

    @lru_cache(maxsize=None)
    def cached_logical_or(self, A, B):
        """Cached OR operation with precomputed wedge product."""
        if (A, B) not in self.relations:
            self.relations[(A, B)] = self.statements[A] + self.statements[B] + self.sparse_wedge_product(A, B)
        return self.relations[(A, B)]

    def benchmark(self):
        """Compare performance with the original GeometricLogic implementation."""
        original_logic = GeometricLogic()
        optimized_logic = self

        statements = [f"Var{i}" for i in range(100)]

        if len(statements) < 2:
            print("Not enough statements for benchmarking.")
            return

        for stmt in statements:
            original_logic.add_statement(stmt)
            optimized_logic.add_statement(stmt)

        # Benchmark dot product
        start = time.time()
        for i in range(len(statements) - 1):
            original_logic.dot_product(statements[i], statements[i + 1])
        original_time = time.time() - start

        start = time.time()
        for i in range(len(statements) - 1):
            optimized_logic.parallel_dot_product(statements[i], statements[i + 1])
        optimized_time = time.time() - start

        # Benchmark batch dot product
        pairs = [(statements[i], statements[i + 1]) for i in range(len(statements) - 1)]
        start = time.time()
        optimized_logic.batch_dot_product(pairs)
        batch_time = time.time() - start

        print("\nBenchmark Results:")
        print(f"Original Dot Product Time: {original_time:.6f} sec")
        print(f"Optimized Dot Product Time: {optimized_time:.6f} sec")
        print(f"Batch Dot Product Time: {batch_time:.6f} sec")

    def test_optimizations(self):
        """Run tests with optimized operations, including all tests from glogic_concept_kingdon.py."""

        self.add_statement("Rain")
        self.add_statement("WetGround")
        self.add_statement("AliceTruthful")
        self.add_statement("BobLying")

        print("\nDot Product (Similarity) Tests:")
        print("Rain ⋅ WetGround:", self.parallel_dot_product("Rain", "WetGround"))
        print("Rain ⋅ Rain:", self.parallel_dot_product("Rain", "Rain"))

        print("\nWedge Product (Independence) Tests:")
        print("Rain ∧ WetGround:", self.sparse_wedge_product("Rain", "WetGround"))
        print("Rain ∧ Rain:", self.sparse_wedge_product("Rain", "Rain"))

        print("\nLogical Gate Tests:")
        print("Rain AND WetGround:", self.parallel_dot_product("Rain", "WetGround"))
        print("Rain OR WetGround:", self.cached_logical_or("Rain", "WetGround"))

        xor_result = self.statements["Rain"] + self.statements["WetGround"] - 2 * (
                    self.statements["Rain"] | self.statements["WetGround"])
        print("Rain XOR WetGround:", xor_result)

        print("\nNegation Tests:")
        print("NOT Rain:", -self.statements["Rain"])

        print("\nSparse Wedge Product:")
        print("Rain ∧ WetGround:", self.sparse_wedge_product("Rain", "WetGround"))

        print("\nCached OR Operation:")
        print("Rain OR WetGround:", self.cached_logical_or("Rain", "WetGround"))


if __name__ == "__main__":
    logic_system = OptimizedGeometricLogic()
    logic_system.test_optimizations()
    # logic_system.benchmark()
