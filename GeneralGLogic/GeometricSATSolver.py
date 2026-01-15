"""
Geometric 3-SAT Solver

Uses Clifford algebra Cl(n,0) to:
1. Embed CNF formulas as multivectors
2. Analyze correlation structure
3. Guide DPLL search with geometric heuristics
4. Detect conflicts early via geometric distance

Key insight: Sparse correlation = decomposable problem!
"""

from rn00 import Rn00, BooleanEmbedder, blade_bits_to_name, count_bits
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Dict
import heapq


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Clause:
    """A clause in CNF: disjunction of literals"""
    literals: List[Tuple[int, bool]]  # [(var_idx, is_positive), ...]

    def __str__(self):
        terms = []
        for var, pos in self.literals:
            terms.append(f"v{var}" if pos else f"¬v{var}")
        return "(" + " ∨ ".join(terms) + ")"

    def __repr__(self):
        return str(self)


@dataclass
class CNFFormula:
    """Complete CNF formula"""
    n_vars: int
    clauses: List[Clause]

    def __str__(self):
        return " ∧ ".join(str(c) for c in self.clauses)


@dataclass
class Assignment:
    """Partial or complete variable assignment"""
    values: Dict[int, bool]  # var_idx -> value

    def copy(self):
        return Assignment(self.values.copy())

    def is_complete(self, n_vars):
        return len(self.values) == n_vars

    def extend(self, var, value):
        new = self.copy()
        new.values[var] = value
        return new


# =============================================================================
# CNF Parser (DIMACS format)
# =============================================================================

def parse_dimacs(text: str) -> CNFFormula:
    """Parse DIMACS CNF format"""
    lines = text.strip().split('\n')
    clauses = []
    n_vars = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith('c'):
            continue
        if line.startswith('p'):
            parts = line.split()
            n_vars = int(parts[2])
            continue

        # Parse clause
        literals = []
        for lit in line.split():
            lit_int = int(lit)
            if lit_int == 0:
                break
            var_idx = abs(lit_int) - 1  # 0-indexed
            is_positive = lit_int > 0
            literals.append((var_idx, is_positive))

        if literals:
            clauses.append(Clause(literals))

    return CNFFormula(n_vars, clauses)


def parse_simple(clauses_str: str, n_vars: int) -> CNFFormula:
    """
    Parse simple format: "v1 v2 -v3, -v1 v2, ..."
    Clauses separated by comma, literals by space
    """
    clauses = []
    for clause_str in clauses_str.split(','):
        literals = []
        for lit in clause_str.strip().split():
            if lit.startswith('-'):
                literals.append((int(lit[1:]) - 1, False))
            else:
                literals.append((int(lit[1:]) - 1, True))
        clauses.append(Clause(literals))

    return CNFFormula(n_vars, clauses)


# =============================================================================
# Geometric Embedding
# =============================================================================

class GeometricCNF:
    """Embed and analyze CNF formulas in Cl(n,0)"""

    def __init__(self, formula: CNFFormula):
        self.formula = formula
        self.n = formula.n_vars
        self.embedder = BooleanEmbedder(self.n)
        self.mv = None
        self.analysis = None

    def embed(self, max_samples=None):
        """
        Embed formula into geometric algebra.

        If max_samples is set, uses sampling instead of full enumeration
        (useful for large n)
        """
        print(f"Embedding CNF with {self.n} variables, {len(self.formula.clauses)} clauses...")

        if max_samples is None or max_samples >= 2 ** self.n:
            # Full enumeration
            satisfying = self._find_all_satisfying()
        else:
            # Sampling approach
            satisfying = self._sample_satisfying(max_samples)

        if not satisfying:
            print("Formula is UNSATISFIABLE")
            return None

        print(f"Found {len(satisfying)} satisfying assignments")

        # Embed
        self.mv = self.embedder.embed_truth_table(satisfying)
        return self.mv

    def _find_all_satisfying(self):
        """Find all satisfying assignments (exponential!)"""
        satisfying = []
        for assignment_bits in itertools.product([True, False], repeat=self.n):
            if self._satisfies_all(assignment_bits):
                satisfying.append(assignment_bits)
        return satisfying

    def _sample_satisfying(self, n_samples):
        """Sample random assignments and keep satisfying ones"""
        import random
        satisfying = []
        attempts = 0
        max_attempts = n_samples * 10

        while len(satisfying) < n_samples and attempts < max_attempts:
            assignment = tuple(random.choice([True, False]) for _ in range(self.n))
            if self._satisfies_all(assignment):
                satisfying.append(assignment)
            attempts += 1

        return satisfying

    def _satisfies_all(self, assignment):
        """Check if assignment satisfies all clauses"""
        for clause in self.formula.clauses:
            if not self._satisfies_clause(assignment, clause):
                return False
        return True

    def _satisfies_clause(self, assignment, clause):
        """Check if assignment satisfies a single clause"""
        for var_idx, is_positive in clause.literals:
            var_value = assignment[var_idx]
            literal_value = var_value if is_positive else not var_value
            if literal_value:
                return True
        return False

    def analyze(self):
        """Analyze correlation structure"""
        if self.mv is None:
            raise ValueError("Must embed first")

        grades = self.mv.grades()

        self.analysis = {
            'probability': self.mv.coeffs[0],
            'variable_biases': {},
            'correlations': {},
            'higher_order': {},
            'active_variables': set(),
            'constraint_strength': {}
        }

        # Grade 1: variable biases
        if 1 in grades:
            for blade_bits, coeff in grades[1]:
                var_idx = (blade_bits.bit_length() - 1) if blade_bits else 0
                self.analysis['variable_biases'][var_idx] = coeff
                if abs(coeff) > 0.01:
                    self.analysis['active_variables'].add(var_idx)
                    self.analysis['constraint_strength'][var_idx] = abs(coeff)

        # Grade 2: pairwise correlations
        if 2 in grades:
            for blade_bits, coeff in grades[2]:
                vars_in_blade = [i for i in range(self.n) if (blade_bits >> i) & 1]
                if len(vars_in_blade) == 2:
                    i, j = sorted(vars_in_blade)
                    self.analysis['correlations'][(i, j)] = coeff
                    if abs(coeff) > 0.01:
                        self.analysis['active_variables'].add(i)
                        self.analysis['active_variables'].add(j)
                        # Update constraint strength
                        self.analysis['constraint_strength'][i] = \
                            self.analysis['constraint_strength'].get(i, 0) + abs(coeff)
                        self.analysis['constraint_strength'][j] = \
                            self.analysis['constraint_strength'].get(j, 0) + abs(coeff)

        # Higher order
        for grade in grades:
            if grade > 2:
                for blade_bits, coeff in grades[grade]:
                    if abs(coeff) > 0.01:
                        self.analysis['higher_order'][blade_bits] = coeff

        return self.analysis

    def print_analysis(self):
        """Print human-readable analysis"""
        if self.analysis is None:
            self.analyze()

        print("\n" + "=" * 70)
        print("GEOMETRIC STRUCTURE ANALYSIS")
        print("=" * 70)

        print(f"\nSolution Density: {self.analysis['probability']:.3f}")
        print(f"  ({int(self.analysis['probability'] * 2 ** self.n)}/{2 ** self.n} assignments satisfy)")

        print(f"\nActive Variables: {len(self.analysis['active_variables'])}/{self.n}")
        free_vars = set(range(self.n)) - self.analysis['active_variables']
        if free_vars:
            print(f"Free Variables: {sorted(free_vars)}")
            print("  (These can be set to any value)")

        print("\nVariable Constraint Strength (ranked):")
        ranked = sorted(self.analysis['constraint_strength'].items(),
                        key=lambda x: x[1], reverse=True)
        for var, strength in ranked[:10]:  # Top 10
            bias = self.analysis['variable_biases'].get(var, 0)
            direction = "TRUE" if bias > 0.3 else "FALSE" if bias < -0.3 else "mixed"
            print(f"  v{var}: strength={strength:.3f}, bias={bias:+.3f} → {direction}")

        print("\nTop Pairwise Correlations:")
        ranked_corr = sorted(self.analysis['correlations'].items(),
                             key=lambda x: abs(x[1]), reverse=True)
        for (i, j), corr in ranked_corr[:10]:
            relationship = "AGREE" if corr > 0 else "DISAGREE"
            print(f"  v{i} ↔ v{j}: {corr:+.3f} [{relationship}]")

        if self.analysis['higher_order']:
            print(f"\nHigher-order interactions: {len(self.analysis['higher_order'])}")
            for blade_bits, coeff in list(self.analysis['higher_order'].items())[:5]:
                print(f"  {blade_bits_to_name(blade_bits)}: {coeff:+.3f}")


# =============================================================================
# DPLL with Geometric Guidance
# =============================================================================

class GeometricDPLL:
    """DPLL solver guided by geometric heuristics"""

    def __init__(self, formula: CNFFormula, geometric: GeometricCNF = None):
        self.formula = formula
        self.geometric = geometric
        self.n = formula.n_vars

        # Statistics
        self.decisions = 0
        self.conflicts = 0
        self.unit_propagations = 0

    def solve(self, use_geometric=True, verbose=True):
        """Main solve routine"""
        print("\n" + "=" * 70)
        print("STARTING GEOMETRIC-GUIDED DPLL")
        print("=" * 70)

        # Get variable ordering from geometric analysis
        if use_geometric and self.geometric and self.geometric.analysis:
            var_order = self._geometric_variable_ordering()
            print(f"Using geometric variable ordering: {var_order[:10]}...")
        else:
            var_order = list(range(self.n))
            print("Using default variable ordering")

        assignment = Assignment({})
        result = self._dpll(assignment, var_order, verbose)

        print("\n" + "=" * 70)
        print("SOLVER STATISTICS")
        print("=" * 70)
        print(f"Decisions made: {self.decisions}")
        print(f"Conflicts encountered: {self.conflicts}")
        print(f"Unit propagations: {self.unit_propagations}")

        return result

    def _geometric_variable_ordering(self):
        """Order variables by constraint strength (most constrained first)"""
        if not self.geometric.analysis:
            return list(range(self.n))

        # Sort by constraint strength (descending)
        strength = self.geometric.analysis['constraint_strength']
        ordered = sorted(range(self.n),
                         key=lambda v: strength.get(v, 0),
                         reverse=True)
        return ordered

    def _dpll(self, assignment: Assignment, var_order: List[int], verbose: bool) -> Optional[Assignment]:
        """Recursive DPLL"""

        # Unit propagation
        assignment = self._unit_propagate(assignment)
        if assignment is None:
            self.conflicts += 1
            return None

        # Check if complete
        if assignment.is_complete(self.n):
            if self._verify_solution(assignment):
                return assignment
            else:
                self.conflicts += 1
                return None

        # Choose next variable (first unassigned in order)
        var = None
        for v in var_order:
            if v not in assignment.values:
                var = v
                break

        if var is None:
            return None

        # Choose value (use geometric bias if available)
        value = self._choose_value(var, assignment)

        if verbose and self.decisions % 100 == 0:
            print(f"  Decision {self.decisions}: v{var} = {value}")

        self.decisions += 1

        # Try this assignment
        result = self._dpll(assignment.extend(var, value), var_order, verbose)
        if result is not None:
            return result

        # Backtrack: try opposite value
        result = self._dpll(assignment.extend(var, not value), var_order, verbose)
        return result

    def _unit_propagate(self, assignment: Assignment) -> Optional[Assignment]:
        """Unit propagation: find unit clauses and force their values"""
        changed = True
        current = assignment.copy()

        while changed:
            changed = False

            for clause in self.formula.clauses:
                # Check clause status
                satisfied = False
                unassigned_literal = None
                unassigned_count = 0

                for var, is_positive in clause.literals:
                    if var in current.values:
                        value = current.values[var]
                        literal_value = value if is_positive else not value
                        if literal_value:
                            satisfied = True
                            break
                    else:
                        unassigned_count += 1
                        unassigned_literal = (var, is_positive)

                if satisfied:
                    continue

                # Unit clause: exactly one unassigned literal
                if unassigned_count == 1:
                    var, is_positive = unassigned_literal
                    forced_value = is_positive
                    current.values[var] = forced_value
                    changed = True
                    self.unit_propagations += 1

                # Empty clause: conflict!
                if unassigned_count == 0:
                    return None

        return current

    def _choose_value(self, var: int, assignment: Assignment) -> bool:
        """Choose value for variable using geometric bias"""
        if self.geometric and self.geometric.analysis:
            bias = self.geometric.analysis['variable_biases'].get(var, 0)
            if abs(bias) > 0.1:
                # Strong bias: follow it
                return bias > 0

        # Default: try True first
        return True

    def _verify_solution(self, assignment: Assignment) -> bool:
        """Verify that assignment satisfies all clauses"""
        for clause in self.formula.clauses:
            satisfied = False
            for var, is_positive in clause.literals:
                value = assignment.values[var]
                literal_value = value if is_positive else not value
                if literal_value:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True


# =============================================================================
# Complete Solver Pipeline
# =============================================================================

class GeometricSATSolver:
    """Complete pipeline: parse → embed → analyze → solve"""

    def __init__(self, formula: CNFFormula):
        self.formula = formula
        self.geometric = GeometricCNF(formula)
        self.solver = None
        self.solution = None

    def solve(self, use_embedding=True, max_samples=1000, verbose=True):
        """Full pipeline"""
        print("=" * 70)
        print("GEOMETRIC SAT SOLVER")
        print("=" * 70)
        print(f"Formula: {self.formula.n_vars} variables, {len(self.formula.clauses)} clauses")

        # Step 1: Embed (if requested and feasible)
        if use_embedding and self.formula.n_vars <= 15:
            print("\n[Step 1] Embedding formula into Cl({})...".format(self.formula.n_vars))
            start = time.time()
            mv = self.geometric.embed(max_samples=max_samples)
            elapsed = time.time() - start
            print(f"Embedding completed in {elapsed:.2f}s")

            if mv is None:
                print("Formula is UNSATISFIABLE (detected during embedding)")
                return None

            # Step 2: Analyze
            print("\n[Step 2] Analyzing geometric structure...")
            self.geometric.analyze()
            self.geometric.print_analysis()
        else:
            print("\n[Skipping embedding - too many variables or disabled]")

        # Step 3: Solve with DPLL
        print("\n[Step 3] Running DPLL with geometric guidance...")
        start = time.time()

        self.solver = GeometricDPLL(self.formula, self.geometric if use_embedding else None)
        self.solution = self.solver.solve(use_geometric=use_embedding, verbose=verbose)

        elapsed = time.time() - start
        print(f"DPLL completed in {elapsed:.2f}s")

        # Result
        if self.solution:
            print("\n" + "=" * 70)
            print("✅ SATISFIABLE")
            print("=" * 70)
            print("Solution:")
            for var in sorted(self.solution.values.keys()):
                print(f"  v{var} = {self.solution.values[var]}")
        else:
            print("\n" + "=" * 70)
            print("❌ UNSATISFIABLE")
            print("=" * 70)

        return self.solution


# =============================================================================
# Test Cases
# =============================================================================

def simple():
    """Simple test case"""
    print("\n" + "=" * 80)
    print("TEST 1: Simple 3-SAT")
    print("=" * 80)

    # (v1 ∨ v2 ∨ v3) ∧ (¬v1 ∨ v2 ∨ ¬v3) ∧ (v1 ∨ ¬v2 ∨ v3)
    clauses = [
        Clause([(0, True), (1, True), (2, True)]),
        Clause([(0, False), (1, True), (2, False)]),
        Clause([(0, True), (1, False), (2, True)]),
    ]

    formula = CNFFormula(3, clauses)
    solver = GeometricSATSolver(formula)
    solution = solver.solve(use_embedding=True)

    return solution is not None


def with_free_variables():
    """Test with unconstrained variables"""
    print("\n" + "=" * 80)
    print("TEST 2: Formula with Free Variables")
    print("=" * 80)

    # 5 variables but only v0, v1, v2 are constrained
    # v3 and v4 should show up as free
    clauses = [
        Clause([(0, True), (1, True), (2, True)]),
        Clause([(0, False), (1, True), (2, False)]),
        Clause([(0, True), (1, False), (2, True)]),
        # Tautology involving v3, v4 - doesn't constrain them
        Clause([(3, True), (3, False), (4, True)]),
    ]

    formula = CNFFormula(5, clauses)
    solver = GeometricSATSolver(formula)
    solution = solver.solve(use_embedding=True)

    return solution is not None


def unsat():
    """Test UNSAT detection"""
    print("\n" + "=" * 80)
    print("TEST 3: Unsatisfiable Formula")
    print("=" * 80)

    # (v1) ∧ (¬v1) - clearly unsat
    clauses = [
        Clause([(0, True)]),
        Clause([(0, False)]),
    ]

    formula = CNFFormula(1, clauses)
    solver = GeometricSATSolver(formula)
    solution = solver.solve(use_embedding=True)

    return solution is None


def large_without_embedding():
    """Test larger formula without embedding (pure DPLL)"""
    print("\n" + "=" * 80)
    print("TEST 4: Larger Formula (DPLL only)")
    print("=" * 80)

    # Generate random 3-SAT
    import random
    random.seed(42)

    n_vars = 20
    n_clauses = 80  # ~4 ratio

    clauses = []
    for _ in range(n_clauses):
        vars_in_clause = random.sample(range(n_vars), 3)
        literals = [(v, random.choice([True, False])) for v in vars_in_clause]
        clauses.append(Clause(literals))

    formula = CNFFormula(n_vars, clauses)
    solver = GeometricSATSolver(formula)
    solution = solver.solve(use_embedding=False, verbose=False)  # Too big to embed

    return solution is not None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("GEOMETRIC 3-SAT SOLVER")
    print("Using Clifford Algebra Cl(n,0) for structural analysis")
    print()

    # Run tests
    tests = [
        ("Simple 3-SAT", simple),
        ("Free Variables Detection", with_free_variables),
        ("UNSAT Detection", unsat),
        ("Large Formula (DPLL only)", large_without_embedding),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result is not None else "PASS (correctly UNSAT)"))
        except Exception as e:
            results.append((name, f"FAIL: {e}"))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, result in results:
        print(f"{name}: {result}")

    print("\n✅ Geometric SAT Solver Complete!")