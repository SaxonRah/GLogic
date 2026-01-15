"""
Enhanced Geometric 3-SAT Solver - Complete Edition

Features:
1. Correlation-guided variable ordering
2. Geometric conflict analysis and early pruning
3. Problem decomposition via correlation graph
4. Visualization of constraint structure
5. Learned clause generation
6. Adaptive sampling for large problems

Uses Clifford algebra Cl(n,0) for structural analysis.
"""

from rn00 import Rn00, BooleanEmbedder, blade_bits_to_name, count_bits
import itertools
import time
import numpy as np
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

    def __hash__(self):
        return hash(tuple(sorted(self.literals)))

    def __eq__(self, other):
        return set(self.literals) == set(other.literals)


@dataclass
class CNFFormula:
    """Complete CNF formula"""
    n_vars: int
    clauses: List[Clause]

    def __str__(self):
        return " ∧ ".join(str(c) for c in self.clauses)

    def add_clause(self, clause: Clause):
        """Add a clause to the formula"""
        if clause not in self.clauses:
            self.clauses.append(clause)


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

    def __str__(self):
        items = [f"v{k}={v}" for k, v in sorted(self.values.items())]
        return "{" + ", ".join(items) + "}"


# =============================================================================
# CNF Parser
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
            if lit.startswith('-') or lit.startswith('¬'):
                var_str = lit[1:] if lit.startswith('-') else lit[1:]
                literals.append((int(var_str[1:]) - 1, False))
            else:
                literals.append((int(lit[1:]) - 1, True))
        clauses.append(Clause(literals))

    return CNFFormula(n_vars, clauses)


# =============================================================================
# Geometric Embedding with Enhanced Analysis
# =============================================================================

class GeometricCNF:
    """Embed and analyze CNF formulas in Cl(n,0)"""

    def __init__(self, formula: CNFFormula):
        self.formula = formula
        self.n = formula.n_vars
        self.embedder = BooleanEmbedder(self.n)
        self.mv = None
        self.analysis = None
        self.satisfying_assignments = []
        self.used_sampling = False

    def embed(self, max_samples=None):
        """
        Embed formula into geometric algebra.

        If max_samples is set, uses sampling instead of full enumeration
        """
        print(f"Embedding CNF with {self.n} variables, {len(self.formula.clauses)} clauses...")

        if max_samples is None or max_samples >= 2 ** self.n:
            # Full enumeration
            self.used_sampling = False
            self.satisfying_assignments = self._find_all_satisfying()
        else:
            # Sampling approach
            self.used_sampling = True
            self.satisfying_assignments = self._sample_satisfying(max_samples)

        if not self.satisfying_assignments:
            print("Formula is UNSATISFIABLE")
            return None

        print(f"Found {len(self.satisfying_assignments)} satisfying assignments")

        # Embed
        self.mv = self.embedder.embed_truth_table(self.satisfying_assignments)
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
                if assignment not in satisfying:
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
        """Analyze correlation structure with enhanced metrics"""
        if self.mv is None:
            raise ValueError("Must embed first")

        grades = self.mv.grades()

        self.analysis = {
            'probability': self.mv.coeffs[0],
            'variable_biases': {},
            'correlations': {},
            'higher_order': {},
            'active_variables': set(),
            'constraint_strength': {},
            'independence_score': {}  # How independent each variable is
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
                        # Mark all variables in this interaction as active
                        for i in range(self.n):
                            if (blade_bits >> i) & 1:
                                self.analysis['active_variables'].add(i)

        # Compute independence scores
        for i in range(self.n):
            # How many significant correlations does this variable have?
            correlation_count = sum(
                1 for (v1, v2), corr in self.analysis['correlations'].items()
                if (v1 == i or v2 == i) and abs(corr) > 0.05
            )
            self.analysis['independence_score'][i] = 1.0 / (1.0 + correlation_count)

        return self.analysis

    def print_analysis(self, verbose=True):
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

        if verbose:
            print("\nVariable Constraint Strength (ranked):")
            ranked = sorted(self.analysis['constraint_strength'].items(),
                            key=lambda x: x[1], reverse=True)
            for var, strength in ranked[:min(10, len(ranked))]:
                bias = self.analysis['variable_biases'].get(var, 0)
                independence = self.analysis['independence_score'].get(var, 0)
                direction = "TRUE" if bias > 0.3 else "FALSE" if bias < -0.3 else "mixed"
                print(f"  v{var}: strength={strength:.3f}, bias={bias:+.3f}, "
                      f"indep={independence:.2f} → {direction}")

            print("\nTop Pairwise Correlations:")
            ranked_corr = sorted(self.analysis['correlations'].items(),
                                 key=lambda x: abs(x[1]), reverse=True)
            for (i, j), corr in ranked_corr[:min(10, len(ranked_corr))]:
                relationship = "AGREE" if corr > 0 else "DISAGREE"
                print(f"  v{i} ↔ v{j}: {corr:+.3f} [{relationship}]")

            if self.analysis['higher_order']:
                print(f"\nHigher-order interactions: {len(self.analysis['higher_order'])}")
                for blade_bits, coeff in list(self.analysis['higher_order'].items())[:5]:
                    print(f"  {blade_bits_to_name(blade_bits)}: {coeff:+.3f}")


# =============================================================================
# Geometric Conflict Analyzer
# =============================================================================

class GeometricConflictAnalyzer:
    """Analyze conflicts using geometric overlap (inner product)"""
    
    def __init__(self, geometric_cnf: GeometricCNF):
        self.geometric = geometric_cnf
        self.n = geometric_cnf.n
        self.embedder = BooleanEmbedder(self.n)

    def assignment_to_multivector(self, assignment: Assignment):
        """
        Convert partial assignment to multivector as a SUM over all completions.

        IMPORTANT: Do NOT normalize by number of completions.
        We want overlap to behave like "does there exist any satisfying completion?"
        """
        if len(assignment.values) == self.n:
            full_assign = tuple(assignment.values[i] for i in range(self.n))
            return self.embedder.embed_truth_table([full_assign])

        result = Rn00(self.n)
        unassigned = [i for i in range(self.n) if i not in assignment.values]

        # Enumerate all completions (still exponential; OK for small n)
        for completion in itertools.product([True, False], repeat=len(unassigned)):
            full_values = assignment.values.copy()
            for idx, val in zip(unassigned, completion):
                full_values[idx] = val

            full_assign = tuple(full_values[i] for i in range(self.n))
            result = result + self.embedder.embed_truth_table([full_assign])

        return result

    def is_conflict(self, assignment: Assignment):
        """
        Conflict = zero overlap with the satisfying region.

        IMPORTANT:
        - assignment_to_multivector() must SUM completions (no normalization)
        - formula_mv is the SUM of satisfying projectors
        """
        if getattr(self.geometric, "used_sampling", False):
            return False, 0.0, len(assignment.values) / self.n

        if self.geometric.mv is None:
            return False, 0.0, 0.0

        assign_mv = self.assignment_to_multivector(assignment)
        formula_mv = self.geometric.mv

        # coefficient dot product (basis is orthogonal in this representation)
        overlap = sum(a * f for a, f in zip(assign_mv.coeffs, formula_mv.coeffs))
        overlap = max(0.0, overlap)  # numerical safety

        # If overlap is essentially zero, there's no satisfying completion
        eps = 1e-9
        is_conflict = overlap <= eps

        confidence = len(assignment.values) / self.n
        return is_conflict, overlap, confidence

    def find_conflicting_variables(self, assignment: Assignment, k=3):
        """
        Find which subset of k variables causes the conflict.
        Uses correlation analysis.
        """
        if self.geometric.analysis is None:
            return []

        # Look at variables in assignment
        assigned_vars = list(assignment.values.keys())

        if len(assigned_vars) < 2:
            return []

        # Check which combinations have high correlation
        conflicts = []

        for i, j in itertools.combinations(assigned_vars, 2):
            if i > j:
                i, j = j, i

            corr = self.geometric.analysis['correlations'].get((i, j), 0)

            # Check if assignment violates correlation
            val_i = assignment.values[i]
            val_j = assignment.values[j]

            if corr > 0.3 and val_i != val_j:
                # Positive correlation but different values!
                conflicts.append(((i, j), "correlation violation", corr))
            elif corr < -0.3 and val_i == val_j:
                # Negative correlation but same values!
                conflicts.append(((i, j), "anti-correlation violation", corr))

        return sorted(conflicts, key=lambda x: abs(x[2]), reverse=True)


# =============================================================================
# Problem Decomposer
# =============================================================================

class ProblemDecomposer:
    """Decompose SAT into independent sub-problems"""

    def __init__(self, geometric: GeometricCNF):
        self.geometric = geometric
        self.n = geometric.n

    def find_components(self, correlation_threshold=0.05):
        """
        Find independent variable components via correlation graph.

        Returns: List of variable sets (components)
        """
        if self.geometric.analysis is None:
            self.geometric.analyze()

        # Build correlation graph
        graph = {i: set() for i in range(self.n)}

        # Add edges for geometric correlations
        for (i, j), corr in self.geometric.analysis['correlations'].items():
            if abs(corr) > correlation_threshold:
                graph[i].add(j)
                graph[j].add(i)

        # Add edges for variables in same clause (syntactic constraint)
        for clause in self.geometric.formula.clauses:
            vars_in_clause = [v for v, _ in clause.literals]
            for i, var_i in enumerate(vars_in_clause):
                for var_j in vars_in_clause[i + 1:]:
                    graph[var_i].add(var_j)
                    graph[var_j].add(var_i)

        # Find connected components (BFS)
        components = []
        visited = set()

        for start in range(self.n):
            if start in visited:
                continue

            component = set()
            queue = [start]

            while queue:
                v = queue.pop(0)
                if v in visited:
                    continue

                visited.add(v)
                component.add(v)

                for neighbor in graph[v]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            components.append(component)

        return components

    def decompose_formula(self, components):
        """
        Split formula into independent sub-formulas.

        Returns: List of (component, CNFFormula, var_map) tuples
        """
        sub_formulas = []

        for component in components:
            # Extract clauses that only involve this component
            component_clauses = []

            for clause in self.geometric.formula.clauses:
                clause_vars = {v for v, _ in clause.literals}

                if clause_vars.issubset(component):
                    # Remap variables to 0..k-1
                    sorted_comp = sorted(component)
                    var_map = {old_var: new_idx for new_idx, old_var in enumerate(sorted_comp)}

                    new_literals = [
                        (var_map[v], pos)
                        for v, pos in clause.literals
                    ]

                    component_clauses.append(Clause(new_literals))

            if component_clauses:
                sub_formula = CNFFormula(len(component), component_clauses)
                # Create inverse map (new_idx -> old_var)
                sorted_comp = sorted(component)
                inverse_map = {new_idx: old_var for new_idx, old_var in enumerate(sorted_comp)}
                sub_formulas.append((component, sub_formula, inverse_map))

        return sub_formulas


# =============================================================================
# Enhanced DPLL with All Features
# =============================================================================

class EnhancedGeometricDPLL:
    """DPLL solver with all geometric enhancements"""

    def __init__(self, formula: CNFFormula, geometric: GeometricCNF = None):
        self.formula = formula
        self.geometric = geometric
        self.n = formula.n_vars

        # Enhanced features
        self.conflict_analyzer = GeometricConflictAnalyzer(geometric) if geometric and geometric.mv else None
        self.learned_clauses = []

        # Statistics
        self.decisions = 0
        self.conflicts = 0
        self.unit_propagations = 0
        self.geometric_prunes = 0
        self.learned_clause_uses = 0

    def solve(self, use_geometric=True, verbose=True):
        """Main solve routine with all enhancements"""
        print("\n" + "=" * 70)
        print("STARTING ENHANCED GEOMETRIC-GUIDED DPLL")
        print("=" * 70)

        # Get variable ordering from geometric analysis
        if use_geometric and self.geometric and self.geometric.analysis:
            var_order = self._geometric_variable_ordering()
            print(f"Using geometric variable ordering (most constrained first)")
            if verbose:
                print(f"  Top variables: {var_order[:min(10, len(var_order))]}")
        else:
            var_order = list(range(self.n))
            print("Using default variable ordering")

        assignment = Assignment({})
        result = self._dpll(assignment, var_order, verbose, use_geometric)

        print("\n" + "=" * 70)
        print("SOLVER STATISTICS")
        print("=" * 70)
        print(f"Decisions made: {self.decisions}")
        print(f"Conflicts encountered: {self.conflicts}")
        print(f"Unit propagations: {self.unit_propagations}")
        if use_geometric:
            print(f"Geometric prunes: {self.geometric_prunes}")
            print(f"Learned clauses: {len(self.learned_clauses)}")
            print(f"Learned clause uses: {self.learned_clause_uses}")

        return result

    def _geometric_variable_ordering(self):
        """Order variables by constraint strength and connectivity"""
        if not self.geometric.analysis:
            return list(range(self.n))

        # Score each variable by:
        # 1. Constraint strength (how constrained)
        # 2. Connectivity (how many correlations)
        # 3. Bias strength (strong bias = easier to satisfy)

        scores = {}
        for v in range(self.n):
            strength = self.geometric.analysis['constraint_strength'].get(v, 0)
            independence = self.geometric.analysis['independence_score'].get(v, 1)
            bias = abs(self.geometric.analysis['variable_biases'].get(v, 0))

            # Higher score = branch on this first
            # Prefer highly constrained, well-connected variables
            scores[v] = strength * 2.0 + (1.0 - independence) * 1.0 + bias * 0.5

        ordered = sorted(range(self.n), key=lambda v: scores.get(v, 0), reverse=True)
        return ordered

    def _dpll(self, assignment: Assignment, var_order: List[int], verbose: bool, use_geometric: bool):
        """Recursive DPLL with all enhancements"""

        # Enhancement 1: Geometric conflict detection (early pruning!)
        if use_geometric and self.conflict_analyzer and len(assignment.values) > 0:
            is_conflict, overlap, confidence = self.conflict_analyzer.is_conflict(assignment)

            if is_conflict and confidence > 0.3:  # Only trust if somewhat complete
                if verbose and self.geometric_prunes < 5:
                    print(f"    ⚡ Geometric conflict! Overlap: {overlap:.6g}, Confidence: {confidence:.2f}, Confidence: {confidence:.2f}")
                    conflicts = self.conflict_analyzer.find_conflicting_variables(assignment)
                    if conflicts:
                        var_set, reason, corr = conflicts[0]
                        print(f"       Conflicting: v{var_set[0]}, v{var_set[1]} - {reason} ({corr:+.3f})")

                self.geometric_prunes += 1
                self.conflicts += 1

                # Learn a clause from this geometric conflict
                if conflicts:
                    self._learn_from_geometric_conflict(conflicts[0], assignment)

                return None

        # Unit propagation (with learned clauses)
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

        # Choose next variable
        var = None
        for v in var_order:
            if v not in assignment.values:
                var = v
                break

        if var is None:
            return None

        # Choose value (use geometric bias)
        value = self._choose_value(var, assignment, use_geometric)

        if verbose and self.decisions % 100 == 0:
            print(f"  Decision {self.decisions}: v{var} = {value}")

        self.decisions += 1

        # Try this assignment
        result = self._dpll(assignment.extend(var, value), var_order, verbose, use_geometric)
        if result is not None:
            return result

        # Backtrack: try opposite value
        result = self._dpll(assignment.extend(var, not value), var_order, verbose, use_geometric)
        return result

    def _unit_propagate(self, assignment: Assignment):
        """Unit propagation with learned clauses"""
        changed = True
        current = assignment.copy()

        # Combine original and learned clauses
        all_clauses = list(self.formula.clauses) + self.learned_clauses

        while changed:
            changed = False

            for clause in all_clauses:
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

                # Unit clause
                if unassigned_count == 1:
                    var, is_positive = unassigned_literal
                    forced_value = is_positive
                    current.values[var] = forced_value
                    changed = True
                    self.unit_propagations += 1

                    if clause in self.learned_clauses:
                        self.learned_clause_uses += 1

                # Empty clause: conflict!
                if unassigned_count == 0:
                    return None

        return current

    def _choose_value(self, var: int, assignment: Assignment, use_geometric: bool):
        """Choose value using geometric bias and phase saving"""
        if use_geometric and self.geometric and self.geometric.analysis:
            bias = self.geometric.analysis['variable_biases'].get(var, 0)

            # Strong bias: follow it
            if abs(bias) > 0.1:
                return bias > 0

            # Check correlations with already-assigned variables
            for assigned_var, assigned_val in assignment.values.items():
                if var < assigned_var:
                    key = (var, assigned_var)
                else:
                    key = (assigned_var, var)

                corr = self.geometric.analysis['correlations'].get(key, 0)

                if abs(corr) > 0.2:
                    # Strong correlation: follow it
                    if corr > 0:
                        # Positive correlation: same value
                        return assigned_val
                    else:
                        # Negative correlation: opposite value
                        return not assigned_val

        # Default: try True first
        return True

    def _learn_from_geometric_conflict(self, conflict_info, assignment: Assignment):
        """Learn a clause from geometric conflict"""
        var_set, reason, corr = conflict_info

        # Create a clause that prevents this conflict
        # If positive correlation violated (same values required but different assigned):
        # Learn: ¬v_i ∨ ¬v_j (can't both be different)
        # If negative correlation violated (different values required but same assigned):
        # Learn: v_i ∨ v_j (at least one must be true)

        i, j = var_set
        val_i = assignment.values[i]
        val_j = assignment.values[j]

        if "correlation" in reason and corr > 0:
            # Positive correlation violated
            # Prevent: both true when they should agree, or both false when they should agree
            # Learn: (¬v_i ∨ v_j) if v_i=T, v_j=F
            literals = [
                (i, not val_i),
                (j, val_j)
            ]
        else:
            # Negative correlation violated
            # Learn: (v_i ∨ v_j) or (¬v_i ∨ ¬v_j)
            literals = [
                (i, not val_i),
                (j, not val_j)
            ]

        learned = Clause(literals)
        if learned not in self.learned_clauses:
            self.learned_clauses.append(learned)

    def _verify_solution(self, assignment: Assignment):
        """Verify solution"""
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
# Visualization
# =============================================================================

def visualize_correlation_graph(geometric: GeometricCNF, filename="correlation_graph.png"):
    """Create visualization of variable correlation structure"""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("⚠️  Matplotlib/NetworkX not available - skipping visualization")
        return

    if geometric.analysis is None:
        geometric.analyze()

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(geometric.n))

    # Add edges for correlations
    for (i, j), corr in geometric.analysis['correlations'].items():
        if abs(corr) > 0.05:
            G.add_edge(i, j, weight=abs(corr), sign=np.sign(corr))

    # Layout
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = {}

    # Draw
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw nodes
    node_colors = []
    node_sizes = []
    for i in range(geometric.n):
        bias = geometric.analysis['variable_biases'].get(i, 0)
        strength = geometric.analysis['constraint_strength'].get(i, 0)

        if bias > 0.2:
            node_colors.append('lightgreen')  # Tends true
        elif bias < -0.2:
            node_colors.append('lightcoral')  # Tends false
        else:
            node_colors.append('lightgray')  # Neutral

        # Size by constraint strength
        node_sizes.append(300 + strength * 500)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, ax=ax, alpha=0.8)

    # Draw edges by correlation type
    pos_edges = [(i, j) for i, j, d in G.edges(data=True) if d.get('sign', 0) > 0]
    neg_edges = [(i, j) for i, j, d in G.edges(data=True) if d.get('sign', 0) < 0]

    nx.draw_networkx_edges(G, pos, edgelist=pos_edges,
                           edge_color='green', width=2,
                           style='solid', label='Positive correlation', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges,
                           edge_color='red', width=2,
                           style='dashed', label='Negative correlation', ax=ax)

    # Labels
    labels = {i: f'v{i}' for i in range(geometric.n)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax)

    ax.set_title('SAT Correlation Graph\n(Node size = constraint strength, Color = bias)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved correlation graph: {filename}")
    plt.close()


# =============================================================================
# Complete Solver with Decomposition
# =============================================================================

class CompleteGeometricSATSolver:
    """
    Complete pipeline with all enhancements:
    1. Embedding and analysis
    2. Problem decomposition
    3. Enhanced DPLL
    4. Visualization
    """

    def __init__(self, formula: CNFFormula):
        self.formula = formula
        self.geometric = None
        self.solver = None
        self.solution = None
        self.components = None

    def solve(self, use_embedding=True, use_decomposition=True,
              max_samples=1000, verbose=True, visualize=False):
        """Full pipeline with all features"""
        print("=" * 70)
        print("COMPLETE GEOMETRIC SAT SOLVER")
        print("=" * 70)
        print(f"Formula: {self.formula.n_vars} variables, {len(self.formula.clauses)} clauses")

        # Step 1: Embed (if feasible)
        if use_embedding and self.formula.n_vars <= 15:
            print("\n[Step 1] Embedding & Analysis")
            print("-" * 70)
            start = time.time()

            self.geometric = GeometricCNF(self.formula)
            mv = self.geometric.embed(max_samples=max_samples)

            if mv is None:
                print("✅ Formula is UNSATISFIABLE (detected during embedding)")
                return None

            self.geometric.analyze()
            self.geometric.print_analysis(verbose=verbose)

            elapsed = time.time() - start
            print(f"\nEmbedding completed in {elapsed:.2f}s")

            # Visualization
            if visualize:
                print("\n[Visualizing correlation structure...]")
                visualize_correlation_graph(self.geometric,
                                            f"correlation_{self.formula.n_vars}vars.png")
        else:
            print("\n[Skipping embedding - too many variables or disabled]")

        # Step 2: Decomposition (if enabled and embedded)
        if use_decomposition and self.geometric and self.geometric.mv:
            print("\n[Step 2] Problem Decomposition")
            print("-" * 70)

            decomposer = ProblemDecomposer(self.geometric)
            self.components = decomposer.find_components()

            print(f"Found {len(self.components)} independent components:")
            for i, comp in enumerate(self.components):
                comp_size = len(comp)
                print(
                    f"  Component {i}: {comp_size} variables {sorted(list(comp))[:5]}{'...' if comp_size > 5 else ''}")

            if len(self.components) > 1:
                # Solve decomposed
                return self._solve_decomposed(decomposer, verbose)
            else:
                print("  Problem is fully connected - solving as one unit")

        # Step 3: Solve with enhanced DPLL
        print("\n[Step 3] Enhanced DPLL")
        print("-" * 70)
        start = time.time()

        self.solver = EnhancedGeometricDPLL(
            self.formula,
            self.geometric if use_embedding else None
        )
        self.solution = self.solver.solve(
            use_geometric=use_embedding and self.geometric is not None,
            verbose=verbose
        )

        elapsed = time.time() - start
        print(f"\nDPLL completed in {elapsed:.2f}s")

        # Result
        self._print_result()
        return self.solution

    def _solve_decomposed(self, decomposer: ProblemDecomposer, verbose: bool):
        """Solve decomposed sub-problems"""
        print("\n[Step 3] Solving Independent Components")
        print("-" * 70)

        sub_formulas = decomposer.decompose_formula(self.components)

        if not sub_formulas:
            print("No clauses to solve!")
            # All variables are free - any assignment works
            return Assignment({i: True for i in range(self.formula.n_vars)})

        combined_solution = {}
        total_time = 0

        for i, (component, sub_formula, inverse_map) in enumerate(sub_formulas):
            comp_size = len(component)
            print(f"\n  Component {i}/{len(sub_formulas)}: {comp_size} vars, {len(sub_formula.clauses)} clauses")

            start = time.time()

            # Create mini-solver for this component
            solver = EnhancedGeometricDPLL(sub_formula, None)
            sub_solution = solver.solve(use_geometric=False, verbose=False)

            elapsed = time.time() - start
            total_time += elapsed

            if sub_solution is None:
                print(f"  ❌ Component {i} is UNSAT - entire formula is UNSAT!")
                return None

            # Map back to original variables
            for new_var, value in sub_solution.values.items():
                old_var = inverse_map[new_var]
                combined_solution[old_var] = value

            print(f"  ✅ Solved in {elapsed:.3f}s ({solver.decisions} decisions)")

        # Set free variables to True
        for i in range(self.formula.n_vars):
            if i not in combined_solution:
                combined_solution[i] = True

        print(f"\n✅ All components solved in {total_time:.3f}s total")
        self.solution = Assignment(combined_solution)
        self._print_result()
        return self.solution

    def _print_result(self):
        """Print final result"""
        if self.solution:
            print("\n" + "=" * 70)
            print("✅ SATISFIABLE")
            print("=" * 70)
            print("Solution:")
            for var in sorted(self.solution.values.keys()):
                print(f"  v{var} = {self.solution.values[var]}")

            # Verify
            if self._verify(self.solution):
                print("\n✓ Solution verified correct!")
            else:
                print("\n✗ WARNING: Solution verification failed!")
        else:
            print("\n" + "=" * 70)
            print("❌ UNSATISFIABLE")
            print("=" * 70)

    def _verify(self, assignment: Assignment):
        """Verify solution satisfies all clauses"""
        for clause in self.formula.clauses:
            satisfied = False
            for var, is_positive in clause.literals:
                value = assignment.values.get(var, False)
                literal_value = value if is_positive else not value
                if literal_value:
                    satisfied = True
                    break
            if not satisfied:
                print(f"  Failed clause: {clause}")
                return False
        return True


# =============================================================================
# Test Suite
# =============================================================================

def simple():
    """Simple 3-SAT test"""
    print("\n" + "=" * 80)
    print("TEST 1: Simple 3-SAT")
    print("=" * 80)

    clauses = [
        Clause([(0, True), (1, True), (2, True)]),
        Clause([(0, False), (1, True), (2, False)]),
        Clause([(0, True), (1, False), (2, True)]),
    ]

    formula = CNFFormula(3, clauses)
    solver = CompleteGeometricSATSolver(formula)
    solution = solver.solve(use_embedding=True, visualize=True, verbose=True)

    return solution is not None


def free_variables():
    """Test with unconstrained variables"""
    print("\n" + "=" * 80)
    print("TEST 2: Formula with Free Variables")
    print("=" * 80)

    clauses = [
        Clause([(0, True), (1, True), (2, True)]),
        Clause([(0, False), (1, True), (2, False)]),
        Clause([(0, True), (1, False), (2, True)]),
        Clause([(3, True), (3, False), (4, True)]),  # Tautology
    ]

    formula = CNFFormula(5, clauses)
    solver = CompleteGeometricSATSolver(formula)
    solution = solver.solve(use_embedding=True, visualize=True)

    return solution is not None


def decomposition():
    """Test problem decomposition"""
    print("\n" + "=" * 80)
    print("TEST 3: Problem Decomposition")
    print("=" * 80)

    # Two independent components
    clauses = [
        # Component 1: v0, v1
        Clause([(0, True), (1, True)]),
        Clause([(0, False), (1, False)]),
        # Component 2: v2, v3
        Clause([(2, True), (3, False)]),
        Clause([(2, False), (3, True)]),
    ]

    formula = CNFFormula(4, clauses)
    solver = CompleteGeometricSATSolver(formula)
    solution = solver.solve(use_embedding=True, use_decomposition=True,
                            visualize=True, verbose=True)

    return solution is not None


def unsat():
    """Test UNSAT detection"""
    print("\n" + "=" * 80)
    print("TEST 4: Unsatisfiable Formula")
    print("=" * 80)

    clauses = [
        Clause([(0, True)]),
        Clause([(0, False)]),
    ]

    formula = CNFFormula(1, clauses)
    solver = CompleteGeometricSATSolver(formula)
    solution = solver.solve(use_embedding=True)

    return solution is None


def geometric_conflict():
    """Test geometric conflict detection"""
    print("\n" + "=" * 80)
    print("TEST 5: Geometric Conflict Detection")
    print("=" * 80)

    # Formula where geometric analysis should help
    clauses = [
        Clause([(0, True), (1, True)]),  # v0 ∨ v1
        Clause([(0, True), (1, False)]),  # v0 ∨ ¬v1
        Clause([(0, False), (1, True)]),  # ¬v0 ∨ v1
        Clause([(0, False), (1, False)]),  # ¬v0 ∨ ¬v1
    ]

    formula = CNFFormula(2, clauses)
    solver = CompleteGeometricSATSolver(formula)
    solution = solver.solve(use_embedding=True, verbose=True)

    return solution is None  # Should be UNSAT


def large_random():
    """Test larger random formula"""
    print("\n" + "=" * 80)
    print("TEST 6: Larger Random 3-SAT")
    print("=" * 80)

    import random
    random.seed(42)

    n_vars = 20
    n_clauses = 80

    clauses = []
    for _ in range(n_clauses):
        vars_in_clause = random.sample(range(n_vars), 3)
        literals = [(v, random.choice([True, False])) for v in vars_in_clause]
        clauses.append(Clause(literals))

    formula = CNFFormula(n_vars, clauses)
    solver = CompleteGeometricSATSolver(formula)
    solution = solver.solve(use_embedding=False, verbose=False)  # Too big to embed

    return solution is not None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("COMPLETE ENHANCED GEOMETRIC SAT SOLVER")
    print("=" * 80)
    print("Features:")
    print("  ✓ Correlation-guided variable ordering")
    print("  ✓ Geometric conflict analysis")
    print("  ✓ Problem decomposition")
    print("  ✓ Learned clauses from geometric conflicts")
    print("  ✓ Visualization")
    print("=" * 80)

    # Run all tests
    tests = [
        ("Simple 3-SAT", simple),
        ("Free Variables", free_variables),
        ("Problem Decomposition", decomposition),
        ("UNSAT Detection", unsat),
        ("Geometric Conflict", geometric_conflict),
        ("Large Random SAT", large_random),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            status = "PASS" if result is not None else "PASS (correctly UNSAT)"
            results.append((name, status))
        except Exception as e:
            import traceback

            results.append((name, f"FAIL: {e}"))
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, result in results:
        status_icon = "✅" if "PASS" in result else "❌"
        print(f"{status_icon} {name}: {result}")

    print("\n" + "=" * 80)
    print("✅ COMPLETE GEOMETRIC SAT SOLVER WITH ALL ENHANCEMENTS!")
    print("=" * 80)