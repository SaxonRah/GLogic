# Geometric Boolean Logic: A Complete Framework

> **Embedding Boolean logic into continuous geometric algebra**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Math: Clifford Algebra](https://img.shields.io/badge/Math-Clifford%20Algebra-blue.svg)]()
[![Theory: Complete](https://img.shields.io/badge/Theory-Complete-brightgreen.svg)]()

---

## Table of Contents

- [Overview](#overview)
- [The Algebra: Cl(n,0)](#the-algebra-cln0)
- [Boolean Variables as Signs](#boolean-variables-as-signs)
- [General Multivector Form](#general-multivector-form)
- [Embedding: Truth Tables to Geometry](#embedding-truth-tables-to-geometry)
- [Evaluation: Geometry to Boolean](#evaluation-geometry-to-boolean)
- [Truth Decision Threshold](#truth-decision-threshold)
- [The Core Insight](#the-core-insight)
- [Properties and Advantages](#properties-and-advantages)
- [Examples](#examples)
- [Implementation Notes](#implementation-notes)
- [Mathematical Summary](#mathematical-summary)

---

## Overview

### Goal

Embed Boolean logic with **n variables** into a **continuous geometric algebra** such that:

- ✅ Boolean logic is **exactly recovered** by evaluation
- ✅ **No logic gates or branching** are required
- ✅ All **correlations and structural relationships** are preserved
- ✅ **Continuous interpolation** between Boolean operations becomes possible
- ✅ **Instant correlation measurement** at all orders (O(1) complexity)

### Key Principle

> **Boolean logic is not executed — it is sampled.**

The multivector encodes the complete logical structure as a geometric object. Boolean truth values are recovered by evaluating this geometric object at specific points in the Boolean hypercube $\{-1, +1\}^n$.

---

## The Algebra: Cl(n,0)

For **n Boolean variables**, we use the Clifford algebra:

$$\mathrm{Cl}(n,0)$$

### Properties

**Basis vectors:** $e_1, e_2, \dots, e_n$

Each basis vector satisfies:
$$e_i^2 = +1$$

**Geometric product:** For distinct indices $i \neq j$:
$$e_i e_j = -e_j e_i$$

**Dimension:** The algebra has dimension:
$$\dim(\mathrm{Cl}(n,0)) = 2^n$$

This exactly matches the number of Boolean truth assignments!

### Basis Blades

The complete basis consists of all products of distinct basis vectors:

$$\{e_S \mid S \subseteq \{1,2,\dots,n\}\}$$

where $e_S = \prod_{i \in S} e_i$ (in sorted order) and $e_{\emptyset} = 1$ (the scalar).

**Examples:**

| n | Dimension | Basis Blades |
|---|-----------|--------------|
| 1 | 2 | $1, e_1$ |
| 2 | 4 | $1, e_1, e_2, e_{12}$ |
| 3 | 8 | $1, e_1, e_2, e_3, e_{12}, e_{13}, e_{23}, e_{123}$ |
| 4 | 16 | $1, e_1, \dots, e_4, e_{12}, \dots, e_{1234}$ |

---

## Boolean Variables as Signs

Each Boolean variable is mapped to a **sign value**:

$$p_i \in \{\text{False}, \text{True}\} \quad\longrightarrow\quad s_i \in \{-1, +1\}$$

### Convention

- **True** → $+1$
- **False** → $-1$

This mapping is crucial because:
1. It preserves the structure of Boolean operations
2. Enables polynomial evaluation on $\{-1, +1\}^n$
3. Makes geometric products correspond to logical operations

### The Boolean Hypercube

The $2^n$ possible Boolean assignments form the vertices of a hypercube in $\mathbb{R}^n$:

```
n=2 (Square):          n=3 (Cube):

  (-1,+1)-----(+1,+1)         (+1,+1,+1)
     |           |               /|    /|
     |           |              / |   / |
  (-1,-1)-----(+1,-1)      (-1,+1,+1) |
                              |  |  |  |
                              | (+1,-1,+1)
                              |/    |/
                           (-1,-1,-1)
```

---

## General Multivector Form

Any Boolean formula over $n$ variables is represented by a **multivector**:

$$F = \sum_{S \subseteq \{1,\dots,n\}} a_S \, e_S$$

where:
- $S$ is a subset of variable indices
- $e_S = \prod_{i \in S} e_i$ is the corresponding basis blade
- $a_S \in \mathbb{R}$ is a real coefficient

### Grade Structure

Multivectors decompose by **grade** (number of basis vectors in the product):

| Grade | Interpretation | Example (n=3) |
|-------|----------------|---------------|
| 0 | **Truth probability** | $a_{\emptyset} \cdot 1$ |
| 1 | **Variable bias** | $a_1 e_1 + a_2 e_2 + a_3 e_3$ |
| 2 | **Pairwise correlation** | $a_{12} e_{12} + a_{13} e_{13} + a_{23} e_{23}$ |
| 3 | **Three-way interaction** | $a_{123} e_{123}$ |
| $\vdots$ | $\vdots$ | $\vdots$ |
| n | **Global parity (XOR-like)** | $a_{1\dots n} e_{1\dots n}$ |

### Example: n=2

For two variables, a general formula looks like:

$$F = a_0 \cdot 1 + a_1 e_1 + a_2 e_2 + a_{12} e_{12}$$

- $a_0$: probability the formula is true
- $a_1$: bias toward $P_1$ being true
- $a_2$: bias toward $P_2$ being true  
- $a_{12}$: **correlation** between $P_1$ and $P_2$
  - Positive → variables tend to agree
  - Negative → variables tend to disagree
  - Zero → variables are independent

---

## Embedding: Truth Tables to Geometry

Given a Boolean formula with satisfying assignments $(p_1, \dots, p_n)$:

### Step 1: Convert to Signs

For each satisfying assignment, convert Boolean values to signs:

$$p_i = \text{True} \implies s_i = +1$$
$$p_i = \text{False} \implies s_i = -1$$

### Step 2: Build Projector

For each assignment $s = (s_1, \dots, s_n)$, construct the **quasi-projector**:

$$\Pi(s) = \prod_{i=1}^n \frac{1 + s_i e_i}{2}$$

This product expands to:

$$\Pi(s) = \frac{1}{2^n} \sum_{S \subseteq \{1,\dots,n\}} \left(\prod_{i \in S} s_i\right) e_S$$

**Properties of $\Pi(s)$:**
- Lives in the **Boolean cone** (positive linear combinations form all valid formulas)
- Evaluates to $\frac{1}{2^{n-1}}$ on the assignment $s$
- Evaluates to 0 on assignments maximally different from $s$

### Step 3: Sum Over Satisfying Assignments

The formula is the sum of all its satisfying projectors:

$$F = \sum_{\substack{(p_1,\dots,p_n) \\ \text{satisfies formula}}} \Pi(s_1,\dots,s_n)$$

### Example: AND for n=2

Truth table for $P_1 \land P_2$: only $(T, T)$ satisfies it.

1. Convert: $(T, T) \rightarrow s = (+1, +1)$
2. Build projector:
   $$\Pi(+1,+1) = \frac{1 + e_1}{2} \cdot \frac{1 + e_2}{2} = \frac{1}{4}(1 + e_1 + e_2 + e_{12})$$
3. Result:
   $$P_1 \land P_2 = \frac{1}{4} + \frac{1}{4}e_1 + \frac{1}{4}e_2 + \frac{1}{4}e_{12}$$

**Interpretation:**
- Probability: $\frac{1}{4}$ (25%)
- Both variables biased positive: $+\frac{1}{4}$
- Positive correlation: $+\frac{1}{4}$ (variables agree)

---

## Evaluation: Geometry to Boolean

To evaluate formula $F$ on a specific Boolean input $(p_1, \dots, p_n)$:

### Step 1: Convert Inputs to Signs

$$s_i = \begin{cases} +1 & \text{if } p_i = \text{True} \\ -1 & \text{if } p_i = \text{False} \end{cases}$$

### Step 2: Polynomial Evaluation

Treat the multivector as a **multilinear polynomial** on $\{-1, +1\}^n$:

$$F(s_1, \dots, s_n) = \sum_{S \subseteq \{1,\dots,n\}} a_S \prod_{i \in S} s_i$$

This means:
- Scalar component $a_{\emptyset}$ contributes as-is
- Each $e_i$ acts as "multiplication by $s_i$"
- Each $e_{ij}$ acts as "multiplication by $s_i s_j$"
- And so on for higher grades

### Step 3: Result

The evaluation produces a **real number** $r = F(s_1, \dots, s_n)$.

### Example: Evaluating AND

Formula: $P_1 \land P_2 = \frac{1}{4}(1 + e_1 + e_2 + e_{12})$

On input $(T, T) \rightarrow s = (+1, +1)$:
$$F(+1, +1) = \frac{1}{4}(1 + 1 + 1 + 1) = 1$$

On input $(T, F) \rightarrow s = (+1, -1)$:
$$F(+1, -1) = \frac{1}{4}(1 + 1 - 1 - 1) = 0$$

On input $(F, F) \rightarrow s = (-1, -1)$:
$$F(-1, -1) = \frac{1}{4}(1 - 1 - 1 + 1) = 0$$

---

## Truth Decision Threshold

The evaluation produces a real number. A **threshold** determines the Boolean result:

$$\text{result} \geq \text{threshold} \implies \text{True}$$
$$\text{result} < \text{threshold} \implies \text{False}$$

### Standard Threshold

For formulas embedded from truth tables:

$$\text{threshold} = \frac{1}{2^n}$$

**Why this value?**
- Each satisfying assignment contributes $\frac{1}{2^{n-1}}$ when evaluated at that point
- Non-satisfying assignments contribute $\approx 0$
- The threshold $\frac{1}{2^n}$ cleanly separates these cases

### Evaluation Table

For a properly embedded formula:

| Input Status | Evaluation Result | Boolean Output |
|--------------|-------------------|----------------|
| Satisfies formula | $\frac{1}{2^{n-1}}$ or higher | True |
| Doesn't satisfy | $< \frac{1}{2^n}$ | False |

---

## The Core Insight

### The Fundamental Transformation

```
Boolean Domain          Geometric Domain          Evaluation
─────────────────      ──────────────────        ────────────
                       
Truth Table    ─────>  Multivector       ─────>  Polynomial
{T,F}^n                in Cl(n,0)                on {-1,+1}^n
                       
Discrete                Continuous               Sampled
Logic                   Geometry                 Function
```

### Key Principles

1. **Boolean logic is not executed — it is sampled**
   - The multivector encodes all logical structure at once
   - Evaluation samples this structure at specific points
   - No branching or conditional logic needed

2. **Structure is preserved**
   - All correlations are explicit in the geometric form
   - Grade-k components encode k-way interactions
   - Composition via geometric product maintains structure

3. **Information is never destroyed**
   - Boolean logic: input → output (information lost)
   - Geometric logic: input → position in space (structure preserved)
   - Can analyze, transform, interpolate without losing information

4. **Continuous manifold**
   - Boolean operations are discrete samples
   - Geometric space allows smooth interpolation
   - "Fuzzy" intermediate states exist between Boolean operations

---

## Properties and Advantages

### Exact Equivalence

✅ **Complete expressiveness:** Every Boolean function has a unique multivector representation

✅ **Exact evaluation:** No approximation — Boolean semantics are perfectly preserved

✅ **Bijective mapping:** One-to-one correspondence between truth tables and multivectors (in the Boolean cone)

### Computational Efficiency

| Operation | Boolean Logic | Geometric Logic |
|-----------|---------------|-----------------|
| Storage | $2^n$ bits (truth table) | $2^n$ real coefficients |
| Evaluation | $O(n)$ (gate traversal) | $O(2^n)$ (polynomial eval) |
| Correlation | $O(2^n)$ (scan all inputs) | **O(1)** (read component) |
| Distance | Undefined | $O(2^n)$ (norm computation) |

**Key advantage:** Correlation measurement is **instant** — just read the appropriate coefficient!

### Beyond Boolean

1. **Smooth Interpolation**
   ```
   F_t = (1-t) · F_1 + t · F_2    (0 ≤ t ≤ 1)
   ```
   Creates continuous path between any two Boolean operations

2. **Distance Metric**
   ```
   d(F, G) = ||F - G|| = √(Σ(a_S - b_S)²)
   ```
   Measures how "different" two formulas are geometrically

3. **Geometric Transformations**
   - **Negation:** $\neg F = 1 - F$
   - **Rotation:** Change correlation orientation
   - **Reflection:** Mirror structure in variable space
   - **Projection:** Extract specific interaction orders

4. **Compositional Semantics**
   - Product of formulas includes interaction terms naturally
   - Correlations compose geometrically
   - No need to expand truth tables

### Information Hierarchy

```
Grade 0:  █████████████████  (Probability)
          How often is this true?

Grade 1:  ████ ████ ████     (Variable biases)  
          Which variables tend toward true?

Grade 2:  ██ ██ ██           (Pairwise correlations)
          How do pairs of variables relate?

Grade 3:  █ █                (3-way interactions)
          Complex conditional dependencies

   ⋮         ⋮                      ⋮

Grade n:  █                  (Global parity)
          Overall XOR structure
```

---

## Examples

### Example 1: XOR (n=2)

**Truth table:** $(T,F)$ and $(F,T)$ satisfy $P_1 \oplus P_2$

**Embedding:**
$$\Pi(+1,-1) = \frac{1}{4}(1 + e_1 - e_2 - e_{12})$$
$$\Pi(-1,+1) = \frac{1}{4}(1 - e_1 + e_2 - e_{12})$$

**Sum:**
$$P_1 \oplus P_2 = \frac{1}{2} \cdot 1 - \frac{1}{2} \cdot e_{12}$$

**Analysis:**
- Probability: $\frac{1}{2}$ (50% true)
- Variable biases: 0 (no preference)
- **Correlation: $-\frac{1}{2}$** (strongly disagree!)

### Example 2: IFF (n=2)

**Truth table:** $(T,T)$ and $(F,F)$ satisfy $P_1 \leftrightarrow P_2$

**Embedding:**
$$P_1 \leftrightarrow P_2 = \frac{1}{2} \cdot 1 + \frac{1}{2} \cdot e_{12}$$

**Analysis:**
- Probability: $\frac{1}{2}$ (50% true)
- Variable biases: 0 (no preference)
- **Correlation: $+\frac{1}{2}$** (strongly agree!)

**Key Insight:** XOR and IFF have the **same probability** (50%) but **opposite correlation** ($±\frac{1}{2}$)! Boolean logic cannot distinguish them structurally — geometric logic makes it obvious.

### Example 3: Majority Function (n=3)

**Truth table:** True when at least 2 of 3 variables are true

**Satisfying assignments:** $(T,T,T)$, $(T,T,F)$, $(T,F,T)$, $(F,T,T)$

**Result:**
$$\text{MAJ}_3 = \frac{1}{2} + \frac{1}{4}(e_{12} + e_{13} + e_{23})$$

**Analysis:**
- Probability: $\frac{1}{2}$ (50%)
- All pairwise correlations: $+\frac{1}{4}$ (variables tend to agree)
- Three-way interaction: 0

**Interpretation:** The majority function creates positive pairwise correlations because any two agreeing variables force the output.

### Example 4: 3-Way XOR (n=3)

**Truth table:** True when odd number of variables are true

**Result:**
$$\text{XOR}_3 = \frac{1}{2} - \frac{1}{2} e_{123}$$

**Analysis:**
- Probability: $\frac{1}{2}$ (50%)
- All pairwise correlations: 0 (cancel out)
- **Three-way interaction: $-\frac{1}{2}$** (pure parity!)

**Interpretation:** XOR has no pairwise structure — it's a global, grade-n phenomenon.

---

## Implementation Notes

### Efficient Representation

For sparse formulas (few satisfying assignments), store only non-zero coefficients:

```python
multivector = {
    frozenset(): 0.25,           # scalar
    frozenset({0}): 0.25,        # e_1
    frozenset({1}): 0.25,        # e_2
    frozenset({0,1}): 0.25       # e_12
}
```

### Evaluation Algorithm

```python
def evaluate(multivector, assignment):
    """
    Evaluate multivector at Boolean assignment.
    
    Args:
        multivector: dict mapping frozensets to coefficients
        assignment: tuple of bools
    
    Returns:
        float: evaluation result
    """
    signs = [1 if b else -1 for b in assignment]
    result = 0.0
    
    for subset, coeff in multivector.items():
        term = coeff
        for i in subset:
            term *= signs[i]
        result += term
    
    return result
```

### Optimization Tips

1. **Lazy evaluation:** Only compute needed components
2. **Caching:** Store frequently-used projectors
3. **Sparse representation:** Most real formulas have few terms
4. **Parallel evaluation:** Each term is independent

---

## Mathematical Summary

### The Framework in One Page

**Domain:** Boolean functions on $n$ variables

**Target:** Clifford algebra $\mathrm{Cl}(n,0)$ with dimension $2^n$

**Embedding:** Truth table $\rightarrow$ Multivector
$$F = \sum_{s \text{ satisfies}} \prod_{i=1}^n \frac{1 + s_i e_i}{2}$$

**Evaluation:** Multivector $\rightarrow$ Boolean function
$$F(s_1,\dots,s_n) = \sum_S a_S \prod_{i \in S} s_i$$

**Threshold:** $\text{threshold} = \frac{1}{2^n}$

**Structure:**
- Grade 0: Probability
- Grade 1-n: Correlations and interactions at all orders

**Operations:**
- Negation: $\neg F = 1 - F$
- Conjunction (approx): $F \cdot G$ (geometric product)
- Interpolation: $(1-t)F + tG$
- Distance: $\|F - G\|$

### The Fundamental Theorem

> **Every Boolean function on $n$ variables corresponds to a unique point in the Boolean cone of $\mathrm{Cl}(n,0)$, and can be exactly evaluated via polynomial computation on $\{-1,+1\}^n$.**

---

## Conclusion

### What We've Achieved

$$\boxed{\text{Boolean Logic} \subset \text{Geometric Logic}}$$

Geometric logic is a **strict superset** that:
- Contains all Boolean operations
- Adds continuous structure
- Preserves all information
- Enables new operations
- Maintains computational efficiency

### The Paradigm Shift

**Boolean thinking:** "Is it true or false?"

**Geometric thinking:** "Where does it live in structure space?"

### The Beautiful Truth

$$\textbf{Boolean logic is a projection.}$$

$$\textbf{Geometric logic is the space being projected from.}$$

The 16 Boolean operations for $n=2$ aren't isolated points — they're **samples** of an infinite 4-dimensional geometric manifold. The bivector $e_{12}$ was always there, hidden, encoding the correlation structure Boolean logic couldn't see.

By lifting Boolean logic into geometric algebra, we don't lose anything — we **gain** the continuous structure that was always implicitly present.

---

## References

### Further Reading

- Clifford Algebra: Hestenes, D. (2003). *Oersted Medal Lecture 2002: Reforming the mathematical language of physics*
- Boolean Functions: Crama, Y., & Hammer, P. L. (2011). *Boolean Functions: Theory, Algorithms, and Applications*
- Geometric Algebra: Dorst, L., Fontijne, D., & Mann, S. (2009). *Geometric Algebra for Computer Science*

### Related Work

- Quantum logic and phase space formulations
- Probabilistic logic and Bayesian networks
- Polynomial representations of Boolean functions
- Spectral methods in Boolean analysis
