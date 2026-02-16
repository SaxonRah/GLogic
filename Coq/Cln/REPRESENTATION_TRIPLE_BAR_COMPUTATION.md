# Representation ≡ Computation (via Geometric Algebra)

*A program for a representation-theoretic complexity invariant*

---

## 1. Core Thesis

We propose a model where Boolean computation is understood as:

1. **Reversible algebraic dynamics** — computation occurs inside a multivector algebra via the geometric product (invertible for *certain* classes of elements, e.g. blades/versors under appropriate non-degeneracy conditions).
2. **Constrained by conservation-type laws** — ℓ₁ submultiplicativity bounds representational "mass."
3. **Terminated by irreversible measurement** — projection/evaluation back to Boolean outputs collapses the richer structure.

> **Complexity = minimum unavoidable representational excursion**
> forced inside this reversible algebra before projection can produce a Boolean answer.

This defines a **model-relative invariant**: geometric representational complexity.

---

## 2. Objects and Formal Model

### 2.1 Boolean Domain

```math
\text{Inputs: } \{\pm 1\}^n
```

```math
\text{Boolean functions: } f : \{\pm 1\}^n \to \{0,1\}
```

### 2.2 Multivector Representation

Let

```math
MV_n := \{F : \mathcal{P}([n]) \to \mathbb{Q}\}
```

with basis blades $e_A$ indexed by masks $A \subseteq [n]$.

```math
F = \sum_{A \subseteq [n]} F(A)\,e_A
```

Grade:

```math
\mathrm{grade}(A) = |A|
```

### 2.3 Geometric Product

For blades:

```math
e_A \star e_B = c(A,B)\, e_{A \oplus B}
```

where:

* $A \oplus B$ = symmetric difference (XOR of masks)
* $c(A,B) \in {\pm 1}$ determined by swaps parity and metric signature (unit-metric case)

Extend bilinearly to all multivectors.

### 2.4 Embedding Boolean Functions

```math
\mathrm{embed}(f) \in MV_n
```

such that:

```math
\mathrm{eval}(\mathrm{embed}(f), s) = f(s) \quad \forall s \in \{\pm1\}^n
```

**Important distinction:** there are (at least) two natural embedding styles, and *parity/XOR looks different in each*.

| Embedding style                                | What basis means                      | What XOR looks like                                                                         | Notes                                                      |
| ---------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Delta-basis (projector / point-mass expansion) | basis elements are “input projectors” | generally a *mixture* across multiple grades; often includes scalar + top-grade             | this is the style used in the concrete `n=2,3` experiments |
| Walsh-character basis (Fourier/Walsh basis)    | basis elements are parity characters  | XOR/parity is supported at *top degree* (plus possibly a scalar offset if mapping to {0,1}) | aligns directly with Fourier analysis                      |

In general, we will write parity’s embedding in the form:

```math
\mathrm{embed}(\text{XOR}_n) = \alpha_n \cdot 1 + \beta_n \cdot e_{[n]} + R_n
```

where:

* $\beta_n \neq 0$ captures the **nonzero pseudoscalar (grade-$n$) component**,
* $R_n$ captures any additional lower-grade structure that may appear depending on the embedding style/normalization.

> The *only* guaranteed claim at this stage is: **the grade-$n$ coefficient is nonzero for parity** (in the constructions we care about), not a specific closed form like $1/2$ or $1/2^n$ unless proved.

---

## 3. Foundational Results (Mechanized in Coq)

| Result                           | File                              | Interpretation                                                       |
| -------------------------------- | --------------------------------- | -------------------------------------------------------------------- |
| `embed_correct`                  | (general embedding file)          | Boolean truth tables preserved under embedding                       |
| `embed3_correct`                 | concrete `n=3` file               | Concrete n=3 verification                                            |
| `mv_gp_assoc`                    | general GP file                   | Geometric product is associative                                     |
| `geom_prod_leaves_boolean_space` | concrete `n=2,3` file             | Boolean multivectors not closed under ⋆                              |
| `l1_gp_submultiplicative`        | `finite_l1_submultiplicativity.v` | ℓ₁ mass bounded under geometric product (unit-metric)                |
| parity-grade lemma(s)            | concrete `n=3` file               | Parity has a nonzero highest-grade component in the tested instances |
| `e_square`, `e_anticomm`         | general algebra file              | Clifford relations / sign rules                                      |

> Note: exact file names should match your repo. Keep this table synced to the actual structure.

---

## 4. Algebraic Laws of Computation

### Law 1 — Reversibility (Qualified)

There exist important classes of multivectors for which inversion is definable and provable (e.g. **basis blades** and products of invertible blades / **versors** under suitable non-degeneracy assumptions).

A safe starting point for mechanization is blade invertibility:

```math
e_A \star e_A^{-1} = 1
\quad\text{(for non-null blades, with an explicitly defined inverse)}
```

> We do **not** assume “every multivector has an inverse.” Invertibility is a property to prove for a class of elements, not a global axiom.

### Law 2 — Norm Constraint (ℓ₁ Submultiplicativity)

```math
\|F \star G\|_1 \le \|F\|_1 \cdot \|G\|_1
```

This is proved (in your development) under the “unit metric” assumptions ensuring $|c(A,B)|=1$.

### Law 3 — Projection Irreversibility

The evaluation map

```math
\mathrm{eval}: MV_n \times \{\pm 1\}^n \to \mathbb{Q}
```

recovers Boolean values on embedded Boolean functions, but in general returns rationals for intermediate multivectors. Any *Boolean decision* requires an additional projection/thresholding step, which is information-losing.

### Law 4 — Complexity as Excursion

Hard functions force large excursions in representation space before projection can yield a Boolean answer.

---

## 5. Excursion Measures

### 5.1 Boolean Distance

```math
\mathrm{BoolDist}(F) = \inf_{g:\{\pm1\}^n\to\{0,1\}} \|F - \mathrm{embed}(g)\|_1
```

> This is defined as an **infimum over all Boolean functions**, not by enumerating them (since there are $2^{2^n}$ of them).

### 5.2 Grade Support

```math
\mathrm{GradeSupp}(F) = \{k : \exists A,\ |A|=k,\ F(A)\neq 0\}
```

```math
\mathrm{MaxGrade}(F) = \max(\mathrm{GradeSupp}(F))
```

### 5.3 Support Size

```math
\mathrm{Supp}(F) = |\{A : F(A) \neq 0\}|
```

### 5.4 Composite Measure (Lexicographic)

```math
\mathcal{E}(F) := (\mathrm{MaxGrade}(F),\ \mathrm{BoolDist}(F))
```

ordered lexicographically. This avoids logs and is Coq-friendly.

---

## 6. Geometric Representational Complexity

```math
\mathcal{E}(f) = \inf_{\text{GA programs computing } f} \ \sup_t \ \operatorname{Dist}(F_t, \mathcal{B})
```

where:

* $F_t$ = intermediate multivector state at step $t$
* $\mathcal{B}$ = Boolean subspace ${\mathrm{embed}(g) : g \text{ Boolean}}$
* $\mathrm{Dist}$ = chosen excursion metric (e.g. BoolDist, MaxGrade, support, or a combination)

> We use $\inf/\sup$ (not min/max) because existence of exact minimizers may require extra finiteness/compactness assumptions.

---

## 7. Grade Dynamics (Key Structural Insight)

### 7.1 Grade Under XOR of Masks

For masks $A, B$:

```math
\mathrm{grade}(A \oplus B) = |A| + |B| - 2|A \cap B|
```

This implies:

```math
\big||A|-|B|\big| \le \mathrm{grade}(A \oplus B) \le |A| + |B|
```

and a parity constraint (grades change in steps of 2).

### 7.2 Grade Support Closure Under Product

```math
\mathrm{GradeSupp}(F \star G) \subseteq \bigcup_{j \in S_F,\, k \in S_G} \{|j-k|,\, |j-k|+2,\, \ldots,\, j+k\}
```

### 7.3 Grade Bounds (Inequalities)

```math
\mathrm{MaxGrade}(F \star G) \le \mathrm{MaxGrade}(F) + \mathrm{MaxGrade}(G)
```

```math
\mathrm{MaxGrade}(F + G) \le \max(\mathrm{MaxGrade}(F), \mathrm{MaxGrade}(G))
```

> Equality can fail due to cancellation, so we state the general (always true) inequality.

---

## 8. First Lower-Bound Target: Parity

### 8.1 Parity Embedding Structure (What we actually need)

We only require:

```math
\text{The coefficient of } e_{[n]} \text{ in } \mathrm{embed}(\text{XOR}_n) \text{ is nonzero.}
```

Equivalently:

```math
\mathrm{embed}(\text{XOR}_n)([n]) \neq 0.
```

This is the “spectral signature” needed for a grade excursion bound.

### 8.2 Grade Reachability Theorem (Careful statement)

> **Theorem (Parity Excursion Lower Bound — construction form):**
> Any GA program that *constructs* $\mathrm{embed}(\text{XOR}_n)$ from basis vectors $e_1, \ldots, e_n$ using $+$ and $\star$ must produce an intermediate state with $\mathrm{MaxGrade} \ge n$ (hence $=n$ since grades are in $[0,n]$).

**Proof sketch:**

1. Basis vectors have grade 1, scalars have grade 0
2. Products obey $\mathrm{MaxGrade}(F \star G) \le \mathrm{MaxGrade}(F) + \mathrm{MaxGrade}(G)$
3. Sums obey $\mathrm{MaxGrade}(F + G) \le \max(\mathrm{MaxGrade}(F), \mathrm{MaxGrade}(G))$
4. Final constructed multivector has a nonzero grade-$n$ coefficient
5. Therefore the maximum grade attained during construction is at least $n$

> This is a **first genuine excursion lower bound**: producing parity’s representation forces the computation to reach top grade (in this construction sense).

---

## 9. GA Programs (Formal Definition)

```coq
Inductive GA_expr (n : nat) : Type :=
  | Basis  : Fin.t n -> GA_expr n           (* basis vector e_i *)
  | Scalar : Q -> GA_expr n                 (* scalar constant *)
  | Add    : GA_expr n -> GA_expr n -> GA_expr n
  | Mul    : GA_expr n -> GA_expr n -> GA_expr n.  (* geometric product *)

Fixpoint eval_expr {n} (sq : Vector.t Q n) (e : GA_expr n) : MV n :=
  match e with
  | Basis i    => basis (mask_single i)
  | Scalar c   => mv_scale c mv_one
  | Add e1 e2  => mv_add (eval_expr sq e1) (eval_expr sq e2)
  | Mul e1 e2  => mv_gp n sq (eval_expr sq e1) (eval_expr sq e2)
  end.
```

A purely syntactic “max grade during evaluation” should avoid re-evaluating products in a way that presumes properties not yet proved. A safer approach is:

* define `max_grade (eval_expr sq e)` (final grade),
* and define a separate traversal function that takes `max` over all subexpressions’ final grades.

```coq
Fixpoint max_grade_during {n} (sq : Vector.t Q n) (e : GA_expr n) : nat :=
  match e with
  | Basis _   => 1
  | Scalar _  => 0
  | Add e1 e2 => max (max_grade_during sq e1) (max_grade_during sq e2)
  | Mul e1 e2 =>
      max (max (max_grade_during sq e1) (max_grade_during sq e2))
          (max_grade (mv_gp n sq (eval_expr sq e1) (eval_expr sq e2)))
  end.
```

---

## 10. Coq Development Roadmap

### Phase 1: Grade Infrastructure (List-based, proof-friendly)

This follows the pattern already used across your files (heavy `Vector.to_list` + list lemmas). We count `true` entries using `count_occ`, which tends to rewrite more cleanly than `filter/length`.

```coq
(* Grade = popcount of mask (count trues in the list view) *)
Definition grade {n} (m : Mask n) : nat :=
  List.count_occ Bool.bool_dec (Vector.to_list m) true.

Definition max_grade {n} (F : MV n) : nat :=
  fold_right max 0
    (map (fun m => if Qeq_bool (F m) 0 then 0 else grade m)
         (all_masks n)).

(* Full mask has grade n *)
Lemma grade_full_mask : forall n,
  grade (Vector.const true n) = n.
Proof.
Admitted.

(* Grade under XOR/symmetric difference *)
Lemma grade_xor_formula : forall n (A B : Mask n),
  grade (mask_xor A B)
  = grade A + grade B - 2 * grade (mask_and A B).
Proof.
Admitted.
```

> If your `Mask n` is `Vector.t bool n` (as in your current development), this definition is consistent with the list-based style you already use for `swaps_parity`, `metric_factor`, and related combinatorics.

### Phase 2: Grade Evolution Lemmas

Rather than “nonzero sum implies a nonzero term” (finicky), start with the contrapositive “if all summands are zero then the sum is zero.”

```coq
(* If every summand is zero, the coefficient is zero. *)
Lemma gp_coeff_zero_if_all_summands_zero :
  forall n sq (F G : MV n) (U : Mask n),
    (forall A : Mask n,
        (* replace with your actual summand expression in mv_gp *)
        True) ->
    mv_gp n sq F G U == 0.
Proof.
Admitted.
```

Then derive grade bounds:

```coq
Lemma max_grade_gp_bound : forall n sq (F G : MV n),
  max_grade (mv_gp n sq F G) <= max_grade F + max_grade G.
Proof.
Admitted.

Lemma max_grade_add_bound : forall n (F G : MV n),
  max_grade (mv_add F G) <= max (max_grade F) (max_grade G).
Proof.
Admitted.
```

### Phase 3: Parity Structure (only what’s needed)

Define XOR/parity:

```coq
Definition XOR_n_func {n} (c : Corner n) : bool :=
  fold_right xorb false
    (Vector.to_list
      (Vector.map (fun s => match s with Pos => true | Neg => false end) c)).
```

Prove the key spectral signature:

```coq
Theorem XOR_has_grade_n_component : forall n,
  n > 0 ->
  embed (@XOR_n_func n) (Vector.const true n) <> 0.
Proof.
Admitted.
```

> Do **not** assert a closed form like `1/(2^n)` unless you’ve derived it from your exact embedding definition.

### Phase 4: Excursion Lower Bound (construction form)

```coq
Definition builds_XOR {n} (sq : Vector.t Q n) (e : GA_expr n) : Prop :=
  eval_expr sq e = embed (@XOR_n_func n).

Theorem parity_excursion_lower_bound : forall n sq (e : GA_expr n),
  n > 0 ->
  builds_XOR sq e ->
  max_grade_during sq e >= n.
Proof.
Admitted.
```

### Phase 5: ℓ₁ Trajectory Analysis

```coq
Lemma l1_add_bound : forall n (F G : MV n),
  l1_norm (mv_add F G) <= l1_norm F + l1_norm G.
Proof.
Admitted.

Lemma l1_gp_bound : forall n sq (F G : MV n),
  l1_norm (mv_gp n sq F G) <= l1_norm F * l1_norm G.
Proof.
Admitted.
```

### Phase 6: Boolean Distance (Infimum form)

```coq
(* A Prop-based definition: d is an upper bound if some boolean g achieves it. *)
Definition BoolDist_upper {n} (F : MV n) (d : Q) : Prop :=
  exists g : Corner n -> bool,
    d == l1_norm (mv_sub F (embed g)).

(* BoolDist as an infimum of upper bounds. *)
Definition boolean_distance {n} (F : MV n) : Q :=
  infimum (BoolDist_upper F).

Lemma boolean_distance_zero_iff : forall n (F : MV n),
  boolean_distance F == 0 ->
  exists g, F = embed g.
Proof.
Admitted.
```

---

## 11. Research Program

### A — Simulation Theorems

Establish complexity-preserving translations:

| Source Model              | Target         | Overhead |
| ------------------------- | -------------- | -------- |
| Boolean formulas          | GA expressions | O(size)  |
| Circuits (bounded fan-in) | GA programs    | O(size)  |

**Caution:** “depth ↔ grade” is not automatic; it requires a carefully designed encoding. Start with formulas and prove what you can.

### B — Upper Bounds (Easy Functions)

Show small excursion suffices for families like AND/OR/decision trees, with respect to *chosen* excursion metrics (grade, ℓ₁, BoolDist).

### C — Lower Bounds (Hard Functions)

Prove unavoidable excursion:

```math
\mathcal{E}(\text{XOR}_n) \ge n \quad \text{(via MaxGrade)}
```

### D — Separation Theorem

Ultimate goal:

```math
\exists f_n : \mathcal{E}(f_n) \ge 2^{\Omega(n)}
```

Candidates: SAT, clique indicator, functions requiring exponential formula size.

---

## 12. Theoretical Connections

### 12.1 Fourier Analysis on Boolean Hypercube

In Walsh-character embeddings, coefficients correspond to Fourier coefficients:

```math
\hat{f}(S) = \mathbb{E}_{x}[f(x)\chi_S(x)], \quad \chi_S(x)=\prod_{i\in S} x_i
```

**What GA adds:** multiplicative structure and grade mixing under $\star$.

### 12.2 Linial–Mansour–Nisan Connection (Correct translation)

LMN: AC⁰ functions have Fourier mass concentrated on low degrees.

A faithful GA translation is a **tail-mass** statement, not a strict max-grade cutoff. For example, define high-grade mass:

```math
\mathrm{HighGradeMass}_k(f) = \sum_{|S|>k} |\hat f(S)|
```

Then LMN suggests: for AC⁰, there exists $k=\mathrm{polylog}(n)$ such that $\mathrm{HighGradeMass}_k(f)$ is small.

### 12.3 Communication Complexity Analogy (Heuristic)

| Communication Complexity | Geometric Representational Complexity |
| ------------------------ | ------------------------------------- |
| Bits exchanged           | Grade reached / mass spread           |
| Protocol tree            | GA expression tree                    |
| Rectangle covers         | Blade decompositions                  |

---

## 13. Conjectures

### Conjecture 1 — Excursion Hierarchy

```math
\mathcal{E}^{-1}(O(1)) \subsetneq \mathcal{E}^{-1}(O(\log n)) \subsetneq \mathcal{E}^{-1}(O(n)) \subsetneq \mathcal{E}^{-1}(2^{O(n)})
```

### Conjecture 2 — AC⁰ Tail-Mass Characterization (Refined)

```math
f \in \mathrm{AC}^0 \implies \exists k=\mathrm{polylog}(n): \mathrm{HighGradeMass}_k(f) \text{ is small.}
```

### Conjecture 3 — Formula Size Lower Bound (Model-relative)

```math
L(f) \ge 2^{\Omega(\mathcal{E}(f))}
```

where $L(f)$ is formula size, and $\mathcal{E}$ is a carefully chosen excursion metric compatible with the simulation theorem.

### Conjecture 4 — Norm–Excursion Tradeoff (Refined)

There exists a step $t$ such that a product of “staying close to Boolean” and “having controlled norm” cannot hold simultaneously beyond a threshold. For example, one possible formal shape:

```math
\exists t:\ \mathrm{BoolDist}(F_t) \cdot \|F_t\|_1 \ge \Omega(1)
```

(Exact constants/conditions depend on embedding/threshold operator.)

---

## 14. Summary: The Physics of Computation (Analogy)

| Physical System  | Computational Analog                           |
| ---------------- | ---------------------------------------------- |
| State space      | Multivector algebra $MV_n$                     |
| Dynamics         | Geometric product (reversible for key classes) |
| Conservation law | ℓ₁ submultiplicativity                         |
| Measurement      | Projection to Boolean                          |
| Irreversibility  | Information loss at projection                 |
| Complexity       | Minimum excursion before measurement           |

> **Thesis:** Computation is trajectory through a constrained representation space. Complexity measures the unavoidable geometric distortion required before projection can yield a Boolean answer.

---

## 15. Repository Structure

```
geometric-computation/
├── Cln_Basis.v                         # Masks, corners, signs
├── Cln_Multivector.v                   # MV type, addition, scaling
├── Cln_GeometricProduct.v              # Product definition, associativity
├── Cln_Embedding.v                     # embed, eval, correctness
├── Cln_Grade.v                         # Grade definitions, evolution lemmas
├── finite_l1_submultiplicativity.v     # ℓ₁ norm, submultiplicativity
├── Cln_Excursion.v                     # Excursion measures, lower bounds
├── Cln_BooleanEmbedding.v              # Concrete n=2,3 computations
└── README.md                           # This document
```

---

## 16. Next Steps (Immediate)

1. **Prove `grade_xor_formula`** — combinatorial core of grade dynamics
2. **Prove `max_grade_gp_bound`** — grade accumulation bound under product
3. **Prove `XOR_has_grade_n_component`** — parity’s spectral signature (nonzero pseudoscalar coefficient)
4. **Prove `parity_excursion_lower_bound`** — first genuine construction-based lower bound

These establish that excursion is not philosophical but **algebraically forced**.

---

## References

* Hestenes, D. & Sobczyk, G. *Clifford Algebra to Geometric Calculus* (1984)
* O'Donnell, R. *Analysis of Boolean Functions* (2014)
* Linial, N., Mansour, Y., Nisan, N. "Constant depth circuits, Fourier transform, and learnability" (1993)
* Arora, S. & Barak, B. *Computational Complexity: A Modern Approach* (2009)

---

*A geometric invariant theory of computation.*
