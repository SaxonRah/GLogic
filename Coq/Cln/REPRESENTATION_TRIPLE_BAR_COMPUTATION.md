I've now thoroughly reviewed all four Coq files in the project. Here's the updated markdown reflecting the actual current state of the development:

---

# Representation ≡ Computation (via Geometric Algebra)

*A program for a representation-theoretic complexity invariant*

---

## 1. Core Thesis

We propose a model where Boolean computation is understood as:

1. **Reversible algebraic dynamics** — computation occurs inside a multivector algebra via the geometric product (invertible for certain classes of elements, e.g. blades/versors under appropriate non-degeneracy conditions).
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

In Coq, `MV n := Mask n -> Q` where `Mask n := Vector.t bool n`, and grade is the popcount:

```coq
Definition grade {n} (m : Mask n) : nat :=
  List.count_occ Bool.bool_dec (Vector.to_list m) true.
```

### 2.3 Geometric Product

For basis blades:

```math
e_A \star e_B = \mathrm{sgn}(A,B) \cdot \mathrm{met}(A,B) \cdot e_{A \oplus B}
```

where:

* $A \oplus B$ = symmetric difference (XOR of masks)
* $\mathrm{sgn}(A,B) = (-1)^{|\{(i \in A, j \in B) : j < i\}|}$ (swap parity)
* $\mathrm{met}(A,B) = \prod_{i \in A \cap B} \mathrm{sq}_i$ (metric factor from the quadratic form)

Extended bilinearly to all multivectors:

```coq
Definition mv_gp (n : nat) (sq : Vector.t Q n) (F G : MV n) : MV n :=
  fun U =>
    sumQ (List.map (fun A =>
      sumQ (List.map (fun B =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * G B * c)%Q else 0%Q
      ) (all_masks n))
    ) (all_masks n)).
```

### 2.4 Embedding Boolean Functions

The embedding uses the projector–character expansion:

```math
\Pi(a) = 2^{-n} \sum_S \chi_S(a)\, e_S, \qquad
\mathrm{embed}(f) = \sum_{a \in \{\pm 1\}^n} f(a)\, \Pi(a)
```

such that:

```math
\mathrm{eval}(\mathrm{embed}(f), s) = f(s) \quad \forall s \in \{\pm1\}^n
```

**Important distinction:** there are (at least) two natural embedding styles, and *parity/XOR looks different in each*.

| Embedding style                                | What basis means                      | What XOR looks like                                                                         | Notes                                        |
| ---------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Delta-basis (projector / point-mass expansion) | basis elements are "input projectors" | generally a *mixture* across multiple grades; often includes scalar + top-grade             | this is the style used in the Coq development |
| Walsh-character basis (Fourier/Walsh basis)    | basis elements are parity characters  | XOR/parity is supported at *top degree* (plus possibly a scalar offset if mapping to {0,1}) | aligns directly with Fourier analysis        |

In general, we will write parity's embedding in the form:

```math
\mathrm{embed}(\text{XOR}_n) = \alpha_n \cdot 1 + \beta_n \cdot e_{[n]} + R_n
```

where $\beta_n \neq 0$ captures the **nonzero pseudoscalar (grade-$n$) component**.

> The key proven fact is: **the grade-$n$ coefficient is nonzero for parity**. In particular, the Coq development shows $\mathrm{embed}(\text{XOR}_n)(\text{full mask}) = (1/2^n) \cdot \text{xor\_sum}(n)$ where $\text{xor\_sum}(n) = (-2)^{n-1}$, which is nonzero for all $n \ge 1$.

---

## 3. Foundational Results (Mechanized in Coq)

The development is organized across four files. All results below are **fully proved** (no `Admitted`) unless explicitly marked.

| Result | File | Status | Interpretation |
| --- | --- | --- | --- |
| `embed_correct` | `Cln_Full.v` | ✅ Proved | Boolean truth tables preserved under embedding |
| `mv_gp_assoc` | `Cln_Full.v` | ✅ Proved | Geometric product is associative (via sign & metric cocycles) |
| `mv_gp_one_l`, `mv_gp_one_r` | `Cln_Full.v` | ✅ Proved | Scalar 1 is the identity for the geometric product |
| `mv_gp_add_l`, `mv_gp_add_r` | `Cln_Full.v` | ✅ Proved | Geometric product is bilinear |
| `mv_gp_basis` | `Cln_Full.v` | ✅ Proved | Closed-form product on basis blades |
| `e_square` | `Cln_Full.v` | ✅ Proved | Clifford relation: $e_i^2 = \mathrm{sq}_i$ |
| `e_anticomm` | `Cln_Full.v` | ✅ Proved | Clifford relation: $e_i e_j = -e_j e_i$ for $i \neq j$ |
| `swaps_parity_cocycle` | `Cln_Full.v` | ✅ Proved | Sign associativity cocycle |
| `metric_factor_cocycle` | `Cln_Full.v` | ✅ Proved | Metric associativity cocycle |
| `l1_gp_submultiplicative` | `Cln_finite_l1_submultiplicativity.v` | ✅ Proved | $\lVert F \star G \rVert_1 \le \lVert F \rVert_1 \cdot \lVert G \rVert_1$ (unit metric) |
| `l1_add_bound` | `Cln_finite_l1_submultiplicativity.v` | ✅ Proved | $\lVert F + G \rVert_1 \le \lVert F \rVert_1 + \lVert G \rVert_1$ |
| `l1_norm_eval_le` | `Cln_finite_l1_submultiplicativity.v` | ✅ Proved | Static ℓ₁ bound from GA expression structure |
| `grade_bounded_gp` / `max_grade_gp_le` | `Cln_Grade.v` | ✅ Proved | $\mathrm{MaxGrade}(F \star G) \le \mathrm{MaxGrade}(F) + \mathrm{MaxGrade}(G)$ |
| `grade_bounded_add` / `max_grade_add_le` | `Cln_Grade.v` | ✅ Proved | $\mathrm{MaxGrade}(F + G) \le \max(\mathrm{MaxGrade}(F), \mathrm{MaxGrade}(G))$ |
| `eval_grade_bounded` | `Cln_Grade.v` | ✅ Proved | Every GA expression respects its syntactic grade bound |
| `excursion_lower_bound` | `Cln_Grade.v` | ✅ Proved | Generic excursion lower bound from nonzero high-grade coefficient |
| `XOR_has_grade_n_component` | `Cln_Grade.v` | ✅ Proved | Parity has nonzero pseudoscalar coefficient |
| **`parity_excursion`** | `Cln_Grade.v` | ✅ **Proved** | **Any GA program computing parity must reach grade $n$** |
| `bool_dist_embed` | `Cln_BoolDist.v` | ✅ Proved | Embedded Boolean functions have BoolDist 0 |
| `bool_dist_wrt_add` | `Cln_BoolDist.v` | ✅ Proved | Addition errors add (witness-relative form) |
| `bool_dist_wrt_gp` | `Cln_BoolDist.v` | ✅ Proved | GP error decomposes bilinearly |
| `gp_error_split` | `Cln_BoolDist.v` | ✅ Proved | $F \star G - F_0 \star G_0 = F \star (G - G_0) + (F - F_0) \star G_0$ |
| `translate` (BoolFormula → GA_expr) | `Cln_BoolDist.v` | ✅ Defined | Simulation translation from Boolean formulas |
| `translate_correct` | `Cln_BoolDist.v` | ⬜ Admitted | Correctness of the Boolean formula translation |

---

## 4. Algebraic Laws of Computation

### Law 1 — Associativity and Clifford Relations (Proved)

The geometric product is associative, with Clifford relations:

```math
e_i \star e_i = \mathrm{sq}_i, \qquad e_i \star e_j = -e_j \star e_i \ (i \neq j)
```

Associativity is proved via two cocycle identities — one for swap parity, one for the metric factor — each established by induction on dimension with boolean case analysis:

```coq
Lemma swaps_parity_cocycle : forall n (A B C : Mask n),
  xorb (swaps_parity A B) (swaps_parity (mask_xor A B) C)
  = xorb (swaps_parity B C) (swaps_parity A (mask_xor B C)).

Lemma metric_factor_cocycle : forall n (sq : Vector.t Q n) (A B C : Mask n),
  (metric_factor sq A B * metric_factor sq (mask_xor A B) C)%Q
  == (metric_factor sq B C * metric_factor sq A (mask_xor B C))%Q.
```

### Law 2 — Norm Constraint (ℓ₁ Submultiplicativity, Proved)

```math
\|F \star G\|_1 \le \|F\|_1 \cdot \|G\|_1
```

Proved under the unit metric hypothesis ($|\mathrm{sq}_i| = 1$) via `basis_mul_coeff_abs1`, which gives $|c(A,B)| = 1$ for all mask pairs.

Additionally, a static ℓ₁ bound from expression structure is proved:

```coq
Theorem l1_norm_eval_le : forall (e : GA_expr n),
  l1_norm (eval_expr sq e) <= l1_bound e.
```

where `l1_bound` tracks ℓ₁ norm through expression syntax (1 for basis, |c| for scalars, sum for addition, product for multiplication).

### Law 3 — Projection Irreversibility

The evaluation map

```math
\mathrm{eval}: MV_n \times \{\pm 1\}^n \to \mathbb{Q}
```

recovers Boolean values on embedded Boolean functions, but in general returns rationals for intermediate multivectors. Any *Boolean decision* requires an additional projection/thresholding step, which is information-losing.

### Law 4 — Complexity as Excursion

Hard functions force large excursions in representation space before projection can yield a Boolean answer. This is now a **theorem** for parity, not just a conjecture.

---

## 5. Excursion Measures

### 5.1 Boolean Distance (Witness-based)

The development uses a **witness-relative** formulation (avoids infimum over $2^{2^n}$ functions):

```coq
Definition bool_dist_le {n} (F : MV n) (d : Q) : Prop :=
  exists g : Corner n -> bool,
    l1_norm (mv_sub F (embed g)) <= d.

Definition bool_dist_wrt {n} (F : MV n) (g : Corner n -> bool) : Q :=
  l1_norm (mv_sub F (embed g)).
```

Key proved properties:
- `bool_dist_embed`: embedded functions have distance 0
- `bool_dist_wrt_add`: errors add under multivector addition
- `bool_dist_wrt_gp`: errors decompose bilinearly under geometric product
- `bool_dist_add_absorb`: one-sided absorbing bound for addition

> Note: the naive triangle inequality `bool_dist_le (mv_add F G) (dF + dG)` and the GP bound `bool_dist_le (mv_gp sq F G) (dF + dG + ‖F‖₁ + ‖G‖₁)` are both **false** (documented in Coq comments). The correct bounds use witness-relative decompositions.

### 5.2 Grade Support

```math
\mathrm{GradeSupp}(F) = \{k : \exists A,\ |A|=k,\ F(A)\neq 0\}
```

```math
\mathrm{MaxGrade}(F) = \max(\mathrm{GradeSupp}(F))
```

In Coq:

```coq
Definition max_grade {n} (F : MV n) : nat :=
  fold_right max 0
    (map (fun m => if Qeq_bool (F m) 0 then 0 else grade m)
         (all_masks n)).
```

The `grade_bounded` predicate provides a flexible alternative: `grade_bounded F k` means all coefficients above grade $k$ are zero.

### 5.3 Composite Measure (Lexicographic)

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
* $\mathcal{B}$ = Boolean subspace $\{\mathrm{embed}(g) : g \text{ Boolean}\}$
* $\mathrm{Dist}$ = chosen excursion metric (e.g. BoolDist, MaxGrade, support, or a combination)

---

## 7. Grade Dynamics (Proved Structural Results)

### 7.1 Grade Under XOR of Masks

```coq
Lemma grade_basis_mul_mask_le : forall n (A B : Mask n),
  (grade (basis_mul_mask A B) <= grade A + grade B)%nat.
```

### 7.2 Grade Evolution Under Operations (Proved)

```coq
Lemma max_grade_gp_le : forall n sq (F G : MV n),
  (max_grade (mv_gp sq F G) <= max_grade F + max_grade G)%nat.

Lemma max_grade_add_le : forall n (F G : MV n),
  (max_grade (mv_add F G) <= Nat.max (max_grade F) (max_grade G))%nat.
```

Both proved via the `grade_bounded` framework, which propagates grade bounds compositionally through expressions.

### 7.3 GA Expression Grade Analysis (Proved)

```coq
Fixpoint grade_bound {n} (e : GA_expr n) : nat :=
  match e with
  | Basis _   => 1%nat
  | Scalar _  => 0%nat
  | Add e1 e2 => Nat.max (grade_bound e1) (grade_bound e2)
  | Mul e1 e2 => (grade_bound e1 + grade_bound e2)%nat
  end.

Theorem eval_grade_bounded : forall n sq (e : GA_expr n),
  grade_bounded (eval_expr sq e) (grade_bound e).
```

---

## 8. The Parity Lower Bound (Fully Proved)

### 8.1 Parity Embedding Structure

The embedding of XOR at the full mask factors as:

```coq
Lemma embed_XOR_full_mask : forall n,
  embed (@XOR_n_func n) (Vector.const true n)
  == ((1 / pow2 n) * xor_sum n)%Q.
```

where `xor_sum` satisfies the recurrence:

```coq
Lemma xor_sum_base : xor_sum 1 == 1.
Lemma xor_sum_step : forall n, (n > 0)%nat ->
  xor_sum (S n) == ((-2) * xor_sum n)%Q.
```

(The recurrence uses `chi_full_sum_zero`: the sum of all characters at the full mask vanishes for dimension ≥ 1.)

This gives $\text{xor\_sum}(n) = (-2)^{n-1}$, which is nonzero:

```coq
Lemma xor_sum_nonzero : forall n, (n > 0)%nat -> ~(xor_sum n == 0).
```

### 8.2 Nonzero Pseudoscalar Coefficient (Proved)

```coq
Lemma XOR_has_grade_n_component : forall n,
  (n > 0)%nat ->
  ~(embed (@XOR_n_func n) (Vector.const true n) == 0).
```

### 8.3 Generic Excursion Lower Bound (Proved)

```coq
Theorem excursion_lower_bound :
  forall n sq (e : GA_expr n) (F : MV n),
    eval_expr sq e = F ->
    (exists m : Mask n, (grade m >= n)%nat /\ ~(F m == 0)) ->
    (max_grade_during sq e >= n)%nat.
```

This is a general theorem: if the target multivector has any nonzero coefficient at grade ≥ $n$, then the max grade during evaluation must reach $n$.

### 8.4 Parity Excursion Theorem (Proved)

```coq
Theorem parity_excursion :
  forall n sq (e : GA_expr n),
    (n > 0)%nat ->
    eval_expr sq e = embed (@XOR_n_func n) ->
    (max_grade_during sq e >= n)%nat.
```

**Proof.** Instantiate `excursion_lower_bound` with `m := Vector.const true n`. The grade of the full mask equals $n$ by `grade_full_mask`, and the coefficient is nonzero by `XOR_has_grade_n_component`. ∎

> This is the **first fully mechanized excursion lower bound** in this framework: any GA program computing parity must reach the top grade.

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
  | Basis i   => basis (mask_single i)
  | Scalar c  => mv_scale c mv_one
  | Add e1 e2 => mv_add (eval_expr sq e1) (eval_expr sq e2)
  | Mul e1 e2 => mv_gp sq (eval_expr sq e1) (eval_expr sq e2)
  end.
```

The excursion tracker computes the maximum grade seen at any intermediate step:

```coq
Fixpoint max_grade_during {n} (sq : Vector.t Q n) (e : GA_expr n) : nat :=
  match e with
  | Basis _   => 1%nat
  | Scalar _  => 0%nat
  | Add e1 e2 =>
      Nat.max (max_grade_during sq e1) (max_grade_during sq e2)
  | Mul e1 e2 =>
      Nat.max
        (Nat.max (max_grade_during sq e1) (max_grade_during sq e2))
        (max_grade (mv_gp sq (eval_expr sq e1) (eval_expr sq e2)))
  end.
```

with the key linking lemma (proved):

```coq
Lemma max_grade_le_during : forall n sq (e : GA_expr n),
  (max_grade (eval_expr sq e) <= max_grade_during sq e)%nat.
```

---

## 10. Boolean Formula Simulation (Defined, Correctness Admitted)

A translation from Boolean formulas to GA expressions is defined:

```coq
Inductive BoolFormula (n : nat) : Type :=
  | BVar   : Fin.t n -> BoolFormula n
  | BConst : bool -> BoolFormula n
  | BAnd   : BoolFormula n -> BoolFormula n -> BoolFormula n
  | BOr    : BoolFormula n -> BoolFormula n -> BoolFormula n
  | BNot   : BoolFormula n -> BoolFormula n.

Fixpoint translate {n} (phi : BoolFormula n) : GA_expr n :=
  match phi with
  | BVar i       => Mul (Scalar (1#2)) (Add (Scalar 1) (Basis i))
  | BConst true  => Scalar 1
  | BConst false => Scalar 0
  | BAnd p q     => Mul (translate p) (translate q)
  | BNot p       => Add (Scalar 1) (Mul (Scalar (-1)) (translate p))
  | BOr p q      => Add (Add (translate p) (translate q))
                        (Mul (Scalar (-1)) (Mul (translate p) (translate q)))
  end.
```

The encoding maps:
- Variables $x_i$ to the projector $\frac{1}{2}(1 + e_i)$
- AND to the geometric product (idempotent projectors multiply correctly)
- NOT to $1 - P$
- OR via De Morgan: $P + Q - PQ$

```coq
Theorem translate_correct : forall n sq (phi : BoolFormula n),
  (forall i, Vector.nth sq i == 1) ->
  forall m, eval_expr sq (translate phi) m == embed (eval_bf phi) m.
Proof. Admitted.
```

> **Status:** The translation is defined and the correctness statement is in place. The proof is admitted pending a compositional induction argument through the embedding.

---

## 11. Coq File Structure

| File | Contents |
| --- | --- |
| `Cln_Full.v` | Basis infrastructure (Sign, Corner, Mask), enumerations with NoDup/completeness, multivectors (MV n), characters (χ), evaluation, Boolean embedding (Π, embed, embed_correct), geometric product (mv_gp), associativity (via cocycle proofs), bilinearity, identity laws, Clifford relations |
| `Cln_Grade.v` | Grade (popcount), grade bounds for distinguished masks, grade evolution under +/⋆, GA_expr syntax and evaluation, grade_bound, max_grade_during, excursion_lower_bound, XOR_n_func, xor_sum recurrence, **parity_excursion theorem** |
| `Cln_finite_l1_submultiplicativity.v` | ℓ₁ norm, submultiplicativity theorem (under unit metric), triangle inequality, l1_bound, l1_norm_eval_le |
| `Cln_BoolDist.v` | Boolean distance (witness-based), error propagation lemmas, GP error decomposition, BoolFormula type, translate function, translate_correct (admitted) |

---

## 12. Research Program

### A — Simulation Theorems (In Progress)

| Source Model | Target | Status |
| --- | --- | --- |
| Boolean formulas (AND/OR/NOT) | GA expressions | Translation defined; correctness admitted |
| Circuits (bounded fan-in) | GA programs | Not yet started |

The `translate` function provides an O(size) embedding of Boolean formulas into GA expressions. Proving `translate_correct` is the immediate next milestone.

### B — Upper Bounds (Easy Functions)

Show small excursion suffices for families like AND/OR/decision trees, with respect to chosen excursion metrics (grade, ℓ₁, BoolDist). The `l1_bound` analysis already gives static ℓ₁ upper bounds for any GA expression from its syntax.

### C — Lower Bounds (Hard Functions) ✅ Complete

The parity excursion lower bound is **fully proved**:

```math
\mathcal{E}(\text{XOR}_n) \ge n \quad \text{(via MaxGrade)}
```

The proof chain:
1. `embed_XOR_full_mask`: factor the pseudoscalar coefficient
2. `xor_sum_step` + `xor_sum_base`: establish the recurrence $\text{xor\_sum}(S\,n) = -2 \cdot \text{xor\_sum}(n)$
3. `xor_sum_nonzero`: the recurrence preserves nonzero-ness
4. `XOR_has_grade_n_component`: combine (1–3) to show the coefficient is nonzero
5. `excursion_lower_bound`: generic theorem from nonzero high-grade coefficient
6. `parity_excursion`: instantiate with the full mask

### D — Separation Theorem

Ultimate goal:

```math
\exists f_n : \mathcal{E}(f_n) \ge 2^{\Omega(n)}
```

Candidates: SAT, clique indicator, functions requiring exponential formula size.

---

## 13. Theoretical Connections

### 13.1 Fourier Analysis on Boolean Hypercube

In Walsh-character embeddings, coefficients correspond to Fourier coefficients:

```math
\hat{f}(S) = \mathbb{E}_{x}[f(x)\chi_S(x)], \quad \chi_S(x)=\prod_{i\in S} x_i
```

**What GA adds:** multiplicative structure and grade mixing under ⋆. The Coq development uses the delta-basis embedding, but the Fourier connection is visible through the `xor_sum` recurrence.

### 13.2 Linial–Mansour–Nisan Connection (Correct translation)

LMN: AC⁰ functions have Fourier mass concentrated on low degrees.

A faithful GA translation is a **tail-mass** statement, not a strict max-grade cutoff. For example, define high-grade mass:

```math
\mathrm{HighGradeMass}_k(f) = \sum_{|S|>k} |\hat f(S)|
```

Then LMN suggests: for AC⁰, there exists $k=\mathrm{polylog}(n)$ such that $\mathrm{HighGradeMass}_k(f)$ is small.

### 13.3 Communication Complexity Analogy (Heuristic)

| Communication Complexity | Geometric Representational Complexity |
| --- | --- |
| Bits exchanged | Grade reached / mass spread |
| Protocol tree | GA expression tree |
| Rectangle covers | Blade decompositions |

---

## 14. Conjectures

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

There exists a step $t$ such that a product of "staying close to Boolean" and "having controlled norm" cannot hold simultaneously beyond a threshold. For example, one possible formal shape:

```math
\exists t:\ \mathrm{BoolDist}(F_t) \cdot \|F_t\|_1 \ge \Omega(1)
```

(Exact constants/conditions depend on embedding/threshold operator.)

---

## 15. Summary: The Physics of Computation (Analogy)

| Physical System | Computational Analog |
| --- | --- |
| State space | Multivector algebra $MV_n$ |
| Dynamics | Geometric product (associative, with Clifford relations) |
| Conservation law | ℓ₁ submultiplicativity |
| Measurement | Projection to Boolean |
| Irreversibility | Information loss at projection |
| Complexity | Minimum excursion before measurement |

> **Thesis:** Computation is trajectory through a constrained representation space. Complexity measures the unavoidable geometric distortion required before projection can yield a Boolean answer.

> **Milestone achieved:** The parity excursion lower bound is the first fully mechanized instance of this thesis — any GA program computing XOR must reach the top grade, regardless of how it is constructed.

## References

* Hestenes, D. & Sobczyk, G. *Clifford Algebra to Geometric Calculus* (1984)
* O'Donnell, R. *Analysis of Boolean Functions* (2014)
* Linial, N., Mansour, Y., Nisan, N. "Constant depth circuits, Fourier transform, and learnability" (1993)
* Arora, S. & Barak, B. *Computational Complexity: A Modern Approach* (2009)

---

*A geometric invariant theory of computation — now with its first mechanized lower bound.*