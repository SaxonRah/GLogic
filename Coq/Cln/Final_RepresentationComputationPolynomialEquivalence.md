_# Representation–Computation Polynomial Equivalence

## A Coq Research Roadmap

**Goal (NOT P vs NP):**

> Establish that structured representations (GAProg / geometric algebra programs) and computational models (circuits, lambda) are **polynomially equivalent**.

If successful, this yields a general theory of:

* representations as first-class computational objects,
* cost-preserving compilation between models,
* structural limits on compressibility (rigidity).

**P vs NP is *not* the primary objective.**
It would only appear later as a corollary *if* rigidity + bridge theorems align.

---

# 0. Clarifying the Claim

We are **not** claiming:

```
representation ≡ computation
```

That statement is false without qualifiers.

We aim to prove instead:

> Representation and computation are **polynomially equivalent under a compiler and evaluator**.

Formally, for a computational model M and representation language R:

### Bridge A — Computation → Representation

Efficient computations compile into compact representations.

```
f ∈ P_M  ⇒  ∃ rep_R(f) with size ≤ poly(n)
```

### Bridge B — Representation → Computation

Compact representations can be evaluated efficiently.

```
size(rep_R(f)) ≤ poly(n)  ⇒  f ∈ P_M
```

### Bridge C — Compositional Closure

Composition in M corresponds to composition in R without global expansion.

---

# 1. Models in the Project

We will connect three worlds:

## (A) Lambda calculus (existing Coq libraries)

Role:

* universal computational model
* baseline semantics

## (B) Circuits (non-uniform computation)

Role:

* hardware-like composition
* fanout + DAG structure

## (C) GAProg (geometric algebra programs)

Role:

* representation language
* algebraic + compositional
* supports sharing and structure

---

# 2. Step One — GAProg in Coq

Define a **sharing-aware representation language**.

Important:
Plain inductives = trees.
We need DAG semantics.

Recommended representation:

```coq
Inductive Expr :=
| EConst : Q -> Expr
| EVar   : nat -> Expr
| EAdd   : nat -> nat -> Expr
| EMul   : nat -> nat -> Expr
| EScale : Q -> nat -> Expr.

Record Prog := {
  defs : list Expr;
  out  : nat
}.
```

### Required definitions

```
eval_prog   : Prog -> input -> Q
size_prog   : Prog -> nat
depth_prog  : Prog -> nat
```

---

## Coq Risks — Step 1

### Structural

* DAG modeling inside Coq is awkward.
* Node IDs require invariants (acyclicity, valid references).

### Proof engineering

* evaluation becomes dependent on lookup correctness
* proofs must reason about index validity

### Performance

* large programs = slow proof checking
* extraction may be necessary early

### Conceptual

* GA semantics vs arithmetic semantics alignment
* grade bounds must not silently explode

---

# 3. Step Two — Circuits → GAProg Compilation

Define a circuit language with explicit sharing:

```
Record Circuit := {
  gates : list Gate;
  out   : nat
}.
```

Compile gate-by-gate:

```
compile : Circuit -> Prog
```

### Theorem Targets

Correctness:

```
eval_prog (compile C) = eval_circuit C
```

Size preservation:

```
size_prog (compile C) ≤ c * size(C) + k
```

Depth preservation:

```
depth_prog (compile C) ≤ c' * depth(C) + k'
```

---

## Coq Risks — Step 2

### Fanout modeling

* circuits allow reuse
* naive compilation may duplicate nodes

### Sharing invariants

* need to prove compiled graph preserves reuse
* otherwise DAG collapses into tree

### Arithmetic encoding

* Boolean semantics embedded in Q
* must prove exactness (not approximation)

### Proof size

* structural recursion across large gate lists

---

# 4. Step Three — Lambda Fragment → Circuits

We **reuse existing lambda calculus formalizations**.

We DO NOT reimplement lambda.

Instead:

* isolate a Boolean/circuit fragment
* embed into circuits

```
lambda_fragment → circuits
```

Then composition gives:

```
lambda_fragment → circuits → GAProg
```

---

## Coq Risks — Step 3

### Cost model mismatch

* lambda libraries often lack step-cost semantics
* evaluation ≠ time complexity

### Encoding overhead

* Church encodings can blow up representation size
* must prove bounded translation

### Reduction semantics

* small-step vs big-step differences
* normalization cost vs evaluation cost

---

# 5. Step Four — Representation–Computation Bridge Theorems

Goal: establish polynomial equivalence.

### Bridge A

```
efficient computation ⇒ compact GAProg
```

### Bridge B

```
compact GAProg ⇒ efficient evaluation
```

### Bridge C

```
composition preserved without expansion
```

---

## Coq Risks — Step 4

### Uniform vs non-uniform

* circuits are non-uniform
* lambda is uniform
* must prevent advice-like loopholes

### Evaluation cost model

* size ≠ runtime
* need cost-preserving semantics

### Hidden normalization

* algebraic simplification may secretly expand

---

# 6. Step Five — Geometric Rigidity

Rigidity hypothesis:

> Some functions require intrinsically large representations.

Formal interface:

```coq
Parameter HardFamily : nat -> BoolFn.
Parameter LB : nat -> nat.

Axiom geometric_rigidity :
  forall n p,
    Represents p (HardFamily n) ->
    size_prog p ≥ LB n.
```

---

## Coq Risks — Step 5

### Expressiveness gap

* representation language might be too strong
* could encode function implicitly

### Proof barrier

* rigidity likely independent of current techniques
* may require new mathematics

### Model mismatch

* rigidity must apply to *all* programs, not just expanded forms

---

# 7. Larger Goal

This project aims to prove:

> Structured representations and computational processes are polynomially equivalent objects.

Consequences:

* programs = compressed computations
* circuits = materialized representations
* lambda = universal generator
* GAProg = structural intermediate representation

This reframes computation as:

```
evaluation of representations
```

---

# 8. Where SAT Fits

SAT only enters if all prior pieces hold:

1. computation ↔ representation equivalence
2. rigidity lower bound
3. SAT forces rigid representations

Then:

```
SAT cannot stay compact
⇒ SAT not polynomial-time
⇒ P ≠ NP
```

This is a **corollary**, not the main theorem.

---

# 9. Potential Failure Modes (Global)

### Model alignment failure

* lambda, circuits, GAProg measure cost differently

### Representation leakage

* hidden exponential structure encoded implicitly

### Evaluation blow-up

* representation compact, evaluation expensive

### Rigidity collapse

* some encoding avoids rigidity assumptions

### Coq complexity ceiling

* proofs become too large to maintain

---

# 10. Immediate Implementation Order

1. GAProg DAG model
2. Circuit compiler + size theorem
3. Lambda fragment embedding
4. Bridge theorems
5. Rigidity interface

Do NOT attempt SAT before step 5 stabilizes.

---

# 11. Research Significance

Even without rigidity or SAT:

This yields a **new theory of computation**:

* representations as algebraic computational objects
* compilers as structure-preserving morphisms
* evaluation as representation execution

This stands independently of complexity separation.

---

# 12. Summary

We are building:

* not a P vs NP proof,
* but a **unified representation–computation theory**.

Target statement:

> Computation and structured representation are polynomially equivalent under faithful compilation and evaluation.

If rigidity holds:

> Some problems cannot admit compact representations.

If SAT forces such rigidity:

> P ≠ NP.

But the deeper result is:

> **Computation itself is fundamentally representation-driven.**_
