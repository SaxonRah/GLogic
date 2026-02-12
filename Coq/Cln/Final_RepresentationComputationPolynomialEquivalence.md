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

---

---

---

**File-by-file Coq module plan** that plugs directly into the existing naming + structure:

Current the "math substrate":

* `Cln_Basis.v` (Sign / Corner / Mask, enumerations, decidable eq)
* `Cln_Multivector.v` (`MV n := Mask n -> Q`, linear ops, finite sums, `Eval_s`)
* `Cln_BooleanEmbedding.v` (Π(a), `Embed`, correctness `Eval_s(Embed f)=f(s)`)
* `Cln_GeometricProduct.v` (eventual `mv_gp`; right now looks like a roadmap stub)

The next stage is: **GAProg (representation language) + circuits + compilers + cost theorems**.

---

## Directory + build order

"Phase2" layer:

1. `Cln_GAProg.v`
2. `Cln_GAProg_Cost.v`
3. `Cln_Circuit.v`
4. `Cln_Circuit_Eval.v`
5. `Cln_CircuitToGAProg.v`
6. `Cln_LambdaFragment.v` *(optional / later)*
7. `Cln_LambdaToCircuit.v` *(optional / later)*
8. `Cln_BridgeTheorems.v`
9. `Cln_Rigidity_Interface.v`

Everything after this can be "applications" (SAT, rigidity experiments, etc.), but this spine is the core.

---

## 1) `Cln_GAProg.v` — a sharing-aware representation language (DAG)

**Depends on:** `Cln_Basis`, `Cln_Multivector` (and `QArith`, `List`, etc.)

### Goal

Define a program IR that is **DAG-by-construction** (via `defs : list Expr; out : nat`), so "size" reflects **sharing**, not tree blowups.

### Minimal content

* A type for "inputs" (you have two common choices):

  * *Boolean input semantics:* `Corner n -> Q` / `Corner n -> bool`
  * *Program input semantics:* `list Q` indexed by nat vars
    (I'd keep it nat-var based for the compiler, then later connect vars ↔ corners.)
* An expression language that matches what you'll need for compiling circuits into *multivector arithmetic*:

  * `EConst : Q -> Expr`
  * `EVar : nat -> Expr`
  * `EAdd : nat -> nat -> Expr`
  * `EMul : nat -> nat -> Expr`
  * `EScale : Q -> nat -> Expr`
  * **(optional, but very useful)** `ENeg : nat -> Expr` as sugar for `EScale (-1)`

### Program record + invariants

```coq
Record GAProg := { defs : list Expr; out : nat }.
Definition wf_prog (p:GAProg) : Prop := out < length p.(defs) /\ (* plus per-node bounds *).
```

### Semantics

* `eval_node : list Q -> nat -> Q` (with bounds proofs)
* `eval_prog : GAProg -> list Q -> Q`

**Key lemmas you'll want immediately**

* lookup lemmas (`nth_error` / bounds → definitional equalities)
* "evaluation respects wf": if `wf_prog p` then `eval_prog` is total/unique

---

## 2) `Cln_GAProg_Cost.v` — size/depth + "no hidden expansion" facts

**Depends on:** `Cln_GAProg`

### Minimal definitions

* `size_prog : GAProg -> nat := length defs`
* `node_cost : Expr -> nat` (optional, but helps later)
* `depth_node / depth_prog` via a dependency graph recurrence:

  * `depth(i) = 1 + max(depth deps)`
  * `depth_prog = depth(out)`

### Must-have lemmas

* Depth monotonicity: if node j depends on i then `depth i < depth j`
* "Appending defs doesn't change earlier evals": needed for compilers that build programs incrementally.

  * If `p` is a prefix of `p'`, evaluation of nodes `< length p.defs` is preserved.

This file pays for itself later; without it, your compiler proofs get painful.

---

## 3) `Cln_Circuit.v` — circuit syntax with explicit fanout/sharing

**Depends on:** basic Coq libs only (keep it clean)

### Minimal

```coq
Inductive Gate :=
| GConst : bool -> Gate
| GInput : nat -> Gate
| GNot   : nat -> Gate
| GAnd   : nat -> nat -> Gate
| GXor   : nat -> nat -> Gate
| GOr    : nat -> nat -> Gate.  (* optional if derived *)
Record Circuit := { gates : list Gate; out : nat }.
Definition wf_circuit : Prop := (* indices < position in list, out < length gates *).
```

You can keep OR as derived (`a OR b = a XOR b XOR (a AND b)` in {0,1}) if you want minimal gates.

---

## 4) `Cln_Circuit_Eval.v` — Boolean semantics + cost measures for circuits

**Depends on:** `Cln_Circuit`

### Semantics

* Input assignment: `inputs : nat -> bool` or `list bool`
* Node evaluation: `eval_gate_at : nat -> bool`
* `eval_circuit : Circuit -> (nat -> bool) -> bool`

### Cost

* `size_c : nat := length gates`
* `depth_c : nat` computed similarly to GAProg depth

### Key lemmas

* evaluation totality under `wf_circuit`
* useful boolean identities you'll reuse in correctness proofs

---

## 5) `Cln_CircuitToGAProg.v` — the compiler + 3 preservation theorems

**Depends on:** `Cln_GAProg`, `Cln_GAProg_Cost`, `Cln_Circuit`, `Cln_Circuit_Eval`

### The compiler

Gate-by-gate compilation into `GAProg` nodes.

Important: decide *your boolean encoding in Q*:

* `bQ : bool -> Q` with `bQ true = 1`, `bQ false = 0`
* prove correctness of each compiled gate wrt `bQ`

### Minimal theorems (these are the milestone)

1. **Correctness**

```coq
Theorem compile_correct :
  forall C ρ,
    wf_circuit C ->
    eval_prog (compile C) (encode_inputs ρ) == bQ (eval_circuit C ρ).
```

2. **Size preservation**

```coq
Theorem compile_size :
  forall C, size_prog (compile C) <= a * size_c C + b.
```

3. **Depth preservation**

```coq
Theorem compile_depth :
  forall C, depth_prog (compile C) <= a' * depth_c C + b'.
```

You'll also want a lemma: compilation is "prefix-extending" so you can prove correctness by induction on the gate list.

---

## 6) `Cln_LambdaFragment.v` — optional, but scoped

**Depends on:** (ideally) an existing lambda calculus library; if not, keep it tiny.

The point is **not** to formalize full lambda. You define a *fragment* whose programs correspond to circuits (e.g., straight-line, no recursion).

Minimal: syntax + evaluation (big-step is fine), plus a "size" measure that makes sense.

---

## 7) `Cln_LambdaToCircuit.v` — optional compiler

**Depends on:** `Cln_LambdaFragment`, `Cln_Circuit`

Goal: show your fragment compiles to circuits with polynomial overhead.

This is where you handle the "uniform vs non-uniform" story carefully (to avoid advice-like loopholes).

---

## 8) `Cln_BridgeTheorems.v` — the packaged "polynomial equivalence" statement

**Depends on:** everything above (+ optionally lambda bits)

This file is where you state the bridges *cleanly* and prove them from the compilers + evaluators you already wrote.

Deliverables:

* **Bridge A (computation → representation)**: circuit families compile to GAProg families with poly size.
* **Bridge B (representation → computation)**: evaluating GAProg is poly in program size (this is mostly your `eval_prog` complexity model; in Coq it's usually a theorem about a cost function, not actual runtime).
* **Bridge C (compositional closure)**: composing circuits corresponds to linking GAProg without global expansion.

This is your "paper theorem file."

---

## 9) `Cln_Rigidity_Interface.v` — keep rigidity as an interface, not a blocker

**Depends on:** `Cln_GAProg_Cost` (and whatever function-spec type you use)

You define:

* what it means for a `GAProg` to "represent" a Boolean function family
* then *parameterize* a hardness family + LB function

```coq
Parameter HardFamily : nat -> (nat -> bool) -> bool.  (* or Corner n -> bool *)
Parameter LB : nat -> nat.

Axiom geometric_rigidity :
  forall n p,
    Represents p (HardFamily n) ->
    size_prog p >= LB n.
```

This lets you keep building everything else without needing to solve the hardest part early.

---

# What I would do first (the actual next 2–3 files)

If you want the fastest "progress with least pain", do:

1. `Cln_GAProg.v` (defs/out + eval + wf invariants)
2. `Cln_GAProg_Cost.v` (size/depth + prefix-preservation lemma)
3. `Cln_Circuit.v` + `Cln_Circuit_Eval.v`
4. `Cln_CircuitToGAProg.v` (compiler + correctness)

Once (4) compiles, you basically have the project's backbone.
