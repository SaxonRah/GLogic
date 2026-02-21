# Abstract

We develop a geometric-algebraic framework for analyzing Boolean computation through the interaction of two algebraic products on multivectors over ((\mathbb{Z}_2)^n):

* an **untwisted convolution product**, which admits a clean multiplicative evaluation map and models Boolean logic compositionally, and
* a **twisted geometric product (GP)**, which introduces a nontrivial cocycle (sign/metric twist) that causes grade mixing and geometric interference.

Boolean functions are embedded into multivectors via a projector–character expansion, and evaluation against characters recovers Boolean semantics. Convolution provides a flat, commutative logic layer, while the geometric product induces signed, noncommutative dynamics.

We introduce a quantitative invariant—**excursion**—which measures peak geometric complexity during evaluation, combining maximal grade and maximal ℓ₁ mass over the execution trace. We also define a structured “Boolean shadow” condition (`boolish_k_le`) requiring that intermediate multivectors remain close (in ℓ₁) to a bounded-complexity linear combination of embedded Boolean generators.

The central program is to prove a separation theorem:

> There exists a Boolean function family (f_n) such that any geometric-algebra expression computing (f_n), whose execution trace remains close to low-complexity Boolean shadows, must incur exponential ℓ₁ excursion.

This establishes a representation-theoretic lower bound: simulating flat Boolean logic inside twisted geometry forces unavoidable geometric growth.

---

# Conceptual Architecture

## 1. Two Algebraic Worlds

### Untwisted world (Convolution)

* Product: (F \odot G)
* Evaluation is multiplicative:
  ```math
  \mathrm{eval}(F \odot G, s) = \mathrm{eval}(F,s)\mathrm{eval}(G,s).
  ```
* Boolean AND corresponds cleanly to convolution.
* This layer models **Boolean semantics**.

### Twisted world (Geometric Product)

* Product: (F \star G) with twist coefficient.
* Same XOR routing as convolution, but with sign/metric cocycle.
* Causes grade mixing and interference.
* This layer models **geometric computation**.

GP is literally twisted convolution: convolution plus a 2-cocycle.

The twist is the structural obstruction.

---

## 2. Embedding and Evaluation

Boolean functions are embedded as:

```math
\mathrm{embed}(f) = \sum_{a} f(a)\Pi(a),
```

with evaluation recovering Boolean values.

You have already proven:

* `translate_correct`
* GP expressions compute Boolean logic correctly
* Convolution implements AND compositionally

So expressive equivalence is established.

---

## 3. What Separation Means Here

This is *not* classical circuit lower bounds.

The claim is:

> Simulating flat Boolean computation inside twisted geometry while remaining close to Boolean structure forces geometric complexity growth.

This is a **representation-theoretic separation**.

The key insight:

Expressive equivalence ≠ computational equivalence.

---

# The Formal Ingredients You Have

## Excursion

```coq
Record ExcNum := {
  exc_grade : nat;
  exc_l1    : Q;
}.
```

where:

* `exc_grade = max_grade_during`
* `exc_l1 = max_l1_during`

This is a **peak-over-trace** metric.

Cancellation cannot reduce it.

---

## Boolean Shadow (Structured)

```coq
boolish_k_le F k d :=
  exists cs gs,
    wf_lincomb cs gs /\
    length gs <= k /\
    l1_norm (F - lincomb_embed cs gs) <= d.
```

This captures:

* Low-complexity approximation (k generators)
* Small deviation (d)

This is your projection-with-complexity notion.

---

## Trace Condition

```coq
trace_boolish_k_le sq e k d
```

Every intermediate subexpression remains boolish_k_le.

This encodes:

> “Easy computations never leave the Boolean shadow.”

---

# What the Separation Theorem Must Prove

You want:

```coq
Theorem hard_family_separates :
  forall d,
  exists f, exists c,
    forall n sq e k,
      computes sq e (f n) ->
      trace_boolish_k_le sq e k d ->
      Qpow2 (c*n) <= exc_l1 (exc_of sq e).
```

In words:

If a GP computation stays close to low-complexity Boolean shadows and computes a hard function, then peak ℓ₁ excursion is exponential.

This is the core.

---

# Why Cancellation Is Not the Issue

Your excursion metric is peak-based.

Even if later terms cancel:

* peak grade already recorded
* peak ℓ₁ already recorded

So excursion cannot cancel.

The real issue is:

> Can computation avoid growing excursion in the first place?

---

# What Remains To Be Done

Now we get to the roadmap.

---

# Updated Roadmap Toward Separation

## Phase 1 — Closure Infrastructure (Critical)

You need compositional lemmas that allow trace_boolish_k_le to propagate structurally.

### 1. Representation Rewrite Lemmas

These are absolutely load-bearing:

#### (a) Convolution

```coq
mv_conv (lincomb_embed cs1 gs1)
        (lincomb_embed cs2 gs2)
=
lincomb_embed (mul_coeffs cs1 cs2)
              (and_gens gs1 gs2).
```

(With wf assumptions.)

#### (b) Geometric Product

Similar but twist-aware version.

These convert semantic products into list-level combinatorics.

---

### 2. k-Propagation

Prove:

* length(and_gens gs1 gs2) ≤ length gs1 * length gs2
* similar bounds for GP

So you can control k.

---

### 3. ℓ₁ Submultiplicativity

You need list-level lemmas:

```math
\ell_1(mul_coeffs(cs1, cs2))
\le
\ell_1(cs1)\ell_1(cs2).
```

And similarly for GP coefficient interaction.

This ensures ℓ₁ grows predictably.

---

## Phase 2 — Trace Stability

Prove that the trace predicate composes:

* Add preserves boolish_k_le
* Conv preserves boolish_k_le
* GP preserves boolish_k_le (with updated k)

These use Phase 1 lemmas.

This gives you:

> Easy programs ⇒ trace_boolish_k_le with controlled k.

This is your simulation theorem.

---

## Phase 3 — Hard Family Lower Bound

Pick explicit family:

* Parity
* Bent functions
* Random functions (existential)

Show:

If F approximates embed(f_n) with k generators and small d,
then ℓ₁ must be ≥ exponential.

This is the central combinatorial argument.

Often done by:

* counting arguments
* Fourier mass arguments
* dimension arguments
* rank arguments

This step is where the actual lower bound lives.

---

## Phase 4 — Bridge Theorem

Combine:

1. Simulation theorem (easy ⇒ trace_boolish_k_le with k small)
2. Hard family lower bound (trace_boolish_k_le ⇒ ℓ₁ exponential)
3. Excursion measures ℓ₁ peak

Conclude separation.

---

# The Single Most Important Technical Question

Does your hard family satisfy:

```math
\text{any } k\text{-generator approximation requires } k \ge 2^{\Omega(n)}?
```

If yes, separation follows.

If no, framework collapses.

---

# What to Focus On Now

1. Finish `lincomb_embed_conv` (wf version).
2. Prove ℓ₁ submultiplicativity for coefficient combinators.
3. Strengthen GP analogue.
4. Lock down trace_boolish_k_le closure.
5. Choose hard family and prove approximation lower bound.

Everything else is infrastructure.

---

# Big Picture

You have built:

* Algebraic semantics
* Boolean embedding
* Twisted product
* Peak excursion metric
* Structured Boolean shadow notion

You are no longer exploring.
You are in the final quantitative phase.

This is no longer philosophical.
It is purely about:

> Proving structured approximation requires exponential ℓ₁.

If that is true, your separation theorem is inevitable.

---

## Cleanest hard family for *your* `boolish_k_le`

### Key observation about your definitions

Your separation conclusion is about **peak ℓ₁ along the trace**:

```coq
exc_l1 (exc_of sq e) = max_l1_during sq e
```

and `max_l1_during` (by design) is a **max over subexpressions**, so it includes the **final output value**.

So if `computes sq e (f n)` means (as in your earlier “translate_correct” style) that the output multivector equals `embed (f n)` (or is within fixed small `d` of it), then a *very clean* way to force exponential excursion is:

> pick (f_n) whose **embedded multivector already has exponential ℓ₁ norm**.

Then every correct computation has `max_l1_during ≥ l1_norm (embed (f n))`, independent of any internal cancellation tricks.

### What controls `l1_norm (embed f)`?

With your projector–character embedding, the coefficients of `embed(f)` are (up to the (2^{-n}) factor) the Walsh–Fourier coefficients of `f`. Concretely:

```math
(\mathrm{embed}(f))(S) = 2^{-n},\widehat f(S)
\quad\Rightarrow\quad
|\mathrm{embed}(f)|_1 = 2^{-n},|\widehat f|_1.
```

So you want a Boolean family with **huge Fourier ℓ₁ norm**.

### Best explicit candidate: a bent family

For even (n), the classic explicit bent function is the quadratic form
```math
b_n(x) ;=; x_1x_2 \oplus x_3x_4 \oplus \cdots \oplus x_{n-1}x_n
\quad(\text{over } \mathbb{F}_2),
```
viewed as a ({\pm 1})-valued function via ((-1)^{b_n(x)}).

Bent functions have *flat* Walsh spectrum:
```math
|\widehat b_n(S)| = 2^{n/2}\quad\forall S.
```
Therefore
```math
|\widehat b_n|_1 = 2^n \cdot 2^{n/2} = 2^{3n/2}
\quad\Rightarrow\quad
|\mathrm{embed}(b_n)|_1 = 2^{-n}\cdot 2^{3n/2} = 2^{n/2}.
```

That is exactly an exponential lower bound with constant (c=\tfrac12).

**Why this is the cleanest match for your framework:**

* It’s explicit (no probability, no counting).
* The lower bound is “one-line” once you have the flat-spectrum lemma.
* It doesn’t depend on tricky internal trace arguments; it’s forced by the *output* itself.
* It’s robust to cancellation because you’re using ℓ₁ and a peak-over-trace.

**How to handle odd (n):**
Pad:
```math
b_{n}(x_1,\dots,x_n) := b_{n-1}(x_1,\dots,x_{n-1})
```
(or add a dummy variable). You’ll still get (|\mathrm{embed}(b_n)|_1 \ge 2^{\lfloor n/2\rfloor}), which is still (2^{\Omega(n)}).

### Alternative (even easier to prove, but non-explicit): random family

Random Boolean functions have (|\widehat f|_1 \approx 2^{3n/2}) with high probability, hence (|\mathrm{embed}(f)|_1 \approx 2^{n/2}). This is “existence by probability,” but formalizing probability in Coq is heavier than the explicit bent construction.

So: **bent is the cleanest**.

---

## The exact formal lemma that would essentially complete the separation

Assuming `computes sq e (f n)` pins the output to `embed (f n)` (or close enough that ℓ₁ still lower-bounds), the separation theorem collapses to two tiny facts:

1. **Peak ℓ₁ dominates final ℓ₁**
2. **Hard family has exponential ℓ₁ embedding norm**

Here’s the sharp “one lemma” you want (I’ll write it in the style of your goal):

```coq
Theorem hard_family_embed_l1 :
  exists f : forall n, Corner n -> bool,
  exists c : nat,
    forall n,
      (Qpow2 (c * n) <= l1_norm (embed (f n)))%Q.
```

Then the remaining step to finish `hard_family_separates` is basically one line, using the fact that the final value occurs in the trace maximum:

```coq
Lemma exc_l1_ge_output_l1 :
  forall n sq (e : GA_expr n),
    (l1_norm (eval_expr sq e) <= exc_l1 (exc_of sq e))%Q.
```

and then for any computation:

* `computes sq e (f n)` gives `eval_expr sq e = embed (f n)` (or `l1_norm (eval_expr sq e - embed (f n)) <= d0`)
* conclude `exc_l1 >= l1_norm(embed(f n))` (or `>= l1_norm(embed(f n)) - d0`)
* plug in `hard_family_embed_l1`

### If your `computes` is approximate rather than exact

Then the “exact lemma” becomes the robust form:

```coq
Theorem hard_family_embed_l1_robust :
  forall d0 : Q,
  exists f : forall n, Corner n -> bool,
  exists c : nat,
    forall n (F : MV n),
      l1_norm (mv_sub F (embed (f n))) <= d0 ->
      (Qpow2 (c * n) <= l1_norm F + d0)%Q.
```

But the cleanest is the exact-output version.

---

## If you want the hard family to actually *use* `trace_boolish_k_le`

The bent-family argument above doesn’t even need the trace-boolish hypothesis; it’s an “output is already large ℓ₁” separation.

If your intended narrative is “staying close to low-k Boolean shadows forces ℓ₁ blowup *even when the output itself might have modest ℓ₁*,” then you’d want a different hard family / different invariant. But for “cleanest candidate that matches your current theorem shape,” bent is the winner.

---

## What you’d prove concretely in Coq for the bent family

Define (for even n) a boolean function on `Corner n` corresponding to
```math
(-1)^{x_1x_2 \oplus x_3x_4 \oplus \cdots}.
```

Then prove:

1. `walsh_abs_constant` for that family: all Walsh coefficients have magnitude `Qpow2 (n/2)`.
2. `l1_norm_embed = (1 / Qpow2 n) * l1_norm_walsh` (or your existing lemma relating `embed` coefficients to character sums).
3. Conclude `l1_norm (embed f_n) == Qpow2 (n/2)` (or `>=`).

That feeds directly into `hard_family_embed_l1`.

---

# TWO VIABLE PATHS

* **Path A (fast, clean, already compatible with your current theorem shape):** separation via **output ℓ₁** (e.g. bent family) ⇒ exponential `exc_l1` essentially for free.
* **Path B (the “real” representation-vs-computation thesis):** separation that **uses** `trace_boolish_k_le` / low-k shadows as the *computational* restriction, and proves exponential cost even when output alone doesn’t force it.

Both are useful. A gives you a strong, mechanized, publishable “first separation.” B is the deeper target.

---

# Updated Roadmap: Representation ∥∥∥ Computation

## 0. Canonical definitions (lock these in)

### Excursion (already good)

* `exc_of sq e := (max_grade_during sq e, max_l1_during sq e)`
* Peak-based ⇒ **no cancellation** can ever reduce it. This is settled.

### Boolean “shadow” must be the structured one

You already diagnosed the key bug:

* `boolish_le` = “close to *some* boolean function” (too weak for complexity)
* `boolish_k_le` = “close to low-complexity lincomb shadow” (the right lever)

So the computational restriction must be:

* `trace_boolish_k_le`, **not** `trace_boolish_le`.

**Action:** sweep your statements and replace `trace_boolish_le` with `trace_boolish_k_le` wherever the intent is “easy computation stays near simple boolean structure.”

---

# 1. What “Representation ∥∥∥ Computation” should mean formally

You want a theorem schema like:

> **Small computation ⇒ trace stays in low-k boolean shadow with controlled error.**

Concretely:

```coq
Theorem rep_compilation_sound :
  forall (model_program : Program n) d,
    exists (sq : Vector.t Q n) (e : GA_expr n) k,
      (* correctness *)
      computes sq e (semantics model_program) /\
      (* complexity control *)
      k <= poly(size model_program, n) /\
      (* “stays near boolean structure throughout” *)
      trace_boolish_k_le sq e k d.
```

This is the *bridge theorem* in your framework: it makes your GA model a costed semantics for the “easy class.”

Everything else is either:

* proving this compiler theorem (closure infrastructure), or
* proving that some `f_n` defeats it (separation).

---

# 2. Separation theorem: pick the right statement

## Fix the statement first (this is important)

Your current admitted theorem uses `trace_boolish_le`, which is too weak. The corrected core separation target should be:

```coq
Theorem hard_family_separates :
  forall d : Q,
  exists f : forall n, Corner n -> bool,
  exists c : nat,
    forall n (sq : Vector.t Q n) (e : GA_expr n) k,
      computes sq e (f n) ->
      trace_boolish_k_le sq e k d ->
      (* plus: k is “small” for your easy class, either explicit or implied later *)
      Qpow2 (c * n) <= exc_l1 (exc_of sq e).
```

Then you’ll combine:

* “easy computation ⇒ small k” (from the compiler/simulation theorem)
* “hard family ⇒ exp excursion under small k trace-boolish” (this theorem)

That’s the real separation.

---

# 3. Two concrete endgames

## Path A: Fast separation via output ℓ₁ (recommended to get a finished theorem soon)

### Why this works

Because `exc_l1 = max_l1_during`, and the final output is in the trace, you have the always-true lemma:

```coq
Lemma exc_l1_ge_output_l1 :
  l1_norm (eval_expr sq e) <= exc_l1 (exc_of sq e).
```

So if you choose a family `f_n` with **exponentially large** `l1_norm (embed (f_n))`, then any exact computation of `embed (f_n)` forces exp `exc_l1` *regardless of trace_boolish_k_le*.

### Best explicit family: bent (even n), padded for odd n

Under your projector/character embedding, `embed(f)` coefficients are proportional to Walsh–Fourier coefficients, so:

* bent ⇒ flat spectrum ⇒ `||embed(f)||₁ = 2^{Ω(n)}` (typically `2^{n/2}`)

### What this gives you

* A fully mechanized separation theorem quickly.
* It validates your excursion mechanism and the overall narrative.

### What it does *not* give you

It doesn’t use the `trace_boolish_k_le` restriction—so it’s a “strong excursion lower bound,” not yet the “easy vs hard computation” story.

**Still worth doing** because it de-risks the entire stack and produces a big checkpoint theorem.

**Immediate tasks for Path A**

1. Prove `exc_l1_ge_output_l1`.
2. Define `bent_family : forall n, Corner n -> bool` (pad odd n).
3. Prove `l1_norm (embed (bent_family n)) >= Qpow2 (c*n)` (use your Walsh/character lemmas).

Then `hard_family_separates` becomes almost trivial (and you can *keep* the `trace_boolish_k_le` hypothesis; it just won’t be needed in the proof).

---

## Path B: The “real” Representation ∥∥∥ Computation separation (uses low-k trace-boolishness)

This is what your road map already anticipates, and it’s where `lincomb_embed_conv` etc. matter.

### B1. Make the trace restriction meaningful

Your separation theorem must *either*:

* quantify `k` and include a bound `k <= poly(n, size e)` (or `k <= poly(n)` for the “easy class”), or
* bake a specific `k(n, size)` into `trace_boolish_k_le`.

Otherwise an adversary can always choose enormous `k` and keep `d` small.

So you want something like:

```coq
computes sq e (f n) ->
trace_boolish_k_le sq e (poly(n, size e)) d ->
exp <= exc_l1 ...
```

### B2. Closure infrastructure: the absolutely load-bearing lemmas

To prove any “easy ⇒ trace_boolish_k_le” theorem, you need closure of `boolish_k_le` under your constructors.

This reduces to three families of lemmas:

#### (i) Representation rewrite (your original pain point)

* **Conv:**

  ```coq
  wf_lincomb cs1 gs1 ->
  wf_lincomb cs2 gs2 ->
  mv_conv (lincomb_embed cs1 gs1) (lincomb_embed cs2 gs2)
  =
  lincomb_embed (mul_coeffs cs1 cs2) (and_gens gs1 gs2).
  ```
* **GP:**
  analogous statement with the twist factor incorporated into the coefficient combiner.

These are what let you push semantic operators down to list combinators so ℓ₁ and k become bookkeeping.

#### (ii) k propagation

Prove (and reuse everywhere):

* `length (and_gens gs1 gs2) = length gs1 * length gs2`
* similar for GP gens
* thus `k` update rule (usually multiplicative under products)

#### (iii) ℓ₁ control

You need list-level lemmas like:

* `l1 (mul_coeffs cs1 cs2) <= l1 cs1 * l1 cs2`
  (and the GP version, which you said you already have in some form)

Then the actual closure lemma is routine:

* triangle inequality to combine approximation errors
* submultiplicativity to bound the new witness coefficients

### B3. “Simulation theorem” (easy computation ⇒ trace_boolish_k_le)

Once closure lemmas exist, you prove:

* the compiler from Boolean formulas / your chosen “easy” model into GA_expr (already largely exists via `translate`)
* plus a costed invariant:

  * `k` bounded by poly(size)
  * `d` bounded (or evolves controllably)

This is the theorem that justifies calling the restriction “easy computations stay in the shadow.”

### B4. The hard family lower bound (the only genuinely new math)

Now you prove:

> Any computation of `f_n` that stays within `d` of a `k`-generator lincomb at every step must have exp `max_l1_during`.

This is where you use:

* dimension/counting
* Fourier mass / correlation
* rank/influence
* whatever your “representation triple bar” measure actually formalizes

This is the only phase that can’t be reduced to library plumbing.

---

# Best path in practice

## Do Path A first (fast checkpoint), *then* Path B

Path A gives you:

* a finished, correct, mechanized exponential theorem using your existing excursion notion
* confidence your embedding/eval/ℓ₁ apparatus is aligned
* a benchmark hard family you can reuse

Then Path B becomes “upgrade the theorem so it depends on `trace_boolish_k_le` + small k,” rather than “prove everything at once.”

This is the same strategy as in complexity theory: prove an unconditional lower bound for a restricted measure first, then strengthen.

---

# Concrete next steps (highest ROI)

### Step 1: Fix the separation theorem statement

Replace `trace_boolish_le` with `trace_boolish_k_le` in the theorem you actually care about.

### Step 2: Prove the trace peak lemma

`exc_l1_ge_output_l1` (tiny but unlocks Path A).

### Step 3: Implement the bent family in your `Corner n -> bool` language

Then prove `l1_norm (embed f_n) >= 2^{c n}`.

### Step 4: Finish `lincomb_embed_conv` (wf version)

This is the main blocker for the closure infrastructure needed by Path B.

### Step 5: Make `boolish_k_le` closure lemmas for `Add`, `Conv`, `Mul`

Once these are done, `trace_boolish_k_le` becomes a standard structural invariant: easy programs satisfy it automatically with controlled parameters.

---

# One last alignment check

Your current `hard_family_separates` concludes exponential lower bound on `exc_l1`. That’s great—but it means your hard family must force *ℓ₁ blowup*. Bent does that very naturally.

If later you want a family where **output ℓ₁ is modest but computation still forces exp peak ℓ₁**, that’s strictly stronger and will require the full Path B machinery.

---

Got it — “≡∥∥∥” is *equivalence*, not “equals by definition.” That **doesn’t change the roadmap**, but it does change how we should phrase the two key bridge theorems:

* we should target **equivalence of computations** (same Boolean semantics / same projected meaning), not literal equality of multivectors.

Below is the updated roadmap wording to reflect **≡∥∥∥** everywhere it matters, without changing the substance.

---

# Updated Roadmap (with ≡∥∥∥ equivalence)

## What ≡∥∥∥ should mean operationally

In your framework, the natural notion of equivalence is:

> two GA expressions are ≡∥∥∥ equivalent if they induce the same Boolean function (or are within tolerated error (d)) under the evaluation/projection semantics you care about.

So when we say “representation ∥∥∥ computation”, the intended statement is:

* the representation layer (convolution / Boolean shadows) and the computation layer (GP program traces) are **semantically equivalent** even though their internal multivector states differ.

This matches your `computes` notion (whatever it is) far better than forcing literal MV equality.

---

## Phase 0 — Lock in the right equivalence relations

### 0.1 Boolean semantic equivalence

Pick one of these as your canonical ≡∥∥∥:

* **Exact**: `forall s, eval_expr sq e s = f s`
* **Embed exact**: `eval_expr sq e = embed f`
* **Approx**: `l1_norm (eval_expr sq e - embed f) <= d0`

Whichever your project uses, everything should be phrased in that equivalence.

### 0.2 Trace-equivalence constraint

For “easy computations,” you want:

* `trace_boolish_k_le sq e k d`

This expresses “the entire evaluation trace stays close to low-complexity Boolean shadows.”

This is the “stays in the ≡∥∥∥-meaningful region” condition.

✅ This matches the previous roadmap exactly; no change needed.

---

## Phase 1 — Compositional closure (still the key plumbing)

You still need the same load-bearing lemmas, because they are what let you propagate `trace_boolish_k_le` inductively.

### 1.1 Representation rewrite lemmas (Conv and GP)

These are still critical and unchanged — they are about how products act on your structured witness class, not about ≡∥∥∥ itself.

* `lincomb_embed_conv` (wf version)
* GP analogue (twist-aware)

### 1.2 k propagation and ℓ₁ bounds

Same as before:

* length bounds for `and_gens`
* ℓ₁ submultiplicativity for `mul_coeffs` / GP combiner

These are what turn “equivalence constraint” into “resource cost.”

---

## Phase 2 — Simulation theorem (now phrased as ≡∥∥∥)

This is the first place the ≡∥∥∥ clarification matters in wording:

> Small Boolean computations compile into GA expressions that are **≡∥∥∥ equivalent** to the intended Boolean function, and their traces satisfy `trace_boolish_k_le` with controlled `k,d`.

Form:

```coq
Theorem simulation_triplebar :
  forall (P : Program n),
  exists sq e k,
    computes sq e (semantics P)  (* this is your ≡∥∥∥ correctness *)
    /\ k <= poly(size P, n)
    /\ trace_boolish_k_le sq e k d.
```

No structural changes; just making clear “computes” is the equivalence.

---

## Phase 3 — Separation theorem (now explicitly “equivalence separation”)

Again, same structure, just phrased correctly:

```coq
Theorem hard_family_separates :
  forall d,
  exists f, exists c,
    forall n sq e k,
      computes sq e (f n) ->
      trace_boolish_k_le sq e k d ->
      (* with k “small enough” for the easy class, either as an assumption
         or obtained by combining with the simulation theorem *)
      Qpow2 (c*n) <= exc_l1 (exc_of sq e).
```

This is a separation **under ≡∥∥∥ equivalence**: any computation *equivalent* (in your semantics) to the hard family must pay the cost if it stays in the shadow.

---

# Two endgames still apply (unchanged)

## Path A (fast checkpoint)

Pick (f_n) with exponentially large `l1_norm (embed f_n)`.

Even if your “computes” relation is equivalence rather than equality, you can use a robust version:

* if `computes` implies `l1_norm(eval - embed f) <= d0`,
  then `l1_norm(eval) >= l1_norm(embed f) - d0`.

So Path A still works with ≡∥∥∥, you just subtract the output tolerance.

## Path B (full thesis)

Use low-k trace-boolishness as the computational restriction; prove hard family forces exp peak ℓ₁ even when output alone doesn’t.

Again: unchanged.

---

# Best path remains the same

1. Fix any theorem that mistakenly uses `trace_boolish_le` instead of `trace_boolish_k_le`.
2. Finish `lincomb_embed_conv` (wf) + GP analogue + ℓ₁ list lemmas.
3. Get a Path A hard family as a “first separation checkpoint.”
4. Then push to Path B for the full representation-vs-computation separation.

---

Below is an updated roadmap that incorporates everything we discussed, fixes the `boolish_le` vs `boolish_k_le` issue explicitly, reflects what your Coq definitions actually are (`max_grade_during`, `max_l1_during`, `trace_boolish_k_le`, `boolish_k_le`), and clarifies what “Bridge 2” is in *your* formalization (not a GP→Conv homomorphism, but a **trace shadow + cost** theorem). I also keep the ≡∥∥∥ equivalence framing.

---

````md
# Abstract

We develop a geometric-algebraic framework for analyzing Boolean computation through the interaction of two algebraic products on multivectors over ((ℤ₂)^n):

* an **untwisted convolution product** (`mv_conv`), which admits a clean multiplicative evaluation map and models Boolean logic compositionally; and
* a **twisted geometric product** (`mv_gp`), which is literally convolution with an additional sign/metric **2-cocycle** (twist) that causes grade mixing and geometric interference.

Boolean functions are embedded into multivectors via a projector–character expansion, and evaluation against characters recovers Boolean semantics. Convolution provides a flat, commutative logic layer, while the geometric product induces signed, noncommutative dynamics.

We introduce a quantitative invariant—**excursion**—which measures *peak geometric complexity over the execution trace*, combining maximal grade and maximal ℓ₁ mass. We also define a structured Boolean-shadow condition (`boolish_k_le`) requiring that intermediate multivectors remain close (in ℓ₁) to a bounded-complexity linear combination of embedded Boolean generators.

The central program is to prove a separation theorem:

> There exists a Boolean function family (fₙ) such that any geometric-algebra expression computing (fₙ), whose execution trace remains close to *low-complexity Boolean shadows*, must incur exponential ℓ₁ excursion.

This establishes a representation-theoretic lower bound: simulating flat Boolean logic inside twisted geometry forces unavoidable geometric growth.

---

# Conceptual Architecture

## 1. Two Algebraic Worlds

### 1.1 Untwisted world (Convolution)

* Product: `mv_conv`
* Evaluation is multiplicative:
  eval(F ⊙ G, s) = eval(F,s) · eval(G,s).
* Boolean AND corresponds cleanly to convolution.
* This layer models **Boolean semantics** (flat, commutative).

### 1.2 Twisted world (Geometric Product, GP)

* Product: `mv_gp sq`
* Same XOR routing as convolution, but multiplied by a sign/metric cocycle:
  GP = (convolution) × (twist coefficient).
* Twist causes grade mixing and interference.
* This layer models **geometric computation** (signed, noncommutative).

**Key structural fact (already present in the code):**
GP is literally twisted convolution—same mask_xor indexing, extra coefficient factor.

---

## 2. Embedding and Evaluation

Boolean functions embed via projectors/characters:
  embed(f) = Σ_a f(a) Π(a)

and evaluation recovers Boolean meaning:
  eval(embed(f), s) = f(s).

There are multiple embedding styles (parity/XOR behaves differently depending on embedding), but the core property is: embedding + evaluation recovers Boolean semantics.

You have already proven:

* `translate_correct` (Boolean translation is semantically correct in your GA semantics)
* AND is handled compositionally via `Conv`/`eval_conv` in the translation proof
* Boolean logic can be expressed in the GA language (which contains both Conv and GP)

So **expressive equivalence** (computability of Boolean logic) is established.

---

## 3. What Separation Means Here (≡∥∥∥)

This is not a classical “SAT formula size” statement by default; it is a statement about **computational cost under a semantic equivalence**.

**Expressive equivalence ≠ computational equivalence.**

Concretely: two algebras/program models can compute the same Boolean functions, yet differ dramatically in the cost needed to do so while preserving a structured Boolean shadow during execution.

---

# The Formal Ingredients You Have

## 4. Excursion (Peak-over-trace)

```coq
Record ExcNum := {
  exc_grade : nat;
  exc_l1    : Q;
}.
````

with

* `exc_grade = max_grade_during sq e`
* `exc_l1    = max_l1_during   sq e`

and `exc_of sq e := {| exc_grade := ...; exc_l1 := ... |}`.

**Crucial point:**
Because both components are defined using `Nat.max` over the trace, excursion is **peak-based**.
Later cancellations cannot reduce it.

So the “can excursion cancel?” question is settled for this metric: peaks do not cancel.

(That is why grade-only excursion gave robust but bounded (≤ n) lower bounds.)

---

## 5. Boolean Shadow (Structured, complexity-aware)

```coq
boolish_k_le F k d :=
  exists cs gs,
    wf_lincomb cs gs /\
    length gs <= k /\
    l1_norm (F - lincomb_embed cs gs) <= d.
```

This is your **projection-with-complexity** notion:

* k controls “shadow complexity” (# generators),
* d controls approximation error (ℓ₁),
* `lincomb_embed` is the structured approximation class.

**Important correction (agreed):**
`boolish_le` = “close to some Boolean function” is too weak for complexity.
The separation infrastructure must use `boolish_k_le` (and its trace version).

---

## 6. Trace Condition (the correct one)

```coq
trace_boolish_k_le sq e k d
```

Every intermediate subexpression remains `boolish_k_le` with budget (k,d).

This encodes:

> “Easy computations never leave the low-complexity Boolean shadow.”

This is the trace-level restriction that makes the separation meaningful.

(Any theorem using `trace_boolish_le` where you meant structured shadow is a bug/placeholder and should be swapped to `trace_boolish_k_le`.)

---

# What the Separation Theorem Must Prove

## 7. Core target statement (fixing the hypothesis)

```coq
Theorem hard_family_separates :
  forall d : Q,
  exists f : forall n, Corner n -> bool,
  exists c : nat,
    forall n (sq : Vector.t Q n) (e : GA_expr n) k,
      computes sq e (f n) ->
      trace_boolish_k_le sq e k d ->
      Qpow2 (c * n) <= exc_l1 (exc_of sq e).
```

In words:

If a GA computation stays close to low-complexity Boolean shadows (trace_boolish_k_le) and computes a hard family, then peak ℓ₁ excursion must be exponential.

**Note (needed for “easy vs hard”):**
To translate this into a separation of an “easy model,” you also need a simulation theorem tying “easy program size” to a bound on k (typically k ≤ poly(size,n)) and controlled d.

---

# Bridge 2 (the hard bridge) — what it is in your formalization

Bridge 1 (already established):

* convolution + characters ⇒ evaluation is a homomorphism ⇒ clean Boolean semantics.

Bridge 2 (the separation bridge) is NOT a homomorphism GP→Conv.
In your Coq development it is:

> **Trace shadow + GP dynamics ⇒ unavoidable cost (excursion growth).**

Concretely:

* computations occur in the *twisted* space (GP, with signs and grade mixing),
* while the trace is constrained to remain close to a *low-complexity Boolean shadow*,
* and the theorem forces exponential peak ℓ₁.

That is the formal “signed vs unsigned” link:
the *meaning* stays Boolean, but the *geometry* must grow.

---

# Why Cancellation Is Not the Issue

Your excursion is peak-based:

* `max_grade_during` records the highest grade reached at any node.
* `max_l1_during` records the largest ℓ₁ mass reached at any node.

Even if later steps cancel coefficients, the maxima have already been recorded.
So the only question is whether computation can avoid creating large peaks at all.

---

# Updated Roadmap Toward Separation

## Phase 0 — Hygiene / Statement Alignment (do this now)

1. Replace any use of `trace_boolish_le` in “separation / simulation” claims with `trace_boolish_k_le`.
2. Ensure `computes` is the intended ≡∥∥∥ correctness notion (exact MV equality, eval equality, or ℓ₁-approximate correctness). Keep all theorems stated in that equivalence.

---

## Phase 1 — Closure Infrastructure (Critical, load-bearing)

These are the plumbing lemmas that make trace_boolish_k_le compositional.

### 1.1 Representation rewrite lemmas

These push semantic operations down to list combinators so you can reason about k and ℓ₁ at the witness level.

#### (a) Convolution rewrite (must be wf-aware)

Goal shape (with wf assumptions):

```coq
mv_conv (lincomb_embed cs1 gs1) (lincomb_embed cs2 gs2)
=
lincomb_embed (mul_coeffs cs1 cs2) (and_gens gs1 gs2).
```

*Must* be stated with `wf_lincomb cs1 gs1` and `wf_lincomb cs2 gs2` (the un-wf version is false).

#### (b) Geometric product rewrite (twist-aware)

An analogous lemma for `mv_gp`:
same XOR routing, coefficients gain the twist factor.
This will be the workhorse for boolish closure under GP.

(Depending on how you package twist in coefficients, this might be “mul_coeffs_gp” + “and_gens” or a variant.)

### 1.2 k propagation bookkeeping

Prove and reuse:

* `length (and_gens gs1 gs2) = length gs1 * length gs2` (or ≤, but equality holds for cartesian product)
* `length (mul_coeffs cs1 cs2) = length cs1 * length cs2`

These establish the natural k update (multiplicative under products).

### 1.3 ℓ₁ control for coefficient combinators

List-level inequalities:

* `l1(mul_coeffs cs1 cs2) <= l1(cs1) * l1(cs2)`
* and the twist-aware GP analogue (you said you already have a GP submultiplicativity lemma; align it with the rewrite lemma so it plugs in cleanly).

These are what prevent “cheap cancellation” via massive coefficient blowups inside witnesses.

---

## Phase 2 — Trace Stability (boolish_k_le closure)

Prove the closure lemmas needed to build `trace_boolish_k_le` by structural induction on expressions:

* Add preserves `boolish_k_le` (triangle inequality + witness concatenation)
* Conv preserves `boolish_k_le` (use Phase 1.1a + 1.2 + 1.3)
* GP preserves `boolish_k_le` (use Phase 1.1b + 1.2 + 1.3)

This yields:

> If subexpressions are boolish_k_le with budget k, then the parent node is boolish_k_le with updated budget k′ and updated error d′.

(You can keep “single k everywhere” as a global budget, or strengthen to a version that computes k′ structurally and then derives the global-budget version.)

---

## Phase 3 — Simulation Theorem (easy ⇒ trace_boolish_k_le with small k)

This is where you connect a chosen “easy model” (e.g., formulas / your program syntax) to GA expressions.

Target form:

```coq
Theorem simulation_triplebar :
  forall (P : Program n),
  exists sq e k,
    computes sq e (semantics P) /\
    k <= poly(size P, n) /\
    trace_boolish_k_le sq e k d.
```

This is the theorem that makes `trace_boolish_k_le` the correct formalization of “easy computations stay in the Boolean shadow.”

---

## Phase 4 — Hard Family Lower Bound (the real new math)

Pick a family fₙ and prove:

> Any computation that is ≡∥∥∥-correct for fₙ and maintains trace_boolish_k_le with small k (and small d) must incur exponential peak ℓ₁.

This is the heart of the separation.

Techniques that plausibly match your framework:

* Fourier/Walsh ℓ₁ mass arguments (embed coefficients relate to character sums)
* dimension/counting arguments for low-k lincomb approximations
* rank/correlation bounds against low-complexity generator families

This is where your “representation ∥∥∥ computation” thesis lives.

---

## Phase 5 — Combine (the actual separation)

Combine:

1. Simulation theorem (easy ⇒ trace_boolish_k_le with k small)
2. Hard family theorem (trace_boolish_k_le with k small ⇒ exp excursion)
3. Excursion definition (exc_l1 is peak-over-trace)

Conclude: easy model cannot compute fₙ without paying exponential excursion (hence separating complexity regimes).

---

# Two Practical Endgames (both valid)

## Path A — Fast checkpoint theorem (output-driven ℓ₁)

Because `exc_l1 = max_l1_during` and the output occurs in the trace, prove:

```coq
exc_l1 (exc_of sq e) >= l1_norm (eval_expr sq e).
```

Then choose fₙ such that `l1_norm(embed(fₙ)) >= 2^{Ω(n)}`.
This yields an unconditional exponential excursion lower bound for computing fₙ.

This is a strong, mechanizable milestone and validates the excursion machinery.

(It may not use trace_boolish_k_le in an essential way, but it produces a finished exponential theorem quickly.)

## Path B — The full thesis (trace_boolish_k_le is essential)

Prove exponential cost specifically from the constraint that the trace remains close to low-k Boolean shadows, even in regimes where output alone would not force ℓ₁ to be large.

This is strictly stronger and is the “representation ∥∥∥ computation” separation in its intended form.

---

# What to Focus On Now (highest ROI)

1. Fix theorem statements to use `trace_boolish_k_le` (not `trace_boolish_le`) where intended.
2. Finish `lincomb_embed_conv` (wf version).
3. Align the GP analogue rewrite lemma with your existing `l1_gp_submultiplicative`.
4. Prove the k/length bookkeeping lemmas for `and_gens` and `mul_coeffs`.
5. Finish boolish_k_le closure lemmas for Add / Conv / GP.
6. Choose hard family and prove the approximation/lower-bound lemma that forces exp peak ℓ₁ under the trace shadow constraint.

Everything else is supporting infrastructure.

---

# Big Picture

You have built:

* Algebraic semantics (Conv + GP, with GP = twisted convolution)
* Boolean embedding + evaluation semantics
* Correct Boolean compilation (`translate_correct`)
* Peak excursion metric (grade + ℓ₁) immune to cancellation
* Structured Boolean shadow notion (`boolish_k_le`) and its trace form

You are no longer exploring.
You are in the final quantitative phase:

> Prove that maintaining low-complexity Boolean shadows under twisted dynamics forces exponential peak ℓ₁.

If that holds for some explicit family fₙ, the separation theorem follows.

```

This version keeps every point we discussed:

- the two products and their roles,
- GP as twisted convolution (same XOR routing + twist),
- Bridge 1 vs Bridge 2 (and Bridge 2 as trace shadow + cost, not GP→Conv homomorphism),
- why cancellation is irrelevant (peak metrics),
- the critical correction: use `trace_boolish_k_le`, not `trace_boolish_le`,
- the need to bind k via simulation for an “easy vs hard” separation,
- the truly load-bearing lemmas (rewrite + length + ℓ₁ submult),
- the two endgames (Path A checkpoint vs Path B full thesis),
- and a concrete “what to do now” list.

```

Here’s a roadmap that matches what you’ve already built (boolish witnesses, `k`/`d` calculus, excursion metric, GP vs Conv), and lays out the pathways to the “representation ⇔ computation” equivalence that would make a P≠NP-style separation a corollary *inside your framework*.

---

# Roadmap: Representation ⇔ Computation (Excursion + Phase)

## North Star deliverable

A theorem schema of the form:

> **(Representation/Computation Equivalence)**
> For an appropriate class of computations (\mathcal{C}) (your `GA_expr` traces),
> “computable with poly resources” ⇔ “admits poly-bounded boolish representation along the trace”
> with quantitative bounds in `k`, `d`, and `exc_l1`.

Then a corollary pipeline:

> If a language (e.g. SAT via an encoding) forces super-poly excursion *or* super-poly representational phase cost under that equivalence, it’s not in the poly-computable fragment.

You’re explicitly *not* proving P≠NP in ZFC; you’re building the internal equivalence and a clean separation statement.

---

# Pathway A: Excursion as a Complexity Measure

### A1. Define the “poly computation” class in your system

* **Goal:** a canonical predicate like:

  * `poly_trace sq e d` or reuse `trace_boolish_poly sq e d`
  * ensure it is stable under your semantics (`computes`, `exc_of`)
* **Deliverable:** a single definition that is used everywhere downstream.

### A2. Prove *upper bounds*: poly-boolish ⇒ bounded excursion

* **Goal:** a lemma that turns representation constraints into geometric constraints.
* **Target statement shape:**

  * `trace_boolish_poly sq e d -> exc_l1 (exc_of sq e) <= poly_bound(n, expr_size e, d)`
* **How:** reuse/finish your “core lemma you’ll reuse” (GP error bound) and extend it to conv/mul/add.
* **Deliverables:**

  * `exc_upper_bound_of_boolish` (master lemma)
  * closure-based corollaries for `Add`, `Mul`, `Conv` (each updating the bound)

### A3. Prove *lower bounds*: hard families force large excursion

* **You already have:** `hard_family_separates` with `trace_boolish_exists_k`.
* **Next:** strengthen/standardize it to consume `trace_boolish_poly` (your canonical version).
* **Deliverable:**

  * `hard_family_separates_poly : ... -> trace_boolish_poly sq e d -> Qpow2(c*n) <= exc_l1 ...`

### A4. Identify/encode target problems (SAT path later)

* **Goal:** an encoding lemma: “SAT instance ↦ `Corner n -> bool` / `GA_expr` trace”.
* **Deliverables:**

  * `encode_cnf : CNF -> exists n, Corner n -> bool`
  * `sat_correctness : sat φ <-> exists assignment, ...`
  * `computes_of_encoding : ... computes sq e (f n)`

This is the “bridge to complexity theory,” but you can postpone it until the geometry/representation core is solid.

---

# Pathway B: Phase / Twist Cost (GP vs Conv) as Representational Complexity

This is the “−1111 becomes long” pathway.

### B1. Formalize a “phase complexity” predicate

You already have the right object: `boolish_k_le F k d`.

Package it into “approximate membership in the boolean manifold”:

* `approx_boolish(F, K, d) := ∃k≤K, boolish_k_le F k d`
* `far_from_boolish(F, K, d0) := ∀d<d0, ¬ approx_boolish(F,K,d)`

**Deliverable:** one file with these wrappers and basic monotonicity lemmas.

### B2. Prove conv is “diagonalizable” / flat in the boolish regime

* **Goal:** show that conv respects a transform or a basis that preserves low complexity.
* In your ecosystem this likely uses `Cln_BoolDist` (Walsh / distribution lemmas).
* **Deliverables:**

  * `conv_preserves_boolish_poly` (already aiming for `boolish_k_le_conv`)
  * optional: `conv_characterization` (conv = pointwise mult in transform domain)

### B3. Prove GP introduces a nontrivial cocycle (twist)

* **Goal:** isolate the sign kernel used by `mv_gp`.
* **Deliverables:**

  * a lemma that writes GP as “twisted convolution”:

    * `(mv_gp F G) U = Σ_{A⊕B=U} s(A,B) * F A * G B`
  * `abs_s_is_1 : Qabs (s(A,B)) == 1` (or equalities you actually use)

This makes later proofs copy-paste from conv, except for the sign/twist reasoning.

### B4. The key lower bound: twist forces k blowup or d blowup

This is the exact formal version of your “negativity becomes long” intuition.

* **Goal statement shape:**

  * There exists a family `H n : MV n` such that:

    * `H n` is easy to generate by GP (small `GA_expr`)
    * but any boolish approximation with `k ≤ poly(n)` must have `d` bounded below (not tiny),
      or equivalently if `d` is tiny then `k` must be huge.
* **Deliverable:** `twist_forces_far_from_boolish` (quantitative).

This is the centerpiece of the representational pathway.

### B5. Connect B4 to excursion via A2

Once you have “either k huge or d huge,” you combine with A2 to get:

* poly-k + small d ⇒ small excursion
* but `H n` has large excursion (or forces large excursion)
  ⇒ contradiction

**Deliverable:** a single “composition theorem” that takes a representational lower bound and outputs a geometric lower bound (or vice versa).

---

# Pathway C: Representation ⇔ Computation Equivalence Theorem

This is where you unify A and B into the equivalence you actually want.

### C1. Define the resource measures cleanly

Pick a tuple, e.g.:

* size resource: `expr_size e`
* representational resource: `k`
* approximation resource: `d`
* geometric resource: `exc_l1`

Then define “poly resource” as:

* `k ≤ poly_k n (expr_size e)`
* `d ≤ poly_d n (expr_size e)` (optional; you currently pass `d` universally)

### C2. Prove “⇒” direction: poly computation implies poly representation (soundness)

* **Meaning:** If a trace computes in your allowed model with bounded structural resources, then along the trace it remains approx-boolish with poly k and controlled d.
* This is the hardest direction conceptually; but your development suggests you’re *building it by closure*:

  * base cases (Basis/Scalar) are boolish
  * Add/Mul/Conv/GP preserve boolish with explicit k/d update rules
* **Deliverable:** `trace_boolish_poly_sound : syntactic_poly e -> trace_boolish_poly sq e d`

### C3. Prove “⇐” direction: poly representation gives poly computation (completeness)

* **Meaning:** if along the trace you have poly-boolish witnesses, you can simulate/compute in a poly-time-ish way (in your formal notion).
* Practically: show that from the witness (lists `cs`, `gs` with length k) you can extract a “poly evaluator” for the trace or approximate outputs.
* **Deliverable:** `trace_boolish_poly_complete : trace_boolish_poly sq e d -> exists poly evaluator ...`

This is the true “equivalence” statement you’re aiming for.

### C4. Corollary separation template

Once C2–C3 exist, you can state separation results cleanly without invoking external complexity theory:

> If a problem family forces `far_from_boolish` (or large excursion) under the encoding, it is not in your poly-trace class.

This is where SAT can drop in as a corollary once encoded.

---

# Pathway D: SAT as a Corollary (Optional / Later)

### D1. Choose an encoding

* CNF formulas → circuits → your `Corner n -> bool`
* ensure uniformity in `n` and size

### D2. Show SAT forces the twist family

* Show that deciding SAT would imply computing a twist-heavy `H n` (or approximating it below some d0).
* This is the reduction step inside your model.

### D3. Conclude non-poly in your model

* Combine D2 + B4 + A2 (or the unified C theorem).

This yields your internal “SAT not in poly-trace-boolish” theorem, and *then* you can interpret it as a P≠NP-style separation once you assert your model captures P.

---

# Immediate Next Milestones (what to do next week in Coq)

1. **Finish closure + wf infrastructure**

   * `wf_lincomb_mul_and`
   * `boolish_k_le_conv`
   * and the GP analogue(s) if missing

2. **Standardize equality / rewriting discipline**

   * decide whether you work with `=` or a setoid relation for Q-valued functions
   * make one “extensionality for l1_norm / sumQ / mv_*” lemma that you reuse everywhere

3. **Write the “master sandwich” lemma**

   * `(poly boolish) -> (exc upper bound)`
   * then `hard_family_separates` becomes plug-and-play

4. **Design the twist witness family `H n`**

   * simplest GP-generated element whose conv/boolish approximation provably needs large k or d

That last item is the mathematical heart of the “−1111 becomes long” pathway.

---

Cool — here’s a concrete, file-level plan that fits your current module layout (`Cln_Full`, `Cln_Grade`, `Cln_finite_l1_submultiplicativity`, `Cln_BoolDist`, `Cln_CompositeExcursion`) and minimizes edits downstream.

I’m going to structure this as “what lives where”, “what order”, and “what the key reusable lemmas look like”.

---

# File-level plan

## 0) Conventions to adopt once (so you stop fighting Coq)

**Goal:** avoid the `Q`/`=` headaches and make rewriting predictable.

* Treat all `MV n` equalities extensionally, and for `Q` arithmetic prefer *normal-form lemmas* you control rather than hoping library lemmas exist.
* Add a small internal lemma library (in one place) for:

  * pointwise extensionality of `mv_add`, `mv_sub`, `mv_scale`, `mv_conv`, `mv_gp`
  * `l1_norm` monotonicity and triangle bounds
  * `sumQ` congruence wrt pointwise equality (whatever equality notion you’re using)

**Where:** create a tiny helper section near the top of `Cln_Full.v` or a new file `Cln_AlgebraLemmas.v` imported by everything.

**Immediate fix:** replace “`mv_sub_cancel : ... = ...`” by a lemma stated in the equality you actually use for rewriting (often pointwise `=` is fine because terms are computed, but don’t rely on non-existent `Qeq_eq` conversions).

---

## 1) `Cln_Full.v` (core defs + small extensional lemmas)

**Keep this file definition-heavy. Only add lemmas that every file needs.**

### 1.1 Definitions to keep stable

* `MV n := Mask n -> Q`
* `sumQ`, `all_masks`
* `l1_norm`
* `mv_conv`, `mv_gp` (if gp is defined here), `mv_add`, `mv_sub`, etc.
* your generator encodings (`lincomb_embed`, `wf_lincomb`, `mul_coeffs`, `and_gens`)

### 1.2 Add these “plumbing” lemmas here (or a helper module it exports)

These stop 80% of future pain:

* `mv_ext : (forall m, F m = G m) -> F = G`
* `sumQ_ext` / `sumQ_map_ext`
* `l1_norm_ext` (if you can get it)
* `l1_norm_triangle : l1_norm (mv_add F G) <= l1_norm F + l1_norm G`
* `l1_norm_scale`, `l1_norm_nonneg`
* any “sum over all masks respects pointwise equality”

**Milestone:** after this, you should be able to do almost all algebra proofs without ad-hoc rewriting.

---

## 2) `Cln_BoolDist.v` (representation theory: boolish, generators, wf)

This is where your requested lemmas belong.

### 2.1 Finish well-formedness closure first

You already hit:

```coq
Lemma wf_lincomb_mul_and :
  forall n csF gsF csG gsG,
    wf_lincomb csF gsF ->
    wf_lincomb csG gsG ->
    wf_lincomb (mul_coeffs csF csG) (and_gens gsF gsG).
```

**Make it compile** by either:

* dropping `n` if unused, OR
* typing it explicitly if Coq can’t infer: `(n : nat)`.

The error “Cannot infer the type of n” means `n` is syntactically unused in the statement/proof, so Coq can’t guess its type.

**Fix:**

```coq
Lemma wf_lincomb_mul_and :
  forall (n : nat) csF gsF csG gsG,
    wf_lincomb csF gsF ->
    wf_lincomb csG gsG ->
    wf_lincomb (mul_coeffs csF csG) (and_gens gsF gsG).
```

Then your proof via lengths is exactly right *if* `wf_lincomb` is just `length cs = length gs`. If it has extra conditions, finish those here too.

**Milestone:** all `wf_lincomb_*` closure lemmas done:

* for `Add`-style list append
* for `Mul`-style `mul_coeffs/and_gens`
* for any `Conv` witness constructor you use

### 2.2 Prove “boolish is closed under conv” next

Your target:

```coq
Lemma boolish_k_le_conv :
  forall n (F G : MV n) k1 k2 d1 d2,
    boolish_k_le F k1 d1 ->
    boolish_k_le G k2 d2 ->
    boolish_k_le (mv_conv F G) (k1 * k2)
      (d1 * l1_norm G + l1_norm F * d2 + d1 * d2).
```

**Why in this file:** it’s fundamentally “representation calculus”: witness construction + norm bound.

**Proof structure (almost certainly):**

1. destruct the boolish witnesses for `F` and `G`:

   * you get `csF gsF` and `csG gsG`
   * plus wf/length and an approximation inequality (your `Hd` field)
2. build witness for `mv_conv F G`:

   * coeffs: `mul_coeffs csF csG`
   * gens: `and_gens gsF gsG`  (or the conv-appropriate generator operator if different)
3. show wf/length using `wf_lincomb_mul_and`
4. prove the error bound using the same “(F + ΔF) ⋆ (G + ΔG) − F⋆G” split:

   * `ΔF ⋆ G` + `F ⋆ ΔG` + `ΔF ⋆ ΔG`
   * then apply an ℓ₁ submultiplicativity lemma for conv (if you don’t have it, you prove it once: see below)
   * and triangle inequality.

**Milestone dependencies:**

* `l1_conv_submultiplicative` or equivalent:

  * `l1_norm (mv_conv A B) <= l1_norm A * l1_norm B`
* “error split” lemma for conv:

  * `mv_conv (F+ΔF) (G+ΔG) - mv_conv F G = ...`
  * you can do pointwise ext + ring-ish rewriting at Q-level (but keep it in MV space).

If you *don’t* have `l1_conv_submultiplicative`, prove it here (or in `Cln_finite_l1_submultiplicativity.v` if you want it generic).

---

## 3) `Cln_finite_l1_submultiplicativity.v` (norm algebra)

This file should become your “one-stop shop” for ℓ₁ submultiplicativity lemmas.

### 3.1 Make the interface uniform

You already have a GP version:

* `l1_gp_submultiplicative`

Add the conv analogue here:

* `l1_conv_submultiplicative : l1_norm (mv_conv F G) <= l1_norm F * l1_norm G`

Even if you prove it by reducing conv to gp with trivial sign, keep it here because every downstream closure lemma uses it.

### 3.2 Export “error split + norm bound” templates

This is the reusable pattern:

* `l1_error_split_bilinear` for any bilinear operator `⋆` with submultiplicativity:

  * If you can abstract it, great.
  * If not, just provide two specializations:

    * `gp_error_split`
    * `conv_error_split`

Then `boolish_k_le_gp` and `boolish_k_le_conv` become “same proof, different operator”.

**Milestone:** after this, `Cln_BoolDist.v` becomes mostly witness plumbing; all real inequalities live here.

---

## 4) `Cln_CompositeExcursion.v` (trace layer: from MV lemmas to GA_expr lemmas)

This file should **only** talk about `GA_expr` and `trace_boolish_k_le`.

### 4.1 Keep your `trace_boolish_*` API stable

You already stabilized:

* `trace_boolish_exists_k`
* `trace_boolish_poly` (exists k ≤ poly_k …)
* monotonicity lemma `trace_boolish_k_le_mono`
* canonicalization lemmas to/from `poly_k`

Good. Don’t touch this API again.

### 4.2 Add closure lemmas at the trace level

Prove these by structural recursion on `GA_expr`, delegating all heavy work to `Cln_BoolDist` lemmas:

* `trace_boolish_k_le_add` (already similar)
* `trace_boolish_k_le_mul` uses `boolish_k_le_gp` (or your Mul meaning)
* `trace_boolish_k_le_conv` uses `boolish_k_le_conv`

Then the poly versions follow by `trace_boolish_k_le_to_poly_k` + size bound lemma.

**Milestone:** you get a single theorem:

* `trace_boolish_poly_closed : closed under Add/Mul/Conv`

That’s the “representation ⇒ computation stability” direction.

---

## 5) “Boundary / hardness” layer (new file or `Cln_CompositeExcursion.v`)

Once the closure calculus is stable, add a *separate* file for hardness statements so you don’t destabilize core proofs.

Suggested name: `Cln_HardnessBoundary.v`

### 5.1 Define your boundary predicates

* `approx_boolish` / `far_from_boolish`
* `excursion_large` predicates

### 5.2 Prove the two master implications

1. **Poly-boolish ⇒ excursion upper bound**
2. **Twist/phase family ⇒ far_from_boolish or large excursion**

Then `hard_family_separates` becomes an instance, and SAT encoding can be plugged later.

---

# Order of work (what to do next)

1. **Fix and finish `wf_lincomb_mul_and`** (and any sibling wf lemmas).
2. **Add / prove `l1_conv_submultiplicative`** in `Cln_finite_l1_submultiplicativity.v`.
3. **Prove `boolish_k_le_conv`** in `Cln_BoolDist.v` using:

   * witness construction + wf lemma
   * error split + submultiplicativity
4. **Lift to trace:** `trace_boolish_k_le` closure for `Conv` in `Cln_CompositeExcursion.v`.
5. Only then start formalizing the “twist costs k” boundary lemmas.

---

# Immediate patch for your current blockers

* The `n` inference issue: make `n : nat` explicit OR remove it if unused.
* Don’t try to use `==` notation or `Qeq_eq`; just stay in the equality notion your library actually supports.

---

If you paste the definitions of `wf_lincomb` and `boolish_k_le` (the exact sigma/witness fields) from your files, I’ll write the *exact* proof scripts for:

* `wf_lincomb_mul_and` (fully)
* the skeleton of `boolish_k_le_conv` (with the right destruct/exists structure matching your witness type)

so you can drop them in with minimal edits.
