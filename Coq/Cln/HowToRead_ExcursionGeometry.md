# How to read the Math

## Core objects (the “things” you compute with)

### `n : nat`

* **Meaning:** the dimension / number of Boolean variables.
* Math: we’re working over the group ((\mathbb Z_2)^n).
* Intuition: `n=4` means masks are 4-bit strings like `0101`.

### `Mask n` (sometimes you call it a “mask”, “blade index”, or “basis element”)

* **Meaning:** an element of ((\mathbb Z_2)^n), represented as an `n`-bit vector / subset of `{1..n}`.
* Intuition: “which basis vectors are multiplied together” in geometric algebra.

### `Corner n -> bool`

* **Meaning:** a Boolean function on `n` bits.
* Often `Corner n` is your “Boolean input” type (another view of an `n`-bit vector).
* Intuition: `f : Corner n -> bool` is the target function you want to compute.

### `MV n`

* **Meaning:** a **multivector** in dimension `n`.
* Concretely: a function assigning a rational coefficient to every mask:
  [
  F : \text{Mask}_n \to \mathbb{Q}.
  ]
* You can think of it as a “sparse/dense table” indexed by bitmasks.

### `F, G, X : MV n`

* Generic multivectors.
* Typical roles:

  * `F`, `G`: inputs to a product (`mv_conv F G` or `mv_gp F G`)
  * `X`: a “current state” / intermediate value

### `sq : Vector.t Q n`

* **Meaning:** your “signature” / parameters that decide the GP’s sign rules (metric / cocycle knobs).
* Intuition: it’s what makes GP be “twisted” compared to convolution.

---

## Basis / algebra structure (how masks interact)

### `mask_xor A B`

* **Meaning:** XOR of two masks = group operation in ((\mathbb Z_2)^n).
* Math: (A \oplus B).
* Intuition: combine subsets modulo 2 (toggle membership).

### `all_masks n`

* **Meaning:** list of all masks in dimension `n`.
* Used to define sums/products by iterating over the basis.

### `mask_eq_dec`

* **Meaning:** decidable equality on masks (lets Coq branch on `A = B`).

---

## Two products: convolution vs geometric product

### `mv_conv F G : MV n`

* **Meaning:** **convolution** on the group algebra of ((\mathbb Z_2)^n).
* Coefficient formula:
  [
  (F * G)(U) = \sum_{A \oplus B = U} F(A),G(B).
  ]
* In your code snippet, the `if mask_xor A B = U` is implementing that constraint.
* Intuition: “clean, commutative, no signs.”

### `mv_gp sq F G : MV n` (your GP)

* **Meaning:** **twisted product** (geometric product) that is convolution **plus a sign/phase factor** from a 2-cocycle (and `sq`).
* Shape:
  [
  (F \star G)(U) = \sum_{A \oplus B = U} \sigma_{sq}(A,B), F(A),G(B),
  ]
  where (\sigma_{sq}(A,B)\in{\pm 1}) (in your rational setting: `1` or `-1`).
* Intuition: “like conv, but with interference signs; causes grade-mixing and cancellations.”

### Notation you use informally:

* `⋆` = GP (`mv_gp`)
* `*` or `Conv` = convolution (`mv_conv`)
* “commutator”:
  [
  [B,X] := B\star X - X\star B.
  ]

---

## Size / complexity measures

### `l1` / `|F|_1`

* **Meaning:** (\ell_1)-mass of a multivector:
  [
  |F|*1 := \sum*{U} |F(U)|.
  ]
* Intuition: “total absolute weight over all basis masks.”

### `grade`

* **Meaning:** the Hamming weight of a mask: number of 1-bits.
* For a multivector, “max grade present” means:
  [
  \max{ |U| : F(U)\neq 0}.
  ]
* Intuition: “how many basis vectors multiply together in a term.”

### `excursion`

* **Meaning:** a measure over a *trace* (sequence of intermediate multivectors).
* Your project uses variants like:

  * **grade excursion:** maximum grade seen during the run
  * **ℓ₁ excursion:** maximum (\ell_1) mass seen during the run
* Intuition: “peak complexity during evaluation.”

---

## Traces and evaluation

### `GA_expr n` / `e`

* **Meaning:** an expression/program tree built from operations (conv, gp, additions, constants, variables, etc.) in dimension `n`.

### `eval_expr sq e`

* **Meaning:** evaluate expression `e` under signature `sq`, producing a multivector (and often a trace).

### “trace”

* **Meaning:** list of intermediate states (F_0, F_1, \dots) encountered when evaluating `e`.
* Used to enforce “stays close to easy space” conditions.

### `computes sq e (f n)`

* **Meaning:** correctness statement: evaluating `e` (under `sq`) implements Boolean function `f n` under your embedding/evaluation map.

---

## Projection / rejection decomposition

This is the “A ⊕ B” story.

### (A) (easy subalgebra) and (B) (hard/complement part)

* **Meaning:** you pick a “nice” subspace (A\subseteq MV_n) that you hope is stable under “easy” operations.
* Then everything outside it is “leakage” into (B).

### `π_A` (projection)

* **Meaning:** linear map (MV_n \to MV_n) that keeps only the part in (A).
* Math property: (\pi_A(\pi_A(X))=\pi_A(X)).

### `ρ_B := id - π_A` (rejection / residual)

* **Meaning:** throws away the (A)-part; keeps the “leakage” part.
* So (X = \pi_A(X) + \rho_B(X)).

### Common concrete choices in your setting

1. **grade cutoff projection**

   * (A): multivectors supported on grades (\le r)
   * (\rho): “high-grade part”
2. **centralizer / commuting projection**

   * Fix a blade (B)
   * (A := {X : B\star X = X\star B})
   * (\rho) measures non-commutation leakage (commutator-related)
3. **Fourier/mask-subset projection**

   * (A): only masks in some subset (S\subseteq \text{Mask}_n)
4. **boolish shadow** (not linear)

   * (A): all (k)-term combinations of embedded Boolean generators
   * “projection” is “best approximation” (distance-to-(A) potential)

---

## Your “boolish” names (how they encode “close to easy”)

### `boolish_k_le F k d`

* **Meaning:** “(F) is within (\ell_1) distance (d) of some (k)-simple Boolean-ish combination.”
* In the proof-carrying form you showed:

  * `cs`: coefficients (rationals)
  * `gs`: generators (the actual Boolean-ish basis multivectors)
  * `Hlen`: length bound `<= k`
  * `Hd`: distance bound `<= d`
* Plain English: “there exists a simple shadow built from ≤ k pieces that approximates F.”

### `trace_boolish_k_le sq e k d`

* **Meaning:** every intermediate state in evaluating `e` is `boolish_k_le` with the same `k` and error `d`.

### `trace_boolish_global_le sq e gs d`

* **Meaning:** like above but the generator set `gs` is fixed globally (stronger uniformity).

### `BoolDist` / “distance”

* **Meaning:** your structured distance notion between a multivector and a “Boolean shadow” family.
* Intuition: like (\ell_1) distance but specialized to your embedding.

---

## The projector gadget: `p = (1 - B)/2`

### `B`

* **Meaning:** usually a **unit blade** (basis element) with (B\star B = 1) (or ±1 depending on signature).
* Intuition: “a symmetry you test commutation against.”

### `p := (1 - B)/2`

* **Meaning:** an idempotent-like “projector” built from `B`.
* Used to convert commutators into differences of left/right multiplication.

### Identity you wrote:

[
X\star p - p\star X = \tfrac12(B\star X - X\star B)=\tfrac12[B,X].
]

* **Meaning:** the “noncommuting part” of (X) with respect to (B) is exactly what shows up when you swap the order with `p`.
* Intuition: if (X) commutes with (B), then it also “doesn’t care” about left vs right multiplication by `p`.

---

## Error bound lemma shapes (what the names usually mean)

### `conv_error_bound_l1` (conceptually)

* **Meaning:** convolution is stable: replacing inputs by approximations only changes output proportionally.
* Typical form:
  [
  |F*G - F'*G'|_1 \le |F-F'|_1|G|_1 + |F'|_1|G-G'|_1
  ]
  (or something close, depending on how you state it).

### `gp_error_bound_l1` (conceptually)

* **Meaning:** same style bound for GP, but the twist/grade-mixing matters in later “leakage” arguments.

### `l1_gp_submultiplicative`

* **Meaning:** (\ell_1) norm doesn’t blow up under GP beyond multiplication:
  [
  |F\star G|_1 \le |F|_1 |G|_1.
  ]
* That’s one of your core reusable “tame upper bound” tools.

---

## Tiny operator glossary (so she can decode formulas fast)

* `∀` / `forall` = “for all”
* `∃` / `exists` = “there exists”
* `->` = implication / function arrow
* `:=` = definition
* `+` / `-` on `MV` = pointwise addition/subtraction of coefficients
* `0` / `1` (in MV context) = zero multivector / scalar unit multivector
* `sumQ` = summation in rationals
* `Q` = rational numbers
* `Qabs` / `|q|` = absolute value of a rational
* `<=` with `%nat` vs `%Q` = be careful: nat inequality vs rational inequality

---

## One “wife-friendly” sentence tying it all together

* A multivector `F : MV n` is just a **dictionary from bitmasks to rational weights**.
* `mv_conv` combines masks by XOR with **no signs**.
* `mv_gp` does the same XOR-combine but injects **sign flips** from geometry (`sq`), causing cancellations and grade mixing.
* “Projection/rejection” is picking an “easy” subset of multivectors and tracking how much the computation **leaks** outside it.

---

Yep — I glossed over some of the “math punctuation” and also didn’t spell out the “primed variables” / “Pi” story as cleanly as your original prompt deserves. Here’s the missing glue.

---

## The symbols you called out

### `∥ · ∥` (double bars)

That’s a **norm**: a way to turn an object (here, a multivector) into a nonnegative number measuring its “size”.

In your setting, the one you use constantly is the **ℓ₁ norm**:

[
|F|*1 ;:=; \sum*{U \in \text{Mask}_n} |F(U)|.
]

* `F(U)` is the rational coefficient of basis mask `U`.
* `|F(U)|` is rational absolute value (`Qabs`).
* Intuition: “total absolute mass” of the multivector.

So whenever you see `∥F∥₁`, read: **sum of absolute coefficients**.

---

### “difference” like `F - F'`

This is just **subtraction of multivectors** (pointwise on coefficients):

[
(F - F')(U) ;:=; F(U) - F'(U).
]

So `F - F'` is a new multivector measuring the **coefficient-wise error** between them.

And `∥F - F'∥₁` is the **distance** between them in ℓ₁:

[
|F - F'|_1 = \sum_U |F(U)-F'(U)|.
]

Intuition: “how far did I perturb the state?”

---

### What’s the point of `F` vs `F'`?

The prime (`'`) means **an approximation / projected version / simplified shadow** of the same thing.

Common patterns:

1. **Projection to an easy space**

   * `F` = actual multivector during computation
   * `F' = π_A(F)` = the part of `F` that lies in the “easy subalgebra” (A)

2. **Boolish shadow approximation** (not necessarily linear)

   * `F` = actual state
   * `F'` = “best k-term boolish approximation” to `F`

Either way, the story is:

* `F` is the true state the GP computation produces.
* `F'` is the “easy” surrogate you wish it stayed near.
* `∥F - F'∥₁` measures how badly the computation has “escaped.”

---

### `π` / `Pi` (projection)

In your prompt, `π_A : MV_n → MV_n` is a **projection operator onto the chosen easy subspace** (A).

Projection means:

1. **It lands in (A)**
   [
   \pi_A(X) \in A
   ]

2. **It doesn’t change things already in (A)** (idempotent)
   [
   \pi_A(\pi_A(X)) = \pi_A(X)
   ]

3. **It is linear** (in the “linear projection” versions)
   [
   \pi_A(X+Y)=\pi_A(X)+\pi_A(Y),\quad \pi_A(cX)=c,\pi_A(X).
   ]

In plain terms: **“keep only the easy part, throw away the rest.”**

---

### `ρ` / `rho` (rejection / residual)

You defined it as:

[
\rho_B := \mathrm{id} - \pi_A.
]

That means:

[
\rho_B(X) = X - \pi_A(X).
]

So:

* `π_A(X)` = the part you keep (“easy”).
* `ρ_B(X)` = what’s left over (“leakage”).

And the decomposition is literally:

[
X = \pi_A(X) + \rho_B(X).
]

This is the “projection/rejection” split your prompt is built around.

---

## “Invariant + monotone potential” in the exact sense you mean

### Invariant (what stays true for “easy” computations)

A typical invariant is something like:

[
\forall t,;;|\rho_B(F_t)|_1 \le d.
]

Meaning: **at every step of the trace**, the leakage into (B) stays bounded by `d`.

* `F_t` = the multivector at time/step `t` in the trace.
* `d` = allowed leakage budget.

This is exactly your “trace stays in A (or close to it)” condition.

### Potential (a scalar that measures how far you’ve escaped)

The most direct potential is:

[
\Phi(F) := |\rho_B(F)|_1.
]

Then the invariant above is simply: `Φ(F_t) ≤ d` for all t.

“Monotone” comes in when you show that under certain operations (or for certain hard inputs) this potential **must increase** or **cannot stay small**.

---

## The closure/error-bound lemmas: what each symbol means

When you say:

> If (F,G) are close to A, then Conv(F,G) is close to A with controlled error.

That becomes a statement like:

Let `F' = π_A(F)` and `G' = π_A(G)` (or any approximations in A).
Then

[
|\mathrm{Conv}(F,G) - \mathrm{Conv}(F',G')|_1
]

is bounded by something involving the input errors:

[
\le |F-F'|_1\cdot |G|_1 ;+; |F'|_1\cdot |G-G'|_1.
]

What each piece means:

* `Conv(F,G)` = true product.
* `Conv(F',G')` = “easy-space product” after projecting inputs.
* `∥F-F'∥₁`, `∥G-G'∥₁` = how far inputs are from easy space.
* `∥F∥₁`, `∥F'∥₁`, `∥G∥₁` = sizes (needed because products scale errors).

Same template for GP:

[
|\mathrm{GP}(F,G)-\mathrm{GP}(F',G')|_1 \le \cdots
]

These are “stability bounds”: **small input error ⇒ controlled output error**.

---

## The commutator gadget: what’s being measured

You wrote:

[
X\star p - p\star X = \tfrac12(B\star X - X\star B)
]

Let’s name each component:

* `⋆` = GP (`mv_gp sq`)
* `B` = chosen blade you use to define “commuting space”
* `p = (1-B)/2` = projector built from `B`
* `[B,X] := B⋆X - X⋆B` = commutator (measures non-commutation)

So the identity says:

**“Left-vs-right multiplication by the projector extracts exactly the commutator component (up to 1/2).”**

Then in ℓ₁ terms:

[
|X\star p - p\star X|_1 = \tfrac12 |[B,X]|_1.
]

Meaning:

* If `X` commutes with `B`, commutator is 0, and left/right multiplication agree.
* If `X` does not commute, the difference is measurable as ℓ₁ leakage.

This is why “centralizer projection” is so aligned with GP hardness: the rejected part is literally tied to commutators.

---

## What “A is stable under allowed operations” really means

In your prompt you said “A should be closed (or almost closed).” Concretely:

### Exact closure

[
F\in A,; G\in A \implies \mathrm{Op}(F,G)\in A.
]

### Approximate closure (what you actually use)

If `F` and `G` are close to `A`, then `Op(F,G)` is also close to `A` with a bound.

This is exactly why you keep writing things like:

* “error bound lemma” (replace by projection and pay ∥F−F'∥₁)
* “submultiplicativity” (control how norms blow up)

---

## Filling the “Pi in this context” gap with your four concrete choices

Here’s what `π_A` “actually is” in each option from your original prompt:

1. **Grade cutoff**

   * (A) = multivectors supported only on masks with `popcount(U) ≤ r`.
   * (\pi_A(F)(U)=F(U)) if grade(U)≤r else 0.
   * Linear, simple.

2. **Mask subset / Fourier subset**

   * (A) = support restricted to some subset `S` of masks.
   * (\pi_A(F)(U)=F(U)) if U∈S else 0.
   * Linear.

3. **Centralizer / commuting space**

   * (A={X:[B,X]=0}).
   * `π_A` is trickier: often defined via an averaging operator (project onto fixed points of conjugation) or via eigenspace projectors built from `(1±B)/2`.
   * Still linear, but not “just zeroing coefficients.”

4. **Boolish shadow**

   * Not linear. There isn’t a literal `π_A`; instead you have:

     * “there exists an approximation in A with ≤k terms”
   * Potential becomes: distance-to-best-approximation.

---

## 1) What `∥F∥₁` literally means in your code (and what `∥ · ∥` means)

In `Cln_finite_l1_submultiplicativity.v` you define:

```coq
Definition l1_norm {n} (F : MV n) : Q :=
  sumQ (map (fun U => Qabs (F U)) (all_masks n)).

Notation "∥ F ∥₁" := (l1_norm F) (at level 40).
```

So:

[
|F|*1 ;=; \sum*{U \in \texttt{all_masks n}} |F(U)|.
]

Concrete decode:

* `F : MV n` means `F` is a function `Mask n -> Q` (see below).
* `all_masks n` enumerates every mask (every basis blade index).
* `Qabs (F U)` is absolute value of the rational coefficient at `U`.
* `sumQ` is the finite sum over a list.

So **double bars `∥ ∥`** are not a generic norm here: they are explicitly **ℓ₁ mass of coefficients**.

---

## 2) What `F`, `F'`, and `F - F'` mean (and why primes show up)

### `MV n` and pointwise operations

From `Cln_Full.v`:

```coq
Definition MV (n : nat) : Type := Mask n -> Q.

Definition mv_sub {n} (F G : MV n) : MV n := fun m => (F m - G m)%Q.
Definition mv_add  {n} (F G : MV n) : MV n := fun m => (F m + G m)%Q.
```

So `F - F'` (mathematically) is implemented as:

[
(F - F')(m) := F(m) - F'(m).
]

Then the *distance* / *error size* is:

[
|F - F'|*1
= \sum*{U} |F(U) - F'(U)|.
]

### What the prime means in your proof patterns

In your original prompt, `F'` is “the easy version” of `F`. There are two common meanings:

1. **Linear projection version:**
   `F' := π_A(F)` (projection onto some easy subspace/subalgebra (A)).
   Then `F - F' = ρ_B(F)` is the rejected/leakage part.

2. **Witness approximation version (BoolDist / boolish):**
   `F'` is some “shadow” object you *existentially* get from a witness, like an `embed g` or a `k`-term combination.
   Then `∥F - F'∥₁` is the distance-to-shadow.

Your current BoolDist file is explicitly the second style (witness-based), e.g.:

```coq
Definition bool_dist_wrt {n} (F : MV n) (g : Corner n -> bool) : Q :=
  l1_norm (mv_sub F (embed g)).
```

So there: the “primed thing” is literally `embed g`.

---

## 3) What `π_A` and `ρ_B` mean *in this context* (even if not yet a Coq `Definition`)

In your original prompt you wrote:

* (MV_n = A \oplus B)
* (\pi_A : MV_n \to MV_n) projection onto (A)
* (\rho_B := \mathrm{id} - \pi_A) rejection into (B)

### What “id” is

`\mathrm{id}` just means the identity map:

[
\mathrm{id}(X) = X.
]

### What `π_A` means here

It’s any operator that satisfies the projection behavior:

* **Keeps the easy part:** (\pi_A(X)\in A)
* **Idempotent:** (\pi_A(\pi_A(X))=\pi_A(X))
* Usually linear (for the “linear A ⊕ B” story)

In CLN terms, you’ll instantiate `π_A` in one of these concrete ways:

* **Grade cutoff projection**: keep only masks of grade ≤ r (a literal coefficient filter).
* **Support subset projection**: keep only masks in some subset (S) (also literal filter).
* **Centralizer/commuting projection**: project onto the subspace commuting with a blade (B) (often via averaging or projector tricks).
* **Boolish projection**: not linear—implemented as an existential witness and a distance bound.

### What `ρ_B` means here

Once you choose π, the rejection is always:

[
\rho_B(X) := X - \pi_A(X).
]

and the potential you track is:

[
\Phi(X) := |\rho_B(X)|_1.
]

So “leakage into the hard part” is literally “how big is the rejected component in ℓ₁”.

---

## 4) The `==` vs `=` thing (important for your wife reading proofs)

You’re using Coq’s rational setoid equality:

* `x == y` is **`Qeq`** (equality of rationals in QArith)
* `x = y` is Coq’s **definitional/propositional equality**

So in your files, many lemmas state equalities using `==` and then convert to inequalities using your helper:

```coq
Lemma Qle_of_Qeq : forall x y : Q, x == y -> x <= y.
```

So when you see:

* `... == ...` think “equal as rationals”
* `... <= ...` think “Q-order”

---

## 5) What BoolDist is saying (and what each symbol means)

From `Cln_BoolDist.v`:

```coq
Definition bool_dist_le {n} (F : MV n) (d : Q) : Prop :=
  exists g : Corner n -> bool,
    l1_norm (mv_sub F (embed g)) <= d.
```

Decode:

* `F : MV n` = current multivector
* `d : Q` = allowed error budget
* `exists g : Corner n -> bool` = there is a Boolean function witness `g`
* `embed g : MV n` = the multivector representing that Boolean function
* `l1_norm (F - embed g) <= d` = F is within ℓ₁ distance `d` of a *true embedded Boolean*

So BoolDist is **distance-to-the-Boolean-embedded-subset** in ℓ₁.

And the witness-relative version:

```coq
Definition bool_dist_wrt {n} (F : MV n) (g : Corner n -> bool) : Q :=
  l1_norm (mv_sub F (embed g)).
```

So `bool_dist_wrt F g` is literally “the ℓ₁ error if we claim F is approximating embed(g)”.

---

## 6) How “error bounds” use `F` vs `F'` in your actual files

Your files implement the standard trick:

* rewrite a complicated difference into sums of simpler differences
* use triangle inequality bounds like `l1_add_bound`

Example from `Cln_BoolDist.v` (addition absorb):

```coq
bool_dist_le (mv_add F G) (dF + l1_norm G)
```

Interpretation:

* If `F` is close to `embed gF` (distance ≤ dF),
* then `F + G` is close to the *same* `embed gF`,
* but you pay an extra `∥G∥₁` because adding `G` perturbs you by at most its ℓ₁ mass.

That’s exactly the “difference between F and F′” story:

* `F′` is `embed gF`
* the change `+G` is bounded by `∥G∥₁`

---

## 7) Quick “operator cheat sheet” (the missing math punctuation)

Here are the ones you explicitly mentioned + the ones that tend to confuse helpers:

* `∥F∥₁` = `l1_norm F` = `sum_U |F U|`
* `F - F'` = `mv_sub F F'` = coefficientwise subtraction
* `π_A(F)` = “project F onto the easy subspace A” (you pick what A is)
* `ρ_B(F)` = `F - π_A(F)` = rejected/leakage component
* `id` = identity function (`fun x => x`)
* `A ⊕ B` (math) = direct sum decomposition (unique split into easy + hard)
* `[B, X]` (commutator) = `B⋆X - X⋆B` (GP order-sensitivity measure)

---

## 8) The one-liner that ties “projection/rejection” to your existing BoolDist machinery

Your current BoolDist setup is already a projection/rejection story, just **witness-based** instead of linear:

* “projection” = pick a witness `g` and use `embed g` as the approximant (that’s your “easy model”)
* “rejection magnitude” = `∥F - embed g∥₁`
* “potential” = the smallest such distance over all `g` (if you later define an infimum-style version), or the existence bound `<= d` you use now.

So when you write `F'` in the prose, in today’s Coq it usually means:
**`F' = embed g`** (or `F'` is the boolish k-term combination you witness).

---

## 0) The “legend” for types and notation

### Core types

* `n : nat`
  number of bits / dimension.

* `Mask n`
  an `n`-bit mask (basis index).

* `MV n := Mask n -> Q` (from `Cln_Full.v`)
  a multivector is literally a function “mask ↦ rational coefficient”.

* `Q`
  rationals.

### Pointwise ops on multivectors (also in `Cln_Full.v`)

* `mv_add F G : MV n := fun m => F m + G m`
* `mv_sub F G : MV n := fun m => F m - G m`

So “addition/subtraction of multivectors” is **coefficientwise**.

### ℓ₁ norm (from `Cln_finite_l1_submultiplicativity.v`)

```coq
Definition l1_norm {n} (F : MV n) : Q :=
  sumQ (map (fun U => Qabs (F U)) (all_masks n)).
Notation "∥ F ∥₁" := (l1_norm F).
```

So:
[
|F|*1 = \sum*{U \in \texttt{all_masks n}} |F(U)|.
]

### Equality vs equality

* `==` is `Qeq` (rational equality, setoid)
* `=` is Coq propositional equality (you usually avoid it for `Q`)

---

## 1) `l1_add_bound`: triangle inequality in MV form

**Statement** (in `Cln_finite_l1_submultiplicativity.v`):

```coq
Lemma l1_add_bound : forall n (F G : MV n),
  l1_norm (mv_add F G) <= l1_norm F + l1_norm G.
```

### What each symbol means

* `F, G : MV n`: two multivectors (two coefficient tables).
* `mv_add F G`: coefficientwise addition: `(F+G)(U)=F(U)+G(U)`.
* `<=` is rational order on `Q`.
* `l1_norm`: sum of absolute values of coefficients.

### English meaning

> “If you add two multivectors, the total absolute mass is at most the sum of their masses.”

This is the standard triangle inequality:
[
|F+G|_1 \le |F|_1 + |G|_1.
]

This lemma is *the* glue behind almost every “errors add” argument in your BoolDist file.

---

## 2) `l1_conv_bound`: convolution is ℓ₁-submultiplicative

**Statement** (in `Cln_finite_l1_submultiplicativity.v`):

```coq
Lemma l1_conv_bound :
  forall n (F G : MV n),
    l1_norm (mv_conv F G) <= l1_norm F * l1_norm G.
```

### What the variables mean

* `mv_conv F G` is your untwisted convolution product (group algebra):
  [
  (F * G)(U) = \sum_{A \oplus B = U} F(A)G(B).
  ]

### English meaning

> “Convolution doesn’t amplify ℓ₁ mass by more than multiplicatively.”

Formally:
[
|F * G|_1 \le |F|_1\cdot|G|_1.
]

This is the “conv is tame” upper bound.

---

## 3) `l1_gp_submultiplicative`: GP is also ℓ₁-submultiplicative (under your unit-metric hypothesis)

**Statement** (same file, inside `Section UnitMetric`):

```coq
Theorem l1_gp_submultiplicative :
  forall (F G : MV n),
    l1_norm (mv_gp sq F G) <= l1_norm F * l1_norm G.
```

### Extra variables here

* `sq : Vector.t Q n` is your signature vector.
* Hypothesis in that section:

  ```coq
  Hypothesis sq_unit : forall (i : Fin.t n),
    Qabs (Vector.nth sq i) == 1.
  ```

  Meaning each `sq[i]` has absolute value 1 (so coefficients are ±1-ish).

### English meaning

> “Even with the GP’s sign twisting, ℓ₁ mass is still submultiplicative.”

[
|F \star G|_1 \le |F|_1 \cdot |G|_1.
]

This is your key “GP doesn’t explode ℓ₁ *by itself*” lemma.

---

## 4) BoolDist: what `bool_dist_le` and `bool_dist_wrt` literally mean

From `Cln_BoolDist.v`:

```coq
Definition bool_dist_le {n} (F : MV n) (d : Q) : Prop :=
  exists g : Corner n -> bool,
    l1_norm (mv_sub F (embed g)) <= d.

Definition bool_dist_wrt {n} (F : MV n) (g : Corner n -> bool) : Q :=
  l1_norm (mv_sub F (embed g)).
```

### Decode

* `embed g : MV n` is your “multivector representing Boolean function g”.
* `mv_sub F (embed g)` is “the error multivector” (coeffwise difference).
* `l1_norm` makes that error into a number.

So:

* `bool_dist_wrt F g` = “how far is F from embed(g) in ℓ₁”
* `bool_dist_le F d` = “there exists *some* Boolean g such that F is within d of embed(g)”

This is exactly the “F vs F′” story:

* your **approximation** is `F' := embed g`
* your “distance” is `∥F - F'∥₁`

---

## 5) “Absorb” lemmas: adding garbage costs at most its ℓ₁

### `bool_dist_add_absorb`

```coq
Lemma bool_dist_add_absorb : forall n (F G : MV n) (dF : Q),
  bool_dist_le F dF ->
  bool_dist_le (mv_add F G) (dF + l1_norm G).
```

**English meaning**

> If F is close to some Boolean embedding, then F+G is still close to the *same* embedding; you just pay an extra error equal to ∥G∥₁.

Why? Because:
[
(F+G)-\mathrm{embed}(g) = (F-\mathrm{embed}(g)) + G
]
then use triangle inequality:
[
|(F-\mathrm{embed}(g)) + G|_1 \le |F-\mathrm{embed}(g)|_1 + |G|_1.
]

### `_r` version

Same idea but absorbing on the other side.

These are the basic “error bookkeeping” lemmas you’ll reuse constantly in trace arguments.

---

## 6) `bool_dist_wrt_add`: “errors add” when you know both targets

```coq
Lemma bool_dist_wrt_add : forall n (F G : MV n) (gF gG : Corner n -> bool),
  l1_norm (mv_sub (mv_add F G) (mv_add (embed gF) (embed gG)))
  <= bool_dist_wrt F gF + bool_dist_wrt G gG.
```

### English meaning

> If F is trying to approximate embed(gF) and G is trying to approximate embed(gG), then F+G approximates embed(gF)+embed(gG), and the total error is at most the sum of the two individual errors.

This is the exact algebra:
[
(F+G) - (\mathrm{embed}(g_F)+\mathrm{embed}(g_G))
= (F-\mathrm{embed}(g_F)) + (G-\mathrm{embed}(g_G))
]
then triangle inequality.

This lemma is the “clean, symmetric” add-error lemma (as opposed to the one-sided absorb versions).

---

## 7) `gp_error_split`: the GP error decomposition you were calling “gp_error_bound_l1 core lemma”

This one is the *heart* of your GP error bound machinery:

```coq
Lemma gp_error_split : forall n (sq : Vector.t Q n) (F G eF eG : MV n) (m : Mask n),
  mv_sub (mv_gp sq F G) (mv_gp sq eF eG) m
  == mv_add (mv_gp sq F (mv_sub G eG)) (mv_gp sq (mv_sub F eF) eG) m.
```

### What each variable means

* `F, G : MV n` are the “true” inputs.
* `eF, eG : MV n` are “easy/embedded/approx” versions (the “primed” objects in prose).
* `mv_gp sq F G` is the GP product (F\star G).
* `m : Mask n` is a single coefficient index (you’re stating equality *pointwise*).

### What the statement says in math

Pointwise at coefficient `m`:
[
(F\star G - eF\star eG)(m)
= \big(F\star (G-eG)\big)(m) ;+; \big((F-eF)\star eG\big)(m).
]

### Why this is exactly “the” error split

It’s the bilinear identity:
[
F\star G - eF\star eG
= F\star(G-eG) + (F-eF)\star eG.
]

So it tells you: **output error = (right-input error term) + (left-input error term)**.

This is the GP version of the standard “add and subtract the middle term” trick:
[
FG - F'G' = F(G-G') + (F-F')G'.
]

That’s precisely the “difference between F and F′” gap you asked about.

---

## 8) `bool_dist_wrt_gp`: the ℓ₁ inequality that drops out of `gp_error_split` + submultiplicativity

```coq
Lemma bool_dist_wrt_gp : forall n (sq : Vector.t Q n) (F G : MV n)
  (gF gG : Corner n -> bool),
  (forall i, Qabs (Vector.nth sq i) == 1) ->
  l1_norm (mv_sub (mv_gp sq F G) (mv_gp sq (embed gF) (embed gG)))
  <= l1_norm F * bool_dist_wrt G gG
   + bool_dist_wrt F gF * l1_norm (embed gG).
```

### Variables

* `F, G : MV n` true inputs.
* `gF, gG : Corner n -> bool` targets you want to approximate.
* `embed gF`, `embed gG` are the approximations `eF`, `eG`.
* `bool_dist_wrt F gF = ∥F - embed gF∥₁`.
* The hypothesis `(forall i, |sq[i]| = 1)` enables `l1_gp_submultiplicative`.

### Translate the inequality into “wife English”

> The GP error between `F⋆G` and `embed(gF)⋆embed(gG)` is controlled by two contributions:
>
> 1. if G is inaccurate (far from embed(gG)), that error gets multiplied by the size of F
> 2. if F is inaccurate, that error gets multiplied by the size of the target right factor embed(gG)

Mathematically it’s exactly:
[
|F\star G - eF\star eG|_1
\le |F|_1\cdot|G-eG|_1 ;+; |F-eF|_1\cdot|eG|_1.
]

Where:

* (eF=\mathrm{embed}(g_F))
* (eG=\mathrm{embed}(g_G))

### What operators are doing the work

This lemma is literally:

1. **error split** (`gp_error_split`)
2. **triangle inequality** (`l1_add_bound`)
3. **submultiplicativity** (`l1_gp_submultiplicative`) applied to each term.

---

## The quick “cheat translation” for your guide

When your wife sees something like:

[
|, \text{(real output)} - \text{(easy output)},|_1 \le \cdots
]

she should read:

* Left side: “how far the computation drifted away from the easy model”
* Right side: “two ways it can drift: right input was off, or left input was off — each amplified by the size of the other factor”

And when she sees a prime in prose (`F'`), in your current Coq it’s usually one of:

* `embed g` (Boolean witness)
* or a `mv_sub` / `mv_add`-constructed approximation.

---
