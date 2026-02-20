# Roadmap (updated): “explosion ⇒ separation” pipeline with current files (all compile)

This is the same pipeline as before, rewritten to match the **current codebase layout** and the fact that these files now all compile cleanly:

- `TraceGeometryCore.v` — axiomatic algebra + basic structure (`A`, `Sem`, `+`, `⊙`, `⋆`, `zero`, etc.)
- `TraceGeometryExplosion.v` — `Split` / parts / `norm1` interface and the **local growth-under-⋆** theorem (`split_growth_under_gp`, etc.)
- `TraceGeometryMachine.v` — stack machine (programs, instructions, traces, `obs`, `output`, `cost`)
- `TraceGeometryPipeline.v` — high-level “glue” pipeline modules (target/spec/correctness bridge + lower-bound skeleton)
- `TraceGeometryStep5Strong.v` — **Step 5 strongest variant**: effective-gp counting + `runs_eff_obs_explosion`
- `TraceGeometryToyNat.v` — toy instance / sanity playground (helpful for testing which axioms are really needed)

The goal remains:

> grade + booldist + ℓ1 + “min over programs”
→ any program computing the target must realize exponential Split explosion
→ cost lower bound / separation.

---

## 0) Pick the target statement precisely (goes into `TraceGeometryPipeline.v` instantiation)
You now have a clean slot for this in the pipeline layer:

- Choose a target family `T : nat -> A`.
- Choose a machine start state `st0 : nat -> SM.Stack`.
- Choose a semantic spec `spec : nat -> Input -> Sem`.
- Define `Correct n p := ∀s, eval_at s (output (st0 n) p) = spec n s`.

Status:
- The *structure* for this is implemented in `TraceGeometryPipeline.v` via `TRACE_GEOMETRY_TARGET`.

Deliverable (next):
- One concrete `Target` module for your intended function family (XOR/parity/etc.).

---

## 1) Evaluation bridge: semantic correctness ⇒ equality in `A`
This is already factored cleanly into the pipeline:

- `TraceGeometryExplosion.v` / core provides the eval-family hook (your `TRACE_GEOMETRY_EVAL_INJ` layer).
- `TraceGeometryPipeline.v` provides `TraceGeometryCorrectnessBridge.correct_output_eq_T`:
  correctness over all inputs
  → `output = T n`
  (via `eval_at_injective`).

Status:
- The lemma exists and is used as the canonical “semantic ⇒ algebraic equality” bridge.

Deliverable (next):
- Instantiate `Input` / `eval_at` for the model you care about (or keep axiomatic for now).

---

## 2) Programs in the game: traces + cost (`TraceGeometryMachine.v`)
You have the machine layer compiling:

- Program type `SM.Prog`, instruction set `SM.Instr`.
- State/stack type `SM.Stack`.
- Observation `SM.obs : Stack -> A`.
- `SM.output st0 p` and `SM.cost p`.

Status:
- Machine is present and usable.
- The “trace discipline” for Step 5 is now handled by `runs_eff` in `TraceGeometryStep5Strong.v` (a trace-producing run relation that mirrors the machine steps you care about).

Deliverable (next):
- A small adapter lemma connecting the machine’s own execution notion (if you have `SM.runs`) to `runs_eff` (or use `runs_eff` as the canonical run semantics for the LB proof).

---

## 3) Define the scalable explosion invariant (Split/Parts)
You’ve standardized on:

- Predicate form: `HasSplitAtLeast x m := ∃S:Split x, length(parts S) ≥ m`.

Status:
- Implemented in `TraceGeometryStep5Strong.v` as the main invariant for Step 5.
- Local combinatorics sit in `TraceGeometryExplosion.v` (theorems about `Split` and growth under `⋆`).

Deliverable (next):
- The “stability” lemmas for instructions other than effective `⋆`-mix:
  - `istep_preserves_obs_split_lb`
  - `istep_gp_ineff_preserves_obs_split_lb`
  These are currently `Parameter`s in `TraceGeometryStep5Strong.v` (exactly the right place).

---

## 4) Local step lemma: one `⋆` step multiplies parts (`TraceGeometryExplosion.v` → machine step)
You already have the core combinatorial fact in `TraceGeometryExplosion.v`:

- combining splits for `x` and `y` yields a split for `x ⋆ y` with multiplicative parts lower bound.

Status:
- In `TraceGeometryStep5Strong.v` this is lifted into:
  - `effective_gp_step_multiplies` (the one-step “× b” accumulator growth lemma).

Deliverable (next):
- If needed, show the machine’s `IGp` step matches the `⋆` operation you use in the `Split` theorem (usually already aligned).

---

## 5) Global explosion along a run (STRONG variant is DONE)
This is now *fully implemented and proven* in `TraceGeometryStep5Strong.v`:

- `runs_eff b M st p t M' tr`
  tracks:
  - an accumulator lower bound `M` on the current observed/top value,
  - a count `t` of **effective gp-mix** events (the ones with witnesses: top ≥ M and second ≥ b),
  - and a full trace `tr`.

Key deliverables now proven:
- `runs_eff_obs_explosion`:
  if the start has split ≥ `M`, then the final obs has split ≥ `M * b^t`.
- `runs_eff_t_le_cost`:
  effective count `t` is bounded by `length p` (hence bounded by cost once you connect cost≥len).

Status:
- **Step 5 strongest variant completed** (and compiled).

Deliverable (next):
- Connect `t` to `SM.cost p` if your cost is not literally `length p`.

---

## 6) Target-specific forcing lemma: computing `T_n` requires huge Parts
This is still the “heart” step, and it lives naturally as a *target module* you plug into the pipeline:

- a theorem of the form:
  - `output p = T n -> HasSplitAtLeast (output p) (Exp n)` (or a norm1 lower bound),
  derived from your grade + booldist + ℓ1 story.

Status:
- `TraceGeometryPipeline.v` provides the place-holder interface:
  `TRACE_GEOMETRY_TARGET_LOWER_BOUND`.
- You haven’t yet instantiated it with the concrete grade/booldist argument (that’s the next big math step).

Deliverable (next):
- One lemma (even axiomatized first) stating `T n` forces exponential Parts / norm1.

---

## 7) Assemble: huge Parts ⇒ many effective gp ⇒ cost lower bound
Now that Step 5 strong exists, the pipeline assembly in `TraceGeometryPipeline.v` becomes concrete:

- correctness → equality (`CorrectnessBridge`)
- equality → huge Parts (Step 6 lemma)
- Step 5 strong: Parts can only grow as `M * b^t` where `t` counts effective mixes
- algebra → lower bound on `t`
- `t ≤ cost` → cost LB

Status:
- `TraceGeometryPipeline.v` already contains the skeleton theorem
  `cost_lower_bound_from_explosion` with an `Admitted`/placeholder spot for Step 5 accounting.
- With `TraceGeometryStep5Strong.v`, the “accounting lemma” is no longer a black box: you can replace the old axiom with a lemma that extracts `t` from a `runs_eff` run and uses `runs_eff_obs_explosion`.

Deliverable (next):
- Replace the Step-5 axiom in `TraceGeometryPipeline.v` with a lemma that uses `runs_eff_obs_explosion` + `runs_eff_t_le_cost`.

---

## 8) Min over all programs (optional wrapper)
Once you prove:
- `∀p, Computes n p -> cost p ≥ LB n`,
you can define:
- `MinCost n := inf/min over p computing T n of cost p`,
and get `MinCost n ≥ LB n` for free.

Status:
- Not wired yet, but straightforward once Step 7 is finished.

Deliverable (next):
- a small definition + corollary lemma in the pipeline layer.

---

## 9) Models / instantiations (sanity + real)
You already have `TraceGeometryToyNat.v` as a testbed.

Status:
- Toy model compiles.
- Real “Cln” (or stronger) model instantiation remains to be built/plugged into:
  - eval-injectivity,
  - Split stability for non-effective steps,
  - target forcing lemma (Step 6).

Deliverable (next):
- One “sanity” instantiation where Step 6 fails (so no separation),
- then the intended model where Step 6 holds (so separation goes through).

---

# What you have now
A clean decomposition with **Step 5 strong fully mechanized**:

**Correctness (via eval-injectivity)**  
→ `output = T_n`  
→ (Step 6) `HasSplitAtLeast (T_n) Exp(n)`  
→ (Step 5 strong) any run needs `t` effective gp-mixes so that `M*b^t ≥ Exp(n)`  
→ hence `t` is large  
→ `cost ≥ t` gives the lower bound.

The remaining heavy lift is Step 6 + the small adapter lemmas that plug `runs_eff` into the pipeline’s cost theorem.
