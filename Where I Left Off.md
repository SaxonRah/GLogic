# GLogic — where I left off

*Written 31 Aug 2026, after the ℓ₁ Walsh norm approach failed.*

Read the screens (§3) before writing any new code. They would have caught the last
failure on day one.

---

## 1. What the repo actually contains

19,000 lines of Coq across 11 files. Honest ledger:

| File | Qed | Admitted | Axioms | Status |
|---|---|---|---|---|
| `Cl2_BooleanEmbedding.v` | 100 | 0 | 0 | solid |
| `Cln_Full.v` | 111 | 0 | 0 | solid |
| `Cln_SupportAlgebra.v` | 98 | 0 | 0 | solid |
| `Cln_BoolDist.v` | 66 | 0 | 0 | solid |
| `Cln_Grade.v` | 46 | 0 | 0 | solid |
| `Cln_finite_l1_submultiplicativity.v` | 20 | 0 | 0 | solid |
| `Cln_CompositeExcursion.v` | 217 | 26 | 2 params | partly salvageable |
| `Cln_DAG.v` | 20 | **41** | 0 | not load-bearing |
| `Cln_Nonuniform_Separation.v` | 6 | 0 | 2 + hyp | harness only |
| `RepComp_P.v` | 4 | 0 | 5 | harness only |
| `PneqNP_from_dimension.v` | 3 | 0 | 3 + 7 hyps | harness only; won't compile |

**The bottom two-thirds is real, verified, admit-free infrastructure.** The top third
was built to serve the separation and doesn't survive it.

### Two facts to remember about the code

The embedding is the Fourier transform. `Pi` has coefficients
`(1/2ⁿ)·χ_S(α)`, so `ι(F) = Σ_S f̂(S)·e_S`. Grade 0 is `E[f]`, grade 1 are the Chow
parameters, grade 2 are the pairwise Fourier coefficients. `eval` is multilinear
polynomial evaluation, not a Clifford operation. This is Boolean Fourier analysis with
basis-blade labels — which is fine, and is the part worth keeping.

`PneqNP_from_dimension.v` uses `have` (an ssreflect tactic) while importing only Bool,
List, Arith, Lia. It does not compile. There is no `_CoqProject` outside `Cln/`.

### Two errors to fix if the whitepaper is ever revived

- The AND-coincidence success rate is **(n+1)/2n**, not n/(n+1). The data (100%, 75%,
  66.7%, 62.5%) converges to 50%, not 100%.
- Theorem 3's failure has a cause, not just a caveat. Boolean AND is pointwise
  multiplication of indicators = convolution in the Walsh basis, where `χ_S·χ_T =
  χ_{S△T}` with **no sign**. Clifford gives `e_S·e_T = ε(S,T)·e_{S△T}`. In the
  untwisted group algebra ℝ[(ℤ/2)ⁿ], `ι(F)ι(G) = ι(F∧G)` holds exactly, all F, G, all n
  — no projection operator, no NNLS. The cocycle is the only thing breaking it.

---

## 2. Why ℓ₁ Walsh norm died

Recorded so it never gets re-derived from scratch.

**The counterexample.** Inner product `IP = ⊕ᵢ(xᵢ ∧ yᵢ)` has a linear-size circuit
(n ANDs, n XORs) and the *maximum possible* ℓ₁ Walsh norm: all 2ⁿ coefficients equal
±2^(−n/2), giving ‖f̂‖₁ = 2^(n/2), which saturates the Cauchy–Schwarz ceiling
‖f̂‖₁ ≤ 2^(n/2)‖f̂‖₂. The measure is *anti-correlated* with real hardness at the extreme.

**The structural cap.** `l1_gp_submultiplicative` gives ‖xy‖₁ ≤ ‖x‖₁‖y‖₁. An expression
of size s over generators with ‖g‖₁ ≤ C therefore has ‖·‖₁ ≤ C^s. To force ‖f̂‖₁ =
2^(n/2) needs only s ≥ n / (2 log C). **Any submultiplicative measure bounded by
2^(n/2) can never prove better than a linear size lower bound.** This is a ceiling on
the method, not a gap in the proof.

**What was actually right.** The geometry was fine. The embedding was correct, the grade
structure was real, the picture genuinely showed correlation as bivector content. What
failed was the *ruler*, not the space.

---

## 3. The screens — run these first, always

### Screen 1: the IP test
If μ(IP) is large, **discard**. IP has a linear-size circuit, so any measure calling it
hard is not measuring circuit complexity.

Kills at once: spectral norm, discrepancy, approximate rank, γ₂, sign-rank — the entire
communication-complexity family. IP is genuinely hard in those models and trivially easy
for general circuits.

**Generalized version.** Fixed battery, μ must be small on all of them:
IP, parity, majority, mod-3, and a random linear-size circuit. Most candidates die here
within an afternoon.

### Screen 2: the natural proofs test
If μ is efficiently computable **and** large for most Boolean functions, **discard**.
Razborov–Rudich: such a measure breaks pseudorandom function generators.

Fail one condition deliberately — either make μ hard to compute, or make it small on
random functions.

### Screen 3: the ceiling test
If μ is submultiplicative and bounded by M on n-variable functions, it can prove size
lower bounds no better than log_C(M). **Compute the ceiling before building anything.**
A measure whose best achievable bound is polynomial cannot reach superpolynomial no
matter how the proof goes.

### Why the screens bite together
Random functions are hard. Screen 2 forces μ small on most hard functions. Screen 1
forces μ small on IP. If μ is nonetheless large on SAT, it isn't detecting hardness at
all — it's detecting some *structural* property SAT has that random hard functions lack.
Any surviving candidate must name that property explicitly.

This is why every live program (GCT, multiplicative complexity, stabilizer rank,
tameness) is organized around structure rather than around hardness.

---

## 4. Candidate rulers

| Ruler | S1 (IP) | S2 (natural) | Ceiling | Best known bound | Notes |
|---|---|---|---|---|---|
| ℓ₁ Walsh norm | **FAIL** | fail | linear | — | dead, see §2 |
| Discrepancy / γ₂ / sign-rank | **FAIL** | — | — | — | dead, same counterexample |
| Multiplicative complexity | pass (n) | pass | none | ~2n | honest, stuck for decades |
| Stabilizer rank | pass (**rank 1**) | pass | none | Ω(n) | best fit for existing code |
| Gowers norms U^k, k≥3 | pass at k≥4 | pass | — | — | analytic successor to Fourier |
| Nielsen geodesic metric | pass by construction | pass | none | none | reformulation, not a tool |
| GCT | pass | pass by design | none | none | only program built *around* S2 |
| o-minimal tameness | pass | pass (rare property) | — | none | §6 of the old notes |

**IP has stabilizer rank 1** — it's a quadratic form over GF(2), so Σ(−1)^{IP(x)}|x⟩ is a
stabilizer state. The function that killed ℓ₁ is maximally *easy* under stabilizer rank.
Cleanest possible signal that the measure points the right way.

---

## 5. Salvage list

**Keep and repackage:**
- `Cl2_BooleanEmbedding.v`, `Cln_Full.v`, `Cln_SupportAlgebra.v`, `Cln_BoolDist.v`,
  `Cln_Grade.v` — a ℚ-exact, admit-free, *executable* Boolean Fourier analysis library.
  No floating point. No equivalent exists in Coq as far as I know.
- `Cln_finite_l1_submultiplicativity.v` — the theorem is true and useful even though the
  application failed.
- The CNF compiler in `CompositeExcursion` (`compile_cnf_correct`,
  `compile_cnf_coeff_correct`, `cnf_exists_small_rep`) — correctness and coefficient
  bounds are proven and independent of the separation story.

**Cut:**
- `PneqNP_from_dimension.v`, `RepComp_P.v`, `Cln_Nonuniform_Separation.v`. These assume
  SAT hardness as a hypothesis and derive P≠NP in two lines. Leaving them in the repo
  will get the solid 500-lemma development dismissed on sight, which it doesn't deserve.
- `Cln_DAG.v` (41 admits), and the excursion/blowup machinery in `CompositeExcursion`
  above the CNF compiler.

**Where the Clifford twist genuinely earns its keep:** Cl(n,0)⊗ℂ is the Pauli algebra;
the sign cocycle *is* the Pauli commutation structure. The thing that made the measure
useless for SAT is the thing that makes the algebra useful for stabilizer formalism and
quantum compiler verification. Also matchgates / Gottesman–Knill.

---

## 6. Two forks — pick one deliberately

### Fork A: engineering (finishable)
Verified complexity geometry as infrastructure. ℚ-exact Cl(n,0) with a grade-weighted
Nielsen metric, geodesic length, and the correspondence to circuit cost formalized.
Absent from every proof assistant. Payoff is a tool, not a theorem.

Note the catch honestly: a Nielsen metric passes all three screens *by construction*,
because it's defined to equal circuit cost. It relocates the difficulty rather than
solving it. Nobody has extracted a superpolynomial bound from it in twenty years. Build
it as infrastructure, not as an attack.

### Fork B: research (open-ended)
Pick a ruler that isn't definitionally circuit cost but still passes the screens.
**Stabilizer rank is the best fit** for the existing Clifford infrastructure. Both it and
multiplicative complexity are stuck at Ω(n), which is the actual frontier.

### The key methodological inversion
Nielsen doesn't pick a natural metric and hope. He *defines* the metric by penalizing
hard directions so that geodesic length equals circuit cost by construction. **The
geometry doesn't discover the cost function; the cost function builds the geometry.**

Last time the approach was: find a measure inside a space I liked. The move that works
is: build the space around a measure that's already justified.

---

## 7. Concrete next actions

1. **Split the repo.** Fourier library in one place, speculative separation work archived
   in another. Add `_CoqProject` files. Fix the `have`/ssreflect import or delete the
   file.
2. **Write the screening harness first.** A Coq or Python module that, given any candidate
   μ, computes it on: IP, parity, majority, mod-3, a random linear-size circuit, and a
   random function. This is a day's work now that the substrate exists. It is the actual
   deliverable from the last few months.
3. **Only then** implement the new ruler, and run it through the harness *before* proving
   anything about it.
4. Consider writing up the ℓ₁ negative result on its own. "IP maximizes ℓ₁ Walsh norm and
   has a linear-size circuit; submultiplicativity caps any such measure at linear lower
   bounds" is a clean, checkable barrier statement with machine-checked support. People
   burn years on barriers they never manage to state.

---

## 8. Things not to do again

- Don't build 19,000 lines before testing the measure on IP.
- Don't trust a picture. Vividness is not significance. IP's spectrum *looks* maximally
  complex — spread across all 2ⁿ coefficients — and it's n ANDs and n XORs. The eye
  reports "complicated" and complicated is not hard. Keep a numerical check between every
  picture and every conclusion.
- Don't keep a harness that assumes the hard hypothesis next to real work. `Hypothesis
  SAT_not_in_RP` → `Theorem P_neq_NP` is A → A, and readers will judge the whole repo by
  it.
- Don't let a failed ruler contaminate the framing. Geometric complexity is not fringe:
  Nielsen's geodesic reformulation, Mulmuley's GCT, Bürgisser's condition-number theory
  all live there. The instinct was right. One candidate invariant died, with a proof
  rather than a shrug.

---

## 9. The one thing that went right

The correct diagnosis of the IP problem was written into the comments of
`Cln_CompositeExcursion.v` *before* anyone else looked at it:

> the lower bound is coming from: IP has big ℓ₁ Walsh mass in your fixed basis...
> That is not the "intermediate blowup forced by constrained composition" story.
> It's "the output itself is huge under this measure."

Finding the flaw and writing it down instead of routing around it is the skill that
transfers. It's rarer than the Coq fluency. Keep doing that.
