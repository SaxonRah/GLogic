(*
  ============================================================
  PneqNP_from_dimension.v
  ============================================================

  This file imports RepComp_P.v and sets up the *P vs NP harness*.

  The key design change vs your earlier "FULL" file:
    - We do NOT assume vague "time_bounds_representation" globally.
    - We reuse the theorem InP <-> InRP from RepComp_P.v.
    - P≠NP becomes a corollary of ONE explicit separation hypothesis:

        SAT_family ∉ InRP

  plus a standard NP-completeness story (here left abstract).

  This isolates the real research work into a single lemma:
    SAT_not_in_RP_for_your_rep_language

  ============================================================
*)

From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Arith.Arith.
From Coq Require Import Lia.

Import ListNotations.
Open Scope nat_scope.

Require Import RepComp_P.

(* ============================================================ *)
(* PART 1: NP interface (abstract)                              *)
(* ============================================================ *)

(*
  We only need a minimal notion of NP for a corollary:
    - InNP: families decidable by poly-time verifier with witness

  You can replace this with your own full SAT/CNF development later.
*)

Parameter InNP : FnFamily -> Prop.

(* SAT family (NP-complete target) *)
Parameter SAT_family : FnFamily.

Axiom SAT_in_NP : InNP SAT_family.

(* One way to connect NP to SAT: NP-completeness via reductions.
   We keep this abstract as well. *)
Parameter polytime_manyone_reduces : FnFamily -> FnFamily -> Prop.

Axiom SAT_is_NP_complete :
  forall F, InNP F -> polytime_manyone_reduces F SAT_family.

(*
  Standard closure lemma: if F reduces to G and G in P then F in P.
  (You can later prove this once reductions are formalized.)
*)
Axiom reduction_preserves_P :
  forall F G, polytime_manyone_reduces F G -> InP G -> InP F.

(* ============================================================ *)
(* PART 2: Choose a representation language Rep                 *)
(* ============================================================ *)

(*
  Here we fix a Rep language. This file is parametrized over RepLang,
  but you’ll typically instantiate it with your candidate Rep:

    - uniform circuits (baseline sanity)
    - GA/Walsh reps (your target)

  The only requirement: you must have the two bridges from RepComp_P:
    eval_as_alg and compile, with the axioms/lemmas they require.
*)

Section WithRep.

Context {Rep : nat -> Type}.
Context `{RepLang Rep}.

(* Bridges from RepComp_P *)
Context (eval_as_alg : RepFamily Rep -> Alg)
        (compile : Alg -> RepFamily Rep).

Hypothesis eval_as_alg_correct :
  forall (Rfam : RepFamily Rep) n x,
    wf n x ->
    run n (eval_as_alg Rfam) x = rep_eval n (Rfam n) x.

Hypothesis eval_as_alg_time :
  forall (Rfam : RepFamily Rep) n x,
    wf n x ->
    time n (eval_as_alg Rfam) x <= rep_eval_time n (Rfam n) x + 1.

Hypothesis compile_correct :
  forall (A:Alg) n x,
    wf n x ->
    rep_eval n (compile A n) x = run n A x.

Hypothesis compile_size_poly :
  forall (A:Alg),
    polytime A ->
    exists p, poly p /\ forall n, rep_size n (compile A n) <= p n.

Hypothesis compile_eval_poly :
  forall (A:Alg),
    polytime A ->
    exists p, poly p /\ forall n x, wf n x ->
      rep_eval_time n (compile A n) x <= p n.

(* Shorthand: InRP for this RepLang *)
Let InRP' := InRP (Rep:=Rep).

(* Bring in the P <-> RP theorem specialized to these bridges *)
Theorem P_iff_RP :
  forall F, InP F <-> InRP' F.
Proof.
  intro F.
  apply (Representation_equals_Computation_for_P
           (Rep:=Rep)
           (eval_as_alg:=eval_as_alg)
           (compile:=compile));
  assumption.
Qed.

(* ============================================================ *)
(* PART 3: The ONE hard hypothesis you need for P≠NP            *)
(* ============================================================ *)

(*
  This is where your GA/Walsh “dimension/invariant” work must land:

    SAT_not_in_RP : SAT_family does not have poly-size/poly-eval reps
                    in YOUR representation language.

  Proving this is essentially the separation step.
*)
Hypothesis SAT_not_in_RP : ~ InRP' SAT_family.

(* ============================================================ *)
(* PART 4: P ≠ NP as a corollary (clean harness)                *)
(* ============================================================ *)

Theorem P_neq_NP_from_SAT_not_in_RP :
  ~ (forall F, InNP F -> InP F).
Proof.
  intro H.
  (* If everything in NP is in P, SAT is in P *)
  have SAT_in_P : InP SAT_family := H SAT_family SAT_in_NP.

  (* Convert SAT_in_P to SAT_in_RP using P <-> RP *)
  have SAT_in_RP : InRP' SAT_family.
  { apply (proj1 (P_iff_RP SAT_family)). exact SAT_in_P. }

  contradiction.
Qed.

(*
  Optional stronger statement using NP-completeness:
    If SAT is NP-complete and SAT ∉ RP, then NP ⊄ P.
  (Already implied by above, but you can also show "exists F in NP not in P".)
*)
Theorem exists_NP_not_P :
  exists F, InNP F /\ ~ InP F.
Proof.
  (* Contraposition: if all NP were in P, we contradict SAT_not_in_RP *)
  exists SAT_family.
  split; [exact SAT_in_NP|].
  intro SAT_in_P.
  have SAT_in_RP : InRP' SAT_family.
  { apply (proj1 (P_iff_RP SAT_family)). exact SAT_in_P. }
  contradiction.
Qed.

End WithRep.

(*
  ============================================================
  End PneqNP_from_dimension.v

  What you do next:

  1) Instantiate RepLang with your chosen representation:
       - circuits: should be doable; SAT_not_in_RP is unknown.
       - GA/Walsh: you aim to prove SAT_not_in_RP using your invariants.

  2) Your research target becomes ONE lemma:
       SAT_not_in_RP

     Everything else is bookkeeping that this file already handles.
  ============================================================
*)
