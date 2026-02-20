(***
  TraceGeometryPipeline.v

  A “glue” layer that follows the TODO_TraceGeometry.md roadmap and
  connects:

    - TraceGeometryCore.v
    - TraceGeometryExplosion.v
    - TraceGeometryMachine.v

  into a single lower-bound pipeline statement.

  This file is intentionally *parametric*: the target family, its spec,
  and the machine/trace accounting are supplied as modules/axioms.

  You can instantiate it with your concrete Cln (or other) model later.
*)

Unset Universe Polymorphism.
Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

From Stdlib Require Import List Arith Lia PeanoNat.
Import ListNotations.

From TraceGeometry Require Import TraceGeometryCore.
From TraceGeometry Require Import TraceGeometryExplosion.
From TraceGeometry Require Import TraceGeometryMachine.
From TraceGeometry Require Import TraceGeometryStep5Strong.

Declare Scope tg_scope.
Delimit Scope tg_scope with tg.
Open Scope tg_scope.

(************************************************************)
(* Step 0: target family + correctness predicate              *)
(************************************************************)

Module Type TRACE_GEOMETRY_TARGET
  (F  : TRACE_GEOMETRY_FULL)
  (SM : TRACE_GEOMETRY_STACK_MACHINE(F.TG)).

  Module TG := F.TG.
  Module EI := F.EI.
  Import TG EI.

  Parameter T : nat -> TG.A.

  (* Your machine start-state can depend on n. *)
  Parameter st0 : nat -> SM.Stack.

  (* A semantic spec: what the target “means” on an input. *)
  Parameter spec : nat -> EI.Input -> TG.Sem.

  (* The usual correctness notion: output matches spec on every input. *)
  Definition Correct (n : nat) (p : SM.Prog) : Prop :=
    forall s : EI.Input,
      EI.eval_at s (SM.output (st0 n) p) = spec n s.

  (* Glue axiom: spec is realized by the semantic evaluation of T n. *)
  Axiom spec_sound :
    forall n s, EI.eval_at s (T n) = spec n s.

End TRACE_GEOMETRY_TARGET.

(************************************************************)
(* Step 1: correctness ⇒ equality in A via eval injectivity  *)
(************************************************************)

Module TraceGeometryCorrectnessBridge
  (F  : TRACE_GEOMETRY_FULL)
  (SM : TRACE_GEOMETRY_STACK_MACHINE(F.TG))
  (Tgt : TRACE_GEOMETRY_TARGET(F)(SM)).

  Module TG := F.TG.
  Module EI := F.EI.
  Import TG EI.

  Lemma correct_output_eq_T :
    forall (n : nat) (p : SM.Prog),
      Tgt.Correct n p -> SM.output (Tgt.st0 n) p = Tgt.T n.
  Proof.
    intros n p Hcorr.
    apply EI.eval_at_injective.
    intro s.
    specialize (Hcorr s).
    (* Turn the RHS spec into eval_at s (T n) *)
    rewrite <- (Tgt.spec_sound n s) in Hcorr.
    exact Hcorr.
  Qed.

End TraceGeometryCorrectnessBridge.

(************************************************************)
(* Step 2/3/4: split-richness invariant + one-step gp growth *)
(************************************************************)

Module TraceGeometryMachineExplosion
  (F  : TRACE_GEOMETRY_FULL)
  (SM : TRACE_GEOMETRY_STACK_MACHINE(F.TG)).

  Module TG := F.TG.
  Module EX := F.EX.
  Module GR := F.GR.
  Module SP := GR.S.
  Import TG EX GR SP.

  (* “HasSplitAtLeast x m” is the predicate-form invariant from the roadmap. *)
  Definition HasSplitAtLeast (x : A) (m : nat) : Prop :=
    exists S : Split x,
      (length (split_parts S) >= m)%nat.

  (* A machine transition is a gp-step iff it matches the IGp rule. *)
  Definition IsGpStep (st st' : SM.Stack) : Prop :=
    exists (x y : A) (tail : SM.Stack),
      st = x :: y :: tail /\
      st' = (x ⋆ y)%tg :: tail.

  Lemma gp_step_has_split_mult :
    forall st st' x y tail m n,
      st = x :: y :: tail ->
      st' = (x ⋆ y)%tg :: tail ->
      HasSplitAtLeast x m ->
      HasSplitAtLeast y n ->
      HasSplitAtLeast (x ⋆ y)%tg (m * n).
  Proof.
    intros st st' x y tail m n Hst Hst' [SX HSX] [SY HSY].
    subst.
    destruct (split_growth_under_gp (x:=x) (y:=y) SX SY) as [SZ Hlen].
    exists SZ.
    (* From growth-under-gp we get:
         len(parts SZ) >= len(parts SX) * len(parts SY)
       and we are assuming:
         len(parts SX) >= m, len(parts SY) >= n.
     *)
    nia.
  Qed.

End TraceGeometryMachineExplosion.

(************************************************************)
(* Step 5: counting gp-mix events and global growth along a run *)
(************************************************************)

Module TraceGeometryRunAccounting
  (F  : TRACE_GEOMETRY_FULL)
  (SM : TRACE_GEOMETRY_STACK_MACHINE(F.TG)).

  Module TG := F.TG.
  Import TG.

  (* Syntactic counting of IGp instructions (the simplest roadmap option). *)
  Fixpoint count_gp (p : SM.Prog) : nat :=
    match p with
    | [] => 0%nat
    | i :: tl =>
        match i with
        | SM.IGp => S (count_gp tl)
        | _ => count_gp tl
        end
    end.

  Lemma count_gp_le_cost :
    forall p, (count_gp p <= SM.cost p)%nat.
  Proof.
    induction p as [|i tl IH]; simpl; auto.
    destruct i; simpl; lia.
  Qed.

  (*
    For a sharp separation lower bound, you usually need to count
    “effective gp-mix” steps (those where BOTH operands are split-rich).

    This is intentionally left abstract here: you can refine it to the
    stronger variant later.
  *)

End TraceGeometryRunAccounting.

(************************************************************)
(* Step 6: target-specific “must be huge” lower bound          *)
(************************************************************)

Module Type TRACE_GEOMETRY_TARGET_LOWER_BOUND
  (F  : TRACE_GEOMETRY_FULL).

  Module TG := F.TG.
  Module EX := F.EX.
  Module SP := F.GR.S.
  Import TG EX SP.

  Parameter T : nat -> A.

  (* A convenient concrete exponential. Replace as desired. *)
  Definition Exp (n : nat) : nat := Nat.pow 2 n.

  (*
    “Computing T_n forces huge Parts / huge norm1.”

    This is where your grade + booldist + ℓ1 machinery pays rent.
    You can state it either in terms of Split length or directly norm1.
  *)
  Axiom target_requires_split_parts :
    forall n,
      exists S : Split (T n),
        (length (split_parts S) >= Exp n)%nat.

End TRACE_GEOMETRY_TARGET_LOWER_BOUND.

(************************************************************)
(* Step 7/8: assemble into a cost lower bound (skeleton)      *)
(************************************************************)

Module TraceGeometryLowerBoundPipeline
  (F  : TRACE_GEOMETRY_FULL)
  (SM : TRACE_GEOMETRY_STACK_MACHINE(F.TG))
  (Tgt : TRACE_GEOMETRY_TARGET(F)(SM))
  (LB  : TRACE_GEOMETRY_TARGET_LOWER_BOUND(F)
          with Definition T := Tgt.T).

  Module TG := F.TG.
  Module EI := F.EI.
  Module EX := F.EX.
  Module GR := F.GR.
  Module SP := GR.S.

  Module Br := TraceGeometryCorrectnessBridge(F)(SM)(Tgt).
  Module MX := TraceGeometryMachineExplosion(F)(SM).
  Module AC := TraceGeometryRunAccounting(F)(SM).

  Import TG EI EX GR SP.
  Import Br MX AC.

  (*
    The remaining ingredient is the *global* growth bound:

      along a run, the number of split parts can grow at most
      multiplicatively with each IGp (or each “effective gp-mix”).

    This is the part you’ll specialize once you decide your exact
    machine semantics and invariant (Step 5 in the roadmap).

    We state it here as an axiom so the pipeline theorem can be written
    cleanly, and you can later replace it with an actual proof.
  *)

  Axiom parts_growth_bound_by_gp_count :
    forall (n : nat) (p : SM.Prog),
      (* If the program runs, then the output’s split length is bounded
         by an exponential in the number of gp instructions. *)
      (exists S : Split (SM.output (Tgt.st0 n) p),
         length (split_parts S) <= Nat.pow 2 (count_gp p))%nat.

  (* A minimal “min over programs” definition (Step 8). *)
  Definition MinCost (n : nat) : nat :=
    Nat.min
      (* this is a placeholder default; you will probably replace MinCost
         with an infimum over programs once you have a finite search space
         or a well-founded minimum principle. *)
      (0%nat)
      (0%nat).

  (*
    Pipeline theorem (Step 7):

      semantic correctness
        ⇒ output = T n
        ⇒ output has Exp(n) many split parts
        ⇒ (by parts growth bound) need enough gp-count
        ⇒ cost lower bound.

    Note: the last step uses count_gp <= cost.
  *)
  Theorem cost_lower_bound_from_explosion :
    forall (n : nat) (p : SM.Prog),
      Tgt.Correct n p ->
      (SM.cost p >= n)%nat.
  Proof.
    (***
      Roadmap sketch:
        1) correctness -> output = T n   (Br.correct_output_eq_T)
        2) T n has Exp n split parts    (LB.target_requires_split_parts)
        3) global accounting bounds output split parts by gp-count
        4) gp-count <= cost
        5) conclude cost >= n

      Fill in once you have the Step-5 lemma that relates the machine’s
      gp events to the Split-growth axiom.
    ***)
  Admitted.


End TraceGeometryLowerBoundPipeline.


(************************************************************)
(* Step 7 (strong, reusable): cost LB from effective count   *)
(*                                                          *)
(* This is the “finished” Step 7 glue, parameterized by:     *)
(*   - an adapter from the machine program p to a runs_eff   *)
(*     trace (Step 2 adapter)                                *)
(*   - a base split-richness fact at the start observation    *)
(*   - a Step 6 forcing lemma giving n <= t for correct p     *)
(*                                                          *)
(* See TraceGeometryStep7Assemble.v for a concrete packaging. *)
(************************************************************)

Module Type TRACE_GEOMETRY_STEP7_STRONG_ASSUMPTIONS
  (F  : TRACE_GEOMETRY_FULL)
  (SM : TRACE_GEOMETRY_STACK_MACHINE(F.TG))
  (Tgt : TRACE_GEOMETRY_TARGET(F)(SM)).

  Module S5 := TraceGeometryStep5Strong(F)(SM).

  Definition b : nat := 2%nat.
  Definition M0 : nat := 1%nat.

  Axiom runs_eff_sound_for_output :
    forall (n : nat) (p : SM.Prog),
      exists (t : nat) (M' : nat) (tr : list SM.Stack),
        S5.runs_eff b M0 (Tgt.st0 n) p t M' tr /\
        SM.obs (List.last tr (Tgt.st0 n)) = SM.output (Tgt.st0 n) p.

  Axiom start_has_split_1 :
    forall n, S5.HasSplitAtLeast (SM.obs (Tgt.st0 n)) M0.

  Axiom target_forces_effective_count :
    forall (n : nat) (p : SM.Prog) (t : nat) (M' : nat) (tr : list SM.Stack),
      Tgt.Correct n p ->
      S5.runs_eff b M0 (Tgt.st0 n) p t M' tr ->
      (n <= t)%nat.

End TRACE_GEOMETRY_STEP7_STRONG_ASSUMPTIONS.

Module TraceGeometryLowerBoundPipelineStrong
  (F  : TRACE_GEOMETRY_FULL)
  (SM : TRACE_GEOMETRY_STACK_MACHINE(F.TG))
  (Tgt : TRACE_GEOMETRY_TARGET(F)(SM))
  (A7  : TRACE_GEOMETRY_STEP7_STRONG_ASSUMPTIONS(F)(SM)(Tgt)).

  Module S5 := TraceGeometryStep5Strong(F)(SM).

  Theorem cost_lower_bound_from_effective_count :
    forall (n : nat) (p : SM.Prog),
      Tgt.Correct n p ->
      (SM.cost p >= n)%nat.
  Proof.
    intros n p Hcorr.
    destruct (A7.runs_eff_sound_for_output n p) as [t [M' [tr [Hruns _]]]].

    assert (Hnt : (n <= t)%nat)
      by (eapply A7.target_forces_effective_count; eauto).

    pose proof (A7.S5.runs_eff_t_le_cost Hruns) as Htlen.

    unfold SM.cost.
    lia.
  Qed.

End TraceGeometryLowerBoundPipelineStrong.
