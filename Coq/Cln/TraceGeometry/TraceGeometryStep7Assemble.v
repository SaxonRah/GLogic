Unset Universe Polymorphism.
Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

From Stdlib Require Import List Arith Lia.
Import ListNotations.

From TraceGeometry Require Import TraceGeometryCore.
From TraceGeometry Require Import TraceGeometryExplosion.
From TraceGeometry Require Import TraceGeometryMachine.
From TraceGeometry Require Import TraceGeometryPipeline.
From TraceGeometry Require Import TraceGeometryStep5Strong.

Declare Scope tg_scope.
Delimit Scope tg_scope with tg.
Open Scope tg_scope.

(*************************************************************)
(* Step 7 (assemble, strong):                                *)
(*                                                          *)
(* This file is the “glue” that actually uses Step 5 strong  *)
(* (TraceGeometryStep5Strong.v) to turn a *target-specific*  *)
(* forcing lemma into a cost lower bound.                    *)
(*                                                          *)
(* Roadmap ref: TODO_TraceGeometry.md §7                     *)
(*                                                          *)
(* IMPORTANT: Step 5 strong proves:                          *)
(*   - t (effective gp-mix count) <= length p (hence <= cost) *)
(*   - an explosion *lower bound* on split-richness along a   *)
(*     disciplined run (runs_eff).                            *)
(*                                                          *)
(* To conclude cost >= n, you still need a Step 6-style       *)
(* *forcing lemma* that guarantees any correct program must   *)
(* have at least n effective mixes (or at least LB n).        *)
(* That forcing lemma is the “math heart” and is kept abstract *)
(* here.                                                     *)
(*************************************************************)

Module TraceGeometryStep7Assemble
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
  Module S5 := TraceGeometryStep5Strong(F)(SM).

  Import TG EI EX GR SP.
  Import Br.

  (************************************************************)
  (* Parameters to connect the machine execution to runs_eff. *)
  (************************************************************)

  (* Run Step-5-Strong with b=2 and M0=1 (the simplest base). *)
  Definition b : nat := 2%nat.
  Definition M0 : nat := 1%nat.

  (*
    A1) For any program p, provide a disciplined runs_eff trace starting
        from the same initial stack as SM.output, and relate the final
        observed value to the machine output.

    In a concrete instantiation you will discharge this by proving
    that your machine execution semantics coincides with S5.istep/runs_i,
    and then building runs_eff by classifying each IGp step as effective
    or ineffective.
  *)
  Axiom runs_eff_sound_for_output :
    forall (n : nat) (p : SM.Prog),
      exists (t : nat) (M' : nat) (tr : list SM.Stack),
        S5.runs_eff b M0 (Tgt.st0 n) p t M' tr /\
        SM.obs (List.last tr (Tgt.st0 n)) = SM.output (Tgt.st0 n) p.

  (*
    A2) Base split-richness at the start observation.
    In most models this is easy (often obs(st0 n) is zero or a constant
    with a canonical split of length 1).
  *)
  Axiom start_has_split_1 :
    forall n, S5.HasSplitAtLeast (SM.obs (Tgt.st0 n)) M0.

  (************************************************************)
  (* Step 6 hook: “correctness forces many effective mixes”.  *)
  (************************************************************)

  (*
    This is the *only* truly target-dependent assumption needed to turn
    Step 5’s accounting into a cost lower bound.

    Typical ways to prove it (later) include:
      - grade / booldist / ℓ1 arguments that show T n demands exponential
        split explosion, plus
      - instruction stability lemmas (Step 3) that ensure explosion can only
        be created by effective gp-mix steps, so you get a lower bound on t.

    For now we keep it abstract so Step 7 is mechanically finished.
  *)
  Axiom target_forces_effective_count :
    forall (n : nat) (p : SM.Prog) (t : nat) (M' : nat) (tr : list SM.Stack),
      Tgt.Correct n p ->
      S5.runs_eff b M0 (Tgt.st0 n) p t M' tr ->
      (n <= t)%nat.

  (************************************************************)
  (* Step 7 theorem: Correct -> cost >= n.                    *)
  (************************************************************)

  Theorem cost_lower_bound_from_effective_count :
    forall (n : nat) (p : SM.Prog),
      Tgt.Correct n p ->
      (SM.cost p >= n)%nat.
  Proof.
    intros n p Hcorr.

    destruct (runs_eff_sound_for_output n p) as [t [M' [tr [Hruns _Heq]]]].

    (* Step 6: correctness forces many effective gp-mixes. *)
    assert (Hnt : (n <= t)%nat) by (eapply target_forces_effective_count; eauto).

    (* Step 5 meta: t <= length p. *)
    assert (Htlen : (t <= length p)%nat) by (eapply S5.runs_eff_t_le_cost; eauto).

    (* cost is at least length p (this is how cost is defined in the machine file). *)
    (* In this codebase TraceGeometryMachine.v defines cost = length p. *)
    (* We keep the proof robust to future tweaks via simple rewriting. *)
    unfold SM.cost.
    lia.
  Qed.

End TraceGeometryStep7Assemble.
