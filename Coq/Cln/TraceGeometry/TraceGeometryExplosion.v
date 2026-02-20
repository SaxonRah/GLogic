Unset Universe Polymorphism.
Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

From Stdlib Require Import List Arith Lia.
Import ListNotations.

From TraceGeometry Require Import TraceGeometryCore.

Declare Scope tg_scope.
Delimit Scope tg_scope with tg.
Open Scope tg_scope.

(************************************************************)
(* Eval family + injectivity (Walsh bridge)                  *)
(************************************************************)

Module Type TRACE_GEOMETRY_EVAL_INJ (TG : TRACE_GEOMETRY_CORE).
  Import TG.
  Open Scope tg_scope.

  Parameter Input : Type.
  Parameter eval_at : Input -> A -> Sem.

  Axiom eval_at_zero :
    forall s : Input, eval_at s zero = sem_zero.

  Axiom eval_at_add :
    forall (s : Input) (x y : A),
      eval_at s (x + y) = sem_add (eval_at s x) (eval_at s y).

  Axiom eval_at_conv :
    forall (s : Input) (x y : A),
      eval_at s (x ⊙ y) = sem_mul (eval_at s x) (eval_at s y).

  Axiom eval_at_injective :
    forall x y : A,
      (forall s : Input, eval_at s x = eval_at s y) -> x = y.
End TRACE_GEOMETRY_EVAL_INJ.

(************************************************************)
(* Explosion primitives (no Split parameter!)                *)
(************************************************************)

Module Type TRACE_GEOMETRY_EXPLOSION_PRIMS (TG : TRACE_GEOMETRY_CORE).
  Import TG.
  Open Scope tg_scope.

  Parameter Supp : Type.
  Parameter supp : A -> Supp.
  Parameter sep  : nat -> Supp -> Supp -> Prop.

  Parameter norm1 : A -> nat.

  Axiom norm1_zero : norm1 zero = 0%nat.

  (* subadditivity is optional but useful *)
  Axiom norm1_add_le :
  forall x y : A,
    (norm1 (x + y)%tg <= (norm1 x + norm1 y)%nat)%nat.


  (* anti-cancellation under separation *)
  Axiom norm1_add_ge_if_sep :
    forall (k : nat) (x y : A),
      sep k (supp x) (supp y) ->
      (norm1 (x + y)%tg >= norm1 x + norm1 y)%nat.

  (* nonzero pieces have at least unit mass (ℓ1-like) *)
  Axiom norm1_nonzero_ge_1 :
    forall a : A, a <> zero -> (norm1 a >= 1)%nat.

End TRACE_GEOMETRY_EXPLOSION_PRIMS.

(************************************************************)
(* Split is DEFINED (inductive), not axiomatized             *)
(************************************************************)

Module TraceGeometrySplit (TG : TRACE_GEOMETRY_CORE)
                         (EX : TRACE_GEOMETRY_EXPLOSION_PRIMS(TG)).
  Import TG EX.
  Open Scope tg_scope.

(*   Fixpoint sumA (xs : list A) : A :=
    fold_right add zero xs. *)

  Definition sumA (xs : list A) : A := fold_right add zero xs.

  Fixpoint sum_norm1 (xs : list A) : nat :=
    match xs with
    | [] => 0%nat
    | a :: tl => (norm1 a + sum_norm1 tl)%nat
    end.

  (*
    split_list k xs means:
      - xs is a list of “separated” nonzero components at scale k
      - separation is stated as: head is sep from the sum of the tail
        (this is the right abstraction for witness-relative BoolDist stability).
  *)
  Inductive split_list (k : nat) : list A -> Prop :=
  | split_nil : split_list k []
  | split_cons :
      forall a tl,
        a <> zero ->
        split_list k tl ->
        sep k (supp a) (supp (sumA tl)) ->
        split_list k (a :: tl).

  Record Split (x : A) := {
    split_scale : nat;
    split_parts : list A;
    split_eq : x = sumA split_parts;
    split_ok : split_list split_scale split_parts
  }.

  Lemma norm1_sumA_ge_sum_norm1 :
    forall k xs,
      split_list k xs ->
      (norm1 (sumA xs) >= sum_norm1 xs)%nat.
  Proof.
    intros k xs H.
    induction H; simpl.
    - rewrite norm1_zero. lia.
    - (* norm1(a + sumA tl) >= norm1 a + norm1(sumA tl) *)
      assert (Hge :
      (norm1 ((a + sumA tl)%tg) >= norm1 a + norm1 (sumA tl))%nat).
    {
      apply (norm1_add_ge_if_sep (k:=k) (x:=a) (y:=sumA tl)).
      exact H1.
    }
    unfold sumA in Hge, IHsplit_list.

    unfold sumA.
    lia.
  Qed.

  Lemma norm1_of_Split_ge_sum :
    forall x (S : Split x),
      (norm1 x >= sum_norm1 (split_parts S))%nat.
  Proof.
    intros x S.
    replace (norm1 x) with (norm1 (sumA (split_parts S))).
    - (* now matches norm1_sumA_ge_sum_norm1 *)
      apply (norm1_sumA_ge_sum_norm1 (k := split_scale S) (xs := split_parts S)).
      exact (split_ok S).
    - symmetry. exact (f_equal norm1 (split_eq S)).
  Qed.

  Lemma sum_norm1_ge_length_if_nonzero :
    forall k xs,
      split_list k xs ->
      (sum_norm1 xs >= length xs)%nat.
  Proof.
    intros k xs H.
    induction H; simpl.
    - lia.
    - pose proof (@norm1_nonzero_ge_1 a H) as Ha1.
      lia.
  Qed.

  Lemma norm1_of_Split_ge_length :
    forall x (S : Split x),
      (norm1 x >= length (split_parts S))%nat.
  Proof.
    intros x S.
    pose proof (norm1_of_Split_ge_sum (x:=x) S) as Hsum.
    assert (Hlen : (sum_norm1 (split_parts S) >= length (split_parts S))%nat).
    { apply (sum_norm1_ge_length_if_nonzero (k := split_scale S)).
      exact (split_ok S). }
    lia.
  Qed.

End TraceGeometrySplit.

(************************************************************)
(* Growth axiom: ⋆ multiplies the number of separated parts  *)
(************************************************************)

Module Type TRACE_GEOMETRY_SPLIT_GROWTH
  (TG : TRACE_GEOMETRY_CORE)
  (EX : TRACE_GEOMETRY_EXPLOSION_PRIMS(TG)).
  Module S := TraceGeometrySplit(TG)(EX).
  Import TG EX S.
  Open Scope tg_scope.

  Axiom split_growth_under_gp :
    forall x y (SX : Split x) (SY : Split y),
      exists SZ : Split (x ⋆ y),
        (length (split_parts SZ) >=
           length (split_parts SX) * length (split_parts SY))%nat.
End TRACE_GEOMETRY_SPLIT_GROWTH.

(************************************************************)
(* Combined “full” package                                   *)
(************************************************************)

Module Type TRACE_GEOMETRY_FULL.
  Declare Module TG : TRACE_GEOMETRY_CORE.
  Declare Module EI : TRACE_GEOMETRY_EVAL_INJ(TG).
  Declare Module EX : TRACE_GEOMETRY_EXPLOSION_PRIMS(TG).
  Declare Module GR : TRACE_GEOMETRY_SPLIT_GROWTH(TG)(EX).
End TRACE_GEOMETRY_FULL.

(************************************************************)
(* Derived explosion theorem                                 *)
(************************************************************)

Module TraceGeometryExplosionDerived (F : TRACE_GEOMETRY_FULL).
  Module TG := F.TG.
  Module EI := F.EI.
  Module EX := F.EX.
  Module GR := F.GR.
  
  Module SP := GR.S.
  (* Module SP := TraceGeometrySplit(TG)(EX). *)

  Import TG EX SP GR.
  Open Scope tg_scope.

  Theorem explosion_one_step_norm1 :
    forall x y (SX : Split x) (SY : Split y),
      exists SZ : Split (x ⋆ y),
        (norm1 (x ⋆ y) >=
           length (split_parts SX) * length (split_parts SY))%nat.
  Proof.
    intros x y SX SY.
    destruct (split_growth_under_gp SX SY) as [SZ Hlen].
    exists SZ.
    pose proof (norm1_of_Split_ge_length (x := x ⋆ y) SZ) as Hn.
    lia.
  Qed.

End TraceGeometryExplosionDerived.
