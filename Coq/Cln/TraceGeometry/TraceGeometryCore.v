(*
  TraceGeometryCore.v  (clean layout)

  - Module Type TRACE_GEOMETRY_CORE: only Parameters/Axioms/Notations.
  - Module TraceGeometryDerived(TG): derived defs (wf_trace, Exc_trace, etc.).
  - Module TraceGeometryPlayground(TG): extra utilities (trace_reaches, reaches).
*)

Unset Universe Polymorphism.
Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

From Stdlib Require Import List Arith.
Import ListNotations.

(************************************************************)
(* Scopes + Notations                                        *)
(************************************************************)

Declare Scope tg_scope.
Delimit Scope tg_scope with tg.
Open Scope tg_scope.

(************************************************************)
(* Minimal Axiomatic Core (SIGNATURE ONLY)                   *)
(************************************************************)

Module Type TRACE_GEOMETRY_CORE.

  (* Carriers *)
  Parameter A : Set.
  Parameter K : Set.
  Parameter Sem : Set.

  (* Additive / scalar structure *)
  Parameter zero : A.
  Parameter add  : A -> A -> A.
  Parameter neg  : A -> A.
  Parameter smul : K -> A -> A.

  (* Two multiplications *)
  Parameter conv : A -> A -> A.  (* ⊙  semantic/untwisted *)
  Parameter gp   : A -> A -> A.  (* ⋆  deformed/twisted  *)

  (* Semantics *)
  Parameter sem_zero : Sem.
  Parameter sem_add  : Sem -> Sem -> Sem.
  Parameter sem_mul  : Sem -> Sem -> Sem.

  Parameter eval : A -> Sem.
  Parameter Semantic : A -> Prop.

  (* Excursion *)
  Parameter Exc : A -> nat.

  (* Step relation (traces) *)
  Parameter step : A -> A -> Prop.

  (* Notations *)
  Infix "+" := add : tg_scope.
  Notation "- x" := (neg x) : tg_scope.
  Notation "c • x" := (smul c x) (at level 40, left associativity) : tg_scope.
  Infix "⊙" := conv (at level 40, left associativity) : tg_scope.
  Infix "⋆" := gp   (at level 40, left associativity) : tg_scope.

  (* Axioms: additive group (minimal) *)
  Axiom add_assoc     : forall x y z : A, x + (y + z) = (x + y) + z.
  Axiom add_comm      : forall x y   : A, x + y = y + x.
  Axiom add_zero_l    : forall x     : A, zero + x = x.
  Axiom add_left_inv  : forall x     : A, (-x) + x = zero.

  (* Axioms: scalar distribution (minimal) *)
  Axiom smul_distr_add :
    forall (c : K) (x y : A), c • (x + y) = (c • x) + (c • y).

  (* Axioms: conv ring laws (semantic algebra) *)
  Axiom conv_assoc : forall x y z : A, x ⊙ (y ⊙ z) = (x ⊙ y) ⊙ z.
  Axiom conv_comm  : forall x y   : A, x ⊙ y = y ⊙ x.
  Axiom conv_add_l : forall x y z : A, (x + y) ⊙ z = (x ⊙ z) + (y ⊙ z).
  Axiom conv_add_r : forall x y z : A, x ⊙ (y + z) = (x ⊙ y) + (x ⊙ z).

  (* Axioms: gp laws (deformed algebra) *)
  Axiom gp_assoc : forall x y z : A, x ⋆ (y ⋆ z) = (x ⋆ y) ⋆ z.
  Axiom gp_add_l : forall x y z : A, (x + y) ⋆ z = (x ⋆ z) + (y ⋆ z).
  Axiom gp_add_r : forall x y z : A, x ⋆ (y + z) = (x ⋆ y) + (x ⋆ z).

  (* Axioms: eval is a homomorphism for conv *)
  Axiom eval_zero_ax : eval zero = sem_zero.
  Axiom eval_add_ax  : forall x y : A, eval (x + y) = sem_add (eval x) (eval y).
  Axiom eval_conv_ax : forall x y : A, eval (x ⊙ y) = sem_mul (eval x) (eval y).

  (* Axioms: excursion behavior *)
  Axiom Exc_semantic_zero : forall x : A, Semantic x -> Exc x = 0%nat.
  Axiom Exc_step_mono     : forall x y : A, step x y -> (Exc x <= Exc y)%nat.

  Axiom Exc_add_le :
    forall x y : A, (Exc ((x + y)%tg) <= Nat.max (Exc x) (Exc y))%nat.

  Axiom Exc_conv_le :
    forall x y : A, (Exc (x ⊙ y) <= Nat.max (Exc x) (Exc y))%nat.

  Axiom Exc_gp_le :
    forall x y : A, (Exc (x ⋆ y) <= (Exc x + Exc y))%nat.

  (* Optional: Gauge invariance hook *)
  Parameter Gauge : Type.
  Parameter gauge : Gauge -> A -> A.

  Axiom Exc_gauge_invariant :
    forall (g : Gauge) (x : A), Exc (gauge g x) = Exc x.

End TRACE_GEOMETRY_CORE.

(************************************************************)
(* Derived trace machinery (NOT part of the signature)       *)
(************************************************************)

Module TraceGeometryDerived (TG : TRACE_GEOMETRY_CORE).
  Import TG.
  Open Scope tg_scope.

  Fixpoint wf_trace_aux (prev : A) (rest : list A) : Prop :=
    match rest with
    | [] => True
    | x :: tl => step prev x /\ wf_trace_aux x tl
    end.

  Definition wf_trace (t : list A) : Prop :=
    match t with
    | [] => True
    | x :: tl => wf_trace_aux x tl
    end.

  Fixpoint Exc_trace (t : list A) : nat :=
    match t with
    | [] => 0%nat
    | x :: tl => Nat.max (Exc x) (Exc_trace tl)
    end.

  Lemma Exc_trace_ge_head :
    forall x tl, (Exc x <= Exc_trace (x :: tl))%nat.
  Proof.
    intros x tl. simpl. apply Nat.le_max_l.
  Qed.
End TraceGeometryDerived.

(************************************************************)
(* Optional playground utilities                             *)
(************************************************************)

Module TraceGeometryPlayground (TG : TRACE_GEOMETRY_CORE).
  Module D := TraceGeometryDerived(TG).
  Import TG.
  Import D.
  Open Scope tg_scope.

  Definition trace_reaches (t : list A) (k : nat) : Prop :=
    wf_trace t /\ (k <= Exc_trace t)%nat.

  Inductive reaches : A -> nat -> A -> Prop :=
  | reaches0 : forall x, reaches x 0 x
  | reachesS : forall x n y z, reaches x n y -> step y z -> reaches x (S n) z.

End TraceGeometryPlayground.
