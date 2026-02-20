From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

From TraceGeometry Require Import TraceGeometryCore.

Module ToyZ_Compiles <: TRACE_GEOMETRY_CORE.

  Definition A := Z.
  Definition K := Z.
  Definition Sem := Z.

  Definition zero : A := 0%Z.
  Definition add  : A -> A -> A := Z.add.
  Definition neg  : A -> A := Z.opp.
  Definition smul : K -> A -> A := Z.mul.

  (* Key point: conv must distribute over add, so take multiplication. *)
  Definition conv : A -> A -> A := Z.mul.
  Definition gp   : A -> A -> A := Z.mul.

  Definition sem_zero : Sem := 0%Z.
  Definition sem_add  : Sem -> Sem -> Sem := Z.add.
  Definition sem_mul  : Sem -> Sem -> Sem := Z.mul.

  Definition eval : A -> Sem := fun x => x.

  Definition Semantic (_ : A) : Prop := True.

  (* Trivial excursion, so all Exc_* obligations are immediate. *)
  Definition Exc (_ : A) : nat := 0%nat.

  Definition step (x y : A) : Prop := (x <= y)%Z.

  Infix "+" := add : tg_scope.
  Notation "- x" := (neg x) : tg_scope.
  Notation "c • x" := (smul c x) (at level 40, left associativity) : tg_scope.
  Infix "⊙" := conv (at level 40, left associativity) : tg_scope.
  Infix "⋆" := gp (at level 40, left associativity) : tg_scope.

  Lemma add_assoc : forall x y z : A, x + (y + z) = (x + y) + z.
  Proof. intros; unfold add; lia. Qed.

  Lemma add_comm : forall x y : A, x + y = y + x.
  Proof. intros; unfold add; lia. Qed.

  Lemma add_zero_l : forall x : A, zero + x = x.
  Proof. intros; unfold zero, add; lia. Qed.

  Lemma add_left_inv : forall x : A, (-x) + x = zero.
  Proof. intros; unfold neg, add, zero; lia. Qed.

  Lemma smul_distr_add :
    forall (c : K) (x y : A), c • (x + y) = (c • x) + (c • y).
  Proof.
    intros c x y.
    unfold smul, add.
    now rewrite Z.mul_add_distr_l.
  Qed.
  
  Lemma conv_assoc : forall x y z : A, x ⊙ (y ⊙ z) = (x ⊙ y) ⊙ z.
    Proof.
      intros x y z.
      unfold conv.
      now rewrite Z.mul_assoc.
    Qed.

  Lemma conv_comm : forall x y : A, x ⊙ y = y ⊙ x.
    Proof.
      intros x y.
      unfold conv.
      now rewrite Z.mul_comm.
    Qed.

  Lemma conv_add_l : forall x y z : A, (x + y) ⊙ z = (x ⊙ z) + (y ⊙ z).
  Proof.
    intros x y z.
    unfold conv, add.
    now rewrite Z.mul_add_distr_r.
  Qed.

  Lemma conv_add_r : forall x y z : A, x ⊙ (y + z) = (x ⊙ y) + (x ⊙ z).
  Proof.
    intros x y z.
    unfold conv, add.
    now rewrite Z.mul_add_distr_l.
  Qed.

  Lemma gp_assoc : forall x y z : A, x ⋆ (y ⋆ z) = (x ⋆ y) ⋆ z.
    Proof.
      intros x y z.
      unfold gp.
      now rewrite Z.mul_assoc.
    Qed.

  Lemma gp_add_l : forall x y z : A, (x + y) ⋆ z = (x ⋆ z) + (y ⋆ z).
  Proof.
    intros x y z.
    unfold gp, add.
    now rewrite Z.mul_add_distr_r.
  Qed.

  Lemma gp_add_r : forall x y z : A, x ⋆ (y + z) = (x ⋆ y) + (x ⋆ z).
  Proof.
    intros x y z.
    unfold gp, add.
    now rewrite Z.mul_add_distr_l.
  Qed.

  Lemma eval_zero_ax : eval zero = sem_zero.
  Proof. reflexivity. Qed.

  Lemma eval_add_ax : forall x y : A, eval (x + y) = sem_add (eval x) (eval y).
  Proof. reflexivity. Qed.

  Lemma eval_conv_ax : forall x y : A, eval (x ⊙ y) = sem_mul (eval x) (eval y).
  Proof. reflexivity. Qed.

  Lemma Exc_semantic_zero : forall x : A, Semantic x -> Exc x = 0%nat.
  Proof. intros; reflexivity. Qed.

  Lemma Exc_step_mono :
    forall x y : A, step x y -> (Exc x <= Exc y)%nat.
  Proof.
    intros x y _.
    unfold Exc.
    apply Nat.le_refl.
  Qed.
  
  Lemma Exc_add_le :
    forall x y : A,
      (Exc ((x + y)%tg) <= Nat.max (Exc x) (Exc y))%nat.
  Proof.
    intros x y.
    unfold Exc.
    simpl.
    apply Nat.le_0_l.
  Qed.

  Lemma Exc_conv_le :
    forall x y : A,
      (Exc ((x ⊙ y)%tg) <= Nat.max (Exc x) (Exc y))%nat.
  Proof.
    intros x y.
    unfold Exc.
    simpl.
    apply Nat.le_0_l.
  Qed.

  Lemma Exc_gp_le :
    forall x y : A,
      (Exc ((x ⋆ y)%tg) <= (Exc x + Exc y))%nat.
  Proof.
    intros x y.
    unfold Exc.
    simpl.
    apply Nat.le_refl.
  Qed.


  Definition Gauge : Type := unit.
  Definition gauge (_ : Gauge) (x : A) : A := x.

  Lemma Exc_gauge_invariant :
    forall (g : Gauge) (x : A), Exc (gauge g x) = Exc x.
  Proof. intros; reflexivity. Qed.

End ToyZ_Compiles.
