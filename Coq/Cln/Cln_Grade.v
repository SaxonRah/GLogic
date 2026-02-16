(*
  ============================================================
  File: Cln_Grade.v
  ============================================================

  Grade infrastructure for parity excursion lower bounds.
*)

Require Import Cln_Full.

From Coq Require Import List Bool Arith QArith Vectors.Vector.
From Coq Require Import Setoid Morphisms Lia.
Require Import Coq.Program.Equality.
From Coq Require Import Compare_dec.

Import ListNotations.
Import VectorNotations.

From Coq Require Import QArith.Qring.
Open Scope Q_scope.

Set Implicit Arguments.

(* ============================================================ *)
(* Grade = popcount of boolean mask                              *)
(* ============================================================ *)

Definition grade {n} (m : Mask n) : nat :=
  List.count_occ Bool.bool_dec (Vector.to_list m) true.

Lemma grade_cons_true : forall n (m : Mask n),
  grade (Vector.cons _ true _ m) = S (grade m).
Proof.
  intros n m.
  unfold grade. rewrite to_list_cons. simpl.
  destruct (Bool.bool_dec true true) as [_|H].
  - reflexivity.
  - exfalso; apply H; reflexivity.
Qed.

Lemma grade_cons_false : forall n (m : Mask n),
  grade (Vector.cons _ false _ m) = grade m.
Proof.
  intros n m.
  unfold grade. rewrite to_list_cons. simpl.
  destruct (Bool.bool_dec false true) as [H|_].
  - discriminate.
  - reflexivity.
Qed.

(* ============================================================ *)
(* Grade of distinguished masks                                  *)
(* ============================================================ *)

Lemma grade_empty_mask : forall n,
  grade (mask_empty (n:=n)) = 0%nat.
Proof.
  induction n as [|n IH].
  - reflexivity.
  - unfold mask_empty in *.
    change (Vector.const false (S n))
      with (Vector.cons _ false _ (Vector.const false n)).
    rewrite grade_cons_false.
    exact IH.
Qed.

Lemma grade_full_mask : forall n,
  grade (Vector.const true n) = n.
Proof.
  induction n as [|n IH].
  - reflexivity.
  - change (Vector.const true (S n))
      with (Vector.cons _ true _ (Vector.const true n)).
    rewrite grade_cons_true.
    f_equal.
    exact IH.
Qed.

Lemma grade_single : forall n (i : Fin.t n),
  grade (mask_single i) = 1%nat.
Proof.
  induction i as [|n i IH].
  - cbn [mask_single].
    rewrite grade_cons_true. rewrite grade_empty_mask. reflexivity.
  - cbn [mask_single].
    rewrite grade_cons_false. exact IH.
Qed.

(* ============================================================ *)
(* General grade bounds                                          *)
(* ============================================================ *)

Lemma grade_le_n : forall n (m : Mask n), (grade m <= n)%nat.
Proof.
  induction n as [|n IH]; intro m.
  - dependent destruction m.
    cbn. (* or: simpl. *)
    lia.
  - dependent destruction m. destruct h.
    + rewrite grade_cons_true. specialize (IH m). lia.
    + rewrite grade_cons_false. specialize (IH m). lia.
Qed.

Lemma grade_xor_le : forall n (A B : Mask n),
  (grade (mask_xor A B) <= grade A + grade B)%nat.
Proof.
  induction n as [|n IH]; intros A B.
  - dependent destruction A; dependent destruction B. simpl. lia.
  - dependent destruction A; dependent destruction B.
    rewrite mask_xor_cons.
    destruct h, h0; simpl (xorb _ _).
    + rewrite grade_cons_false, grade_cons_true, grade_cons_true.
      specialize (IH A B). lia.
    + rewrite grade_cons_true, grade_cons_true, grade_cons_false.
      specialize (IH A B). lia.
    + rewrite grade_cons_true, grade_cons_false, grade_cons_true.
      specialize (IH A B). lia.
    + rewrite grade_cons_false, grade_cons_false, grade_cons_false.
      exact (IH A B).
Qed.

Lemma grade_basis_mul_mask_le : forall n (A B : Mask n),
  (grade (basis_mul_mask A B) <= grade A + grade B)%nat.
Proof.
  intros n A B. unfold basis_mul_mask. apply grade_xor_le.
Qed.

(* ============================================================ *)
(* Qeq decidability                                              *)
(* ============================================================ *)

Lemma Qeq_bool_false_neq : forall x y : Q,
  Qeq_bool x y = false -> ~(x == y).
Proof.
  intros x y H Heq.
  apply Qeq_eq_bool in Heq. congruence.
Qed.

Definition Qeq_dec (x y : Q) : {x == y} + {~(x == y)}.
Proof.
  destruct (Qeq_bool x y) eqn:H.
  - left. apply Qeq_bool_eq. exact H.
  - right. exact (@Qeq_bool_false_neq x y H).
Defined.

(* ============================================================ *)
(* Max grade of a multivector                                    *)
(* ============================================================ *)

Definition max_grade {n} (F : MV n) : nat :=
  List.fold_right Nat.max 0%nat
    (List.map (fun m => if Qeq_bool (F m) 0 then 0%nat else grade m)
              (all_masks n)).

(* --- Helpers for fold_right Nat.max --- *)

Lemma fold_max_ge_In :
  forall (l : list nat) (x : nat),
    List.In x l -> (x <= List.fold_right Nat.max 0%nat l)%nat.
Proof.
  induction l as [|a tl IH]; intros x Hx; simpl.
  - inversion Hx.
  - destruct Hx as [<- | Htl].
    + lia.
    + specialize (IH x Htl). lia.
Qed.

Lemma fold_max_le_bound :
  forall (l : list nat) (k : nat),
    (forall x, List.In x l -> (x <= k)%nat) ->
    (List.fold_right Nat.max 0%nat l <= k)%nat.
Proof.
  induction l as [|a tl IH]; intros k Hk; simpl.
  - lia.
  - assert (Ha : (a <= k)%nat) by (apply Hk; left; reflexivity).
    assert (Htl : (List.fold_right Nat.max 0%nat tl <= k)%nat).
    { apply IH. intros x Hx. apply Hk. right. exact Hx. }
    lia.
Qed.

(* --- Specification lemmas --- *)

Lemma max_grade_spec :
  forall n (F : MV n) (m : Mask n),
    ~(F m == 0) -> (grade m <= max_grade F)%nat.
Proof.
  intros n F m Hne.
  unfold max_grade.
  apply fold_max_ge_In.
  apply List.in_map_iff.
  exists m. split.
  - destruct (Qeq_bool (F m) 0) eqn:Heq.
    + exfalso. apply Hne. apply Qeq_bool_eq. exact Heq.
    + reflexivity.
  - apply all_masks_complete.
Qed.

Lemma max_grade_le_n :
  forall n (F : MV n), (max_grade F <= n)%nat.
Proof.
  intros n F.
  unfold max_grade.
  apply fold_max_le_bound.
  intros x Hx.
  apply List.in_map_iff in Hx.
  destruct Hx as [m [Hm _]]. subst x.
  destruct (Qeq_bool (F m) 0).
  - lia.
  - apply grade_le_n.
Qed.

(* ============================================================ *)
(* Grade-bounded predicate                                       *)
(* ============================================================ *)

Definition grade_bounded {n} (F : MV n) (k : nat) : Prop :=
  forall m : Mask n, (grade m > k)%nat -> F m == 0.

Lemma max_grade_bounded :
  forall n (F : MV n), grade_bounded F (max_grade F).
Proof.
  intros n F m Hgt.
  destruct (Qeq_dec (F m) 0) as [Hz|Hnz].
  - exact Hz.
  - exfalso. pose proof (max_grade_spec F m Hnz). lia.
Qed.

Lemma bounded_implies_max_grade_le :
  forall n (F : MV n) (k : nat),
    grade_bounded F k -> (max_grade F <= k)%nat.
Proof.
  intros n F k HF.
  unfold max_grade.
  apply fold_max_le_bound.
  intros x Hx.
  apply List.in_map_iff in Hx.
  destruct Hx as [m [Hm Hin]]. subst x.
  destruct (Qeq_bool (F m) 0) eqn:Heq.
  - lia.
  - destruct (le_gt_dec (grade m) k) as [Hle|Hgt].
    + exact Hle.
    + exfalso.
      apply (@Qeq_bool_false_neq _ _ Heq).
      exact (HF m Hgt).
Qed.

Lemma grade_bounded_mono :
  forall n (F : MV n) (j k : nat),
    grade_bounded F j -> (j <= k)%nat -> grade_bounded F k.
Proof.
  intros n F j k HF Hjk m Hm. apply HF. lia.
Qed.

Lemma grade_bounded_zero :
  forall n (k : nat), grade_bounded (@mv_zero n) k.
Proof.
  intros n k m _. unfold mv_zero. reflexivity.
Qed.

(* ============================================================ *)
(* Grade evolution under addition                                *)
(* ============================================================ *)

Lemma grade_bounded_add :
  forall n (F G : MV n) (j k : nat),
    grade_bounded F j ->
    grade_bounded G k ->
    grade_bounded (mv_add F G) (Nat.max j k).
Proof.
  intros n F G j k HF HG m Hm.
  unfold mv_add.
  assert (HFm : F m == 0) by (apply HF; lia).
  assert (HGm : G m == 0) by (apply HG; lia).
  rewrite HFm, HGm. ring.
Qed.

Lemma max_grade_add_le :
  forall n (F G : MV n),
    (max_grade (mv_add F G) <= Nat.max (max_grade F) (max_grade G))%nat.
Proof.
  intros n F G.
  apply bounded_implies_max_grade_le.
  apply grade_bounded_add; apply max_grade_bounded.
Qed.

(* ============================================================ *)
(* Grade evolution under geometric product                       *)
(* ============================================================ *)

Lemma grade_bounded_gp :
  forall n (sq : Vector.t Q n) (F G : MV n) (j k : nat),
    grade_bounded F j ->
    grade_bounded G k ->
    grade_bounded (mv_gp sq F G) (j + k)%nat.
Proof.
  intros n sq F G j k HF HG U HU.
  unfold mv_gp.

  eapply Qeq_trans.
  { apply (@sumQ_map_ext (Mask n)
      _ (fun _ => 0%Q) (all_masks n)).
    intros A _.

    eapply Qeq_trans.
    { apply (@sumQ_map_ext (Mask n)
        _ (fun _ => 0%Q) (all_masks n)).
      intros B _. simpl.

      destruct (mask_eq_dec (basis_mul_mask A B) U) as [HAB|_].
      - assert (Hgrade : (grade A + grade B > j + k)%nat).
        { pose proof (grade_basis_mul_mask_le A B) as Hle.
          rewrite HAB in Hle. lia. }
          
        destruct (le_gt_dec (grade A) j) as [HleA | HgtA].
        + (* grade A <= j *)
          assert (HgtB : (grade B > k)%nat) by lia.
          rewrite (HG B HgtB). ring.
        + (* grade A > j *)
          rewrite (HF A HgtA). ring.
      - reflexivity.
    }
    exact (sumQ_map_const0 (A:=Mask n) (all_masks n)).
  }
  exact (sumQ_map_const0 (A:=Mask n) (all_masks n)).
Qed.

Lemma max_grade_gp_le :
  forall n (sq : Vector.t Q n) (F G : MV n),
    (max_grade (mv_gp sq F G) <= max_grade F + max_grade G)%nat.
Proof.
  intros n sq F G.
  apply bounded_implies_max_grade_le.
  apply grade_bounded_gp; apply max_grade_bounded.
Qed.

(* ============================================================ *)
(* Base cases for GA expression building blocks                  *)
(* ============================================================ *)

Lemma grade_bounded_basis_single :
  forall n (i : Fin.t n), grade_bounded (basis (mask_single i)) 1%nat.
Proof.
  intros n i m Hm. unfold basis.
  destruct (mask_eq_dec m (mask_single i)) as [Heq|_].
  - exfalso. rewrite Heq in Hm. rewrite grade_single in Hm. lia.
  - reflexivity.
Qed.

Lemma grade_bounded_scale_one :
  forall n (c : Q), grade_bounded (mv_scale c (@mv_one n)) 0%nat.
Proof.
  intros n c m Hm. unfold mv_scale, mv_one, basis.
  destruct (mask_eq_dec m (mask_empty (n:=n))) as [Heq|_].
  - exfalso. rewrite Heq in Hm. rewrite grade_empty_mask in Hm. lia.
  - ring.
Qed.

(* ============================================================ *)
(* GA expressions: syntax, evaluation, grade analysis            *)
(* ============================================================ *)

Inductive GA_expr (n : nat) : Type :=
  | Basis  : Fin.t n -> GA_expr n
  | Scalar : Q -> GA_expr n
  | Add    : GA_expr n -> GA_expr n -> GA_expr n
  | Mul    : GA_expr n -> GA_expr n -> GA_expr n.

Arguments Basis {n}.
Arguments Scalar {n}.
Arguments Add {n}.
Arguments Mul {n}.

Fixpoint eval_expr {n} (sq : Vector.t Q n) (e : GA_expr n) : MV n :=
  match e with
  | Basis i   => basis (mask_single i)
  | Scalar c  => mv_scale c mv_one
  | Add e1 e2 => mv_add (eval_expr sq e1) (eval_expr sq e2)
  | Mul e1 e2 => mv_gp sq (eval_expr sq e1) (eval_expr sq e2)
  end.

Fixpoint grade_bound {n} (e : GA_expr n) : nat :=
  match e with
  | Basis _   => 1%nat
  | Scalar _  => 0%nat
  | Add e1 e2 => Nat.max (grade_bound e1) (grade_bound e2)
  | Mul e1 e2 => (grade_bound e1 + grade_bound e2)%nat
  end.

Theorem eval_grade_bounded :
  forall n (sq : Vector.t Q n) (e : GA_expr n),
    grade_bounded (eval_expr sq e) (grade_bound e).
Proof.
  intros n sq e.
  induction e as [i | c | e1 IH1 e2 IH2 | e1 IH1 e2 IH2]; simpl.
  - exact (grade_bounded_basis_single i).
  - exact (grade_bounded_scale_one c).
  - exact (grade_bounded_add IH1 IH2).
  - exact (grade_bounded_gp sq IH1 IH2).
Qed.

Corollary max_grade_eval_le :
  forall n (sq : Vector.t Q n) (e : GA_expr n),
    (max_grade (eval_expr sq e) <= grade_bound e)%nat.
Proof.
  intros. apply bounded_implies_max_grade_le. apply eval_grade_bounded.
Qed.

(* ============================================================ *)
(* Max grade during evaluation (excursion tracker)               *)
(* ============================================================ *)

Fixpoint max_grade_during {n} (sq : Vector.t Q n) (e : GA_expr n) : nat :=
  match e with
  | Basis _   => 1%nat
  | Scalar _  => 0%nat
  | Add e1 e2 =>
      Nat.max (max_grade_during sq e1)
              (max_grade_during sq e2)
  | Mul e1 e2 =>
      Nat.max
        (Nat.max (max_grade_during sq e1)
                 (max_grade_during sq e2))
        (max_grade (mv_gp sq (eval_expr sq e1) (eval_expr sq e2)))
  end.

Lemma max_grade_le_during :
  forall n (sq : Vector.t Q n) (e : GA_expr n),
    (max_grade (eval_expr sq e) <= max_grade_during sq e)%nat.
Proof.
  intros n sq e.
  induction e as [i | c | e1 IH1 e2 IH2 | e1 IH1 e2 IH2]; simpl.
  - apply bounded_implies_max_grade_le.
    exact (grade_bounded_basis_single i).
  - apply bounded_implies_max_grade_le.
    exact (grade_bounded_scale_one c).
  - eapply Nat.le_trans.
    + exact (max_grade_add_le (eval_expr sq e1) (eval_expr sq e2)).
    + apply Nat.max_le_compat; assumption.
  - apply Nat.le_max_r.
Qed.

(* ============================================================ *)
(* Generic excursion lower bound                                 *)
(* ============================================================ *)

Theorem excursion_lower_bound :
  forall n (sq : Vector.t Q n) (e : GA_expr n) (F : MV n),
    eval_expr sq e = F ->
    (exists m : Mask n, (grade m >= n)%nat /\ ~(F m == 0)) ->
    (max_grade_during sq e >= n)%nat.
Proof.
  intros n sq e F Heval [m [Hgrade Hne]].
  assert (H1 : (grade m <= max_grade F)%nat).
  { apply max_grade_spec. exact Hne. }
  assert (H2 : (max_grade (eval_expr sq e) <= max_grade_during sq e)%nat).
  { apply max_grade_le_during. }
  subst F. lia.
Qed.

(* ============================================================ *)
(* Template for the parity instantiation                         *)
(* ============================================================ *)

(*
  To complete the parity excursion lower bound, prove:

  Lemma XOR_has_grade_n_component : forall n,
    (n > 0)%nat ->
    ~(embed (@XOR_n_func n) (Vector.const true n) == 0).

  Then:

  Theorem parity_excursion : forall n sq (e : GA_expr n),
    (n > 0)%nat ->
    eval_expr sq e = embed (@XOR_n_func n) ->
    (max_grade_during sq e >= n)%nat.
  Proof.
    intros n sq e Hn Heval.
    apply (excursion_lower_bound sq e Heval).
    exists (Vector.const true n).
    split.
    - rewrite grade_full_mask. lia.
    - exact (XOR_has_grade_n_component Hn).
  Qed.
*)