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

(* ============================================================ *)
(* XOR / Parity function                                         *)
(* ============================================================ *)

Definition sign_to_bool (s : Sign) : bool :=
  match s with Pos => true | Neg => false end.

Definition XOR_n_func {n} (c : Corner n) : bool :=
  List.fold_right xorb false
    (Vector.to_list (Vector.map sign_to_bool c)).

Lemma XOR_n_func_cons :
  forall n (h : Sign) (c : Corner n),
    @XOR_n_func (S n) (Vector.cons _ h _ c)
    = xorb (sign_to_bool h) (@XOR_n_func n c).
Proof.
  intros n h c.
  unfold XOR_n_func.
  cbn [Vector.map].
  rewrite to_list_cons.
  simpl.
  reflexivity.
Qed.

Lemma bQ_negb : forall b : bool, bQ (negb b) == (1 - bQ b)%Q.
Proof. destruct b; unfold bQ; simpl; ring. Qed.

(* ============================================================ *)
(* The inner Fourier sum at the full mask                        *)
(* ============================================================ *)

Definition xor_sum (n : nat) : Q :=
  sumQ (List.map
    (fun a : Corner n =>
       (bQ (@XOR_n_func n a) * chi' (Vector.const true n) a)%Q)
    (all_corners n)).

Lemma embed_XOR_full_mask :
  forall n,
    embed (@XOR_n_func n) (Vector.const true n)
    == ((1 / pow2 n) * xor_sum n)%Q.
Proof.
  intro n.
  unfold embed, xor_sum, Pi.
  unfold chi'.              (* <— this is the key line *)
  rewrite <- sumQ_map_scale_l.
  apply sumQ_map_ext; intros a Ha.
  ring.
Qed.

(* ============================================================ *)
(* Sum of chi(full mask, ·) over all corners = 0  (dim >= 1)    *)
(* ============================================================ *)

Lemma chi_full_sum_zero :
  forall n,
    sumQ (List.map
      (fun a : Corner (S n) =>
         chi' (Vector.const true (S n)) a)
      (all_corners (S n)))
    == 0.
Proof.
  intro n.
  simpl (all_corners (S n)).
  rewrite List.map_app.
  rewrite !List.map_map.
  rewrite sumQ_app.
  (* Pos half: sQ Pos = 1, so chi(full, Pos::a) = 1 * chi(full_tail, a) *)
  assert (HPos :
    sumQ (List.map
      (fun x => chi' (Vector.const true (S n)) (Pos :: x))
      (all_corners n))
    ==
    sumQ (List.map
      (fun a => chi' (Vector.const true n) a)
      (all_corners n))).
  { apply sumQ_map_ext; intros a Ha.
    rewrite (@chi_true_cons n (Vector.const true n) a Pos).
    simpl (sQ Pos). ring. }
  (* Neg half: sQ Neg = -1, so chi(full, Neg::a) = -chi(full_tail, a) *)
  assert (HNeg :
    sumQ (List.map
      (fun x => chi' (Vector.const true (S n)) (Neg :: x))
      (all_corners n))
    ==
    ((-1) * sumQ (List.map
      (fun a => chi' (Vector.const true n) a)
      (all_corners n)))%Q).
  { rewrite <- sumQ_map_scale_l.
    apply sumQ_map_ext; intros a Ha.
    rewrite (@chi_true_cons n (Vector.const true n) a Neg).
    simpl (sQ Neg). ring. }
  rewrite HPos, HNeg. ring.
Qed.

(* ============================================================ *)
(* Base case: xor_sum 1 = 1                                      *)
(* ============================================================ *)

Lemma xor_sum_base : xor_sum 1 == 1.
Proof.
  unfold xor_sum, XOR_n_func, sign_to_bool, bQ.
  simpl.
  unfold chi', chi, sQ.
  ring.
Qed.

(* ============================================================ *)
(* Recurrence: xor_sum (S n) = -2 * xor_sum n   (for n >= 1)   *)
(* ============================================================ *)

From Coq Require Import FunctionalExtensionality.

Lemma sumQ_map_sub :
  forall (A : Type) (f g : A -> Q) (l : list A),
    sumQ (List.map (fun x => (f x - g x)%Q) l) ==
    (sumQ (List.map f l) - sumQ (List.map g l))%Q.
Proof.
  intros A f g l.

  (* (f - g) == (f + (-g)) pointwise under sumQ(map ...) *)
  transitivity (sumQ (List.map (fun x => (f x + (- g x))%Q) l)).
  - apply sumQ_map_ext; intros x Hx; ring.
  - (* use sumQ_map_add *)
    rewrite (sumQ_map_add (A:=A) f (fun x => (- g x)%Q) l).

    (* rewrite the second summand: sumQ(map (fun x => -g x)) == (-1) * sumQ(map g) *)
    assert (Hneg :
      sumQ (List.map (fun x => (- g x)%Q) l) ==
      ((-1) * sumQ (List.map g l))%Q).
    {
      transitivity (sumQ (List.map (fun x => ((-1) * g x)%Q) l)).
      - apply sumQ_map_ext; intros x Hx; ring.
      - rewrite (sumQ_map_scale_l (A:=A) (-1)%Q g l).
        reflexivity.
    }
    rewrite Hneg.

    (* now: a + (-1)*b == a - b *)
    ring.
Qed.

Lemma xor_sum_step :
  forall n, (n > 0)%nat ->
    xor_sum (S n) == ((-2) * xor_sum n)%Q.
Proof.
  intros n Hn.
  unfold xor_sum at 1.
  simpl (all_corners (S n)).
  rewrite List.map_app.
  rewrite !List.map_map.
  rewrite sumQ_app.

  (* --- Pos branch --- *)
  assert (HPos :
    sumQ (List.map
      (fun a =>
        (bQ (@XOR_n_func (S n) (Pos :: a)) *
         chi' (Vector.const true (S n)) (Pos :: a))%Q)
      (all_corners n))
    ==
    (sumQ (List.map
      (fun a => chi' (Vector.const true n) a)
      (all_corners n))
     - xor_sum n)%Q).
  {
    unfold xor_sum.
    (* fold RHS difference into one map using sumQ_map_sub *)      
    rewrite <- (@sumQ_map_sub (Corner n)
      (fun a => chi' (Vector.const true n) a)
      (fun a => (bQ (@XOR_n_func n a) * chi' (Vector.const true n) a)%Q)
      (all_corners n)).


    apply sumQ_map_ext; intros a Ha.
    rewrite XOR_n_func_cons. simpl (sign_to_bool Pos).
    rewrite Bool.xorb_true_l.
    rewrite (@chi_true_cons n (Vector.const true n) a Pos).
    simpl (sQ Pos).
    rewrite bQ_negb.
    ring.
  }

  (* --- Neg branch --- *)
  assert (HNeg :
    sumQ (List.map
      (fun a =>
        (bQ (@XOR_n_func (S n) (Neg :: a)) *
         chi' (Vector.const true (S n)) (Neg :: a))%Q)
      (all_corners n))
    ==
    ((-1) * xor_sum n)%Q).
  {
    unfold xor_sum.
    rewrite <- sumQ_map_scale_l.
    apply sumQ_map_ext; intros a Ha.
    rewrite XOR_n_func_cons. simpl (sign_to_bool Neg).
    rewrite Bool.xorb_false_l.
    rewrite (@chi_true_cons n (Vector.const true n) a Neg).
    simpl (sQ Neg).
    ring.
  }

  rewrite HPos, HNeg.

  (* chi full sum is 0 for dimension >= 1 *)
  destruct n as [|n']; [lia|].
  rewrite (chi_full_sum_zero n').
  ring.
Qed.

(* ============================================================ *)
(* Rational nonzero helpers                                      *)
(* ============================================================ *)

Lemma Qmul_nonzero_l :
  forall p q : Q, ~(p == 0) -> ~(q == 0) -> ~(p * q == 0).
Proof.
  intros p q Hp Hq Hpq.
  apply Qmult_integral in Hpq.
  destruct Hpq; contradiction.
Qed.

Lemma Qneg2_nonzero : ~((-2)%Q == 0).
Proof. unfold Qeq; simpl; discriminate. Qed.

Lemma Qinv_nonzero : forall q : Q, ~(q == 0) -> ~(/q == 0).
Proof.
  intros q Hq Hinv.
  assert (H1 : q * /q == 1) by (apply Qmult_inv_r; exact Hq).
  assert (H2 : q * /q == 0) by (rewrite Hinv; ring).
  assert (H3 : (1 == 0)%Q) by (eapply Qeq_trans; [symmetry; exact H1 | exact H2]).
  unfold Qeq in H3; simpl in H3; discriminate H3.
Qed.

Lemma Qdiv1_nonzero : forall q : Q, ~(q == 0) -> ~((1 / q) == 0).
Proof.
  intros q Hq. unfold Qdiv.
  apply Qmul_nonzero_l.
  - unfold Qeq; simpl; discriminate.
  - exact (Qinv_nonzero Hq).
Qed.

(* ============================================================ *)
(* xor_sum n is nonzero for n >= 1                               *)
(* ============================================================ *)

Lemma xor_sum_nonzero :
  forall n, (n > 0)%nat -> ~(xor_sum n == 0).
Proof.
  intro n.
  induction n as [|n IHn]; intro Hn.
  - lia.
  - destruct n as [|n'].
    + (* n = 0, proving xor_sum 1 ≠ 0 *)
      rewrite xor_sum_base.
      unfold Qeq; simpl; discriminate.
    + (* n = S n', proving xor_sum (S (S n')) ≠ 0 *)
      rewrite (@xor_sum_step (S n') ltac:(lia)).
      apply Qmul_nonzero_l.
      * exact Qneg2_nonzero.
      * apply IHn. lia.
Qed.

(* ============================================================ *)
(* XOR has nonzero grade-n component (pseudoscalar coefficient)  *)
(* ============================================================ *)

Lemma XOR_has_grade_n_component :
  forall n,
    (n > 0)%nat ->
    ~(embed (@XOR_n_func n) (Vector.const true n) == 0).
Proof.
  intros n Hn Hembed.
  rewrite embed_XOR_full_mask in Hembed.
  apply Qmult_integral in Hembed.
  destruct Hembed as [Hdiv | Hxor].
  - exact (Qdiv1_nonzero (pow2_nonzero n) Hdiv).
  - exact (xor_sum_nonzero Hn Hxor).
Qed.

(* ============================================================ *)
(* Parity excursion lower bound                                  *)
(* ============================================================ *)

Theorem parity_excursion :
  forall n sq (e : GA_expr n),
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
