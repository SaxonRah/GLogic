(*
  ============================================================
  File: Cln_Multivector.v
  ============================================================

  Multivectors in Cl(n) (coefficient model) over rationals:

    MV n := Mask n -> Q    (coefficients on blades e_S)

  Also includes:
    * basic linear structure ( 0, +, scalar * )
    * finite sums over enumerations
    * evaluation functional Eval_s(F) = Σ_S F_S χ_S(s)

  This file is purely linear-algebraic, but the indexing is
  explicitly by GA basis blades (masks).
*)

Require Import Cln_Basis.

From Coq Require Import List.
From Coq Require Import Bool.
From Coq Require Import Arith.
From Coq Require Import QArith.
From Coq Require Import Vectors.Vector.
From Coq Require Import Setoid.
From Coq Require Import Morphisms.
From Coq Require Import Ring.

Import ListNotations.

From Coq Require Import QArith.Qring.
Open Scope Q_scope.


Set Implicit Arguments.

(* ============================================================ *)
(* Multivectors as coefficient functions                          *)
(* ============================================================ *)

Definition MV (n : nat) : Type := Mask n -> Q.

Definition mv_zero {n} : MV n := fun _ => 0%Q.
Definition mv_add  {n} (F G : MV n) : MV n := fun m => (F m + G m)%Q.
Definition mv_scale {n} (k : Q) (F : MV n) : MV n := fun m => (k * F m)%Q.

Infix "⊕" := mv_add (at level 50, left associativity).

(* Basis blade e_S as Kronecker delta coefficient function *)
Definition basis {n} (S : Mask n) : MV n :=
  fun T => if mask_eq_dec T S then 1%Q else 0%Q.

(* ============================================================ *)
(* Finite sums over lists                                         *)
(* ============================================================ *)

Fixpoint sumQ (xs : list Q) : Q :=
  match xs with
  | List.nil => 0%Q
  | List.cons x tl => (x + sumQ tl)%Q
  end.

Lemma sumQ_app : forall xs ys,
  sumQ (xs ++ ys) == (sumQ xs + sumQ ys)%Q.
Proof.
  induction xs; intros ys; simpl.
  - ring.
  - rewrite IHxs. ring.
Qed.

Lemma sumQ_map_ext :
  forall (A : Type) (f g : A -> Q) (l : list A),
    (forall x, List.In x l -> f x == g x) ->
    sumQ (List.map f l) == sumQ (List.map g l).
Proof.
  intros A f g l.
  induction l as [|a tl IH]; intros H; simpl.
  - reflexivity.
  - (* head *)
    apply Qplus_comp.
    + apply H. left; reflexivity.
    + apply IH. intros x Hx. apply H. right; exact Hx.
Qed.

Lemma sumQ_map_add :
  forall (A : Type) (f g : A -> Q) (l : list A),
    sumQ (List.map (fun x => (f x + g x)%Q) l) ==
    (sumQ (List.map f l) + sumQ (List.map g l))%Q.
Proof.
  induction l as [|a tl IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

Lemma sumQ_map_scale_l :
  forall (A : Type) (k : Q) (f : A -> Q) (l : list A),
    sumQ (List.map (fun x => (k * f x)%Q) l) ==
    (k * sumQ (List.map f l))%Q.
Proof.
  induction l as [|a tl IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

(* ============================================================ *)
(* Characters χ_S(s): product of selected coordinates              *)
(* ============================================================ *)

Definition sQ (s : Sign) : Q :=
  match s with
  | Pos => 1%Q
  | Neg => (-1)%Q
  end.

Lemma sQ_sq1 : forall s, (sQ s * sQ s)%Q == 1%Q.
Proof. destruct s; simpl; ring. Qed.

Lemma sQ_mul_eqb :
  forall a b,
    (sQ a * sQ b)%Q == (if sign_eqb a b then 1%Q else (-1)%Q).
Proof.
  destruct a, b; simpl; reflexivity.
Qed.

Fixpoint chi (n : nat) : Mask n -> Corner n -> Q :=
  match n with
  | O => fun _ _ => 1%Q
  | S n' =>
      fun m s =>
        let mh := Vector.hd m in
        let mt := Vector.tl m in
        let sh := Vector.hd s in
        let st := Vector.tl s in
        ((if mh then sQ sh else 1%Q) * (@chi n' mt st))%Q
  end.

(* ============================================================ *)
(* Chi lemmas (use chi')                                         *)
(* ============================================================ *)

Definition chi' {n : nat} : Mask n -> Corner n -> Q := @chi n.

Lemma chi_false_cons :
  forall n (m : Mask n) (s : Corner n) (h : Sign),
    chi' (n := S n) (Vector.cons bool false n m)
                   (Vector.cons Sign h n s)
    ==
    chi' (n := n) m s.
Proof.
  intros n m s h.
  unfold chi', chi. simpl.
  ring.
Qed.

Lemma chi_true_cons :
  forall n (m : Mask n) (s : Corner n) (h : Sign),
    chi' (n := S n) (Vector.cons bool true n m)
                   (Vector.cons Sign h n s)
    ==
    (sQ h * chi' (n := n) m s)%Q.
Proof.
  intros n m s h.
  unfold chi', chi. simpl.
  ring.
Qed.


(* ============================================================ *)
(* pow2                                                          *)
(* ============================================================ *)

Fixpoint pow2 (n : nat) : Q :=
  match n with
  | O => 1%Q
  | S k => (2%Q * pow2 k)%Q
  end.

Lemma pow2_nonzero : forall n, ~(pow2 n == 0%Q).
Proof.
  induction n; simpl.
  - (* pow2 0 = 1 *)
    intro H; discriminate.
  - intro H.
    apply Qmult_integral in H.
    destruct H as [H2|Hk].
    + discriminate.
    + apply IHn; exact Hk.
Qed.

(* ============================================================ *)
(* Evaluation functional                                          *)
(* ============================================================ *)

Definition eval {n} (F : MV n) (s : Corner n) : Q :=
  sumQ (List.map (fun m => (F m * chi' m s)%Q) (all_masks n)).

Lemma eval_add : forall n (F G : MV n) (s : Corner n),
  eval (mv_add F G) s == (eval F s + eval G s)%Q.
Proof.
  intros n F G s.
  unfold eval, mv_add.
  rewrite <- sumQ_map_add.
  apply sumQ_map_ext; intros m Hm; simpl; ring.
Qed.

Lemma eval_scale : forall n (k : Q) (F : MV n) (s : Corner n),
  eval (mv_scale k F) s == (k * eval F s)%Q.
Proof.
  intros n k F s.
  unfold eval, mv_scale.
  rewrite <- sumQ_map_scale_l.
  apply sumQ_map_ext; intros m Hm; simpl; ring.
Qed.

(* ============================================================ *)
(* Swap finite sums over corners (Fubini)                         *)
(* ============================================================ *)

(* Helper: sum of a list of zeros is zero *)
Lemma sumQ_map_zero :
  forall (A : Type) (l : list A),
    sumQ (List.map (fun _ => 0%Q) l) == 0%Q.
Proof.
  intros A l.
  induction l as [|a tl IH]; simpl.
  - reflexivity.
  - rewrite IH. simpl. reflexivity.
Qed.

Lemma eval_sum_over_corners :
  forall n (cs : list (Corner n)) (H : Corner n -> MV n) (s : Corner n),
    eval (fun m => sumQ (List.map (fun a => H a m) cs)) s ==
    sumQ (List.map (fun a => eval (H a) s) cs).
Proof.
  intros n cs.
  induction cs as [|a tl IH]; intros H s; simpl.
  - (* cs = [] *)
    unfold eval. simpl.
    (* LHS: sumQ (map (fun m => 0 * chi) all_masks) == 0 *)
    apply Qeq_trans with (sumQ (List.map (fun _ : Mask n => 0%Q) (all_masks n))).
    + apply sumQ_map_ext; intros m Hm; simpl; reflexivity.
    + apply sumQ_map_zero.
  - (* cs = a :: tl *)
    unfold eval. simpl.
    (* Distribute the multiplication across the pointwise sum *)
    apply Qeq_trans with
      (sumQ (List.map
               (fun m : Mask n =>
                  ((H a m + sumQ (List.map (fun a0 : Corner n => H a0 m) tl)) * chi' m s)%Q)
               (all_masks n))).
    + reflexivity.
    + (* Split into two sums using sumQ_map_add after rewriting the mapped function *)
      apply Qeq_trans with
        (sumQ (List.map (fun m : Mask n => (H a m * chi' m s)%Q) (all_masks n)) +
         sumQ (List.map (fun m : Mask n =>
                           (sumQ (List.map (fun a0 : Corner n => H a0 m) tl) * chi' m s)%Q)
                        (all_masks n)))%Q.
      * (* use sumQ_map_add *)
        rewrite <- sumQ_map_add.
        apply sumQ_map_ext; intros m Hm; simpl.
        (* (x+y)*z = x*z + y*z *)
        ring.
      * (* Now identify the first term as eval(H a) s and the second via IH *)
        rewrite <- (IH H s).
        unfold eval. simpl.
        ring.
Qed.

