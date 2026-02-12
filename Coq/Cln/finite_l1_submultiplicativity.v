(*
  ============================================================
  File: finite_l1_submultiplicativity.v
  ============================================================

  Finite ℓ₁ submultiplicativity for the geometric product:

    ∥ F ⋆ G ∥₁ ≤ ∥F∥₁ · ∥G∥₁

  under the unit-metric hypothesis: for all A,B, |basis_mul_coeff sq A B| = 1,
  which follows from sq_i = ±1.

  This file avoids fragile inductions over (all_masks n) by using the helper:
    sumQ_map_le
  to lift pointwise inequalities into summed inequalities.

  Plug-in points:
    - basis_mul_coeff_abs1 : prove in Cln_GeometricProduct.v under sq_i=±1
    - sumQ_all_masks_pick  : you already proved this; import it and delete the Hypothesis
*)

Require Import Cln_Basis.
Require Import Cln_Multivector.
Require Import Cln_GeometricProduct.

From Coq Require Import List Bool Arith Lia QArith.
From Coq Require Import QArith.Qabs.
From Coq Require Import Setoid Morphisms Ring.
Import ListNotations.

Open Scope Q_scope.
Set Implicit Arguments.

(* ------------------------------------------------------------ *)
(* sumQ and basic map lemmas (as in your codebase)                *)
(* ------------------------------------------------------------ *)

Fixpoint sumQ (xs : list Q) : Q :=
  match xs with
  | [] => 0%Q
  | x :: tl => (x + sumQ tl)%Q
  end.

Lemma sumQ_map_ext :
  forall (A : Type) (f g : A -> Q) (l : list A),
    (forall x, In x l -> f x == g x) ->
    sumQ (map f l) == sumQ (map g l).
Proof.
  intros A f g l.
  induction l as [|a tl IH]; intros H; simpl.
  - reflexivity.
  - apply Qplus_comp.
    + apply H. left; reflexivity.
    + apply IH. intros x Hx. apply H. right; exact Hx.
Qed.

Lemma sumQ_map_add :
  forall (A : Type) (f g : A -> Q) (l : list A),
    sumQ (map (fun x => (f x + g x)%Q) l)
    ==
    (sumQ (map f l) + sumQ (map g l))%Q.
Proof.
  induction l as [|a tl IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

Lemma sumQ_map_scale_l :
  forall (A : Type) (k : Q) (f : A -> Q) (l : list A),
    sumQ (map (fun x => (k * f x)%Q) l)
    ==
    (k * sumQ (map f l))%Q.
Proof.
  induction l as [|a tl IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

Lemma sumQ_map_le :
  forall (A : Type) (f g : A -> Q) (l : list A),
    (forall x, In x l -> f x <= g x) ->
    sumQ (map f l) <= sumQ (map g l).
Proof.
  induction l as [|a tl IH]; intros H; simpl.
  - lra.
  - apply Qplus_le_compat.
    + apply H. left; reflexivity.
    + apply IH. intros x Hx. apply H. right; exact Hx.
Qed.

Lemma sumQ_map_const0 :
  forall (A : Type) (l : list A),
    sumQ (map (fun _ => 0%Q) l) == 0%Q.
Proof.
  induction l as [|a tl IH]; simpl.
  - reflexivity.
  - rewrite IH. ring.
Qed.

(* ------------------------------------------------------------ *)
(* ℓ₁ norm over MV coefficients                                  *)
(* ------------------------------------------------------------ *)

Definition l1_norm {n} (F : MV n) : Q :=
  sumQ (map (fun U => Qabs (F U)) (all_masks n)).

Notation "∥ F ∥₁" := (l1_norm F) (at level 40).

(* ------------------------------------------------------------ *)
(* Absolute-value lemmas                                          *)
(* ------------------------------------------------------------ *)

Lemma Qplus_le_compat_l : forall x y z : Q,
  y <= z -> x + y <= x + z.
Proof. intros. lra. Qed.

Lemma Qabs_sumQ_le :
  forall xs : list Q,
    Qabs (sumQ xs) <= sumQ (map Qabs xs).
Proof.
  induction xs as [|x tl IH]; simpl.
  - rewrite Qabs_Q0. lra.
  - eapply Qle_trans.
    + apply Qabs_triangle.
    + apply Qplus_le_compat_l. exact IH.
Qed.

Lemma Qabs_sumQ_map_le :
  forall (A:Type) (xs:list A) (f:A->Q),
    Qabs (sumQ (map f xs)) <= sumQ (map (fun a => Qabs (f a)) xs).
Proof.
  intros A xs f.
  exact (Qabs_sumQ_le (map f xs)).
Qed.

Lemma Qle_of_Qeq :
  forall x y : Q, x == y -> x <= y.
Proof.
  intros x y Hxy.
  setoid_rewrite Hxy.
  apply Qle_refl.
Qed.

(* ------------------------------------------------------------ *)
(* Fubini swap for sumQ                                           *)
(* ------------------------------------------------------------ *)

Lemma sumQ_fubini :
  forall (A B : Type) (la : list A) (lb : list B) (h : A -> B -> Q),
    sumQ (map (fun a => sumQ (map (fun b => h a b) lb)) la)
    ==
    sumQ (map (fun b => sumQ (map (fun a => h a b) la)) lb).
Proof.
  intros A B la.
  induction la as [|a tl IH]; intros lb h; simpl.
  - rewrite sumQ_map_const0. reflexivity.
  - eapply Qeq_trans.
    2: {
      apply (sumQ_map_ext (A:=B)
        (fun b => sumQ (map (fun a0 => h a0 b) (a :: tl)))
        (fun b => (h a b + sumQ (map (fun a0 => h a0 b) tl))%Q)
        lb).
      intros b Hb. simpl. reflexivity.
    }
    rewrite (sumQ_map_add (A:=B)
      (fun b => h a b)
      (fun b => sumQ (map (fun a0 => h a0 b) tl))
      lb).
    rewrite <- IH.
    ring.
Qed.

(* ------------------------------------------------------------ *)
(* Unit-metric assumptions + delta collapse                        *)
(* ------------------------------------------------------------ *)

Section UnitMetric.

Context {n : nat}.
Context (sq : Vector.t Q n).

Hypothesis basis_mul_coeff_abs1 :
  forall (A B : Mask n),
    Qabs (basis_mul_coeff sq A B) == 1.

Hypothesis sumQ_all_masks_pick :
  forall (f : Mask n -> Q) (U : Mask n),
    sumQ (map (fun m => if mask_eq_dec m U then f m else 0%Q) (all_masks n))
    == f U.

Lemma sumU_xor_delta :
  forall (A B : Mask n),
    sumQ (map (fun U =>
      if mask_eq_dec (basis_mul_mask A B) U then 1%Q else 0%Q) (all_masks n))
    == 1%Q.
Proof.
  intros A B.
  eapply Qeq_trans.
  2: { apply (sumQ_all_masks_pick (f:=fun _ => 1%Q) (U:=basis_mul_mask A B)). }
  apply sumQ_map_ext. intros m Hm.
  destruct (mask_eq_dec m (basis_mul_mask A B)); ring.
Qed.

(* ------------------------------------------------------------ *)
(* Main theorem                                                   *)
(* ------------------------------------------------------------ *)

Theorem l1_gp_submultiplicative :
  forall (F G : MV n),
    l1_norm (mv_gp n sq F G) <= l1_norm F * l1_norm G.
Proof.
  intros F G.
  unfold l1_norm, mv_gp.
  set (MS := all_masks n).

  set (term := fun (U A B : Mask n) =>
    let c := basis_mul_coeff sq A B in
    if mask_eq_dec (basis_mul_mask A B) U
    then (F A * G B * c)%Q else 0%Q).

  set (inner := fun (U : Mask n) =>
    sumQ (map (fun A : Mask n =>
      sumQ (map (fun B : Mask n => term U A B) MS)) MS)).

  (* Step 1: push abs through both finite sums via sumQ_map_le *)
  assert (Hstep1 :
  sumQ (map (fun U => Qabs (inner U)) MS)
  <=
  sumQ (map (fun U =>
    sumQ (map (fun A =>
      sumQ (map (fun B => Qabs (term U A B)) MS)) MS)) MS)).
    {
      apply sumQ_map_le.
      intros U HU.
      eapply Qle_trans.
      - apply Qabs_sumQ_map_le.
      - apply sumQ_map_le.
        intros A HA.
        apply Qabs_sumQ_map_le.
    }

  eapply Qle_trans.
  - exact Hstep1.
  - apply Qle_of_Qeq.

    (* Rewrite the bigsum to ∥F∥₁ · ∥G∥₁ *)

    (* Expand Qabs(term) and use |coeff|=1 *)
    eapply Qeq_trans.
    2: {
      apply (sumQ_map_ext (A:=Mask n)
        (fun U =>
          sumQ (map (fun A =>
            sumQ (map (fun B => Qabs (term U A B)) MS)) MS))
        (fun U =>
          sumQ (map (fun A =>
            sumQ (map (fun B =>
              if mask_eq_dec (basis_mul_mask A B) U
              then (Qabs (F A) * Qabs (G B) * Qabs (basis_mul_coeff sq A B))%Q
              else 0%Q) MS)) MS))
        MS).
      intros U HU.
      apply sumQ_map_ext; intros A HA.
      apply sumQ_map_ext; intros B HB.
      unfold term.
      destruct (mask_eq_dec (basis_mul_mask A B) U); simpl.
      - repeat rewrite Qabs_Qmult. ring.
      - rewrite Qabs_Q0. ring.
    }

    eapply Qeq_trans.
    2: {
      apply (sumQ_map_ext (A:=Mask n)
        (fun U =>
          sumQ (map (fun A =>
            sumQ (map (fun B =>
              if mask_eq_dec (basis_mul_mask A B) U
              then (Qabs (F A) * Qabs (G B) * Qabs (basis_mul_coeff sq A B))%Q
              else 0%Q) MS)) MS))
        (fun U =>
          sumQ (map (fun A =>
            sumQ (map (fun B =>
              if mask_eq_dec (basis_mul_mask A B) U
              then (Qabs (F A) * Qabs (G B))%Q
              else 0%Q) MS)) MS))
        MS).
      intros U HU.
      apply sumQ_map_ext; intros A HA.
      apply sumQ_map_ext; intros B HB.
      destruct (mask_eq_dec (basis_mul_mask A B) U); simpl.
      - rewrite basis_mul_coeff_abs1. ring.
      - ring.
    }

    (* Swap Σ_U Σ_A Σ_B to Σ_A Σ_B Σ_U using Fubini twice *)
    rewrite (sumQ_fubini (A:=Mask n) (B:=Mask n)
      (la:=MS) (lb:=MS)
      (h:=fun U A =>
        sumQ (map (fun B =>
          if mask_eq_dec (basis_mul_mask A B) U
          then (Qabs (F A) * Qabs (G B))%Q else 0%Q) MS))).

    eapply Qeq_trans.
    2: {
      apply (sumQ_map_ext (A:=Mask n)
        (fun A =>
          sumQ (map (fun U =>
            sumQ (map (fun B =>
              if mask_eq_dec (basis_mul_mask A B) U
              then (Qabs (F A) * Qabs (G B))%Q else 0%Q) MS)) MS))
        (fun A =>
          sumQ (map (fun B =>
            sumQ (map (fun U =>
              if mask_eq_dec (basis_mul_mask A B) U
              then (Qabs (F A) * Qabs (G B))%Q else 0%Q) MS)) MS))
        MS).
      intros A HA.
      apply sumQ_fubini.
    }

    (* Collapse the U-sum to 1 and factor constants *)
    eapply Qeq_trans.
    2: {
      apply (sumQ_map_ext (A:=Mask n)
        (fun A =>
          sumQ (map (fun B =>
            sumQ (map (fun U =>
              if mask_eq_dec (basis_mul_mask A B) U
              then (Qabs (F A) * Qabs (G B))%Q else 0%Q) MS)) MS))
        (fun A =>
          sumQ (map (fun B =>
            (Qabs (F A) * Qabs (G B))%Q *
            sumQ (map (fun U =>
              if mask_eq_dec (basis_mul_mask A B) U then 1%Q else 0%Q) MS)) MS))
        MS).
      intros A HA.
      apply sumQ_map_ext; intros B HB.
      rewrite <- (sumQ_map_scale_l (A:=Mask n)
        (k:=(Qabs (F A) * Qabs (G B))%Q)
        (f:=fun U => if mask_eq_dec (basis_mul_mask A B) U then 1%Q else 0%Q)
        MS).
      apply sumQ_map_ext; intros U HU.
      destruct (mask_eq_dec (basis_mul_mask A B) U); ring.
    }

    eapply Qeq_trans.
    2: {
      apply (sumQ_map_ext (A:=Mask n)
        (fun A =>
          sumQ (map (fun B =>
            (Qabs (F A) * Qabs (G B))%Q *
            sumQ (map (fun U =>
              if mask_eq_dec (basis_mul_mask A B) U then 1%Q else 0%Q) MS)) MS))
        (fun A =>
          sumQ (map (fun B =>
            (Qabs (F A) * Qabs (G B))%Q * 1%Q) MS))
        MS).
      intros A HA.
      apply sumQ_map_ext; intros B HB.
      rewrite sumU_xor_delta. ring.
    }

    (* Factor Σ_A Σ_B |F A||G B| = (Σ_A |F A|)(Σ_B |G B|) *)
    eapply Qeq_trans.
    2: {
      apply (sumQ_map_ext (A:=Mask n)
        (fun A =>
          sumQ (map (fun B => (Qabs (F A) * Qabs (G B))%Q) MS))
        (fun A =>
          (Qabs (F A) * sumQ (map (fun B => Qabs (G B)) MS))%Q)
        MS).
      intros A HA.
      rewrite <- (sumQ_map_scale_l (A:=Mask n)
        (k:=Qabs (F A))
        (f:=fun B => Qabs (G B)) MS).
      apply sumQ_map_ext; intros B HB. ring.
    }

    rewrite <- (sumQ_map_scale_l (A:=Mask n)
      (k:=sumQ (map (fun B => Qabs (G B)) MS))
      (f:=fun A => Qabs (F A)) MS).
    ring.
Qed.

End UnitMetric.