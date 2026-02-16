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

Require Import Cln_Full.
Require Import Cln_Grade.

From Coq Require Import List Bool Arith Lia QArith.
From Coq Require Import QArith.Qabs.
From Coq Require Import Setoid Morphisms Ring.
Import ListNotations.

Open Scope Q_scope.
Set Implicit Arguments.

(* -------------------------------------------------------------------------
Definition Qabs (q:Q) : Q := if Qle_bool 0 q then q else (-q).

Lemma l1_add_bound : forall n (F G : MV n),
  l1_norm (mv_add F G) <= l1_norm F + l1_norm G.
Proof.
Admitted.

Lemma l1_gp_bound : forall n sq (F G : MV n),
  l1_norm (mv_gp n sq F G) <= l1_norm F * l1_norm G.
Proof.
Admitted.
------------------------------------------------------------------------- *)


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
  - apply Qle_refl.
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
Proof.
  intros x y z H.
  apply Qplus_le_compat.
  - apply Qle_refl.
  - exact H.
Qed.

Lemma Qabs_sumQ_le :
  forall xs : list Q,
    Qabs (sumQ xs) <= sumQ (map Qabs xs).
Proof.
  induction xs as [|x tl IH]; simpl.
  - apply Qle_refl.
  - eapply Qle_trans.
    + apply Qabs_triangle.
    + apply Qplus_le_compat_l. exact IH.
Qed.

Lemma Qabs_sumQ_map_le :
  forall (A:Type) (xs:list A) (f:A->Q),
    Qabs (sumQ (map f xs)) <= sumQ (map (fun a => Qabs (f a)) xs).
Proof.
  intros A xs f.
  eapply Qle_trans.
  - exact (Qabs_sumQ_le (map f xs)).
  - (* goal is now: sumQ (map Qabs (map f xs)) <= sumQ (map (fun a => Qabs (f a)) xs) *)
    rewrite map_map.       (* rewrites map Qabs (map f xs) to map (fun a => Qabs (f a)) xs *)
    apply Qle_refl.
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
  2: { apply (sumQ_all_masks_pick (fun _ => 1%Q) (basis_mul_mask A B)). }
  apply sumQ_map_ext. intros m Hm.
  destruct (mask_eq_dec (basis_mul_mask A B) m) as [Hab|Hab].
  - (* equal *)
    subst m.
    destruct (mask_eq_dec (basis_mul_mask A B) (basis_mul_mask A B)) as [_|Hneq].
    + reflexivity.
    + exfalso; apply Hneq; reflexivity.
  - (* not equal *)
    destruct (mask_eq_dec m (basis_mul_mask A B)) as [Hba|_].
    + exfalso. apply Hab. now symmetry.
    + reflexivity.
Qed.

(* ------------------------------------------------------------ *)
(* Main theorem                                                   *)
(* ------------------------------------------------------------ *)

Theorem l1_gp_submultiplicative :
  forall (F G : MV n),
    l1_norm (mv_gp sq F G) <= l1_norm F * l1_norm G.
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
    
    eapply Qeq_trans with
    (y :=
      sumQ (map (fun U : Mask n =>
        sumQ (map (fun A : Mask n =>
          sumQ (map (fun B : Mask n =>
            if mask_eq_dec (basis_mul_mask A B) U
            then (Qabs (F A) * Qabs (G B) * Qabs (basis_mul_coeff sq A B))%Q
            else 0%Q) MS)) MS)) MS)).
      + (* Goal 1 *)
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
        destruct (mask_eq_dec (basis_mul_mask A B) U) as [Heq|Hneq].
        * (* then-branch *)
          (* term reduces to F A * G B * c *)
          (* push abs through multiplication *)
          repeat rewrite Qabs_Qmult.
          (* now both sides are definitionally the same *)
          apply Qeq_refl.
        * (* else-branch *)
          (* term reduces to 0, so Qabs 0 = 0 *)
          (* after simpl, both sides are 0 *)
          simpl.
          apply Qeq_refl.
    
       + eapply Qeq_trans with
        (y :=
          sumQ
            (map
               (fun U : Mask n =>
                sumQ
                  (map
                     (fun A : Mask n =>
                      sumQ
                        (map
                           (fun B : Mask n =>
                            if mask_eq_dec (basis_mul_mask A B) U
                            then (Qabs (F A) * Qabs (G B))%Q
                            else 0%Q) MS)) MS)) MS)).
          * (* prove: old bigsum == new bigsum *)
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
            destruct (mask_eq_dec (basis_mul_mask A B) U) as [Heq|Hneq].
            -- (* then: use |coeff| = 1 *)
              rewrite basis_mul_coeff_abs1.
              (* now: Qabs(F A)*Qabs(G B)*1 == Qabs(F A)*Qabs(G B) *)
              (* avoid ring: just rewrite by 1 and reflexivity *)
              rewrite Qmult_1_r.
              apply Qeq_refl.
            -- (* else: both sides are 0 *)
              apply Qeq_refl.
          * (* continue from the simplified bigsum *)
            (* this is where your Fubini steps resume *)
            (* Fubini: swap Σ_U Σ_A *)
            rewrite (sumQ_fubini (A:=Mask n) (B:=Mask n)
              MS MS
              (fun U A =>
                 sumQ (map (fun B =>
                   if mask_eq_dec (basis_mul_mask A B) U
                   then (Qabs (F A) * Qabs (G B))%Q else 0%Q) MS))).

            (* Now: Σ_A Σ_U Σ_B ...  ->  Σ_A Σ_B Σ_U ... *)
            eapply Qeq_trans with
              (y :=
                sumQ (map (fun A : Mask n =>
                  sumQ (map (fun B : Mask n =>
                    sumQ (map (fun U : Mask n =>
                      if mask_eq_dec (basis_mul_mask A B) U
                      then (Qabs (F A) * Qabs (G B))%Q else 0%Q) MS)) MS)) MS)).
            apply sumQ_map_ext; intros b Hb.
            apply (sumQ_fubini MS MS
              (fun a B : Mask n =>
                if mask_eq_dec (basis_mul_mask b B) a
                then (Qabs (F b) * Qabs (G B))%Q
                else 0%Q)).
            
            eapply Qeq_trans with
              (y :=
                sumQ (map (fun A : Mask n =>
                  sumQ (map (fun B : Mask n =>
                    (Qabs (F A) * Qabs (G B))%Q *
                    sumQ (map (fun U : Mask n =>
                      if mask_eq_dec (basis_mul_mask A B) U then 1%Q else 0%Q) MS)) MS)) MS)).
            apply sumQ_map_ext; intros A HA.
            apply sumQ_map_ext; intros B HB.
            rewrite <- (sumQ_map_scale_l
              (Qabs (F A) * Qabs (G B))%Q
              (fun U : Mask n =>
                 if mask_eq_dec (basis_mul_mask A B) U then 1%Q else 0%Q)
              MS).

            apply sumQ_map_ext; intros U HU.
            destruct (mask_eq_dec (basis_mul_mask A B) U); ring.
            
              (* Collapse the U-sum to 1 using sumU_xor_delta *)
              eapply Qeq_trans with
                (y :=
                  sumQ
                    (map (fun A : Mask n =>
                      sumQ
                        (map (fun B : Mask n =>
                          Qabs (F A) * Qabs (G B) * 1%Q) MS)) MS)).
              apply sumQ_map_ext; intros A HA.
              apply sumQ_map_ext; intros B HB.
              unfold MS.
              rewrite sumU_xor_delta.
              apply Qeq_refl.

              (* simplify *1 *)
              eapply Qeq_trans with
                (y :=
                  sumQ
                    (map (fun A : Mask n =>
                      sumQ (map (fun B : Mask n =>
                        (Qabs (F A) * Qabs (G B))%Q) MS)) MS)).
              apply sumQ_map_ext; intros A HA.
              apply sumQ_map_ext; intros B HB.
              rewrite Qmult_1_r.
              apply Qeq_refl.

              (* Factor the B-sum: Σ_B |F A||G B| = |F A| * Σ_B |G B| *)
              eapply Qeq_trans with
                (y :=
                  sumQ
                    (map (fun A : Mask n =>
                      (Qabs (F A) * sumQ (map (fun B : Mask n => Qabs (G B)) MS))%Q) MS)).
              apply sumQ_map_ext; intros A HA.
              rewrite <- (sumQ_map_scale_l (Qabs (F A))
                (fun B : Mask n => Qabs (G B)) MS).
              apply sumQ_map_ext; intros B HB. ring.

              (* Factor the A-sum: Σ_A |F A| * K = K * Σ_A |F A| *)
              
              set (K := sumQ (map (fun B : Mask n => Qabs (G B)) MS)).

              (* rewrite pointwise: Qabs(F A) * K  ==  K * Qabs(F A) *)
              eapply Qeq_trans with
                (y := sumQ (map (fun A : Mask n => (K * Qabs (F A))%Q) MS)).
              apply sumQ_map_ext; intros A HA.
              unfold K.
              ring.  (* or: rewrite Qmult_comm; reflexivity *)
              (* now scale out K *)
              rewrite (sumQ_map_scale_l K (fun A : Mask n => Qabs (F A)) MS).
              rewrite Qmult_comm.
              apply Qeq_refl.
Qed.

End UnitMetric.