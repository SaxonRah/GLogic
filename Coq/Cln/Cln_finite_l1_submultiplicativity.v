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
Require Import Coq.Program.Equality.

Import ListNotations.

Open Scope Q_scope.
Set Implicit Arguments.

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

Lemma metric_factor_abs1_gen : forall n (sq : Vector.t Q n),
  (forall i : Fin.t n, Qabs (Vector.nth sq i) == 1) ->
  forall (A B : Mask n),
    Qabs (metric_factor sq A B) == 1.
Proof.
  induction n as [|p IH]; intros sq Hsq A B.
  - dependent destruction sq. dependent destruction A. dependent destruction B.
    unfold metric_factor. simpl. reflexivity.
  - dependent destruction sq. dependent destruction A. dependent destruction B.
    rewrite (@metric_factor_cons p h sq h0 h1 A B).
    rewrite Qabs_Qmult.
    rewrite (IH sq).
    + destruct (h0 && h1) eqn:Hab.
      * rewrite (Hsq Fin.F1). ring.
      * rewrite Qabs_pos; [ring | discriminate].
    + intros i. exact (Hsq (Fin.FS i)).
Qed.

(* Delta collapse for convolution (no sq dependency) *)
Lemma sumU_xor_delta_conv :
  forall n (A B : Mask n),
    sumQ (map (fun U =>
      if mask_eq_dec (mask_xor A B) U then 1%Q else 0%Q) (all_masks n))
    == 1%Q.
Proof.
  intros m A B.
  eapply Qeq_trans.
  2: { apply (sumQ_all_masks_pick (fun _ => 1%Q) (mask_xor A B)). }
  apply sumQ_map_ext. intros l Hm.
  destruct (mask_eq_dec (mask_xor A B) l) as [Hab|Hab].
  - subst l.
    destruct (mask_eq_dec (mask_xor A B) (mask_xor A B)) as [_|Hneq].
    + reflexivity.
    + exfalso; apply Hneq; reflexivity.
  - destruct (mask_eq_dec l (mask_xor A B)) as [Hba|_].
    + exfalso. apply Hab. now symmetry.
    + reflexivity.
Qed.

Lemma l1_conv_bound :
  forall n (F G : MV n),
    l1_norm (mv_conv F G) <= l1_norm F * l1_norm G.
Proof.
  intros n F G.
  unfold l1_norm, mv_conv.
  set (MS := all_masks n).

  set (term := fun (U A B : Mask n) =>
    if mask_eq_dec (mask_xor A B) U
    then (F A * G B)%Q else 0%Q).

  set (inner := fun (U : Mask n) =>
    sumQ (map (fun A : Mask n =>
      sumQ (map (fun B : Mask n => term U A B) MS)) MS)).

  (* Step 1: push abs through both finite sums *)
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

    (* Step 2: Expand |term| — no coefficient to worry about *)
    eapply Qeq_trans with
      (y :=
        sumQ (map (fun U : Mask n =>
          sumQ (map (fun A : Mask n =>
            sumQ (map (fun B : Mask n =>
              if mask_eq_dec (mask_xor A B) U
              then (Qabs (F A) * Qabs (G B))%Q
              else 0%Q) MS)) MS)) MS)).
    + apply sumQ_map_ext; intros U HU.
      apply sumQ_map_ext; intros A HA.
      apply sumQ_map_ext; intros B HB.
      unfold term.
      destruct (mask_eq_dec (mask_xor A B) U) as [Heq|Hneq].
      * rewrite Qabs_Qmult. apply Qeq_refl.
      * simpl. apply Qeq_refl.

    + (* Step 3: Fubini — swap U innermost *)
      rewrite (sumQ_fubini (A:=Mask n) (B:=Mask n)
        MS MS
        (fun U A =>
           sumQ (map (fun B =>
             if mask_eq_dec (mask_xor A B) U
             then (Qabs (F A) * Qabs (G B))%Q else 0%Q) MS))).

      eapply Qeq_trans with
        (y :=
          sumQ (map (fun A : Mask n =>
            sumQ (map (fun B : Mask n =>
              sumQ (map (fun U : Mask n =>
                if mask_eq_dec (mask_xor A B) U
                then (Qabs (F A) * Qabs (G B))%Q else 0%Q) MS)) MS)) MS)).
      { apply sumQ_map_ext; intros A HA.
        apply (sumQ_fubini MS MS
          (fun U B : Mask n =>
            if mask_eq_dec (mask_xor A B) U
            then (Qabs (F A) * Qabs (G B))%Q
            else 0%Q)). }

      (* Step 4: Collapse U-sum via delta *)
      eapply Qeq_trans with
        (y :=
          sumQ (map (fun A : Mask n =>
            sumQ (map (fun B : Mask n =>
              (Qabs (F A) * Qabs (G B))%Q *
              sumQ (map (fun U : Mask n =>
                if mask_eq_dec (mask_xor A B) U then 1%Q else 0%Q) MS)) MS)) MS)).
      { apply sumQ_map_ext; intros A HA.
        apply sumQ_map_ext; intros B HB.
        rewrite <- (sumQ_map_scale_l
          (Qabs (F A) * Qabs (G B))%Q
          (fun U : Mask n =>
             if mask_eq_dec (mask_xor A B) U then 1%Q else 0%Q)
          MS).
        apply sumQ_map_ext; intros U HU.
        destruct (mask_eq_dec (mask_xor A B) U); ring. }

      eapply Qeq_trans with
        (y :=
          sumQ (map (fun A : Mask n =>
            sumQ (map (fun B : Mask n =>
              Qabs (F A) * Qabs (G B) * 1%Q) MS)) MS)).
      { apply sumQ_map_ext; intros A HA.
        apply sumQ_map_ext; intros B HB.
        unfold MS.
        rewrite sumU_xor_delta_conv.
        apply Qeq_refl. }

      (* Step 5: Simplify *1 *)
      eapply Qeq_trans with
        (y :=
          sumQ (map (fun A : Mask n =>
            sumQ (map (fun B : Mask n =>
              (Qabs (F A) * Qabs (G B))%Q) MS)) MS)).
      { apply sumQ_map_ext; intros A HA.
        apply sumQ_map_ext; intros B HB.
        rewrite Qmult_1_r. apply Qeq_refl. }

      (* Step 6: Factor into ‖F‖₁ · ‖G‖₁ *)
      eapply Qeq_trans with
        (y :=
          sumQ (map (fun A : Mask n =>
            (Qabs (F A) * sumQ (map (fun B : Mask n => Qabs (G B)) MS))%Q) MS)).
      { apply sumQ_map_ext; intros A HA.
        rewrite <- (sumQ_map_scale_l (Qabs (F A))
          (fun B : Mask n => Qabs (G B)) MS).
        apply sumQ_map_ext; intros B HB. ring. }

      set (K := sumQ (map (fun B : Mask n => Qabs (G B)) MS)).

      eapply Qeq_trans with
        (y := sumQ (map (fun A : Mask n => (K * Qabs (F A))%Q) MS)).
      { apply sumQ_map_ext; intros A HA. unfold K. ring. }

      rewrite (sumQ_map_scale_l K (fun A : Mask n => Qabs (F A)) MS).
      rewrite Qmult_comm.
      apply Qeq_refl.
Qed.


Section UnitMetric.

Context {n : nat}.
Context (sq : Vector.t Q n).

Hypothesis sq_unit : forall (i : Fin.t n), Qabs (Vector.nth sq i) == 1.

Lemma metric_factor_abs1 : forall (A B : Mask n),
  Qabs (metric_factor sq A B) == 1.
Proof.
  exact (metric_factor_abs1_gen sq sq_unit).
Qed.

Lemma basis_mul_coeff_abs1 : forall (A B : Mask n),
  Qabs (basis_mul_coeff sq A B) == 1.
Proof.
  intros A B. unfold basis_mul_coeff.
  rewrite Qabs_Qmult.
  rewrite metric_factor_abs1.
  assert (H : Qabs (sgnQ (swaps_parity A B)) == 1).
  { destruct (swaps_parity A B); unfold sgnQ, Qabs; simpl; reflexivity. }
  rewrite H. ring.
Qed.

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
          *
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

Lemma l1_add_bound : forall n (F G : MV n),
  l1_norm (mv_add F G) <= l1_norm F + l1_norm G.
Proof.
  intros m F G.
  unfold l1_norm, mv_add.
  rewrite <- sumQ_map_add.
  apply sumQ_map_le.
  intros U _.
  apply Qabs_triangle.
Qed.

Lemma l1_gp_bound : forall (F G : MV n),
  l1_norm (mv_gp sq F G) <= l1_norm F * l1_norm G.
Proof.
  intros; apply l1_gp_submultiplicative.
Qed.


(* Static ℓ₁ bound from expression structure *)
Fixpoint l1_bound {n} (e : GA_expr n) : Q :=
  match e with
  | Basis _   => 1
  | Scalar c  => Qabs c
  | Cln_Grade.Add e1 e2 => l1_bound e1 + l1_bound e2
  | Cln_Grade.Mul e1 e2 => l1_bound e1 * l1_bound e2
  | Cln_Grade.Conv e1 e2 => l1_bound e1 * l1_bound e2
  end.

Lemma l1_norm_nonneg : forall (F : MV n),
  0 <= l1_norm F.
Proof.
  intro F. unfold l1_norm.
  induction (all_masks n) as [|m tl IH]; simpl.
  - apply Qle_refl.
  - apply Qle_trans with (y := (0 + 0)%Q).
    + ring_simplify. apply Qle_refl.
    + apply Qplus_le_compat.
      * apply Qabs_nonneg.
      * exact IH.
Qed.

Lemma l1_norm_basis : forall (i : Fin.t n),
  l1_norm (basis (mask_single i)) == 1.
Proof.
  intro i. unfold l1_norm, basis.
  eapply Qeq_trans.
  - apply (sumQ_map_ext _
      (fun m => if mask_eq_dec m (mask_single i) then 1 else 0)).
    intros m Hm.
    destruct (mask_eq_dec m (mask_single i)) as [Heq|Hneq].
    + rewrite Qabs_pos; [reflexivity | discriminate].
    + rewrite Qabs_pos; [reflexivity | apply Qle_refl].
  - apply sumQ_all_masks_pick.
Qed.

Lemma l1_norm_scale_one : forall (c : Q),
  l1_norm (mv_scale c (@mv_one n)) == Qabs c.
Proof.
  intro c. unfold l1_norm, mv_scale, mv_one, basis.
  eapply Qeq_trans.
  - apply (sumQ_map_ext _
      (fun m => if mask_eq_dec m (mask_empty (n:=n)) then Qabs c else 0)).
    intros m Hm.
    destruct (mask_eq_dec m (mask_empty (n:=n))) as [Heq|Hneq].
    + rewrite Qabs_Qmult.
      assert (H1 : Qabs 1 == 1) by (rewrite Qabs_pos; [reflexivity | discriminate]).
      rewrite H1. ring.
      
    + rewrite Qmult_0_r. rewrite Qabs_pos; [reflexivity | apply Qle_refl].
  - apply sumQ_all_masks_pick.
Qed.

Lemma Qmult_le_compat_nonneg : forall a b c d : Q,
  0 <= a -> 0 <= c -> a <= b -> c <= d -> a * c <= b * d.
Proof.
  intros a b c d Ha Hc Hab Hcd.
  apply Qle_trans with (y := (b * c)%Q).
  - destruct (Qle_lt_or_eq _ _ Hc) as [Hc'|Hc'].
    + apply Qmult_le_r; assumption.
    + setoid_rewrite <- Hc'. ring_simplify. apply Qle_refl.
  - destruct (Qle_lt_or_eq _ _ (Qle_trans _ _ _ Ha Hab)) as [Hb'|Hb'].
    + apply Qmult_le_l; assumption.
    + setoid_rewrite <- Hb'. ring_simplify. apply Qle_refl.
Qed.

Theorem l1_norm_eval_le :
  forall (e : GA_expr n),
    l1_norm (eval_expr sq e) <= l1_bound e.
Proof.
  intros e.
  induction e as [i | c | e1 IH1 e2 IH2 | e1 IH1 e2 IH2 | e1 IH1 e2 IH2]; simpl.
  - (* Basis *)
    apply Qle_of_Qeq. apply l1_norm_basis.
  - (* Scalar *)
    apply Qle_of_Qeq. apply l1_norm_scale_one.
  - (* Add *)
    eapply Qle_trans.
    + apply l1_add_bound.
    + apply Qplus_le_compat; assumption.
  - (* Mul *)
    eapply Qle_trans.
    + apply l1_gp_bound.
    + apply Qmult_le_compat_nonneg.
      * apply l1_norm_nonneg.
      * apply l1_norm_nonneg.
      * exact IH1.
      * exact IH2.
  - (* Conv *)
    eapply Qle_trans.
    + apply l1_conv_bound.
    + apply Qmult_le_compat_nonneg.
      * apply l1_norm_nonneg.
      * apply l1_norm_nonneg.
      * exact IH1.
      * exact IH2.
Qed.

End UnitMetric.

