(* ============================================================ *)
(* Phase 6: Boolean Distance (relational / witness-based)        *)
(* ============================================================ *)


Require Import Cln_Full.
Require Import Cln_Grade.
Require Import Cln_finite_l1_submultiplicativity.


From Coq Require Import QArith.
From Coq Require Import QArith.Qabs.

Open Scope Q_scope.

Definition bool_dist_le {n} (F : MV n) (d : Q) : Prop :=
  exists g : Corner n -> bool,
    l1_norm (mv_sub F (embed g)) <= d.

Lemma bool_dist_embed : forall n (g : Corner n -> bool),
  bool_dist_le (embed g) 0.
Proof.
  intros n g. exists g.
  unfold mv_sub, l1_norm.
  apply Qle_of_Qeq.
  eapply Qeq_trans.
  - apply (sumQ_map_ext _ (fun _ : Mask n => 0%Q)).
    intros m Hm.
    assert (H : (embed g m - embed g m)%Q == 0%Q) by ring.
    rewrite H. rewrite Qabs_pos; [reflexivity | apply Qle_refl].
  - apply sumQ_map_const0.
Qed.

(* Triangle inequality for BoolDist under addition
Lemma bool_dist_add : forall n (F G : MV n) (dF dG : Q),
  bool_dist_le F dF -> bool_dist_le G dG ->
  bool_dist_le (mv_add F G) (dF + dG).
This is false 
*)

(* BoolDist under geometric product
Lemma bool_dist_gp : forall n (sq : Vector.t Q n) (F G : MV n) (dF dG : Q),
  bool_dist_le F dF -> bool_dist_le G dG ->
  bool_dist_le (mv_gp sq F G) (dF + dG + l1_norm F + l1_norm G).
This is false 
*)

Lemma l1_norm_ext : forall n (F G : MV n),
  (forall m, F m == G m) -> l1_norm F == l1_norm G.
Proof.
  intros n F G Hext.
  unfold l1_norm.
  apply sumQ_map_ext.
  intros m _. setoid_rewrite (Hext m). reflexivity.
Qed.

Lemma bool_dist_le_mono : forall n (F : MV n) (d d' : Q),
  bool_dist_le F d -> d <= d' -> bool_dist_le F d'.
Proof.
  intros n F d d' [g Hg] Hdd'.
  exists g. eapply Qle_trans; eauto.
Qed.

Lemma bool_dist_add_absorb : forall n (F G : MV n) (dF : Q),
  bool_dist_le F dF ->
  bool_dist_le (mv_add F G) (dF + l1_norm G).
Proof.
  intros n F G dF [gF HgF].
  exists gF.
  assert (Hext : l1_norm (mv_sub (mv_add F G) (embed gF)) ==
                 l1_norm (mv_add (mv_sub F (embed gF)) G)).
  { apply l1_norm_ext. intro m. unfold mv_sub, mv_add. ring. }
  eapply Qle_trans.
  - apply Qle_of_Qeq. exact Hext.
  - eapply Qle_trans.
    + apply l1_add_bound.
    + apply Qplus_le_compat.
      * exact HgF.
      * apply Qle_refl.
Qed.

Lemma bool_dist_add_absorb_r : forall n (F G : MV n) (dG : Q),
  bool_dist_le G dG ->
  bool_dist_le (mv_add F G) (l1_norm F + dG).
Proof.
  intros n F G dG [gG HgG].
  exists gG.
  assert (Hext : l1_norm (mv_sub (mv_add F G) (embed gG)) ==
                 l1_norm (mv_add F (mv_sub G (embed gG)))).
  { apply l1_norm_ext. intro m. unfold mv_sub, mv_add. ring. }
  eapply Qle_trans.
  - apply Qle_of_Qeq. exact Hext.
  - eapply Qle_trans.
    + apply l1_add_bound.
    + apply Qplus_le_compat.
      * apply Qle_refl.
      * exact HgG.
Qed.

(* ============================================================ *)
(* Witness-relative Boolean distance                             *)
(* ============================================================ *)
Definition bool_dist_wrt {n} (F : MV n) (g : Corner n -> bool) : Q :=
  l1_norm (mv_sub F (embed g)).

Lemma bool_dist_le_iff : forall n (F : MV n) (d : Q),
  bool_dist_le F d <-> exists g, bool_dist_wrt F g <= d.
Proof.
  intros n F d. unfold bool_dist_le, bool_dist_wrt. tauto.
Qed.

Lemma bool_dist_wrt_self : forall n (g : Corner n -> bool),
  bool_dist_wrt (embed g) g == 0.
Proof.
  intros n g. unfold bool_dist_wrt.
  assert (Hext : l1_norm (mv_sub (embed g) (embed g)) == l1_norm (@mv_zero n)).
  { apply l1_norm_ext. intro m. unfold mv_sub, mv_zero. ring. }
  eapply Qeq_trans.
  - exact Hext.
  - unfold l1_norm, mv_zero.
    eapply Qeq_trans.
    + apply sumQ_map_ext. intros m _.
      rewrite Qabs_pos; [reflexivity | apply Qle_refl].
    + apply sumQ_map_const0.
Qed.

(* ============================================================ *)
(* Addition: errors add                                          *)
(* ============================================================ *)

Lemma mv_sub_add_split : forall n (F G : MV n) (gF gG : Corner n -> bool) (m : Mask n),
  mv_sub (mv_add F G) (mv_add (embed gF) (embed gG)) m
  == mv_add (mv_sub F (embed gF)) (mv_sub G (embed gG)) m.
Proof.
  intros. unfold mv_sub, mv_add. ring.
Qed.

Lemma bool_dist_wrt_add : forall n (F G : MV n) (gF gG : Corner n -> bool),
  l1_norm (mv_sub (mv_add F G) (mv_add (embed gF) (embed gG)))
  <= bool_dist_wrt F gF + bool_dist_wrt G gG.
Proof.
  intros n F G gF gG.
  unfold bool_dist_wrt.
  eapply Qle_trans.
  - apply Qle_of_Qeq.
    apply l1_norm_ext. intro m. apply mv_sub_add_split.
  - apply l1_add_bound.
Qed.

(* ============================================================ *)
(* Geometric product: bilinear error decomposition               *)
(* ============================================================ *)

Lemma mv_gp_ext_r :
  forall n (sq : Vector.t Q n) (F G1 G2 : MV n) (U : Mask n),
    (forall m, G1 m == G2 m) ->
    mv_gp sq F G1 U == mv_gp sq F G2 U.
Proof.
  intros n sq F G1 G2 U H.
  unfold mv_gp.
  apply sumQ_map_ext; intros A _.
  apply sumQ_map_ext; intros B _.
  destruct (mask_eq_dec (basis_mul_mask A B) U); simpl; [| reflexivity].
  setoid_rewrite (H B).
  reflexivity.
Qed.

Lemma mv_gp_ext_l :
  forall n (sq : Vector.t Q n) (F1 F2 G : MV n) (U : Mask n),
    (forall m, F1 m == F2 m) ->
    mv_gp sq F1 G U == mv_gp sq F2 G U.
Proof.
  intros n sq F1 F2 G U H.
  unfold mv_gp.
  apply sumQ_map_ext; intros A _.
  apply sumQ_map_ext; intros B _.
  destruct (mask_eq_dec (basis_mul_mask A B) U); simpl; [| reflexivity].
  setoid_rewrite (H A).
  reflexivity.
Qed.

Lemma Qeq_minus_l : forall a b c : Q, a + b == c -> a == c - b.
Proof. intros a b c H. setoid_rewrite <- H. ring. Qed.

Lemma mv_gp_sub_r : forall n (sq : Vector.t Q n) (F G H : MV n) (m : Mask n),
  mv_gp sq F (mv_sub G H) m == (mv_gp sq F G m - mv_gp sq F H m)%Q.
Proof.
  intros n sq F G H m.
  apply Qeq_minus_l.
  eapply Qeq_trans.
  - symmetry. apply mv_gp_add_r.
  - apply mv_gp_ext_r. intro u. unfold mv_add, mv_sub. ring.
Qed.

Lemma mv_gp_sub_l : forall n (sq : Vector.t Q n) (F G H : MV n) (m : Mask n),
  mv_gp sq (mv_sub F G) H m == (mv_gp sq F H m - mv_gp sq G H m)%Q.
Proof.
  intros n sq F G H m.
  apply Qeq_minus_l.
  eapply Qeq_trans.
  - symmetry. apply mv_gp_add_l.
  - apply mv_gp_ext_l. intro u. unfold mv_add, mv_sub. ring.
Qed.

Lemma gp_error_split : forall n (sq : Vector.t Q n) (F G eF eG : MV n) (m : Mask n),
  mv_sub (mv_gp sq F G) (mv_gp sq eF eG) m
  == mv_add (mv_gp sq F (mv_sub G eG)) (mv_gp sq (mv_sub F eF) eG) m.
Proof.
  intros.
  unfold mv_sub at 1, mv_add.
  eapply Qeq_trans.
  - instantiate
      (1 := ((mv_gp sq F G m - mv_gp sq F eG m)
           + (mv_gp sq F eG m - mv_gp sq eF eG m))%Q).
    ring.
  - apply Qplus_comp.
    + symmetry. apply mv_gp_sub_r.
    + symmetry. apply mv_gp_sub_l.
Qed.

Lemma bool_dist_wrt_gp : forall n (sq : Vector.t Q n) (F G : MV n)
  (gF gG : Corner n -> bool),
  (forall i, Qabs (Vector.nth sq i) == 1) ->
  l1_norm (mv_sub (mv_gp sq F G) (mv_gp sq (embed gF) (embed gG)))
  <= l1_norm F * bool_dist_wrt G gG
   + bool_dist_wrt F gF * l1_norm (embed gG).
Proof.
  intros n sq F G gF gG Hsq.
  unfold bool_dist_wrt.
  eapply Qle_trans.
  - apply Qle_of_Qeq.
    apply l1_norm_ext. intro m. apply gp_error_split.
  - eapply Qle_trans.
    + apply l1_add_bound.
    + apply Qplus_le_compat.
      * apply l1_gp_submultiplicative; assumption.
      * apply l1_gp_submultiplicative; assumption.
Qed.

(*
-------------------------------------------------------------------------------
*)

Inductive BoolFormula (n : nat) : Type :=
  | BVar   : Fin.t n -> BoolFormula n            (* variable x_i *)
  | BConst : bool -> BoolFormula n               (* true / false *)
  | BAnd   : BoolFormula n -> BoolFormula n -> BoolFormula n
  | BOr    : BoolFormula n -> BoolFormula n -> BoolFormula n
  | BNot   : BoolFormula n -> BoolFormula n.

Arguments BVar {n}.
Arguments BConst {n}.
Arguments BAnd {n}.
Arguments BOr {n}.
Arguments BNot {n}.

Fixpoint eval_bf {n} (phi : BoolFormula n) (c : Corner n) : bool :=
  match phi with
  | BVar i     => match Vector.nth c i with Pos => true | Neg => false end
  | BConst b   => b
  | BAnd p q   => andb (eval_bf p c) (eval_bf q c)
  | BOr p q    => orb (eval_bf p c) (eval_bf q c)
  | BNot p     => negb (eval_bf p c)
  end.

Fixpoint translate {n} (phi : BoolFormula n) : GA_expr n :=
  match phi with
  | BVar i     => Mul (Scalar (1#2)) (Add (Scalar 1) (Basis i))
  | BConst true  => Scalar 1
  | BConst false => Scalar 0
  | BAnd p q   => Mul (translate p) (translate q)
  | BNot p     => Add (Scalar 1) (Mul (Scalar (-1)) (translate p))
  | BOr p q    => Add (Add (translate p) (translate q))
                      (Mul (Scalar (-1)) (Mul (translate p) (translate q)))
  end.

Theorem translate_correct : forall n (sq : Vector.t Q n) (phi : BoolFormula n),
  (forall i, Vector.nth sq i == 1) ->
  forall m, eval_expr sq (translate phi) m == embed (eval_bf phi) m.
Proof.
Admitted.