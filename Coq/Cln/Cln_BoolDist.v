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

(*
-------------------------------------------------------------------------------
*)

(* ============================================================ *)
(* Auxiliary: character product identity                          *)
(*   chi(A,s) * chi(B,s) == chi(A xor B, s)                     *)
(* ============================================================ *)

Require Import Coq.Program.Equality.

Lemma mask_xor_self : forall n (A : Mask n), mask_xor A A = mask_empty.
Proof.
  induction n; intro A; dependent destruction A; simpl.
  - reflexivity.
  - f_equal. + destruct h; reflexivity. + apply IHn.
Qed.

Lemma xorb_eq_false_implies_eq : forall a b : bool, xorb a b = false -> a = b.
Proof. destruct a, b; cbn; try discriminate; reflexivity. Qed.

Lemma VectorDef_nth_const :
  forall (A : Type) (a : A) n (i : Fin.t n),
    VectorDef.nth (Vector.const a n) i = a.
Proof.
  intros A a n.
  induction n as [| n IH]; intro i.
  - inversion i.
  - dependent destruction i.
    + cbn. reflexivity.
    + cbn. apply IH.
Qed.

Lemma mask_xor_eq_empty_iff : forall n (A B : Mask n),
  mask_xor A B = mask_empty -> A = B.
Proof.
  intros n A B H.
  apply Vector.eq_nth_iff; intro i.
  intros p2 Hp2; subst p2.

  pose proof (f_equal (fun v => VectorDef.nth v i) H) as Hi.
  unfold mask_xor, mask_empty in Hi.

  pose proof
    (@Vector.nth_map2 bool bool bool xorb n A B i i i eq_refl eq_refl) as Hmap.
  rewrite Hmap in Hi.

  (* turn RHS into false *)
  rewrite (VectorDef_nth_const bool false n i) in Hi.
  cbn in Hi.

  exact (xorb_eq_false_implies_eq _ _ Hi).
Qed.

(* Helper: Fubini for finite double sums *)
Lemma sumQ_swap :
  forall (A B : Type) (f : A -> B -> Q) (la : list A) (lb : list B),
    sumQ (List.map (fun a => sumQ (List.map (fun b => f a b) lb)) la)
    == sumQ (List.map (fun b => sumQ (List.map (fun a => f a b) la)) lb).
Proof.
  intros A B f la lb.
  induction la as [|a tla IH]; simpl.
  - symmetry. apply sumQ_map_const0.
  - rewrite IH. rewrite <- sumQ_map_add.
    apply sumQ_map_ext; intros b _. ring.
Qed.

Lemma chi_mul :
  forall n (A B : Mask n) (s : Corner n),
    (chi' A s * chi' B s)%Q == chi' (mask_xor A B) s.
Proof.
  induction n as [|n IH]; intros A B s.
  - dependent destruction A. dependent destruction B. dependent destruction s.
    simpl. ring.
  - dependent destruction A. dependent destruction B. dependent destruction s.
    cbn [mask_xor Vector.map2].
    destruct h, h0; simpl.
    + (* true, true => xorb true true = false *)
      (* Goal: sQ h1 * chi' A s * (sQ h1 * chi' B s) == 1 * chi' (mask_xor A B) s *)
      rewrite <- IH.
      (* Goal: ... == 1 * (chi' A s * chi' B s) *)
      eapply Qeq_trans with ((sQ h1 * sQ h1) * (chi' A s * chi' B s))%Q.
      * ring.
      * rewrite sQ_sq1. ring.
    + (* true, false *)
      rewrite <- IH. ring.
    + (* false, true *)
      rewrite <- IH. ring.
    + (* false, false *)
      rewrite <- IH. ring.
Qed.

(* ============================================================ *)
(* Character orthogonality over corners (dual Walsh sum)         *)
(*   Σ_s chi(m1,s) * chi(m2,s) = 2^n * δ_{m1,m2}              *)
(* ============================================================ *)

Definition corner_walsh_sum {n} (m1 m2 : Mask n) : Q :=
  sumQ (List.map (fun s => (chi' m1 s * chi' m2 s)%Q) (all_corners n)).

Lemma corner_walsh_sum_via_xor :
  forall n (m1 m2 : Mask n),
    corner_walsh_sum m1 m2
    == sumQ (List.map (fun s => chi' (mask_xor m1 m2) s) (all_corners n)).
Proof.
  intros n m1 m2. unfold corner_walsh_sum.
  apply sumQ_map_ext; intros s _. apply chi_mul.
Qed.

From Coq Require Import Vectors.Vector.
Import VectorNotations.

Lemma chi_corner_sum_empty :
  forall n,
    sumQ (List.map (fun s => chi' (mask_empty (n:=n)) s) (all_corners n))
    == pow2 n.
Proof.
  induction n as [|n IH].
  - simpl. ring.
  - simpl (all_corners (S n)).
    rewrite List.map_app, !List.map_map, sumQ_app.
    assert (HPos :
      sumQ (List.map (fun x => chi' (mask_empty (n:=S n)) (Pos :: x)) (all_corners n))
      == sumQ (List.map (fun s => chi' (mask_empty (n:=n)) s) (all_corners n))).
    { apply sumQ_map_ext; intros s _.
      unfold mask_empty. change (Vector.const false (S n))
        with (Vector.cons _ false _ (Vector.const false n)).
      rewrite chi_false_cons. reflexivity. }
    assert (HNeg :
      sumQ (List.map (fun x => chi' (mask_empty (n:=S n)) (Neg :: x)) (all_corners n))
      == sumQ (List.map (fun s => chi' (mask_empty (n:=n)) s) (all_corners n))).
    { apply sumQ_map_ext; intros s _.
      unfold mask_empty. change (Vector.const false (S n))
        with (Vector.cons _ false _ (Vector.const false n)).
      rewrite chi_false_cons. reflexivity. }
    rewrite HPos, HNeg, IH. simpl. ring.
Qed.

Lemma chi_corner_sum_nonempty :
  forall n (m : Mask n),
    m <> mask_empty ->
    sumQ (List.map (fun s => chi' m s) (all_corners n)) == 0.
Proof.
  induction n as [|n IH]; intros m Hne.
  - dependent destruction m. exfalso. apply Hne. reflexivity.
  - dependent destruction m.
    destruct h.
    + 
      simpl (all_corners (S n)).
      rewrite List.map_app, !List.map_map, sumQ_app.
      assert (HPos :
        sumQ (List.map (fun x => chi' (true :: m) (Pos :: x)) (all_corners n))
        == sumQ (List.map (fun s => chi' m s) (all_corners n))).
      { apply sumQ_map_ext; intros s _. rewrite chi_true_cons. simpl. ring. }
      assert (HNeg :
        sumQ (List.map (fun x => chi' (true :: m) (Neg :: x)) (all_corners n))
        == ((-1) * sumQ (List.map (fun s => chi' m s) (all_corners n)))%Q).
      { rewrite <- sumQ_map_scale_l.
        apply sumQ_map_ext; intros s _. rewrite chi_true_cons. simpl. ring. }
      rewrite HPos, HNeg. ring.
    + 
      assert (Hm : m <> mask_empty).
      { intro Heq. apply Hne.
        unfold mask_empty in *. rewrite Heq. reflexivity. }
      simpl (all_corners (S n)).
      rewrite List.map_app, !List.map_map, sumQ_app.
      assert (HPos :
        sumQ (List.map (fun x => chi' (false :: m) (Pos :: x)) (all_corners n))
        == sumQ (List.map (fun s => chi' m s) (all_corners n))).
      { apply sumQ_map_ext; intros s _. rewrite chi_false_cons. reflexivity. }
      assert (HNeg :
        sumQ (List.map (fun x => chi' (false :: m) (Neg :: x)) (all_corners n))
        == sumQ (List.map (fun s => chi' m s) (all_corners n))).
      { apply sumQ_map_ext; intros s _. rewrite chi_false_cons. reflexivity. }
      rewrite HPos, HNeg. rewrite (IH m Hm). ring.
Qed.

Lemma corner_walsh_sum_closed :
  forall n (m1 m2 : Mask n),
    corner_walsh_sum m1 m2
    == (if mask_eq_dec m1 m2 then pow2 n else 0).
Proof.
  intros n m1 m2.
  rewrite corner_walsh_sum_via_xor.
  destruct (mask_eq_dec m1 m2) as [Heq|Hneq].
  - subst m2. rewrite mask_xor_self. apply chi_corner_sum_empty.
  - apply chi_corner_sum_nonempty.
    intro Habs. apply Hneq. apply mask_xor_eq_empty_iff. exact Habs.
Qed.

(* ============================================================ *)
(* Eval extensionality: eval determines MV coefficients          *)
(* ============================================================ *)

Lemma Qminus_diag' (x : Q) : (x - x)%Q == 0%Q.
Proof.
  unfold Qminus.
  rewrite Qplus_opp_r.
  reflexivity.
Qed.

Lemma sumQ_all_masks_pick_Q :
  forall n (f : Mask n -> Q) (U : Mask n) (k : Q),
    sumQ (List.map (fun m => if mask_eq_dec m U then (f m * k)%Q else 0%Q) (all_masks n))
    == (f U * k)%Q.
Proof.
  intros n f U k.
  exact (@sumQ_all_masks_pick n (fun m => (f m * k)%Q) U).
Qed.

Lemma sumQ_all_masks_pick_const_Q :
  forall n (U : Mask n) (a : Q),
    sumQ (List.map (fun m => if mask_eq_dec m U then a else 0%Q) (all_masks n))
    == a.
Proof.
  intros n U a.
  exact (@sumQ_all_masks_pick n (fun _ => a) U).
Qed.

Lemma Qeq_plus_r : forall a b c : Q, a == b -> (a + c)%Q == (b + c)%Q.
Proof.
  intros a b c Hab.
  setoid_rewrite Hab.
  reflexivity.
Qed.

Lemma eval_extensionality :
  forall n (F G : MV n),
    (forall s : Corner n, eval F s == eval G s) ->
    forall m : Mask n, F m == G m.
Proof.
  intros n F G Heval m.
  assert (Hzero :
    sumQ (List.map (fun s => ((eval F s - eval G s) * chi' m s)%Q) (all_corners n)) == 0).
  { eapply Qeq_trans.
    -
      change 0%Q with (sumQ (List.map (fun _ : Corner n => 0%Q) (all_corners n))).
      refine (sumQ_map_ext
          (A := Corner n)
          (fun s => ((eval F s - eval G s) * chi' m s)%Q)
          (fun _ : Corner n => 0%Q)
          (all_corners n)
          _).
      intros s _Hin.
      setoid_rewrite (Heval s).
      change (eval G s - eval G s)%Q with (eval G s + (- eval G s))%Q.
      rewrite Qplus_opp_r.
      rewrite Qmult_0_l.
      reflexivity.

    - apply sumQ_map_const0.
  }
  
  assert (Hexpand :
    sumQ (List.map (fun s => ((eval F s - eval G s) * chi' m s)%Q) (all_corners n))
    ==
    sumQ (List.map (fun s =>
      sumQ (List.map (fun m' => ((F m' - G m') * chi' m' s * chi' m s)%Q)
                     (all_masks n)))
      (all_corners n))).
  { apply sumQ_map_ext; intros s _.
    unfold eval.
    rewrite <- sumQ_map_sub.
    rewrite <- sumQ_map_scale_r.
    apply sumQ_map_ext; intros m' _.
    ring.
  }

  assert (Hswap :
    sumQ (List.map (fun s =>
      sumQ (List.map (fun m' => ((F m' - G m') * chi' m' s * chi' m s)%Q)
                     (all_masks n)))
      (all_corners n))
    == ((F m - G m) * pow2 n)%Q).
  {
    rewrite sumQ_swap.

    eapply Qeq_trans.
    
    apply sumQ_map_ext; intros m' _.
    eapply Qeq_trans.
    - refine (sumQ_map_ext
              (A := Corner n)
              (fun s => ((F m' - G m') * chi' m' s * chi' m s)%Q)
              (fun s => ((F m' - G m') * (chi' m' s * chi' m s))%Q)
              (all_corners n)
              _).
      intros s _Hin.
      rewrite Qmult_assoc.
      reflexivity.

    - rewrite <- sumQ_map_scale_l.
      reflexivity.
    -
      eapply Qeq_trans.
      + refine (sumQ_map_ext
                (A := Mask n)
                (fun m' =>
                   ((F m' - G m') *
                     sumQ (List.map (fun s : Corner n => (chi' m' s * chi' m s)%Q)
                                    (all_corners n)))%Q)
                (fun m' =>
                   ((F m' - G m') *
                     (if mask_eq_dec m' m then pow2 n else 0))%Q)
                (all_masks n)
                _).
        intros m' _Hin.
        f_equal.
        unfold corner_walsh_sum.
        
        change (sumQ (List.map (fun s : Corner n => (chi' m' s * chi' m s)%Q) (all_corners n)))
          with (corner_walsh_sum m' m).

        rewrite corner_walsh_sum_closed.
        reflexivity.
        
      +
        eapply Qeq_trans.
        *
          refine (sumQ_map_ext
                    (A := Mask n)
                    (fun m' => ((F m' - G m') * (if mask_eq_dec m' m then pow2 n else 0))%Q)
                    (fun m' => (if mask_eq_dec m' m then ((F m' - G m') * pow2 n)%Q else 0%Q))
                    (all_masks n)
                    _).
          intros m' _Hin.
          destruct (mask_eq_dec m' m) as [H|H].
          -- subst. ring.
          -- ring.
        * exact (@sumQ_all_masks_pick_Q n (fun x => (F x - G x)%Q) m (pow2 n)).
  }

  assert (Hprod : ((F m - G m) * pow2 n)%Q == 0).
  { rewrite <- Hswap, <- Hexpand. exact Hzero. }

  apply Qmult_integral in Hprod.
  destruct Hprod as [Hdiff | Hpow].
  -
    change (F m - G m)%Q with (F m + (- G m))%Q in Hdiff.
    eapply Qeq_trans.
    *
      eapply Qeq_trans.
      + rewrite <- (Qplus_0_r (F m)). reflexivity.
      +
        rewrite <- (Qplus_opp_r (G m)).
        repeat rewrite Qplus_assoc.
        rewrite <- Qplus_assoc.
        rewrite Qplus_opp_r.
        rewrite Qplus_0_r.
        reflexivity.
    *
      change (F m - G m)%Q with (F m + (- G m))%Q in Hdiff.
      pose proof (Qeq_plus_r _ _ (G m) Hdiff) as Hadd.
      rewrite <- Qplus_assoc in Hadd.
      rewrite (Qplus_comm (- G m) (G m)) in Hadd.
      rewrite Qplus_opp_r in Hadd.
      rewrite Qplus_0_r in Hadd.
      rewrite Qplus_0_l in Hadd.
      exact Hadd.
  - exfalso. exact (pow2_nonzero n Hpow).
Qed.


(* ============================================================ *)
(* embed of constant true = scalar 1 (mv_one)                   *)
(* ============================================================ *)

Lemma chi_mask_empty :
  forall n (s : Corner n),
    chi' (mask_empty (n:=n)) s == 1%Q.
Proof.
  induction n as [|n IH].
  - intro s. simpl. ring.
  - intro s.
    dependent destruction s.
    unfold mask_empty.
    change (Vector.const false (S n))
      with (Vector.cons _ false _ (Vector.const false n)).
    rewrite (chi_false_cons (Vector.const false n) s h).
    exact (IH s).
Qed.


Lemma sumQ_all_masks_pick_chi :
  forall n (s : Corner n),
    sumQ (List.map (fun m0 : Mask n =>
      ((if mask_eq_dec m0 mask_empty then 1%Q else 0%Q) * chi' m0 s)%Q)
      (all_masks n))
    == 1%Q.
Proof.
  intros n s.

  (* First, rewrite the integrand into the "pick_Q" shape:
       (if m0=empty then (chi' m0 s * 1) else 0)
     rather than (if ... then 1 else 0) * chi' m0 s. *)
  eapply Qeq_trans.
  - refine (sumQ_map_ext
              (A := Mask n)
              (fun m0 =>
                 ((if mask_eq_dec m0 mask_empty then 1%Q else 0%Q) * chi' m0 s)%Q)
              (fun m0 =>
                 (if mask_eq_dec m0 mask_empty then (chi' m0 s * 1%Q)%Q else 0%Q))
              (all_masks n)
              _).
    intros m0 _Hin.
    destruct (mask_eq_dec m0 mask_empty) as [H|H].
    + subst. ring.
    + (* (0 * chi) == 0 *)
      ring.

  - (* Now apply the general pick lemma with f := chi' _ s and k := 1 *)
    eapply Qeq_trans.
    + exact (@sumQ_all_masks_pick_Q n (fun x : Mask n => (chi' x s)%Q) mask_empty 1%Q).
    + rewrite (chi_mask_empty n s).
      ring.
Qed.

Lemma embed_const_true : forall n (m : Mask n),
  embed (fun _ : Corner n => true) m == mv_one m.
Proof.
  intros n m.
  apply eval_extensionality.
  intro s.
  rewrite embed_correct.
  unfold eval, mv_one, basis.
  eapply Qeq_trans.
  2: { symmetry. apply sumQ_all_masks_pick_chi. }
  simpl. reflexivity.
Qed.

Lemma eval_basis : forall n (M : Mask n) (s : Corner n),
  eval (basis M) s == chi' M s.
Proof.
Admitted.
(*  intros n M s.
  unfold eval, basis.
  (* eval(basis M) s = sum_m (if m=M then 1 else 0) * chi' m s *)
  eapply Qeq_trans.
  - apply sumQ_map_ext; intros m _.
    destruct (mask_eq_dec m M); ring.
  - (* pick lemma from Cln_Full *)
    (* shape: sumQ (map (fun m => if m=M then (chi' m s) else 0) all_masks) == chi' M s *)
    exact (@sumQ_all_masks_pick n (fun m => (chi' m s)%Q) M).
Qed. *)

Lemma eval_mv_one : forall n (s : Corner n),
  eval (@mv_one n) s == 1%Q.
Proof.
Admitted.
(*  intros n s.
  unfold mv_one.
  (* mv_one = basis empty mask in your development *)
  (* if mv_one is defined differently, adapt this line *)
  unfold mv_one, basis.
  (* use eval_basis with M = mask_empty *)
  rewrite eval_basis.
  (* chi' empty s == 1 *)
  (* you already proved chi_mask_empty in Cln_BoolDist.v *)
  apply chi_mask_empty.
Qed. *)

Lemma translate_eval_correct :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n),
    (forall i, Vector.nth sq i == 1) ->
    forall s : Corner n,
      eval (eval_expr sq (translate phi)) s == bQ (eval_bf phi s).
Proof.
  (* main induction; see below *)
Admitted.

Lemma eval_gp_embed :
  forall n (sq : Vector.t Q n) (f g : Corner n -> bool) (s : Corner n),
    (forall i, Vector.nth sq i == 1) ->
    eval (mv_gp sq (embed f) (embed g)) s == (bQ (f s) * bQ (g s))%Q.
Proof.
Admitted.

Lemma Pi_gp_delta :
  forall n (sq : Vector.t Q n) (a b s : Corner n),
    (forall i, Vector.nth sq i == 1) ->
    eval (mv_gp sq (Pi a) (Pi b)) s
    == (if corner_eqb a b then if corner_eqb a s then 1%Q else 0%Q else 0%Q).
Proof.
Admitted.


Theorem translate_correct :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n),
    (forall i, Vector.nth sq i == 1) ->
    forall m, eval_expr sq (translate phi) m == embed (eval_bf phi) m.
Proof.
  intros n sq phi Hsq m.
  (* Use eval_extensionality with F = eval_expr..., G = embed... *)
  refine (eval_extensionality n
            (eval_expr sq (translate phi))
            (embed (eval_bf phi))
            _ m).
  intro s.
  rewrite (translate_eval_correct n sq phi Hsq s).
  rewrite embed_correct.
  reflexivity.
Qed.

Lemma translate_eval_correct_final :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n),
    (forall i, Vector.nth sq i == 1) ->
    forall s : Corner n,
      eval (eval_expr sq (translate phi)) s == bQ (eval_bf phi s).
Proof.
  intros n sq phi Hsq.
  induction phi; intro s; cbn [translate eval_bf eval_expr].
  - (* BVar *)
    (* rewrite gp with left scalar into mv_scale, then eval *)
    (* finish with (1/2)(1+sQ) computation *)
  - (* BConst b *)
    destruct b; cbn.
    + (* true *) rewrite <- (eval_mv_one n s).
      (* use embed_correct if you prefer, or direct eval of Scalar 1 *)
      (* Scalar 1 evaluates to 1 *)
    + (* false *) (* Scalar 0 evaluates to 0 *)
  - (* BAnd *)
    (* use IHphi1, IHphi2, and eval_gp_embed *)
    rewrite (eval_gp_embed n sq (eval_bf phi1) (eval_bf phi2) s Hsq).
    rewrite IHphi1 by assumption.
    rewrite IHphi2 by assumption.
    (* bQ(andb ..) = product *)
    destruct (eval_bf phi1 s), (eval_bf phi2 s); cbn; ring.
  - (* BOr *)
    (* use identity bQ(or) = x + y - x*y, plus eval_gp_embed *)
    (* and IHs; then case split on booleans; ring *)
  - (* BNot *)
    (* identity bQ(negb b) = 1 - bQ(b) *)
Qed.

