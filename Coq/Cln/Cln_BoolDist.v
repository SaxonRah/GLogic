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

(* Sum of chi(m, ·) over all corners = 2^n if m = empty, 0 otherwise *)
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
    + (* head bit is true: m = true :: m0 *)
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
    + (* head bit is false: m = false :: m0, with m0 <> empty *)
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

Lemma eval_extensionality :
  forall n (F G : MV n),
    (forall s : Corner n, eval F s == eval G s) ->
    forall m : Mask n, F m == G m.
Proof.
  intros n F G Heval m.
  (* Strategy: multiply eval(F,s) - eval(G,s) by chi(m,s) and sum over s.
     LHS sums to 0 (by Heval). RHS gives (F m - G m) * 2^n by orthogonality. *)
  assert (Hzero :
    sumQ (List.map (fun s => ((eval F s - eval G s) * chi' m s)%Q) (all_corners n)) == 0).
  { eapply Qeq_trans.
    - apply sumQ_map_ext; intros s _.
      assert (Hs : eval F s - eval G s == 0) by (rewrite (Heval s); ring).
      rewrite Hs. ring.
    - apply sumQ_map_const0. }

  (* Expand eval and distribute *)
  assert (Hexpand :
    sumQ (List.map (fun s => ((eval F s - eval G s) * chi' m s)%Q) (all_corners n))
    == sumQ (List.map (fun s =>
         (sumQ (List.map (fun m' => ((F m' - G m') * chi' m' s * chi' m s)%Q)
                         (all_masks n)))
       ) (all_corners n))).
  { apply sumQ_map_ext; intros s _.
    unfold eval.
    rewrite <- sumQ_map_sub.
    rewrite <- sumQ_map_scale_l.
    apply Qeq_trans with
      (sumQ (List.map (fun m' => ((F m' - G m') * (chi' m' s * chi' m s))%Q)
                       (all_masks n))).
    - apply sumQ_map_ext; intros m' _. ring.
    - apply sumQ_map_ext; intros m' _. ring. }

  (* Swap sums (Fubini) *)
  (* After swapping, inner sum over s gives corner_walsh_sum,
     which is 2^n * δ_{m',m}. Only m'=m survives. *)
  assert (Hswap :
    sumQ (List.map (fun s =>
      sumQ (List.map (fun m' => ((F m' - G m') * chi' m' s * chi' m s)%Q)
                     (all_masks n)))
      (all_corners n))
    == ((F m - G m) * pow2 n)%Q).
  { (* This requires Fubini + orthogonality. The full proof is:
       swap sums to get Σ_{m'} (F m' - G m') * Σ_s chi(m',s)*chi(m,s)
       = Σ_{m'} (F m' - G m') * corner_walsh_sum m' m
       = (F m - G m) * 2^n  *)
    admit. (* See proof strategy below *) }
    
  (*
  
  rewrite sumQ_swap.
  eapply Qeq_trans.
  - apply sumQ_map_ext; intros m' _.
    eapply Qeq_trans.
    + apply sumQ_map_ext; intros s _. ring. (* factor out (F m' - G m') *)
    + rewrite sumQ_map_scale_l. reflexivity.
  - (* Now: Σ_{m'} (F m' - G m') * corner_walsh_sum m' m *)
    eapply Qeq_trans.
    + apply sumQ_map_ext; intros m' _.
      unfold corner_walsh_sum.
      rewrite corner_walsh_sum_closed.
      destruct (mask_eq_dec m' m); ring.
    + apply sumQ_all_masks_pick_Q. (* Σ_{m'} [if m'=m then x else 0] = x *)
  
  *)

  (* From Hzero and Hexpand and Hswap: (F m - G m) * 2^n == 0 *)
  assert (Hprod : ((F m - G m) * pow2 n)%Q == 0).
  { rewrite <- Hswap, <- Hexpand. exact Hzero. }

  (* Since 2^n ≠ 0, F m == G m *)
  apply Qmult_integral in Hprod.
  destruct Hprod as [Hdiff | Hpow].
  - lra. (* or: apply Qeq... from Hdiff *)
  - exfalso. exact (pow2_nonzero n Hpow).
Qed.


(* ============================================================ *)
(* embed of constant true = scalar 1 (mv_one)                   *)
(* ============================================================ *)

Lemma embed_const_true : forall n (m : Mask n),
  embed (fun _ : Corner n => true) m == mv_one m.
Proof.
  intros n m.
  apply eval_extensionality.
  intro s.
  rewrite embed_correct.
  unfold eval, mv_one, basis.
  eapply Qeq_trans.
  2: { symmetry. apply sumQ_all_masks_pick_chi. (* Σ_m [if m=empty then 1 else 0]*chi(m,s) = chi(empty,s) = 1 *) }
  simpl. reflexivity.
Qed.


Theorem translate_correct : forall n (sq : Vector.t Q n) (phi : BoolFormula n),
  (forall i, Vector.nth sq i == 1) ->
  forall m, eval_expr sq (translate phi) m == embed (eval_bf phi) m.
Proof.
Admitted.