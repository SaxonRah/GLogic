Require Import Cln_Full.
Require Import Cln_Grade.
Require Import Cln_BoolDist.

From Coq Require Import List QArith.
From Coq Require Import Lia.
Require Import Coq.Program.Equality.

Import ListNotations.

(* ------------------------------------------------------------------ *)
(* Small list helpers (avoid depending on library lemma names)         *)
(* ------------------------------------------------------------------ *)

Lemma length_filter_le :
  forall (A : Type) (p : A -> bool) (l : list A),
    (length (filter p l) <= length l)%nat.
Proof.
  intros A p l; induction l as [|a tl IH]; simpl; auto.
  destruct (p a); simpl; lia.
Qed.

Lemma NoDup_filter :
  forall (A : Type) (p : A -> bool) (l : list A),
    NoDup l -> NoDup (filter p l).
Proof.
  intros A p l Hnd.
  induction Hnd as [|a tl Hnotin Hnd IH]; simpl.
  - constructor.
  - destruct (p a) eqn:Ha.
    + constructor.
      * intro Hin.
        apply filter_In in Hin as [Hin _].
        contradiction.
      * exact IH.
    + exact IH.
Qed.

Lemma NoDup_incl_length :
  forall (A : Type) (l1 l2 : list A),
    NoDup l1 -> incl l1 l2 -> (length l1 <= length l2)%nat.
Proof.
  intros A l1; induction l1 as [|a tl IH];
    intros l2 Hnd Hincl; simpl.
  - lia.
  - inversion Hnd as [|a' tl' Hnotin Hndtl]; subst.
    assert (Hin : In a l2) by (apply Hincl; left; reflexivity).
    apply in_split in Hin as [lL [lR ->]].
    simpl.
    assert (Hincl' : incl tl (lL ++ lR)).
    {
      intros x Hx.
      specialize (Hincl x (or_intror Hx)).
      apply in_app_or in Hincl as [HinL|HinR].
      - apply in_or_app. left; exact HinL.
      - simpl in HinR. destruct HinR as [HxEq|HinR].
        + subst. exfalso. apply Hnotin. exact Hx.
        + apply in_or_app. right; exact HinR.
    }
    specialize (IH (lL ++ lR) Hndtl Hincl').

    rewrite length_app in IH.
    rewrite length_app. simpl.
    lia.
Qed.

Lemma filter_ext_in :
  forall (A : Type) (l : list A) (p q : A -> bool),
    (forall x, In x l -> p x = q x) ->
    filter p l = filter q l.
Proof.
  intros A l; induction l as [|a tl IH]; intros p q Hpq; simpl; auto.
  rewrite (Hpq a (or_introl eq_refl)).
  destruct (q a) eqn:Hqa; simpl.
  - f_equal.
    rewrite (IH p q (fun x Hx => Hpq x (or_intror Hx))).
    reflexivity.
  - rewrite (IH p q (fun x Hx => Hpq x (or_intror Hx))).
    reflexivity.
Qed.

Lemma all_masks_length_pow2 :
  forall n, length (all_masks n) = Nat.pow 2 n.
Proof.
  induction n; simpl.
  - reflexivity.
  - rewrite length_app.
    repeat rewrite length_map.
    set (k := length (all_masks n)).
    change (k + k = 2 ^ n + (2 ^ n + 0))%nat.
    subst k.
    rewrite IHn.
    lia.
Qed.

Lemma Qeq_bool_10 : Qeq_bool 1 0 = false.
Proof. vm_compute. reflexivity. Qed.

Lemma Qeq_bool_00 : Qeq_bool 0 0 = true.
Proof. vm_compute. reflexivity. Qed.

(* Count nonzero coefficients *)
Definition support_size {n} (F : MV n) : nat :=
  length (List.filter (fun m => negb (Qeq_bool (F m) 0))
                 (all_masks n)).

Lemma support_size_ext :
  forall n (F G : MV n),
    (forall m, F m == G m) ->
    support_size F = support_size G.
Proof.
  intros n F G Heq.
  unfold support_size.
  apply f_equal.
  apply filter_ext_in.
  intros m _.
  destruct (Qeq_bool (F m) 0) eqn:Hf;
  destruct (Qeq_bool (G m) 0) eqn:Hg; try reflexivity.
  - (* Hf = true, Hg = false *)
    exfalso.
    (* turn Hg into ~(G m == 0) *)
    pose proof (Qeq_bool_false_neq Hg) as Hneq.
    apply Hneq.
    (* prove G m == 0 using Heq and Hf *)
    apply Qeq_trans with (y := F m).
    + apply Qeq_sym. apply Heq.
    + apply Qeq_bool_eq. exact Hf.
  - (* Hf = false, Hg = true *)
    exfalso.
    pose proof (Qeq_bool_false_neq Hf) as Hneq.
    apply Hneq.
    apply Qeq_trans with (y := G m).
    + apply Heq.
    + apply Qeq_bool_eq. exact Hg.
Qed.

        (* Support cardinality *)


(* The support set as a list *)
Definition support_list {n} (F : MV n) : list (Mask n) :=
  filter (fun m => negb (Qeq_bool (F m) 0)) (all_masks n).

        (* Lemma Block 1 *)

Lemma support_size_le_2n : forall n (F : MV n),
  (support_size F <= Nat.pow 2 n)%nat.
Proof.
  intros n F.
  unfold support_size.
  eapply Nat.le_trans.
  - apply length_filter_le.
  - rewrite all_masks_length_pow2.
    lia.
Qed.

Lemma support_size_zero : forall n,
  support_size (@mv_zero n) = 0%nat.
Proof.
  intro n.
  unfold support_size, mv_zero.
  rewrite Qeq_bool_00.
  simpl.
  induction (all_masks n); simpl; auto.
Qed.

Lemma filter_all_false_nil :
  forall (A : Type) (p : A -> bool) (l : list A),
    (forall x, In x l -> p x = false) ->
    filter p l = [].
Proof.
  intros A p l Hall.
  induction l as [|h t IH]; simpl; auto.
  assert (Hp : p h = false) by (apply Hall; left; reflexivity).
  rewrite Hp.
  apply IH.
  intros x Hinx.
  apply Hall. right. exact Hinx.
Qed.

Lemma filter_singleton :
  forall (A : Type) (p : A -> bool) (l : list A) (a : A),
    NoDup l ->
    In a l ->
    p a = true ->
    (forall x, In x l -> x <> a -> p x = false) ->
    filter p l = [a].
Proof.
  intros A p l.
  induction l as [|h t IH]; intros a Hnd Hin Hp Hall; simpl in *.
  - contradiction.
  - inversion Hnd as [|h' t' Hnotin Hndt]; subst.
    simpl in Hin. destruct Hin as [Hin|Hin].
    + subst h.
      rewrite Hp. f_equal.
      apply filter_all_false_nil.
      intros x Hinx.
      apply Hall.
      * right; exact Hinx.
      * intro Heq; subst x. exact (Hnotin Hinx).
    + (* a is in tail *)
      assert (Hha : h <> a) by (intro Heq; subst h; contradiction).
      assert (Hph : p h = false).
      { apply Hall; [left; reflexivity | exact Hha]. }
      rewrite Hph.
      apply IH.
      * exact Hndt.
      * exact Hin.
      * exact Hp.
      * intros x Hinx Hxa.
        apply Hall; [right; exact Hinx | exact Hxa].
Qed.

Lemma support_size_basis : forall n (i : Fin.t n),
  support_size (basis (mask_single i)) = 1%nat.
Proof.
  intros n i.
  unfold support_size.
  set (m0 := mask_single i).
  set (p := fun m : Mask n => negb (Qeq_bool (basis m0 m) 0)).

  assert (Hnodup : NoDup (all_masks n)) by apply all_masks_nodup.
  assert (Hin : In m0 (all_masks n)) by apply all_masks_complete.

  assert (Hp0 : p m0 = true).
  {
    unfold p, basis.
    destruct (mask_eq_dec m0 m0) as [_|Hc]; [|contradiction].
    rewrite Qeq_bool_10. reflexivity.
  }

  assert (Hall : forall x, In x (all_masks n) -> x <> m0 -> p x = false).
  {
    intros x Hinx Hneq.
    unfold p, basis.
    destruct (mask_eq_dec x m0) as [Heq|Hne].
    - contradiction.
    - (* basis is 0 on x, so Qeq_bool 0 0 = true, negb -> false *)
      rewrite Qeq_bool_00. reflexivity.
  }

  (* now the filter is exactly [m0], so length is 1 *)
  rewrite (filter_singleton (Mask n) p (all_masks n) m0 Hnodup Hin Hp0 Hall).
  simpl. reflexivity.
Qed.

Lemma support_size_mv_one : forall n,
  support_size (@mv_one n) = 1%nat.
Proof.
  intro n.
  unfold mv_one, support_size.
  set (m0 := (@mask_empty n)).
  set (p := fun m : Mask n => negb (Qeq_bool (basis m0 m) 0)).

  assert (Hnodup : NoDup (all_masks n)) by apply all_masks_nodup.
  assert (Hin : In m0 (all_masks n)) by apply all_masks_complete.

  assert (Hp0 : p m0 = true).
  {
    unfold p, basis.
    destruct (mask_eq_dec m0 m0) as [_|Hc]; [|contradiction].
    rewrite Qeq_bool_10. reflexivity.
  }

  assert (Hall : forall x, In x (all_masks n) -> x <> m0 -> p x = false).
  {
    intros x Hinx Hneq.
    unfold p, basis.
    destruct (mask_eq_dec x m0) as [Heq|Hne].
    - contradiction.
    - rewrite Qeq_bool_00. reflexivity.
  }

  rewrite (filter_singleton (Mask n) p (all_masks n) m0 Hnodup Hin Hp0 Hall).
  simpl. reflexivity.
Qed.

Lemma support_size_scale : forall n (c : Q) (F : MV n),
  ~(c == 0) ->
  support_size (mv_scale c F) = support_size F.
Proof.
  intros n c F Hc.
  unfold support_size.
  apply f_equal.
  apply filter_ext_in.
  intros m _.
  unfold mv_scale.
  destruct (Qeq_bool (F m) 0) eqn:Hfm.
  - apply Qeq_bool_eq in Hfm.
    rewrite Hfm. rewrite Qmult_0_r.
    rewrite Qeq_bool_00. reflexivity.
  - assert (Hnz : ~(F m == 0)) by (exact (Qeq_bool_false_neq Hfm)).
    destruct (Qeq_bool (c * F m) 0) eqn:Hcm; try reflexivity.
    apply Qeq_bool_eq in Hcm.
    exfalso.
    apply Hnz.
    apply Qmult_integral in Hcm as [Hc0|Hf0]; [contradiction|exact Hf0].
Qed.

        (* Lemma Block 2 *)

(* Add can at most double support *)
Lemma support_size_add : forall n (F G : MV n),
  (support_size (mv_add F G) <= support_size F + support_size G)%nat.
Proof.
  intros n F G.
  unfold support_size.
  set (SF := filter (fun m : Mask n => negb (Qeq_bool (F m) 0)) (all_masks n)).
  set (SG := filter (fun m : Mask n => negb (Qeq_bool (G m) 0)) (all_masks n)).
  set (SH := filter (fun m : Mask n => negb (Qeq_bool (mv_add F G m) 0)) (all_masks n)).

  assert (Hincl : incl SH (SF ++ SG)).
  {
    intros m Hin.
    apply filter_In in Hin as [HinAll Hp].
    assert (Hsupp : supp (mv_add F G) m).
    {
      unfold supp.
      destruct (Qeq_bool (mv_add F G m) 0) eqn:Heq; simpl in Hp; try discriminate.
      apply (Qeq_bool_false_neq Heq).
    }
    destruct (supp_add n F G m Hsupp) as [HsF|HsG].
    - apply in_or_app. left.
      apply filter_In. split; [exact HinAll|].
      destruct (Qeq_bool (F m) 0) eqn:Heq; simpl.
      * exfalso. apply HsF. apply Qeq_bool_eq. exact Heq.
      * reflexivity.
    - apply in_or_app. right.
      apply filter_In. split; [exact HinAll|].
      destruct (Qeq_bool (G m) 0) eqn:Heq; simpl.
      * exfalso. apply HsG. apply Qeq_bool_eq. exact Heq.
      * reflexivity.
  }

  assert (Hnd : NoDup SH) by (apply NoDup_filter, all_masks_nodup).
  eapply Nat.le_trans.
  - eapply NoDup_incl_length; eauto.
  - rewrite length_app. lia.
Qed.

(* Conv: support of F⊙G ⊆ {A⊕B : A ∈ supp(F), B ∈ supp(G)} *)
(* So |supp(F⊙G)| ≤ |supp(F)| * |supp(G)| *)
Lemma support_size_conv : forall n (F G : MV n),
  (support_size (mv_conv F G) <= support_size F * support_size G)%nat.
Proof.
  intros n F G.
  unfold support_size.
  set (SF := filter (fun m : Mask n => negb (Qeq_bool (F m) 0)) (all_masks n)).
  set (SG := filter (fun m : Mask n => negb (Qeq_bool (G m) 0)) (all_masks n)).
  set (SH := filter (fun m : Mask n => negb (Qeq_bool (mv_conv F G m) 0)) (all_masks n)).

  (* all xor-combinations of support lists *)
  set (xor_sumset :=
    concat (map (fun A => map (fun B => mask_xor A B) SG) SF)).

  (* length of the xor_sumset list is |SF| * |SG| *)
  assert (Hlen : length xor_sumset = (length SF * length SG)%nat).
  {
    unfold xor_sumset.
    induction SF as [|A tl IH]; simpl.
    - lia.
    - rewrite length_app, length_map, IH. lia.
  }

  (* SH ⊆ xor_sumset *)
  assert (Hincl : incl SH xor_sumset).
  {
    intros U HinU.
    apply filter_In in HinU as [HinAll HpU].
    assert (Hsupp : supp (mv_conv F G) U).
    {
      unfold supp.
      destruct (Qeq_bool (mv_conv F G U) 0) eqn:Heq; simpl in HpU; try discriminate.
      apply (Qeq_bool_false_neq Heq).
    }
    destruct (support_conv_subset_xor n F G U Hsupp)
      as [A [B [HsA [HsB Hxor]]]].
    subst U.

    assert (HinA : In A SF).
    {
      unfold SF.
      apply filter_In. split.
      - apply all_masks_complete.
      - destruct (Qeq_bool (F A) 0) eqn:Heq; simpl.
        + exfalso. apply HsA. apply Qeq_bool_eq. exact Heq.
        + reflexivity.
    }
    assert (HinB : In B SG).
    {
      unfold SG.
      apply filter_In. split.
      - apply all_masks_complete.
      - destruct (Qeq_bool (G B) 0) eqn:Heq; simpl.
        + exfalso. apply HsB. apply Qeq_bool_eq. exact Heq.
        + reflexivity.
    }

    unfold xor_sumset.
    apply in_concat.
    exists (map (fun B0 : Mask n => mask_xor A B0) SG).
    split.
    - apply in_map_iff. exists A. split; [reflexivity | exact HinA].
    - apply in_map. exact HinB.
  }

  assert (Hnd : NoDup SH) by (apply NoDup_filter, all_masks_nodup).
  eapply Nat.le_trans.
  - eapply NoDup_incl_length; eauto.
  - rewrite Hlen. unfold SF, SG.
    lia.
Qed.

        (* Lemma Block 3 *)

(* Static support size bound from expression structure *)

(*
OLD support_size_bound
Fixpoint support_size_bound {n} (e : GA_expr n) : nat :=
  match e with
  | Basis _    => 1
  | Scalar _   => 1
  | Cln_Grade.Add e1 e2  => support_size_bound e1 + support_size_bound e2
  | Mul _ _    => Nat.pow 2 n  (* worst case; keeps Block 3 mechanical *)
  | Conv e1 e2 => support_size_bound e1 * support_size_bound e2
  end.
*)
Fixpoint support_size_bound {n} (e : GA_expr n) : nat :=
  match e with
  | Basis _    => 1
  | Scalar _   => 1
  | Cln_Grade.Add e1 e2  => support_size_bound e1 + support_size_bound e2
  | Mul e1 e2  =>
      match e1, e2 with
      | Scalar _, _ => support_size_bound e2
      | _, Scalar _ => support_size_bound e1
      | _, _        => Nat.pow 2 n
      end
  | Conv e1 e2 => support_size_bound e1 * support_size_bound e2
  end.

Lemma eval_mul_scalar_l :
  forall n (sq : Vector.t Q n) (q : Q) (e : GA_expr n),
    forall m,
      eval_expr sq (Mul (Scalar q) e) m
      ==
      mv_scale q (eval_expr sq e) m.
Proof.
  intros n sq q e m.
  simpl.  (* eval_expr Mul / Scalar *)
  rewrite (@mv_gp_scale_l n sq q (@mv_one n) (eval_expr sq e) m).
  unfold mv_scale.
  rewrite (@mv_gp_one_l n sq (eval_expr sq e) m).
  ring.
Qed.

Lemma eval_mul_scalar_r :
  forall n (sq : Vector.t Q n) (q : Q) (e : GA_expr n),
    forall m,
      eval_expr sq (Mul e (Scalar q)) m
      ==
      mv_scale q (eval_expr sq e) m.
Proof.
  intros n sq q e m.
  simpl.
  rewrite (@mv_gp_scale_r n sq q (eval_expr sq e) (@mv_one n) m).
  unfold mv_scale.
  rewrite (@mv_gp_one_r n sq (eval_expr sq e) m).
  ring.
Qed.

Lemma support_size_scalar_mul_l :
  forall n (sq : Vector.t Q n) (q : Q) (e : GA_expr n),
    (support_size (eval_expr sq (Mul (Scalar q) e))
     <= support_size (eval_expr sq e))%nat.
Proof.
  intros n sq q e.
  rewrite (support_size_ext _ _ _ (eval_mul_scalar_l n sq q e)).
  destruct (Qeq_dec q 0) as [Hq0|Hq0].
  - assert (Hz : forall m, mv_scale q (eval_expr sq e) m == mv_zero m).
    { intro m. unfold mv_scale, mv_zero. rewrite Hq0. ring. }
    rewrite (support_size_ext _ _ _ Hz).
    rewrite support_size_zero. lia.
  - rewrite (support_size_scale n q _ Hq0). lia.
Qed.

Lemma support_size_scalar_mul_r :
  forall n (sq : Vector.t Q n) (q : Q) (e : GA_expr n),
    (support_size (eval_expr sq (Mul e (Scalar q)))
     <= support_size (eval_expr sq e))%nat.
Proof.
  intros n sq q e.
  rewrite (support_size_ext _ _ _ (eval_mul_scalar_r n sq q e)).
  destruct (Qeq_dec q 0) as [Hq0|Hq0].
  - assert (Hz : forall m, mv_scale q (eval_expr sq e) m == mv_zero m).
    { intro m. unfold mv_scale, mv_zero. rewrite Hq0. ring. }
    rewrite (support_size_ext _ _ _ Hz).
    rewrite support_size_zero. lia.
  - rewrite (support_size_scale n q _ Hq0). lia.
Qed.

Theorem eval_support_size_le :
  forall n (sq : Vector.t Q n) (e : GA_expr n),
    (support_size (eval_expr sq e) <= support_size_bound e)%nat.
Proof.
  intros n sq e.
  induction e as [i|c|e1 IH1 e2 IH2|e1 IH1 e2 IH2|e1 IH1 e2 IH2]; simpl.
  - (* Basis *) rewrite support_size_basis. lia.
  - (* Scalar *)
    destruct (Qeq_dec c 0) as [Hc0|Hc0].
    + assert (He : forall m, mv_scale c (@mv_one n) m == mv_zero m).
      { intro m. unfold mv_scale, mv_zero. rewrite Hc0. ring. }
      rewrite (support_size_ext _ _ _ He). rewrite support_size_zero. lia.
    + rewrite (support_size_scale n c mv_one Hc0).
      rewrite support_size_mv_one. lia.
  - (* Add *)
    eapply Nat.le_trans; [apply support_size_add | lia].
  - (* Mul *)
    destruct e1 as [?|q1|? ?|? ?|? ?].
    2: { (* Scalar q1 * e2 — handle separately *)
      simpl support_size_bound.
      eapply Nat.le_trans;
        [apply support_size_scalar_mul_l | exact IH2].
    }
    (* Non-Scalar * e2: destruct e2 *)
    all: destruct e2 as [?|q2|? ?|? ?|? ?];
         simpl support_size_bound.
    (* Non-Scalar * Non-Scalar: crude 2^n bound *)
    all: try (eapply Nat.le_trans;
              [apply support_size_le_2n | lia]).
    (* Non-Scalar * Scalar q2: right helper *)
    all: eapply Nat.le_trans;
         [apply support_size_scalar_mul_r | exact IH1].
  - (* Conv *)
    eapply Nat.le_trans; [apply support_size_conv | nia].
Qed.

        (* Lemma Block 4 *)

Fixpoint formula_size {n} (phi : BoolFormula n) : nat :=
  match phi with
  | BVar _   => 1
  | BConst _ => 1
  | BAnd p q => 1 + formula_size p + formula_size q
  | BOr p q  => 1 + formula_size p + formula_size q
  | BNot p   => 1 + formula_size p
  end.

Lemma formula_size_pos : forall n (phi : BoolFormula n),
  (formula_size phi >= 1)%nat.
Proof. intros n phi; destruct phi; simpl; lia. Qed.

Lemma pow2_pos : forall k, (1 <= Nat.pow 2 k)%nat.
Proof. induction k; simpl; lia. Qed.

Lemma pow2_ge_2 : forall k, (k >= 1)%nat -> (2 <= Nat.pow 2 k)%nat.
Proof.
  intros [|k'] Hk; [lia|]. simpl. pose proof (pow2_pos k'). lia.
Qed.


Lemma support_size_bound_translate :
  forall n (phi : BoolFormula n),
    (support_size_bound (translate phi) <= Nat.pow 2 (formula_size phi))%nat.
Proof.
  intros n phi.
  induction phi as [i|[]|p IHp q IHq|p IHp q IHq|p IHp].
  - (* BVar: Mul (Scalar _) (Add (Scalar _) (Basis _))
       support_size_bound = 1 + 1 = 2, formula_size = 1, 2^1 = 2 *)
    simpl. lia.
  - (* BConst true: Scalar 1, bound = 1 ≤ 2 *)
    simpl. lia.
  - (* BConst false: Scalar 0, bound = 1 ≤ 2 *)
    simpl. lia.
  - (* BAnd p q: Conv (translate p) (translate q)
       bound = ssb_p * ssb_q, need ≤ 2^(1 + sp + sq) *)
    simpl support_size_bound. simpl formula_size.
    (* ssb_p * ssb_q ≤ 2^sp * 2^sq = 2^(sp+sq) ≤ 2^(1+sp+sq) *)
    transitivity (Nat.pow 2 (formula_size p) * Nat.pow 2 (formula_size q))%nat.
    { apply Nat.mul_le_mono; assumption. }
    rewrite <- Nat.pow_add_r.
    apply Nat.pow_le_mono_r; lia.
  - (* BOr p q: Add (Add tp tq) (Mul (Scalar -1) (Conv tp tq))
       bound = (ssb_p + ssb_q) + ssb_p * ssb_q
       need ≤ 2^(1 + sp + sq) = 2 * 2^sp * 2^sq *)
    simpl support_size_bound. simpl formula_size.
    pose proof (pow2_ge_2 (formula_size p) (formula_size_pos n p)).
    pose proof (pow2_ge_2 (formula_size q) (formula_size_pos n q)).
    (* Step 1: replace bounds via IH *)
    transitivity
      (Nat.pow 2 (formula_size p) + Nat.pow 2 (formula_size q)
       + Nat.pow 2 (formula_size p) * Nat.pow 2 (formula_size q))%nat.
    { assert (support_size_bound (translate p) * support_size_bound (translate q)
              <= Nat.pow 2 (formula_size p) * Nat.pow 2 (formula_size q))%nat
        by (apply Nat.mul_le_mono; lia).
      lia. }
    (* Step 2: a + b + a*b ≤ 2*a*b for a,b ≥ 2  
       (because a + b ≤ a*b when a,b ≥ 2) *)
    transitivity (2 * (Nat.pow 2 (formula_size p) * Nat.pow 2 (formula_size q)))%nat.
    { nia. }
    (* Step 3: 2 * 2^sp * 2^sq = 2^(1+sp+sq) *)
    rewrite <- Nat.pow_add_r.
    change (1 + formula_size p + formula_size q)%nat
      with (S (formula_size p + formula_size q)).
    simpl Nat.pow. lia.
  - (* BNot p: Add (Scalar 1) (Mul (Scalar -1) (translate p))
       bound = 1 + ssb_p, need ≤ 2^(1 + sp) = 2 * 2^sp *)
    simpl support_size_bound. simpl formula_size.
    pose proof (pow2_pos (formula_size p)).
    (* 1 + ssb_p ≤ 1 + 2^sp ≤ 2^sp + 2^sp = 2 * 2^sp *)
    change (1 + formula_size p)%nat with (S (formula_size p)).
    simpl Nat.pow. lia.
Qed.

Corollary translate_support_size_bound :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n),
    (support_size (eval_expr sq (translate phi))
     <= Nat.pow 2 (formula_size phi))%nat.
Proof.
  intros n sq phi.
  eapply Nat.le_trans.
  - apply eval_support_size_le.
  - apply support_size_bound_translate.
Qed.

        (* Lemma Block 5 *)

(* ============================================================ *)
(* (-1)^n in Q                                                   *)
(* ============================================================ *)

Definition neg1_pow (k : nat) : Q :=
  if Nat.even k then 1%Q else (-(1))%Q.

Lemma neg1_pow_S : forall k, neg1_pow (S k) == (- neg1_pow k)%Q.
Proof.
  intro k; unfold neg1_pow.
  rewrite Nat.even_succ.
  unfold Nat.odd.

  destruct (Nat.even k) eqn:Hev; simpl; ring.
Qed.

Lemma neg1_pow_neq_0 : forall k, ~(neg1_pow k == 0).
Proof.
  intro k; unfold neg1_pow; destruct (Nat.even k);
    unfold Qeq; simpl; discriminate.
Qed.

(* ============================================================ *)
(* Core Fourier identity:                                        *)
(*   2 * bQ(XOR(s)) + (-1)^n * chi(full_mask, s) == 1          *)
(*                                                               *)
(* This says the {0,1}-valued XOR function decomposes into       *)
(* exactly two Walsh characters: the trivial one and the full.   *)
(* ============================================================ *)

Lemma bQ_XOR_plus_chi : forall n (s : Corner n),
  (2 * bQ (XOR_n_func s) + neg1_pow n * chi' (Vector.const true n) s == 1)%Q.
Proof.
  induction n as [|n IHn]; intro s.
  - (* n = 0: XOR(empty) = false, chi(empty,empty) = 1, neg1_pow 0 = 1 *)
    dependent destruction s.
    unfold XOR_n_func, bQ, neg1_pow, chi'; simpl; ring.
  - dependent destruction s; rename h into hd; rename s into tl.
    rewrite XOR_n_func_cons.
    change (Vector.const true (S n))
      with (Vector.cons _ true _ (Vector.const true n)).
    rewrite chi_true_cons, neg1_pow_S.
    specialize (IHn tl).
    destruct hd; simpl sign_to_bool; simpl sQ.
    + (* Pos: xorb true (XOR tl) = negb (XOR tl), sQ Pos = 1 *)
      rewrite xorb_true_l, bQ_negb.
      (* Goal: 2*(1 - bQ(XOR tl)) + (-neg1_pow n)*(1*chi(full,tl)) == 1 *)
      (* = 2 - [2*bQ(XOR tl) + neg1_pow n * chi(full,tl)] == 1 *)
      setoid_replace
        (2 * (1 - bQ (XOR_n_func tl))
         + - neg1_pow n * (1 * chi' (Vector.const true n) tl))%Q
        with
        (2 - (2 * bQ (XOR_n_func tl)
              + neg1_pow n * chi' (Vector.const true n) tl))%Q
        by ring.
      rewrite IHn; ring.
    + (* Neg: xorb false (XOR tl) = XOR tl, sQ Neg = -1 *)
      rewrite xorb_false_l.
      (* Goal: 2*bQ(XOR tl) + (-neg1_pow n)*(-1*chi(full,tl)) == 1 *)
      (* = 2*bQ(XOR tl) + neg1_pow n * chi(full,tl) == 1 = IH *)
      setoid_replace
        (2 * bQ (XOR_n_func tl)
         + - neg1_pow n * (-1 * chi' (Vector.const true n) tl))%Q
        with
        (2 * bQ (XOR_n_func tl)
         + neg1_pow n * chi' (Vector.const true n) tl)%Q
        by ring.
      exact IHn.
Qed.

(* ============================================================ *)
(* Explicit 2-term multivector equal to embed(XOR)               *)
(* ============================================================ *)

Definition XOR_explicit {n} : MV n :=
  mv_add (mv_scale (1#2) (basis mask_empty))
         (mv_scale (- neg1_pow n * (1#2)) (basis (Vector.const true n))).

Lemma eval_XOR_explicit : forall n (s : Corner n),
  eval (@XOR_explicit n) s == bQ (XOR_n_func s).
Proof.
  intros n s.
  unfold XOR_explicit.
  rewrite eval_add, !eval_scale, !eval_basis, chi_mask_empty.
  (* Goal: (1#2)*1 + (-neg1_pow n*(1#2)) * chi'(full,s) == bQ(XOR s) *)
  (* From bQ_XOR_plus_chi: 2*bQ(XOR s) + neg1_pow n * chi'(full,s) == 1 *)
  pose proof (bQ_XOR_plus_chi n s) as H.
  (* Strategy: rewrite both sides to (1 - neg1_pow n * chi'(full,s)) * (1#2) *)
  setoid_replace
    ((1 # 2) * 1 + - neg1_pow n * (1 # 2) * chi' (Vector.const true n) s)%Q
    with
    ((1 - neg1_pow n * chi' (Vector.const true n) s) * (1 # 2))%Q
    by ring.
  setoid_replace (bQ (XOR_n_func s))
    with (2 * bQ (XOR_n_func s) * (1 # 2))%Q
    by ring.
  apply Qmult_comp; [|reflexivity].
  (* Goal: 1 - neg1_pow n * chi'(full,s) == 2 * bQ(XOR s) *)
  (* From H: rewrite 1 → 2*bQ + neg1_pow*chi, then ring *)
  rewrite <- H; ring.
Qed.

(* ============================================================ *)
(* Transfer: embed(XOR) == XOR_explicit coefficient-wise         *)
(* ============================================================ *)

Lemma embed_XOR_eq_explicit : forall n (m : Mask n),
  embed (@XOR_n_func n) m == (@XOR_explicit n) m.
Proof.
  intros n.
  apply eval_extensionality.
  intro s.
  rewrite embed_correct.
  symmetry.
  apply eval_XOR_explicit.
Qed.

(* ============================================================ *)
(* mask_empty ≠ full_mask for n ≥ 1                              *)
(* ============================================================ *)

Lemma mask_empty_neq_full : forall n,
  (n > 0)%nat -> @mask_empty n <> Vector.const true n.
Proof.
  intros [|n'] Hn; [lia|].
  intro H.
  pose proof (f_equal (fun v => Vector.hd v) H) as Hhd.
  simpl in Hhd. discriminate.
Qed.

(* ============================================================ *)
(* XOR_explicit has nonzero coefficients at empty and full       *)
(* ============================================================ *)

Lemma XOR_explicit_empty_nonzero : forall n,
  (n > 0)%nat -> ~(@XOR_explicit n mask_empty == 0).
Proof.
  intros n Hn.
  unfold XOR_explicit, mv_add, mv_scale, basis.
  destruct (mask_eq_dec mask_empty mask_empty) as [_|Habs];
    [|exfalso; apply Habs; reflexivity].
  destruct (mask_eq_dec mask_empty (Vector.const true n)) as [Heq|Hne].
  - exfalso. exact (mask_empty_neq_full n Hn Heq).
  - (* value = (1#2)*1 + (-neg1_pow n * (1#2))*0 = 1#2 *)
    intro H.
    assert (Hval : ((1 # 2) * 1 + - neg1_pow n * (1 # 2) * 0)%Q == (1#2)%Q) by ring.
    rewrite Hval in H.
    unfold Qeq in H; simpl in H; discriminate.
Qed.

Lemma XOR_explicit_full_nonzero : forall n,
  (n > 0)%nat -> ~(@XOR_explicit n (Vector.const true n) == 0).
Proof.
  intros n Hn.
  unfold XOR_explicit, mv_add, mv_scale, basis.
  destruct (mask_eq_dec (Vector.const true n) mask_empty) as [Heq|Hne].
  - exfalso. exact (mask_empty_neq_full n Hn (eq_sym Heq)).
  - destruct (mask_eq_dec (Vector.const true n) (Vector.const true n))
      as [_|Habs]; [|exfalso; apply Habs; reflexivity].
    (* value = (1#2)*0 + (-neg1_pow n * (1#2))*1 = -neg1_pow n * (1#2) *)
    intro H.
    assert (Hval : ((1 # 2) * 0 + - neg1_pow n * (1 # 2) * 1)%Q
                   == (- neg1_pow n * (1 # 2))%Q) by ring.
    rewrite Hval in H.
    apply (neg1_pow_neq_0 n).
    (* from -neg1_pow n * (1#2) == 0 derive neg1_pow n == 0 *)
    destruct (Qeq_dec (neg1_pow n) 0) as [Hz|Hnz]; [exact Hz|].
    exfalso. apply Hnz.
    apply Qmult_integral in H as [H|H].
    + (* -neg1_pow n == 0 implies neg1_pow n == 0 *)
      assert (neg1_pow n == - - neg1_pow n)%Q by ring.
      rewrite H0, H. ring.
    + unfold Qeq in H; simpl in H; discriminate.
Qed.

(* ============================================================ *)
(* XOR_explicit is zero at all other masks                       *)
(* ============================================================ *)

Lemma XOR_explicit_other_zero : forall n (m : Mask n),
  m <> mask_empty ->
  m <> Vector.const true n ->
  @XOR_explicit n m == 0.
Proof.
  intros n m Hne Hnf.
  unfold XOR_explicit, mv_add, mv_scale, basis.
  destruct (mask_eq_dec m mask_empty) as [Heq|_];
    [contradiction|].
  destruct (mask_eq_dec m (Vector.const true n)) as [Heq|_];
    [contradiction|].
  ring.
Qed.

(* ============================================================ *)
(* Lower bound: 2 distinct nonzero coefficients ⟹ support ≥ 2  *)
(* ============================================================ *)

Lemma support_size_at_least_two :
  forall n (F : MV n) (a b : Mask n),
    a <> b -> ~(F a == 0) -> ~(F b == 0) ->
    (2 <= support_size F)%nat.
Proof.
  intros n F a b Hneq Ha Hb.
  unfold support_size.
  set (p := fun m : Mask n => negb (Qeq_bool (F m) 0)).
  assert (HinA : In a (filter p (all_masks n))).
  { apply filter_In. split; [apply all_masks_complete|].
    unfold p. destruct (Qeq_bool (F a) 0) eqn:E.
    + exfalso. apply Ha. apply Qeq_bool_eq. exact E.
    + reflexivity. }
  assert (HinB : In b (filter p (all_masks n))).
  { apply filter_In. split; [apply all_masks_complete|].
    unfold p. destruct (Qeq_bool (F b) 0) eqn:E.
    + exfalso. apply Hb. apply Qeq_bool_eq. exact E.
    + reflexivity. }
  (* [a; b] is NoDup and included in the filter, so length ≥ 2 *)
  assert (Hincl : incl [a; b] (filter p (all_masks n))).
  { intros x [->|[->|[]]]; assumption. }
  assert (Hnd : NoDup [a; b]).
  { constructor.
    - simpl. intros [H|[]]. apply Hneq. now symmetry.
    - constructor; [simpl; tauto | constructor]. }
  pose proof (@NoDup_incl_length (Mask n) [a;b] (filter p (all_masks n)) Hnd Hincl) as Hle.
  simpl in Hle. lia.
Qed.

(* ============================================================ *)
(* support_size(XOR_explicit) ≤ 2                                *)
(* ============================================================ *)

Lemma support_size_basis_general : forall n (M : Mask n),
  support_size (basis M) = 1%nat.
Proof.
  intros n M.
  unfold support_size.
  set (p := fun m : Mask n => negb (Qeq_bool (basis M m) 0)).
  rewrite (filter_singleton (Mask n) p (all_masks n) M
             (all_masks_nodup n) (all_masks_complete M)).
  - simpl. reflexivity.
  - (* p M = true *)
    unfold p, basis.
    destruct (mask_eq_dec M M) as [_|Hc]; [|contradiction].
    rewrite Qeq_bool_10. reflexivity.
  - (* p x = false for x ≠ M *)
    intros x _ Hneq.
    unfold p, basis.
    destruct (mask_eq_dec x M) as [Heq|_];
      [contradiction|].
    rewrite Qeq_bool_00. reflexivity.
Qed.

Lemma support_size_XOR_explicit_le :
  forall n, (support_size (@XOR_explicit n) <= 2)%nat.
Proof.
  intro n.
  unfold XOR_explicit.
  eapply Nat.le_trans; [apply support_size_add|].
  assert (H1 : ~((1 # 2) == 0)%Q)
    by (unfold Qeq; simpl; discriminate).
  assert (H2 : ~((- neg1_pow n * (1 # 2)) == 0)%Q).
  { intro H. apply (neg1_pow_neq_0 n).
    apply Qmult_integral in H as [H|H].
    - assert (neg1_pow n == - - neg1_pow n)%Q by ring.
      rewrite H0, H. ring.
    - unfold Qeq in H; simpl in H; discriminate. }
  rewrite (support_size_scale n _ _ H1).
  rewrite (support_size_scale n _ _ H2).
  rewrite !support_size_basis_general.
  lia.
Qed.

(* ============================================================ *)
(* The main result                                               *)
(* ============================================================ *)

Lemma support_size_XOR : forall n,
  (n > 0)%nat ->
  support_size (embed (@XOR_n_func n)) = 2%nat.
Proof.
  intros n Hn.
  (* Transfer from embed(XOR) to XOR_explicit *)
  rewrite (support_size_ext n _ _ (embed_XOR_eq_explicit n)).
  (* Upper bound: ≤ 2 *)
  assert (Hle := support_size_XOR_explicit_le n).
  (* Lower bound: ≥ 2 *)
  assert (Hge := @support_size_at_least_two n
                   (@XOR_explicit n) mask_empty (Vector.const true n)
                   (mask_empty_neq_full n Hn)
                   (XOR_explicit_empty_nonzero n Hn)
                   (XOR_explicit_full_nonzero n Hn)).
  lia.
Qed.

(* ============================================================ *)
(* Inner Product mod 2:  IP(x₀,y₀,x₁,y₁,...) = ⊕ᵢ (xᵢ ∧ yᵢ)  *)
(* Variables are paired consecutively: (0,1), (2,3), (4,5), ... *)
(* ============================================================ *)

Fixpoint IP_raw (l : list Sign) : bool :=
  match l with
  | nil         => false
  | _ :: nil    => false            (* odd — shouldn't happen *)
  | x :: y :: rest =>
      xorb (andb (sign_to_bool x) (sign_to_bool y))
           (IP_raw rest)
  end.

Definition IP_n_func {n} (c : Corner n) : bool :=
  IP_raw (Vector.to_list c).

Lemma IP_n_func_cons2 :
  forall n (x y : Sign) (c : Corner n),
    @IP_n_func (S (S n)) (Vector.cons _ x _ (Vector.cons _ y _ c))
    = xorb (andb (sign_to_bool x) (sign_to_bool y))
           (@IP_n_func n c).
Proof.
  intros n x y c.
  unfold IP_n_func.
  rewrite !to_list_cons.
  simpl IP_raw.
  reflexivity.
Qed.

(* ============================================================ *)
(* Corrected: IP on 2m variables has FULL support (2^{2m})       *)
(*                                                               *)
(* Proof sketch (tensor product structure):                      *)
(*   (-1)^{IP(s)} = Π_i (-1)^{AND(s_{2i}, s_{2i+1})}           *)
(*   Each factor has all 4 Walsh coefficients = ±1/2            *)
(*   Product over m independent pairs: all 4^m = 2^{2m} nonzero *)
(*   bQ(IP) = (1 - (-1)^IP)/2 inherits full support             *)
(*   For M≠∅: embed(IP)(M) = -(±1)/2^{m+1} ≠ 0                 *)
(*   For M=∅: embed(IP)(∅) = (2^m - 1)/2^{m+1} ≠ 0             *)
(* ============================================================ *)

(* Step 1: (-1)^b as a Q value *)
Definition signed (b : bool) : Q := if b then (-1)%Q else 1%Q.

(* Step 2: signed is multiplicative over XOR *)
Lemma signed_xorb : forall a b, signed (xorb a b) == (signed a * signed b)%Q.
Proof. destruct a, b; simpl; ring. Qed.

(* Step 3: The "signed Walsh coefficient" *)
Definition signed_walsh {n} (f : Corner n -> bool) (M : Mask n) : Q :=
  sumQ (List.map (fun a => (signed (f a) * chi' M a)%Q) (all_corners n)).

(* Step 4: bQ in terms of signed *)
Lemma bQ_via_signed : forall b, bQ b == ((1 - signed b) * (1#2))%Q.
Proof. destruct b; simpl; ring. Qed.

(* 2-variable AND function *)
Definition AND_2 (c : Corner 2) : bool :=
  andb (sign_to_bool (Vector.hd c))
       (sign_to_bool (Vector.hd (Vector.tl c))).

(* 2-variable signed Walsh coefficient:
   Σ over all 4 corners of dim 2 *)
Definition signed_walsh_2 (f : Corner 2 -> bool) (b1 b2 : bool) : Q :=
  signed_walsh f (Vector.cons _ b1 _ (Vector.cons _ b2 _ (Vector.nil _))).

(* Product of 2-variable factors, one per coordinate pair *)
Fixpoint prod_pairs_raw (l : list bool) : Q :=
  match l with
  | []            => 1%Q
  | [_]           => 1%Q            (* odd — shouldn't happen *)
  | b1 :: b2 :: rest =>
      (signed_walsh_2 AND_2 b1 b2 * prod_pairs_raw rest)%Q
  end.

Definition prod_pairs (m : nat) (M : Mask (m + m)) : Q :=
  prod_pairs_raw (Vector.to_list M).

(* The arithmetic fact *)
Lemma Sn_plus_Sn : forall m, S (S ((m + m)%nat)) = (S m + S m)%nat.
Proof.
  intro m. lia.
Qed.

(* Transport for vectors *)
Definition mask_cast {n1 n2} (H : n1 = n2) (M : Mask n1) : Mask n2 :=
  eq_rect n1 (Vector.t bool) M n2 H.

(* KEY: to_list erases transport *)
Lemma to_list_cast : forall n1 n2 (H : n1 = n2) (M : Mask n1),
  Vector.to_list (mask_cast H M) = Vector.to_list M.
Proof. intros. subst. reflexivity. Qed.

(* Now the cons2 lemma types correctly *)
Lemma prod_pairs_cons2 : forall m (b1 b2 : bool) (M : Mask (m + m)),
  prod_pairs (S m) (mask_cast (Sn_plus_Sn m)
    (Vector.cons _ b1 _ (Vector.cons _ b2 _ M)))
  = (signed_walsh_2 AND_2 b1 b2 * prod_pairs m M)%Q.
Proof.
  intros.
  unfold prod_pairs.
  rewrite to_list_cast.
  rewrite !to_list_cons.
  simpl prod_pairs_raw.
  reflexivity.
Qed.

Lemma vector_peel2 : forall (A : Type) (n : nat) (v : Vector.t A (S (S n))),
  exists (a1 a2 : A) (v' : Vector.t A n),
    Vector.to_list v = a1 :: a2 :: Vector.to_list v'.
Proof.
  intros A n v.
  dependent destruction v. rename h into a1.
  dependent destruction v. rename h into a2.
  exists a1, a2, v.
  rewrite !to_list_cons. reflexivity.
Qed.

Lemma Sm_plus_Sm : forall m, (S m + S m)%nat = (S (S (m + m)%nat)).
Proof. intro m. lia. Qed.

Lemma mask_decompose2 : forall m (M : Mask (S m + S m)),
  exists b1 b2 (M'' : Mask (m + m)),
    Vector.to_list M = b1 :: b2 :: Vector.to_list M''.
Proof.
  intros m M.
  destruct (vector_peel2 bool (m + m) (mask_cast (Sm_plus_Sm m) M))
    as [b1 [b2 [M'' Hlist]]].
  exists b1, b2, M''.
  rewrite to_list_cast in Hlist.
  exact Hlist.
Qed.

Lemma prod_pairs_unfold : forall m (M : Mask (S m + S m)),
  forall b1 b2 (M'' : Mask (m + m)),
    Vector.to_list M = b1 :: b2 :: Vector.to_list M'' ->
    prod_pairs (S m) M = (signed_walsh_2 AND_2 b1 b2 * prod_pairs m M'')%Q.
Proof.
  intros m M b1 b2 M'' Hlist.
  unfold prod_pairs. rewrite Hlist. simpl prod_pairs_raw. reflexivity.
Qed.

Definition inject_Z (z : Z) : Q := z # 1.

(* Step 5: embed in terms of signed_walsh *)

Require Import Coq.Setoids.Setoid.
Require Import Coq.Classes.Morphisms.

Lemma Qmult_eq_compat_l' : forall a b c : Q, b == c -> a * b == a * c.
Proof.
  intros a b c H; now setoid_rewrite H.
Qed.

Lemma Qmult_eq_compat_r' : forall a b c : Q, b == c -> b * a == c * a.
Proof.
  intros a b c H; now setoid_rewrite H.
Qed.

Lemma Qminus_eq_compat_l' : forall a b c : Q, a == b -> a - c == b - c.
Proof. intros a b c H; unfold Qminus; now setoid_rewrite H. Qed.
Lemma Qminus_eq_compat_r' : forall a b c : Q, a == b -> c - a == c - b.
Proof. intros a b c H; unfold Qminus; now setoid_rewrite H. Qed.

Lemma embed_via_signed_walsh : forall n (f : Corner n -> bool) (M : Mask n),
  embed f M == ((1 / pow2 n) * ((1#2) * (if mask_eq_dec M mask_empty then pow2 n else 0)
                - (1#2) * signed_walsh f M))%Q.
Proof.
  intros n f M.
  unfold embed, Pi, signed_walsh.

  (* Step 1: pull out 1/pow2 n *)
  eapply Qeq_trans.
  { apply (sumQ_map_ext (A := Corner n)
      (fun a => (bQ (f a) * (1 / pow2 n * chi M a))%Q)
      (fun a => ((1 / pow2 n) * (bQ (f a) * chi' M a))%Q)
      (all_corners n)).
    intros a _. unfold chi'. ring. }
  rewrite (sumQ_map_scale_l (A := Corner n) (1 / pow2 n)%Q
    (fun a => (bQ (f a) * chi' M a)%Q) (all_corners n)).

  apply Qmult_eq_compat_l'.

  (* Step 2: rewrite bQ via signed, split into difference *)
  eapply Qeq_trans.
  { apply (sumQ_map_ext (A := Corner n)
      (fun a => (bQ (f a) * chi' M a)%Q)
      (fun a => ((1#2) * chi' M a - (1#2) * (signed (f a) * chi' M a))%Q)
      (all_corners n)).
    intros a _. rewrite bQ_via_signed. ring. }

  (* Step 3: split sum of differences into difference of sums *)
  rewrite (sumQ_map_sub
    (fun a : Corner n => ((1#2) * chi' M a)%Q)
    (fun a : Corner n => ((1#2) * (signed (f a) * chi' M a))%Q)
    (all_corners n)).

  (* Step 4: factor (1#2) out of each sum — FORWARD direction *)
  rewrite (sumQ_map_scale_l (A := Corner n) (1#2)%Q
    (fun a => chi' M a) (all_corners n)).
  rewrite (sumQ_map_scale_l (A := Corner n) (1#2)%Q
    (fun a => (signed (f a) * chi' M a)%Q) (all_corners n)).

  (* Step 5: character orthogonality + ring *)
  assert (Hchi_sum : sumQ (List.map (fun a => chi' M a) (all_corners n))
                     == if mask_eq_dec M mask_empty then pow2 n else 0).
  { destruct (mask_eq_dec M mask_empty) as [Heq|Hneq].
    - subst M. apply chi_corner_sum_empty.
    - apply chi_corner_sum_nonempty. exact Hneq. }
  rewrite Hchi_sum.
  ring.
Qed.

Lemma all_corners_2 :
  all_corners 2 = [Vector.cons _ Pos _ (Vector.cons _ Pos _ (Vector.nil _));
                   Vector.cons _ Pos _ (Vector.cons _ Neg _ (Vector.nil _));
                   Vector.cons _ Neg _ (Vector.cons _ Pos _ (Vector.nil _));
                   Vector.cons _ Neg _ (Vector.cons _ Neg _ (Vector.nil _))].
Proof. reflexivity. Qed.

Lemma signed_walsh_AND_2var : forall (b1 b2 : bool),
  signed_walsh_2 AND_2 b1 b2 ==
    if b1 then (if b2 then (-2)%Q else (-2)%Q)
    else (if b2 then (-2)%Q else 2%Q).
Proof.
  intros b1 b2.
  unfold signed_walsh_2, signed_walsh, AND_2, signed, chi', chi, sQ,
         sign_to_bool.
  destruct b1, b2; rewrite all_corners_2; simpl; ring.
Qed.

(* Step 8: Therefore signed_walsh(IP)(M) = ±2^m for every M *)
Require Import Coq.Lists.List.
Import ListNotations.

(* 1-step split is definitional for your Fixpoint *)

Lemma all_corners_S : forall n,
  all_corners (S n)
  =
    map (fun c => Vector.cons Sign Pos n c) (all_corners n)
 ++ map (fun c => Vector.cons Sign Neg n c) (all_corners n).
Proof.
  intro n. simpl. reflexivity.
Qed.

(* 2-step split into 4 quadrants *)
Lemma all_corners_SS : forall n,
  all_corners (S (S n))
  =
     map (fun c => Vector.cons Sign Pos (S n)
                   (Vector.cons Sign Pos n c)) (all_corners n)
  ++ map (fun c => Vector.cons Sign Pos (S n)
                   (Vector.cons Sign Neg n c)) (all_corners n)
  ++ map (fun c => Vector.cons Sign Neg (S n)
                   (Vector.cons Sign Pos n c)) (all_corners n)
  ++ map (fun c => Vector.cons Sign Neg (S n)
                   (Vector.cons Sign Neg n c)) (all_corners n).
Proof.
  intro n.
  simpl.
  rewrite map_app.
  rewrite map_app.
  repeat rewrite map_map.
  repeat rewrite app_assoc.
  reflexivity.
Qed.

Lemma chi'_cons2_factor :
  forall n (b1 b2 : bool) (m : Mask n) (x y : Sign) (c : Corner n),
    chi' (Vector.cons _ b1 _ (Vector.cons _ b2 _ m))
         (Vector.cons _ x  _ (Vector.cons _ y  _ c))
    ==
    ((if b1 then sQ x else 1%Q) *
     (if b2 then sQ y else 1%Q) *
     chi' m c)%Q.
Proof.
  intros n b1 b2 m x y c.
  destruct b1, b2; simpl.
  all: try (rewrite chi_true_cons; rewrite chi_true_cons).
  all: try (rewrite chi_true_cons; rewrite chi_false_cons).
  all: try (rewrite chi_false_cons; rewrite chi_true_cons).
  all: try (rewrite chi_false_cons; rewrite chi_false_cons).
  all: (* now just associativity / ones *)
       repeat rewrite Qmult_assoc;
       repeat rewrite Qmult_1_l;
       repeat rewrite Qmult_1_r;
       reflexivity.
Qed.

Lemma to_list_inj : forall (A : Type) (n : nat) (v1 v2 : Vector.t A n),
  Vector.to_list v1 = Vector.to_list v2 -> v1 = v2.
Proof.
  intros A n. induction n; intros v1 v2 H.
  - dependent destruction v1. dependent destruction v2. reflexivity.
  - dependent destruction v1. dependent destruction v2.
    rewrite !to_list_cons in H.
    injection H as Hh Ht. subst.
    f_equal. apply IHn. exact Ht.
Qed.

Definition corner_cast {n1 n2} (H : n1 = n2) (c : Corner n1) : Corner n2 :=
  eq_rect n1 Corner c n2 H.

Lemma to_list_corner_cast : forall n1 n2 (H : n1 = n2) (c : Corner n1),
  Vector.to_list (corner_cast H c) = Vector.to_list c.
Proof. intros. subst. reflexivity. Qed.

(* Pull out a left-multiplicative constant from sumQ(map ...) *)


Lemma Qmult_plus_distr_l_Qeq : forall k x y : Q,
  k * (x + y) == k * x + k * y.
Proof.
  intros k x y.
  (* field works with Qeq goals *)
  ring.
Qed.

Lemma sumQ_map_mul_l :
  forall (A:Type) (k:Q) (l:list A) (f:A->Q),
    sumQ (map (fun x => k * f x) l) == k * sumQ (map f l).
Proof.
  intros A k l f. induction l as [|a l IH]; simpl.
  - (* [] *) simpl; ring.   (* or reflexivity depending on sumQ *)
  - rewrite IH.
    symmetry.
    (* this line should now work if it's the Qeq lemma *)
    apply Qmult_plus_distr_l_Qeq.
Qed.

Lemma Qmult_plus_distr_r_Qeq : forall k x y : Q,
  (x + y) * k == x * k + y * k.
Proof.
  intros k x y.
  ring.
Qed.

(* Same thing but for right multiplication if you need it *)
Lemma sumQ_map_mul_r :
  forall (A:Type) (k:Q) (l:list A) (f:A->Q),
    sumQ (map (fun x => (f x * k)%Q) l) == (sumQ (map f l) * k)%Q.
Proof.
  intros A k l f. induction l as [|a l IH]; simpl.
  - ring.
  - rewrite IH.
    (* goal is: f a * k + sumQ(map f l) * k == (f a + sumQ(map f l)) * k *)
    (* use symmetry + right distributivity *)
    symmetry.
    apply Qmult_plus_distr_r_Qeq.
Qed.

(* If this is true in your development; often chi is just chi' *)
Lemma chi_eq_chi' : forall n (m:Mask n) (c:Corner n),
  chi m c == chi' m c.
Proof.
  unfold chi'.
  reflexivity.
Qed.

Lemma sumQ_flat_map :
  forall (A B : Type) (l : list A) (g : A -> list B) (f : B -> Q),
    sumQ (map f (flat_map g l))
    ==
    sumQ (map (fun a => sumQ (map f (g a))) l).
Proof.
  intros A B l; induction l as [|a l IH]; intros g f; cbn.
  - reflexivity.
  - rewrite !map_app, !sumQ_app. cbn.
    rewrite IH. reflexivity.
Qed.

Lemma map_flat_map :
  forall (A B C : Type) (l : list A) (g : A -> list B) (f : B -> C),
    map f (flat_map g l) = flat_map (fun a => map f (g a)) l.
Proof.
  intros A B C l; induction l as [|a l IH]; intros g f; cbn.
  - reflexivity.
  - rewrite map_app, IH. reflexivity.
Qed.

Definition cons2 {n} (c2 : Corner 2) (x : Corner n) : Corner (S (S n)) :=
  Vector.cons Sign (Vector.hd c2) (S n)
    (Vector.cons Sign (Vector.hd (Vector.tl c2)) n x).

Lemma all_corners_SS_flatmap :
  forall n,
    all_corners (S (S n)) =
      flat_map
        (fun c2 : Corner 2 =>
           map (fun x : Corner n => cons2 c2 x) (all_corners n))
        (all_corners 2).
Proof.
  intro n.
  rewrite all_corners_SS.
  rewrite all_corners_2. cbn [flat_map].
  unfold cons2.
  (* now just normalize *)
  cbn.
  rewrite app_nil_r.
  reflexivity.
Qed.

Lemma signed_walsh_tensor2 :
  forall n (g : Corner 2 -> bool) (h : Corner n -> bool)
         (b1 b2 : bool) (m : Mask n),
    signed_walsh
      (fun a : Corner (S (S n)) =>
         xorb (g (Vector.cons _ (Vector.hd a) _
                  (Vector.cons _ (Vector.hd (Vector.tl a)) _ (Vector.nil _))))
              (h (Vector.tl (Vector.tl a))))
      (Vector.cons _ b1 _ (Vector.cons _ b2 _ m))
    ==
    (signed_walsh_2 g b1 b2 * signed_walsh h m)%Q.
Proof.
  intros n g h b1 b2 m.
  unfold signed_walsh at 1.
  rewrite all_corners_SS_flatmap.
  
  (* name the integrand to keep rewriting manageable *)
  set (F :=
    fun a : Corner (S (S n)) =>
      signed
        (xorb
           (g (Vector.cons Sign (Vector.hd a) 1
                 (Vector.cons Sign (Vector.hd (Vector.tl a)) 0 (Vector.nil Sign))))
           (h (Vector.tl (Vector.tl a))))
      * chi' (Vector.cons bool b1 (S n) (Vector.cons bool b2 n m)) a).

  (* push map F through flat_map *)
  rewrite (map_flat_map
            (Corner 2) (Corner (S (S n))) Q
            (all_corners 2)
            (fun c2 : Corner 2 => map (fun x : Corner n => cons2 c2 x) (all_corners n))
            F).
  
  eapply Qeq_trans.
  {
    rewrite <- (map_id (flat_map
      (fun a : Corner 2 => map F (map (fun x : Corner n => cons2 a x) (all_corners n)))
      (all_corners 2))).

    rewrite (sumQ_flat_map
      (Corner 2) Q
      (all_corners 2)
      (fun a : Corner 2 => map F (map (fun x : Corner n => cons2 a x) (all_corners n)))
      (fun q : Q => q)).
    
    reflexivity.
  }
  cbn.
  
  (* --- cleanup: remove map (fun q => q) and fuse map-map --- *)
  repeat (change (fun q : Q => q) with (@id Q); rewrite map_id).
  repeat rewrite map_map.

  (* name the 4 explicit 2-corners *)
  set (cPP := Vector.cons Sign Pos 1 (Vector.cons Sign Pos 0 (Vector.nil Sign))).
  set (cPN := Vector.cons Sign Pos 1 (Vector.cons Sign Neg 0 (Vector.nil Sign))).
  set (cNP := Vector.cons Sign Neg 1 (Vector.cons Sign Pos 0 (Vector.nil Sign))).
  set (cNN := Vector.cons Sign Neg 1 (Vector.cons Sign Neg 0 (Vector.nil Sign))).

  (* name the common Walsh sum for h *)
  set (SW := signed_walsh h m).

  (* also name the four quadrant sums (now in the fused form) *)
  set (SPP := sumQ (map (fun x : Corner n => F (cons2 cPP x)) (all_corners n))).
  set (SPN := sumQ (map (fun x : Corner n => F (cons2 cPN x)) (all_corners n))).
  set (SNP := sumQ (map (fun x : Corner n => F (cons2 cNP x)) (all_corners n))).
  set (SNN := sumQ (map (fun x : Corner n => F (cons2 cNN x)) (all_corners n))).

  assert (SPP ==
          (signed (g cPP) *
           ((if b1 then 1 else 1) * ((if b2 then 1 else 1) * 1)) *
           SW)%Q) as SPP_eq.
  {
    unfold SPP, SW, F, signed_walsh.

    set (K :=
      (signed (g cPP) *
       ((if b1 then 1 else 1) * ((if b2 then 1 else 1) * 1)))%Q).

    eapply Qeq_trans.
    2: {
      apply sumQ_map_mul_l.
    }
    
    apply sumQ_map_ext; intros x Hx.
    unfold K.
    unfold cPP.
    cbn [cons2 Vector.hd Vector.tl].
    rewrite signed_xorb.
    rewrite (chi'_cons2_factor n b1 b2 m Pos Pos x).
    destruct b1, b2; cbn; ring.
  }

  assert (SPN ==
          (signed (g cPN) *
           ((if b1 then 1 else 1) * ((if b2 then -1 else 1) * 1)) *
           SW)%Q) as SPN_eq.
  {
    unfold SPN, SW, F, signed_walsh.
    set (K :=
      (signed (g cPN) *
       ((if b1 then 1 else 1) * ((if b2 then -1 else 1) * 1)))%Q).
    eapply Qeq_trans.
    2: { apply sumQ_map_mul_l. }
    apply sumQ_map_ext; intros x Hx.
    unfold K.
    unfold cPN.
    cbn [cons2 Vector.hd Vector.tl].
    rewrite signed_xorb.
    rewrite (chi'_cons2_factor n b1 b2 m Pos Neg x).
    destruct b1, b2; cbn; ring.
  }

  assert (SNP ==
          (signed (g cNP) *
           ((if b1 then -1 else 1) * ((if b2 then 1 else 1) * 1)) *
           SW)%Q) as SNP_eq.
  {
    unfold SNP, SW, F, signed_walsh.
    set (K :=
      (signed (g cNP) *
       ((if b1 then -1 else 1) * ((if b2 then 1 else 1) * 1)))%Q).
    eapply Qeq_trans.
    2: { apply sumQ_map_mul_l. }
    apply sumQ_map_ext; intros x Hx.
    unfold K.
    unfold cNP.
    cbn [cons2 Vector.hd Vector.tl].
    rewrite signed_xorb.
    rewrite (chi'_cons2_factor n b1 b2 m Neg Pos x).
    destruct b1, b2; cbn; ring.
  }

  assert (SNN ==
          (signed (g cNN) *
           ((if b1 then -1 else 1) * ((if b2 then -1 else 1) * 1)) *
           SW)%Q) as SNN_eq.
  {
    unfold SNN, SW, F, signed_walsh.
    set (K :=
      (signed (g cNN) *
       ((if b1 then -1 else 1) * ((if b2 then -1 else 1) * 1)))%Q).
    eapply Qeq_trans.
    2: { apply sumQ_map_mul_l. }
    apply sumQ_map_ext; intros x Hx.
    unfold K.
    unfold cNN.
    cbn [cons2 Vector.hd Vector.tl].
    rewrite signed_xorb.
    rewrite (chi'_cons2_factor n b1 b2 m Neg Neg x).
    destruct b1, b2; cbn; ring.
  }

  (* now finish the main goal *)
  rewrite SPP_eq, SPN_eq, SNP_eq, SNN_eq.

  (* the remaining statement is pure Q algebra;
     easiest is to case split b1 b2 to remove ifs and ring. *)
  destruct b1, b2; cbn; ring.
Qed.

Lemma signed_walsh_IP_factored : forall m (M : Mask (m + m)),
  signed_walsh (@IP_n_func (m + m)) M == prod_pairs m M.
Proof.
  induction m as [|m IH]; intro M.
  - dependent destruction M.
    unfold signed_walsh, prod_pairs, IP_n_func, signed.
    cbn. ring.
  -
    dependent destruction M.
    
    pose proof (Nat.add_succ_r m m) as e.
    
    remember (eq_rect (m + S m)%nat (fun n : nat => Mask n) M (S (m + m)) e)
      as M1' eqn:HM1'.
      
    dependent destruction M1'.

(* Helper: eq_rect on Mask preserves to_list *)
    assert (to_list_eqrect : forall n1 n2 (H : n1 = n2) (v : Mask n1),
      Vector.to_list (eq_rect n1 (fun n => Mask n) v n2 H) = Vector.to_list v).
    { intros. subst. reflexivity. }

    (* Helper: signed_walsh respects mask_cast *)
    assert (sw_cast : forall n1 n2 (H : n1 = n2) (Mx : Mask n1),
      signed_walsh (@IP_n_func n1) Mx ==
      signed_walsh (@IP_n_func n2) (mask_cast H Mx)).
    { intros. subst. reflexivity. }

    (* Step 1: Relate to_list M to h0 :: to_list M1' *)
    assert (HtolistM : Vector.to_list M = h0 :: Vector.to_list M1').
    {
      assert (Hc := to_list_eqrect _ _ e M).
      rewrite <- HM1' in Hc.
      rewrite to_list_cons in Hc.
      symmetry. exact Hc.
    }

    (* Step 2: Simplify RHS (prod_pairs) *)
    unfold prod_pairs.
    
    change (Vector.to_list (Vector.cons bool h (m + S m) M))
      with (h :: Vector.to_list M).
    rewrite HtolistM.
    simpl prod_pairs_raw.
    change (prod_pairs_raw (Vector.to_list M1')) with (prod_pairs m M1').
    rewrite <- (IH M1').
    
    assert (edim : S (m + S m) = S (S (m + m))) by lia.

    eapply Qeq_trans.
    { apply (sw_cast _ _ edim (Vector.cons _ h _ M)). }
    
    assert (Hcast_eq : mask_cast edim (Vector.cons _ h _ M) =
      Vector.cons _ h _ (Vector.cons _ h0 _ M1')).
    {
      apply to_list_inj.
      rewrite to_list_cast, !to_list_cons.
      f_equal. exact HtolistM.
    }
    rewrite Hcast_eq.
    eapply Qeq_trans.
    2: { apply (signed_walsh_tensor2 (m + m) AND_2 (@IP_n_func (m + m)) h h0 M1'). }

    unfold signed_walsh.
    apply sumQ_map_ext. intros a _.
    assert (HIP : @IP_n_func (S (S (m + m))) a =
      xorb (AND_2 (Vector.cons _ (Vector.hd a) _
                   (Vector.cons _ (Vector.hd (Vector.tl a)) _ (Vector.nil _))))
           (@IP_n_func (m + m) (Vector.tl (Vector.tl a)))).
    {
      rewrite (Vector.eta a), (Vector.eta (Vector.tl a)).
      apply IP_n_func_cons2.
    }
    rewrite HIP. reflexivity.
Qed.

Lemma signed_walsh_IP_magnitude : forall m (M : Mask (m + m)),
  (m > 0)%nat ->
  exists s : bool, signed_walsh (@IP_n_func (m + m)) M 
                    == signed s * inject_Z (Z.pow 2 (Z.of_nat m)).
Proof.
  intros m M Hm.
  assert (Hfact := signed_walsh_IP_factored m M).
  enough (H : exists s : bool,
    prod_pairs m M == signed s * inject_Z (2 ^ Z.of_nat m)).
  { destruct H as [s Hs]. exists s. eapply Qeq_trans; eassumption. }
  clear Hfact. revert M Hm.
  induction m as [|m' IH']; intros M Hm; [lia|].
  destruct (mask_decompose2 m' M) as [b1 [b2 [M'' Hlist]]].
  rewrite (prod_pairs_unfold m' M b1 b2 M'' Hlist).
  destruct m' as [|m''].
  - (* m = 1: single factor *)
    dependent destruction M''.
    unfold prod_pairs. simpl.
    assert (Haw := signed_walsh_AND_2var b1 b2).
    destruct b1, b2;    
    [ exists true | exists true | exists true | exists false ];
    (setoid_rewrite Haw;
     unfold signed, inject_Z; simpl; ring).

  - (* m = S (S m''): use IH *)
    assert (Hm'' : (S m'' > 0)%nat) by lia.
    destruct (IH' M'' Hm'') as [s' Hs'].
    assert (Haw := signed_walsh_AND_2var b1 b2).
    assert (Hprod : signed_walsh_2 AND_2 b1 b2 * prod_pairs (S m'') M'' ==
      (if b1 then if b2 then -2 else -2 else if b2 then -2 else 2) *
      (signed s' * inject_Z (2 ^ Z.of_nat (S m'')))).
    { eapply Qeq_trans.
      - apply Qmult_eq_compat_l'. exact Hs'.
      - apply Qmult_eq_compat_r'. exact Haw. }
    set (K := inject_Z (2 ^ Z.of_nat (S m''))) in *.
    
    assert (Hpow : inject_Z (2 ^ Z.of_nat (S (S m''))) == (inject_Z 2) * K).
    {
      subst K.
      (* rewrite the exponent: Z.of_nat (S (S m'')) = Z.succ (Z.of_nat (S m'')) *)
      rewrite Nat2Z.inj_succ.
      (* turn 2^(succ e) into 2^e * 2 in Z *)
      rewrite Z.pow_succ_r by lia.
      (* move the Z-multiplication out through inject_Z *)
      rewrite inject_Z_mult.
      
      change inject_Z with QArith_base.inject_Z.
      reflexivity.
    }
    
    destruct b1, b2; simpl in Hprod.
    + (* true,true : factor = -2 *)
      exists (negb s').
      eapply Qeq_trans; [ exact Hprod | ].
      rewrite Hpow.
      clearbody K.
      change (inject_Z 2) with (2%Q).
      destruct s'; unfold signed; simpl; ring.

    + (* b1=true, b2=false : factor = -2 *)
      exists (negb s').
      eapply Qeq_trans; [ exact Hprod | ].
      rewrite Hpow.
      clearbody K.
      change (inject_Z 2) with (2%Q).
      destruct s'; unfold signed; simpl; ring.

    + (* b1=false, b2=true : factor = -2 *)
      exists (negb s').
      eapply Qeq_trans; [ exact Hprod | ].
      rewrite Hpow.
      clearbody K.
      change (inject_Z 2) with (2%Q).
      destruct s'; unfold signed; simpl; ring.

    + (* b1=false, b2=false : factor = +2 *)
      exists s'.
      eapply Qeq_trans; [ exact Hprod | ].
      rewrite Hpow.
      clearbody K.
      change (inject_Z 2) with (2%Q).
      destruct s'; unfold signed; simpl; ring.
Qed.

(* Step 9: embed(IP)(M) ≠ 0 for all M *)

Lemma pow2_injectZ : forall n,
  pow2 n == inject_Z (2 ^ Z.of_nat n).
Proof.
  induction n as [|n IH].
  - simpl. reflexivity.
  -
    cbn [pow2].
    rewrite IH.
    rewrite Nat2Z.inj_succ.
    rewrite Z.pow_succ_r by lia.
    rewrite inject_Z_mult.
    change (inject_Z 2) with (2%Q).
    change (QArith_base.inject_Z 2) with (2%Q).
    reflexivity.
Qed.

Lemma Zpow2_nat_eq_1 : forall n,
  Z.pow 2 (Z.of_nat n) = 1%Z -> n = 0%nat.
Proof.
  induction n as [|n IH]; intro H; [reflexivity|].
  rewrite Nat2Z.inj_succ in H.
  rewrite Z.pow_succ_r in H by lia.
  assert (Hdiv : (2 | 1)%Z).
  { exists (Z.pow 2 (Z.of_nat n)). lia. }
  now destruct Hdiv as [k Hk]; lia.
Qed.

Lemma embed_IP_all_nonzero : forall m (M : Mask (m + m)),
  (m > 0)%nat ->
  ~ (embed (@IP_n_func (m + m)) M == 0).
Proof.
  intros m M Hm Habs.
  destruct (signed_walsh_IP_magnitude m M Hm) as [s Hs].
  assert (Hembed := embed_via_signed_walsh (m+m) (@IP_n_func (m+m)) M).
  
  (* signed_walsh ≠ 0 *)
  assert (Hsw_nz : ~ (signed_walsh (@IP_n_func (m+m)) M == 0)).
  { intro Hsw0.

    (* from Hs and Hsw0 *)
    assert (Hc : signed s * inject_Z (2 ^ Z.of_nat m) == 0)
      by (eapply Qeq_trans; [symmetry; exact Hs | exact Hsw0]).

    (* signed s ≠ 0 *)
    assert (Hsigned_nz : ~ (signed s == 0)).
    { destruct s; cbv [signed]; unfold Qeq; simpl; discriminate. }

    (* inject_Z (2^m) ≠ 0 *)
    assert (Hinj_nz : ~ (inject_Z (2 ^ Z.of_nat m) == 0)).
    { intro Hin0.
      (* turn inject_Z _ == 0 into Z equality *)
      assert (Hz0 : (2 ^ Z.of_nat m)%Z = 0%Z).
      {
        apply (QArith_base.inject_Z_injective (2 ^ Z.of_nat m) 0%Z).
        (* Goal: inject_Z (2^m) == inject_Z 0 *)
        (* But 0 : Q reduces to inject_Z 0 *)
        change (inject_Z (2 ^ Z.of_nat m) == inject_Z 0%Z).
        exact Hin0.
      }
      (* but Z.pow is never 0 when base ≠ 0 *)
      apply (Z.pow_nonzero 2 (Z.of_nat m)) in Hz0; try lia.
    }

    (* now split the product *)
    apply Qmult_integral in Hc as [Hbad | Hbad].
    - exact (Hsigned_nz Hbad).
    - exact (Hinj_nz Hbad).
  }
  
 (* From Hembed and Habs: the inner expression == 0 *)
  assert (H0 : (1 / pow2 (m + m) *
    ((1 # 2) * (if mask_eq_dec M mask_empty then pow2 (m + m) else 0) -
     (1 # 2) * signed_walsh (@IP_n_func (m + m)) M)) == 0).
  { eapply Qeq_trans; [symmetry; exact Hembed | exact Habs]. }

 (* 1/pow2 ≠ 0 *)
  assert (Hdiv_nz : ~ (1 / pow2 (m + m) == 0)).
  { intro Hk.
    assert (Hp : pow2 (m+m) * (1 / pow2 (m+m)) == 1)
      by (field; apply pow2_nonzero).
    setoid_rewrite Hk in Hp.
    assert (Hbad : (1 == 0)%Q) by (eapply Qeq_trans; [symmetry; exact Hp|]; ring).
    unfold Qeq in Hbad; simpl in Hbad; discriminate.
  }
  apply Qmult_integral in H0 as [H0 | H0].
  { exact (Hdiv_nz H0). }

  destruct (mask_eq_dec M mask_empty) as [Meq | Mneq].
  
  - (* M = empty: (1#2)*pow2(m+m) - (1#2)*sw == 0, so pow2(m+m) == sw *)
    subst M.
    assert (Hsw_eq : signed_walsh (@IP_n_func (m+m)) mask_empty == pow2 (m+m)).
    { assert (H' : (1#2) * (pow2 (m+m) - signed_walsh (@IP_n_func (m+m)) mask_empty) == 0)
        by (setoid_rewrite <- H0; ring).
      apply Qmult_integral in H' as [H'|H'].
      - unfold Qeq in H'; simpl in H'; discriminate.
      - setoid_replace (signed_walsh (@IP_n_func (m+m)) mask_empty)
          with (pow2 (m+m) - (pow2 (m+m) - signed_walsh (@IP_n_func (m+m)) mask_empty))%Q
          by ring.
        setoid_rewrite H'. ring. }
    (* So signed s * 2^m == 2^(m+m) *)
    assert (Hz : signed s * inject_Z (2 ^ Z.of_nat m) == pow2 (m+m)).
    { eapply Qeq_trans; [symmetry; exact Hs | exact Hsw_eq]. }
    setoid_rewrite pow2_injectZ in Hz.
    destruct s; unfold signed in Hz.
    
    + (* s = true: -1 * 2^m == 2^(m+m), impossible *)
      assert (Hbad : inject_Z (2 ^ Z.of_nat (m+m)) == -1 * inject_Z (2 ^ Z.of_nat m))
        by (eapply Qeq_trans; [symmetry; exact Hz |]; ring).
      unfold inject_Z, Qeq in Hbad; simpl in Hbad.
      rewrite !Z.mul_1_r in Hbad.
      pose proof (Z.pow_pos_nonneg 2 (Z.of_nat (m+m)) ltac:(lia) ltac:(lia)) as Hpos1.
      pose proof (Z.pow_pos_nonneg 2 (Z.of_nat m) ltac:(lia) ltac:(lia)) as Hpos2.
      destruct (2 ^ Z.of_nat m)%Z eqn:Em; lia.

    + (* s = false: 1 * 2^m == 2^(m+m), so 2^m == 2^(m+m), impossible for m>0 *)
      assert (Hbad : inject_Z (2 ^ Z.of_nat (m+m)) == inject_Z (2 ^ Z.of_nat m)).
      { eapply Qeq_trans; [symmetry; exact Hz |]; ring. }
      unfold inject_Z, Qeq in Hbad; simpl in Hbad.
      rewrite !Z.mul_1_r in Hbad.
      rewrite Nat2Z.inj_add, Z.pow_add_r in Hbad by lia.
      pose proof (Z.pow_pos_nonneg 2 (Z.of_nat m) ltac:(lia) ltac:(lia)).
      assert (H2m1 : (2 ^ Z.of_nat m = 1)%Z) by nia.
      apply Zpow2_nat_eq_1 in H2m1; lia.

  - (* M ≠ empty: -(1#2)*sw == 0, so sw == 0, contradiction *)
    apply Hsw_nz.
    assert (H' : (-(1#2)) * signed_walsh (@IP_n_func (m+m)) M == 0).
    { setoid_rewrite <- H0; ring. }
    apply Qmult_integral in H' as [H'|H'].
    + unfold Qeq in H'; simpl in H'; discriminate.
    + exact H'.
Qed.

(* Step 10: Full support follows *)
Lemma support_size_IP : forall m,
  (m > 0)%nat ->
  support_size (embed (@IP_n_func (m + m))) = Nat.pow 2 (m + m).
Proof.
  intros m Hm.
  assert (Hle := support_size_le_2n (m + m) (embed (@IP_n_func (m + m)))).
  assert (Hge : (Nat.pow 2 (m + m) <= support_size (embed (@IP_n_func (m + m))))%nat).
  { rewrite <- (all_masks_length_pow2 (m + m)).
    unfold support_size.
    apply NoDup_incl_length.
    - apply all_masks_nodup.
    - intros x Hin.
      apply filter_In. split; [exact Hin|].
      (* Show negb (Qeq_bool (embed IP x) 0) = true *)
      destruct (Qeq_bool (embed (@IP_n_func (m + m)) x) 0) eqn:Heq.
      + exfalso.
        apply (embed_IP_all_nonzero m x Hm).
        apply Qeq_bool_eq. exact Heq.
      + reflexivity. }
  lia.
Qed.

        (* The support-based separation theorem *)

Theorem IP_formula_size_lower_bound :
  forall m (sq : Vector.t Q (m + m)) (phi : BoolFormula (m + m)),
    (m > 0)%nat ->
    (forall i, Vector.nth sq i == 1) ->
    eval_bf phi = @IP_n_func (m + m) ->
    (formula_size phi >= m)%nat.
Proof.
  intros m sq phi Hm Hsq Hbf.

  (* Step 1: translate_correct + Hbf gives pointwise equality *)
  assert (Hpt : forall mm : Mask (m + m),
    eval_expr sq (translate phi) mm == embed (@IP_n_func (m + m)) mm).
  {
    intro mm.
    rewrite (translate_correct (m + m) sq phi Hsq mm).
    rewrite Hbf. apply Qeq_refl.
  }

  (* Step 2: support sizes are equal *)
  assert (Hss : support_size (eval_expr sq (translate phi))
                = support_size (embed (@IP_n_func (m + m)))).
  { apply (support_size_ext (m + m) _ _ Hpt). }

  (* Step 3: upper bound from formula structure *)
  assert (Hub : (support_size (eval_expr sq (translate phi))
                 <= Nat.pow 2 (formula_size phi))%nat).
  { apply translate_support_size_bound. }

  (* Step 4: lower bound from IP Fourier analysis *)
  assert (Hlb : support_size (embed (@IP_n_func (m + m)))
                = Nat.pow 2 (m + m)).
  { apply support_size_IP. exact Hm. }

  (* Step 5: chain:  2^(m+m) = supp(embed IP) = supp(eval_expr ...) ≤ 2^(formula_size phi) *)
  assert (Hpow : (Nat.pow 2 (m + m) <= Nat.pow 2 (formula_size phi))%nat).
  { lia. }

  (* Step 6: monotonicity of 2^x gives m+m ≤ formula_size phi, hence m ≤ formula_size phi *)
  destruct (le_gt_dec m (formula_size phi)) as [Hle|Hlt]; [exact Hle|].
  exfalso.
  assert (Hfs : (formula_size phi < m + m)%nat) by lia.
  assert (Hpow2 : (Nat.pow 2 (formula_size phi) < Nat.pow 2 (m + m))%nat).
  { apply Nat.pow_lt_mono_r; lia. }
  lia.
Qed.

Corollary IP_formula_size_lower_bound_tight :
  forall m (sq : Vector.t Q (m + m)) (phi : BoolFormula (m + m)),
    (m > 0)%nat ->
    (forall i, Vector.nth sq i == 1) ->
    eval_bf phi = @IP_n_func (m + m) ->
    (formula_size phi >= m + m)%nat.
Proof.
  intros m sq phi Hm Hsq Hbf.

  (* Pointwise equality of the translated GA object with embed(IP) *)
  assert (Hpt : forall mm : Mask (m + m),
            eval_expr sq (translate phi) mm
            == embed (@IP_n_func (m + m)) mm).
  {
    intro mm.
    rewrite (translate_correct (m + m) sq phi Hsq mm).
    rewrite Hbf.
    apply Qeq_refl.
  }

  (* Hence equal support sizes *)
  assert (Hss :
    support_size (eval_expr sq (translate phi))
    = support_size (embed (@IP_n_func (m + m)))).
  { apply (support_size_ext (m + m) _ _ Hpt). }

  (* Structural upper bound: support_size ≤ 2^(formula_size) *)
  assert (Hub :
    (support_size (eval_expr sq (translate phi))
     <= Nat.pow 2 (formula_size phi))%nat).
  { apply translate_support_size_bound. }

  (* Fourier fact: IP has full support *)
  assert (Hlb :
    support_size (embed (@IP_n_func (m + m)))
    = Nat.pow 2 (m + m)).
  { apply support_size_IP; exact Hm. }

  (* Chain them to get 2^(2m) ≤ 2^(formula_size phi) *)
  assert (Hpowle : (Nat.pow 2 (m + m) <= Nat.pow 2 (formula_size phi))%nat).
  {
    (* Goal: 2^(m+m) <= 2^(size) *)
    rewrite <- Hlb.          (* replace 2^(m+m) by support_size(embed IP) *)
    rewrite <- Hss.          (* replace support_size(embed IP) by support_size(eval_expr ...) *)
    exact Hub.
  }

  (* Monotonicity of pow base 2 gives (m+m) ≤ formula_size phi *)
  destruct (le_gt_dec (m + m) (formula_size phi)) as [Hle | Hgt].
  - exact Hle.
  - exfalso.
    assert (Hlt : (formula_size phi < m + m)%nat) by lia.
    assert (Hpowlt : (Nat.pow 2 (formula_size phi) < Nat.pow 2 (m + m))%nat).
    { apply Nat.pow_lt_mono_r; lia. }
    lia.
Qed.

Lemma formula_size_lower_bound_from_support :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n) (f : Corner n -> bool),
    (forall i, Vector.nth sq i == 1) ->
    eval_bf phi = f ->
    (support_size (embed f) <= Nat.pow 2 (formula_size phi))%nat.
Proof.
  intros n sq phi f Hsq Heq.

  (* Pointwise equality between eval_expr(translate phi) and embed f *)
  assert (Hpt :
    forall m : Mask n,
      eval_expr sq (translate phi) m == embed f m).
  {
    intro m.
    rewrite (translate_correct n sq phi Hsq m).
    rewrite Heq.
    apply Qeq_refl.
  }

  (* Equal support sizes by extensionality *)
  assert (Hss :
    support_size (eval_expr sq (translate phi))
    = support_size (embed f)).
  { apply (support_size_ext n _ _ Hpt). }

  (* Structural upper bound on translated expressions *)
  assert (Hub :
    (support_size (eval_expr sq (translate phi))
     <= Nat.pow 2 (formula_size phi))%nat).
  { apply translate_support_size_bound. }

  (* Finish by rewriting Hub using Hss *)
  rewrite <- Hss.
  exact Hub.
Qed.

Corollary formula_size_lower_bound_from_support_k :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n) (f : Corner n -> bool) k,
    (forall i, Vector.nth sq i == 1) ->
    eval_bf phi = f ->
    (k <= support_size (embed f))%nat ->
    (k <= Nat.pow 2 (formula_size phi))%nat.
Proof.
  intros n sq phi f k Hsq Heq Hk.
  eapply Nat.le_trans.
  - exact Hk.
  - apply (formula_size_lower_bound_from_support n sq phi f Hsq Heq).
Qed.

Corollary formula_size_lower_bound_from_support_pow2 :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n) (f : Corner n -> bool) t,
    (forall i, Vector.nth sq i == 1) ->
    eval_bf phi = f ->
    (Nat.pow 2 t <= support_size (embed f))%nat ->
    (t <= formula_size phi)%nat.
Proof.
  intros n sq phi f t Hsq Heq Hpow.

  assert (Hupper :
    (support_size (embed f) <= Nat.pow 2 (formula_size phi))%nat).
  { apply (formula_size_lower_bound_from_support n sq phi f Hsq Heq). }

  assert (Hle :
    (Nat.pow 2 t <= Nat.pow 2 (formula_size phi))%nat).
  {
    eapply Nat.le_trans.
    - exact Hpow.
    - exact Hupper.
  }

  destruct (le_gt_dec t (formula_size phi)) as [Hts | Hts].
  - exact Hts.
  - exfalso.
    assert (Hlt : (formula_size phi < t)%nat) by lia.
    assert (Hpowlt : (Nat.pow 2 (formula_size phi) < Nat.pow 2 t)%nat).
    { apply Nat.pow_lt_mono_r; lia. }
    lia.
Qed.

Corollary formula_size_full_support :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n) (f : Corner n -> bool),
    (forall i, Vector.nth sq i == 1) ->
    eval_bf phi = f ->
    support_size (embed f) = Nat.pow 2 n ->
    (formula_size phi >= n)%nat.
Proof.
  intros n sq phi f Hsq Heq Hfull.
  (* turn equality into the >= hypothesis needed by _pow2 *)
  apply (formula_size_lower_bound_from_support_pow2
           n sq phi f n Hsq Heq).
  rewrite Hfull.
  apply Nat.le_refl.
Qed.

(* Eval compute in (support_size (embed (@XOR_n_func 3))).  == 2 *)

Corollary support_size_embed_XOR :
  forall n,
    (n > 0)%nat ->
    support_size (embed (@XOR_n_func n)) = 2%nat.
Proof.
  intros n Hn.
  apply support_size_XOR; exact Hn.
Qed.

Corollary IP_formula_size_lower_bound_via_full_support :
  forall m (sq : Vector.t Q (m + m)) (phi : BoolFormula (m + m)),
    (m > 0)%nat ->
    (forall i, Vector.nth sq i == 1) ->
    eval_bf phi = @IP_n_func (m + m) ->
    (formula_size phi >= m + m)%nat.
Proof.
  intros m sq phi Hm Hsq Heq.
  (* Use the full-support corollary specialized to n := m+m and f := IP *)
  eapply formula_size_full_support.
  - exact Hsq.
  - exact Heq.
  - apply support_size_IP; exact Hm.
Qed.


Corollary XOR_support_size_small :
  forall n,
    (n > 0)%nat ->
    (support_size (embed (@XOR_n_func n)) <= 2)%nat.
Proof.
  intros n Hn.
  rewrite (support_size_XOR n Hn).
  lia.
Qed.

Corollary full_support_implies_size_ge_n :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n) (f : Corner n -> bool),
    (forall i, Vector.nth sq i == 1) ->
    eval_bf phi = f ->
    support_size (embed f) = Nat.pow 2 n ->
    (formula_size phi >= n)%nat.
Proof.
  intros n sq phi f Hsq Heq Hfull.
  apply (formula_size_full_support n sq phi f Hsq Heq Hfull).
Qed.

Fixpoint occurs_var {n} (i : Fin.t n) (phi : BoolFormula n) : Prop :=
  match phi with
  | BVar j     => i = j
  | BConst _   => False
  | BAnd p q   => occurs_var i p \/ occurs_var i q
  | BOr  p q   => occurs_var i p \/ occurs_var i q
  | BNot p     => occurs_var i p
  end.

Lemma length_filter_eq_Forall :
  forall (A : Type) (p : A -> bool) (l : list A),
    length (List.filter p l) = length l ->
    List.Forall (fun x => p x = true) l.
Proof.
  intros A p l.
  induction l as [|a l IH]; simpl; intro Hlen.
  - constructor.
  - destruct (p a) eqn:Ha.
    + constructor.
      * exact Ha.
      * apply IH.
        apply Nat.succ_inj.
        exact Hlen.
    + exfalso.
      (* Here: p a = false, so filter (a::l) = filter l and Hlen becomes: length(filter p l) = S(length l) *)
      simpl in Hlen.

      (* Use the partition identity to get length(filter p l) <= length l *)
      assert (Hpart :
        (length (List.filter p l) +
         length (List.filter (fun x => negb (p x)) l))%nat = length l).
      { apply List.filter_length. }

      (* Hence length(filter p l) <= length l *)
      assert (Hle : (length (List.filter p l) <= length l)%nat).
      { rewrite <- Hpart. apply Nat.le_add_r. }

      lia.
Qed.

Lemma full_support_all_masks_nz :
  forall n (F : MV n),
    support_size F = Nat.pow 2 n ->
    forall m, List.In m (all_masks n) ->
      negb (Qeq_bool (F m) 0) = true.
Proof.
  intros n F Hfull m Hin.
  unfold support_size in Hfull.
  set (p := fun mm : Mask n => negb (Qeq_bool (F mm) 0)) in *.

  assert (Hall_len : length (all_masks n) = Nat.pow 2 n).
  { apply all_masks_length_pow2. }

  assert (Hlen : length (List.filter p (all_masks n)) = length (all_masks n)).
  { rewrite Hall_len; exact Hfull. }

  pose proof (length_filter_eq_Forall (Mask n) p (all_masks n) Hlen) as Hfor.
  (* Hfor : Forall (fun x => p x = true) (all_masks n) *)

  (* Convert Forall into a pointwise fact using proj1 of the <-> lemma *)
  pose proof (proj1 (List.Forall_forall (fun x : Mask n => p x = true) (all_masks n)) Hfor)
    as Hall.

  (* Now apply to m *)
  unfold p in Hall.
  exact (Hall m Hin).
Qed.

Lemma nz_bool_to_neq0 :
  forall q : Q,
    negb (Qeq_bool q 0) = true ->
    ~ q == 0.
Proof.
  intros q Hnz Heq.
  assert (Hb : Qeq_bool q 0 = true).
  { apply Qeq_eq_bool. exact Heq. }
  rewrite Hb in Hnz.
  simpl in Hnz.
  discriminate.
Qed.

Lemma full_support_all_nonzero :
  forall n (F : MV n),
    support_size F = Nat.pow 2 n ->
    forall m, In m (all_masks n) -> ~(F m == 0).
Proof.
  intros n F Hfull m Hin Heq.
  (* from full support, boolean-nonzero holds on all masks *)
  pose proof (full_support_all_masks_nz n F Hfull m Hin) as Hnz_bool.
  (* convert boolean-nonzero to Prop-nonzero *)
  apply (nz_bool_to_neq0 (F m) Hnz_bool).
  exact Heq.
Qed.

Lemma eval_bf_independent_if_not_occurs :
  forall n (phi : BoolFormula n) i,
    ~ occurs_var (n:=n) i phi ->
    forall c1 c2 : Corner n,
      (forall j, j <> i -> Vector.nth c1 j = Vector.nth c2 j) ->
      eval_bf phi c1 = eval_bf phi c2.
Proof.
  intros n phi.
  induction phi as [j|b|p IHp q IHq|p IHp q IHq|p IHp]; intros i Hno c1 c2 Hagree; simpl.
  - (* BVar j *)
    simpl in Hno.
    assert (Hij : j <> i).
    { intro Hji. apply Hno. subst. reflexivity. }
    (* use Hagree at index j *)
    specialize (Hagree j Hij).
    (* eval_bf reads Vector.nth at j; rewrite it *)
    rewrite Hagree. reflexivity.
  - (* BConst *)
    reflexivity.
 - (* BAnd *)
    simpl in Hno.
    assert (Hno_p : ~ occurs_var i p).
    { intro Hp. apply Hno. left. exact Hp. }
    assert (Hno_q : ~ occurs_var i q).
    { intro Hq. apply Hno. right. exact Hq. }
    rewrite (IHp i Hno_p c1 c2 Hagree).
    rewrite (IHq i Hno_q c1 c2 Hagree).
    reflexivity.

  - (* BOr *)
    simpl in Hno.
    assert (Hno_p : ~ occurs_var i p).
    { intro Hp. apply Hno. left. exact Hp. }
    assert (Hno_q : ~ occurs_var i q).
    { intro Hq. apply Hno. right. exact Hq. }
    rewrite (IHp i Hno_p c1 c2 Hagree).
    rewrite (IHq i Hno_q c1 c2 Hagree).
    reflexivity.

  - (* BNot *)
    simpl in Hno.
    rewrite (IHp i Hno c1 c2 Hagree).
    reflexivity.
Qed.

Require Import Classical_Prop.

Lemma inner_sum_independent_zero :
  forall n (i : Fin.t n) (f : Corner n -> bool),
    (forall c1 c2 : Corner n,
      (forall j : Fin.t n, j <> i -> Vector.nth c1 j = Vector.nth c2 j) ->
      f c1 = f c2) ->
    sumQ (List.map (fun a => (bQ (f a) * sQ (Vector.nth a i))%Q)
                   (all_corners n)) == 0.
Proof.
  induction n as [|n' IHn]; intros i f Hind.
  - inversion i.
  - dependent destruction i.
    + (* i = F1: pair Pos-headed with Neg-headed, cancel by independence *)
      simpl (all_corners (S n')).
      rewrite map_app, !map_map, sumQ_app.
      (* Combine the two halves into a single sum of paired terms *)
      rewrite <- sumQ_map_add.
      apply sumQ_map_all_zero.
      intros t _.
      (* Vector.nth (Pos :: t) F1 = Pos, Vector.nth (Neg :: t) F1 = Neg *)
      simpl Vector.nth. simpl sQ.
      (* Independence: f(Pos::t) = f(Neg::t) *)
      assert (Heq : f (Vector.cons _ Pos _ t) = f (Vector.cons _ Neg _ t)).
      { apply Hind. intros j Hj.
        dependent destruction j.
        - exfalso; apply Hj; reflexivity.
        - simpl. reflexivity. }
      rewrite Heq. ring.
    + (* i = FS i0 *)
      simpl (all_corners (S n')).
      rewrite map_app, !map_map, sumQ_app.
      (* Both halves simplify: Vector.nth (h :: t) (FS i) = Vector.nth t i *)
      (* Apply IH to each half directly *)
      assert (HIH_pos :
        sumQ (List.map (fun t : Corner n' =>
          (bQ (f (Vector.cons _ Pos _ t)) * sQ (Vector.nth t i))%Q)
          (all_corners n')) == 0).
      { apply (IHn i (fun t => f (Vector.cons _ Pos _ t))).
        intros c1 c2 Hagree.
        apply Hind. intros j Hj.
        dependent destruction j.
        - simpl. reflexivity.
        - simpl. apply Hagree.
          intro Heq; subst; exact (Hj eq_refl). }
      assert (HIH_neg :
        sumQ (List.map (fun t : Corner n' =>
          (bQ (f (Vector.cons _ Neg _ t)) * sQ (Vector.nth t i))%Q)
          (all_corners n')) == 0).
      { apply (IHn i (fun t => f (Vector.cons _ Neg _ t))).
        intros c1 c2 Hagree.
        apply Hind. intros j Hj.
        dependent destruction j.
        - simpl. reflexivity.
        - simpl. apply Hagree.
          intro Heq; subst; exact (Hj eq_refl). }
      (* Now rewrite the goal to match these *)
      assert (Hgoal :
        sumQ (List.map (fun x : Corner n' =>
          (bQ (f (Vector.cons _ Pos _ x)) * sQ (Vector.nth (Vector.cons _ Pos _ x) (Fin.FS i)))%Q)
          (all_corners n'))
        ==
        sumQ (List.map (fun t : Corner n' =>
          (bQ (f (Vector.cons _ Pos _ t)) * sQ (Vector.nth t i))%Q)
          (all_corners n'))).
      { apply sumQ_map_ext; intros t _. simpl Vector.nth. reflexivity. }
      assert (Hgoal2 :
        sumQ (List.map (fun x : Corner n' =>
          (bQ (f (Vector.cons _ Neg _ x)) * sQ (Vector.nth (Vector.cons _ Neg _ x) (Fin.FS i)))%Q)
          (all_corners n'))
        ==
        sumQ (List.map (fun t : Corner n' =>
          (bQ (f (Vector.cons _ Neg _ t)) * sQ (Vector.nth t i))%Q)
          (all_corners n'))).
      { apply sumQ_map_ext; intros t _. simpl Vector.nth. reflexivity. }
      rewrite Hgoal, Hgoal2, HIH_pos, HIH_neg. ring.
Qed.

Lemma embed_mask_single_zero_if_independent :
  forall n (f : Corner n -> bool) (i : Fin.t n),
    (forall c1 c2 : Corner n,
      (forall j, j <> i -> Vector.nth c1 j = Vector.nth c2 j) ->
      f c1 = f c2) ->
    embed f (mask_single i) == 0.
Proof.
  intros n f i Hind.
  unfold embed, Pi.
  eapply Qeq_trans.
  { apply (@sumQ_map_ext (Corner n)
      (fun a => bQ (f a) * (1 / pow2 n * chi (mask_single i) a))%Q
      (fun a => (1 / pow2 n) * (bQ (f a) * sQ (Vector.nth a i)))%Q
      (all_corners n)).
    intros a _.
    rewrite chi_mask_single. ring. }
  rewrite sumQ_map_scale_l.
  rewrite (inner_sum_independent_zero n i f Hind).
  ring.
Qed.

Corollary full_support_implies_reads_all :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n) (f : Corner n -> bool),
    (forall i, Vector.nth sq i == 1) ->
    eval_bf phi = f ->
    support_size (embed f) = Nat.pow 2 n ->
    forall i, occurs_var i phi.
Proof.
  intros n sq phi f Hsq Heq Hfull i.

  destruct (classic (occurs_var i phi)) as [Hocc | Hno].
  - exact Hocc.
  - (* now Hno : ~ occurs_var i phi, derive contradiction *)
    assert (Hind :
      forall c1 c2 : Corner n,
        (forall j, j <> i -> Vector.nth c1 j = Vector.nth c2 j) ->
        f c1 = f c2).
    {
      intros c1 c2 Hagree.
      rewrite <- Heq.
      apply (eval_bf_independent_if_not_occurs n phi i Hno c1 c2 Hagree).
    }

    assert (Hzero : embed f (mask_single i) == 0).
    { apply (embed_mask_single_zero_if_independent n f i Hind). }

    assert (Hnz : ~ embed f (mask_single i) == 0).
    {
      apply (full_support_all_nonzero n (embed f) Hfull (mask_single i)).
      apply all_masks_complete.
    }

    exfalso. exact (Hnz Hzero).
Qed.

Theorem max_grade_of_full_support :
  forall n (F : MV n),
    support_size F = Nat.pow 2 n ->
    max_grade F = n.
Proof.
  intros n F Hfull.

  assert (Hnz_bool : negb (Qeq_bool (F (Vector.const true n)) 0) = true).
  {
    apply (full_support_all_masks_nz n F Hfull).
    apply all_masks_complete.
  }

  assert (Hnz : ~ F (Vector.const true n) == 0).
  { apply nz_bool_to_neq0. exact Hnz_bool. }

  assert (Hge : (n <= max_grade F)%nat).
  {
    pose proof (max_grade_spec F (Vector.const true n) Hnz) as Hle.
    rewrite grade_full_mask in Hle.
    exact Hle.
  }

  pose proof (max_grade_le_n F) as Hle_n.
  lia.
Qed.

Theorem full_support_implies_max_grade_n :
  forall n (F : MV n),
    support_size F = Nat.pow 2 n ->
    max_grade F = n.
Proof.
  intros n F Hfull.
  set (U := Vector.const true n).

  (* boolean nonzero from full support *)
  assert (HnzU_bool : negb (Qeq_bool (F U) 0) = true).
  {
    apply (full_support_all_masks_nz n F Hfull U).
    apply all_masks_complete.
  }

  (* convert boolean nonzero to Prop nonzero *)
  assert (HnzU : ~ F U == 0).
  { apply nz_bool_to_neq0. exact HnzU_bool. }

  (* grade(U)=n <= max_grade F *)
  assert (Hge : (n <= max_grade F)%nat).
  {
    pose proof (max_grade_spec F U HnzU) as Hle.
    rewrite grade_full_mask in Hle.
    exact Hle.
  }

  (* always max_grade F <= n *)
  pose proof (max_grade_le_n F) as Hle_n.
  lia.
Qed.

Corollary support_or_max_grade :
  forall n (F : MV n),
    support_size F <> Nat.pow 2 n \/ max_grade F = n.
Proof.
  intros n F.
  destruct (Nat.eq_dec (support_size F) (Nat.pow 2 n)) as [Heq|Hneq].
  - right. apply full_support_implies_max_grade_n. exact Heq.
  - left. exact Hneq.
Qed.

