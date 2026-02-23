(*
  ============================================================
  File: Cln_CompositeExcursion.v
  ============================================================
*)

Require Import Cln_Full.
Require Import Cln_Grade.
Require Import Cln_finite_l1_submultiplicativity.
Require Import Cln_BoolDist.

From Coq Require Import FunctionalExtensionality.
From Coq Require Import List Lia Arith.
From Coq Require Import QArith.
From Coq Require Import QArith.QArith_base.
From Coq Require Import QArith.Qabs.
From Coq Require Import Bool.

Import ListNotations.
Open Scope Q_scope.
Set Implicit Arguments.

Definition Qmax (a b : Q) : Q :=
  if Qle_bool a b then b else a.

Lemma Qmax_l : forall a b, a <= Qmax a b.
Proof.
  intros a b.
  unfold Qmax.
  destruct (Qle_bool a b) eqn:H.
  - apply Qle_bool_iff in H. exact H.
  - apply Qle_refl.
Qed.

Lemma Qmax_r : forall a b, b <= Qmax a b.
Proof.
  intros a b; unfold Qmax.
  destruct (Qle_bool a b) eqn:H.
  - apply Qle_refl.
  -
    assert (Hnot : ~(a <= b)).
    { intro Hab.
      apply (proj2 (Qle_bool_iff a b)) in Hab.
      rewrite Hab in H; discriminate.
    }
    destruct (Qcompare_spec b a) as [Hc | Hc | Hc].
    all: try (
      apply Qlt_le_weak; exact Hc
    ).
    all: try (
      rewrite Hc; apply Qle_refl
    ).
    exfalso.
    apply Hnot.
    apply Qlt_le_weak.
    exact Hc.
Qed.

Fixpoint list_Qmax (xs : list Q) : Q :=
  match xs with
  | [] => 0
  | x :: tl => Qmax x (list_Qmax tl)
  end.

Fixpoint max_l1_during {n} (sq : Vector.t Q n) (e : GA_expr n) : Q :=
  match e with
  | Basis _   => 1
  | Scalar c  => Qabs c
  | Cln_Grade.Add e1 e2 =>
      max_l1_during sq e1 + max_l1_during sq e2
  | Mul e1 e2 =>
      Qmax (Qmax (max_l1_during sq e1) (max_l1_during sq e2))
           (l1_norm (mv_gp sq (eval_expr sq e1) (eval_expr sq e2)))
  | Conv e1 e2 =>
      Qmax (Qmax (max_l1_during sq e1) (max_l1_during sq e2))
           (l1_norm (mv_conv (eval_expr sq e1) (eval_expr sq e2)))
  end.

Lemma l1_eval_le_max_l1_during :
  forall n (sq : Vector.t Q n) (e : GA_expr n),
    l1_norm (eval_expr sq e) <= max_l1_during sq e.
Proof.
  intros n sq e.
  induction e as [i|c|e1 IH1 e2 IH2|e1 IH1 e2 IH2|e1 IH1 e2 IH2]; simpl.
  - apply Qle_of_Qeq. apply l1_norm_basis.
  - apply Qle_of_Qeq. apply l1_norm_scale_one.
  -
    eapply Qle_trans.
    + apply l1_add_bound.
    +
      apply Qplus_le_compat; assumption.
  - apply Qmax_r.
  - apply Qmax_r.
Qed.

Definition dist_to {n} (F : MV n) (f : Corner n -> bool) : Q :=
  bool_dist_wrt F f.

Definition boolish_le {n} (F : MV n) (d : Q) : Prop :=
  exists g : Corner n -> bool, bool_dist_wrt F g <= d.

Fixpoint max_booldist_during {n}
  (sq : Vector.t Q n)
  (f  : Corner n -> bool)
  (e  : GA_expr n) : Q :=
  match e with
  | Basis _  => dist_to (eval_expr sq e) f
  | Scalar _ => dist_to (eval_expr sq e) f
  | Cln_Grade.Add e1 e2 =>
      Qmax
        (Qmax (max_booldist_during sq f e1)
              (max_booldist_during sq f e2))
        (dist_to (eval_expr sq (Cln_Grade.Add e1 e2)) f)
  | Mul e1 e2 =>
      Qmax
        (Qmax (max_booldist_during sq f e1)
              (max_booldist_during sq f e2))
        (dist_to (mv_gp sq (eval_expr sq e1) (eval_expr sq e2)) f)
  | Conv e1 e2 =>
      Qmax
        (Qmax (max_booldist_during sq f e1)
              (max_booldist_during sq f e2))
        (dist_to (mv_conv (eval_expr sq e1) (eval_expr sq e2)) f)
  end.

Lemma booldist_eval_le_max_booldist_during :
  forall n (sq : Vector.t Q n) (f : Corner n -> bool) (e : GA_expr n),
    dist_to (eval_expr sq e) f <= max_booldist_during sq f e.
Proof.
  intros n sq f e.
  induction e as [i|c|e1 IH1 e2 IH2|e1 IH1 e2 IH2|e1 IH1 e2 IH2]; simpl.
  - apply Qle_refl.
  - apply Qle_refl.
  - (* Add *) apply Qmax_r.
  - (* Mul *) apply Qmax_r.
  - (* Conv *) apply Qmax_r.
Qed.

Fixpoint trace_boolish_le {n} (sq : Vector.t Q n) (e : GA_expr n) (d : Q) : Prop :=
  match e with
  | Basis _ | Scalar _ => boolish_le (eval_expr sq e) d
  | Cln_Grade.Add e1 e2 =>
      trace_boolish_le sq e1 d /\ trace_boolish_le sq e2 d /\
      boolish_le (eval_expr sq (Cln_Grade.Add e1 e2)) d
  | Mul e1 e2 =>
      trace_boolish_le sq e1 d /\ trace_boolish_le sq e2 d /\
      boolish_le (mv_gp sq (eval_expr sq e1) (eval_expr sq e2)) d
  | Conv e1 e2 =>
      trace_boolish_le sq e1 d /\ trace_boolish_le sq e2 d /\
      boolish_le (mv_conv (eval_expr sq e1) (eval_expr sq e2)) d
  end.

Record ExcNum := {
  exc_grade : nat;
  exc_l1    : Q;
}.

Definition exc_of {n} (sq : Vector.t Q n) (e : GA_expr n) : ExcNum :=
  {| exc_grade := max_grade_during sq e;
     exc_l1    := max_l1_during sq e |}.

Definition exc_le (a b : ExcNum) : Prop :=
  (exc_grade a < exc_grade b)%nat \/
  (exc_grade a = exc_grade b /\ exc_l1 a < exc_l1 b) \/
  (exc_grade a = exc_grade b /\ exc_l1 a == exc_l1 b).

Definition exc_pre (a b : ExcNum) : Prop :=
  (exc_grade a <= exc_grade b)%nat /\ exc_l1 a <= exc_l1 b.

Definition computes {n} (sq : Vector.t Q n) (e : GA_expr n) (f : Corner n -> bool) : Prop :=
  forall m : Mask n, eval_expr sq e m == embed f m.

Fixpoint expr_size {n} (e : GA_expr n) : nat :=
  match e with
  | Basis _ => 1
  | Scalar _ => 1
  | Cln_Grade.Add e1 e2 => 1 + expr_size e1 + expr_size e2
  | Mul e1 e2 => 1 + expr_size e1 + expr_size e2
  | Conv e1 e2 => 1 + expr_size e1 + expr_size e2
  end.

Definition easy_under
  (B : nat -> ExcNum) (d : Q)
  {n} (sq : Vector.t Q n) (f : Corner n -> bool) : Prop :=
  exists e : GA_expr n,
    computes sq e f /\
    exc_pre (exc_of sq e) (B (expr_size e)) /\
    trace_boolish_le sq e d.

Definition hard_under
  (B : nat -> ExcNum) (d : Q)
  {n} (sq : Vector.t Q n) (f : Corner n -> bool) : Prop :=
  forall e : GA_expr n,
    computes sq e f ->
    ~ (exc_pre (exc_of sq e) (B (expr_size e)) /\ trace_boolish_le sq e d).

Lemma exc_pre_refl : forall a, exc_pre a a.
Proof.
  intro a; split; [lia | apply Qle_refl].
Qed.

Lemma exc_pre_trans : forall a b c,
  exc_pre a b -> exc_pre b c -> exc_pre a c.
Proof.
  intros a b c Hab Hbc.
  destruct Hab as [Hg1 Hl1].
  destruct Hbc as [Hg2 Hl2].
  repeat split.
  - lia.
  - eapply Qle_trans; eauto.
Qed.

Definition exc_score (E : ExcNum) : Q :=
  (inject_Z (Z.of_nat (exc_grade E))) + exc_l1 E.

Parameter proj_bool : forall n, MV n -> Corner n -> bool.

Definition represents {n} (sq : Vector.t Q n) (e : GA_expr n) (f : Corner n -> bool) : Prop :=
  forall x : Corner n, proj_bool (eval_expr sq e) x = f x.

Definition looks_hard (B : nat -> ExcNum) (d : Q) {n} (sq : Vector.t Q n) (f : Corner n -> bool) : Prop :=
  exists e : GA_expr n,
    computes sq e f /\
    ~ (exc_pre (exc_of sq e) (B (expr_size e)) /\ trace_boolish_le sq e d).

Lemma dist_to_unfold :
  forall n (F : MV n) (f : Corner n -> bool),
    dist_to F f == l1_norm (mv_sub F (embed f)).
Proof. intros; reflexivity. Qed.

(* Every correct program ends with BoolDist == 0 *)
Lemma computes_implies_final_booldist_zero :
  forall n (sq : Vector.t Q n) (e : GA_expr n) (f : Corner n -> bool),
    computes sq e f ->
    dist_to (eval_expr sq e) f == 0.
Proof.
  intros n sq e f Hcomp.
  unfold dist_to, bool_dist_wrt.
  assert (Hext :
    l1_norm (mv_sub (eval_expr sq e) (embed f)) == l1_norm (@mv_zero n)).
  { apply l1_norm_ext.
    intro m.
    unfold mv_sub, mv_zero.
    specialize (Hcomp m).
    rewrite Hcomp.
    ring.
  }
  eapply Qeq_trans.
  - exact Hext.
  -
    unfold l1_norm, mv_zero.
    eapply Qeq_trans.
    + apply sumQ_map_ext. intros m _.
      rewrite Qabs_pos; [reflexivity | apply Qle_refl].
    + apply sumQ_map_const0.
Qed.

Lemma trace_boolish_le_sub_left :
  forall n (sq : Vector.t Q n) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_le sq (Cln_Grade.Add e1 e2) d -> trace_boolish_le sq e1 d.
Proof.
  intros n sq d e1 e2 H.
  simpl in H.
  tauto.
Qed.

Lemma trace_boolish_le_sub_right_Add :
  forall n (sq : Vector.t Q n) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_le sq (Cln_Grade.Add e1 e2) d ->
    trace_boolish_le sq e2 d.
Proof.
  intros n sq d e1 e2 H.
  simpl in H.
  tauto.
Qed.

Lemma trace_boolish_le_sub_left_Mul :
  forall n (sq : Vector.t Q n) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_le sq (Mul e1 e2) d ->
    trace_boolish_le sq e1 d.
Proof.
  intros n sq d e1 e2 H.
  simpl in H.
  tauto.
Qed.

Lemma trace_boolish_le_sub_right_Mul :
  forall n (sq : Vector.t Q n) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_le sq (Mul e1 e2) d ->
    trace_boolish_le sq e2 d.
Proof.
  intros n sq d e1 e2 H.
  simpl in H.
  tauto.
Qed.

Lemma trace_boolish_le_sub_left_Conv :
  forall n (sq : Vector.t Q n) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_le sq (Conv e1 e2) d ->
    trace_boolish_le sq e1 d.
Proof.
  intros n sq d e1 e2 H.
  simpl in H.
  tauto.
Qed.

Lemma trace_boolish_le_sub_right_Conv :
  forall n (sq : Vector.t Q n) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_le sq (Conv e1 e2) d ->
    trace_boolish_le sq e2 d.
Proof.
  intros n sq d e1 e2 H.
  simpl in H.
  tauto.
Qed.

Lemma trace_boolish_le_node_Add :
  forall n (sq : Vector.t Q n) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_le sq (Cln_Grade.Add e1 e2) d ->
    boolish_le (eval_expr sq (Cln_Grade.Add e1 e2)) d.
Proof.
  intros n sq d e1 e2 H.
  simpl in H.
  tauto.
Qed.

Lemma trace_boolish_le_node_Mul :
  forall n (sq : Vector.t Q n) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_le sq (Mul e1 e2) d ->
    boolish_le (mv_gp sq (eval_expr sq e1) (eval_expr sq e2)) d.
Proof.
  intros n sq d e1 e2 H.
  simpl in H.
  tauto.
Qed.

Lemma trace_boolish_le_node_Conv :
  forall n (sq : Vector.t Q n) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_le sq (Conv e1 e2) d ->
    boolish_le (mv_conv (eval_expr sq e1) (eval_expr sq e2)) d.
Proof.
  intros n sq d e1 e2 H.
  simpl in H.
  tauto.
Qed.

Lemma computes_implies_final_boolish_0 :
  forall n (sq : Vector.t Q n) (e : GA_expr n) (f : Corner n -> bool),
    computes sq e f ->
    boolish_le (eval_expr sq e) 0.
Proof.
  intros n sq e f Hcomp.
  unfold boolish_le.
  exists f.
  (* dist_to is definitional equal to bool_dist_wrt, so the lemma gives equality *)
  assert (Hz : bool_dist_wrt (eval_expr sq e) f == 0).
  { exact (computes_implies_final_booldist_zero (n:=n) (sq:=sq) (e:=e) (f:=f) Hcomp). }
  (* equality implies <= *)
  apply (Qle_trans _ 0).
  - apply Qle_of_Qeq. exact Hz.
  - apply Qle_refl.
Qed.

Record EasyCompiler := {
  R : nat -> Type;

  target : forall {n : nat}, R n -> Corner n -> bool;
  compile : forall {n : nat}, R n -> GA_expr n;

  B : nat -> ExcNum;
  d0 : Q;

  compile_correct :
    forall {n : nat} (sq : Vector.t Q n) (r : R n),
      computes sq (compile (n:=n) r) (target (n:=n) r);

  compile_exc_bound :
    forall {n : nat} (sq : Vector.t Q n) (r : R n),
      exc_pre (exc_of sq (compile (n:=n) r)) (B (expr_size (compile (n:=n) r)));

  compile_boolish_bound :
    forall {n : nat} (sq : Vector.t Q n) (r : R n),
      trace_boolish_le sq (compile (n:=n) r) d0
}.

Definition compile' (C : EasyCompiler) {n} (r : R C n) : GA_expr n :=
  compile C (n:=n) r.

Definition target' (C : EasyCompiler) {n} (r : R C n) : Corner n -> bool :=
  target C (n:=n) r.

(*
    those compile' and target' definitions allow
      computes sq (compile' C r) (target' C r)
*)


(* Definition easy (C : EasyCompiler) {n} (f : Corner n -> bool) : Prop :=
  exists r : R C n, forall x, target' C r x = f x. *)
Definition easy (C : EasyCompiler) {n} (f : Corner n -> bool) : Prop :=
  exists r : R C n, forall m : Mask n, embed (target' C r) m == embed f m.

Lemma easy_implies_easy_under :
  forall (C : EasyCompiler) n (sq : Vector.t Q n) (f : Corner n -> bool),
    easy C (n:=n) f ->
    easy_under (B C) (d0 C) sq f.
Proof.
  intros C n sq f [r Htgt_embed].
  exists (compile' C r).
  split.
  - (* computes *)
    intros m.
    specialize (compile_correct C (n:=n) sq r) as Hcomp.
    specialize (Hcomp m).
    (* Hcomp: eval_expr sq (compile' C r) m == embed (target' C r) m *)
    (* rewrite RHS using Htgt_embed *)
    eapply Qeq_trans; [exact Hcomp |].
    exact (Htgt_embed m).
  - split.
    + apply compile_exc_bound.
    + apply compile_boolish_bound.
Qed.

Definition poly_B (k : nat) : nat -> ExcNum :=
  fun s => {| exc_grade := s^k; exc_l1 := inject_Z (Z.of_nat (s^k)) |}.

Fixpoint lincomb_embed {n}
  (cs : list Q) (gs : list (Corner n -> bool)) : MV n :=
  match cs, gs with
  | [], [] => mv_zero
  | c :: cs', g :: gs' =>
      mv_add (mv_scale c (embed g)) (lincomb_embed cs' gs')
  | _, _ => mv_zero (* or any default; enforce equal lengths in proofs *)
  end.

Definition wf_lincomb {n} (cs : list Q) (gs : list (Corner n -> bool)) : Prop :=
  length cs = length gs.

Definition boolish_k_le {n} (F : MV n) (k : nat) (d : Q) : Prop :=
  exists (cs : list Q) (gs : list (Corner n -> bool)),
    wf_lincomb cs gs /\
    (length gs <= k)%nat /\
    l1_norm (mv_sub F (lincomb_embed cs gs)) <= d.

Theorem all_easy_are_easy_under :
  forall (C : EasyCompiler) n (sq : Vector.t Q n) (f : Corner n -> bool),
    easy C (n:=n) f ->
    easy_under (B C) (d0 C) sq f.
Proof.
  intros C n sq f Heasy.
  exact (@easy_implies_easy_under C n sq f Heasy).
Qed.

Lemma mv_add_apply :
  forall n (F G : MV n) (m : Mask n),
    (F ⊕ G) m == F m + G m.
Proof. reflexivity. Qed.

Lemma lincomb_embed_app :
  forall (n : nat)
         (cs1 cs2 : list Q)
         (gs1 gs2 : list (Corner n -> bool)),
    wf_lincomb cs1 gs1 ->
    wf_lincomb cs2 gs2 ->
    forall m : Mask n,
      lincomb_embed (cs1 ++ cs2) (gs1 ++ gs2) m
      == mv_add (lincomb_embed cs1 gs1) (lincomb_embed cs2 gs2) m.
Proof.
  intros n cs1 cs2 gs1 gs2 Hwf1 Hwf2 m.
  revert gs1 Hwf1 m.
  induction cs1 as [|c cs1 IH]; intros gs1 Hwf1 m; simpl in *.
  -
    destruct gs1 as [|g gs1]; simpl in *.
    + unfold mv_add, mv_zero; ring.
    + discriminate.
  - destruct gs1 as [|g gs1]; simpl in *.
    + discriminate.
    +
      specialize (IH gs1).
      assert (Hwf1' : wf_lincomb cs1 gs1).
      { unfold wf_lincomb in *; simpl in *; lia. }
      specialize (IH Hwf1' m).
      unfold mv_add, mv_scale; simpl.
      rewrite IH.
      rewrite mv_add_apply.
      rewrite Qplus_assoc.
      reflexivity.
Qed.

Lemma wf_lincomb_app :
  forall n (cs1 cs2 : list Q) (gs1 gs2 : list (Corner n -> bool)),
    wf_lincomb cs1 gs1 ->
    wf_lincomb cs2 gs2 ->
    wf_lincomb (cs1 ++ cs2) (gs1 ++ gs2).
Proof.
  intros n cs1 cs2 gs1 gs2 H1 H2.
  unfold wf_lincomb in *.
  now rewrite length_app, length_app, H1, H2.
Qed.

Lemma mv_sub_add_split_general :
  forall n (F1 F2 G1 G2 : MV n) (m : Mask n),
    mv_sub (F1 ⊕ F2) (G1 ⊕ G2) m == (mv_sub F1 G1 ⊕ mv_sub F2 G2) m.
Proof.
  intros n F1 F2 G1 G2 m.
  unfold mv_sub.
  (* expand ⊕ at m *)
  rewrite (@mv_add_apply n F1 F2 m).
  rewrite (@mv_add_apply n G1 G2 m).
  rewrite (@mv_add_apply n (mv_sub F1 G1) (mv_sub F2 G2) m).
  unfold mv_sub.
  ring.
Qed.

Lemma boolish_k_le_add :
  forall n (F G : MV n) k1 k2 d1 d2,
    boolish_k_le F k1 d1 ->
    boolish_k_le G k2 d2 ->
    boolish_k_le (F ⊕ G) (k1 + k2) (d1 + d2).
Proof.
  intros n F G k1 k2 d1 d2 HF HG.
  destruct HF as [csF [gsF [HwfF [HlenF HdistF]]]].
  destruct HG as [csG [gsG [HwfG [HlenG HdistG]]]].

  unfold boolish_k_le.
  exists (csF ++ csG), (gsF ++ gsG).
  repeat split.
  - (* wf *)
    apply (wf_lincomb_app (n:=n)); assumption.
  - (* length bound *)
    rewrite length_app; lia.
  - (* distance bound *)
    set (AF := lincomb_embed csF gsF).
    set (AG := lincomb_embed csG gsG).
    assert (Hlin_app :
      forall m : Mask n,
        lincomb_embed (csF ++ csG) (gsF ++ gsG) m == (AF ⊕ AG) m).
    {
      intro m. unfold AF, AG.
      apply lincomb_embed_app; assumption.
    }
    assert (Hnorm_rewrite :
      l1_norm (mv_sub (F ⊕ G) (lincomb_embed (csF ++ csG) (gsF ++ gsG)))
      == l1_norm (mv_sub (F ⊕ G) (AF ⊕ AG))).
    {
      apply l1_norm_ext.
      intro m.
      unfold mv_sub.
      rewrite (Hlin_app m).
      reflexivity.
    }
    eapply Qle_trans.
    + apply Qle_of_Qeq. exact Hnorm_rewrite.
    + eapply Qle_trans.
      * 
        eapply Qle_trans.
        -- (* rewrite inside l1_norm *)
           apply (Qle_of_Qeq).
           apply l1_norm_ext.
           intro m.
           apply mv_sub_add_split_general.
        -- (* triangle inequality *)
           apply l1_add_bound.
      * 
        unfold AF, AG in *.
        apply Qplus_le_compat; assumption.
Qed.

Definition Qpow2 (k : nat) : Q := inject_Z (Z.pow 2 (Z.of_nat k)).

Fixpoint trace_boolish_k_le {n}
  (sq : Vector.t Q n) (e : GA_expr n) (k : nat) (d : Q) : Prop :=
  match e with
  | Basis _ | Scalar _ => boolish_k_le (eval_expr sq e) k d
  | Cln_Grade.Add e1 e2 =>
      trace_boolish_k_le sq e1 k d /\
      trace_boolish_k_le sq e2 k d /\
      boolish_k_le (eval_expr sq (Cln_Grade.Add e1 e2)) k d
  | Mul e1 e2 =>
      trace_boolish_k_le sq e1 k d /\
      trace_boolish_k_le sq e2 k d /\
      boolish_k_le (mv_gp sq (eval_expr sq e1) (eval_expr sq e2)) k d
  | Conv e1 e2 =>
      trace_boolish_k_le sq e1 k d /\
      trace_boolish_k_le sq e2 k d /\
      boolish_k_le (mv_conv (eval_expr sq e1) (eval_expr sq e2)) k d
  end.

(* subexpression lemmas: Add *)
Lemma trace_boolish_k_le_sub_left_Add :
  forall n (sq : Vector.t Q n) (k : nat) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_k_le sq (Cln_Grade.Add e1 e2) k d ->
    trace_boolish_k_le sq e1 k d.
Proof. intros; simpl in *; tauto. Qed.

Lemma trace_boolish_k_le_sub_right_Add :
  forall n (sq : Vector.t Q n) (k : nat) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_k_le sq (Cln_Grade.Add e1 e2) k d ->
    trace_boolish_k_le sq e2 k d.
Proof. intros; simpl in *; tauto. Qed.

Lemma trace_boolish_k_le_node_Add :
  forall n (sq : Vector.t Q n) (k : nat) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_k_le sq (Cln_Grade.Add e1 e2) k d ->
    boolish_k_le (eval_expr sq (Cln_Grade.Add e1 e2)) k d.
Proof. intros; simpl in *; tauto. Qed.

(* subexpression lemmas: Mul *)
Lemma trace_boolish_k_le_sub_left_Mul :
  forall n (sq : Vector.t Q n) (k : nat) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_k_le sq (Mul e1 e2) k d ->
    trace_boolish_k_le sq e1 k d.
Proof. intros; simpl in *; tauto. Qed.

Lemma trace_boolish_k_le_sub_right_Mul :
  forall n (sq : Vector.t Q n) (k : nat) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_k_le sq (Mul e1 e2) k d ->
    trace_boolish_k_le sq e2 k d.
Proof. intros; simpl in *; tauto. Qed.

Lemma trace_boolish_k_le_node_Mul :
  forall n (sq : Vector.t Q n) (k : nat) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_k_le sq (Mul e1 e2) k d ->
    boolish_k_le (mv_gp sq (eval_expr sq e1) (eval_expr sq e2)) k d.
Proof. intros; simpl in *; tauto. Qed.

(* subexpression lemmas: Conv *)
Lemma trace_boolish_k_le_sub_left_Conv :
  forall n (sq : Vector.t Q n) (k : nat) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_k_le sq (Conv e1 e2) k d ->
    trace_boolish_k_le sq e1 k d.
Proof. intros; simpl in *; tauto. Qed.

Lemma trace_boolish_k_le_sub_right_Conv :
  forall n (sq : Vector.t Q n) (k : nat) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_k_le sq (Conv e1 e2) k d ->
    trace_boolish_k_le sq e2 k d.
Proof. intros; simpl in *; tauto. Qed.

Lemma trace_boolish_k_le_node_Conv :
  forall n (sq : Vector.t Q n) (k : nat) (d : Q) (e1 e2 : GA_expr n),
    trace_boolish_k_le sq (Conv e1 e2) k d ->
    boolish_k_le (mv_conv (eval_expr sq e1) (eval_expr sq e2)) k d.
Proof. intros; simpl in *; tauto. Qed.

Lemma mv_add_sub_cancel_r :
  forall n (G eG : MV n) (m : Mask n),
    (eG ⊕ mv_sub G eG) m == G m.
Proof.
  intros n G eG m.
  unfold mv_add, mv_sub.
  ring.
Qed.

Lemma mv_conv_add_l :
  forall n (F1 F2 G : MV n) (U : Mask n),
    mv_conv (F1 ⊕ F2) G U == (mv_conv F1 G ⊕ mv_conv F2 G) U.
Proof.
  intros n F1 F2 G U.
  unfold mv_conv.
  rewrite (@mv_add_apply n (mv_conv F1 G) (mv_conv F2 G) U).
  set (MS := all_masks n).

  eapply Qeq_trans.
  - (* pointwise: push mv_add inside the kernel *)
    apply (sumQ_map_ext (A := Mask n)
        (fun A =>
           sumQ (map (fun B =>
             if mask_eq_dec (mask_xor A B) U
             then (F1 ⊕ F2) A * G B else 0) MS))
        (fun A =>
           sumQ (map (fun B =>
             (if mask_eq_dec (mask_xor A B) U then F1 A * G B else 0) +
             (if mask_eq_dec (mask_xor A B) U then F2 A * G B else 0)) MS))
        MS).
    intros A HA.

  apply (sumQ_map_ext (A := Mask n)
          (fun B =>
             if mask_eq_dec (mask_xor A B) U
             then (F1 A + F2 A) * G B else 0)
          (fun B =>
             (if mask_eq_dec (mask_xor A B) U then F1 A * G B else 0) +
             (if mask_eq_dec (mask_xor A B) U then F2 A * G B else 0))
          MS).
  intros B HB.
  destruct (mask_eq_dec (mask_xor A B) U); ring.

  - (* now split sums: inner first (pointwise), then outer *)
    set (inner1 := fun A : Mask n =>
      sumQ (map (fun B0 : Mask n =>
        if mask_eq_dec (mask_xor A B0) U then F1 A * G B0 else 0) MS)).
    set (inner2 := fun A : Mask n =>
      sumQ (map (fun B0 : Mask n =>
        if mask_eq_dec (mask_xor A B0) U then F2 A * G B0 else 0) MS)).

    eapply Qeq_trans.
    + (* rewrite each A-summand using inner sumQ_map_add *)
      apply (sumQ_map_ext (A := Mask n)
              (fun A =>
                 sumQ (map (fun B0 =>
                   (if mask_eq_dec (mask_xor A B0) U then F1 A * G B0 else 0) +
                   (if mask_eq_dec (mask_xor A B0) U then F2 A * G B0 else 0)) MS))
              (fun A => (inner1 A + inner2 A)%Q)
              MS).
      intros A HA.
      unfold inner1, inner2.
      (* THIS is where sumQ_map_add applies: inside the B0-sum *)
      apply (sumQ_map_add (A := Mask n)
        (fun B0 =>
           if mask_eq_dec (mask_xor A B0) U then F1 A * G B0 else 0)
        (fun B0 =>
           if mask_eq_dec (mask_xor A B0) U then F2 A * G B0 else 0)
        MS).

    + (* now split the outer sum over A *)
      unfold inner1, inner2.
      apply (sumQ_map_add (A := Mask n)
              (fun A =>
                 sumQ (map (fun B0 =>
                   if mask_eq_dec (mask_xor A B0) U then F1 A * G B0 else 0) MS))
              (fun A =>
                 sumQ (map (fun B0 =>
                   if mask_eq_dec (mask_xor A B0) U then F2 A * G B0 else 0) MS))
              MS).
Qed.

Lemma mv_conv_add_r :
  forall n (F G1 G2 : MV n) (U : Mask n),
    mv_conv F (G1 ⊕ G2) U == (mv_conv F G1 ⊕ mv_conv F G2) U.
Proof.
  intros n F G1 G2 U.
  unfold mv_conv.
  rewrite (@mv_add_apply n (mv_conv F G1) (mv_conv F G2) U).
  set (MS := all_masks n).

  (* Expand the RHS mv_add at U, then split the double sum on the LHS *)
  eapply Qeq_trans.
  - (* pointwise: expand (G1 ⊕ G2) B0 inside the kernel *)
    apply (sumQ_map_ext (A := Mask n)
            (fun A : Mask n =>
               sumQ (map (fun B0 : Mask n =>
                 if mask_eq_dec (mask_xor A B0) U
                 then F A * (G1 ⊕ G2) B0
                 else 0) MS))
            (fun A : Mask n =>
               sumQ (map (fun B0 : Mask n =>
                 (if mask_eq_dec (mask_xor A B0) U then F A * G1 B0 else 0) +
                 (if mask_eq_dec (mask_xor A B0) U then F A * G2 B0 else 0)) MS))
            MS).
    intros A HA.
    apply (sumQ_map_ext (A := Mask n)
            (fun B0 : Mask n =>
               if mask_eq_dec (mask_xor A B0) U
               then F A * (G1 ⊕ G2) B0
               else 0)
            (fun B0 : Mask n =>
               (if mask_eq_dec (mask_xor A B0) U then F A * G1 B0 else 0) +
               (if mask_eq_dec (mask_xor A B0) U then F A * G2 B0 else 0))
            MS).
    intros B0 HB0.
    unfold mv_add. (* so (G1 ⊕ G2) B0 becomes G1 B0 + G2 B0 *)
    destruct (mask_eq_dec (mask_xor A B0) U); ring.

  - (* now split sums: inner (over B0) first, then outer (over A) *)
    set (inner1 := fun A : Mask n =>
      sumQ (map (fun B0 : Mask n =>
        if mask_eq_dec (mask_xor A B0) U then F A * G1 B0 else 0) MS)).
    set (inner2 := fun A : Mask n =>
      sumQ (map (fun B0 : Mask n =>
        if mask_eq_dec (mask_xor A B0) U then F A * G2 B0 else 0) MS)).

    eapply Qeq_trans.
    + (* rewrite each A-summand using inner sumQ_map_add *)
      apply (sumQ_map_ext (A := Mask n)
              (fun A : Mask n =>
                 sumQ (map (fun B0 : Mask n =>
                   (if mask_eq_dec (mask_xor A B0) U then F A * G1 B0 else 0) +
                   (if mask_eq_dec (mask_xor A B0) U then F A * G2 B0 else 0)) MS))
              (fun A : Mask n => (inner1 A + inner2 A)%Q)
              MS).
      intros A HA.
      unfold inner1, inner2.
      apply (sumQ_map_add
              (fun B0 : Mask n =>
                 if mask_eq_dec (mask_xor A B0) U then F A * G1 B0 else 0)
              (fun B0 : Mask n =>
                 if mask_eq_dec (mask_xor A B0) U then F A * G2 B0 else 0)
              MS).
    + (* now split the outer sum over A *)
      unfold inner1, inner2.
      apply (sumQ_map_add
              (fun A : Mask n =>
                 sumQ (map (fun B0 : Mask n =>
                   if mask_eq_dec (mask_xor A B0) U then F A * G1 B0 else 0) MS))
              (fun A : Mask n =>
                 sumQ (map (fun B0 : Mask n =>
                   if mask_eq_dec (mask_xor A B0) U then F A * G2 B0 else 0) MS))
              MS).
Qed.

Lemma conv_error_split :
  forall (n : nat) (F G eF eG : MV n) (U : Mask n),
    mv_sub (mv_conv F G) (mv_conv eF eG) U
    ==
    (mv_conv F (mv_sub G eG) ⊕ mv_conv (mv_sub F eF) eG) U.
Proof.
  intros n F G eF eG U.
  unfold mv_sub.
  assert (HGdecomp :
    mv_conv F G U == mv_conv F (eG ⊕ mv_sub G eG) U).
  {
    unfold mv_conv.
    apply (sumQ_map_ext (A := Mask n)
            (fun A =>
               sumQ (map (fun B =>
                 if mask_eq_dec (mask_xor A B) U then F A * G B else 0) (all_masks n)))
            (fun A =>
               sumQ (map (fun B =>
                 if mask_eq_dec (mask_xor A B) U then F A * (eG ⊕ mv_sub G eG) B else 0) (all_masks n)))
            (all_masks n)).
    intros A HA.
    apply (sumQ_map_ext (A := Mask n)
            (fun B =>
               if mask_eq_dec (mask_xor A B) U then F A * G B else 0)
            (fun B =>
               if mask_eq_dec (mask_xor A B) U then F A * (eG ⊕ mv_sub G eG) B else 0)
            (all_masks n)).
    intros B HB.
    destruct (mask_eq_dec (mask_xor A B) U) as [HAB|HAB].
    - unfold mv_add, mv_sub. ring.
    - reflexivity.
  }

  rewrite HGdecomp.
  rewrite (@mv_conv_add_r n F eG (mv_sub G eG) U).
  assert (HFdecomp :
    mv_conv F eG U == mv_conv (eF ⊕ mv_sub F eF) eG U).
  {
    unfold mv_conv.
    apply (sumQ_map_ext (A := Mask n)); intros A HA.
    apply (sumQ_map_ext (A := Mask n)); intros B HB.
    destruct (mask_eq_dec (mask_xor A B) U) as [HAB|HAB].
    - unfold mv_add, mv_sub. ring.
    - reflexivity.
  }

  set (dF := mv_sub F eF).
  set (dG := mv_sub G eG).

  rewrite (@mv_add_apply n (mv_conv F eG) (mv_conv F dG) U).
  change (mv_conv F (fun m : Mask n => G m - eG m) ⊕ mv_conv (fun m : Mask n => F m - eF m) eG) with
       (mv_conv F dG ⊕ mv_conv dF eG).

  rewrite (@mv_add_apply n (mv_conv F dG) (mv_conv dF eG) U).
  rewrite HFdecomp.

  set (T := mv_conv (eF ⊕ mv_sub F eF) eG U).

  assert (HT : T == (mv_conv eF eG ⊕ mv_conv dF eG) U).
  {
    unfold T.
    fold dF.
    exact (@mv_conv_add_l n eF dF eG U).
  }

  set (X := mv_conv F dG U).
  set (Y := mv_conv eF eG U).
  set (Z := mv_conv dF eG U).

  assert (HT' : T == mv_conv eF eG U + mv_conv dF eG U).
  { eapply Qeq_trans; [ exact HT | exact (@mv_add_apply n (mv_conv eF eG) (mv_conv dF eG) U) ]. }

  rewrite HT'.
  change (mv_conv eF eG U + mv_conv dF eG U + mv_conv F dG U - mv_conv eF eG U
          == mv_conv F dG U + mv_conv dF eG U).
  ring.
Qed.

Lemma gp_error_bound_l1 :
  forall (n : nat) (sq : Vector.t Q n) (F G eF eG : MV n),
    (forall i : Fin.t n, Qabs (Vector.nth sq i) == 1) ->
    l1_norm (mv_sub (mv_gp sq F G) (mv_gp sq eF eG))
    <= l1_norm F * l1_norm (mv_sub G eG)
     + l1_norm (mv_sub F eF) * l1_norm eG.
Proof.
  intros n sq F G eF eG Hsq.
  eapply Qle_trans.
  - apply Qle_of_Qeq.
    apply l1_norm_ext; intro m.
    apply (@gp_error_split n sq F G eF eG m).
  - eapply Qle_trans.
    + apply l1_add_bound.
    + apply Qplus_le_compat.
      * eapply l1_gp_submultiplicative.
        exact Hsq.
      * eapply l1_gp_submultiplicative.
        exact Hsq.
Qed.

Lemma boolish_k_le_mono :
  forall n (F : MV n) k1 k2 d,
    (k1 <= k2)%nat ->
    boolish_k_le F k1 d ->
    boolish_k_le F k2 d.
Proof.
  intros n F k1 k2 d Hle [cs [gs [Hwf [Hlen Hd]]]].
  exists cs, gs; repeat split; try assumption.
  - lia.
Qed.

Definition poly1 (n : nat) (a c : nat) : nat :=
  c * Nat.pow (n + 1) a.

Definition poly_bound1 (f : nat -> nat) : Prop :=
  exists a c,
    forall n, (f n <= poly1 n a c)%nat.

Definition poly_in_nat (s x : nat) (a b c : nat) : nat :=
  c * (Nat.pow (s + 1) a) * (Nat.pow (x + 1) b).

Definition poly_bound2 (P : nat -> nat -> nat) : Prop :=
  exists (a b c : nat),
    forall (s x : nat),
      (P s x <= poly_in_nat s x a b c)%nat.

Definition dominates_poly (f : nat -> nat) : Prop :=
  forall a c,
    exists N, forall n, (n >= N)%nat ->
      (poly1 n a c < f n)%nat.

Definition dominates_poly2 (F : nat -> nat -> nat) : Prop :=
  forall a b c,
    exists N, forall s x,
      (s >= N)%nat ->
      (poly_in_nat s x a b c < F s x)%nat.

Definition superpoly (f : nat -> nat) : Prop := dominates_poly f.

Definition exp_lb (f : nat -> nat) : Prop :=
  exists c, forall n, (Nat.pow 2 (c*n) <= f n)%nat.


(* ------------------------------------------------------------ *)
(* Basic facts about pow / monotonicity                          *)
(* ------------------------------------------------------------ *)
Local Open Scope nat_scope.
Local Close Scope Q_scope.

Lemma pow_le_succ :
  forall base e,
    (1 <= base)%nat ->
    (base ^ e <= base ^ (S e))%nat.
Proof.
  intros base e Hb.
  (* exponent monotonicity *)
  apply (Nat.pow_le_mono_r base e (S e)).
  - lia.
  - lia.
Qed.

Lemma pow_mono_exp :
  forall base e1 e2,
    (1 <= base)%nat ->
    (e1 <= e2)%nat ->
    (base ^ e1 <= base ^ e2)%nat.
Proof.
  intros base e1 e2 Hb He.
  apply (Nat.pow_le_mono_r base e1 e2).
  - (* base <> 0 *)
    lia.
  - exact He.
Qed.

Lemma pow_mono_base :
  forall base1 base2 e,
    (base1 <= base2)%nat ->
    (Nat.pow base1 e <= Nat.pow base2 e)%nat.
Proof.
  intros base1 base2 e Hle.
  (* In this Coq version: pow_le_mono_l is monotone in the base *)
  apply (Nat.pow_le_mono_l base1 base2 e).
  exact Hle.
Qed.

(* Useful corollaries for (s+1) and (x+1) bases *)
Lemma pow_s_mono :
  forall s1 s2 a,
    (s1 <= s2)%nat ->
    (Nat.pow (s1 + 1) a <= Nat.pow (s2 + 1) a)%nat.
Proof.
  intros s1 s2 a H.
  apply pow_mono_base. lia.
Qed.

Lemma pow_x_mono :
  forall x1 x2 b,
    (x1 <= x2)%nat ->
    (Nat.pow (x1 + 1) b <= Nat.pow (x2 + 1) b)%nat.
Proof.
  intros x1 x2 b H.
  apply pow_mono_base. lia.
Qed.

(* ------------------------------------------------------------ *)
(* Monotonicity of poly_in_nat                                   *)
(* ------------------------------------------------------------ *)
Lemma poly_in_nat_mono_a :
  forall s x a1 a2 b c,
    (a1 <= a2)%nat ->
    poly_in_nat s x a1 b c <= poly_in_nat s x a2 b c.
Proof.
  intros s x a1 a2 b c Ha.
  unfold poly_in_nat.

  (* Put both sides into the form (c * (s+1)^a) * (x+1)^b *)
  repeat rewrite Nat.mul_assoc.

  (* multiply both sides on the right by the same factor preserves <= *)
  apply Nat.mul_le_mono_r.

  (* now goal: c * (s+1)^a1 <= c * (s+1)^a2 *)
  apply Nat.mul_le_mono_l.

  (* now goal: (s+1)^a1 <= (s+1)^a2 *)
  apply pow_mono_exp; lia.
Qed.

Lemma poly_in_nat_mono_b :
  forall s x a b1 b2 c,
    (b1 <= b2)%nat ->
    poly_in_nat s x a b1 c <= poly_in_nat s x a b2 c.
Proof.
  intros s x a b1 b2 c Hb.
  unfold poly_in_nat.
  (* force the shape (c*(s+1)^a) * (x+1)^b *)
  repeat rewrite Nat.mul_assoc.
  apply Nat.mul_le_mono_l.
  apply pow_mono_exp; lia.
Qed.

Lemma poly_in_nat_mono_s :
  forall s1 s2 x a b c,
    (s1 <= s2)%nat ->
    poly_in_nat s1 x a b c <= poly_in_nat s2 x a b c.
Proof.
  intros s1 s2 x a b c Hs.
  unfold poly_in_nat.
  (* regroup as (c * (s+1)^a) * (x+1)^b *)
  repeat rewrite Nat.mul_assoc.
  apply Nat.mul_le_mono_r.
  (* now prove: c * (s1+1)^a <= c * (s2+1)^a *)
  apply Nat.mul_le_mono_l.
  apply pow_mono_base.
  lia.
Qed.

Lemma poly_in_nat_mono_x :
  forall s x1 x2 a b c,
    (x1 <= x2)%nat ->
    poly_in_nat s x1 a b c <= poly_in_nat s x2 a b c.
Proof.
  intros s x1 x2 a b c Hx.
  unfold poly_in_nat.
  (* c * (s+1)^a is a common left factor *)
  apply Nat.mul_le_mono_l.
  (* now prove: (x1+1)^b <= (x2+1)^b *)
  apply pow_mono_base.
  lia.
Qed.

(* A handy “upgrade exponents to max” lemma *)
Lemma poly_in_nat_le_with_max :
  forall s x a a' b b' c,
    poly_in_nat s x a b c
    <= poly_in_nat s x (Nat.max a a') (Nat.max b b') c.
Proof.
  intros s x a a' b b' c.
  eapply Nat.le_trans.
  - (* raise a to max a a' *)
    eapply poly_in_nat_mono_a.
    apply Nat.le_max_l.
  - (* raise b to max b b' *)
    eapply poly_in_nat_mono_b.
    apply Nat.le_max_l.
Qed.

(* ------------------------------------------------------------ *)
(* Algebraic closure at the envelope level                       *)
(* ------------------------------------------------------------ *)

(* Exact multiplicativity: envelope * envelope = envelope with added exponents *)
Lemma poly_in_nat_mul_exact :
  forall s x a1 b1 c1 a2 b2 c2,
    poly_in_nat s x a1 b1 c1 * poly_in_nat s x a2 b2 c2
    =
    poly_in_nat s x (a1 + a2) (b1 + b2) (c1 * c2).
Proof.
  intros s x a1 b1 c1 a2 b2 c2.
  unfold poly_in_nat.
  (* pow_add_r: base^(m+n) = base^m * base^n *)
  rewrite Nat.pow_add_r.
  rewrite Nat.pow_add_r.
  nia.
Qed.

(* Additive closure (loose but clean): use max exponents and add constants *)
Lemma poly_in_nat_add_const_le :
  forall s x a b c1 c2,
    poly_in_nat s x a b c1 + poly_in_nat s x a b c2
    <= poly_in_nat s x a b (c1 + c2).
Proof.
  intros s x a b c1 c2.
  unfold poly_in_nat.
  nia.
Qed.

Lemma poly_in_nat_add_le :
  forall s x a1 b1 c1 a2 b2 c2,
    poly_in_nat s x a1 b1 c1 + poly_in_nat s x a2 b2 c2
    <= poly_in_nat s x (Nat.max a1 a2) (Nat.max b1 b2) (c1 + c2).
Proof.
  intros s x a1 b1 c1 a2 b2 c2.
  set (A := Nat.max a1 a2).
  set (B := Nat.max b1 b2).

  eapply Nat.le_trans
    with (m := poly_in_nat s x A B c1 + poly_in_nat s x A B c2).
  - (* bound each summand up to (A,B) *)
    apply Nat.add_le_mono.
    + eapply Nat.le_trans.
      * eapply poly_in_nat_mono_a. apply Nat.le_max_l.
      * eapply poly_in_nat_mono_b. apply Nat.le_max_l.
    + eapply Nat.le_trans.
      * eapply poly_in_nat_mono_a. apply Nat.le_max_r.
      * eapply poly_in_nat_mono_b. apply Nat.le_max_r.
  - (* combine constants *)
    apply poly_in_nat_add_const_le.
Qed.

(* Scaling constant closure: c0 * poly <= poly with constant multiplied *)
Lemma poly_in_nat_scale :
  forall s x a b c c0,
    c0 * poly_in_nat s x a b c
    = poly_in_nat s x a b (c0 * c).
Proof.
  intros s x a b c c0.
  unfold poly_in_nat.
  nia.
Qed.

(* ------------------------------------------------------------ *)
(* Closure lemmas for poly_bound2                                *)
(* ------------------------------------------------------------ *)

Definition P_add (P Q : nat -> nat -> nat) : nat -> nat -> nat :=
  fun s x => P s x + Q s x.

Definition P_mul (P Q : nat -> nat -> nat) : nat -> nat -> nat :=
  fun s x => P s x * Q s x.

Definition P_scale (c0 : nat) (P : nat -> nat -> nat) : nat -> nat -> nat :=
  fun s x => c0 * P s x.

Lemma poly_bound2_mono :
  forall P Q,
    (forall s x, (P s x <= Q s x)%nat) ->
    poly_bound2 Q ->
    poly_bound2 P.
Proof.
  intros P Q Hle [a [b [c HQ]]].
  exists a, b, c. intros s x.
  eapply Nat.le_trans; [apply Hle | apply HQ].
Qed.

Lemma poly_bound2_add :
  forall P Q,
    poly_bound2 P ->
    poly_bound2 Q ->
    poly_bound2 (P_add P Q).
Proof.
  intros P Q [a1 [b1 [c1 HP]]] [a2 [b2 [c2 HQ]]].
  exists (Nat.max a1 a2), (Nat.max b1 b2), (c1 + c2).
  intros s x.
  unfold P_add.
  eapply Nat.le_trans.
  - apply Nat.add_le_mono; [apply HP | apply HQ].
  - apply poly_in_nat_add_le.
Qed.

Lemma poly_bound2_mul :
  forall P Q,
    poly_bound2 P ->
    poly_bound2 Q ->
    poly_bound2 (P_mul P Q).
Proof.
  intros P Q [a1 [b1 [c1 HP]]] [a2 [b2 [c2 HQ]]].
  exists (a1 + a2), (b1 + b2), (c1 * c2).
  intros s x.
  unfold P_mul.
  eapply Nat.le_trans.
  - apply Nat.mul_le_mono; [apply HP | apply HQ].
  - rewrite <- poly_in_nat_mul_exact.
    apply Nat.le_refl.
Qed.

Lemma poly_bound2_scale :
  forall c0 P,
    poly_bound2 P ->
    poly_bound2 (P_scale c0 P).
Proof.
  intros c0 P [a [b [c HP]]].
  exists a, b, (c0 * c).
  intros s x.
  unfold P_scale.
  eapply Nat.le_trans.
  - apply Nat.mul_le_mono_l. apply HP.
  - rewrite <- poly_in_nat_scale.
    apply Nat.le_refl.
Qed.

Lemma l1_conv_submultiplicative :
  forall n (F G : MV n),
    (l1_norm (mv_conv F G) <= l1_norm F * l1_norm G)%Q.
Proof.
  intros n F G.
  exact (@l1_conv_bound n F G).
Qed.

Definition and_gen {n} (g1 g2 : Corner n -> bool) : Corner n -> bool :=
  fun x => andb (g1 x) (g2 x).

Definition and_gens {n}
  (gs1 gs2 : list (Corner n -> bool)) : list (Corner n -> bool) :=
  concat (map (fun g1 => map (and_gen g1) gs2) gs1).
  
Definition mul_coeffs (cs1 cs2 : list Q) : list Q :=
  concat (map (fun c1 => map (fun c2 => (c1 * c2)%Q) cs2) cs1).

(*
===============================================================================
*)

Definition trace_boolish_exists_k {n}
  (sq : Vector.t Q n) (e : GA_expr n) (d : Q) : Prop :=
  exists k : nat, trace_boolish_k_le sq e k d.

Lemma computes_l1_eq :
  forall n (sq : Vector.t Q n) (e : GA_expr n) (f : Corner n -> bool),
    computes sq e f ->
    l1_norm (eval_expr sq e) = l1_norm (embed f).
Proof.
  intros n sq e f Hcomp.
  apply l1_norm_ext. intro m.
  exact (Hcomp m).
Qed.

Theorem hard_family_separates :
  forall d : Q,
  exists f : forall n, Corner n -> bool,
  exists c : nat,
    forall n (sq : Vector.t Q n) (e : GA_expr n),
      computes sq e (f n) ->
      trace_boolish_exists_k sq e d ->
      (Qpow2 (c * n) <= exc_l1 (exc_of sq e))%Q.
Proof.
Admitted.

Lemma l1_norm_embed_IP_ge_pow2 :
  forall m,
    (m >= 2)%nat ->
    (Qpow2 (m - 2) <= l1_norm (embed (@IP_n_func (m+m))))%Q.
Proof.
(*
Then in hard_family_separates_div2, choose f n := IP_n_func n and c := 1, and use Nat.div2 (m+m) = m.
*)
Admitted.

Theorem hard_family_separates_div2 :
  forall d : Q,
  exists f : forall n, Corner n -> bool,
  exists c : nat,
    forall n (sq : Vector.t Q n) (e : GA_expr n),
      computes sq e (f n) ->
      trace_boolish_exists_k sq e d ->
      (Qpow2 (c * Nat.div2 n) <= exc_l1 (exc_of sq e))%Q.
Proof.
Admitted.

Definition pow2 (k : nat) : nat :=
  Nat.pow 2 k.
  
Definition poly_k (n s : nat) : nat :=
  pow2 s.
  
(* Definition trace_boolish_poly {n}
  (sq : Vector.t Q n) (e : GA_expr n) (d : Q) : Prop :=
  trace_boolish_k_le sq e (poly_k n (expr_size e)) d. *)

Definition trace_boolish_poly {n}
  (sq : Vector.t Q n) (e : GA_expr n) (d : Q) : Prop :=
  exists k : nat,
    (k <= poly_k n (expr_size e))%nat /\
    trace_boolish_k_le sq e k d.

Lemma trace_boolish_k_le_mono :
  forall n (sq : Vector.t Q n) (e : GA_expr n) k1 k2 d,
    (k1 <= k2)%nat ->
    trace_boolish_k_le sq e k1 d ->
    trace_boolish_k_le sq e k2 d.
Proof.
  intros n sq e.
  induction e; intros k1 k2 d Hle Htr; simpl in *.
  - (* Basis *) eapply boolish_k_le_mono; eauto.
  - (* Scalar *) eapply boolish_k_le_mono; eauto.
  - (* Add *)
    destruct Htr as [H1 [H2 H3]].
    repeat split.
    + eapply IHe1; eauto.
    + eapply IHe2; eauto.
    + eapply boolish_k_le_mono; eauto.
  - (* Mul *)
    destruct Htr as [H1 [H2 H3]].
    repeat split.
    + eapply IHe1; eauto.
    + eapply IHe2; eauto.
    + eapply boolish_k_le_mono; eauto.
  - (* Conv *)
    destruct Htr as [H1 [H2 H3]].
    repeat split.
    + eapply IHe1; eauto.
    + eapply IHe2; eauto.
    + eapply boolish_k_le_mono; eauto.
Qed.

Lemma trace_boolish_k_le_to_poly_k :
  forall n (sq : Vector.t Q n) (e : GA_expr n) k d,
    trace_boolish_k_le sq e k d ->
    (k <= poly_k n (expr_size e))%nat ->
    trace_boolish_k_le sq e (poly_k n (expr_size e)) d.
Proof.
  intros n sq e k d Htr Hk.
  eapply trace_boolish_k_le_mono; eauto.
Qed.

Lemma trace_boolish_poly_to_canonical :
  forall n (sq : Vector.t Q n) (e : GA_expr n) d,
    trace_boolish_poly sq e d ->
    trace_boolish_k_le sq e (poly_k n (expr_size e)) d.
Proof.
  intros n sq e d [k [Hk Htr]].
  eapply trace_boolish_k_le_to_poly_k; eauto.
Qed.

Lemma trace_boolish_canonical_to_poly :
  forall n (sq : Vector.t Q n) (e : GA_expr n) d,
    trace_boolish_k_le sq e (poly_k n (expr_size e)) d ->
    trace_boolish_poly sq e d.
Proof.
  intros n sq e d H.
  exists (poly_k n (expr_size e)).
  now split.
Qed.

Lemma trace_boolish_poly_elim :
  forall n (sq : Vector.t Q n) (e : GA_expr n) d,
    trace_boolish_poly sq e d ->
    trace_boolish_k_le sq e (poly_k n (expr_size e)) d.
Proof.
  intros n sq e d Hpoly.
  apply trace_boolish_poly_to_canonical; exact Hpoly.
Qed.

Lemma mul_coeffs_cons :
  forall (c : Q) cs1 cs2,
    mul_coeffs (c :: cs1) cs2 =
    (map (fun c2 => (c * c2)%Q) cs2) ++ mul_coeffs cs1 cs2.
Proof.
  intros c cs1 cs2.
  unfold mul_coeffs. simpl. reflexivity.
Qed.

Lemma and_gens_cons {n} :
  forall (g : Corner n -> bool) gs1 gs2,
    and_gens (g :: gs1) gs2 =
    (map (and_gen g) gs2) ++ and_gens gs1 gs2.
Proof.
  intros g gs1 gs2.
  unfold and_gens. simpl. reflexivity.
Qed.

Lemma length_and_gens {n} :
  forall (gs1 gs2 : list (Corner n -> bool)),
    length (and_gens gs1 gs2) = (length gs1 * length gs2)%nat.
Proof.
  intros gs1 gs2.
  induction gs1 as [|g gs1 IH]; simpl.
  - reflexivity.
  - rewrite and_gens_cons.
    rewrite length_app, length_map.
    rewrite IH.
    lia.
Qed.

Lemma length_mul_coeffs :
  forall (cs1 cs2 : list Q),
    length (mul_coeffs cs1 cs2) = (length cs1 * length cs2)%nat.
Proof.
  intros cs1 cs2.
  induction cs1 as [|c cs1 IH]; simpl.
  - reflexivity.
  - rewrite mul_coeffs_cons.
    rewrite length_app, length_map.
    rewrite IH.
    lia.
Qed.

Lemma wf_lincomb_mul_and :
  forall (n : nat)
         (csF csG : list Q)
         (gsF gsG : list (Corner n -> bool)),
    wf_lincomb csF gsF ->
    wf_lincomb csG gsG ->
    wf_lincomb (mul_coeffs csF csG) (and_gens gsF gsG).
Proof.
  intros n csF csG gsF gsG HwfF HwfG.
  unfold wf_lincomb in *.
  rewrite length_mul_coeffs, length_and_gens.
  now rewrite HwfF, HwfG.
Qed.

Lemma l1_sub_bound :
  forall n (F G : MV n),
    (l1_norm (mv_sub F G) <= l1_norm F + l1_norm G)%Q.
Proof.
  intros n F G.
  unfold l1_norm, mv_sub.

  eapply Qle_trans.
  - (* lift pointwise inequality through sum *)
    apply (@sumQ_map_le (Mask n)
             (fun U => Qabs (F U + (- G U))%Q)
             (fun U => (Qabs (F U) + Qabs (G U))%Q)
             (all_masks n)).
    intros U HU.
    (* |x + (-y)| <= |x| + |y| *)
    eapply Qle_trans.
    + apply Qabs_triangle.
    + rewrite Qabs_opp. exact (Qle_refl _).
  - (* sum of (a+b) = sum a + sum b *)
    rewrite <- sumQ_map_add.
    apply Qle_refl.
Qed.

Lemma mv_sub_cancel_qeq :
  forall n (G G0 : MV n) m,
    Qeq (mv_sub G (mv_sub G G0) m) (G0 m).
Proof.
  intros n G G0 m.
  unfold mv_sub, Qminus.
  ring.
Qed.

Lemma conv_error_bound_l1 :
  forall (n : nat) (F G eF eG : MV n),
    ( l1_norm (mv_sub (mv_conv F G) (mv_conv eF eG))
      <= l1_norm F * l1_norm (mv_sub G eG)
       + l1_norm (mv_sub F eF) * l1_norm eG )%Q.
Proof.
  intros n F G eF eG.
  eapply Qle_trans.
  - apply Qle_of_Qeq.
    apply l1_norm_ext; intro U.
    apply (@conv_error_split n F G eF eG U).
  - eapply Qle_trans.
    + apply l1_add_bound.
    + apply Qplus_le_compat.
      * (* || F ⋆ (G-eG) ||₁ <= ||F||₁ ||G-eG||₁ *)
        apply l1_conv_bound.
      * (* || (F-eF) ⋆ eG ||₁ <= ||F-eF||₁ ||eG||₁ *)
        apply l1_conv_bound.
Qed.

Lemma trace_boolish_k_le_subexpr_l :
  forall n (sq : Vector.t Q n) (e1 e2 : GA_expr n) k d,
    trace_boolish_k_le sq (Mul e1 e2) k d ->
    trace_boolish_k_le sq e1 k d.
Proof. intros; simpl in *; tauto. Qed.

Lemma trace_boolish_k_le_subexpr_r :
  forall n (sq : Vector.t Q n) (e1 e2 : GA_expr n) k d,
    trace_boolish_k_le sq (Mul e1 e2) k d ->
    trace_boolish_k_le sq e2 k d.
Proof. intros; simpl in *; tauto. Qed.

Lemma trace_boolish_k_le_subexpr_conv_l :
  forall n (sq : Vector.t Q n) (e1 e2 : GA_expr n) k d,
    trace_boolish_k_le sq (Conv e1 e2) k d ->
    trace_boolish_k_le sq e1 k d.
Proof. intros; simpl in *; tauto. Qed.

Lemma trace_boolish_k_le_subexpr_conv_r :
  forall n (sq : Vector.t Q n) (e1 e2 : GA_expr n) k d,
    trace_boolish_k_le sq (Conv e1 e2) k d ->
    trace_boolish_k_le sq e2 k d.
Proof. intros; simpl in *; tauto. Qed.

Open Scope Q_scope.
Lemma gp_boolish_witness_error_only :
  forall n (sq : Vector.t Q n) (F G : MV n) k d,
    (forall i : Fin.t n, Qabs (Vector.nth sq i) == 1) ->
    boolish_k_le F k d ->
    boolish_k_le G k d ->
    exists LF LG,
      l1_norm (mv_sub F LF) <= d /\
      l1_norm (mv_sub G LG) <= d /\
      l1_norm (mv_sub (mv_gp sq F G) (mv_gp sq LF LG))
      <= l1_norm F * d + d * l1_norm LG.
Proof.
  intros n sq F G k d Hsig HF HG.
  destruct HF as [csF [gsF [HwfF [HlenF HdF]]]].
  destruct HG as [csG [gsG [HwfG [HlenG HdG]]]].
  set (LF := lincomb_embed csF gsF).
  set (LG := lincomb_embed csG gsG).
  exists LF, LG; repeat split; try assumption.
  eapply Qle_trans.
  - apply gp_error_bound_l1; auto.
  - apply Qplus_le_compat.
    + (* ∥F∥₁ * ∥G - LG∥₁ <= ∥F∥₁ * d *)
      apply Qmult_le_compat_r with (z := l1_norm F) in HdG.
      * setoid_rewrite Qmult_comm in HdG at 1.
        setoid_rewrite Qmult_comm in HdG at 2.
        exact HdG.
      * apply l1_norm_nonneg.
    + (* ∥F - LF∥₁ * ∥LG∥₁ <= d * ∥LG∥₁ *)
      apply Qmult_le_compat_r; [exact HdF | apply l1_norm_nonneg].
Qed.

(* Later do :

Definition embed_gp_closed_up_to {n : nat} (sq : Vector.t Q n) (delta : Q) : Prop :=
  forall (g h : Corner n -> bool),
    exists gh : Corner n -> bool,
      l1_norm (mv_sub (mv_gp sq (embed g) (embed h)) (embed gh)) <= delta.

Lemma boolish_k_le_gp_of_boolish_k_le_up_to :
  forall n (sq : Vector.t Q n) (F G : MV n) k d delta,
    (forall i : Fin.t n, Qabs (Vector.nth sq i) == 1) ->
    embed_gp_closed_up_to sq delta ->
    boolish_k_le F k d ->
    boolish_k_le G k d ->
    exists cs gs,
      wf_lincomb cs gs /\
      (length gs <= (k * k))%nat /\
      l1_norm (mv_sub (mv_gp sq F G) (lincomb_embed cs gs))
      <=
        (* the GP “input approximation” error *)
        (l1_norm F) * d
      + d * (l1_norm (lincomb_embed cs gs))
        (* plus the embed-closure slack accumulated across k*k terms;
           in the proof it will look like (sum_abs(csF)*sum_abs(csG))*delta *)
      + delta * (l1_norm (lincomb_embed cs gs)).
Proof.
Admitted.
*)
Close Scope Q_scope.

Lemma boolish_k_le_conv_of_boolish_k_le :
  forall n (F G : MV n) k d,
    boolish_k_le F k d ->
    boolish_k_le G k d ->
    exists k' d',
      boolish_k_le (mv_conv F G) k' d'.
Proof.
Admitted.

Lemma boolish_k_le_gp_propagate :
  forall n (sq : Vector.t Q n)
         (F G : MV n)
         k d,
    (forall i : Fin.t n, Qabs (Vector.nth sq i) == 1) ->
    boolish_k_le F k d ->
    boolish_k_le G k d ->
    exists cs gs k' d',
      wf_lincomb cs gs /\
      (length gs <= k')%nat /\
      l1_norm (mv_sub (mv_gp sq F G)
                      (lincomb_embed cs gs)) <= d'.
Proof.
Admitted.

Lemma boolish_k_le_conv :
  forall n (F G : MV n) k1 k2 d1 d2,
    boolish_k_le F k1 d1 ->
    boolish_k_le G k2 d2 ->
    boolish_k_le (mv_conv F G) (k1 * k2)
      (d1 * l1_norm G + l1_norm F * d2 + d1 * d2).
Proof.
  intros n F G k1 k2 d1 d2
         [csF [gsF [HwfF [HlenF HdF]]]]
         [csG [gsG [HwfG [HlenG HdG]]]].

  set (F0 := lincomb_embed csF gsF).
  set (G0 := lincomb_embed csG gsG).

  exists (mul_coeffs csF csG), (and_gens gsF gsG).
  repeat split.
  - (* wf *)
    eapply wf_lincomb_mul_and; eauto.
  - (* length bound *)
    rewrite length_and_gens.
    (* |gsF|*|gsG| <= k1*k2 *)
    apply Nat.mul_le_mono; lia.
  - (* error bound *)
    (* Rewrite witness as mv_conv F0 G0 using lincomb_embed_conv *)
    set (W := lincomb_embed (mul_coeffs csF csG) (and_gens gsF gsG)).
    assert (HW : mv_conv F0 G0 = W).
    {
      subst W F0 G0.
      apply (lincomb_embed_conv (n:=n)); assumption.
    }

    (* Reduce to bounding || mv_conv F G - mv_conv F0 G0 ||_1 *)
    eapply Qle_trans.
    + (* replace W by mv_conv F0 G0 inside l1_norm *)
      apply Qle_of_Qeq.
      apply l1_norm_ext; intro m.
      unfold W.
      rewrite <- HW.
      reflexivity.
    + (* Now use conv_error_split + triangle + submultiplicativity *)
      eapply Qle_trans.
      * (* split via conv_error_split pointwise, then l1_add_bound *)
        eapply Qle_trans.
        -- apply Qle_of_Qeq.
           apply l1_norm_ext; intro m.
           exact (conv_error_split (n:=n) (F:=F) (G:=G) (eF:=F0) (eG:=G0) m).
        -- eapply Qle_trans.
           ++ apply l1_add_bound.
           ++ apply Qplus_le_compat.
              ** (* first term *)
                 eapply Qle_trans.
                 --- apply l1_conv_submultiplicative.
                 --- (* <= ||F|| * d2 *)
                     apply Qmult_le_compat_l.
                     { apply l1_norm_nonneg. }
                     exact HdG.
              ** (* second term *)
                 eapply Qle_trans.
                 --- apply l1_conv_submultiplicative.
                 --- (* <= d1 * ||G0|| *)
                     apply Qmult_le_compat_r.
                     { apply l1_norm_nonneg. }
                     exact HdF.
      * (* bound ||G0|| <= ||G|| + d2, then algebra *)
        (* First: ||G0|| = ||G - (G-G0)|| <= ||G|| + ||G-G0|| <= ||G|| + d2 *)
        assert (HG0_le : l1_norm G0 <= l1_norm G + d2).
        {
          subst G0.
          (* G0 = G - (G-G0) *)
          rewrite <- (mv_sub_cancel (n:=n) (G:=G) (G0:=lincomb_embed csG gsG)).
          eapply Qle_trans.
          - apply l1_sub_bound.
          - apply Qplus_le_compat.
            + apply Qle_refl.
            + exact HdG.
        }

        (* Use HG0_le to rewrite d1*||G0|| <= d1*(||G||+d2) *)
        (* and then expand to match goal *)
        (* Current bound from previous step is:
             ||F||*d2 + d1*||G0|| *)
        eapply Qle_trans.
        -- (* replace d1*||G0|| by d1*(||G||+d2) *)
           apply Qplus_le_compat.
           ++ apply Qle_refl.
           ++ apply Qmult_le_compat_l.
              { (* need 0 <= d1; follows from HdF since l1_norm >=0 *)
                eapply Qle_trans; [apply l1_norm_nonneg | exact HdF]. }
              exact HG0_le
        -- (* algebra: ||F||*d2 + d1*(||G||+d2) = d1*||G|| + ||F||*d2 + d1*d2 *)
           (* expand and reorder *)
           ring_simplify.
           (* `ring_simplify` may or may not close; if it doesn't, use: *)
           ring.
Qed.

(*
===============================================================================


Lemma mv_conv_scale_l :
  forall n (c : Q) (F G : MV n),
    mv_conv (mv_scale c F) G = mv_scale c (mv_conv F G).
Proof.
  intros n c F G.
  apply functional_extensionality; intro U.
  unfold mv_conv, mv_scale.
  (* pull c through both sums *)
  rewrite <- sumQ_map_scale_l.
  apply Qeq_trans with
    (c * sumQ
       (map (fun A =>
          sumQ (map (fun B =>
            if mask_eq_dec (mask_xor A B) U
            then (F A * G B)%Q else 0%Q) (all_masks n)))
         (all_masks n)))%Q.
  - (* show LHS equals c * ... by rewriting inner sums *)
    apply sumQ_map_ext; intros A _.
    rewrite <- sumQ_map_scale_l.
    apply sumQ_map_ext; intros B _.
    destruct (mask_eq_dec (mask_xor A B) U); simpl; ring.
  - reflexivity.
Qed.

Lemma mv_conv_scale_r :
  forall n (c : Q) (F G : MV n),
    mv_conv F (mv_scale c G) = mv_scale c (mv_conv F G).
Proof.
  intros n c F G.
  apply functional_extensionality; intro U.
  unfold mv_conv, mv_scale.
  rewrite <- sumQ_map_scale_l.
  apply Qeq_trans with
    (c * sumQ
       (map (fun A =>
          sumQ (map (fun B =>
            if mask_eq_dec (mask_xor A B) U
            then (F A * G B)%Q else 0%Q) (all_masks n)))
         (all_masks n)))%Q.
  - apply sumQ_map_ext; intros A _.
    rewrite <- sumQ_map_scale_l.
    apply sumQ_map_ext; intros B _.
    destruct (mask_eq_dec (mask_xor A B) U); simpl; ring.
  - reflexivity.
Qed.


(* --- key Fourier-basis fact: Pi convolution is Kronecker --- *)

Lemma Pi_conv :
  forall n (a b : Corner n),
    mv_conv (Pi a) (Pi b)
    =
    if corner_eqb a b then Pi a else (@mv_zero n).
Proof.
  intros n a b.
  apply functional_extensionality; intro U.
  unfold mv_conv, Pi, mv_zero.

  (* rewrite the double sum by choosing B = A xor U *)
  (* We keep your existing style: sumQ_map_ext and mask_eq_dec splitting. *)
  (* The key identity is chi_mul and Walsh orthogonality walsh_sum_masks_closed. *)

  (* First: collapse the B-sum to only B = xor A U contributions. *)
  (* This is a standard “indicator picks one term” trick; we use your list facts implicitly. *)
  (* We’ll directly transform the inner map by extensionality and then use a lemma:
       sum_{B} if (A xor B = U) then f B else 0 = f (A xor U)
     over all_masks. This is already provable from all_masks completeness.
     If you already have such a lemma, replace the admitted block with it. *)

  (* --- If you don't yet have the “indicator picks unique B” lemma, you can keep Pi_conv
         admitted temporarily and still finish lincomb_embed_conv_wf. But below is the
         intended endgame using Walsh orthogonality. --- *)

Admitted.


Lemma embed_conv_and :
  forall n (g1 g2 : Corner n -> bool),
    mv_conv (embed g1) (embed g2) = embed (and_gen g1 g2).
Proof.
  intros n g1 g2.
  apply functional_extensionality; intro U.
  unfold embed, and_gen.

  (* expand embed as sum over corners of bQ * Pi, then use bilinearity + Pi_conv *)
  (* embed g m = Σ_a bQ(g a) * Pi a m *)

  (* We do it pointwise at U, by pushing mv_conv through sums.
     It’s cleaner to use the already-proved bilinearity lemmas for mv_conv
     together with a “lincomb over corners” view; but embed is defined as a sumQ map.
     So we reason with sumQ_map_ext / sumQ_map_add / sumQ_map_scale_l. *)

Admitted.

(* --- list-structure lemmas for mul_coeffs/and_gens --- *)

Lemma mul_coeffs_cons :
  forall (c : Q) cs1 cs2,
    mul_coeffs (c :: cs1) cs2 =
    (map (fun c2 => (c * c2)%Q) cs2) ++ mul_coeffs cs1 cs2.
Proof.
  intros c cs1 cs2.
  unfold mul_coeffs. simpl.
  rewrite concat_app. reflexivity.
Qed.

Lemma and_gens_cons {n} :
  forall (g : Corner n -> bool) gs1 gs2,
    and_gens (g :: gs1) gs2 =
    (map (and_gen g) gs2) ++ and_gens gs1 gs2.
Proof.
  intros g gs1 gs2.
  unfold and_gens. simpl.
  rewrite concat_app. reflexivity.
Qed.

(* --- the lemma you need, with the right hypotheses --- *)

Lemma lincomb_embed_conv {n : nat} :
  forall (cs1 cs2 : list Q)
         (gs1 gs2 : list (Corner n -> bool)),
    wf_lincomb cs1 gs1 ->
    wf_lincomb cs2 gs2 ->
    mv_conv (lincomb_embed cs1 gs1) (lincomb_embed cs2 gs2)
    =
    lincomb_embed (mul_coeffs cs1 cs2) (and_gens gs1 gs2).
Proof.
  intros cs1 cs2 gs1 gs2 Hwf1 Hwf2.
  revert gs1 Hwf1.
  induction cs1 as [|c cs1 IH]; intros gs1 Hwf1.
  - destruct gs1 as [|g gs1]; simpl in *.
    + (* [] [] *)
      apply functional_extensionality; intro U.
      unfold mv_conv, mv_zero.
      (* mv_conv 0 X = 0 *)
      rewrite sumQ_map_const0. reflexivity.
    + discriminate.
  - destruct gs1 as [|g gs1]; simpl in *.
    + discriminate.
    + (* main step *)
      assert (Hwf1' : wf_lincomb cs1 gs1).
      { unfold wf_lincomb in *; simpl in *; lia. }

      (* unfold head+tail and use bilinearity *)
      rewrite mv_conv_add_l.
      rewrite mv_conv_add_r.
      rewrite mv_conv_scale_l.
      rewrite mv_conv_scale_r.

      (* identify mv_conv (embed g) (lincomb_embed cs2 gs2) as lincomb with and_gens *)
      (* We do it by induction on cs2/gs2 inside lincomb_embed, but since we have
         embed_conv_and, the outer induction is enough using lincomb_embed recursion. *)

      (* Now rewrite RHS using cons structure and your lincomb_embed_app (needs wf!) *)
      rewrite mul_coeffs_cons.
      rewrite and_gens_cons.

      (* Split lincomb_embed over ++ using your proved lemma lincomb_embed_app *)
      (* First, show both parts are wf_lincomb *)
      assert (Hwf_head : wf_lincomb (map (fun c2 => (c*c2)%Q) cs2)
                                   (map (and_gen g) gs2)).
      { unfold wf_lincomb in *.
        rewrite map_length, map_length. exact Hwf2. }

      assert (Hwf_tail : wf_lincomb (mul_coeffs cs1 cs2) (and_gens gs1 gs2)).
      { (* this is where you may want a dedicated wf lemma for mul_coeffs/and_gens;
           but under wf cs1 gs1 and wf cs2 gs2 it’s true because both are cartesian products. *)
        unfold wf_lincomb in *.
        (* length mul_coeffs = |cs1|*|cs2|, length and_gens = |gs1|*|gs2| *)
        (* prove these two length facts once and reuse *)
        admit.
      }

      apply functional_extensionality; intro U.
      (* use lincomb_embed_app pointwise *)
      rewrite (lincomb_embed_app (n:=n)
                (cs1:=map (fun c2 => (c*c2)%Q) cs2)
                (cs2:=mul_coeffs cs1 cs2)
                (gs1:=map (and_gen g) gs2)
                (gs2:=and_gens gs1 gs2)
                Hwf_head Hwf_tail U).
      (* Now it remains to match the two summands with the LHS decomposition. *)

      (* First summand: c·embed g convolved with lincomb2 *)
      (* This is exactly “scale then distribute then use embed_conv_and” *)
      (* Again, best done with a helper lemma:
           mv_conv (embed g) (lincomb_embed cs2 gs2)
           = lincomb_embed cs2 (map (and_gen g) gs2)
         under wf cs2 gs2. *)
      admit.
Qed.


Lemma boolish_k_le_conv :
  forall n (F G : MV n) k1 k2 d1 d2,
    boolish_k_le F k1 d1 ->
    boolish_k_le G k2 d2 ->
    boolish_k_le (mv_conv F G) (k1 * k2)
      (d1 * l1_norm G + l1_norm F * d2 + d1 * d2).
Proof.
  intros n F G k1 k2 d1 d2
         [csF [gsF [HwfF [HlenF HdF]]]]
         [csG [gsG [HwfG [HlenG HdG]]]].

  set (F0 := lincomb_embed csF gsF).
  set (G0 := lincomb_embed csG gsG).

  (* Witness: product lincomb *)
  exists (mul_coeffs csF csG), (and_gens gsF gsG).
  repeat split.
  - (* wf_lincomb for the new witness *)
    (* You need: wf_lincomb_mul / wf_lincomb_and_gens *)
    admit.
  - (* length bound: |csF|*|csG| <= k1*k2 *)
    (* If mul_coeffs is cartesian product, length is length csF * length csG. *)
    (* Use HlenF, HlenG and Nat.mul_le_mono / lia. *)
    admit.
  - (* error bound *)
    (* We bound: ||F*G - F0*G0||_1 *)
    (* Add/subtract: F0*G and F0*G0 *)
    eapply Qle_trans.
    2: {
      (* final algebra to match stated RHS *)
      admit.
    }

    (* Step 1: triangle split *)
    (* ||F*G - F0*G0|| <= ||F*G - F0*G|| + ||F0*G - F0*G0|| *)
    admit.

    (* Step 2: first term = ||(F-F0)*G||, bound by d1 * ||G|| *)
    (* use l1_conv_submultiplicative and HdF *)

    (* Step 3: second term = ||F0*(G-G0)||, bound by ||F0|| * d2 *)
    (* use l1_conv_submultiplicative and HdG *)

    (* Step 4: replace ||F0|| by ||F|| + d1 to get:
         (||F|| + d1)*d2 = ||F||*d2 + d1*d2
       using triangle: ||F0|| <= ||F|| + ||F-F0|| <= ||F|| + d1 *)
 

    (* Step 5: use lincomb_embed_conv to identify mv_conv F0 G0 with witness lincomb *)

Admitted.

Lemma lincomb_embed_gp :
  forall (n : nat) (sq : Vector.t Q n)
         (csF csG : list Q)
         (gsF gsG : list (Corner n -> bool))
         (U : Mask n),
    @mv_gp n sq (lincomb_embed csF gsF) (lincomb_embed csG gsG) U
    =
    lincomb_embed (mul_coeffs csF csG) (and_gens gsF gsG) U.
Proof.
Admitted.

Lemma boolish_k_le_gp :
  forall n (sq : Vector.t Q n) (F G eF eG : MV n) k1 k2 d1 d2,
    (forall i : Fin.t n, Qabs (Vector.nth sq i) = 1%Q) ->
    boolish_k_le F k1 d1 ->
    boolish_k_le G k2 d2 ->
    boolish_k_le (mv_gp sq F G) (k1 * k2)
      (d1 * l1_norm G + l1_norm F * d2 + d1*d2).
Proof.
Admitted.

Lemma trace_boolish_le_implies_small_k :
  exists P : nat -> nat -> nat,
  forall n (sq : Vector.t Q n) (e : GA_expr n) (d : Q),
    trace_boolish_le sq e d ->
    exists k,
      trace_boolish_k_le sq e k d /\
      (k <= P (expr_size e)
              (Z.to_nat (Qnum (exc_l1 (exc_of sq e)))))%nat.
Proof.
Admitted.

Lemma trace_boolish_le_to_k :
  forall n (sq : Vector.t Q n) (e : GA_expr n) (d : Q),
    trace_boolish_le sq e d ->
    exists k,
      trace_boolish_k_le sq e k d
      /\ (* k bounded by a function of exc_l1 (exc_of sq e) *) True.
Proof.
Admitted.

Lemma trace_k_bound_by_size :
  forall (n : nat) (sq : Vector.t Q n) (e : GA_expr n) (d : Q) (k : nat),
    trace_boolish_k_le sq e k d ->
    exists a b c,
      k <= poly_in_nat (expr_size e)
                       (Z.to_nat (Qnum (exc_l1 (exc_of sq e))))
                       a b c.
Proof.
Admitted.

Fixpoint trace_boolish_global_le {n}
  (sq : Vector.t Q n) (e : GA_expr n)
  (gs : list (Corner n -> bool)) (d : Q) : Prop :=
  match e with
  | Basis _ | Scalar _ =>
      exists cs, wf_lincomb cs gs /\
        (l1_norm (mv_sub (eval_expr sq e) (lincomb_embed cs gs)) <= d)%Q

  | Cln_Grade.Add e1 e2 =>
      trace_boolish_global_le sq e1 gs d /\
      trace_boolish_global_le sq e2 gs d /\
      exists cs, wf_lincomb cs gs /\
        (l1_norm (mv_sub (eval_expr sq (Cln_Grade.Add e1 e2)) (lincomb_embed cs gs)) <= d)%Q

  | Mul e1 e2 =>
      trace_boolish_global_le sq e1 gs d /\
      trace_boolish_global_le sq e2 gs d /\
      exists cs, wf_lincomb cs gs /\
        (l1_norm (mv_sub (eval_expr sq (Mul e1 e2)) (lincomb_embed cs gs)) <= d)%Q

  | Conv e1 e2 =>
      trace_boolish_global_le sq e1 gs d /\
      trace_boolish_global_le sq e2 gs d /\
      exists cs, wf_lincomb cs gs /\
        (l1_norm (mv_sub (eval_expr sq (Conv e1 e2)) (lincomb_embed cs gs)) <= d)%Q
  end.

Lemma trace_boolish_k_le_of_global {n : nat} :
  forall (sq : Vector.t Q n) (e : GA_expr n)
         (gs : list (Corner n -> bool)) (d : Q),
    trace_boolish_global_le sq e gs d ->
    exists k : nat, trace_boolish_k_le sq e k d.
Proof.
Admitted.

Theorem hard_family_separates :
  forall d : Q,
  exists f : forall n, Corner n -> bool,
  exists c : nat,
    forall n (sq : Vector.t Q n) (e : GA_expr n) k,
      computes sq e (f n) ->
      trace_boolish_k_le sq e k d ->
      (* optional: k <= poly(n, size e) or k <= poly(n) *)
      (Qpow2 (c * n) <= exc_l1 (exc_of sq e))%Q.
Proof.
Admitted.
*)