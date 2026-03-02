(*
  ============================================================
  File: Cln_CompositeExcursion.v
  ============================================================
*)

Require Import Cln_Full.
Require Import Cln_Grade.
Require Import Cln_finite_l1_submultiplicativity.
Require Import Cln_BoolDist.
Require Import Cln_SupportAlgebra.

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

Lemma boolish_k_le_tol_mono :
  forall n (F : MV n) k d1 d2,
    d1 <= d2 ->
    boolish_k_le F k d1 ->
    boolish_k_le F k d2.
Proof.
  intros n F k d1 d2 Hle [cs [gs [Hwf [Hlen Hd]]]].
  exists cs, gs; repeat split; try assumption.
  eapply Qle_trans; eassumption.
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


(* Helper: |signed s| = 1 *)
Open Scope Q_scope.

Lemma Qabs_signed : forall s : bool,
  Qabs (signed s) == 1.
Proof.
  destruct s; unfold signed; simpl.
  - reflexivity.
  - reflexivity.
Qed.

Lemma pow2_add : forall a b, pow2 (a + b) == pow2 a * pow2 b.
Proof.
  induction a as [|a IHa]; intro b; simpl.
  - ring.
  - rewrite IHa. ring.
Qed.

Lemma pow2_pos_Q : forall n, 0 < pow2 n.
Proof.
  induction n as [|n IH]; simpl.
  - reflexivity.
  - apply Qmult_lt_0_compat.
    + unfold Qlt; simpl; lia.
    + exact IH.
Qed.

Lemma pow2_S_eq : forall n, pow2 (S n) == 2 * pow2 n.
Proof. intro n. simpl pow2. ring. Qed.

Lemma pow2_ge_1 : forall n, 1 <= pow2 n.
Proof.
  induction n as [|n IH]; simpl.
  - apply Qle_refl.
  - apply (Qle_trans _ (pow2 n)).
    + exact IH.
    +
      assert (H12 : (1:Q) <= 2).
      { unfold Qle; simpl; lia. }
      assert (H0 : 0 <= pow2 n).
      { apply Qlt_le_weak. exact (pow2_pos_Q n). }
      pose proof (Qmult_le_compat_r (1:Q) 2 (pow2 n) H12 H0) as H.
      rewrite Qmult_1_l in H.
      exact H.
Qed.

Lemma pow2_le_mono : forall a b, (a <= b)%nat -> pow2 a <= pow2 b.
Proof.
  intros a b Hab.
  replace b with (a + (b - a))%nat by lia.
  rewrite pow2_add.
  rewrite <- (Qmult_1_r (pow2 a)) at 1.
  apply Qmult_le_l.
  - apply pow2_pos_Q.
  - apply pow2_ge_1.
Qed.

(* ============================================================ *)
(* The pointwise lower bound                                     *)
(* ============================================================ *)

Lemma embed_IP_abs_lower : forall m (M : Mask (m + m)),
  (m >= 1)%nat ->
  1 / pow2 (m + 1) <= Qabs (embed (@IP_n_func (m + m)) M).
Proof.
  intros m M Hm.
  assert (Hm' : (m > 0)%nat) by lia.
  destruct (signed_walsh_IP_magnitude m M Hm') as [s Hs].
  assert (Hembed := embed_via_signed_walsh (m+m) (@IP_n_func (m+m)) M).

  set (p := pow2 m) in *.
  assert (Hpmm : pow2 (m + m) == p * p)
    by (subst p; rewrite <- pow2_add; reflexivity).
  assert (Hpm1 : pow2 (m + 1) == 2 * p)
    by (subst p; replace (m+1)%nat with (S m) by lia; apply pow2_S_eq).
  assert (Hp_pos : 0 < p) by (subst p; apply pow2_pos_Q).
  assert (Hp_nz : ~ p == 0) by (subst p; apply pow2_nonzero).

  assert (Hp_ge2 : 2 <= p).
    { subst p.
      destruct m as [|m']; [lia|].
      simpl.
      
      pose proof (pow2_ge_1 m') as Hpow1.
      setoid_replace ((1 + 1) * pow2 m') with (pow2 m' + pow2 m') by ring.
      setoid_replace 2 with (1 + 1)%Q by ring.
      apply Qplus_le_compat; exact Hpow1.
    }

  assert (H2p_pos : 0 < 2 * p)
    by (apply Qmult_lt_0_compat; [reflexivity | exact Hp_pos]).

  (* Replace inject_Z(2^m) by p in Hembed *)
  setoid_rewrite Hs in Hembed.
  setoid_rewrite <- pow2_injectZ in Hembed.
  (* NOTE: do NOT rewrite Hpmm in Hembed here —
     it won't penetrate the if-then-else *)

  destruct (mask_eq_dec M mask_empty) as [Meq | Mneq].

  - (* ═══════════ M = ∅ ═══════════ *)
    subst M.
    (* The if resolved, exposing pow2(m+m). We rewrite in each Hval. *)

    destruct s.

    + (* ── s = true: signed true = -1 ── *)
      (* embed = 1/(p²)·((1/2)·p² - (1/2)·(-1·p)) = 1/2 + 1/(2p) *)
        
        
      assert (Hval : embed (@IP_n_func (m+m)) mask_empty == (1#2) + 1/(2*p)).
      {
        eapply Qeq_trans; [exact Hembed|].
        setoid_rewrite Hpmm.
        change (pow2 m) with p.
        unfold signed.
        ring_simplify.
        setoid_replace (1/(2*p) + (1#2)) with ((1#2) + 1/(2*p)) by ring.
        change (p ^ 2) with (p * p).

        (* prove p*p nonzero in the right (setoid) sense *)
        assert (Hpp_nz : ~ (p * p) == 0).
        { intro Hpp.
          destruct (Qmult_integral p p) as [Hp0|Hp0]; try exact Hpp;
          apply Hp_nz; exact Hp0.
        }
        ring_simplify.
        change (p ^ 2) with (p * p).
        setoid_replace ((1#2) * (1 / (p*p)) * (p*p))
          with ((1#2) * ((1 / (p*p)) * (p*p))) by ring.
        ring_simplify.
        field. exact Hp_nz.

      }

      (* embed > 0 *)
      assert (Hpos : 0 < (1#2) + 1/(2*p)).
      { apply Qlt_le_trans with (1#2); [reflexivity|].
        rewrite <- (Qplus_0_r (1#2)) at 1.
        apply Qplus_le_compat; [apply Qle_refl|].
        apply Qlt_le_weak.
        apply Qlt_shift_div_l; [exact H2p_pos|].
        ring_simplify. reflexivity. }

      setoid_rewrite Hval.
      setoid_rewrite (Qabs_pos _ (Qlt_le_weak _ _ Hpos)).
      setoid_rewrite Hpm1.
      (* Goal: 1/(2*p) ≤ (1#2) + 1/(2*p) *)
      rewrite <- (Qplus_0_l (1/(2*p))) at 1.
      apply Qplus_le_compat; [discriminate | apply Qle_refl].

    + (* ── s = false: signed false = 1 ── *)
      (* embed = 1/(p²)·((1/2)·p² - (1/2)·(1·p)) = 1/2 - 1/(2p) *)
      assert (Hval : embed (@IP_n_func (m+m)) mask_empty == (1#2) - 1/(2*p)).
      { eapply Qeq_trans; [exact Hembed|].
        setoid_rewrite Hpmm.
        unfold signed.
        change (pow2 m) with p.
        field. exact Hp_nz. }

      (* ── Key arithmetic chain ── *)

      (* Step A: 1 ≤ (1#2) * p, since p ≥ 2 *)
      assert (Hhalf_p : 1 <= (1#2) * p).
      { eapply Qle_trans with ((1#2) * 2).
        - unfold Qle; simpl; lia.   (* 1 ≤ 1 *)
        - apply Qmult_le_l; [reflexivity | exact Hp_ge2]. }

      (* Step B: 1/p ≤ 1/2 *)
      assert (H1p : 1/p <= (1#2)).
      {
        (* multiply both sides by (2*p) > 0 *)
        apply (Qmult_le_l _ _ (2*p)); [ exact H2p_pos | ].

        (* goal is now: (2*p) * (1/p) <= (2*p) * (1#2) *)
        (* simplify; this should reduce to 2 <= p *)
        field_simplify; try exact Hp_nz; try discriminate.
        (* after field_simplify, the goal should be 2 <= p *)
        exact Hp_ge2.
      }

      assert (Hfrac_le1 : 1/(2*p) <= 1/p).
      {
        (* multiply both sides by p > 0 *)
        apply (Qmult_le_l _ _ p); [ exact Hp_pos | ].
        (* p*(1/(2*p)) <= p*(1/p) *)
        field_simplify; try exact Hp_nz; try discriminate.
      }
      assert (Hfrac_le : 1/(2*p) <= (1#2)).
      { eapply Qle_trans; [ exact Hfrac_le1 | exact H1p ]. }
      
      (* Step D: embed ≥ 0, since 1/(2p) ≤ 1/2 *)
      assert (Hpos : 0 <= (1#2) - 1/(2*p)).
      { apply -> Qle_minus_iff. exact Hfrac_le. }

      setoid_rewrite Hval.
      setoid_rewrite (Qabs_pos _ Hpos).
      setoid_rewrite Hpm1.

      (* Goal: 1/(2*p) ≤ (1#2) - 1/(2*p) *)
      (* ↔ 0 ≤ (1#2) - 1/(2p) - 1/(2p) == (1#2) - 1/p *)
      apply Qle_minus_iff.
      (* Goal: 0 ≤ (1#2) - 1/(2*p) + - (1/(2*p)) *)
      (* Simplify the RHS to (1#2) - 1/p *)
      assert (Hdiff : (1#2) - 1/(2*p) + -(1/(2*p)) == (1#2) - 1/p).
      { field. exact Hp_nz. }
      setoid_rewrite Hdiff.
      (* Goal: 0 ≤ (1#2) - 1/p *)
      apply -> Qle_minus_iff.
      (* Goal: 1/p ≤ (1#2) *)
      exact H1p.

  - (* ═══════════ M ≠ ∅ ═══════════ *)
    (* The if gave 0, so embed = 1/(p²)·(0 - (1/2)·(signed s · p)) *)
    destruct s.

    + (* ── s = true: signed true = -1, so embed = 1/(2p) ── *)
      assert (Hval : embed (@IP_n_func (m+m)) M == 1/(2*p)).
      { eapply Qeq_trans; [exact Hembed|].
        setoid_rewrite Hpmm.
        change (pow2 m) with p.
        unfold signed.
        field. exact Hp_nz.
      }
      
      assert (Hpos : 0 < 1/(2*p)).
      { apply Qlt_shift_div_l; [exact H2p_pos|].
        ring_simplify. reflexivity.
      }

      setoid_rewrite Hval.
      setoid_rewrite (Qabs_pos _ (Qlt_le_weak _ _ Hpos)).
      setoid_rewrite Hpm1.
      apply Qle_refl.

    + (* ── s = false: signed false = 1, so embed = -1/(2p) ── *)
      assert (Hval : embed (@IP_n_func (m+m)) M == -(1/(2*p))).
      { eapply Qeq_trans; [exact Hembed|].
        setoid_rewrite Hpmm.
        change (pow2 m) with p.
        unfold signed.
        field. exact Hp_nz.
      }

      assert (Hfrac_pos : 0 < 1/(2*p)).
      { apply Qlt_shift_div_l; [exact H2p_pos|].
        ring_simplify. reflexivity.
      }

      assert (Hneg : -(1/(2*p)) <= 0).
      { setoid_replace 0 with (-(0)) by ring.
        apply Qopp_le_compat.
        apply Qlt_le_weak. exact Hfrac_pos.
      }

      setoid_rewrite Hval.
      setoid_rewrite (Qabs_neg _ Hneg).
      setoid_replace (-(-(1/(2*p)))) with (1/(2*p)) by ring.
      setoid_rewrite Hpm1.
      apply Qle_refl.
Qed.

Lemma sumQ_map_lower_bound :
  forall (A : Type) (f : A -> Q) (l : list A) (c : Q),
    (forall x, List.In x l -> c <= f x) ->
    (inject_Z (Z.of_nat (length l)) * c <= sumQ (List.map f l))%Q.
Proof.
  intros A f l c Hbound.
  induction l as [|a tl IH]; simpl.
  - (* base: inject_Z 0 * c <= 0 *)
    ring_simplify. apply Qle_refl.
  - (* step: (1 + length tl) * c <= f a + sumQ (map f tl) *)
    rewrite Zpos_P_of_succ_nat.
    change (Z.succ (Z.of_nat (length tl))) with (Z.of_nat (length tl) + 1)%Z.
    setoid_rewrite inject_Z_plus.
    change (inject_Z 1) with 1.
    setoid_replace ((inject_Z (Z.of_nat (length tl)) + 1) * c)
      with (c + inject_Z (Z.of_nat (length tl)) * c) by ring.
    apply Qplus_le_compat.
    + apply Hbound. left. reflexivity.
    + apply IH. intros x Hx. apply Hbound. right. exact Hx.
Qed.

Theorem l1_norm_embed_IP_lower_bound : forall m,
  (m >= 2)%nat ->
  (pow2 (m - 1) <= l1_norm (embed (@IP_n_func (m + m))))%Q.
Proof.
  intros m Hm.
  unfold l1_norm.
  
  (* Step 1: Each |embed(IP)(M)| ≥ 1/pow2(m+1) *)
  assert (Hpointwise : forall M, List.In M (all_masks (m+m)) ->
    1 / pow2 (m + 1) <= Qabs (embed (@IP_n_func (m + m)) M)).
  { intros M _. apply embed_IP_abs_lower. lia. }
  
  (* Step 2: Sum ≥ |masks| * (1/pow2(m+1)) *)
  assert (Hsum : inject_Z (Z.of_nat (length (all_masks (m+m)))) * (1 / pow2 (m+1))
                 <= sumQ (List.map (fun M => Qabs (embed (@IP_n_func (m+m)) M))
                                   (all_masks (m+m)))).
  { apply sumQ_map_lower_bound. exact Hpointwise. }
  
  (* Step 3: |all_masks(2m)| = 2^(2m) *)
  rewrite all_masks_length_pow2 in Hsum.
  
  (* Step 4: 2^(2m) / 2^(m+1) = 2^(m-1) *)
  (* 2^(2m) * (1/2^(m+1)) = 2^(2m) / 2^(m+1) = 2^(2m - m - 1) = 2^(m-1) *)
  eapply Qle_trans; [|exact Hsum].
  
  assert (Hpow_split : pow2 (m - 1) * pow2 (m + 1) == pow2 (m + m)).
  { rewrite <- pow2_add. f_equiv. lia. }

  assert (Hrewrite : pow2 (m - 1) == inject_Z (Z.of_nat (Nat.pow 2 (m+m))) * (1 / pow2 (m+1))).
  {
    assert (Hconv : inject_Z (Z.of_nat (Nat.pow 2 (m+m))) == pow2 (m+m)).
    { rewrite Nat2Z.inj_pow. simpl (Z.of_nat 2).
      symmetry. rewrite pow2_injectZ.
      change Cln_SupportAlgebra.inject_Z with QArith_base.inject_Z.
      reflexivity. }
    assert (Hpm1_nz : ~ pow2 (m + 1) == 0) by apply pow2_nonzero.
    setoid_rewrite Hconv.
    (* Goal: pow2 (m - 1) == pow2 (m + m) * (1 / pow2 (m + 1)) *)
    setoid_rewrite <- Hpow_split.
    (* Goal: pow2 (m - 1) == pow2 (m - 1) * pow2 (m + 1) * (1 / pow2 (m + 1)) *)
    field. exact Hpm1_nz.
  }  
  apply Qle_of_Qeq. exact Hrewrite.
Qed.

(* 
Might be too weak
*)
      Definition trace_boolish_exists_k {n}
        (sq : Vector.t Q n) (e : GA_expr n) (d : Q) : Prop :=
        exists k : nat, trace_boolish_k_le sq e k d.
(*
Might be too weak
*)

(* Definition trace_boolish_poly {n}
  (sq : Vector.t Q n) (e : GA_expr n) (d : Q) : Prop :=
  exists k, (k <= poly n)%nat /\ trace_boolish_k_le sq e k d. *)

(* This trace_boolish_poly_size needs a size_expr, if we go by expr size *)
Fixpoint size_expr {n} (e : GA_expr n) : nat :=
  match e with
  | Basis _ | Scalar _ => 1
  | Cln_Grade.Add e1 e2 => 1 + size_expr e1 + size_expr e2
  | Mul e1 e2 => 1 + size_expr e1 + size_expr e2
  | Conv e1 e2 => 1 + size_expr e1 + size_expr e2
  end.

Definition trace_boolish_poly_size {n}
  (sq : Vector.t Q n) (e : GA_expr n) (d : Q) : Prop :=
  exists k, (k <= Nat.pow (size_expr e) 3)%nat /\ trace_boolish_k_le sq e k d.
(* Use any polynomial. ^3 is just a placeholder. *)

Definition trace_boolish_poly_n {n}
  (sq : Vector.t Q n) (e : GA_expr n) (d : Q) : Prop :=
  exists k, (k <= Nat.pow n 3)%nat /\ trace_boolish_k_le sq e k d.
(* This is bound by n (dimension) *)

Lemma trace_boolish_poly_size_implies_exists_k :
  forall n (sq : Vector.t Q n) (e : GA_expr n) d,
    trace_boolish_poly_size sq e d ->
    trace_boolish_exists_k sq e d.
Proof.
  intros n sq e d [k [_ Hk]].
  exists k. exact Hk.
Qed.

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

Lemma trace_boolish_poly_size_to_k_le :
  forall n (sq : Vector.t Q n) (e : GA_expr n) d,
    trace_boolish_poly_size sq e d ->
    trace_boolish_k_le sq e (Nat.pow (size_expr e) 3) d.
Proof.
  intros n sq e d [k [Hk Htr]].
  eapply trace_boolish_k_le_mono; eauto.
Qed.

(*
========================================================================
*)

Require Import Coq.Program.Equality.
Require Import Coq.QArith.Qabs.

Definition eval_bounded {n} (F : MV n) (C : Q) : Prop :=
  forall s : Corner n, Qabs (eval F s) <= C.

Definition eval_close_bool {n} (F : MV n) (d : Q) : Prop :=
  exists g : Corner n -> bool,
    forall s : Corner n, Qabs (eval F s - bQ (g s)) <= d.


Lemma Qabs_sQ_1 : forall h : Sign, Qabs (sQ h) == 1.
Proof. destruct h; simpl; reflexivity. Qed.

Lemma Qabs_chi'_1 :
  forall n (m : Mask n) (s : Corner n),
    Qabs (chi' m s) == 1.
Proof.
  induction n as [|n IH]; intros m s.
  - dependent destruction m. dependent destruction s. simpl. reflexivity.
  - dependent destruction m. dependent destruction s.
    cbn [chi' chi].
    destruct h; simpl.
    + (* mh = true *)
      change (Qabs ((sQ h0) * (chi m s)) == 1).
      rewrite Qabs_Qmult.
      rewrite Qabs_sQ_1.
      rewrite IH.
      ring.
    + (* mh = false *)
      (* factor is 1 *)
      change (Qabs (1 * (chi m s)) == 1).
      rewrite Qabs_Qmult.
      simpl. rewrite IH. ring.
Qed.

Lemma eval_abs_le_l1 :
  forall n (F : MV n) (s : Corner n),
    Qabs (eval F s) <= l1_norm F.
Proof.
  intros n F s.
  unfold eval, l1_norm.

  eapply Qle_trans.
  - exact (Qabs_sumQ_map_le (A := Mask n) (all_masks n)
            (fun m => (F m * chi' m s)%Q)).
  -
    apply Qle_of_Qeq.
    apply sumQ_map_ext; intros m _.
    rewrite Qabs_Qmult.
    rewrite (@Qabs_chi'_1 n m s).
    ring.
Qed.

Lemma bool_dist_le_implies_eval_close :
  forall n (F : MV n) (d : Q),
    bool_dist_le F d -> eval_close_bool F d.
Proof.
  intros n F d [g Hg].
  exists g.
  intro s.

  rewrite <- (@embed_correct n g s).
  assert (Hlin :
  (eval F s - eval (embed g) s)%Q
  ==
  eval (mv_sub F (embed g)) s).
  {
    unfold mv_sub.
    unfold eval.
    rewrite <- (@sumQ_map_sub
              (Mask n)
              (fun m => (F m * chi' m s)%Q)
              (fun m => (embed g m * chi' m s)%Q)
              (all_masks n)).
    f_equal.
    change (Qeq
    (sumQ (map (fun x : Mask n => (F x * chi' x s - embed g x * chi' x s)%Q) (all_masks n)))
    (sumQ (map (fun m : Mask n => ((F m - embed g m) * chi' m s)%Q) (all_masks n)))).
    apply sumQ_map_ext; intros m Hm.
    ring.
  }
  setoid_rewrite Hlin.
  eapply Qle_trans.
  - apply eval_abs_le_l1.
  - exact Hg.
Qed.

Require Import Coq.micromega.Lra.

Lemma Qabs_bQ_le_1 : forall b : bool, Qabs (bQ b) <= 1.
Proof.
  intro b; destruct b; simpl.
  - (* goal: 1 <= 1 *)
    apply Qle_refl.
  - (* goal: 0 <= 1 *)
    unfold Qle; simpl; lia.
Qed.

Lemma eval_close_bool_implies_eval_bounded_1pd :
  forall n (F : MV n) (d : Q),
    eval_close_bool F d ->
    eval_bounded F (1 + d).
Proof.
  intros n F d [g Hg] s.
  specialize (Hg s).

  (* triangle inequality: |(x) + (y)| <= |x| + |y| *)
  assert (Htri :
    Qabs ((eval F s - bQ (g s)) + bQ (g s))
      <= Qabs (eval F s - bQ (g s)) + Qabs (bQ (g s))).
  { exact (Qabs_triangle (eval F s - bQ (g s)) (bQ (g s))). }

  (* simplify the LHS: (eval - bQ) + bQ = eval *)
  setoid_replace (eval F s - bQ (g s) + bQ (g s)) with (eval F s) in Htri by ring.

  eapply Qle_trans.
  - exact Htri.
  - eapply Qle_trans.
    + (* pin the RHS to d + 1 so we don't get an evar ?t *)
      apply (Qplus_le_compat
               (Qabs (eval F s - bQ (g s))) d
               (Qabs (bQ (g s))) 1).
      * exact Hg.
      * apply Qabs_bQ_le_1.
    + (* d + 1 == 1 + d *)
      apply Qle_of_Qeq.
      ring.
Qed.

Lemma sumQ_mul_r :
  forall (A : Type) (f : A -> Q) (k : Q) (l : list A),
    (sumQ (List.map f l) * k)%Q == sumQ (List.map (fun x => (f x * k)%Q) l).
Proof.
  intros A f k l. induction l as [|a tl IH]; simpl.
  - ring.
  - rewrite <- IH. ring.
Qed.

Lemma coeff_via_eval :
  forall n (F : MV n) (m : Mask n),
    F m ==
    (1 / pow2 n) *
      sumQ (List.map (fun s => (eval F s * chi' m s)%Q) (all_corners n)).
Proof.
  intros n F m.
  unfold eval.

  (* Work on RHS via symmetry *)
  symmetry.

  (* Step 1: distribute χ(m,s) into inner sum
     (Σ_{m0} F(m0)·χ(m0,s)) · χ(m,s)  =  Σ_{m0} F(m0)·χ(m0,s)·χ(m,s) *)
  eapply Qeq_trans.
  { apply Qmult_comp; [reflexivity|].
    apply sumQ_map_ext; intros s _.
    apply sumQ_mul_r. }

  (* Step 2: reassociate each term
     F(m0)·χ(m0,s)·χ(m,s)  =  F(m0) · (χ(m0,s) · χ(m,s)) *)
  eapply Qeq_trans.
  { apply Qmult_comp; [reflexivity|].
    apply sumQ_map_ext; intros s _.
    apply sumQ_map_ext; intros m0 _.
    (* (F m0 * chi' m0 s) * chi' m s = F m0 * (chi' m0 s * chi' m s) *)
    rewrite Qmult_assoc.
    reflexivity. }

  (* Step 3: swap sums  Σ_s Σ_{m0} ... = Σ_{m0} Σ_s ... *)
  eapply Qeq_trans.
  { apply Qmult_comp; [reflexivity|].
    apply (@sumQ_swap (Corner n) (Mask n)
            (fun s m0 => (F m0 * (chi' m0 s * chi' m s))%Q)
            (all_corners n) (all_masks n)). }

  (* Step 4: factor out F(m0) *)
  eapply Qeq_trans.
  { apply Qmult_comp; [reflexivity|].
    apply sumQ_map_ext; intros m0 _.
    (* rewrite the inner function as c * h a, then apply scale lemma *)
    (* here c = F m0, h a = (chi' m0 a * chi' m a) *)
    exact (@sumQ_map_scale_l (Corner n)
         (F m0)
         (fun a => (chi' m0 a * chi' m a)%Q)
         (all_corners n)). }

  (* Step 5: apply corner-Walsh orthogonality
     Σ_s χ(m0,s)·χ(m,s) = if m0=m then pow2 n else 0 *)
  eapply Qeq_trans.
  { apply Qmult_comp; [reflexivity|].
    apply sumQ_map_ext; intros m0 _.
    apply Qmult_comp; [reflexivity|].
    apply corner_walsh_sum_ortho. }

  (* Step 6: push the "if" outside the product *)
  assert (Hinner :
    sumQ (map (fun m0 : Mask n =>
                 (F m0 * (if mask_eq_dec m0 m then pow2 n else 0))%Q)
              (all_masks n))
    ==
    sumQ (map (fun m0 : Mask n =>
                 if mask_eq_dec m0 m then (F m0 * pow2 n)%Q else 0)
              (all_masks n))).
  {
    apply sumQ_map_ext; intros m0 _.
    destruct (mask_eq_dec m0 m) as [Heq|Hneq].
    - subst m0. simpl. reflexivity.
    - simpl. rewrite Qmult_0_r. reflexivity.
  }
  setoid_rewrite Hinner.

  (* Step 7: collapse sum via pick lemma
     Σ_{m0} if m0=m then F(m0)·pow2 n else 0  ==  F(m)·pow2 n *)
  eapply Qeq_trans.
  { apply Qmult_comp; [reflexivity|].
    apply (@sumQ_all_masks_pick n (fun m0 => (F m0 * pow2 n)%Q) m). }

  (* Step 8: cancel  (1/pow2 n) · (F(m) · pow2 n) == F(m) *)
  (* pow2 n > 0 so division is valid *)
  unfold pow2. (* exposes inject_Z (2^n) *)
  field.
  (* remaining obligation: pow2 n ≠ 0, which follows from 2^n > 0 *)
  apply pow2_nonzero. (* you may need to prove this if you don't have it *)
Qed.

Lemma Qlt_0_1' : (0 < 1)%Q.
Proof. unfold Qlt; simpl; lia. Qed.

Lemma Qlt_0_2' : (0 < 2)%Q.
Proof. unfold Qlt; simpl; lia. Qed.

Lemma Qle_0_1' : (0 <= 1)%Q.
Proof. unfold Qle; simpl; lia. Qed.

Lemma pow2_Qpos : forall n, (0 < pow2 n)%Q.
Proof.
  induction n; simpl.
  - exact Qlt_0_1'.
  - apply Qmult_lt_0_compat.
    + exact Qlt_0_2'.
    + exact IHn.
Qed.

Lemma Qmult_le_compat_l' : forall c a b : Q,
  0 <= c ->
  a <= b ->
  c * a <= c * b.
Proof.
  intros c a b Hc Hab.
  (* use: x <= y  <->  0 <= y - x *)
  apply (Qle_minus_iff (c*a) (c*b)).
  (* goal becomes: 0 <= c*b - c*a *)
  (* factor the difference *)
  ring_simplify.
  assert (Hfact : (c * b + -1 * c * a == c * (b - a))%Q).
  { ring. }
  setoid_rewrite Hfact.
  (* now: 0 <= c * (b - a) *)
  apply Qmult_le_0_compat.
  - exact Hc.
  - (* 0 <= b - a *)
    apply (Qle_minus_iff a b). exact Hab.
Qed.

Lemma abs_coeff_le_avg_abs_eval :
  forall n (F : MV n) (m : Mask n),
    Qabs (F m) <=
      (1 / pow2 n) *
        sumQ (List.map (fun s => Qabs (eval F s)) (all_corners n)).
Proof.
  intros n F m.
  rewrite (@coeff_via_eval n F m).

  (* Step 1: |c · X| = |c| · |X| *)
  rewrite Qabs_Qmult.

  (* Step 2: |1/pow2 n| = 1/pow2 n since 1/pow2 n ≥ 0 *)
  assert (Hcpos : 0 <= 1 / pow2 n).
  {
    apply Qlt_le_weak.
    unfold Qdiv.
    apply Qmult_lt_0_compat.
    - exact Qlt_0_1'.
    - apply Qinv_lt_0_compat.
      apply pow2_Qpos.
  }
  rewrite (Qabs_pos _ Hcpos).

  (* Step 3: c · |X| ≤ c · Y, reduce to |X| ≤ Y by monotone mult *)
  apply (@Qmult_le_compat_l' (1 / pow2 n));
    [ exact Hcpos | ].

  (* Step 4: triangle inequality on the sum
     |Σ_s eval(F)(s) · χ(m,s)| ≤ Σ_s |eval(F)(s) · χ(m,s)| *)
  eapply Qle_trans.
  - exact (Qabs_sumQ_map_le (A := Corner n) (all_corners n)
             (fun s => (eval F s * chi' m s)%Q)).

  (* Step 5: pointwise |f(s) · χ(m,s)| = |f(s)| since |χ| = 1 *)
  - apply Qle_of_Qeq.
    apply sumQ_map_ext; intros s _.
    rewrite Qabs_Qmult.
    rewrite Qabs_chi'_1.
    ring.
Qed.

Lemma sumQ_const_all_masks :
  forall n (k : Q),
    sumQ (List.map (fun _ : Mask n => k) (all_masks n)) == pow2 n * k.
Proof.
  induction n as [|n IH]; intro k; simpl.
  - (* all_masks 0 = [[]],  pow2 0 = 1 *)
    ring.
  - (* all_masks (S n) = map (false::·) ++ map (true::·) *)
    rewrite map_app.
    rewrite sumQ_app.
    (* map (fun _ => k) (map (false::·) ms) has same length as ms *)
    rewrite List.map_map.
    rewrite List.map_map.
    (* both inner maps collapse to (fun _ => k) *)
    (* so each half equals sumQ (map (fun _ => k) (all_masks n)) == pow2 n * k *)
    setoid_rewrite IH.
    (* pow2 n * k + pow2 n * k == 2 * pow2 n * k *)
    ring.
Qed.

Lemma Qdiv_mul_cancel_l :
  forall (q r : Q),
    ~ q == 0 ->
    (1 / q) * (q * r) == r.
Proof.
  intros q r Hq.
  (* field works because goal is Qeq (==) *)
  field.
  exact Hq.
Qed.

Lemma l1_le_sum_abs_eval :
  forall n (F : MV n),
    l1_norm F <=
      sumQ (List.map (fun s => Qabs (eval F s)) (all_corners n)).
Proof.
  intros n F.
  unfold l1_norm.

  set (S := sumQ (List.map (fun s => Qabs (eval F s)) (all_corners n))).

  (* Step 1: pointwise bound |F(m)| ≤ (1/pow2 n) * S *)
  eapply Qle_trans.
  { apply sumQ_map_le.
    intros m _. apply (@abs_coeff_le_avg_abs_eval n F m). }

  (* Goal: sumQ (map (fun _ => (1/pow2 n) * S) (all_masks n)) <= S *)

  (* Step 2: factor out constant *)
  eapply Qle_trans.
  { apply Qle_of_Qeq.
    (* current goal is: sumQ (map (fun _ => (1/pow2 n) * BIG) ms) == ?y
       so we want to rewrite it to: (1/pow2 n) * sumQ (map (fun _ => BIG) ms) *)
    (* Use the scale lemma in the forward direction *)
    exact (@sumQ_map_scale_l (Mask n) (1 / pow2 n)
             (fun _ : Mask n =>
                sumQ (map (fun s : Corner n => Qabs (eval F s)) (all_corners n)))
             (all_masks n)).
  }
  
  (* Step 3: sum of constant over all masks = pow2 n * S *)
  eapply Qle_trans.
  { apply Qle_of_Qeq.
    apply Qmult_comp; [reflexivity|].
    apply sumQ_const_all_masks. }

  (* Step 4: cancel (1/pow2 n) * (pow2 n * S) == S *)
  apply Qle_of_Qeq.
  unfold S.
  apply (@Qdiv_mul_cancel_l
         (pow2 n)
         (sumQ (map (fun s : Corner n => Qabs (eval F s)) (all_corners n)))).
  apply pow2_nonzero.
Qed.

Lemma sumQ_const_all_corners :
  forall n (k : Q),
    sumQ (List.map (fun _ : Corner n => k) (all_corners n)) == pow2 n * k.
Proof.
  (* identical structure to sumQ_const_all_masks, but with all_corners *)
  induction n as [|n IH]; intro k; simpl.
  - ring.
  - rewrite map_app.
    rewrite sumQ_app.
    rewrite List.map_map.
    rewrite List.map_map.
    setoid_rewrite IH.
    ring.
Qed.

Lemma sum_abs_eval_le_pow2C :
  forall n (F : MV n) (C : Q),
    eval_bounded F C ->
    sumQ (List.map (fun s => Qabs (eval F s)) (all_corners n)) <= (pow2 n) * C.
Proof.
  intros n F C Hb.

  (* Step 1: pointwise -> sum inequality *)
  eapply Qle_trans.
  - apply sumQ_map_le.
    intros s _.  (* s ranges over all_corners n *)
    (* get the bound from Hb *)
    exact (Hb s).   (* or: specialize (Hb s); exact Hb *)
  - (* Step 2: collapse sum of constant C over all corners *)
    apply Qle_of_Qeq.
    (* sumQ (map (fun _ => C) (all_corners n)) == pow2 n * C *)
    rewrite (sumQ_const_all_corners n C).
    reflexivity.
Qed.

Theorem eval_bounded_implies_l1_bound :
  forall n (F : MV n) (C : Q),
    eval_bounded F C ->
    l1_norm F <= (pow2 n) * C.
Proof.
  intros n F C Hb.
  eapply Qle_trans.
  - apply l1_le_sum_abs_eval.
  - apply sum_abs_eval_le_pow2C; exact Hb.
Qed.

Lemma sum_abs_eval_le_pow2_l1 :
  forall n (F : MV n),
    sumQ (map (fun s => Qabs (eval F s)) (all_corners n))
    <= (pow2 n) * l1_norm F.
Proof.
  intros n F.

  (* Step 1: pointwise |eval(F)(s)| ≤ l1_norm F *)
  eapply Qle_trans.
  - apply sumQ_map_le.
    intros s _.
    exact (@eval_abs_le_l1 n F s).

  (* Step 2: sum of constant l1_norm F over all corners = pow2 n * l1_norm F *)
  - apply Qle_of_Qeq.
    apply sumQ_const_all_corners.
Qed.

Corollary eval_bounded_by_l1 :
  forall n (F : MV n),
    eval_bounded F (l1_norm F).
Proof.
  intros n F s.
  exact (eval_abs_le_l1 (n:=n) F s).
Qed.

Theorem l1_and_sum_abs_eval_equiv :
  forall n (F : MV n),
    l1_norm F <= sumQ (map (fun s => Qabs (eval F s)) (all_corners n))
    /\
    sumQ (map (fun s => Qabs (eval F s)) (all_corners n))
      <= pow2 n * l1_norm F.
Proof.
  intros n F.
  split.
  - apply l1_le_sum_abs_eval.
  - apply sum_abs_eval_le_pow2_l1.
Qed.

(*
Fully proved norm equivalence package:

    eval_abs_le_l1          :  |eval(F)(s)| ≤ ||F||₁

    eval_bounded_by_l1      :  eval_bounded F (||F||₁)

    l1_le_sum_abs_eval      :  ||F||₁ ≤ Σ_s |eval(F)(s)|

    sum_abs_eval_le_pow2_l1 :  Σ_s |eval(F)(s)| ≤ 2ⁿ · ||F||₁

    eval_bounded_implies_l1 :  eval_bounded F C → ||F||₁ ≤ 2ⁿ · C

    bool_dist → eval_close  :  ||F - embed(g)||₁ ≤ d → |eval(F)(s) - g(s)| ≤ d

    eval_close → eval_bound :  eval_close_bool F d → eval_bounded F (1+d)

========================================================================
*)

Lemma chi_xor_mul :
  forall n (A B : Mask n) (s : Corner n),
    chi' (mask_xor A B) s == (chi' A s * chi' B s)%Q.
Proof.
  induction n as [|n IH]; intros A B s.
  - dependent destruction A. dependent destruction B0.
    dependent destruction s. simpl. ring.
  - dependent destruction A. dependent destruction B0.
    dependent destruction s.
    unfold mask_xor. simpl.
    cbn [chi' chi].
    destruct h, h0, h1; simpl; rewrite IH; ring.
Qed.

Lemma if_dec_0_mul :
  forall (P : Prop) (d : {P}+{~P}) (a b : Q),
    (if d then a else 0) * b == if d then a * b else 0.
Proof. intros; destruct d; ring. Qed.

Lemma if_mask_eq_dec_subst :
  forall n (X U : Mask n) (f : Mask n -> Q) (b : Q),
    (if mask_eq_dec X U then f U else b)
    == (if mask_eq_dec X U then f X else b).
Proof. intros; destruct (mask_eq_dec X U); [subst; reflexivity | reflexivity]. Qed.

Lemma sumQ_all_masks_pick' :
  forall n (f : Mask n -> Q) (tgt : Mask n),
    sumQ (map (fun m => if mask_eq_dec m tgt then f m else 0) (all_masks n))
    == f tgt.
Proof.
  intros.
  enough (H : sumQ (map (fun m => if mask_eq_dec m tgt then f m else 0) (all_masks n))
          == sumQ (map (fun m => if mask_eq_dec m tgt then f tgt else 0) (all_masks n))).
    { eapply Qeq_trans. exact H. apply (@sumQ_all_masks_pick n (fun _ => f tgt) tgt). }
  apply sumQ_map_ext; intros m _.
  destruct (mask_eq_dec m tgt) as [Heq|Hneq].
  - subst. reflexivity.
  - reflexivity.
Qed.

Lemma eval_conv_pointwise :
  forall n (F G : MV n) (s : Corner n),
    eval (mv_conv F G) s == (eval F s * eval G s)%Q.
Proof.
  intros n F G s.
  unfold eval at 1. unfold mv_conv.

  (* Step 1: distribute χ(U,s) into inner sums *)
  eapply Qeq_trans.
  { apply sumQ_map_ext; intros U _. apply sumQ_mul_r. }
  eapply Qeq_trans.
  { apply sumQ_map_ext; intros U _.
    apply sumQ_map_ext; intros A _. apply sumQ_mul_r. }

  (* Step 2a: distribute χ(U,s) into the conditional *)
  eapply Qeq_trans.
  { apply sumQ_map_ext; intros U _.
    apply sumQ_map_ext; intros A _.
    apply sumQ_map_ext; intros B _.
    apply if_dec_0_mul. }

  (* Step 3: swap Σ_U and Σ_A *)
  eapply Qeq_trans.
  { apply sumQ_swap. }

  (* Step 4: swap Σ_U and Σ_B inside each A *)
  eapply Qeq_trans.
  { apply sumQ_map_ext; intros A _.
    apply sumQ_swap. }

  (* Step 5: flip mask_eq_dec to match pick lemma *)
  eapply Qeq_trans.
  { apply sumQ_map_ext; intros A _.
    apply sumQ_map_ext; intros B _.
    apply sumQ_map_ext; intros U _.
    apply if_mask_eq_dec_sym. }

  (* Step 6: collapse Σ_U via pick lemma *)
  eapply Qeq_trans.
  { apply sumQ_map_ext; intros A _.
    apply sumQ_map_ext; intros B _.
    apply (@sumQ_all_masks_pick' n
      (fun U => (F A * G B * chi' U s)%Q)
      (mask_xor A B)). }

  (* Step 7: apply chi_xor_mul, rearrange *)
  apply (Qeq_trans _
    (sumQ (map (fun A =>
      sumQ (map (fun B =>
        (F A * chi' A s) * (G B * chi' B s))%Q
        (all_masks n)))
      (all_masks n))) _).
  { apply sumQ_map_ext; intros A _.
    apply sumQ_map_ext; intros B _.
    rewrite chi_xor_mul. ring. }

  (* Step 8: factor inner sum *)
  apply (Qeq_trans _
    (sumQ (map (fun A =>
      (F A * chi' A s) * sumQ (map (fun B => G B * chi' B s) (all_masks n)))
      (all_masks n))) _).
  { apply sumQ_map_ext; intros A _.
    apply sumQ_map_scale_l. }

  (* Step 9: factor outer sum = product of two evals *)
  symmetry. unfold eval.
  apply sumQ_mul_r.
Qed.

Lemma eval_abs_le_bound :
  forall n (F:MV n) (C:Q) s, eval_bounded F C -> Qabs (eval F s) <= C.
Proof.
  intros n F C s Hb. exact (Hb s).
Qed.

Lemma eval_bounded_conv :
  forall n (F G : MV n) (CF CG : Q),
    eval_bounded F CF -> eval_bounded G CG ->
    eval_bounded (mv_conv F G) (CF * CG).
Proof.
  intros n F G CF CG HF HG s.
  rewrite eval_conv_pointwise.
  rewrite Qabs_Qmult.
  apply Qmult_le_compat_nonneg.
  - split; [apply Qabs_nonneg | exact (HF s)].
  - split; [apply Qabs_nonneg | exact (HG s)].
Qed.

Lemma eval_sub_pointwise :
  forall n (F G : MV n) (s : Corner n),
    eval (mv_sub F G) s == (eval F s - eval G s)%Q.
Proof.
  intros n F G s.
  unfold mv_sub, eval.
  rewrite <- (@sumQ_map_sub (Mask n)
    (fun m => (F m * chi' m s)%Q)
    (fun m => (G m * chi' m s)%Q)
    (all_masks n)).
  apply sumQ_map_ext; intros m _.
  ring.
Qed.

Lemma Qabs_triangle_sub : forall x y : Q,
  Qabs (x - y) <= Qabs x + Qabs y.
Proof.
  intros x y.
  setoid_replace (x - y) with (x + (-y)) by ring.
  eapply Qle_trans.
  - apply Qabs_triangle.
  - apply Qplus_le_compat.
    + apply Qle_refl.
    + rewrite Qabs_opp. apply Qle_refl.
Qed.

Lemma eval_bounded_sub :
  forall n (F G : MV n) (CF CG : Q),
    eval_bounded F CF ->
    eval_bounded G CG ->
    eval_bounded (mv_sub F G) (CF + CG).
Proof.
  intros n F G CF CG HF HG s.
  rewrite eval_sub_pointwise.
  eapply Qle_trans.
  - apply Qabs_triangle_sub.
  - apply Qplus_le_compat.
    + exact (HF s).
    + exact (HG s).
Qed.

Lemma eval_add_pointwise :
  forall n (F G : MV n) (s : Corner n),
    eval (mv_add F G) s == (eval F s + eval G s)%Q.
Proof.
  intros n F G s.
  unfold mv_add, eval.
  rewrite <- (@sumQ_map_add (Mask n)
    (fun m => (F m * chi' m s)%Q)
    (fun m => (G m * chi' m s)%Q)
    (all_masks n)).
  apply sumQ_map_ext; intros m _. ring.
Qed.

Lemma eval_bounded_add :
  forall n (F G : MV n) (CF CG : Q),
    eval_bounded F CF ->
    eval_bounded G CG ->
    eval_bounded (mv_add F G) (CF + CG).
Proof.
  intros n F G CF CG HF HG s.
  rewrite eval_add_pointwise.
  eapply Qle_trans.
  - (* |x+y| <= |x|+|y| *)
    apply Qabs_triangle.
  - apply Qplus_le_compat; [exact (HF s) | exact (HG s)].
Qed.

Lemma eval_scale_pointwise :
  forall n (c : Q) (F : MV n) (s : Corner n),
    eval (mv_scale c F) s == c * eval F s.
Proof.
  intros n c F s.
  unfold eval, mv_scale.
  eapply Qeq_trans.
  - apply sumQ_map_ext; intros m _.
    rewrite Qmult_assoc.
    rewrite (Qmult_comm c (F m)).
    rewrite <- Qmult_assoc.
    reflexivity.
  - apply Qeq_sym.
    symmetry.
    apply (@sumQ_map_scale_l (Mask n) c
             (fun m => (F m * chi' m s)%Q)
             (all_masks n)).
Qed.

Lemma eval_bounded_scale :
  forall n (c : Q) (F : MV n) (C : Q),
    eval_bounded F C ->
    eval_bounded (mv_scale c F) (Qabs c * C).
Proof.
  intros n c F C HF s.
  rewrite eval_scale_pointwise.
  rewrite Qabs_Qmult.
  apply Qmult_le_compat_nonneg.
  - split; [apply Qabs_nonneg | apply Qle_refl].
  - split; [apply Qabs_nonneg | exact (HF s)].
Qed.

Definition eval_close_bool' {n} (F : MV n) (d : Q) : Prop :=
  exists g : Corner n -> bool,
    forall s, Qabs (eval F s - bQ (g s)) <= d.

Lemma bQ_andb_mul : forall a b : bool,
  bQ (andb a b) == (bQ a * bQ b)%Q.
Proof.
  intros a b; destruct a, b; simpl; ring.
Qed.

Fixpoint corner_pos (n : nat) : Corner n :=
  match n with
  | O => Vector.nil Sign
  | S n' => Vector.cons Sign (Pos : Sign) n' (corner_pos n')
  end.

Lemma eval_close_bool_nonneg :
  forall n (F : MV n) d,
    eval_close_bool F d -> 0 <= d.
Proof.
  intros n F d [g Hg].
  pose (s0 := corner_pos n).
  specialize (Hg s0).
  eapply Qle_trans.
  - apply Qabs_nonneg.
  - exact Hg.
Qed.

Lemma Qabs_triangle3 : forall x y z : Q,
  Qabs (x + y + z) <= Qabs x + Qabs y + Qabs z.
Proof.
  intros x y z.
  (* rewrite x+y+z as x + (y+z) *)
  setoid_replace (x + y + z) with (x + (y + z)) by ring.
  eapply Qle_trans.
  - apply Qabs_triangle.
  - (* reassociate RHS so it's Qabs x + (Qabs y + Qabs z) *)
    setoid_replace (Qabs x + Qabs y + Qabs z)
      with (Qabs x + (Qabs y + Qabs z)) by ring.
    apply Qplus_le_compat_l.
    apply Qabs_triangle.
Qed.

Lemma eval_close_bool_conv :
  forall n (F G : MV n) dF dG,
    eval_close_bool F dF ->
    eval_close_bool G dG ->
    eval_close_bool (mv_conv F G) (dF + dG + dF*dG).
Proof.
  intros n F G dF dG HF HG.
  destruct HF as [g Hg].
  destruct HG as [h Hh].
  exists (fun s => andb (g s) (h s)).
  intro s.

  (* abbreviations *)
  set (a  := eval F s).
  set (b  := eval G s).
  set (ga := bQ (g s)).
  set (hb := bQ (h s)).
  set (e  := (a - ga)%Q).
  set (f  := (b - hb)%Q).

  (* Error bounds at this s *)
  assert (He : Qabs e <= dF).
  { unfold e, a, ga. exact (Hg s). }
  assert (Hf : Qabs f <= dG).
  { unfold f, b, hb. exact (Hh s). }
  
  (* Nonnegativity of dF and dG (use a fixed corner) *)
  assert (HdF0 : 0 <= dF).
  { pose (s0 := corner_pos n).
    specialize (Hg s0).
    eapply Qle_trans; [apply Qabs_nonneg | exact Hg]. }
  assert (HdG0 : 0 <= dG).
  { pose (s0 := corner_pos n).
    specialize (Hh s0).
    eapply Qle_trans; [apply Qabs_nonneg | exact Hh]. }

  (* switch to the AND target and use pointwise convolution *)
  rewrite eval_conv_pointwise.
  rewrite bQ_andb_mul.
  unfold a, b, ga, hb, e, f.

  (* Expand: (ga+e)(hb+f) - ga*hb = ga*f + hb*e + e*f *)
  setoid_replace (eval F s * eval G s - bQ (g s) * bQ (h s))%Q
    with (bQ (g s) * (eval G s - bQ (h s))
          + bQ (h s) * (eval F s - bQ (g s))
          + (eval F s - bQ (g s)) * (eval G s - bQ (h s)))%Q
    by ring.

  (* Triangle: |x+y+z| <= |x| + |y| + |z| *)
  eapply Qle_trans.
  - (* 3-term triangle in one shot *)
  apply Qabs_triangle3.
  - (* bound each term *)
    (* rename the three terms for readability *)
    set (t1 := (bQ (g s) * (eval G s - bQ (h s)))%Q).
    set (t2 := (bQ (h s) * (eval F s - bQ (g s)))%Q).
    set (t3 := ((eval F s - bQ (g s)) * (eval G s - bQ (h s)))%Q).

    (* now we need: |t1| + |t2| + |t3| <= dF + dG + dF*dG *)
    (* we’ll bound |t1|<=dG, |t2|<=dF, |t3|<=dF*dG *)

    eapply Qle_trans.
    + (* replace t1,t2,t3 and apply pointwise bounds *)
      apply Qplus_le_compat.
      * apply Qplus_le_compat.
        -- (* |t1| <= dG *)
           subst t1.
           rewrite Qabs_Qmult.
           eapply Qle_trans.
           ++ apply Qmult_le_compat_nonneg.
              ** split; [apply Qabs_nonneg | apply Qabs_bQ_le_1].
              ** split; [apply Qabs_nonneg | exact Hf].
           ++ (* 1 * dG = dG *)
              rewrite Qmult_1_l.
              apply Qle_refl.
        -- (* |t2| <= dF *)
           subst t2.
           rewrite Qabs_Qmult.
           eapply Qle_trans.
           ++ apply Qmult_le_compat_nonneg.
              ** split; [apply Qabs_nonneg | apply Qabs_bQ_le_1].
              ** split; [apply Qabs_nonneg | exact He].
           ++
              rewrite Qmult_1_l.
              apply Qle_refl.
      * (* |t3| <= dF*dG *)
        subst t3.
        rewrite Qabs_Qmult.
        (* |e| <= dF and |f| <= dG, with nonneg, so |e||f| <= dF*dG *)
        apply Qmult_le_compat_nonneg.
        -- split; [apply Qabs_nonneg | exact He].
        -- split; [apply Qabs_nonneg | exact Hf].
    + (* reorder dG + dF + dF*dG into dF + dG + dF*dG *)
      apply Qle_of_Qeq. ring.
Qed.

Definition constMV {n} (c : Q) : MV n :=
  mv_scale c (basis mask_empty).

Lemma eval_constMV :
  forall n (c : Q) (s : Corner n),
    eval (@constMV n c) s == c.
Proof.
  intros n c s.
  unfold constMV.
  rewrite eval_scale, eval_basis, chi_mask_empty.
  ring.
Qed.

Lemma eval_close_bool_not :
  forall n (F : MV n) d,
    eval_close_bool F d ->
    eval_close_bool (mv_sub (constMV 1) F) d.
Proof.
  intros n F d [g Hg].
  exists (fun s => negb (g s)).
  intro s.
  rewrite eval_sub_pointwise.
  rewrite eval_constMV.
  rewrite bQ_negb.
  (* goal: |1 - eval F s - (1 - bQ (g s))| <= d *)
  setoid_replace (1 - eval F s - (1 - bQ (g s)))
    with (-(eval F s - bQ (g s)))%Q by ring.
  rewrite Qabs_opp.
  exact (Hg s).
Qed.


Lemma bQ_orb_mul : forall a b : bool,
  bQ (orb a b) == (bQ a + bQ b - bQ a * bQ b)%Q.
Proof.
  intros a b; destruct a, b; simpl; ring.
Qed.

Lemma Qabs_1_minus_bQ_le_1 : forall b : bool, Qabs (1 - bQ b) <= 1.
Proof.
  intro b; destruct b; unfold bQ, Qminus, Qabs, Qle; simpl; lia.
Qed.

Lemma eval_close_bool_or :
  forall n (F G : MV n) dF dG,
    eval_close_bool F dF ->
    eval_close_bool G dG ->
    eval_close_bool
      (mv_sub (mv_add F G) (mv_conv F G))
      (dF + dG + dF * dG).
Proof.
  intros n F G dF dG HF HG.
  destruct HF as [g Hg].
  destruct HG as [h Hh].
  exists (fun s => orb (g s) (h s)).
  intro s.

  rewrite eval_sub_pointwise.
  rewrite eval_add_pointwise.
  rewrite eval_conv_pointwise.
  rewrite bQ_orb_mul.

  (* goal: |(f+g - f*g) - (bf+bh - bf*bh)| <= dF+dG+dF*dG *)
  (* rewrite as same decomposition used in conv proof *)
  set (a  := eval F s).
  set (b  := eval G s).
  set (ga := bQ (g s)).
  set (hb := bQ (h s)).
  set (e  := (a - ga)%Q).
  set (f  := (b - hb)%Q).

  assert (He : Qabs e <= dF) by exact (Hg s).
  assert (Hf : Qabs f <= dG) by exact (Hh s).

  assert (HdF0 : 0 <= dF).
  { eapply Qle_trans; [apply Qabs_nonneg | exact (Hg (corner_pos n))]. }
  assert (HdG0 : 0 <= dG).
  { eapply Qle_trans; [apply Qabs_nonneg | exact (Hh (corner_pos n))]. }

  (* Key: (a+b - a*b) - (ga+hb - ga*hb)
       = e + f - (ga*f + hb*e + e*f)
       = e*(1 - hb) + f*(1 - ga) - e*f *)
  setoid_replace (a + b - a * b - (ga + hb - ga * hb))
    with (e * (1 - hb) + f * (1 - ga) - e * f)%Q
    by (unfold e, f, a, b, ga, hb; ring).

  eapply Qle_trans.
  - apply Qabs_triangle_sub.
  - eapply Qle_trans.
    + apply Qplus_le_compat.
      * apply Qabs_triangle.
      * apply Qle_refl.
    + (* now bound each of three terms *)
      (* |e*(1-hb)| <= dF since |1-hb| <= 1 for hb in {0,1} *)
      (* |f*(1-ga)| <= dG since |1-ga| <= 1 *)
      (* |e*f| <= dF*dG *)

      assert (Hga1 : Qabs (1 - ga) <= 1).
      { unfold ga. apply Qabs_1_minus_bQ_le_1. }

      assert (Hhb1 : Qabs (1 - hb) <= 1).
      { unfold hb. apply Qabs_1_minus_bQ_le_1. }

      eapply Qle_trans.
      * apply Qplus_le_compat.
        -- apply Qplus_le_compat.
           ++ rewrite Qabs_Qmult.
              apply Qmult_le_compat_nonneg.
              ** split; [apply Qabs_nonneg | exact He].
              ** split; [apply Qabs_nonneg | exact Hhb1].
           ++ rewrite Qabs_Qmult.
              apply Qmult_le_compat_nonneg.
              ** split; [apply Qabs_nonneg | exact Hf].
              ** split; [apply Qabs_nonneg | exact Hga1].
        -- rewrite Qabs_Qmult.
           apply Qmult_le_compat_nonneg.
           ++ split; [apply Qabs_nonneg | exact He].
           ++ split; [apply Qabs_nonneg | exact Hf].
      * (* dF*1 + dG*1 + dF*dG = dF + dG + dF*dG *)
        apply Qle_of_Qeq. ring.
Qed.

(* A literal is a variable index plus a negation flag. *)
Definition Lit (n : nat) := (Fin.t n * bool)%type.
Definition Clause (n : nat) := list (Lit n).
Definition CNF (n : nat) := list (Clause n).

Section CNFCompiler.
  Context {n : nat}.
  
  Definition chiMV {n} (i : Fin.t n) : MV n :=
  basis (mask_single i).
  
  Definition varMV {n} (i : Fin.t n) : MV n :=
    mv_scale (1#2) (mv_add (constMV 1) (chiMV i)).
  
  (* Boolean connectives in MV-land *)
  Definition mv_not (F : MV n) : MV n :=
    mv_sub (constMV 1) F.

  Definition mv_and (F G : MV n) : MV n :=
    mv_conv F G.

  Definition mv_or (F G : MV n) : MV n :=
    mv_sub (mv_add F G) (mv_conv F G).

  (* Compile a literal: x_i or ¬x_i *)
  Definition compile_lit (l : Lit n) : MV n :=
    let '(i, neg) := l in
    if neg then mv_not (varMV i) else varMV i.

  (* Compile a clause = OR of literals.
     Identity for OR is FALSE = 0. *)
  Fixpoint compile_clause (c : Clause n) : MV n :=
    match c with
    | [] => constMV 0
    | l :: cs => mv_or (compile_lit l) (compile_clause cs)
    end.

  (* Compile a CNF = AND of clauses.
     Identity for AND is TRUE = 1. *)
  Fixpoint compile_cnf (phi : CNF n) : MV n :=
    match phi with
    | [] => constMV 1
    | c :: cs => mv_and (compile_clause c) (compile_cnf cs)
    end.
End CNFCompiler.

Lemma eval_chiMV :
  forall n (i : Fin.t n) (s : Corner n),
    eval (@chiMV n i) s == chi' (mask_single i) s.
Proof.
  intros n i s. unfold chiMV.
  rewrite eval_basis. reflexivity.
Qed.

Lemma chi_single_is_sQ'' :
  forall n (i : Fin.t n) (s : Corner n),
    chi' (mask_single i) s == sQ (Vector.nth s i).
Proof.
  (* This depends on your definitions of chi'/chi and mask_single.
     Usually proved by induction on i (Fin.t n) using Vector.nth. *)
Admitted.

Lemma chi_single_is_sQ :
  forall n (i : Fin.t n) (s : Corner n),
    chi' (mask_single i) s == sQ (Vector.nth s i).
Proof.
  intros n i s.
  revert s.
  induction i as [n'|n' j IH]; intro s.
  - (* i = F1, so n = S n' *)
    (* prove: chi' (const false) = 1 *)
    assert (Hconst :
      forall n (s : Corner n),
        chi' (Vector.const false n) s == 1%Q).
    { intro n0. induction n0 as [|n0 IHn0]; intro s0.
      - (* n0 = 0 *)
        dependent destruction s0. simpl. reflexivity.
      - (* n0 = S n0 *)
        dependent destruction s0.
        (* s0 = h :: s0 *)
        (* peel the leading false in the mask *)
        rewrite (@chi_false_cons n0 (Vector.const false n0) s0 h).
        apply IHn0.
    }

    dependent destruction s.
    (* s = h :: st *)
    (* mask_single F1 = true :: const false *)
    cbn [mask_single].
    rewrite (@chi_true_cons n' (Vector.const false n') s h).
    rewrite (Hconst n' s).
    (* Vector.nth (h::st) F1 = h *)
    simpl.
    ring.

  - (* i = FS j, so n = S n' *)
    dependent destruction s.
    (* mask_single (FS j) = false :: mask_single j *)
    cbn [mask_single].
    rewrite (@chi_false_cons n' (mask_single j) s h).
    rewrite (IH s).
    simpl.
    reflexivity.
Qed.

Lemma bQ_corner_bit :
  forall h : Sign,
    bQ (sign_to_bool h) == (1#2) * (1 + sQ h).
Proof.
  intro h; destruct h; unfold sign_to_bool; simpl.
  - (* Pos *)
    (* goal: bQ false == 1/2 * (1 - sQ Pos) *)
    (* bQ false = 0, sQ Pos = 1 *)
    unfold bQ, sQ; simpl.
    (* goal becomes: 0 == (1#2) * (1 - 1) *)
    field.
  - (* Neg *)
    (* bQ true = 1, sQ Neg = -1 *)
    unfold bQ, sQ; simpl.
    (* goal becomes: 1 == (1#2) * (1 - (-1)) *)
    field.
Qed.

Definition corner_bit {n} (i : Fin.t n) (s : Corner n) : bool :=
  sign_to_bool (Vector.nth s i).

Lemma bQ_sign_to_bool_plus :
  forall h : Sign,
    bQ (match h with Pos => true | Neg => false end)
    == (1#2) * (1 + sQ h).
Proof.
  intro h; destruct h; simpl.
  - (* Pos *) vm_compute. reflexivity.
  - (* Neg *) vm_compute. reflexivity.
Qed.

Lemma eval_varMV :
  forall n (i : Fin.t n) (s : Corner n),
    eval (@varMV n i) s == bQ (corner_bit i s).
Proof.
  intros n i s.
  unfold varMV.
  rewrite eval_scale.
  rewrite eval_add_pointwise.
  rewrite eval_constMV.
  rewrite eval_chiMV.
  rewrite chi_single_is_sQ.
  unfold corner_bit.
  (* reduce sign_to_bool by case on the sign *)
  destruct (Vector.nth s i); simpl; field; discriminate.
Qed.

Lemma eval_close_bool_varMV :
  forall n (i : Fin.t n),
    eval_close_bool (@varMV n i) 0.
Proof.
  intros n i.
  exists (fun s => corner_bit i s).
  intro s.
  rewrite eval_varMV.
  (* |bQ(bit)-bQ(bit)| = 0 *)
  setoid_replace (bQ (corner_bit i s) - bQ (corner_bit i s)) with 0%Q by ring.
  simpl. apply Qle_refl.
Qed.

Lemma eval_close_bool_const0 : forall n, eval_close_bool (@constMV n 0) 0.
Proof.
  intros n. exists (fun _ => false). intro s.
  rewrite eval_constMV. simpl.
  apply Qle_refl.
Qed.

Lemma eval_close_bool_const1 : forall n, eval_close_bool (@constMV n 1) 0.
Proof.
  intros n. exists (fun _ => true). intro s.
  rewrite eval_constMV. simpl.
  apply Qle_refl.
Qed.

Lemma compile_lit_eval_close :
  forall n (l : Lit n),
    eval_close_bool (compile_lit (n:=n) l) 0.
Proof.
  intros n l.
  destruct l as [i neg]. simpl.
  destruct neg.
  - (* negated literal *)
    apply eval_close_bool_not.
    apply eval_close_bool_varMV.
  - (* positive literal *)
    apply eval_close_bool_varMV.
Qed.

Lemma eval_close_bool_or_mv_or :
  forall n (F G : MV n) dF dG,
    eval_close_bool F dF ->
    eval_close_bool G dG ->
    eval_close_bool (mv_or F G) (dF + dG + dF*dG).
Proof.
  intros n F G dF dG HF HG.
  unfold mv_or.
  exact (@eval_close_bool_or n F G dF dG HF HG).
Qed.

Lemma compile_clause_eval_close :
  forall n (c : Clause n),
    eval_close_bool (compile_clause (n:=n) c) 0.
Proof.
  intros n c; induction c as [|l cs IH]; simpl.
  - apply eval_close_bool_const0.
  
  - (* OR preserves d=0 *)
    (* First get the bound (0+0+0*0) *)
    assert (H :
      eval_close_bool (mv_or (compile_lit l) (compile_clause cs)) (0 + 0 + 0*0)).
    { eapply (eval_close_bool_or_mv_or (n:=n)
                (F:=compile_lit l) (G:=compile_clause cs)
                (dF:=0) (dG:=0));
      [ apply compile_lit_eval_close | exact IH ]. }

    (* Then rewrite that bound to 0 *)
    (* 0 + 0 + 0*0 == 0 *)
    replace (0 + 0 + 0 * 0)%Q with 0%Q in H by (vm_compute; reflexivity).
    exact H.
Qed.

Lemma compile_cnf_eval_close :
  forall n (phi : CNF n),
    eval_close_bool (compile_cnf (n:=n) phi) 0.
Proof.
  intros n phi; induction phi as [|c cs IH]; simpl.
  - apply eval_close_bool_const1.
  -
    unfold mv_and.

    assert (H :
      eval_close_bool (mv_conv (compile_clause c) (compile_cnf cs))
                     (0 + 0 + 0 * 0)).
    { eapply (eval_close_bool_conv (n:=n)
              (F:=compile_clause c) (G:=compile_cnf cs)
              (dF:=0) (dG:=0)).
      - apply compile_clause_eval_close.
      - exact IH.
    }

    replace (0 + 0 + 0 * 0)%Q with 0%Q in H by (vm_compute; reflexivity).
    exact H.
Qed.

Corollary compile_cnf_eval_bounded :
  forall n (phi : CNF n),
    eval_bounded (compile_cnf (n:=n) phi) 1.
Proof.
  intros n phi.
  apply (eval_close_bool_implies_eval_bounded_1pd (n:=n) (F:=compile_cnf (n:=n) phi) (d:=0)).
  apply compile_cnf_eval_close.
Qed.

Corollary compile_cnf_l1_bound :
  forall n (phi : CNF n),
    l1_norm (compile_cnf (n:=n) phi) <= pow2 n.
Proof.
  intros n phi.
  (* first get the bound with *1 *)
  eapply Qle_trans.
  - apply (eval_bounded_implies_l1_bound
            (n:=n) (F:=compile_cnf (n:=n) phi) (C:=1)).
    apply compile_cnf_eval_bounded.
  - (* simplify pow2 n * 1 to pow2 n *)
    apply Qle_of_Qeq.
    rewrite Qmult_1_r.
    reflexivity.
Qed.

Lemma Qabs_le_0_eq :
  forall x : Q, Qabs x <= 0 -> x == 0.
Proof.
  intros x Hle.
  assert (H0 : Qabs x == 0).
  { apply Qle_antisym; [exact Hle | apply Qabs_nonneg]. }
  destruct (Qlt_le_dec x 0) as [Hxlt | Hxge].
  - (* x < 0 *)
    (* Qabs x = -x *)
    assert (Hxle : x <= 0) by (apply Qlt_le_weak; exact Hxlt).
    rewrite (Qabs_neg _ Hxle) in H0.
    (* -x = 0 -> x = 0 *)
    assert (H0' : (-1) * ((-1) * x) == (-1) * 0).
    { now rewrite H0. }
    ring_simplify in H0'.
    exact H0'.
  - (* 0 <= x *)
    rewrite (Qabs_pos _ Hxge) in H0.
    exact H0.
Qed.

Lemma eval_close_bool_0_implies_pointwise :
  forall n (F : MV n),
    eval_close_bool F 0 ->
    exists g : Corner n -> bool,
      forall s : Corner n, eval F s == bQ (g s).
Proof.
  intros n F [g Hg].
  exists g.
  intro s.
  specialize (Hg s).
  apply Qabs_le_0_eq in Hg.
  apply (Qplus_inj_r _ _ (bQ (g s))).

  setoid_replace (eval F s + bQ (g s))
    with ((eval F s - bQ (g s)) + 2 * bQ (g s)) by ring.
  rewrite Hg.
  ring.
Qed.

Corollary compile_cnf_correct :
  forall n (phi : CNF n),
    exists g : Corner n -> bool,
      forall s : Corner n,
        eval (compile_cnf (n:=n) phi) s == bQ (g s).
Proof.
  intros n phi.
  apply eval_close_bool_0_implies_pointwise.
  apply compile_cnf_eval_close.
Qed.

Lemma eval_pointwise_eq_implies_coeff_eq :
  forall n (F G : MV n),
    (forall s : Corner n, eval F s == eval G s) ->
    forall m : Mask n, F m == G m.
Proof.
  intros n F G Heq m.
  (* expand both sides using coeff_via_eval *)
  rewrite (@coeff_via_eval n F m).
  rewrite (@coeff_via_eval n G m).
  (* same prefactor; push Heq through the sum *)
  apply Qmult_comp; [reflexivity|].
  apply sumQ_map_ext; intros s _.
  rewrite (Heq s).
  reflexivity.
Qed.

Lemma eval_eq_bQ_implies_coeff_eq_embed :
  forall n (F : MV n) (f : Corner n -> bool),
    (forall s : Corner n, eval F s == bQ (f s)) ->
    forall m : Mask n, F m == embed f m.
Proof.
  intros n F f Hs m.
  eapply eval_pointwise_eq_implies_coeff_eq with (G := embed f).
  - intro s.
    rewrite embed_correct.
    exact (Hs s).
Qed.

Corollary compile_cnf_coeff_correct :
  forall n (phi : CNF n),
    exists g : Corner n -> bool,
      forall m : Mask n,
        (compile_cnf (n:=n) phi) m == embed g m.
Proof.
  intros n phi.
  destruct (@compile_cnf_correct n phi) as [g Hg].
  exists g.
  intro m.
  apply (@eval_eq_bQ_implies_coeff_eq_embed n (@compile_cnf n phi) g).
  exact Hg.
Qed.

Theorem cnf_exists_small_rep :
  forall n (phi : CNF n),
    exists F : MV n,
      (* F computes phi *)
      (exists g, forall s, eval F s == bQ (g s)) /\
      l1_norm F <= pow2 n.
Proof.
  intros n phi.
  exists (@compile_cnf n phi).
  split.
  - (* correctness in eval/bQ form *)
    destruct (@compile_cnf_correct n phi) as [g Hg].
    exists g. exact Hg.
  - (* l1 bound *)
    exact (@compile_cnf_l1_bound n phi).
Qed.

(* --- CNF semantics via BoolFormula --- *)
Definition bf_of_lit {n} (l : Lit n) : BoolFormula n :=
  let '(i, neg) := l in
  if neg then BNot (BVar i) else BVar i.

Fixpoint bf_of_clause {n} (c : Clause n) : BoolFormula n :=
  match c with
  | [] => BConst false
  | l :: cs => BOr (bf_of_lit l) (bf_of_clause cs)
  end.

Fixpoint bf_of_cnf {n} (phi : CNF n) : BoolFormula n :=
  match phi with
  | [] => BConst true
  | c :: cs => BAnd (bf_of_clause c) (bf_of_cnf cs)
  end.

Definition lit_sem {n} (l : Lit n) (s : Corner n) : bool :=
  let '(i, neg) := l in
  if neg then negb (corner_bit i s) else corner_bit i s.

Fixpoint clause_sem {n} (c : Clause n) (s : Corner n) : bool :=
  match c with
  | [] => false
  | l :: cs => orb (lit_sem l s) (clause_sem cs s)
  end.

Fixpoint cnf_sem {n} (phi : CNF n) (s : Corner n) : bool :=
  match phi with
  | [] => true
  | c :: cs => andb (clause_sem c s) (cnf_sem cs s)
  end.

Definition compile_cnf_expr {n} (phi : CNF n) : GA_expr n :=
  translate (bf_of_cnf phi).

Lemma eval_mv_not :
  forall n (F : MV n) (s : Corner n),
    eval (mv_not (n:=n) F) s == (1 - eval F s)%Q.
Proof.
  intros n F s.
  unfold mv_not.
  rewrite eval_sub_pointwise.
  rewrite eval_constMV.
  ring.
Qed.

Lemma eval_mv_or :
  forall n (F G : MV n) (s : Corner n),
    eval (mv_or (n:=n) F G) s
    == (eval F s + eval G s - eval (mv_conv F G) s)%Q.
Proof.
  intros n F G s.
  unfold mv_or.
  rewrite eval_sub_pointwise.
  rewrite eval_add_pointwise.
  rewrite eval_conv_pointwise.
  ring.
Qed.

Lemma eval_compile_lit :
  forall n (l : Lit n) (s : Corner n),
    eval (compile_lit (n:=n) l) s == bQ (lit_sem l s).
Proof.
  intros n [i neg] s; simpl.
  unfold lit_sem; simpl.
  destruct neg.
  - (* negated *)
    unfold compile_lit; simpl.
    unfold mv_not.
    rewrite eval_sub_pointwise.
    rewrite eval_constMV.
    rewrite eval_varMV.
    rewrite bQ_negb.
    ring.
  - (* positive *)
    unfold compile_lit; simpl.
    rewrite eval_varMV.
    reflexivity.
Qed.

Lemma eval_compile_clause :
  forall n (c : Clause n) (s : Corner n),
    eval (compile_clause (n:=n) c) s == bQ (clause_sem c s).
Proof.
  intros n c; induction c as [|l cs IH]; intro s; simpl.
  - rewrite eval_constMV. reflexivity.
  - rewrite eval_mv_or.
    rewrite eval_compile_lit.
    rewrite IH.
    rewrite eval_conv_pointwise.
    rewrite eval_compile_lit.
    rewrite IH.
    (* now use the bQ lemma for OR *)
    rewrite bQ_orb_mul.
    ring.
Qed.

Lemma compile_cnf_eval_correct :
  forall n (phi : CNF n) (s : Corner n),
    eval (compile_cnf (n:=n) phi) s == bQ (cnf_sem (n:=n) phi s).
Proof.
  intros n phi; induction phi as [|c cs IH]; intro s; simpl.
  - (* empty CNF = true *)
    rewrite eval_constMV. reflexivity.
  - (* AND via conv *)
    unfold mv_and.
    rewrite eval_conv_pointwise.
    rewrite eval_compile_clause.
    rewrite IH.
    rewrite bQ_andb_mul.
    ring.
Qed.

Theorem cnf_small_rep_correct :
  forall n (phi : CNF n),
    exists F : MV n,
      (forall s, eval F s == bQ (cnf_sem phi s)) /\
      l1_norm F <= pow2 n.
Proof.
  intros n phi.
  exists (compile_cnf (n:=n) phi).
  split.
  - intro s. apply compile_cnf_eval_correct.
  - apply compile_cnf_l1_bound.
Qed.

Lemma eval_bf_of_lit :
  forall n (l : Lit n) (s : Corner n),
    eval_bf (bf_of_lit (n:=n) l) s = lit_sem (n:=n) l s.
Proof.
  intros n [i neg] s; simpl.
  unfold lit_sem; simpl.
  destruct neg; reflexivity.
Qed.

Lemma eval_bf_of_clause :
  forall n (c : Clause n) (s : Corner n),
    eval_bf (bf_of_clause (n:=n) c) s = clause_sem (n:=n) c s.
Proof.
  intros n c; induction c as [|l cs IH]; intro s; simpl.
  - reflexivity.
  - rewrite eval_bf_of_lit.
    rewrite IH.
    reflexivity.
Qed.

Lemma eval_bf_of_cnf :
  forall n (phi : CNF n) (s : Corner n),
    eval_bf (bf_of_cnf (n:=n) phi) s = cnf_sem (n:=n) phi s.
Proof.
  intros n phi; induction phi as [|c cs IH]; intro s; simpl.
  - reflexivity.
  - rewrite eval_bf_of_clause.
    rewrite IH.
    reflexivity.
Qed.

Lemma embed_ext :
  forall n (f g : Corner n -> bool),
    (forall s, f s = g s) ->
    forall m, embed (n:=n) f m == embed (n:=n) g m.
Proof.
  intros n f g Hfg m.
  unfold embed.
  (* reduce to equality of the sums; then apply sumQ extensionality *)
  (* If embed is literally (1/pow2 n) * sumQ(...), we can just rewrite inside and reflexivity. *)
  f_equal.  (* often works if embed is definitional with "*"; if not, use ring below *)
  apply sumQ_map_ext.
  intros s _.
  rewrite Hfg.
  reflexivity.
Qed.

Theorem compile_cnf_expr_sound :
  forall n (phi : CNF n) (sq : Vector.t Q n),
    (forall i, Vector.nth sq i == 1) ->
    computes sq (compile_cnf_expr phi) (cnf_sem phi)
    /\
    (forall m, eval_expr sq (compile_cnf_expr phi) m == embed (cnf_sem phi) m).
Proof.
  intros n phi sq Hsq.
  split.
  - unfold computes, compile_cnf_expr.
    intro m.
    (* translate_correct gives embed(eval_bf (bf_of_cnf phi)) *)
    eapply Qeq_trans.
    + apply (@translate_correct n sq (bf_of_cnf phi) Hsq m).
    + (* convert eval_bf (bf_of_cnf phi) to cnf_sem phi *)
      apply embed_ext; intro s.
      apply eval_bf_of_cnf.
  - unfold compile_cnf_expr.
    intro m.
    eapply Qeq_trans.
    + apply (@translate_correct n sq (bf_of_cnf phi) Hsq m).
    + apply embed_ext; intro s.
      apply eval_bf_of_cnf.
Qed.

Theorem eval_expr_compile_cnf_expr_eq_compile_cnf :
  forall n (phi : CNF n) (sq : Vector.t Q n),
    (forall i, Vector.nth sq i == 1) ->
    forall m,
      eval_expr sq (compile_cnf_expr phi) m
      ==
      compile_cnf (n:=n) phi m.
Proof.
  intros n phi sq Hsq m.

  (* from expr compiler soundness *)
  destruct (@compile_cnf_expr_sound n phi sq Hsq)
    as [_ Hexpr].
  specialize (Hexpr m).
  (* Hexpr : eval_expr ... m == embed (cnf_sem phi) m *)

  (* from MV compiler eval correctness -> coeff correctness *)
  pose proof
    (@eval_eq_bQ_implies_coeff_eq_embed
       n
       (@compile_cnf n phi)
       (@cnf_sem n phi)
       (fun s => @compile_cnf_eval_correct n phi s)
       m)
    as Hcnf.
  (* Hcnf : compile_cnf phi m == embed (cnf_sem phi) m *)

  (* chain them *)
  eapply Qeq_trans; [exact Hexpr |].
  symmetry; exact Hcnf.
Qed.

Lemma translate_boolish_le_0 :
  forall n (sq : Vector.t Q n) (psi : BoolFormula n),
    (forall i, Vector.nth sq i == 1) ->
    boolish_le (eval_expr sq (translate psi)) 0.
Proof.
  intros n sq psi Hsq.
  unfold boolish_le.
  (* witness g := eval_bf psi *)
  exists (eval_bf psi).
  (* bool_dist_wrt = l1_norm (F - embed g) *)
  unfold bool_dist_wrt.
  (* show mv_sub is identically 0 by translate_correct *)
  assert (Hext :
    l1_norm (mv_sub (eval_expr sq (translate psi)) (embed (eval_bf psi)))
    ==
    l1_norm (@mv_zero n)).
  { apply l1_norm_ext; intro m.
    unfold mv_sub, mv_zero.
    rewrite (@translate_correct n sq psi Hsq m).
    ring.
  }
  rewrite Hext.
  (* l1_norm mv_zero = 0; you already proved this pattern earlier *)
  unfold l1_norm, mv_zero.
  eapply Qle_trans.
  - apply Qle_of_Qeq.
    eapply Qeq_trans.
    + apply sumQ_map_ext. intros m _. rewrite Qabs_pos; [reflexivity|apply Qle_refl].
    + apply sumQ_map_const0.
  - apply Qle_refl.
Qed.

Lemma trace_boolish_le_Basis :
  forall n (sq : Vector.t Q n) (i : Fin.t n) d,
    boolish_le (eval_expr sq (Basis i)) d ->
    trace_boolish_le sq (Basis i) d.
Proof. intros; simpl; assumption. Qed.

Lemma trace_boolish_le_Scalar :
  forall n (sq : Vector.t Q n) (c : Q) d,
    boolish_le (eval_expr sq (Scalar c)) d ->
    trace_boolish_le sq (Scalar c) d.
Proof. intros. simpl. assumption. Qed.

Lemma mv_scale_1_pointwise :
  forall n (F : MV n) (m : Mask n),
    mv_scale 1 F m == F m.
Proof.
  intros n F m.
  unfold mv_scale.
  ring.
Qed.

Lemma mv_sub_self_pointwise :
  forall n (F : MV n) (m : Mask n),
    mv_sub F F m == mv_zero m.
Proof.
  intros n F m.
  unfold mv_sub, mv_zero.
  ring.
Qed.

Lemma mv_sub_self_zero_pointwise :
  forall n (F : MV n) (m : Mask n),
    mv_sub F F m == 0.
Proof.
  intros n F m.
  unfold mv_sub.
  ring.
Qed.

Lemma l1_norm_zero :
  forall n, l1_norm (@mv_zero n) == 0.
Proof.
  intro n.
  unfold l1_norm, mv_zero.
  (* l1_norm mv_zero = sumQ (map (fun U => Qabs 0) (all_masks n)) *)
  eapply Qeq_trans.
  - (* rewrite Qabs 0 to 0 pointwise *)
    apply sumQ_map_ext.
    intros U HU.
    simpl. reflexivity.
  - (* sum of zeros is zero *)
    apply sumQ_map_const0.
Qed.

Lemma mv_sub_self :
  forall n (F : MV n),
    (forall m : Mask n, mv_sub F F m == mv_zero m).
Proof.
  intros n F m.
  apply mv_sub_self_pointwise.
Qed.

Lemma boolish_le_embed_0 :
  forall n (f : Corner n -> bool),
    boolish_le (embed f) 0.
Proof.
  intros n f.
  unfold boolish_le.
  exists f.
  eapply Qle_trans.
  -
    apply Qle_of_Qeq.
    eapply Qeq_trans.
    +
      apply l1_norm_ext.
      intro m.
      apply mv_sub_self_pointwise.
    + apply l1_norm_zero.
  - apply Qle_refl.
Qed.

Lemma boolish_k_le_of_eq :
  forall n (F G : MV n) k d,
    (forall m, F m == G m) ->
    boolish_k_le G k d ->
    boolish_k_le F k d.
Proof.
  intros n F G k d Heq Hb.
  unfold boolish_k_le in *.
  destruct Hb as [cs [gs [Hwf [Hlen Hdist]]]].
  exists cs, gs.
  repeat split; try assumption.

  eapply Qle_trans.
  - apply Qle_of_Qeq.
    refine (@l1_norm_ext n
              (mv_sub F (lincomb_embed cs gs))
              (mv_sub G (lincomb_embed cs gs)) _).
    intro m.
    unfold mv_sub.
    setoid_rewrite (Heq m).
    reflexivity.
  - exact Hdist.
Qed.

Lemma boolish_k_le_embed_1_0 :
  forall n (f : Corner n -> bool),
    boolish_k_le (embed f) 1 0.
Proof.
  intros n f.
  unfold boolish_k_le.
  exists (1%Q :: nil), (f :: nil).
  repeat split.
  - (* wf_lincomb *)
    unfold wf_lincomb; simpl; reflexivity.
  - (* length <= 1 *)
    simpl.
    eapply Qle_trans.
    + apply Qle_of_Qeq.
      refine (@l1_norm_ext n
                (mv_sub (embed f) (mv_scale 1 (embed f) ⊕ mv_zero))
                mv_zero _).
      intro m.
      unfold mv_sub, mv_add, mv_scale, mv_zero.
      (* now it’s pure Q arithmetic pointwise *)
      ring.
    + rewrite l1_norm_zero.
      apply Qle_refl.
Qed.

Definition bf_true {n} : Corner n -> bool := fun _ => true.

Definition bf_var {n} (i : Fin.t n) : Corner n -> bool :=
  fun s => match Vector.nth s i with Pos => true | Neg => false end.

Definition bf_notvar {n} (i : Fin.t n) : Corner n -> bool :=
  fun s => negb (bf_var i s).

Lemma lincomb_embed_scale :
  forall n (a : Q) (cs : list Q) (gs : list (Corner n -> bool)) (m : Mask n),
    wf_lincomb cs gs ->
    lincomb_embed (map (fun c => (a * c)%Q) cs) gs m
    ==
    mv_scale a (lincomb_embed cs gs) m.
Proof.
  intros n a cs.
  induction cs as [|c cs IH]; intros gs m Hwf.
  -
    destruct gs as [|g gs].
    +
      unfold lincomb_embed; simpl.
      unfold mv_scale, mv_zero; simpl.
      ring.
    +
      unfold wf_lincomb in Hwf; simpl in Hwf; discriminate.
  - (* cs = c :: cs *)
    destruct gs as [|g gs].
    + unfold wf_lincomb in Hwf; simpl in Hwf; discriminate.
    +
      unfold lincomb_embed; simpl.

      assert (Hwf' : wf_lincomb cs gs).
      { unfold wf_lincomb in *.
        simpl in Hwf.
        inversion Hwf.
        reflexivity. }

      specialize (IH gs m Hwf').
      unfold mv_add; simpl.
      setoid_rewrite IH.
      unfold mv_scale; simpl.

      change ((fix lincomb_embed (n0 : nat) (cs0 : list Q) (gs0 : list (Corner n0 -> bool)) {struct cs0} : MV n0 := _) n cs gs m)
        with (lincomb_embed cs gs m).

      ring.
Qed.

Lemma l1_norm_scale :
  forall n (a : Q) (F : MV n),
    ∥ mv_scale a F ∥₁ == (Qabs a * ∥F∥₁)%Q.
Proof.
  intros n a F.
  unfold l1_norm.
  unfold mv_scale.
  
  eapply Qeq_trans.
  - apply (@sumQ_map_ext
             (Mask n)
             (fun m => Qabs (a * F m))
             (fun m => Qabs a * Qabs (F m))
             (all_masks n)).
    intros m _.
    rewrite Qabs_Qmult.
    ring.
  - rewrite <- (@sumQ_map_mul_l
                  (Mask n)
                  (Qabs a)
                  (all_masks n)
                  (fun m => Qabs (F m))).
    reflexivity.
Qed.

Lemma boolish_k_le_scale_0 :
  forall n (a : Q) (F : MV n) k,
    boolish_k_le F k 0 ->
    boolish_k_le (mv_scale a F) k 0.
Proof.
  intros n a F k Hb.
  unfold boolish_k_le in *.
  destruct Hb as [cs [gs [Hwf [Hlen Hdist]]]].
  exists (map (fun c => (a * c)%Q) cs), gs.
  repeat split.
  - (* wf_lincomb *)
    unfold wf_lincomb in *.
    now rewrite length_map.
  - (* length gs <= k *)
    exact Hlen.
  - (* distance *)
    (* rewrite the norm to a scaled norm of the old residual *)
    eapply Qle_trans.
    + (* First: show the MV inside the norm is exactly mv_scale a (old residual) *)
      apply Qle_of_Qeq.
      apply (@l1_norm_ext n
               (mv_sub (mv_scale a F)
                       (lincomb_embed (map (fun c => (a * c)%Q) cs) gs))
               (mv_scale a (mv_sub F (lincomb_embed cs gs))) ).
      intro m.
      unfold mv_sub.
      (* expand mv_scale at coefficients *)
      unfold mv_scale.
      (* scale the lincomb back to mv_scale a (lincomb_embed cs gs) *)
      rewrite (@lincomb_embed_scale n a cs gs m Hwf).
      (* now it's pure Q arithmetic: a*F m - a*L m == a*(F m - L m) *)
      unfold mv_scale; simpl.
      ring.
    + (* Now apply l1_norm_scale and Hdist *)
      rewrite l1_norm_scale.
      (* Qabs a >= 0 *)
      assert (Ha : 0 <= Qabs a).
      { apply Qabs_nonneg. }
      (* multiply Hdist by Qabs a *)
      eapply Qle_trans.
      * apply Qmult_le_compat_l'; try exact Ha.
        exact Hdist.
      * (* Qabs a * 0 == 0 *)
        rewrite Qmult_0_r.  (* x * 0 = 0 *)
        apply Qle_refl.
Qed.

Lemma mv_one_eq_embed_true :
  forall n, forall m : Mask n, mv_one (n:=n) m == embed (n:=n) (@bf_true n) m.
Proof.
  intros n m.
  apply (@eval_pointwise_eq_implies_coeff_eq (n)
           (@mv_one (n)) (@embed (n) (@bf_true n))).
  intro s.
  rewrite embed_correct.
  rewrite eval_mv_one.
  reflexivity.
Qed.

Lemma boolish_k_le_scalar_1_0 :
  forall n (sq : Vector.t Q n) (c : Q),
    boolish_k_le (eval_expr sq (Scalar c)) 1 0.
Proof.
  intros n sq c.
  unfold boolish_k_le.
  exists (c :: nil), ((@bf_true n) :: nil).
  refine (conj _ (conj _ _)).
  - (* wf_lincomb *)
    unfold wf_lincomb; simpl; reflexivity.
  - (* length gs <= 1 *)
    simpl. apply le_n.
  - (* distance *)
    eapply Qle_trans.
    + apply Qle_of_Qeq.
      refine (@l1_norm_ext n
                (mv_sub (eval_expr sq (Scalar c)) (lincomb_embed (c :: nil) ((@bf_true n) :: nil)))
                mv_zero _).
      intro m.
      unfold mv_sub.
      unfold lincomb_embed; simpl.
      cbn [eval_expr].

      unfold mv_scale.
      setoid_rewrite (@mv_one_eq_embed_true n m).
      unfold mv_add, mv_zero; simpl.
      ring.
    + rewrite l1_norm_zero.
      apply Qle_refl.
Qed.

Lemma trace_boolish_k_le_root :
  forall n (sq : Vector.t Q n) (e : GA_expr n) k d,
    trace_boolish_k_le sq e k d ->
    boolish_k_le (eval_expr sq e) k d.
Proof.
  intros n sq e.
  induction e; intros k d Ht; simpl in *.
  - (* Scalar *)
    exact Ht.
  - (* Basis *)
    exact Ht.
  - (* Add *)
    destruct Ht as [Ht1 [Ht2 Hnode]].
    exact Hnode.
  - (* Mul / gp *)
    destruct Ht as [Ht1 [Ht2 Hnode]].
    exact Hnode.
  - (* Conv *)
    destruct Ht as [Ht1 [Ht2 Hnode]].
    exact Hnode.
Qed.

Lemma boolish_k_le_mv_scale_mv_one_1_0 :
  forall n (sq : Vector.t Q n) (c : Q),
    @boolish_k_le n (mv_scale c (@mv_one n)) 1 0.
Proof.
  intros n sq c.
  (* reuse scalar lemma + definitional equality of eval_expr *)
  eapply boolish_k_le_of_eq.
  - intro m. cbn [eval_expr]. reflexivity.
  - apply (@boolish_k_le_scalar_1_0 n sq c).
Qed.

Lemma bQ_var_minus_notvar :
  forall n (i : Fin.t n) (s : Corner n),
    bQ (bf_var i s) - bQ (bf_notvar i s) == sQ (Vector.nth s i).
Proof.
  intros n i s.
  unfold bf_var, bf_notvar.
  destruct (Vector.nth s i) eqn:Hnth; simpl.
  - (* Pos *)
    (* Goal is: 1 - bQ (negb (bf_var i s)) == 1 *)
    (* Re-expand bf_var so it exposes the match, then rewrite that match using Hnth *)
    unfold bf_var.
    rewrite Hnth.
    simpl.
    ring.
  - (* Neg *)
    unfold bf_var.
    rewrite Hnth.
    simpl.
    ring.
Qed.

Lemma basis_eq_embed_var_minus_notvar :
  forall n (i : Fin.t n) (m : Mask n),
    basis (n:=n) (mask_single i) m
    ==
    mv_sub (embed (n:=n) (bf_var i)) (embed (n:=n) (bf_notvar i)) m.
Proof.
  intros n i m.
  apply (@eval_pointwise_eq_implies_coeff_eq
           n
           (basis (n:=n) (mask_single i))
           (mv_sub (embed (n:=n) (bf_var i)) (embed (n:=n) (bf_notvar i)))).
  intro s.
  rewrite eval_basis.
  rewrite chi_mask_single.
  rewrite eval_sub_pointwise.
  rewrite embed_correct.
  rewrite embed_correct.
  apply Qeq_sym.
  apply bQ_var_minus_notvar.
Qed.

Lemma boolish_k_le_basis_2_0 :
  forall n (i : Fin.t n),
    boolish_k_le (basis (n:=n) (mask_single i)) 2 0.
Proof.
  intros n i.
  unfold boolish_k_le.
  exists (1%Q :: (-1)%Q :: nil), (bf_var i :: bf_notvar i :: nil).
  (* Now prove: wf_lincomb /\ length<=2 /\ norm<=0 *)
  refine (conj _ (conj _ _)).
  - (* wf_lincomb *)
    unfold wf_lincomb; simpl; reflexivity.
  - (* length gs <= 2 *)
    simpl; apply le_n.   (* 2 <= 2 *)
  - (* exact distance 0 *)
    eapply Qle_trans.
    + apply Qle_of_Qeq.
      refine (@l1_norm_ext n
                (mv_sub (basis (n:=n) (mask_single i))
                        (lincomb_embed (1%Q :: (-1)%Q :: nil)
                                      (bf_var i :: bf_notvar i :: nil)))
                mv_zero _).
      intro m.
      unfold mv_sub.
      unfold lincomb_embed; simpl.
      rewrite (@basis_eq_embed_var_minus_notvar n i m).
      unfold mv_sub, mv_add, mv_scale, mv_zero; simpl.
      ring.
    + rewrite l1_norm_zero.
      apply Qle_refl.
Qed.

Lemma mv_mul_scale_l :
  forall n (sq : Vector.t Q n) (c : Q) (B : MV n) (m : Mask n),
    @mv_gp n sq (mv_scale c mv_one) B m == mv_scale c B m.
Proof.
  intros n sq c B m.
  rewrite (@mv_gp_scale_l n sq c mv_one B m).
  unfold mv_scale; simpl.
  change ((mv_one ⋆ B) m) with (@mv_gp n sq mv_one B m).
  rewrite (@mv_gp_one_l n sq B m).
  reflexivity.
Qed.

Lemma mv_mul_one_l n (sq : Vector.t Q n) (B : MV n) m :
  @mv_gp n sq mv_one B m == B m.
Proof. apply mv_gp_one_l. Qed.

Lemma root_boolish_at n (sq : Vector.t Q n) e k k' :
  (k <= k')%nat ->
  trace_boolish_k_le sq e k 0 ->
  boolish_k_le (eval_expr sq e) k' 0.
Proof.
  intros Hle Ht.
  eapply boolish_k_le_mono; [exact Hle|].
  exact (@trace_boolish_k_le_root n sq e k 0 Ht).
Qed.

Lemma boolish_k_le_neg_gp_as_scale n (sq : Vector.t Q n) (F : MV n) k :
  boolish_k_le F k 0 ->
  boolish_k_le (@mv_gp n sq (mv_scale (-1) mv_one) F) k 0.
Proof.
  intro H.
  eapply boolish_k_le_of_eq.
  - intro m. apply mv_mul_scale_l.
  - (* make the scalar be a Q *)
    change (boolish_k_le (mv_scale (-1)%Q F) k 0).
    exact (@boolish_k_le_scale_0 n (-1)%Q F k H).
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

Lemma embed_conv_and :
  forall n (f g : Corner n -> bool) (m : Mask n),
    mv_conv (embed f) (embed g) m == embed (and_gen f g) m.
Proof.
  intros n f g m.
  apply (@eval_pointwise_eq_implies_coeff_eq n).
  intro s.
  rewrite eval_conv.
  rewrite embed_correct, embed_correct, embed_correct.
  symmetry. apply bQ_andb.
Qed.

Lemma sumQ_map_const0 {A} (l : list A) :
  sumQ (map (fun _ => 0%Q) l) == 0%Q.
Proof.
  induction l as [|x xs IH]; simpl.
  - reflexivity.
  - rewrite IH. ring.
Qed.

Lemma mv_conv_zero_l {n : nat} :
  forall (G : MV n) m, mv_conv mv_zero G m == 0.
Proof.
  intros G m.
  unfold mv_conv, mv_zero.

  set (L := all_masks n).

  (* First: show the inner sum is 0 for every A *)
  assert (Hinner : forall A : Mask n,
            sumQ
              (map (fun B : Mask n =>
                      if mask_eq_dec (mask_xor A B) m
                      then (0 * G B)%Q
                      else 0%Q) L) == 0%Q).
  {
    intro A.
    (* rewrite the mapped function to the constant 0 function *)
    transitivity (sumQ (map (fun _ : Mask n => 0%Q) L)).
    - apply sumQ_map_ext; intros B _.
      destruct (mask_eq_dec (mask_xor A B) m); reflexivity.
    - apply sumQ_map_const0.
  }

  (* Now rewrite the outer map using Hinner, making it a map of all zeros *)
  transitivity (sumQ (map (fun _ : Mask n => 0%Q) L)).
  - apply sumQ_map_ext; intros A _.
    exact (Hinner A).
  - apply sumQ_map_const0.
Qed.

Lemma sumQ_map_scale {A} (c : Q) (l : list A) (f : A -> Q) :
  sumQ (map (fun x => (c * f x)%Q) l) == (c * sumQ (map f l))%Q.
Proof.
  induction l as [|x xs IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

Lemma sumQ_map_ext_eq {A} (l : list A) (f g : A -> Q) :
  (forall x, In x l -> f x = g x) ->
  sumQ (map f l) = sumQ (map g l).
Proof.
  induction l as [|a l IH]; intro H; simpl.
  - reflexivity.
  - rewrite (H a (or_introl eq_refl)).
    rewrite IH.
    + reflexivity.
    + intros x Hx. apply H. right. exact Hx.
Qed.

Lemma mv_conv_scale_l :
  forall n (c : Q) (F G : MV n),
    forall m,
      mv_conv (mv_scale c F) G m
      == (mv_scale c (mv_conv F G)) m.
Proof.
  intros n c F G m.
  unfold mv_conv, mv_scale.
  set (L := all_masks n).

  (* inner: factor c out of the B-sum *)
  assert (Hinner : forall A : Mask n,
    sumQ (map (fun B : Mask n =>
      if mask_eq_dec (mask_xor A B) m
      then ((c * F A) * G B)%Q
      else 0%Q) L)
    ==
    (c * sumQ (map (fun B : Mask n =>
      if mask_eq_dec (mask_xor A B) m
      then (F A * G B)%Q
      else 0%Q) L))%Q).
  {
    intro A.
    transitivity
      (sumQ (map (fun B : Mask n =>
        (c * (if mask_eq_dec (mask_xor A B) m
              then (F A * G B)%Q else 0%Q))%Q) L)).
    - apply sumQ_map_ext; intros B _.
      destruct (mask_eq_dec (mask_xor A B) m); ring.
    - apply sumQ_map_scale.
  }

  (* outer: rewrite by Hinner, then factor c out of the A-sum *)
  transitivity
    (sumQ (map (fun A : Mask n =>
      (c * sumQ (map (fun B : Mask n =>
        if mask_eq_dec (mask_xor A B) m
        then (F A * G B)%Q else 0%Q) L))%Q) L)).
  - apply sumQ_map_ext; intros A _.
    apply Hinner.
  - apply sumQ_map_scale.
Qed.

Lemma mv_conv_scale_r :
  forall n (c : Q) (F G : MV n),
    forall m,
      mv_conv F (mv_scale c G) m
      == (mv_scale c (mv_conv F G)) m.
Proof.
  intros n c F G m.
  unfold mv_conv, mv_scale.
  set (L := all_masks n).

  assert (Hinner : forall A : Mask n,
    sumQ (map (fun B : Mask n =>
      if mask_eq_dec (mask_xor A B) m
      then (F A * (c * G B))%Q
      else 0%Q) L)
    ==
    (c * sumQ (map (fun B : Mask n =>
      if mask_eq_dec (mask_xor A B) m
      then (F A * G B)%Q
      else 0%Q) L))%Q).
  {
    intro A.
    transitivity
      (sumQ (map (fun B : Mask n =>
        (c * (if mask_eq_dec (mask_xor A B) m
              then (F A * G B)%Q else 0%Q))%Q) L)).
    - apply sumQ_map_ext; intros B _.
      destruct (mask_eq_dec (mask_xor A B) m); ring.
    - apply sumQ_map_scale.
  }

  transitivity
    (sumQ (map (fun A : Mask n =>
      (c * sumQ (map (fun B : Mask n =>
        if mask_eq_dec (mask_xor A B) m
        then (F A * G B)%Q else 0%Q) L))%Q) L)).
  - apply sumQ_map_ext; intros A _.
    apply Hinner.
  - apply sumQ_map_scale.
Qed.

Lemma mv_conv_embed_lincomb_r {n : nat} :
  forall (f : Corner n -> bool)
         (cs : list Q) (gs : list (Corner n -> bool)),
    wf_lincomb cs gs ->
    forall m,
      mv_conv (embed f) (lincomb_embed cs gs) m
      == lincomb_embed cs (map (and_gen f) gs) m.
Proof.
  intros f cs.
  induction cs as [|c cs IH]; intros gs Hwf m.
  - (* cs = [] *)
    destruct gs; [| unfold wf_lincomb in Hwf; simpl in Hwf; discriminate].
    simpl. unfold mv_conv, mv_zero.

    set (L := all_masks n).

    (* inner sum is 0 for each A *)
    assert (Hinner : forall A : Mask n,
      sumQ (map (fun B : Mask n =>
        if mask_eq_dec (mask_xor A B) m then (embed f A * 0)%Q else 0%Q) L) == 0%Q).
    {
      intro A.
      transitivity (sumQ (map (fun _ : Mask n => 0%Q) L)).
      - apply sumQ_map_ext; intros B _.
        destruct (mask_eq_dec (mask_xor A B) m); ring.
      - apply sumQ_map_const0.
    }

    (* outer sum is then also 0 *)
    transitivity (sumQ (map (fun _ : Mask n => 0%Q) L)).
    + apply sumQ_map_ext; intros A _.
      exact (Hinner A).
    + apply sumQ_map_const0.
  
  - destruct gs as [|g gs]; [unfold wf_lincomb in Hwf; simpl in Hwf; discriminate|].
    assert (Hwf' : wf_lincomb cs gs).
    { unfold wf_lincomb in *; simpl in *; lia. }
    simpl lincomb_embed at 1.
    (* LHS: mv_conv (embed f) (mv_scale c (embed g) ⊕ lincomb cs gs) *)
    rewrite mv_conv_add_r.
    
    unfold mv_add.
    rewrite (@mv_conv_scale_r n c (embed f) (embed g) m).
    (* now: mv_scale c (mv_conv (embed f) (embed g)) ⊕ mv_conv (embed f) (lincomb cs gs) *)
    unfold mv_add.
    rewrite (IH gs Hwf' m).
    
    simpl (lincomb_embed (c :: cs) (map (and_gen f) (g :: gs))) at 1.
    unfold mv_add.
    apply Qplus_comp.
    
    + (* show the scaled conv equals scaled embed *)
      unfold mv_scale.
      (* goal: c * mv_conv (embed f) (embed g) m == c * embed (and_gen f g) m *)
      apply Qmult_comp.
      * reflexivity.              (* c == c *)
      * exact (@embed_conv_and n f g m).
    + (* the tail terms match *)
      * reflexivity.
Qed.

Lemma lincomb_embed_conv {n : nat} :
  forall (cs1 cs2 : list Q)
         (gs1 gs2 : list (Corner n -> bool)),
    wf_lincomb cs1 gs1 ->
    wf_lincomb cs2 gs2 ->
    forall m,
      mv_conv (lincomb_embed cs1 gs1) (lincomb_embed cs2 gs2) m
      ==
      lincomb_embed (mul_coeffs cs1 cs2) (and_gens gs1 gs2) m.
Proof.
  intros cs1 cs2 gs1 gs2 Hwf1 Hwf2.
  revert gs1 Hwf1.
  induction cs1 as [|c cs1 IH]; intros gs1 Hwf1 m.
  -
    destruct gs1 as [|g gs1].
    +
      simpl. apply mv_conv_zero_l.
    +
      unfold wf_lincomb in Hwf1; simpl in Hwf1; discriminate.
  -
    destruct gs1 as [|g gs1].
    + unfold wf_lincomb in Hwf1; simpl in Hwf1; discriminate.
    + assert (Hwf1' : wf_lincomb cs1 gs1).
      { unfold wf_lincomb in *; simpl in *; lia. }
      simpl lincomb_embed at 1.
      eapply Qeq_trans.
      { apply mv_conv_add_l. }
      unfold mv_add.
      
      assert (Hfst :
        mv_conv (mv_scale c (embed g)) (lincomb_embed cs2 gs2) m
        == lincomb_embed (map (fun c2 => (c * c2)%Q) cs2) (map (and_gen g) gs2) m).
      {
        eapply Qeq_trans.
        - apply mv_conv_scale_l.
        - (* now: mv_scale c (mv_conv (embed g) (lincomb_embed cs2 gs2)) m == RHS *)

          (* wf for cs2 vs mapped gs2 *)
          assert (Hwf2' : wf_lincomb cs2 (map (and_gen g) gs2)).
          { unfold wf_lincomb in *.
            rewrite length_map.
            exact Hwf2. }

          (* choose the "middle" as mv_scale c (lincomb_embed cs2 (map ...) ) m *)
          eapply Qeq_trans
            with (y := mv_scale c (lincomb_embed cs2 (map (and_gen g) gs2)) m).

          + (* scale congruence: c * (conv ...) == c * (lincomb ...) *)
            unfold mv_scale.
            apply Qmult_comp; [reflexivity|].
            apply mv_conv_embed_lincomb_r.
            exact Hwf2.

          + (* middle == RHS, via lincomb_embed_scale (reversed) *)
            symmetry.
            exact (@lincomb_embed_scale n c cs2 (map (and_gen g) gs2) m Hwf2').
      }

      assert (Hsnd :
        mv_conv (lincomb_embed cs1 gs1) (lincomb_embed cs2 gs2) m
        == lincomb_embed (mul_coeffs cs1 cs2) (and_gens gs1 gs2) m).
      { apply IH. exact Hwf1'. }

      rewrite Hfst, Hsnd.
      rewrite mul_coeffs_cons, and_gens_cons.
      symmetry.
      eapply Qeq_trans.
      { apply lincomb_embed_app.
        - (* wf for map..cs2, map (and_gen g) gs2 *)
          unfold wf_lincomb. rewrite length_map, length_map.
          unfold wf_lincomb in Hwf2. exact Hwf2.
        - (* wf for mul_coeffs, and_gens *)
          apply wf_lincomb_mul_and; assumption.
      }
      unfold mv_add. reflexivity.
Qed.

Lemma mv_sub_cancel_qeq :
  forall n (G G0 : MV n) m,
    Qeq (mv_sub G (mv_sub G G0) m) (G0 m).
Proof.
  intros n G G0 m.
  unfold mv_sub, Qminus.
  ring.
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
    
    eapply Qle_trans.
    + apply Qle_of_Qeq.
      apply l1_norm_ext; intro m.
      unfold W, F0, G0.
      (* we need: mv_sub (mv_conv F G) W m == mv_sub (mv_conv F G) (mv_conv F0 G0) m *)
      unfold mv_sub.
      apply Qminus_comp; [reflexivity|].
      (* now show W m == mv_conv F0 G0 m *)
      symmetry.
      apply (lincomb_embed_conv (n:=n)); assumption.

    + (* Now use conv_error_split + triangle + submultiplicativity *)
      eapply Qle_trans.
      * (* split via conv_error_split pointwise, then l1_add_bound *)
        eapply Qle_trans.
        -- apply Qle_of_Qeq.
           apply l1_norm_ext; intro m.
           exact (@conv_error_split n F G F0 G0 m).
        -- eapply Qle_trans.
           ++ apply l1_add_bound.
           ++ apply Qplus_le_compat.
              ** (* first term *)
                 eapply Qle_trans.
                 --- apply l1_conv_submultiplicative.
                 --- (* <= ||F|| * d2 *)
                     apply Qmult_le_compat_l'.
                     { apply l1_norm_nonneg. }
                     exact HdG.
              
              ** (* second term *)
                 eapply Qle_trans.
                 --- (* submultiplicativity: ||conv (F-F0) G0|| <= ||F-F0|| * ||G0|| *)
                     apply l1_conv_submultiplicative.
                 --- (* now we are at the product; use HdF' *)
                     assert (HdF' : ∥ mv_sub F F0 ∥₁ <= d1).
                     { subst F0. exact HdF. }
                     eapply Qle_trans.
                     +++ (* multiply HdF' on the right by ||G0|| *)
                         apply Qmult_le_compat_r.
                         **** exact HdF'.
                         **** apply l1_norm_nonneg.   (* 0 <= ||G0|| *)
                     +++ (* clean up: d1 * ||G0|| is exactly what you want *)
                         apply Qle_refl.

      *
        assert (HG0_le : l1_norm G0 <= l1_norm G + d2).
        {
          subst G0.
          (* rewrite G0 as G - (G - G0) *)
          eapply Qle_trans.
          - (* rewrite the norm’s argument pointwise using mv_sub_cancel_qeq *)
            apply Qle_of_Qeq.
            apply l1_norm_ext; intro m.
            (* we want: mv_sub G (mv_sub G (lincomb_embed csG gsG)) m == lincomb_embed csG gsG m *)
            symmetry.
            apply (@mv_sub_cancel_qeq n G (lincomb_embed csG gsG) m).
          - (* now apply triangle bound: ||A|| <= ||G|| + ||G-A|| *)
            eapply Qle_trans.
            + apply l1_sub_bound.    (* ||G - (G - G0)|| <= ||G|| + ||G - G0|| *)
            + apply Qplus_le_compat.
              * apply Qle_refl.
              * exact HdG.
        }
        
        (* Use HG0_le to rewrite d1*||G0|| <= d1*(||G||+d2) *)
        (* and then expand to match goal *)
        (* Current bound from previous step is:
             ||F||*d2 + d1*||G0|| *)
        eapply Qle_trans.
        -- (* replace d1*||G0|| by d1*(||G||+d2) *)
           apply Qplus_le_compat.
           ++ apply Qle_refl.
           ++ apply Qmult_le_compat_l'.
              eapply Qle_trans.
                ** exact (@l1_norm_nonneg n (mv_sub F (lincomb_embed csF gsF))).
                ** exact HdF.
                
        (* Solve goal (1): ||G0|| <= ?b *)
        ** exact HG0_le.
        
        -- (* algebra *)
          ring_simplify.
          apply Qle_refl.
Qed.

Lemma boolish_k_le_d_mono :
  forall n (F : MV n) k d d',
    d <= d' ->
    boolish_k_le F k d ->
    boolish_k_le F k d'.
Proof.
  intros n F k d d' Hle [cs [gs [Hwf [Hlen Hdist]]]].
  exists cs, gs. repeat split; try assumption.
  eapply Qle_trans; [exact Hdist | exact Hle].
Qed.

Lemma boolish_k_le_conv_0 n (F G : MV n) k1 k2 :
  boolish_k_le F k1 0 ->
  boolish_k_le G k2 0 ->
  boolish_k_le (mv_conv F G) (k1 * k2) 0.
Proof.
  intros HF HG.
  specialize (@boolish_k_le_conv n F G k1 k2 0 0 HF HG) as H.

  eapply (boolish_k_le_d_mono
            (n:=n) (F:=mv_conv F G) (k:=k1*k2)
            (d := (0 * (∥ G ∥₁) + ∥ F ∥₁ * 0 + 0 * 0)%Q)
            (d' := 0%Q)).
  - (* prove: 0*||G|| + ||F||*0 + 0*0 <= 0 *)
    apply Qle_of_Qeq.
    ring.
  - exact H.
Qed.

Lemma translate_trace_boolish_exists_k_0 :
  forall n (sq : Vector.t Q n) (psi : BoolFormula n),
    (forall i, Vector.nth sq i == 1) ->
    trace_boolish_exists_k sq (translate psi) 0.
Proof.
  intros n sq psi Hsq.
  unfold trace_boolish_exists_k.
  revert sq Hsq.
  induction psi; intros sq Hsq; simpl.

  - (* Var t *)
    exists 3%nat.
    repeat split.

    + apply (@boolish_k_le_mono n
               (mv_scale (1#2) mv_one) 1%nat 3%nat 0).
      * lia.
      * apply (@boolish_k_le_mv_scale_mv_one_1_0 n sq (1#2)).

    + apply (@boolish_k_le_mono n
               (mv_scale 1 mv_one) 1%nat 3%nat 0).
      * lia.
      * apply (@boolish_k_le_mv_scale_mv_one_1_0 n sq 1).

    + apply (@boolish_k_le_mono n
               (basis (mask_single t)) 2%nat 3%nat 0).
      * lia.
      * apply (@boolish_k_le_basis_2_0 n t).

    + eapply (@boolish_k_le_add n
                (mv_scale 1 mv_one) (basis (mask_single t))
                1%nat 2%nat 0%Q 0%Q).
      * apply (@boolish_k_le_mv_scale_mv_one_1_0 n sq 1).
      * apply (@boolish_k_le_basis_2_0 n t).

    + eapply (@boolish_k_le_of_eq
            n
            (mv_scale (1#2) mv_one ⋆
               (mv_scale 1 mv_one ⊕ basis (mask_single t)))
            (mv_scale (1#2)
               (mv_scale 1 mv_one ⊕ basis (mask_single t)))
            3%nat
            0%Q).
      * intro m.
        rewrite (@mv_mul_scale_l
                  n sq
                  (1#2)
                  (mv_scale 1 mv_one ⊕ basis (mask_single t))
                  m).
        reflexivity.
      
      * apply (@boolish_k_le_scale_0
           n
           (1#2)
           (mv_scale 1 mv_one ⊕ basis (mask_single t))
           3%nat).

        apply (@boolish_k_le_add n (mv_scale 1 mv_one) (basis (mask_single t))
                  1%nat 2%nat 0 0).
        -- apply (@boolish_k_le_mv_scale_mv_one_1_0 n sq 1).
        -- apply (@boolish_k_le_basis_2_0 n t).


  - (* Const b *)
    exists 1%nat.
    destruct b; simpl; cbn [trace_boolish_k_le].
    + apply (@boolish_k_le_scalar_1_0 n sq 1).
    + apply (@boolish_k_le_scalar_1_0 n sq 0).

  - (* And psi1 psi2 : Conv *)
    (* FIX: witness must accommodate k1*k2 from conv, plus k1 and k2 for subtrees *)
    destruct (IHpsi1 sq Hsq) as [k1 Ht1].
    destruct (IHpsi2 sq Hsq) as [k2 Ht2].
    exists (k1 + k2 + k1 * k2)%nat.
    repeat split.

    + (* trace psi1 *)
      apply (@trace_boolish_k_le_mono n sq (translate psi1) k1 (k1 + k2 + k1 * k2)%nat 0).
      * lia.
      * exact Ht1.

    + (* trace psi2 *)
      apply (@trace_boolish_k_le_mono n sq (translate psi2) k2 (k1 + k2 + k1 * k2)%nat 0).
      * lia.
      * exact Ht2.

    + (* node boolish: conv at k1*k2, then mono up *)
      apply (@boolish_k_le_mono n
               (mv_conv (eval_expr sq (translate psi1))
                        (eval_expr sq (translate psi2)))
               (k1 * k2)%nat (k1 + k2 + k1 * k2)%nat 0).
      * lia.
      * apply boolish_k_le_conv_0.
        -- exact (@trace_boolish_k_le_root n sq (translate psi1) k1 0 Ht1).
        -- exact (@trace_boolish_k_le_root n sq (translate psi2) k2 0 Ht2).

  - (* Or psi1 psi2 *)
    (* FIX: witness must accommodate (k1+k2) for inner add and k1*k2 for conv *)
    destruct (IHpsi1 sq Hsq) as [k1 Ht1].
    destruct (IHpsi2 sq Hsq) as [k2 Ht2].
    set (k := S (k1 + k2 + k1 * k2)%nat).
    exists k.
    repeat split.

    + (* (trace psi1 /\ trace psi2 /\ boolish (A⊕B)) *)
      repeat split.
      * apply (@trace_boolish_k_le_mono n sq (translate psi1) k1 k 0).
        -- unfold k. lia.
        -- exact Ht1.
      + apply (@trace_boolish_k_le_mono n sq (translate psi2) k2 k 0).
        -- unfold k. lia.
        -- exact Ht2.
      + (* boolish (A ⊕ B): add at k1+k2, then mono to k *)
        apply (@boolish_k_le_mono n
                 (eval_expr sq (translate psi1) ⊕ eval_expr sq (translate psi2))
                 (k1 + k2)%nat k 0).
        -- unfold k. lia.
        -- apply (@boolish_k_le_add n
                    (eval_expr sq (translate psi1))
                    (eval_expr sq (translate psi2))
                    k1 k2 
                    0%Q 0%Q).
           ++ exact (@trace_boolish_k_le_root n sq (translate psi1) k1 0 Ht1).
           ++ exact (@trace_boolish_k_le_root n sq (translate psi2) k2 0 Ht2).

    + (* (-1)mv_one /\ (trace psi1 /\ trace psi2 /\ conv) /\ (-1)⋆conv *)
      repeat split.
      * (* boolish (-1*mv_one) *)
        apply (@boolish_k_le_mono n
                 (mv_scale (-1) mv_one) 1%nat k 0).
        -- unfold k. lia.
        -- apply (@boolish_k_le_mv_scale_mv_one_1_0 n sq (-1)).
    + (* trace psi1 *)
      apply (@trace_boolish_k_le_mono n sq (translate psi1) k1 k 0).
      -- unfold k. lia.
      -- exact Ht1.
    + (* trace psi2 *)
      apply (@trace_boolish_k_le_mono n sq (translate psi2) k2 k 0).
      -- unfold k. lia.
      -- exact Ht2.
    + (* boolish conv: at k1*k2, then mono to k *)
      apply (@boolish_k_le_mono n
               (mv_conv (eval_expr sq (translate psi1))
                        (eval_expr sq (translate psi2)))
               (k1 * k2)%nat k 0).
      -- unfold k. lia.
      -- apply boolish_k_le_conv_0.
         ++ exact (@trace_boolish_k_le_root n sq (translate psi1) k1 0 Ht1).
         ++ exact (@trace_boolish_k_le_root n sq (translate psi2) k2 0 Ht2).
         
    +
    (* inside the "* (* boolish (-1)⋆conv *)" branch *)
      set (C :=
        mv_conv (eval_expr sq (translate psi1))
                (eval_expr sq (translate psi2))).

      eapply (@boolish_k_le_of_eq
                n
                (mv_scale (-1) mv_one ⋆ C)   (* F: the thing you currently have *)
                (mv_scale (-1) C)            (* G: the thing scale_0 will give *)
                k
                0%Q).
      * intro m.
        rewrite (@mv_mul_scale_l
                  n sq (-1) C m).
        reflexivity.
      * (* now the goal is: boolish_k_le (mv_scale (-1) C) k 0 *)
        apply (@boolish_k_le_scale_0 n (-1) C k).
        apply (@boolish_k_le_mono n C (k1 * k2)%nat k 0).
        -- unfold k. lia.
        -- apply boolish_k_le_conv_0.
          ++ exact (@trace_boolish_k_le_root n sq (translate psi1) k1 0 Ht1).
          ++ exact (@trace_boolish_k_le_root n sq (translate psi2) k2 0 Ht2).

    
    
    
    + (* final boolish: (A⊕B) ⊕ (-1)⋆conv *)
      (* First: use add at (k1+k2) and (k1*k2), then mono up to k if needed *)
      apply (@boolish_k_le_mono
               n
               (eval_expr sq (translate psi1) ⊕ eval_expr sq (translate psi2)
                ⊕ mv_scale (-1) mv_one ⋆
                  mv_conv (eval_expr sq (translate psi1))
                          (eval_expr sq (translate psi2)))
               ((k1 + k2) + (k1 * k2))%nat
               k
               0).
      * unfold k. lia.
      * (* now prove it at ((k1+k2) + (k1*k2)) using boolish_k_le_add *)
        apply (@boolish_k_le_add
                 n
                 (eval_expr sq (translate psi1) ⊕ eval_expr sq (translate psi2))
                 (mv_scale (-1) mv_one ⋆
                   mv_conv (eval_expr sq (translate psi1))
                           (eval_expr sq (translate psi2)))
                 (k1 + k2)%nat
                 (k1 * k2)%nat
                 0
                 0).

        ++ (* boolish(A⊕B) at (k1+k2) *)
          apply (@boolish_k_le_add
                   n
                   (eval_expr sq (translate psi1))
                   (eval_expr sq (translate psi2))
                   k1
                   k2
                   0%Q
                   0%Q).
          -- exact (@trace_boolish_k_le_root n sq (translate psi1) k1 0 Ht1).
          -- exact (@trace_boolish_k_le_root n sq (translate psi2) k2 0 Ht2).

        ++ (* boolish((-1)⋆conv) at (k1*k2) *)
          set (C :=
            mv_conv (eval_expr sq (translate psi1))
                    (eval_expr sq (translate psi2))).

          eapply (@boolish_k_le_of_eq
                    n
                    (mv_scale (-1) mv_one ⋆ C)
                    (mv_scale (-1) C)
                    (k1 * k2)%nat
                    0%Q).
          -- intro m.
             rewrite (@mv_mul_scale_l n sq (-1) C m).
             reflexivity.
          -- apply (@boolish_k_le_scale_0 n (-1) C (k1 * k2)%nat).
             apply boolish_k_le_conv_0.
             ** exact (@trace_boolish_k_le_root n sq (translate psi1) k1 0 Ht1).
             ** exact (@trace_boolish_k_le_root n sq (translate psi2) k2 0 Ht2).

  - (* Not psi *)
    destruct (IHpsi sq Hsq) as [k1 Ht].
    exists (k1 + 1)%nat.
    repeat split.

    + (* boolish (1*mv_one) *)
      apply (@boolish_k_le_mono n (mv_scale 1 mv_one) 1%nat (k1 + 1)%nat 0).
      * lia.
      * apply (@boolish_k_le_mv_scale_mv_one_1_0 n sq 1).

    + (* inner: (-1)mv_one /\ trace /\ (-1)⋆A *)
      repeat split.
      * apply (@boolish_k_le_mono n (mv_scale (-1) mv_one) 1%nat (k1 + 1)%nat 0).
        -- lia.
        -- apply (@boolish_k_le_mv_scale_mv_one_1_0 n sq (-1)).
    + apply (@trace_boolish_k_le_mono n sq (translate psi) k1 (k1 + 1)%nat 0).
      * lia.
      * exact Ht.
    + set (A := eval_expr sq (translate psi)).
      eapply (@boolish_k_le_of_eq n (mv_scale (-1) mv_one ⋆ A) (mv_scale (-1) A)
                (k1 + 1)%nat 0%Q).
      * intro m. rewrite (@mv_mul_scale_l n sq (-1) A m). reflexivity.
      * apply (@boolish_k_le_mono n (mv_scale (-1) A) k1 (k1 + 1)%nat 0).
         -- lia.
         -- apply (@boolish_k_le_scale_0 n (-1) A k1).
            exact (@trace_boolish_k_le_root n sq (translate psi) k1 0 Ht).

    +
      set (A := eval_expr sq (translate psi)).
      replace (k1 + 1)%nat with (1 + k1)%nat by lia.
      apply (@boolish_k_le_add n (mv_scale 1 mv_one) (mv_scale (-1) mv_one ⋆ A)
                1%nat k1 0%Q 0%Q).
      * apply (@boolish_k_le_mv_scale_mv_one_1_0 n sq 1).
      * eapply (@boolish_k_le_of_eq n (mv_scale (-1) mv_one ⋆ A) (mv_scale (-1) A)
                  k1 0%Q).
        -- intro m. rewrite (@mv_mul_scale_l n sq (-1) A m). reflexivity.
        -- apply (@boolish_k_le_scale_0 n (-1) A k1).
           exact (@trace_boolish_k_le_root n sq (translate psi) k1 0 Ht).

Qed.

Theorem cnf_easy_in_booleanish_trace_tracepart :
  forall n (phi : CNF n),
  exists sq e,
    computes sq e (cnf_sem phi) /\
    trace_boolish_exists_k sq e 0.
Proof.
  intros n phi.
  exists (Vector.const 1 n), (compile_cnf_expr phi).
  split.
  - (* computes *)
    destruct (@compile_cnf_expr_sound n phi (Vector.const 1 n))
      as [Hc _].
    + intro i. rewrite VectorDef_nth_const. reflexivity.
    + exact Hc.
  - (* trace exists k *)
    unfold compile_cnf_expr.
    apply translate_trace_boolish_exists_k_0.
    intro i. rewrite VectorDef_nth_const. reflexivity.
Qed.

Theorem cnf_easy_in_booleanish_trace :
  forall n (phi : CNF n),
  exists sq e,
    computes sq e (cnf_sem phi) /\
    trace_boolish_exists_k sq e 0.
Proof.
  intros n phi.
  exists (Vector.const 1 n), (compile_cnf_expr phi).
  split.

  - destruct (compile_cnf_expr_sound phi (Vector.const 1 n)) as [Hc _].
    + intro i. rewrite VectorDef_nth_const. reflexivity.
    + exact Hc.

  - unfold compile_cnf_expr.
    apply translate_trace_boolish_exists_k_0.
    intro i. rewrite VectorDef_nth_const. reflexivity.
Qed.

Theorem IP_exponential_excursion_simple :
    (* This uses no boolish hypothesis, and it gives a clean exponential lower bound. *)
  forall m (sq : Vector.t Q (m+m)) (e : GA_expr (m+m)),
    (m >= 2)%nat ->
    computes sq e (@IP_n_func (m+m)) ->
    (pow2 (m - 1) <= exc_l1 (exc_of sq e))%Q.
Proof.
  intros m sq e Hm Hcomp.
  (* l1(eval) = l1(embed) because computes is pointwise equality *)
  assert (Hl1eq : l1_norm (eval_expr sq e) == l1_norm (embed (@IP_n_func (m+m)))).
  { apply l1_norm_ext; intro M. apply Hcomp. }

  (* l1(eval) ≤ max_l1_during = exc_l1(exc_of ...) *)
  eapply Qle_trans.
  - exact (@l1_norm_embed_IP_lower_bound m Hm).
  - (* rewrite via Hl1eq and then apply l1_eval_le_max_l1_during *)
    eapply Qle_trans.
    + apply Qle_of_Qeq. symmetry. exact Hl1eq.
    + unfold exc_of; simpl. (* exc_l1 is max_l1_during *)
      apply l1_eval_le_max_l1_during.
      
      
(* What IP_exponential_excursion_simple is really doing

  The core inequality it uses is:
    exc_l1 ≥ l1_norm(final_value) (max over intermediates is ≥ final)
  and computes lets you identify the final value with embed(IP)
  and l1_norm(embed(IP)) is exponential

  So the lower bound is coming from:
  IP has big ℓ₁ Walsh mass in your fixed basis,
  therefore any expression computing it must at some point hold an object with big ℓ₁ mass
  (in fact, the final object already has it).
  
  That is not the “intermediate blowup forced by constrained composition” story.
  It’s “the output itself is huge under this measure.”

  That’s still a legitimate lower bound inside your model, but it’s a different kind of lower bound.
  
  ---
  
  If your main separation uses IP_exponential_excursion_simple, then the honest story is:

  “In this model, some functions (like IP) have exponentially large representation mass (Fourier ℓ₁),
  and since the computation must output that object, excursion is forced to be exponential.”

  That is more like a monotone-complexity-style phenomenon
  (“the representation itself is big under this measure”)
  than a dynamical “trace constraint forces intermediate growth” phenomenon.

  It doesn’t make the program bogus, but it shifts the “why this might lift” question:

  Output-mass lower bounds are often model-/representation-dependent.

  Lifting them to general computation typically requires arguing the measure is
  robust/invariant under simulation or compilation into other representations;
  which you already suspected is hard (Morita invariance, basis changes, etc.).

  So the “open problem” paragraph as writen becomes more important,
  because the proof is now even more clearly tied to your chosen embedding/measure.
  
  “open problem” paragraph:
    The open problem is whether this structural, norm-based separation in the restricted algebraic model
    can be robustly lifted to general computation.
    Establishing such robustness would determine whether the framework remains a restricted-model phenomenon 
    or points toward a more fundamental separation.
  
  --- Furthermore: 
  
* `exc_l1 (exc_of sq e) = max_l1_during sq e`
* and `computes` is *pointwise equality of the final multivector with `embed f`*

So **any** lower bound that only uses

> `max_l1_during ≥ l1_norm(final)` and `final = embed f`

is, by construction, an **output-size argument**. It never needs to “look inside.”

That doesn’t mean the framework fails — it means your current invariant (`max_l1_during`) is too permissive to force internal reasoning by itself.

What you want is an invariant (or theorem shape) that *cannot* be discharged by “max ≥ final.”

Below are three ways to fix that, in increasing strength, and all are fully compatible with your existing `exc_of`/`computes`.

---

## 1) Define a “strict” excursion that factors out the final output

### Idea

Measure how much bigger an intermediate gets compared to the **final** object.

Two natural variants:

### (a) Additive strict excursion

```coq
Definition strict_exc_l1 {n} (sq : Vector.t Q n) (e : GA_expr n) : Q :=
  max_l1_during sq e - l1_norm (eval_expr sq e).
```

Now the trivial inequality becomes:

* `strict_exc_l1 ≥ 0`, but you **cannot** prove a positive lower bound from output mass alone.

To get `strict_exc_l1 ≥ 2^(Ω(m))`, you *must* use internal structure / trace constraints.

### (b) Multiplicative blowup ratio

```coq
Definition blowup_ratio_l1 {n} (sq : Vector.t Q n) (e : GA_expr n) : Q :=
  max_l1_during sq e / l1_norm (eval_expr sq e).
```

Again, output size alone gives you `≥ 1`. Anything stronger is inherently “inside-computation.”

### Why this helps your program

If your flagship theorem lower-bounds `strict_exc_l1` or `blowup_ratio_l1` **under booleanish trace**, you’re genuinely tracking internal dynamics.

---

## 2) Change the main theorem to a “peak-before-output” form

Right now, since `max_l1_during` includes the final step, output mass dominates.

So you can define the max *excluding the root* (i.e., all proper subexpressions). If you have an evaluator that can traverse subterms, define:

```coq
(* sketch: maximum l1 among all strict subexpressions *)
Parameter max_l1_strict_subexpr : forall {n}, Vector.t Q n -> GA_expr n -> Q.
```

Then prove a theorem like:

> If `e` computes IP and is booleanish-trace poly, then
> `max_l1_strict_subexpr sq e ≥ 2^(c*m)`.

This *forces* “looking inside,” because the final output is excluded by definition.

If you don’t currently have an accessor for “strict subexpressions,” your trace machinery likely already walks the syntax tree, so this is very feasible to define.

---

## 3) Use booleanish trace to bound *intermediate representational complexity*, not final mass

Given your `trace_boolish_poly_size` setup, the “inside” thing you wanted is something like:

> Every intermediate multivector is close to a combination of ≤ poly masks / or ≤ poly embeds / or has ≤ poly “active spectrum.”

That suggests the missing lemma you were circling:

* **Trace boolish ⇒ bounded intermediate “effective support” / bounded intermediate “coefficient budget.”**

Once you have that, you can prove:

* IP’s flat spectrum cannot be reached without either

  * huge intermediate coefficient mass (→ huge ℓ₁), or
  * huge support growth (→ huge trace/size), etc.

This is the true “internal dynamics” route.

But crucially: **it won’t show up if your conclusion is just `max_l1_during`**, because output already has big ℓ₁. You need one of the “strict” notions above.

---

# What this means for your current theorems

### `IP_exponential_excursion_simple`

With your current definitions, it’s unavoidably output-mass-driven:

* `computes` pins the final value to `embed IP`
* `max_l1_during` counts the final value
* so the proof can ignore the computation structure

So yes: your fear is exactly correct.

### Does the barrier-avoidance narrative still hold?

It becomes conditional:

* The narrative about “tracking internal dynamics” is not demonstrated by `IP_exponential_excursion_simple`.
* The narrative becomes *true* once your main theorem is about **strict excursion / pre-output peak / intermediate constraints**.

---

# Concrete “flagship theorem” shapes that force internal reasoning

Pick one:

### A) Strict excursion tradeoff

```coq
Theorem IP_booleanish_forces_strict_blowup :
  forall d : Q, exists c : nat,
  forall m (sq : Vector.t Q (m+m)) (e : GA_expr (m+m)),
    (m >= 2)%nat ->
    computes sq e (@IP_n_func (m+m)) ->
    trace_boolish_poly_size sq e d ->
    (Qpow2 (c * m) <= strict_exc_l1 sq e)%Q.
```

### B) Peak-before-output tradeoff

```coq
Theorem IP_booleanish_forces_prepeak :
  forall d : Q, exists c : nat,
  forall m sq e,
    computes sq e IP ->
    trace_boolish_poly_size sq e d ->
    Qpow2 (c*m) <= max_l1_strict_subexpr sq e.
```

### C) Ratio blowup

```coq
Theorem IP_booleanish_forces_ratio_blowup :
  forall d : Q, exists c : nat,
  forall m sq e,
    computes sq e IP ->
    trace_boolish_poly_size sq e d ->
    Qpow2 (c*m) <= blowup_ratio_l1 sq e.
```

All three *cannot* be proven by output-size alone. They require real internal structure lemmas.


  
  *)
Qed.

(* --------------------------------------------------------------------------------------------- *)
(* --------------------------------------------------------------------------------------------- *)
(* --------------------------------------------------------------------------------------------- *)


(* ---------------------------------------------------------------------------------------------
Lemma l1_norm_embed_IP_ge :
  forall m,
    (m >= 2)%nat ->
    (Qpow2 (m - 2) <= l1_norm (embed (@IP_n_func (m+m))))%Q.
Proof.
  intros m Hm2.
  assert (Hm : (m > 0)%nat) by lia.
  (* lower bound the sum by summing only over nonempty masks *)
  (* each nonempty term contributes exactly 1 / 2^(m+1) *)
  (* number of nonempty masks is 2^(2m) - 1 *)
Admitted.

Lemma l1_norm_embed_IP_ge_pow2 :
  forall m,
    (m >= 2)%nat ->
    (Qpow2 (m - 2) <= l1_norm (embed (@IP_n_func (m+m))))%Q.
Proof.
(*
  In hard_family_separates_div2,
    choose f n := IP_n_func n
         and c := 1,
         and use Nat.div2 (m+m) = m.
*)
Admitted.

Lemma computes_l1_eq_Qeq :
  forall n (sq : Vector.t Q n) (e : GA_expr n) (f : Corner n -> bool),
    computes sq e f ->
    l1_norm (eval_expr sq e) == l1_norm (embed f).
Proof.
  intros n sq e f Hcomp.
  apply l1_norm_ext.
  intro m. exact (Hcomp m).
Qed.

Theorem IP_exponential_in_booleanish_model :
  forall d : Q,
  exists c : nat,
    forall m (sq : Vector.t Q (m+m)) (e : GA_expr (m+m)),
      (m >= 2)%nat ->
      computes sq e (@IP_n_func (m+m)) ->
      trace_boolish_poly_size sq e d ->
      (Qpow2 (c * m) <= exc_l1 (exc_of sq e))%Q.
Proof.
  intro d.
  destruct (IP_exponential_in_booleanish_model_div2 d) as [c Hc].
  exists c.
  intros m sq e Hm Hcomp Htrace.

  (* apply the old theorem at n := m+m *)
  specialize (Hc (m+m) sq e Hcomp Htrace).

  (* rewrite div2 (m+m) = m *)
  (* option A: using Nat.div2_double with 2*m *)
  assert (Hdiv : Nat.div2 (m+m) = m).
  { (* turn m+m into 2*m *)
    rewrite <- Nat.mul_2_l.
    (* 2*m = m+m *)
    rewrite Nat.div2_double.
    reflexivity.
  }
  (* now rewrite the exponent and finish *)
  rewrite Hdiv in Hc.
  exact Hc.
Qed.

Theorem IP_exponential_in_booleanish_model :
  forall d : Q,
  exists c : nat,
    forall m (sq : Vector.t Q (m+m)) (e : GA_expr (m+m)),
      (m >= 2)%nat ->
      computes sq e (@IP_n_func (m+m)) ->
      trace_boolish_poly_size sq e d ->
      (Qpow2 (c * m) <= exc_l1 (exc_of sq e))%Q.
Proof.
Admitted.

Theorem IP_booleanish_tradeoff :
  forall d : Q,
  exists c : nat,
    forall m (sq : Vector.t Q (m+m)) (e : GA_expr (m+m)),
      (m >= 2)%nat ->
      computes sq e (@IP_n_func (m+m)) ->
      ( trace_boolish_poly_size sq e d ->
          Qpow2 (c * m) <= exc_l1 (exc_of sq e) )
      /\
      ( exc_l1 (exc_of sq e) < Qpow2 (c * m) ->
          ~ trace_boolish_poly_size sq e d ).
Proof.
  (* Second clause is contrapositive of first; 
     both follow from IP_exponential_in_booleanish_model *)
  intro d.
  destruct (IP_exponential_in_booleanish_model d) as [c Hc].
  exists c. intros m sq e Hm Hcomp. split.
  - intro Htrace. exact (Hc m sq e Hm Hcomp Htrace).
  - intros Hlt Htrace.
    apply Qlt_not_le in Hlt. apply Hlt.
    exact (Hc m sq e Hm Hcomp Htrace).
Qed.

Theorem IP_beats_CNF_in_booleanish_trace :
  forall d : Q,
  exists c, forall m (sq : Vector.t Q (m+m)) (e : GA_expr (m+m)),
    (m >= 2)%nat ->
    computes sq e (@IP_n_func (m+m)) ->
    trace_boolish_poly_size sq e d ->
    (Qpow2 (c * m) <= exc_l1 (exc_of sq e))%Q.
Proof.
  intro d.
  destruct (IP_exponential_in_booleanish_model d) as [c Hc].
  exists c.
  exact Hc.
Qed.


(*
========================================================================
*)


Theorem hard_family_beats_all_poly_compilers :
  forall (C : EasyCompiler) (k : nat),
  exists f : forall n, Corner n -> bool,
    forall n (sq : Vector.t Q n),
      ~ easy_under (B C) (d0 C) sq (f n)  (* or: not with poly_B k *)
      .
Proof.
Admitted.

Theorem hard_family_not_easy :
  forall (C : EasyCompiler),
  exists c : nat,
  exists f : forall n, Corner n -> bool,
    forall n,
      (* if f n were easy for compiler C, we'd contradict the hardness bound *)
      ~ easy C (n:=n) (f n).
Proof.
Admitted.

(*              This can be deleted. 

Theorem hard_family_separates_boolish_trace :
  forall d : Q,
  exists f : forall n, Corner n -> bool,
  exists c : nat,
    forall n (sq : Vector.t Q n) (e : GA_expr n),
      computes sq e (f n) ->
      trace_boolish_exists_k sq e d ->
      (Qpow2 (c * Nat.div2 n) <= exc_l1 (exc_of sq e))%Q.
Proof.
Admitted.

                Replace with a thin even-n corollary if you ever need the "for all even n" form

Corollary IP_exponential_even_n :
    forall d : Q,
    exists c : nat,
      forall n (sq : Vector.t Q n) (e : GA_expr n),
        Nat.Even n ->
        (Nat.div2 n >= 2)%nat ->
        computes sq e (@IP_n_func n) ->
        trace_boolish_poly_size sq e d ->
        (Qpow2 (c * Nat.div2 n) <= exc_l1 (exc_of sq e))%Q.
  Proof.
    intros d.
    destruct (IP_exponential_in_booleanish_model d) as [c Hc].
    exists c.
    intros n sq e Heven Hge Hcomp Hbool.
    destruct Heven as [m Hm]. subst n.
    (* now n = 2*m, and Nat.div2 (2*m) = m *)
    (* rewrite 2*m as m+m, apply Hc *)
    ...
  Qed.

*)


Lemma computes_l1_eq :
  forall n (sq : Vector.t Q n) (e : GA_expr n) (f : Corner n -> bool),
    computes sq e f ->
    l1_norm (eval_expr sq e) = l1_norm (embed f).
Proof.
Admitted.

Lemma embed_IP_abs_nonempty :
  forall m (M : Mask (m + m)),
    (m > 0)%nat ->
    M <> mask_empty ->
    Qabs (embed (@IP_n_func (m+m)) M)
    == (1 / pow2 (m+m) * (1#2) * inject_Z (Z.pow 2 (Z.of_nat m)))%Q.
Proof.
  intros m M Hm Hne.
  rewrite (embed_via_signed_walsh (m+m) (@IP_n_func (m+m)) M).
  destruct (mask_eq_dec M mask_empty) as [Heq|Hneq].
  - exfalso; apply Hne; exact Heq.
  - (* nonempty case: the “if” term vanishes *)
    simpl.
    (* becomes: Qabs ( (1/pow2 n) * (0 - (1/2)*signed_walsh) ) *)
    (* pull abs through products *)
    (* use signed_walsh_IP_magnitude to rewrite signed_walsh *)
    destruct (signed_walsh_IP_magnitude m M Hm) as [s Hs].
    admit.
    (* now it’s abs of (1/pow2 n) * (-(1/2) * (signed s * 2^m)) *)
    (* abs(signed s) = 1, abs(-x)=abs(x) *)
    (* finalize *)
Admitted.

Lemma l1_norm_ge_sum_over_subset :
  forall n (F : MV n),
    (* if for all nonempty masks abs(F M) >= a *)
    forall a,
      (forall M, M <> mask_empty -> a <= Qabs (F M)) ->
      (inject_Z (Z.of_nat (pred (length (all_masks n)))) * a
       <= l1_norm F)%Q.
Proof.
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

Theorem booleanish_or_exponential_excursion :
  forall d : Q,
  exists f : forall n, Corner n -> bool,
  exists c : nat,
    forall n (sq : Vector.t Q n) (e : GA_expr n),
      computes sq e (f n) ->
      ( trace_boolish_exists_k sq e d
        -> Qpow2 (c * Nat.div2 n) <= exc_l1 (exc_of sq e) ) /\
      ( ~ trace_boolish_exists_k sq e d
        -> (* optional: show there is some explicit non-booleanish witness *)
           True ).
Proof.
Admitted.

(* This would be ideal over the two options above.

      Exhibit an algorithm/circuit family that computes IP with
        subexponential excursion but violates booleanishness,
                  you get a true separation:


Theorem booleanish_vs_unrestricted_separation :
  exists f : forall n, Corner n -> bool,
    (exists e_easy : forall n, GA_expr n,
        (forall n sq, computes sq (e_easy n) (f n)) /\
        (forall n sq, exc_l1 (exc_of sq (e_easy n)) <= Qpoly n)) /\
    (forall d, exists c,
        forall n sq e,
          computes sq e (f n) ->
          trace_boolish_exists_k sq e d ->
          Qpow2 (c * Nat.div2 n) <= exc_l1 (exc_of sq e)).
Proof.
Admitted.


This is most compelling : "booleanishness costs you exponentially; if you drop it, you can do it cheaply."
  But it requires you to actually build the cheap non-booleanish circuit family.
*)

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

(*
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
*)

(*
===============================================================================




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



*)