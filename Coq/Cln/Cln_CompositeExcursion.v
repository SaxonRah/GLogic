(*
  ============================================================
  File: Cln_CompositeExcursion.v
  ============================================================
*)

Require Import Cln_Full.
Require Import Cln_Grade.
Require Import Cln_finite_l1_submultiplicativity.
Require Import Cln_BoolDist.

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

Fixpoint trace_boolish_k_le {n} (sq : Vector.t Q n) (e : GA_expr n) (k : nat) (d : Q) : Prop :=
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

Theorem hard_family_separates :
  forall d : Q,
  exists f : forall n, Corner n -> bool,
  exists c : nat,
    forall n (sq : Vector.t Q n) (e : GA_expr n),
      computes sq e (f n) ->
      trace_boolish_le sq e d ->
      Qpow2 (c * n) <= exc_l1 (exc_of sq e).
Proof.
Admitted.

