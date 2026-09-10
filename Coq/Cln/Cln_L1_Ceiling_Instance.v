(* ================================================================= *)
(*  Cln_L1_Ceiling_Instance.v                                        *)
(*                                                                   *)
(*  The Fourier-l1 lower-bound method, closed end to end.            *)
(*                                                                   *)
(*  Setting.  Boolean functions f on n = 2m variables, De Morgan +   *)
(*  XOR formulas (leaves: literals and the constant false; gates:    *)
(*  NOT, AND, OR, XOR).  The measure is                              *)
(*                                                                   *)
(*        mu f  :=  2 * l1( Walsh spectrum of (-1)^f )               *)
(*                                                                   *)
(*  i.e. twice the l1 norm of `embed_pm f` from Cln_IP_l1_exact.v.   *)
(*  The factor 2 makes AND exactly submultiplicative, so the         *)
(*  abstract theorems of Ceiling_Formula.v apply with C = 2.         *)
(*                                                                   *)
(*  Main results.                                                    *)
(*                                                                   *)
(*    l1_pm_le_pow2         l1(embed_pm f) <= 2^m    (every f)       *)
(*                          via Walsh inversion, Plancherel and      *)
(*                          Cauchy-Schwarz, all proved here          *)
(*    embed_pm_xor          XOR of functions = untwisted convolution *)
(*                          of spectra (NOT the Clifford product)    *)
(*    l1_method_sound       if 2^k < mu f, every formula computing f *)
(*                          (pointwise) has more than k leaves       *)
(*    l1_method_cap         if 2^k < mu f then k <= m                *)
(*    l1_method_cap_attained   IP certifies k = m                    *)
(*    l1_method_best_is_m   so the best bound the method certifies,  *)
(*                          over all functions, is exactly m = n/2   *)
(*                                                                   *)
(*  No admits.  No axioms beyond eq_rect_eq, which the library       *)
(*  already uses.  Built and checked with Coq 8.20.0.                *)
(* ================================================================= *)

Require Import Cln_Full.
Require Import Cln_Grade.
Require Import Cln_BoolDist.
Require Import Cln_SupportAlgebra.
Require Import Cln_finite_l1_submultiplicativity.
Require Import Cln_IP_l1_exact.
Require Ceiling_Formula.

From Coq Require Import List QArith Qabs Lia Lqa.
Require Import Coq.Program.Equality.
Import ListNotations.
Open Scope Q_scope.

(* ================================================================= *)
(* 1. A small toolkit for finite sums                                *)
(*    All proved by induction on the list, so no higher-order        *)
(*    rewriting under binders is ever needed.                        *)
(* ================================================================= *)

Lemma sq_nonneg : forall x : Q, 0 <= x * x.
Proof.
  intro x. destruct (Qlt_le_dec x 0) as [H|H].
  - setoid_replace (x * x) with ((-x) * (-x)) by ring.
    apply Qmult_le_0_compat; lra.
  - apply Qmult_le_0_compat; lra.
Qed.

Lemma sum_le :
  forall (A : Type) (f g : A -> Q) (l : list A),
    (forall x, In x l -> f x <= g x) ->
    sumQ (map f l) <= sumQ (map g l).
Proof.
  intros A f g l. induction l as [|a l IH]; intro H; cbn [map sumQ].
  - apply Qle_refl.
  - apply Qplus_le_compat.
    + apply H. left. reflexivity.
    + apply IH. intros x Hx. apply H. right. exact Hx.
Qed.

Lemma sum_nonneg :
  forall (A : Type) (f : A -> Q) (l : list A),
    (forall x, 0 <= f x) -> 0 <= sumQ (map f l).
Proof.
  intros A f l H. induction l as [|a l IH]; cbn [map sumQ].
  - apply Qle_refl.
  - specialize (H a). lra.
Qed.

Lemma sum_scale :
  forall (A : Type) (k : Q) (f : A -> Q) (l : list A),
    sumQ (map (fun x => k * f x) l) == k * sumQ (map f l).
Proof.
  intros A k f l. induction l as [|a l IH]; cbn [map sumQ]; [ring|]. rewrite IH. ring.
Qed.

Lemma sum_mul_r :
  forall (A : Type) (f : A -> Q) (c : Q) (l : list A),
    sumQ (map f l) * c == sumQ (map (fun x => f x * c) l).
Proof.
  intros A f c l. induction l as [|a l IH]; cbn [map sumQ]; [ring|]. rewrite <- IH. ring.
Qed.

Lemma sum_mul_sum :
  forall (A B : Type) (a : A -> Q) (b : B -> Q) (l : list A) (l' : list B),
    sumQ (map a l) * sumQ (map b l')
    == sumQ (map (fun i => sumQ (map (fun j => a i * b j) l')) l).
Proof.
  intros A B a b l l'. induction l as [|x l IH]; cbn [map sumQ]; [ring|].
  rewrite <- IH.
  assert (Hx : sumQ (map (fun j => a x * b j) l') == a x * sumQ (map b l')).
  { clear IH. induction l' as [|y l' IH']; cbn [map sumQ]; [ring|]. rewrite IH'. ring. }
  rewrite Hx. ring.
Qed.

(* Fubini for finite sums: reuse the library's sumQ_swap. *)
Lemma sum_swap :
  forall (A B : Type) (h : A -> B -> Q) (la : list A) (lb : list B),
    sumQ (map (fun a => sumQ (map (fun b => h a b) lb)) la)
    == sumQ (map (fun b => sumQ (map (fun a => h a b) la)) lb).
Proof. intros. apply sumQ_swap. Qed.

(* ================================================================= *)
(* 2. Cauchy-Schwarz for finite sums                                 *)
(*    (sum y)^2 <= (sum 1) * (sum y^2), from 2 y_i y_j <= y_i^2+y_j^2 *)
(* ================================================================= *)

Lemma sum_c_plus :
  forall (A : Type) (c : Q) (d : A -> Q) (l : list A),
    sumQ (map (fun j => c + d j) l)
    == c * sumQ (map (fun _ => 1) l) + sumQ (map d l).
Proof.
  intros A c d l. induction l as [|a l IH]; cbn [map sumQ]; [ring|]. rewrite IH. ring.
Qed.

Lemma sum_affine :
  forall (A : Type) (c1 c2 : Q) (z : A -> Q) (l : list A),
    sumQ (map (fun i => c1 * z i + c2) l)
    == c1 * sumQ (map z l) + c2 * sumQ (map (fun _ => 1) l).
Proof.
  intros A c1 c2 z l. induction l as [|a l IH]; cbn [map sumQ]; [ring|].
  rewrite IH. ring.
Qed.

Theorem cauchy_schwarz :
  forall (A : Type) (y : A -> Q) (l : list A),
    sumQ (map y l) * sumQ (map y l)
    <= sumQ (map (fun _ => 1) l) * sumQ (map (fun i => y i * y i) l).
Proof.
  intros A y l.
  rewrite sum_mul_sum.
  (* pointwise:  y_i y_j <= (y_i^2 + y_j^2) / 2 *)
  apply Qle_trans with
    (y := sumQ (map (fun i => sumQ (map (fun j =>
            (1#2) * (y i * y i) + (1#2) * (y j * y j)) l)) l)).
  - apply sum_le. intros i _. apply sum_le. intros j _.
    pose proof (sq_nonneg (y i - y j)). lra.
  - apply Qle_of_Qeq.
    (* inner sum over j *)
    transitivity (sumQ (map (fun i =>
        ((1#2) * sumQ (map (fun _ => 1) l)) * (y i * y i)
        + (1#2) * sumQ (map (fun j => y j * y j) l)) l)).
    { apply sumQ_map_ext. intros i _.
      rewrite sum_c_plus, (sum_scale _ (1#2) (fun j => y j * y j)). ring. }
    (* outer sum over i *)
    rewrite (sum_affine _ ((1#2) * sumQ (map (fun _ => 1) l))
               ((1#2) * sumQ (map (fun j => y j * y j) l)) (fun i => y i * y i)).
    ring.
Qed.

(* ================================================================= *)
(* 3. Characters: |chi| = 1, and orthogonality over corners          *)
(* ================================================================= *)

Lemma Qabs_chi : forall n (M : Mask n) (s : Corner n), Qabs (chi' M s) == 1.
Proof.
  unfold chi'. induction n as [|n IH]; intros M s; cbn [chi].
  - reflexivity.
  - rewrite Qabs_Qmult, IH.
    destruct (Vector.hd M); [destruct (Vector.hd s)|]; reflexivity.
Qed.

Lemma mask_xor_eq_empty : forall n (A B : Mask n), mask_xor A B = mask_empty -> A = B.
Proof.
  intros n A B H.
  assert (E : mask_xor (mask_xor A B) B = A).
  { rewrite mask_xor_assoc, mask_xor_self, mask_xor_empty_r. reflexivity. }
  rewrite H, mask_xor_empty_l in E. symmetry. exact E.
Qed.

Lemma chi_orth :
  forall n (A B : Mask n),
    sumQ (map (fun s => chi' A s * chi' B s) (all_corners n))
    == if mask_eq_dec A B then pow2 n else 0.
Proof.
  intros n A B.
  transitivity (sumQ (map (fun s => chi' (mask_xor A B) s) (all_corners n))).
  { apply sumQ_map_ext. intros s _. apply chi_mul. }
  destruct (mask_eq_dec A B) as [HAB | HAB].
  - subst B. rewrite mask_xor_self. apply chi_corner_sum_empty.
  - apply chi_corner_sum_nonempty. intro H. apply HAB. apply mask_xor_eq_empty. exact H.
Qed.

Lemma sum_corners_one : forall n, sumQ (map (fun _ : Corner n => 1) (all_corners n)) == pow2 n.
Proof.
  intro n. rewrite <- (chi_corner_sum_empty n).
  apply sumQ_map_ext. intros s _. symmetry. apply chi_mask_empty.
Qed.

(* ================================================================= *)
(* 4. Walsh inversion and Plancherel                                 *)
(* ================================================================= *)

(* Every multivector is recovered from its values at the corners. *)
Theorem walsh_inversion :
  forall n (F : MV n) (M : Mask n),
    F M == (1 / pow2 n) * sumQ (map (fun s => eval F s * chi' M s) (all_corners n)).
Proof.
  intros n F M.
  assert (Hsum :
    sumQ (map (fun s => eval F s * chi' M s) (all_corners n)) == F M * pow2 n).
  { unfold eval.
    transitivity (sumQ (map (fun s => sumQ (map (fun A => F A * chi' A s * chi' M s)
                                             (all_masks n))) (all_corners n))).
    { apply sumQ_map_ext. intros s _. rewrite sum_mul_r. reflexivity. }
    rewrite (sum_swap _ _ (fun s A => F A * chi' A s * chi' M s)).
    transitivity (sumQ (map (fun A => if mask_eq_dec A M then F A * pow2 n else 0)
                           (all_masks n))).
    { apply sumQ_map_ext. intros A _.
      transitivity (F A * sumQ (map (fun s => chi' A s * chi' M s) (all_corners n))).
      { rewrite <- sum_scale. apply sumQ_map_ext. intros s _. ring. }
      rewrite chi_orth. destruct (mask_eq_dec A M); ring. }
    apply (sumQ_all_masks_pick (fun A => F A * pow2 n) M). }
  rewrite Hsum. field. apply pow2_nonzero.
Qed.

(* Sum of squared coefficients = average of squared values. *)
Theorem plancherel :
  forall n (F : MV n),
    sumQ (map (fun M => F M * F M) (all_masks n))
    == (1 / pow2 n) * sumQ (map (fun s => eval F s * eval F s) (all_corners n)).
Proof.
  intros n F.
  transitivity (sumQ (map (fun M => (1 / pow2 n) *
                  sumQ (map (fun s => F M * eval F s * chi' M s) (all_corners n)))
                  (all_masks n))).
  { apply sumQ_map_ext. intros M _.
    transitivity (F M * ((1 / pow2 n) *
                    sumQ (map (fun s => eval F s * chi' M s) (all_corners n)))).
    { apply Qmult_comp; [reflexivity | apply walsh_inversion]. }
    transitivity ((1 / pow2 n) *
                  (F M * sumQ (map (fun s => eval F s * chi' M s) (all_corners n)))); [ring|].
    apply Qmult_comp; [reflexivity|].
    rewrite <- (sum_scale _ (F M) (fun s => eval F s * chi' M s)).
    apply sumQ_map_ext. intros s _. ring. }
  rewrite sum_scale. apply Qmult_comp; [reflexivity|].
  rewrite (sum_swap _ _ (fun M s => F M * eval F s * chi' M s)).
  apply sumQ_map_ext. intros s _.
  unfold eval at 3.
  rewrite <- (sum_scale _ (eval F s) (fun m => F m * chi' m s)).
  apply sumQ_map_ext. intros M _. ring.
Qed.

(* ================================================================= *)
(* 5. The upper bound: l1(embed_pm f) <= 2^(n/2)                     *)
(* ================================================================= *)

Lemma signed_sq : forall b, signed b * signed b == 1.
Proof. destruct b; reflexivity. Qed.

(* Parseval for +-1 functions: the squared coefficients sum to 1. *)
Theorem parseval_pm :
  forall n (f : Corner n -> bool),
    sumQ (map (fun M => embed_pm f M * embed_pm f M) (all_masks n)) == 1.
Proof.
  intros n f. rewrite plancherel.
  transitivity ((1 / pow2 n) * sumQ (map (fun _ : Corner n => 1) (all_corners n))).
  { apply Qmult_comp; [reflexivity|]. apply sumQ_map_ext. intros s _.
    rewrite embed_pm_correct. apply signed_sq. }
  rewrite sum_corners_one. field. apply pow2_nonzero.
Qed.

(* Cauchy-Schwarz specialised to l1 over all masks. *)
Lemma l1_sq_le :
  forall n (F : MV n),
    l1_norm F * l1_norm F <= pow2 n * sumQ (map (fun M => F M * F M) (all_masks n)).
Proof.
  intros n F. unfold l1_norm.
  eapply Qle_trans; [apply (cauchy_schwarz _ (fun M => Qabs (F M)))|].
  rewrite (sum_all_masks_const n (fun _ => 1) 1) by (intro; reflexivity).
  apply Qle_of_Qeq. rewrite Qmult_1_r. apply Qmult_comp; [reflexivity|].
  apply sumQ_map_ext. intros M _.
  rewrite <- Qabs_Qmult. apply Qabs_pos. apply sq_nonneg.
Qed.

Lemma l1_nonneg : forall n (F : MV n), 0 <= l1_norm F.
Proof. intros n F. unfold l1_norm. apply sum_nonneg. intro. apply Qabs_nonneg. Qed.

Theorem l1_pm_sq_le :
  forall n (f : Corner n -> bool),
    l1_norm (embed_pm f) * l1_norm (embed_pm f) <= pow2 n.
Proof.
  intros n f. eapply Qle_trans; [apply l1_sq_le|].
  rewrite parseval_pm, Qmult_1_r. apply Qle_refl.
Qed.

(* THE UPPER BOUND.  Every Boolean function on 2m variables has
   l1 <= 2^m, and IP attains it (l1_embed_pm_IP_exact). *)
Theorem l1_pm_le_pow2 :
  forall m (f : Corner (m + m) -> bool), l1_norm (embed_pm f) <= pow2 m.
Proof.
  intros m f.
  pose proof (l1_pm_sq_le (m + m) f) as H.
  rewrite pow2Q_add in H.
  pose proof (l1_nonneg _ (embed_pm f)).
  pose proof (pow2Q_pos m).
  nra.
Qed.


(* ================================================================= *)
(* 6. Spectra of gates                                               *)
(* ================================================================= *)

(* Affine combinations of +-1 values, summed against any weight. *)
Lemma sum_lin :
  forall (A : Type) (u v w z x : A -> Q) (a b c d : Q) (l : list A),
    (forall s, u s == a + b * v s + c * w s + d * z s) ->
    sumQ (map (fun s => u s * x s) l)
    == a * sumQ (map x l) + b * sumQ (map (fun s => v s * x s) l)
       + c * sumQ (map (fun s => w s * x s) l) + d * sumQ (map (fun s => z s * x s) l).
Proof.
  intros A u v w z x a b c d l H.
  induction l as [|t l IH]; cbn [map sumQ]; [ring|].
  rewrite IH, (H t). ring.
Qed.

Lemma sum_affine_le :
  forall (A : Type) (u p q r t : A -> Q) (a b c d : Q) (l : list A),
    (forall x, u x <= a * p x + b * q x + c * r x + d * t x) ->
    sumQ (map u l)
    <= a * sumQ (map p l) + b * sumQ (map q l) + c * sumQ (map r l) + d * sumQ (map t l).
Proof.
  intros A u p q r t a b c d l H.
  induction l as [|x l IH]; cbn [map sumQ]; [lra|].
  specialize (H x). lra.
Qed.

Section Gates.
  Variable n : nat.

  (* Spectrum of the constant function 1: the delta at the empty mask. *)
  Definition delta0 : MV n :=
    fun M => (1 / pow2 n) * sumQ (map (fun s => chi' M s) (all_corners n)).

  Lemma delta0_val :
    forall M, delta0 M == if mask_eq_dec M mask_empty then 1 else 0.
  Proof.
    intro M. unfold delta0.
    transitivity ((1 / pow2 n) *
      sumQ (map (fun s => chi' mask_empty s * chi' M s) (all_corners n))).
    { apply Qmult_comp; [reflexivity|]. apply sumQ_map_ext. intros s _.
      rewrite chi_mask_empty. ring. }
    rewrite chi_orth.
    destruct (mask_eq_dec mask_empty M) as [H1|H1];
      destruct (mask_eq_dec M mask_empty) as [H2|H2];
      try (field; apply pow2_nonzero);
      exfalso; [apply H2 | apply H1]; symmetry; assumption.
  Qed.

  Lemma l1_one_hot :
    forall (F : MV n) (U : Mask n),
      (forall M, Qabs (F M) == if mask_eq_dec M U then 1 else 0) ->
      l1_norm F == 1.
  Proof.
    intros F U H. unfold l1_norm.
    transitivity (sumQ (map (fun M => if mask_eq_dec M U then 1 else 0) (all_masks n))).
    { apply sumQ_map_ext. intros M _. apply H. }
    apply sumQ_all_masks_pick_const_Q.
  Qed.

  Lemma l1_delta0 : l1_norm delta0 == 1.
  Proof.
    apply (l1_one_hot delta0 mask_empty). intro M. rewrite delta0_val.
    destruct (mask_eq_dec M mask_empty); reflexivity.
  Qed.

  (* If (-1)^h is an affine combination of (-1)^f, (-1)^g, (-1)^k,
     then so is its spectrum. *)
  Lemma pm_affine :
    forall (h f g k : Corner n -> bool) (a b c d : Q),
      (forall s, signed (h s) == a + b * signed (f s) + c * signed (g s) + d * signed (k s)) ->
      forall M,
        embed_pm h M
        == a * delta0 M + b * embed_pm f M + c * embed_pm g M + d * embed_pm k M.
  Proof.
    intros h f g k a b c d H M.
    unfold embed_pm, signed_walsh, delta0.
    pose proof (sum_lin _ (fun s => signed (h s)) (fun s => signed (f s))
                  (fun s => signed (g s)) (fun s => signed (k s)) (fun s => chi' M s)
                  a b c d (all_corners n) H) as P.
    cbv beta in P. rewrite P. ring.
  Qed.

  Lemma l1_ext : forall (F G : MV n), (forall M, F M == G M) -> l1_norm F == l1_norm G.
  Proof.
    intros F G H. unfold l1_norm. apply sumQ_map_ext. intros M _. rewrite H. reflexivity.
  Qed.

  (* l1 of an affine combination of spectra: triangle inequality. *)
  Lemma l1_affine_le :
    forall (F D G H K : MV n) (a b c d : Q),
      (forall M, F M == a * D M + b * G M + c * H M + d * K M) ->
      l1_norm F <= Qabs a * l1_norm D + Qabs b * l1_norm G
                   + Qabs c * l1_norm H + Qabs d * l1_norm K.
  Proof.
    intros F D G H K a b c d HF. unfold l1_norm.
    apply (sum_affine_le _ (fun M => Qabs (F M)) (fun M => Qabs (D M))
             (fun M => Qabs (G M)) (fun M => Qabs (H M)) (fun M => Qabs (K M))).
    intro M. rewrite (HF M).
    pose proof (Qabs_triangle (a * D M + b * G M + c * H M) (d * K M)).
    pose proof (Qabs_triangle (a * D M + b * G M) (c * H M)).
    pose proof (Qabs_triangle (a * D M) (b * G M)).
    rewrite !Qabs_Qmult in *. lra.
  Qed.

  (* XOR is the untwisted convolution of spectra.  (The Clifford
     product would carry the sign cocycle and would NOT match.) *)
  Theorem embed_pm_xor :
    forall (f g : Corner n -> bool) M,
      embed_pm (fun s => xorb (f s) (g s)) M == mv_conv (embed_pm f) (embed_pm g) M.
  Proof.
    intros f g M.
    rewrite (walsh_inversion n (mv_conv (embed_pm f) (embed_pm g)) M).
    unfold embed_pm at 1. unfold signed_walsh.
    apply Qmult_comp; [reflexivity|]. apply sumQ_map_ext. intros s _.
    rewrite eval_conv, !embed_pm_correct, signed_xorb. reflexivity.
  Qed.

  (* Every +-1 function has l1 >= 1 (it takes the value +-1 somewhere). *)
  Lemma l1_pm_ge_1 : forall f : Corner n -> bool, 1 <= l1_norm (embed_pm f).
  Proof.
    intro f. set (s0 := Vector.const Pos n).
    assert (E : Qabs (eval (embed_pm f) s0) == 1).
    { rewrite embed_pm_correct. destruct (f s0); reflexivity. }
    rewrite <- E. unfold eval, l1_norm.
    eapply Qle_trans; [apply Qabs_sumQ_map_le|].
    apply Qle_of_Qeq. apply sumQ_map_ext. intros M _.
    rewrite Qabs_Qmult, Qabs_chi. ring.
  Qed.

  (* ---- the four gate bounds ------------------------------------- *)

  Lemma l1_pm_not :
    forall f : Corner n -> bool,
      l1_norm (embed_pm (fun s => negb (f s))) == l1_norm (embed_pm f).
  Proof.
    intro f. unfold l1_norm. apply sumQ_map_ext. intros M _.
    rewrite (pm_affine (fun s => negb (f s)) f f f 0 (-1) 0 0) by
      (intro s; destruct (f s); reflexivity).
    setoid_replace (0 * delta0 M + -1 * embed_pm f M + 0 * embed_pm f M + 0 * embed_pm f M)
      with (- embed_pm f M) by ring.
    apply Qabs_opp.
  Qed.

  Lemma l1_pm_xor_le :
    forall f g : Corner n -> bool,
      l1_norm (embed_pm (fun s => xorb (f s) (g s)))
      <= l1_norm (embed_pm f) * l1_norm (embed_pm g).
  Proof.
    intros f g. rewrite (l1_ext _ _ (embed_pm_xor f g)). apply l1_conv_bound.
  Qed.

  Lemma l1_pm_and_le :
    forall f g : Corner n -> bool,
      l1_norm (embed_pm (fun s => andb (f s) (g s)))
      <= (1#2) * (1 + l1_norm (embed_pm f)) * (1 + l1_norm (embed_pm g)).
  Proof.
    intros f g.
    eapply Qle_trans.
    { apply (l1_affine_le _ delta0 (embed_pm f) (embed_pm g)
               (embed_pm (fun s => xorb (f s) (g s))) (1#2) (1#2) (1#2) (-1#2)).
      apply pm_affine. intro s. destruct (f s), (g s); reflexivity. }
    rewrite l1_delta0.
    pose proof (l1_pm_xor_le f g).
    change (Qabs (1#2)) with (1#2). change (Qabs (-1#2)) with (1#2).
    lra.
  Qed.

  Lemma l1_pm_or_le :
    forall f g : Corner n -> bool,
      l1_norm (embed_pm (fun s => orb (f s) (g s)))
      <= (1#2) * (1 + l1_norm (embed_pm f)) * (1 + l1_norm (embed_pm g)).
  Proof.
    intros f g.
    eapply Qle_trans.
    { apply (l1_affine_le _ delta0 (embed_pm f) (embed_pm g)
               (embed_pm (fun s => xorb (f s) (g s))) (-1#2) (1#2) (1#2) (1#2)).
      apply pm_affine. intro s. destruct (f s), (g s); reflexivity. }
    rewrite l1_delta0.
    pose proof (l1_pm_xor_le f g).
    change (Qabs (1#2)) with (1#2). change (Qabs (-1#2)) with (1#2).
    lra.
  Qed.

  (* ---- leaves: literals and the constant ------------------------ *)

  Definition lit (i : Fin.t n) (s : Corner n) : bool :=
    match Vector.nth s i with Neg => true | Pos => false end.

  Lemma l1_pm_character :
    forall (h : Corner n -> bool) (U : Mask n),
      (forall s, signed (h s) == chi' U s) -> l1_norm (embed_pm h) == 1.
  Proof.
    intros h U H. apply (l1_one_hot _ U). intro M.
    unfold embed_pm, signed_walsh.
    transitivity (Qabs ((1 / pow2 n) *
      sumQ (map (fun s => chi' U s * chi' M s) (all_corners n)))).
    { apply Qabs_wd. apply Qmult_comp; [reflexivity|].
      apply sumQ_map_ext. intros s _. rewrite H. reflexivity. }
    rewrite chi_orth.
    destruct (mask_eq_dec U M) as [H1|H1]; destruct (mask_eq_dec M U) as [H2|H2].
    - rewrite Qmult_comm, Qmult_div_r by apply pow2_nonzero. reflexivity.
    - exfalso. apply H2. symmetry. exact H1.
    - exfalso. apply H1. symmetry. exact H2.
    - rewrite Qmult_0_r. reflexivity.
  Qed.

  Lemma l1_pm_lit : forall i, l1_norm (embed_pm (lit i)) == 1.
  Proof.
    intro i. apply (l1_pm_character _ (mask_single i)). intro s.
    rewrite chi_mask_single. unfold lit. destruct (Vector.nth s i); reflexivity.
  Qed.

  Lemma l1_pm_false : l1_norm (embed_pm (fun _ : Corner n => false)) == 1.
  Proof.
    apply (l1_pm_character _ mask_empty). intro s. rewrite chi_mask_empty. reflexivity.
  Qed.

  (* The measure only depends on the truth table. *)
  Lemma l1_pm_ext :
    forall f g : Corner n -> bool, (forall s, f s = g s) ->
      l1_norm (embed_pm f) == l1_norm (embed_pm g).
  Proof.
    intros f g H. apply l1_ext. intro M. unfold embed_pm, signed_walsh.
    apply Qmult_comp; [reflexivity|]. apply sumQ_map_ext. intros s _. rewrite H. reflexivity.
  Qed.

End Gates.


(* ================================================================= *)
(* 7. The instance: De Morgan + XOR formulas on n = 2m variables     *)
(* ================================================================= *)

Inductive BGate : Type := GAnd | GOr | GXor.

Section Instance.
  Variable m : nat.
  Local Notation N := (m + m)%nat.
  Local Notation BF := (Corner N -> bool).

  (* Leaves: literal x_i, or the constant false.  NOT is the unary
     gate, so negated literals and the constant true cost nothing. *)
  Definition leaf_val (l : option (Fin.t N)) : BF :=
    match l with
    | Some i => lit N i
    | None   => fun _ => false
    end.

  Definition not_val (_ : unit) (f : BF) : BF := fun s => negb (f s).

  Definition bin_val (o : BGate) (f g : BF) : BF :=
    fun s => match o with
             | GAnd => andb (f s) (g s)
             | GOr  => orb  (f s) (g s)
             | GXor => xorb (f s) (g s)
             end.

  (* The measure. *)
  Definition mu (f : BF) : Q := 2 * l1_norm (embed_pm f).

  Lemma mu_ge_2 : forall f, 2 <= mu f.
  Proof. intro f. unfold mu. pose proof (l1_pm_ge_1 N f). lra. Qed.

  (* ---- the four hypotheses of Ceiling_Formula, discharged -------- *)

  Lemma mu_nonneg : forall f, 0 <= mu f.
  Proof. intro f. pose proof (mu_ge_2 f). lra. Qed.

  Lemma mu_leaf : forall l, mu (leaf_val l) <= 2.
  Proof.
    intros [i|]; unfold mu, leaf_val.
    - rewrite l1_pm_lit. apply Qle_refl.
    - rewrite l1_pm_false. apply Qle_refl.
  Qed.

  Lemma mu_unary : forall u f, mu (not_val u f) <= mu f.
  Proof.
    intros u f. unfold mu, not_val. rewrite l1_pm_not. apply Qle_refl.
  Qed.

  Lemma mu_submult : forall o f g, mu (bin_val o f g) <= mu f * mu g.
  Proof.
    intros o f g. unfold mu.
    pose proof (l1_pm_ge_1 N f) as Hf. pose proof (l1_pm_ge_1 N g) as Hg.
    destruct o; unfold bin_val.
    - pose proof (l1_pm_and_le N f g). nra.
    - pose proof (l1_pm_or_le N f g). nra.
    - pose proof (l1_pm_xor_le N f g). nra.
  Qed.

  (* ---- the range of mu: tight upper bound ----------------------- *)

  Lemma qpow2_pow2 : forall k, Ceiling_Formula.qpow 2 k == pow2 k.
  Proof.
    induction k as [|k IH]; cbn [Ceiling_Formula.qpow pow2]; [reflexivity|].
    rewrite IH. reflexivity.
  Qed.

  Theorem mu_le_top : forall f, mu f <= Ceiling_Formula.qpow 2 (S m).
  Proof.
    intro f. rewrite qpow2_pow2. unfold mu. cbn [pow2].
    pose proof (l1_pm_le_pow2 m f). lra.
  Qed.

  Theorem mu_IP_top : mu (@IP_n_func N) == Ceiling_Formula.qpow 2 (S m).
  Proof.
    rewrite qpow2_pow2. unfold mu. rewrite l1_embed_pm_IP_exact. reflexivity.
  Qed.

  (* ---- formulas ------------------------------------------------- *)

  Definition formula := Ceiling_Formula.formula (option (Fin.t N)) unit BGate.

  Definition feval (phi : formula) : BF :=
    Ceiling_Formula.eval BF (option (Fin.t N)) unit BGate leaf_val not_val bin_val phi.

  Definition leaves (phi : formula) : nat :=
    Ceiling_Formula.leaves (option (Fin.t N)) unit BGate phi.

  (* "The l1 method certifies that f needs more than k leaves." *)
  Definition certifies (f : BF) (k : nat) : Prop :=
    Ceiling_Formula.certifies BF mu 2 f k.

  Lemma two_ge_1 : 1 <= 2.
  Proof. discriminate. Qed.

  Lemma mu_ext : forall f g : BF, (forall s, f s = g s) -> mu f == mu g.
  Proof. intros f g H. unfold mu. rewrite (l1_pm_ext N f g H). reflexivity. Qed.

  (* ================================================================ *)
  (* 8. END-TO-END THEOREMS                                           *)
  (* ================================================================ *)

  (* Soundness: a certificate is a genuine formula-size lower bound,
     for every formula that computes f pointwise. *)
  Theorem l1_method_sound :
    forall f k, certifies f k ->
    forall phi : formula, (forall s, feval phi s = f s) -> (k < leaves phi)%nat.
  Proof.
    intros f k Hcert phi Hphi.
    apply (Ceiling_Formula.qpow_lt_exponent 2); [exact two_ge_1|].
    unfold certifies, Ceiling_Formula.certifies in Hcert.
    eapply Qlt_le_trans; [exact Hcert|].
    rewrite <- (mu_ext _ _ Hphi).
    exact (Ceiling_Formula.formula_bound BF (option (Fin.t N)) unit BGate
             leaf_val not_val bin_val mu 2
             mu_nonneg mu_leaf mu_unary mu_submult phi).
  Qed.

  (* The cap: no Boolean function on 2m variables gets a certified
     bound above m = n/2. *)
  Theorem l1_method_cap : forall f k, certifies f k -> (k <= m)%nat.
  Proof.
    intros f k Hcert.
    pose proof (Ceiling_Formula.formula_ceiling BF mu 2 two_ge_1 (S m) mu_le_top f k Hcert).
    lia.
  Qed.

  (* The cap is attained: IP certifies exactly m. *)
  Theorem l1_method_cap_attained : certifies (@IP_n_func N) m.
  Proof.
    unfold certifies, Ceiling_Formula.certifies.
    rewrite mu_IP_top, !qpow2_pow2. cbn [pow2].
    pose proof (pow2Q_pos m). lra.
  Qed.

  (* HEADLINE.  The best formula-size lower bound the Fourier-l1
     method can certify, for any Boolean function on n = 2m
     variables, is exactly m = n/2 leaves; IP reaches it. *)
  Theorem l1_method_best_is_m :
    certifies (@IP_n_func N) m /\ (forall f k, certifies f k -> (k <= m)%nat).
  Proof. split; [exact l1_method_cap_attained | exact l1_method_cap]. Qed.

  (* And what the certificate means for IP concretely: every De
     Morgan + XOR formula for IP has more than m leaves. *)
  Corollary IP_needs_more_than_m_leaves :
    forall phi : formula, (forall s, feval phi s = @IP_n_func N s) -> (m < leaves phi)%nat.
  Proof.
    intros phi H. exact (l1_method_sound _ m l1_method_cap_attained phi H).
  Qed.

End Instance.

Print Assumptions l1_method_best_is_m.
Print Assumptions IP_needs_more_than_m_leaves.
