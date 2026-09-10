(* ================================================================= *)
(*  Cln_IP_l1_exact.v                                                *)
(*                                                                   *)
(*  Exact l1 Walsh norms of the inner-product function               *)
(*      IP(x0,y0,...,x_{m-1},y_{m-1}) = XOR_i (x_i /\ y_i)           *)
(*  on n = 2m variables, in both encodings.  Write p = 2^m.          *)
(*                                                                   *)
(*    +-1 encoding  (-1)^IP :                                        *)
(*      l1_embed_pm_IP_exact     l1 = p                   (exactly)  *)
(*                                                                   *)
(*    0/1 encoding  (the library's `embed`) :                        *)
(*      l1_embed_IP_exact        l1 = p/2 + 1/2 - 1/p     (exactly)  *)
(*      l1_embed_IP_between      2^(m-1) <= l1 < 2^(m-1) + 1/2       *)
(*                                                                   *)
(*  So the existing bound  l1_norm_embed_IP_lower_bound : 2^(m-1) <= *)
(*  l1(embed IP)  is tight up to an additive 1/2.  The clean power   *)
(*  of two belongs to the +-1 encoding, which is the encoding in     *)
(*  which the ceiling theorems (Ceiling_Formula.v) are tight:        *)
(*  l1 of a 2-input AND is 2 = C, XOR on disjoint variables          *)
(*  multiplies l1 exactly, and IP reaches C^m = 2^m = 2^(n/2).       *)
(*                                                                   *)
(*  Proof route.  Every unnormalised +-1 Walsh coefficient of IP     *)
(*  has magnitude exactly p (signed_walsh_IP_magnitude, from         *)
(*  Cln_SupportAlgebra), and the empty-mask coefficient is +p        *)
(*  (proved here, via the pair factorisation).  Everything else is   *)
(*  bookkeeping over all_masks.                                      *)
(*                                                                   *)
(*  Depends only on the admit-free part of the library:              *)
(*    Cln_Full, Cln_Grade, Cln_BoolDist, Cln_SupportAlgebra,         *)
(*    Cln_finite_l1_submultiplicativity.                             *)
(*  No admits.  No new axioms: Print Assumptions shows only          *)
(*  eq_rect_eq, which the library already uses (embed_correct and    *)
(*  signed_walsh_IP_magnitude depend on it via dependent             *)
(*  destruction).  Built and checked with Coq 8.20.0.                *)
(* ================================================================= *)

Require Import Cln_Full.
Require Import Cln_Grade.
Require Import Cln_BoolDist.
Require Import Cln_SupportAlgebra.
Require Import Cln_finite_l1_submultiplicativity.

From Coq Require Import List QArith Qabs Lia.
Require Import Coq.Program.Equality.
Import ListNotations.
Open Scope Q_scope.

(* ----------------------------------------------------------------- *)
(* 1. Facts about pow2 in Q                                          *)
(*    (the library's pow2_pos / pow2_ge_2 are about Nat.pow)         *)
(* ----------------------------------------------------------------- *)

Lemma pow2Q_pos : forall n, 0 < pow2 n.
Proof.
  induction n as [|n IH]; simpl.
  - reflexivity.
  - apply Qmult_lt_0_compat; [reflexivity | exact IH].
Qed.

Lemma pow2Q_ge_1 : forall n, 1 <= pow2 n.
Proof.
  induction n as [|n IH]; simpl.
  - apply Qle_refl.
  - apply Qle_trans with (y := 2 * 1); [discriminate|].
    apply Qmult_le_l; [reflexivity | exact IH].
Qed.

Lemma pow2Q_add : forall a b, pow2 (a + b) == pow2 a * pow2 b.
Proof.
  induction a as [|a IH]; intro b; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

Lemma Qabs_pow2 : forall n, Qabs (pow2 n) == pow2 n.
Proof. intro n. apply Qabs_pos. apply Qlt_le_weak, pow2Q_pos. Qed.

Lemma inv_pow2_nonneg : forall n, 0 <= 1 / pow2 n.
Proof.
  intro n. apply Qle_shift_div_l; [apply pow2Q_pos|].
  rewrite Qmult_0_l. discriminate.
Qed.

Lemma Qabs_inv_pow2 : forall n, Qabs (1 / pow2 n) == 1 / pow2 n.
Proof. intro n. apply Qabs_pos, inv_pow2_nonneg. Qed.

Lemma Qabs_signed : forall s, Qabs (signed s) == 1.
Proof. destruct s; reflexivity. Qed.

(* 0 <= 1/2 - 1/(2p): the empty-mask coefficient of embed IP is
   nonnegative. *)
Lemma half_minus_nonneg : forall k, 0 <= (1#2) - 1 / (2 * pow2 k).
Proof.
  intro k.
  pose proof (pow2Q_pos k) as Hp.
  pose proof (pow2Q_ge_1 k) as H1.
  setoid_replace ((1#2) - 1 / (2 * pow2 k)) with ((pow2 k - 1) / (2 * pow2 k))
    by (field; apply pow2_nonzero).
  apply Qle_shift_div_l.
  - apply Qmult_lt_0_compat; [reflexivity | exact Hp].
  - rewrite Qmult_0_l. apply Qle_minus_iff in H1. exact H1.
Qed.

Lemma inv_2pow2_pos : forall k, 0 < 1 / (2 * pow2 k).
Proof.
  intro k. apply Qlt_shift_div_l.
  - apply Qmult_lt_0_compat; [reflexivity | apply pow2Q_pos].
  - rewrite Qmult_0_l. reflexivity.
Qed.

(* ----------------------------------------------------------------- *)
(* 2. Two summation lemmas over all_masks                            *)
(* ----------------------------------------------------------------- *)

(* A pointwise-constant sum over all 2^n masks. *)
Lemma sum_all_masks_const :
  forall n (g : Mask n -> Q) (c : Q),
    (forall M, g M == c) ->
    sumQ (List.map g (all_masks n)) == pow2 n * c.
Proof.
  induction n as [|n IH]; intros g c Hg.
  - simpl. rewrite Hg. ring.
  - cbn [all_masks pow2].
    rewrite List.map_app, sumQ_app, !List.map_map.
    rewrite (IH _ c), (IH _ c);
      [ring | intro M; apply Hg | intro M; apply Hg].
Qed.

(* Constant everywhere, plus a correction d at a single mask U. *)
Lemma sum_all_masks_split :
  forall n (g : Mask n -> Q) (c d : Q) (U : Mask n),
    (forall M, g M == c + (if mask_eq_dec M U then d else 0)) ->
    sumQ (List.map g (all_masks n)) == pow2 n * c + d.
Proof.
  intros n g c d U Hg.
  transitivity
    (sumQ (List.map (fun M => c + (if mask_eq_dec M U then d else 0)) (all_masks n))).
  { apply sumQ_map_ext. intros M _. apply Hg. }
  pose proof (@sumQ_map_add _ (fun _ : Mask n => c)
                (fun M => if mask_eq_dec M U then d else 0) (all_masks n)) as P.
  cbv beta in P. rewrite P.
  rewrite (sum_all_masks_const n (fun _ => c) c) by (intro; reflexivity).
  rewrite sumQ_all_masks_pick_const_Q.
  reflexivity.
Qed.

(* ----------------------------------------------------------------- *)
(* 3. Walsh coefficients of IP                                       *)
(* ----------------------------------------------------------------- *)

(* Every unnormalised +-1 Walsh coefficient has magnitude exactly 2^m.
   Extends signed_walsh_IP_magnitude to m = 0. *)
Lemma IP_sw_abs :
  forall m (M : Mask (m + m)),
    Qabs (signed_walsh (@IP_n_func (m + m)) M) == pow2 m.
Proof.
  intros m M. destruct m as [|m'].
  - dependent destruction M.
    unfold signed_walsh, IP_n_func, signed. cbn. reflexivity.
  - destruct (signed_walsh_IP_magnitude (S m') M ltac:(lia)) as [s Hs].
    rewrite Hs, Qabs_Qmult, Qabs_signed, <- pow2_injectZ, Qabs_pow2.
    ring.
Qed.

Lemma to_list_const_false :
  forall n, Vector.to_list (Vector.const false n) = List.repeat false n.
Proof.
  induction n as [|n IH]; [reflexivity|].
  change (Vector.const false (S n)) with (Vector.cons bool false n (Vector.const false n)).
  rewrite to_list_cons, IH. reflexivity.
Qed.

Lemma prod_pairs_raw_all_false :
  forall k, prod_pairs_raw (List.repeat false (k + k)) == pow2 k.
Proof.
  induction k as [|k IH]; [reflexivity|].
  replace (S k + S k)%nat with (S (S (k + k))) by lia.
  cbn [List.repeat prod_pairs_raw pow2].
  rewrite IH, (signed_walsh_AND_2var false false).
  cbv iota. ring.
Qed.

(* The empty-mask coefficient is +2^m: IP takes the value 0 on
   (2^(2m) + 2^m)/2 inputs, i.e. 2^m more zeros than ones. *)
Lemma IP_sw_empty :
  forall m, signed_walsh (@IP_n_func (m + m)) mask_empty == pow2 m.
Proof.
  intro m.
  rewrite signed_walsh_IP_factored.
  unfold prod_pairs, mask_empty.
  rewrite to_list_const_false.
  apply prod_pairs_raw_all_false.
Qed.

(* ----------------------------------------------------------------- *)
(* 4. The +-1 encoding inside the library's algebra                  *)
(* ----------------------------------------------------------------- *)

(* Fourier coefficients of (-1)^f. *)
Definition embed_pm {n} (f : Corner n -> bool) : MV n :=
  fun M => (1 / pow2 n) * signed_walsh f M.

(* It is the multivector  e_empty - 2 * embed f,  i.e. the embedding
   of the +-1 function 1 - 2F.  No new structure is introduced. *)
Lemma embed_pm_coeff :
  forall n (f : Corner n -> bool) (M : Mask n),
    embed_pm f M == basis mask_empty M - 2 * embed f M.
Proof.
  intros n f M. unfold embed_pm, basis.
  rewrite (embed_via_signed_walsh n f M).
  destruct (mask_eq_dec M mask_empty); field; apply pow2_nonzero.
Qed.

(* And it evaluates to the +-1 value at every corner. *)
Theorem embed_pm_correct :
  forall n (f : Corner n -> bool) (s : Corner n),
    eval (embed_pm f) s == signed (f s).
Proof.
  intros n f s. unfold eval.
  transitivity
    (sumQ (List.map (fun m => (if mask_eq_dec m mask_empty then chi' m s else 0)
                              + (-2) * (embed f m * chi' m s)) (all_masks n))).
  { apply sumQ_map_ext. intros M _.
    rewrite embed_pm_coeff. unfold basis.
    destruct (mask_eq_dec M mask_empty); ring. }
  pose proof (@sumQ_map_add _
                (fun m => if mask_eq_dec m mask_empty then chi' m s else 0)
                (fun m => (-2) * (embed f m * chi' m s)) (all_masks n)) as P.
  cbv beta in P. rewrite P. clear P.
  pose proof (@sumQ_map_scale_l _ (-2) (fun m => embed f m * chi' m s) (all_masks n)) as P.
  cbv beta in P. rewrite P. clear P.
  pose proof (sumQ_all_masks_pick (fun m => chi' m s) (@mask_empty n)) as P.
  cbv beta in P. rewrite P. clear P.
  change (sumQ (List.map (fun m => embed f m * chi' m s) (all_masks n)))
    with (eval (embed f) s).
  rewrite embed_correct, chi_mask_empty.
  destruct (f s); unfold bQ, signed; ring.
Qed.

(* ----------------------------------------------------------------- *)
(* 5. MAIN RESULT, +-1 encoding:  l1 = 2^m exactly                   *)
(* ----------------------------------------------------------------- *)

Theorem l1_embed_pm_IP_exact :
  forall m, l1_norm (embed_pm (@IP_n_func (m + m))) == pow2 m.
Proof.
  intro m. unfold l1_norm.
  rewrite (sum_all_masks_const _ _ (1 / pow2 m)).
  - rewrite pow2Q_add. field. apply pow2_nonzero.
  - intro M. unfold embed_pm.
    rewrite Qabs_Qmult, Qabs_inv_pow2, IP_sw_abs, pow2Q_add.
    field. apply pow2_nonzero.
Qed.

(* ----------------------------------------------------------------- *)
(* 6. MAIN RESULT, 0/1 encoding:  l1 = p/2 + 1/2 - 1/p exactly       *)
(* ----------------------------------------------------------------- *)

(* Coefficient magnitudes of embed IP: 1/(2p) off the empty mask,
   1/2 - 1/(2p) on it.  Written as "constant + correction at empty". *)
Lemma embed_IP_abs_coeff :
  forall m (M : Mask (m + m)),
    Qabs (embed (@IP_n_func (m + m)) M)
    == 1 / (2 * pow2 m)
       + (if mask_eq_dec M mask_empty then (1#2) - 1 / pow2 m else 0).
Proof.
  intros m M.
  rewrite (embed_via_signed_walsh (m + m) (@IP_n_func (m + m)) M).
  destruct (mask_eq_dec M mask_empty) as [HM | HM].
  - subst M. rewrite IP_sw_empty, pow2Q_add.
    setoid_replace (1 / (pow2 m * pow2 m) * ((1#2) * (pow2 m * pow2 m) - (1#2) * pow2 m))
      with ((1#2) - 1 / (2 * pow2 m)) by (field; apply pow2_nonzero).
    rewrite Qabs_pos by apply half_minus_nonneg.
    field. apply pow2_nonzero.
  - setoid_replace (1 / pow2 (m + m) * ((1#2) * 0 - (1#2) * signed_walsh (@IP_n_func (m + m)) M))
      with ((-(1#2)) * (1 / pow2 (m + m) * signed_walsh (@IP_n_func (m + m)) M)) by ring.
    rewrite Qabs_Qmult, Qabs_Qmult, Qabs_inv_pow2, IP_sw_abs, pow2Q_add.
    assert (Hh : Qabs (-(1#2)) == 1#2) by reflexivity.
    rewrite Hh.
    field. apply pow2_nonzero.
Qed.

Theorem l1_embed_IP_exact :
  forall m,
    l1_norm (embed (@IP_n_func (m + m)))
    == pow2 m * (1#2) + (1#2) - 1 / pow2 m.
Proof.
  intro m. unfold l1_norm.
  rewrite (sum_all_masks_split _ _ (1 / (2 * pow2 m)) ((1#2) - 1 / pow2 m) mask_empty).
  - rewrite pow2Q_add. field. apply pow2_nonzero.
  - apply embed_IP_abs_coeff.
Qed.

(* The existing lower bound 2^(m-1) is tight up to an additive 1/2. *)
Corollary l1_embed_IP_between :
  forall m, (m >= 1)%nat ->
    pow2 (m - 1) <= l1_norm (embed (@IP_n_func (m + m)))
    /\ l1_norm (embed (@IP_n_func (m + m))) < pow2 (m - 1) + (1#2).
Proof.
  intros m Hm. destruct m as [|k]; [lia|].
  replace (S k - 1)%nat with k by lia.
  rewrite l1_embed_IP_exact. cbn [pow2].
  split.
  - apply Qle_minus_iff.
    setoid_replace (2 * pow2 k * (1#2) + (1#2) - 1 / (2 * pow2 k) + - pow2 k)
      with ((1#2) - 1 / (2 * pow2 k)) by (field; apply pow2_nonzero).
    apply half_minus_nonneg.
  - apply Qlt_minus_iff.
    setoid_replace (pow2 k + (1#2) + - (2 * pow2 k * (1#2) + (1#2) - 1 / (2 * pow2 k)))
      with (1 / (2 * pow2 k)) by (field; apply pow2_nonzero).
    apply inv_2pow2_pos.
Qed.

(* ----------------------------------------------------------------- *)
(* 7. Sanity instances (checked by computation of the statements)    *)
(* ----------------------------------------------------------------- *)

(* m = 1 is the 2-input AND: l1 of (-1)^AND is 2 = C, the generator
   bound used in Ceiling_Formula.v; l1 of the 0/1 AND is 1.         *)
Example l1_AND_pm : l1_norm (embed_pm (@IP_n_func (1 + 1))) == 2.
Proof. rewrite l1_embed_pm_IP_exact. reflexivity. Qed.

Example l1_AND_01 : l1_norm (embed (@IP_n_func (1 + 1))) == 1.
Proof. rewrite l1_embed_IP_exact. reflexivity. Qed.

Print Assumptions l1_embed_pm_IP_exact.
Print Assumptions l1_embed_IP_exact.
Print Assumptions l1_embed_IP_between.
Print Assumptions embed_pm_correct.
