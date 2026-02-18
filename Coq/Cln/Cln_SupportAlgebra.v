(*

Dependency Chain for the Separation Theorem

The final theorem `IP_formula_size_lower_bound` needs:

1. **`support_size_IP`**
    — IP mod 2 on 2m variables has 2^m nonzero Fourier coefficients

2. **`translate_support_size_bound`**
    — formula of size s produces a GA expression with ≤ 2^s support

3. A **support-preservation lemma** connecting `translate_correct`
    to support size equality (i.e., if two multivectors agree pointwise, they have the same support)

4. **`Nat.pow_le_mono_r`** or similar to conclude `m ≤ formula_size phi` from `2^m ≤ 2^(formula_size phi)`

-------------------------------------------------------------------------------

### Block 1 — Basic support facts (straightforward)

These are all direct from definitions and should go quickly:

  **`support_size_zero`**:
    Every coefficient of `mv_zero` is 0, so the filter returns `[]`.
    Unfold `support_size`, `mv_zero`, show `Qeq_bool 0 0 = true` for each mask,
    so `negb` gives `false`, filter keeps nothing.

  **`support_size_mv_one`**:
    `mv_one = basis mask_empty`. Only `mask_empty` has coefficient 1 (nonzero);
    all others are 0. You basically need that `Qeq_bool 1 0 = false` and `Qeq_bool 0 0 = true`, then count the filter.

  **`support_size_basis`**:
    Same pattern — `basis (mask_single i)` has exactly one nonzero entry.

  **`support_size_le_2n`**:
    `support_size F` is the length of a filtered sublist of `all_masks n`,
    which has length `2^n`. Use `filter_length_le` or similar.

  **`support_size_scale`**:
    For `c ≠ 0`, the mask `m` has `c * F(m) ≠ 0` iff `F(m) ≠ 0`.
    The subtlety is that `Qeq_bool` works with Leibniz on `Q` but you need `==`-compatibility.
    You may need a helper like `Qeq_bool_iff`.

-------------------------------------------------------------------------------

### Block 2 — Algebraic support bounds

**`support_size_add`**:
  The support of `F + G` is contained in `supp(F) ∪ supp(G)`.
  You already have `supp_add` proved in `Cln_BoolDist.v`.
  The counting argument is: filter on a union is bounded by sum of filter lengths.
  This needs a list-level lemma about filter lengths.

**`support_size_conv`**:
  This is the key multiplicative bound. Support of `F ⊙ G` is contained in `{A ⊕ B : A ∈ supp(F), B ∈ supp(G)}`.
  You already have `support_conv_subset_xor`.
  The counting bound `|S₁ ⊕ S₂| ≤ |S₁| × |S₂|` needs a combinatorial argument 
    — the xor-sumset has at most that many distinct elements.

-------------------------------------------------------------------------------

### Block 3 — Structural support bound

**`eval_support_size_le`**:
  Induction on `GA_expr`, using blocks 1 and 2 at each case.
  The `Mul` case with the worst-case `2^n` fallback makes it easy — you just need `support_size_le_2n`.

-------------------------------------------------------------------------------

### Block 4 — Translation bound

**`support_size_bound_translate`**:
  Induction on `BoolFormula`.
  You need to check what `translate` produces for each case (AND → Conv, OR → combination, NOT → scalar ops)
  and verify the bound tracks through. The `formula_size` definition with the `1 +` at each connective gives you room.

**`translate_support_size_bound`**:
  Immediate corollary combining `eval_support_size_le` and `support_size_bound_translate` via transitivity.

-------------------------------------------------------------------------------

### Block 5 — The hard Fourier-analytic lemmas

**`support_size_XOR`**:
  XOR has exactly 1 nonzero coefficient (the pseudoscalar).
  You already proved `embed_XOR_full_mask` and `xor_sum_nonzero` in `Cln_Grade.v`.
  You need to additionally show all *other* coefficients are zero.
  This requires the Fourier inversion argument — XOR is a single character, so its embedding lands on exactly one mask.

**`support_size_IP`**:
  This is the hardest standalone lemma.
  IP mod 2 on 2m variables has exactly 2^m nonzero Fourier coefficients.
  The key insight: IP decomposes as XOR of m independent AND pairs, and in the Fourier/Walsh basis,
  each AND pair contributes coefficients at two levels, giving 2^m total nonzero terms via a tensor product structure.
  You could prove this via induction on m using `IP_n_func_cons2`.


### The Final Theorem

**`IP_formula_size_lower_bound`**:
  Once you have `support_size_IP` and `translate_support_size_bound`, you need one glue lemma:

```coq
Lemma support_size_eval_eq : forall n (F G : MV n),
  (forall s, eval F s == eval G s) ->
  support_size F = support_size G.
```

This follows from Walsh inversion / evaluation injectivity — if two multivectors agree on all corners, 
they're equal coefficient-wise (by orthogonality of characters).

You have `corner_walsh_sum_ortho` which gives you this. Then chain:

```
2^m = support_size(embed(IP))          [support_size_IP]
    = support_size(eval_expr sq (translate phi))  [glue lemma + translate_correct]
    ≤ 2^(formula_size phi)              [translate_support_size_bound]
```

Therefore `m ≤ formula_size phi`.

---------------------------------------------------------------------------------

The cleanest path is blocks 1 -> ... -> 5 -> final theorem.
Blocks 1–3 are mostly mechanical.
Block 4 depends on `translate`'s definition.
Block 5 (especially `support_size_IP`) is where the real math lives.

*)

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

Theorem eval_support_size_le :
  forall n (sq : Vector.t Q n) (e : GA_expr n),
    (support_size (eval_expr sq e) <= support_size_bound e)%nat.
Proof.
  intros n sq e.
  induction e as [i|c|e1 IH1 e2 IH2|e1 IH1 e2 IH2|e1 IH1 e2 IH2]; simpl.
  - (* Basis *)
    rewrite support_size_basis. lia.

  - (* Scalar *)
    destruct (Qeq_dec c 0) as [Hc0|Hc0].
    + (* c = 0 *)
      assert (He : forall m, mv_scale c (@mv_one n) m == (@mv_zero n) m).
      { intro m. unfold mv_scale, mv_zero. rewrite Hc0. ring. }
      rewrite (support_size_ext n (mv_scale c (@mv_one n)) (@mv_zero n) He).
      rewrite support_size_zero. lia.
    + (* c <> 0 *)
      rewrite (support_size_scale n c (@mv_one n) Hc0).
      rewrite support_size_mv_one. lia.

  - (* Add *)
    eapply Nat.le_trans.
    + apply support_size_add.
    + lia.

  - (* Mul *)
    (* Do NOT simpl here; we want eval_expr (Mul ...) to remain visible for rewrites. *)
    destruct e1 as [i1|q1|a1 b1|a1 b1|a1 b1];
    destruct e2 as [i2|q2|a2 b2|a2 b2|a2 b2].

    + (* Basis * Basis *)
      simpl. eapply Nat.le_trans; [apply support_size_le_2n | lia].

    + (* Basis * Scalar *)
      change (eval_expr sq (Basis i1) ⋆ eval_expr sq (Scalar q2))
        with (eval_expr sq (Mul (Basis i1) (Scalar q2))).
      rewrite (support_size_ext n
                (eval_expr sq (Mul (Basis i1) (Scalar q2)))
                (mv_scale q2 (eval_expr sq (Basis i1)))
                (eval_mul_scalar_r n sq q2 (Basis i1))).
      destruct (Qeq_dec q2 0) as [Hq0|Hq0].
      * assert (Hz : forall m, mv_scale q2 (eval_expr sq (Basis i1)) m == (@mv_zero n) m).
        { intro m. unfold mv_scale, mv_zero. rewrite Hq0. ring. }
        rewrite (support_size_ext n (mv_scale q2 (eval_expr sq (Basis i1))) (@mv_zero n) Hz).
        rewrite support_size_zero. simpl. lia.
      * rewrite (support_size_scale n q2 (eval_expr sq (Basis i1)) Hq0).
        simpl. exact IH1.

    + (* Basis * Add *)
      simpl. eapply Nat.le_trans; [apply support_size_le_2n | lia].

    + (* Basis * Mul *)
      simpl. eapply Nat.le_trans; [apply support_size_le_2n | lia].

    + (* Basis * Conv *)
      simpl. eapply Nat.le_trans; [apply support_size_le_2n | lia].

    + (* Scalar * Basis *)
      change (eval_expr sq (Scalar q1) ⋆ eval_expr sq (Basis i2))
        with (eval_expr sq (Mul (Scalar q1) (Basis i2))).
      rewrite (support_size_ext n
                (eval_expr sq (Mul (Scalar q1) (Basis i2)))
                (mv_scale q1 (eval_expr sq (Basis i2)))
                (eval_mul_scalar_l n sq q1 (Basis i2))).
      destruct (Qeq_dec q1 0) as [Hq0|Hq0].
      * assert (Hz : forall m, mv_scale q1 (eval_expr sq (Basis i2)) m == (@mv_zero n) m).
        { intro m. unfold mv_scale, mv_zero. rewrite Hq0. ring. }
        rewrite (support_size_ext n (mv_scale q1 (eval_expr sq (Basis i2))) (@mv_zero n) Hz).
        rewrite support_size_zero. simpl. lia.
      * rewrite (support_size_scale n q1 (eval_expr sq (Basis i2)) Hq0).
        simpl. exact IH2.
    
    + (* Scalar * Scalar *)
      change (eval_expr sq (Scalar q1) ⋆ eval_expr sq (Scalar q2))
        with (eval_expr sq (Mul (Scalar q1) (Scalar q2))).
      rewrite (support_size_ext n
                (eval_expr sq (Mul (Scalar q1) (Scalar q2)))
                (mv_scale q1 (eval_expr sq (Scalar q2)))
                (eval_mul_scalar_l n sq q1 (Scalar q2))).
      destruct (Qeq_dec q1 0) as [Hq0|Hq0].
      * assert (Hz : forall m, mv_scale q1 (eval_expr sq (Scalar q2)) m == (@mv_zero n) m).
        { intro m. unfold mv_scale, mv_zero. rewrite Hq0. ring. }
        rewrite (support_size_ext n (mv_scale q1 (eval_expr sq (Scalar q2))) (@mv_zero n) Hz).
        rewrite support_size_zero. simpl. lia.
      * rewrite (support_size_scale n q1 (eval_expr sq (Scalar q2)) Hq0).
        simpl. exact IH2.

    + (* Scalar * Add *)
      change (eval_expr sq (Scalar q1) ⋆ eval_expr sq (Cln_Grade.Add a2 b2))
        with (eval_expr sq (Mul (Scalar q1) (Cln_Grade.Add a2 b2))).
      rewrite (support_size_ext n
                (eval_expr sq (Mul (Scalar q1) (Cln_Grade.Add a2 b2)))
                (mv_scale q1 (eval_expr sq (Cln_Grade.Add a2 b2)))
                (eval_mul_scalar_l n sq q1 (Cln_Grade.Add a2 b2))).
      destruct (Qeq_dec q1 0) as [Hq0|Hq0].
      * assert (Hz : forall m, mv_scale q1 (eval_expr sq (Cln_Grade.Add a2 b2)) m == (@mv_zero n) m).
        { intro m. unfold mv_scale, mv_zero. rewrite Hq0. ring. }
        rewrite (support_size_ext n (mv_scale q1 (eval_expr sq (Cln_Grade.Add a2 b2))) (@mv_zero n) Hz).
        rewrite support_size_zero. simpl. lia.
      * rewrite (support_size_scale n q1 (eval_expr sq (Cln_Grade.Add a2 b2)) Hq0).
        simpl. exact IH2.

    + (* Scalar * Mul *)
      change (eval_expr sq (Scalar q1) ⋆ eval_expr sq (Mul a2 b2))
        with (eval_expr sq (Mul (Scalar q1) (Mul a2 b2))).
      rewrite (support_size_ext n
                (eval_expr sq (Mul (Scalar q1) (Mul a2 b2)))
                (mv_scale q1 (eval_expr sq (Mul a2 b2)))
                (eval_mul_scalar_l n sq q1 (Mul a2 b2))).
      destruct (Qeq_dec q1 0) as [Hq0|Hq0].
      * assert (Hz : forall m, mv_scale q1 (eval_expr sq (Mul a2 b2)) m == (@mv_zero n) m).
        { intro m. unfold mv_scale, mv_zero. rewrite Hq0. ring. }
        rewrite (support_size_ext n (mv_scale q1 (eval_expr sq (Mul a2 b2))) (@mv_zero n) Hz).
        rewrite support_size_zero. simpl. lia.
      * rewrite (support_size_scale n q1 (eval_expr sq (Mul a2 b2)) Hq0).
        simpl. exact IH2.

    + (* Scalar * Conv *)
      change (eval_expr sq (Scalar q1) ⋆ eval_expr sq (Conv a2 b2))
        with (eval_expr sq (Mul (Scalar q1) (Conv a2 b2))).
      rewrite (support_size_ext n
                (eval_expr sq (Mul (Scalar q1) (Conv a2 b2)))
                (mv_scale q1 (eval_expr sq (Conv a2 b2)))
                (eval_mul_scalar_l n sq q1 (Conv a2 b2))).
      destruct (Qeq_dec q1 0) as [Hq0|Hq0].
      * assert (Hz : forall m, mv_scale q1 (eval_expr sq (Conv a2 b2)) m == (@mv_zero n) m).
        { intro m. unfold mv_scale, mv_zero. rewrite Hq0. ring. }
        rewrite (support_size_ext n (mv_scale q1 (eval_expr sq (Conv a2 b2))) (@mv_zero n) Hz).
        rewrite support_size_zero. simpl. lia.
      * rewrite (support_size_scale n q1 (eval_expr sq (Conv a2 b2)) Hq0).
        simpl. exact IH2.

    (* remaining cases: neither side is Scalar => crude 2^n bound *)
    all: (simpl; eapply Nat.le_trans; [apply support_size_le_2n | lia]).

  + (* Conv *)
    eapply Nat.le_trans.
    * apply support_size_conv.
    * nia.
Qed.



Theorem eval_support_size_le :
  forall n (sq : Vector.t Q n) (e : GA_expr n),
    (support_size (eval_expr sq e) <= support_size_bound e)%nat.
Proof.
  intros n sq e.
  induction e as [i|c|e1 IH1 e2 IH2|e1 IH1 e2 IH2|e1 IH1 e2 IH2]; simpl.
  - (* Basis *)
    rewrite support_size_basis. lia.
  - (* Scalar *)
    destruct (Qeq_dec c 0) as [Hc0|Hc0].
    + (* zero scalar *)
      assert (He : forall m, mv_scale c (@mv_one n) m == (@mv_zero n) m).
      { intro m. unfold mv_scale, mv_zero. rewrite Hc0. ring. }
      rewrite (support_size_ext n (mv_scale c mv_one) mv_zero He).
      rewrite support_size_zero. lia.
    + (* nonzero scalar *)
      rewrite (support_size_scale n c mv_one Hc0).
      rewrite support_size_mv_one. lia.
  - (* Add *)
    eapply Nat.le_trans.
    + apply support_size_add.
    + lia.
  - (* Mul: crude bound by 2^n *)
    eapply Nat.le_trans.
    + apply support_size_le_2n.
    + lia.
  - (* Conv *)
    eapply Nat.le_trans.
    + apply support_size_conv.
    + nia.
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

Lemma support_size_bound_translate :
  forall n (phi : BoolFormula n),
    (support_size_bound (translate phi) <= Nat.pow 2 (formula_size phi))%nat.
Proof.
Admitted.

Corollary translate_support_size_bound :
  forall n (sq : Vector.t Q n) (phi : BoolFormula n),
    (support_size (eval_expr sq (translate phi))
     <= Nat.pow 2 (formula_size phi))%nat.
Proof.
Admitted.

        (* Lemma Block 5 *)

(* Parity: exactly 1 nonzero coefficient (the pseudoscalar) *)
Lemma support_size_XOR : forall n,
  (n > 0)%nat ->
  support_size (embed (@XOR_n_func n)) = 1%nat.
Proof.
Admitted.

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

(* Majority function on n variables (n odd):
   has Θ(2^n / √n) nonzero Fourier coefficients *)
(* This requires real work — skip for now *)

(* Inner product mod 2: IP(x,y) = ⊕ᵢ (xᵢ ∧ yᵢ)
   on 2n variables, has 2^n nonzero coefficients *)

Lemma support_size_IP : forall m,
  (m > 0)%nat ->
  support_size (embed (@IP_n_func (m + m))) = (Nat.pow 2 m)%nat.
Proof.
Admitted.

        (* The support-based separation theorem *)


Theorem IP_formula_size_lower_bound :
  forall m (sq : Vector.t Q (m + m)) (phi : BoolFormula (m + m)),
    (m > 0)%nat ->
    (forall i, Vector.nth sq i == 1) ->
    eval_bf phi = @IP_n_func (m + m) ->
    (formula_size phi >= m)%nat.
Proof.
  intros m sq phi Hm Hsq Hbf.
  (* From translate_correct + Hbf:
     eval_expr sq (translate phi) is pointwise == embed(IP) *)
  (* From translate_support_size_bound:
     support_size(eval_expr ...) ≤ 2^(formula_size phi) *)
  (* From support_size_IP:
     support_size(embed(IP)) = 2^m *)
  (* Need: support_size is preserved/reflected through translate_correct *)
  (* Combine: 2^m ≤ 2^(formula_size phi), so formula_size ≥ m *)
Admitted.
