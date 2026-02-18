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

        (* Support cardinality *)

(* Count nonzero coefficients *)
Definition support_size {n} (F : MV n) : nat :=
  length (List.filter (fun m => negb (Qeq_bool (F m) 0))
                 (all_masks n)).

(* The support set as a list *)
Definition support_list {n} (F : MV n) : list (Mask n) :=
  filter (fun m => negb (Qeq_bool (F m) 0)) (all_masks n).

        (* Lemma Block 1 *)

Lemma support_size_le_2n : forall n (F : MV n),
  (support_size F <= Nat.pow 2 n)%nat.
Proof.
Admitted.

Lemma support_size_zero : forall n,
  support_size (@mv_zero n) = 0%nat.
Proof.
Admitted.

Lemma support_size_basis : forall n (i : Fin.t n),
  support_size (basis (mask_single i)) = 1%nat.
Proof.
Admitted.

Lemma support_size_mv_one : forall n,
  support_size (@mv_one n) = 1%nat.
Proof.
Admitted.

Lemma support_size_scale : forall n (c : Q) (F : MV n),
  ~(c == 0) ->
  support_size (mv_scale c F) = support_size F.
Proof.
Admitted.

        (* Lemma Block 2 *)

(* Add can at most double support *)
Lemma support_size_add : forall n (F G : MV n),
  (support_size (mv_add F G) <= support_size F + support_size G)%nat.
Proof.
Admitted.

(* Conv: support of F⊙G ⊆ {A⊕B : A ∈ supp(F), B ∈ supp(G)} *)
(* So |supp(F⊙G)| ≤ |supp(F)| * |supp(G)| *)
Lemma support_size_conv : forall n (F G : MV n),
  (support_size (mv_conv F G) <= support_size F * support_size G)%nat.
Proof.
Admitted.

        (* Lemma Block 3 *)

(* Static support size bound from expression structure *)
Fixpoint support_size_bound {n} (e : GA_expr n) : nat :=
  match e with
  | Basis _    => 1
  | Scalar _   => 1
  | Cln_Grade.Add e1 e2  => support_size_bound e1 + support_size_bound e2
  | Mul e1 e2  =>
      match e1, e2 with
      | Scalar _, _ => support_size_bound e2
      | _, Scalar _ => support_size_bound e1
      | _, _        => Nat.pow 2 n  (* worst case *)
      end
  | Conv e1 e2 => support_size_bound e1 * support_size_bound e2
  end.

Theorem eval_support_size_le :
  forall n (sq : Vector.t Q n) (e : GA_expr n),
    (support_size (eval_expr sq e) <= support_size_bound e)%nat.
Proof.
Admitted.

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
