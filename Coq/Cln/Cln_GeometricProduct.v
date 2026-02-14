(*
## Phase 1: Complete `Cln_GeometricProduct.v` (prove all the `Admitted` lemmas)

### Tier 1 - Basic Properties (should be straightforward):
```coq
✗ mv_gp_add_l / mv_gp_add_r     (bilinearity - use sumQ_map_add)
✗ mv_gp_scale_l / mv_gp_scale_r (bilinearity - use sumQ_map_scale_l)
✗ mv_gp_one_l / mv_gp_one_r     (identity - mask_xor with empty)
✗ mv_gp_basis                   (closed form - one term survives)
```

### Tier 2 - Clifford Relations (medium difficulty):
```coq
✗ e_square       (e_i * e_i = sq_i)
✗ e_anticomm     (e_i * e_j = -e_j * e_i when i≠j)
```

### Tier 3 - The Big One (hard!):
```coq
✗ basis_mul_assoc_coeff  (cocycle identity)
✗ mv_gp_assoc           (full associativity)
```

---

## Phase 2: Composition Impossibility

**New file: `Cln_CompositionFailure.v`**

```coq
(* Key theorem: geometric product ≠ functional composition *)
Theorem geom_prod_not_composition :
  forall n (sq : Vector.t Q n),
    n >= 1 ->
    exists (f g : Corner n -> bool),
      (* Geometric product of embeddings *)
      mv_gp n sq (embed f) (embed g)
      ≠
      (* Embedding of composition *)
      embed (fun s => f (apply_corner g s)).
      
(* Specific counterexample: AND squared *)
Theorem AND_geom_square_not_boolean :
  forall n sq,
    n >= 2 ->
    exists s : Corner n,
      let F := embed (corner_and n) in  (* generalized AND *)
      eval (mv_gp n sq F F) s ∉ {0, 1}.
```

---

## Phase 3: Characterize Geometric Product (the mystery)

**New file: `Cln_GeometricAnalysis.v`**

### 3A. geom_square Support Theorem
```coq
Definition geom_square {n} sq (F : MV n) : MV n := mv_gp n sq F F.

Definition geom_square_support {n} sq (f : Corner n -> bool) : nat :=
  (* count nonzero evaluations of geom_square(embed f) *)
  
Theorem single_variable_support_theorem :
  forall n sq f,
    depends_on_single_variable n f ->
    geom_square_support sq f = 2^(n-1).
    
Theorem parity_full_support :
  forall n sq,
    geom_square_support sq (parity_n n) = 2^n.
```

### 3B. Grade Structure Theorem
```coq
(* Parity lives in top grade *)
Theorem parity_top_grade :
  forall n sq,
    let F := embed (parity_n n) in
    (* Only the n-vector component is nonzero *)
    forall m : Mask n,
      grade n m ≠ n -> F m == 0.
      
(* Single-variable functions use only grade 0 and 1 *)
Theorem single_var_low_grade :
  forall n sq f,
    depends_on_single_variable n f ->
    forall m : Mask n,
      grade n m > 1 -> embed f m == 0.
```

### 3C. Fourier vs Geometric Duality
```coq
(* Fourier support measures algebraic complexity *)
(* geom_square support measures geometric complexity *)
(* They're DIFFERENT! *)

Theorem fourier_geom_duality :
  forall n sq f,
    fourier_support f ≠ geom_square_support sq f
    (for most f).
```

---

## Phase 4: Algebraic Structure

**Extend `Cln_GeometricProduct.v` or new file:**

```coq
(* Embedding is injective *)
Theorem embed_injective :
  forall n (f g : Corner n -> bool),
    embed f = embed g -> 
    (forall s, f s = g s).
    
(* Dimension of Boolean embedding space *)
Theorem embed_dimension :
  forall n,
    (* 2^n Boolean functions embed into 2^n dimensional space *)
    (* but they span only a special subspace! *)
    dimension_of_span (image embed) = 2^n.
    
(* NOT a subalgebra *)
Theorem embed_not_subalgebra :
  forall n sq,
    n >= 1 ->
    ~ (forall f g, 
        exists h, mv_gp n sq (embed f) (embed g) = embed h).
```

---

## Priority Order (if you want to do this systematically):

1. **Phase 1, Tier 1** (bilinearity + identity) - straightforward, mechanical
2. **Phase 2** (composition failure) - proves the KEY negative result
3. **Phase 1, Tier 2** (Clifford relations) - shows it's a real Clifford algebra
4. **Phase 3A** (support theorems) - characterizes what geom_prod computes
5. **Phase 1, Tier 3** (associativity) - completes the algebra structure
6. **Phase 3B-C + Phase 4** (deep theory) - research-level questions

The **composition impossibility** (Phase 2) is probably the most important philosophically - it answers "why doesn't this work for computation?"
*)

(*
  ============================================================
  File: Cln_GeometricProduct.v
  ============================================================

  Clifford algebra structure on MV n := Mask n -> Q.

  Key idea on basis blades:
      e_A * e_B = sgn(A,B) * met(A,B) * e_(A xor B)

  - xor picks the resulting blade index
  - sgn encodes anti-commutation (swap parity)
  - met encodes the quadratic form (signature): e_i^2 = sq_i

  This gives an associative unital algebra with Clifford relations.
*)

Require Import Cln_Basis.
Require Import Cln_Multivector.

From Coq Require Import FunctionalExtensionality.

From Coq Require Import Lia.
From Coq Require Import List Bool Arith QArith Vectors.Vector.
From Coq Require Import Setoid Morphisms Ring.
Import ListNotations.

From Coq Require Import Vectors.Vector Bool.
Import VectorNotations.

From Coq Require Import QArith.Qring.
Open Scope Q_scope.
Set Implicit Arguments.

(* ============================================================ *)
(* Mask operations: xor, and, empty, singleton, parity            *)
(* ============================================================ *)

Definition mask_xor {n} (a b : Mask n) : Mask n :=
  Vector.map2 xorb a b.

Definition mask_and {n} (a b : Mask n) : Mask n :=
  Vector.map2 andb a b.

Definition mask_empty {n} : Mask n := Vector.const false n.
  
(* parity of number of true bits (grade parity) *)

Definition grade_parity {n} (m : Mask n) : bool :=
  List.fold_right xorb false (Vector.to_list m).

(* ============================================================ *)
(* Swap parity: (-1)^(# { (i in A, j in B) | j < i })             *)
(* ============================================================ *)

(*
  Recurrence:
    swaps(A,B) = swaps(tail A, tail B) XOR ( (head B) AND odd(tail A) )

  Reason: the only “new” crossings created by stripping heads are:
  pairs where j is the head of B (so j is earlier than every tail index),
  and i ranges over true bits in tail A.
*)
Definition swaps_parity {n} (a b : Mask n) : bool :=
  fst (List.fold_right
         (fun ab st =>
            let '(ai, bi) := ab in
            let '(s, p) := st in
            (xorb s (andb bi p), xorb ai p))
         (false, false)
         (List.combine (Vector.to_list a) (Vector.to_list b))).

Definition sgnQ (b : bool) : Q := if b then (-1)%Q else 1%Q.

(* ============================================================ *)
(* Metric factor for Cl(p,q) via squares vector                   *)
(* ============================================================ *)

(*
  sq : Vector.t Q n  with each entry = +1 or -1 (typically)
  metric_factor(A,B) = ∏_{i where A_i && B_i} sq_i

  This is what turns repeated generators into scalars:
     e_i e_i = sq_i
*)
Definition metric_factor {n} (sq : Vector.t Q n) (a b : Mask n) : Q :=
  List.fold_right Qmult 1%Q
    (List.map (fun '(sq_i, ab) =>
                 let '(ai, bi) := ab in
                 if andb ai bi then sq_i else 1%Q)
      (List.combine (Vector.to_list sq)
        (List.combine (Vector.to_list a) (Vector.to_list b)))).

(* ============================================================ *)
(* Basis-blade multiplication payload                             *)
(* ============================================================ *)
Definition basis_mul_coeff {n} (sq : Vector.t Q n) (A B : Mask n) : Q :=
  (sgnQ (swaps_parity A B) * metric_factor sq A B)%Q.

Definition basis_mul_mask {n} (A B : Mask n) : Mask n := mask_xor A B.

(* ============================================================ *)
(* Geometric product on multivectors                              *)
(* ============================================================ *)

(*
  (F ⋆ G)(U) = Σ_A Σ_B  F(A)*G(B)*coeff(A,B)*[xor(A,B)=U]
*)
Definition mv_gp (n : nat) (sq : Vector.t Q n) (F G : MV n) : MV n :=
  fun U =>
    sumQ (List.map (fun A : Mask n =>
      sumQ (List.map (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * G B * c)%Q else 0%Q
      ) (all_masks n))
    ) (all_masks n)).


Infix "⋆" := (mv_gp _ ) (at level 40). (* usage: (mv_gp n sq F G) *)
(*This won't work as expected*)
Notation "F ⋆[ n , sq ] G" := (mv_gp n sq F G) (at level 40).
(*
OR

Section GP.
  Context {n : nat} (sq : Vector.t Q n).
  Infix "⋆" := (mv_gp n sq) (at level 40).
End GP.

That avoids the “partial application to _” trap.
*)


(* Scalar 1 (the empty blade) *)
Definition mv_one {n} : MV n := basis (mask_empty (n:=n)).

(* ============================================================ *)
(* Basic algebra laws: bilinear + identity                        *)
(* ============================================================ *)
From Coq Require Import FunctionalExtensionality.

Require Import QArith.
Require Import Qcanon.

Lemma Qmult_plus_distr_r_eq : forall x y z : Q,
  (x + y) * z == x * z + y * z.
Proof.
  intros. ring.
Qed.

Lemma Qmult_assoc_eq : forall x y z : Q,
  (x * y) * z == x * (y * z).
Proof.
  intros. ring.
Qed.

Lemma mv_gp_add_l :
  forall n (sq : Vector.t Q n) (F1 F2 G : MV n) (U : Mask n),
    @mv_gp n sq (mv_add F1 F2) G U
    ==
    mv_add (@mv_gp n sq F1 G)
           (@mv_gp n sq F2 G) U.
Proof.
  intros n sq F1 F2 G U.
  unfold mv_gp, mv_add.

  (* Define the two “inner sums” as functions of A *)
  set (inner1 :=
    fun A : Mask n =>
      sumQ (List.map (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F1 A * G B * c)%Q else 0%Q
      ) (all_masks n))).

  set (inner2 :=
    fun A : Mask n =>
      sumQ (List.map (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F2 A * G B * c)%Q else 0%Q
      ) (all_masks n))).

  (* LHS == sumQ(map (fun A => inner1 A + inner2 A) all_masks) *)
  eapply Qeq_trans.
  2: {
    (* now split the outer sum *)
    exact (@sumQ_map_add (Mask n) inner1 inner2 (all_masks n)).
  }

  apply sumQ_map_ext.
  intros A HA.
  subst inner1 inner2.

  (* Inside: split the B-sum *)
  eapply Qeq_trans.
  2: {
    exact (@sumQ_map_add (Mask n)
      (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F1 A * G B * c)%Q else 0%Q)
      (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F2 A * G B * c)%Q else 0%Q)
      (all_masks n)).
  }

  (* Pointwise: (F1+F2) term equals term1 + term2 *)
  apply sumQ_map_ext.
  intros B HB.
  destruct (mask_eq_dec (basis_mul_mask A B) U); simpl; ring.
Qed.

Lemma mv_gp_add_r :
  forall n (sq : Vector.t Q n) (F G1 G2 : MV n) (U : Mask n),
    @mv_gp n sq F (mv_add G1 G2) U
    ==
    mv_add (@mv_gp n sq F G1)
           (@mv_gp n sq F G2) U.
Proof.
  intros n sq F G1 G2 U.
  unfold mv_gp, mv_add.

  (* Define the two A-indexed inner sums (with G1 and G2 separately) *)
  set (inner1 :=
    fun A : Mask n =>
      sumQ (List.map (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * G1 B * c)%Q else 0%Q
      ) (all_masks n))).

  set (inner2 :=
    fun A : Mask n =>
      sumQ (List.map (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * G2 B * c)%Q else 0%Q
      ) (all_masks n))).

  (* Goal: outer sum with (G1+G2) == sumQ(map inner1) + sumQ(map inner2).
     We’ll rewrite LHS into sumQ(map (fun A => inner1 A + inner2 A)),
     then split with sumQ_map_add. *)
  eapply Qeq_trans.
  2: { exact (@sumQ_map_add (Mask n) inner1 inner2 (all_masks n)). }

  (* Rewrite the outer map pointwise in A. *)
  apply sumQ_map_ext.
  intros A HA.
  subst inner1 inner2.

  (* Now show the B-sum distributes: *)
  eapply Qeq_trans.
  2: {
    exact (@sumQ_map_add (Mask n)
      (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * G1 B * c)%Q else 0%Q)
      (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * G2 B * c)%Q else 0%Q)
      (all_masks n)).
  }

  (* First rewrite the mapped term into (t1 + t2) pointwise, then ring. *)
  apply sumQ_map_ext.
  intros B HB.
  destruct (mask_eq_dec (basis_mul_mask A B) U); simpl; ring.
Qed.

Lemma mv_gp_scale_l :
  forall n (sq : Vector.t Q n) (k : Q) (F G : MV n) (U : Mask n),
    @mv_gp n sq (mv_scale k F) G U
    ==
    mv_scale k (@mv_gp n sq F G) U.
Proof.
  intros n sq k F G U.
  unfold mv_gp, mv_scale.

  (* Define the “base” inner sum without the k factor *)
  set (inner_base :=
    fun (A : Mask n) =>
      sumQ (List.map (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * G B * c)%Q else 0%Q
      ) (all_masks n))).

  (* Step 1: show the outer mapped function equals (fun A => k * inner_base A) *)
  eapply Qeq_trans.
  2: {
    (* Step 2: pull k out of the outer sum *)
    unfold mv_scale.
    exact (@sumQ_map_scale_l (Mask n) k inner_base (all_masks n)).
  }

  (* Prove: sumQ(map outer_with_k) == sumQ(map (fun A => k * inner_base A)) *)
  apply sumQ_map_ext.
  intros A HA.
  subst inner_base.

  (* Now work on the inner B-sum for this A *)
  eapply Qeq_trans.
  2: {
    (* Pull k out of the inner sum over B *)
    exact (@sumQ_map_scale_l (Mask n) k
      (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * G B * c)%Q else 0%Q)
      (all_masks n)).
  }

  (* Pointwise: the term with (k * F A) matches k * (term without k) *)
  apply sumQ_map_ext.
  intros B HB.
  destruct (mask_eq_dec (basis_mul_mask A B) U); simpl; ring.
Qed.

Lemma mv_gp_scale_r :
  forall n (sq : Vector.t Q n) (k : Q) (F G : MV n) (U : Mask n),
    @mv_gp n sq F (mv_scale k G) U
    ==
    mv_scale k (@mv_gp n sq F G) U.
Proof.
  intros n sq k F G U.
  unfold mv_gp, mv_scale.

  (* base inner sum (without the k factor) *)
  set (inner_base :=
    fun (A : Mask n) =>
      sumQ (List.map (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * G B * c)%Q else 0%Q
      ) (all_masks n))).

  (* Rewrite outer sum into sumQ(map (fun A => k * inner_base A) ...),
     then pull k out of the outer sum. *)
  eapply Qeq_trans.
  2: {
    exact (@sumQ_map_scale_l (Mask n) k inner_base (all_masks n)).
  }

  (* Show the outer map matches k * inner_base pointwise *)
  apply sumQ_map_ext.
  intros A HA.
  subst inner_base.

  (* Now, for each A, rewrite the inner B-sum into k * (base inner sum),
     then pull k out via sumQ_map_scale_l. *)
  eapply Qeq_trans.
  2: {
    exact (@sumQ_map_scale_l (Mask n) k
      (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * G B * c)%Q
        else 0%Q)
      (all_masks n)).
  }

  (* Pointwise ring normalization under the if *)
  apply sumQ_map_ext.
  intros B HB.
  destruct (mask_eq_dec (basis_mul_mask A B) U); simpl; ring.
Qed.

Require Import Coq.Program.Equality.

(* --- XOR with empty mask --- *)
Lemma mask_xor_empty_l :
  forall n (B : Mask n),
    mask_xor (mask_empty (n:=n)) B = B.
Proof.
  induction n; intro B.
  - dependent destruction B. reflexivity.
  - dependent destruction B.
    simpl [mask_xor mask_empty].
    (* mask_empty = false :: ... ; xorb false h = h *)
    simpl. f_equal. apply IHn.
Qed.

Lemma swaps_parity_empty_l_aux :
  forall n (B : Mask n),
    List.fold_right
      (fun ab st =>
         let '(ai, bi) := ab in
         let '(s, p) := st in
         (xorb s (andb bi p), xorb ai p))
      (false, false)
      (List.combine (Vector.to_list (Vector.const false n)) (Vector.to_list B))
    =
    (false, false).
Proof.
  induction n as [|n IH]; intro B.
  - dependent destruction B. simpl. reflexivity.
  - dependent destruction B. simpl.
    (* Now B is (Vector.cons _ h _ B0) for some h,B0, and simpl exposes it. *)
    (* fold back the unfolded Vector.to_list terms so IH matches *)
    fold (Vector.to_list (Vector.const false n)).
    fold (Vector.to_list B).
    rewrite IH.
    simpl. rewrite Bool.andb_false_r. reflexivity.
Qed.

Lemma swaps_parity_empty_l :
  forall n (B : Mask n),
    swaps_parity (Vector.const false n) B = false.
Proof.
  intros n B.
  unfold swaps_parity.
  (* swaps_parity is fst of that fold *)
  rewrite swaps_parity_empty_l_aux.
  reflexivity.
Qed.

Lemma metric_factor_empty_l :
  forall n (sq : Vector.t Q n) (B : Mask n),
    metric_factor sq (mask_empty (n:=n)) B == 1%Q.
Proof.
  induction n as [|n IH]; intros sq B.
  - dependent destruction sq.
    dependent destruction B.
    simpl. reflexivity.
  - dependent destruction sq.
    dependent destruction B.
    (* Now expand metric_factor just one step, but control simplification. *)
    unfold metric_factor.
    unfold mask_empty.
    (* mask_empty = const false (S n) = cons false (const false n) *)
    simpl.
    (* After simpl, the list being folded starts with factor = 1, since ai=false *)
    (* fold_right Qmult 1 (1 :: rest) == 1 * fold_right ... rest *)
    simpl.

    (* The remaining fold_right/map/combine is exactly the n-case: *)
    (* We want to rewrite it to metric_factor sqt (const false n) bt *)
    (* Instead of folding to_list, just re-expand metric_factor on the tail: *)
    change (List.fold_right Qmult 1%Q
      (List.map
        (fun '(sq_i, (ai, bi)) => if ai && bi then sq_i else 1%Q)
        (List.combine (to_list sq)
          (List.combine (to_list (Vector.const false n)) (to_list B)))))
    with (metric_factor sqt (Vector.const false n) bt).

    (* Now use IH *)
    rewrite (IH sq B).
    ring.
Qed.

Lemma basis_mul_coeff_empty_l :
  forall n (sq : Vector.t Q n) (B : Mask n),
    basis_mul_coeff sq (mask_empty (n:=n)) B == 1%Q.
Proof.
  intros n sq B.
  unfold basis_mul_coeff.
  rewrite swaps_parity_empty_l.
  unfold sgnQ. simpl.
  rewrite metric_factor_empty_l.
  ring.
Qed.

(* --- “Kronecker delta sum” over all_masks ---
   sum_{m in all_masks n} (if m=U then f m else 0) == f U
*)
Local Opaque mask_eq_dec.

Lemma sumQ_all_masks_pick :
  forall n (f : Mask n -> Q) (U : Mask n),
    sumQ (List.map (fun m => if mask_eq_dec m U then f m else 0%Q) (all_masks n))
    == f U.
Proof.
  induction n as [|n IH]; intros f U.
  (* n = 0 case *)
  - dependent destruction U.
  cbn [all_masks sumQ List.map].   (* IMPORTANT: includes List.map *)
  (* goal is now: sumQ [if mask_eq_dec [] [] then f [] else 0] == f [] *)
  cbn [sumQ].                      (* sumQ [x] = x + 0 *)
  destruct (mask_eq_dec ([] : Mask 0) ([] : Mask 0)) as [Heq|Hneq].
  + cbn.                           (* if left Heq then f[] else 0  ==> f[] *)
    rewrite Qplus_0_r.
    apply Qeq_refl.
  + exfalso; apply Hneq; reflexivity.
  
  (* n = S n case *)
  - dependent destruction U.
    rename h into Uh.
    rename U into Ut.
    simpl [all_masks].
    
    rewrite map_app.
    rewrite sumQ_app.

    destruct Uh.

    + (* Uh = true *)
      (* left half = 0 *)
      
      
      assert (Hleft :
        sumQ
          (List.map
             (fun m => if mask_eq_dec m (true :: Ut) then f m else 0%Q)
             (List.map (fun t => false :: t) (all_masks n)))
        == 0%Q).
      {
        (* Turn RHS into a sumQ of zeros so sumQ_map_ext applies *)
        eapply Qeq_trans.
        2: {
          apply (sumQ_map_const0
                   (List.map (fun t => false :: t) (all_masks n))).
        }

        apply sumQ_map_ext; intros m Hm.
        (* show each term equals 0 *)
        apply (proj1 (in_map_iff' (fun t => false :: t) m (all_masks n))) in Hm.
        destruct Hm as [t [Ht_in Ht_eq]]; subst m.

        destruct (mask_eq_dec (false :: t) (true :: Ut)) as [Heq|Hneq].
        - inversion Heq.
        - cbn. apply Qeq_refl.
      }
      
      
      rewrite Hleft.
      rewrite Qplus_0_l.

      (* right half reduces to IH on tails *)
      eapply Qeq_trans.
      2: exact (IH (fun t => f (true :: t)) Ut).

      (* rewrite the LHS so the list is exactly (all_masks n) *)
      rewrite List.map_map.
      cbn.

      apply sumQ_map_ext; intros t Ht.
      destruct (mask_eq_dec t Ut) as [HtEq|HtNeq].
      * subst t.
        destruct (mask_eq_dec (true :: Ut) (true :: Ut)) as [_|Hbad].
        { cbn. apply Qeq_refl. }
        { exfalso; apply Hbad; reflexivity. }
      * destruct (mask_eq_dec (true :: t) (true :: Ut)) as [Heq|Hneq'].
        { exfalso.
          apply HtNeq.
          dependent destruction Heq.
          reflexivity.
        }
        { cbn. apply Qeq_refl. }

    + (* Uh = false *)
      (* right half = 0 *)
      assert (Hright :
        sumQ
          (List.map
             (fun m => if mask_eq_dec m (false :: Ut) then f m else 0%Q)
             (List.map (fun t => true :: t) (all_masks n)))
        == 0%Q).
      {
        eapply Qeq_trans.
        2: {
          apply (sumQ_map_const0
                   (List.map (fun t => true :: t) (all_masks n))).
        }

        apply sumQ_map_ext; intros m Hm.
        apply (proj1 (in_map_iff' (fun t => true :: t) m (all_masks n))) in Hm.
        destruct Hm as [t [Ht_in Ht_eq]]; subst m.

        destruct (mask_eq_dec (true :: t) (false :: Ut)) as [Heq|Hneq].
        - inversion Heq.
        - cbn. apply Qeq_refl.
      }

      rewrite Hright.
      rewrite Qplus_0_r.

      (* left half reduces to IH on tails *)
      eapply Qeq_trans.
      2: exact (IH (fun t => f (false :: t)) Ut).

      rewrite List.map_map.
      cbn.

      apply sumQ_map_ext; intros t Ht.
      destruct (mask_eq_dec t Ut) as [HtEq|HtNeq].
      * subst t.
        destruct (mask_eq_dec (false :: Ut) (false :: Ut)) as [_|Hbad].
        { cbn. apply Qeq_refl. }
        { exfalso; apply Hbad; reflexivity. }
      * destruct (mask_eq_dec (false :: t) (false :: Ut)) as [Heq|Hneq'].
        { exfalso.
          apply HtNeq.
          dependent destruction Heq.
          reflexivity.
        }
        { cbn. apply Qeq_refl. }

Qed.

Lemma mv_gp_one_l :
  forall n (sq : Vector.t Q n) (F : MV n) (U : Mask n),
    @mv_gp n sq mv_one F U == F U.
Proof.
  intros n sq F U.
  unfold mv_gp, mv_one, basis.

  (* outer(A) = Σ_B [A xor B = U] * (delta_{A=empty}) * F(B) * coeff(A,B) *)
  set (outer :=
    fun A : Mask n =>
      sumQ (List.map (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then ((if mask_eq_dec A (mask_empty (n:=n)) then 1%Q else 0%Q) * F B * c)%Q
        else 0%Q
      ) (all_masks n))).

  (* Rewrite whole thing into Σ_A outer(A) *)
  change
    (sumQ
      (List.map
        (fun A : Mask n =>
          sumQ
            (List.map
              (fun B : Mask n =>
                let c := basis_mul_coeff sq A B in
                if mask_eq_dec (basis_mul_mask A B) U
                then ((if mask_eq_dec A (mask_empty (n:=n)) then 1%Q else 0%Q) * F B * c)%Q
                else 0%Q)
              (all_masks n)))
        (all_masks n)) == F U).
  fold outer.

  (* 1) For A ≠ empty, outer A = 0 *)
  assert (Houter0 : forall A : Mask n, A <> mask_empty -> outer A == 0%Q).
  {
    intros A Hne.
    unfold outer.

    eapply Qeq_trans.
    - (* force g := 0 to avoid ?g *)
      apply (@sumQ_map_ext (Mask n)
        (fun B : Mask n =>
           let c := basis_mul_coeff sq A B in
           if mask_eq_dec (basis_mul_mask A B) U
           then ((if mask_eq_dec A mask_empty then 1 else 0) * F B * c)%Q
           else 0%Q)
        (fun _ : Mask n => 0%Q)
        (all_masks n)).


      intros B HB.
      destruct (mask_eq_dec (basis_mul_mask A B) U) as [HAB|HAB]; simpl.
      + destruct (mask_eq_dec A mask_empty) as [Heq|Hneq].
        * exfalso; exact (Hne Heq).
        * (* 0 * F B * c == 0 *)
          ring.
      + (* 0 == 0 *)
        apply Qeq_refl.
    - (* sumQ (map (fun _ => 0) ...) == 0 *)
      exact (sumQ_map_const0 (A:=Mask n) (all_masks n)).
  }

  (* 2) Replace Σ_A outer(A) by the guarded sum that sumQ_all_masks_pick expects *)
  set (E := mask_empty (n:=n)).
  set (guarded :=
    fun A : Mask n =>
    
      if mask_eq_dec A E then outer A else 0%Q).
    (*if mask_eq_dec A (mask_empty (n:=n)) then outer A else 0%Q).*)
  
  eapply Qeq_trans.
    - (* pointwise: outer A == guarded A *)
      apply (@sumQ_map_ext (Mask n) outer guarded (all_masks n)).
      intros A HA.
      unfold guarded.
      destruct (mask_eq_dec A E) as [Heq|HneqE].
      + (* A = E *)
        subst A. apply Qeq_refl.
      + (* A <> E : guarded A = 0 *)
        exact (Houter0 A HneqE).
    - (* now apply the pick lemma on A, picking empty *)
      eapply Qeq_trans.
      + (* guarded has the pick shape *)
        (* guarded A = if A=E then outer A else 0 *)
        (* so this is exactly sumQ_all_masks_pick with U:=E *)
        exact (@sumQ_all_masks_pick n outer E).
      + (* outer(E) == F U *)
        subst E.
        unfold outer.

        (* simplify the delta (if empty=empty then 1 else 0) *)
        destruct (mask_eq_dec mask_empty mask_empty) as [_|Hbad].
        2:{ exfalso; apply Hbad; reflexivity. }
        cbn.

        (* rewrite the term to (if B=U then F B else 0) *)
        eapply Qeq_trans with
          (y := sumQ (List.map (fun B : Mask n =>
                   if mask_eq_dec B U then F B else 0%Q) (all_masks n))).
        * (* goal 1: rewrite the sum to the (if B=U then F B else 0) form *)
          apply (@sumQ_map_ext (Mask n)
                   (fun B : Mask n =>
                      if mask_eq_dec (basis_mul_mask mask_empty B) U
                      then (1%Q * F B * basis_mul_coeff sq mask_empty B)%Q
                      else 0%Q)
                   (fun B : Mask n =>
                      if mask_eq_dec B U then F B else 0%Q)
                   (all_masks n)).
          
          intros B HB.
          unfold basis_mul_mask.
          rewrite (mask_xor_empty_l (n:=n) B).

          destruct (mask_eq_dec B U) as [Heq|Hneq]; simpl.
          { (* B = U *)
            (* Goal: 1 * F B * basis_mul_coeff sq mask_empty B == F B *)

            (* Step 1: replace the rightmost factor using basis_mul_coeff_empty_l *)
            eapply Qeq_trans with (y := (1%Q * F B * 1%Q)%Q).
            { (* show: 1 * F B * coeff == 1 * F B * 1 *)
              (* use compatibility on the RIGHT factor of the outer multiplication *)
              apply Qmult_comp.
              - apply Qeq_refl.   (* left factor: 1 * F B *)
              - exact (basis_mul_coeff_empty_l (n:=n) sq B).
            }
            { (* Step 2: 1*F B*1 == F B *)
              ring.
            }
          }
          { (* B <> U *)
            apply Qeq_refl.
          }
          (*sumQ (List.map (fun B : Mask n => if mask_eq_dec B U then F B else 0) (all_masks n)) == F U*)
          * exact (@sumQ_all_masks_pick n F U).
Qed.


Lemma mask_xor_empty_r :
  forall n (A : Mask n),
    mask_xor A (mask_empty (n:=n)) = A.
Proof.
  induction n as [|n IH]; intro A.
  - dependent destruction A. reflexivity.
  - dependent destruction A.
    simpl [mask_xor mask_empty]. simpl.
    f_equal.
    + (* head bit *)
      destruct h; reflexivity.   (* xorb true false = true, xorb false false = false *)
    + (* tail *)
      apply IH.
Qed.

Lemma swaps_parity_empty_r_aux_fst :
  forall n (A : Mask n),
    fst
      (List.fold_right
         (fun ab st =>
            let '(ai, bi) := ab in
            let '(s, p) := st in
            (xorb s (andb bi p), xorb ai p))
         (false, false)
         (List.combine (Vector.to_list A)
                       (Vector.to_list (Vector.const false n))))
    = false.
Proof.
  induction n as [|n IH]; intro A.
  - dependent destruction A. simpl. reflexivity.
  - dependent destruction A. simpl.
    fold (Vector.to_list A).
    fold (Vector.to_list (Vector.const false n)).
    (* at this point bi = false has already reduced (bi && p) to false *)
    simpl.                    (* fst of the let/pair *)
    destruct (List.fold_right
      (fun ab st : bool * bool =>
         let '(ai, bi) := ab in
         let '(s, p) := st in (xorb s (bi && p), xorb ai p))
      (false, false)
      (combine (to_list A) (to_list (const false n))))
      as [s p] eqn:Hs.
    cbn.                       (* fst (xorb s false, ...) -> xorb s false *)
    rewrite Bool.xorb_false_r.
    (* goal becomes: s = false *)
    (* and IH, rewritten using Hs, gives exactly that *)
    specialize (IH A).
    rewrite Hs in IH.
    exact IH.
Qed.

Lemma swaps_parity_empty_r :
  forall n (A : Mask n),
    swaps_parity A (Vector.const false n) = false.
Proof.
  intros n A.
  unfold swaps_parity.
  apply swaps_parity_empty_r_aux_fst.
Qed.

Lemma metric_factor_empty_r :
  forall n (sq : Vector.t Q n) (A : Mask n),
    metric_factor sq A (mask_empty (n:=n)) == 1%Q.
Proof.
  induction n as [|n IH]; intros sq A.
  - dependent destruction sq.
    dependent destruction A.
    simpl. reflexivity.
  - dependent destruction sq.
    dependent destruction A.
    unfold metric_factor.
    unfold mask_empty.
    simpl.
    simpl.
    change (List.fold_right Qmult 1%Q
      (List.map
        (fun '(sq_i, (ai, bi)) => if ai && bi then sq_i else 1%Q)
        (List.combine (to_list sq)
          (List.combine (to_list A) (to_list (Vector.const false n))))))
    with (metric_factor sqt A (Vector.const false n)).
    rewrite (IH sq A).
    rewrite Bool.andb_false_r.
    simpl.
    ring.
Qed.

Lemma basis_mul_coeff_empty_r :
  forall n (sq : Vector.t Q n) (A : Mask n),
    basis_mul_coeff sq A (mask_empty (n:=n)) == 1%Q.
Proof.
  intros n sq A.
  unfold basis_mul_coeff.
  rewrite swaps_parity_empty_r.
  unfold sgnQ. simpl.
  rewrite metric_factor_empty_r.
  ring.
Qed.

Lemma mv_gp_one_r :
  forall n (sq : Vector.t Q n) (F : MV n) (U : Mask n),
    @mv_gp n sq F (@mv_one n) U == F U.
Proof.
  intros n sq F U.
  unfold mv_gp, mv_one, basis.

  (* inner(A) = Σ_B [A xor B = U] * F(A) * (delta_{B=empty}) * coeff(A,B) *)
  set (inner :=
    fun A : Mask n =>
      sumQ (List.map (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * (if mask_eq_dec B (mask_empty (n:=n)) then 1%Q else 0%Q) * c)%Q
        else 0%Q
      ) (all_masks n))).

  (* Rewrite whole thing into Σ_A inner(A) *)
  change
    (sumQ
      (List.map
        (fun A : Mask n =>
          sumQ
            (List.map
              (fun B : Mask n =>
                let c := basis_mul_coeff sq A B in
                if mask_eq_dec (basis_mul_mask A B) U
                then (F A * (if mask_eq_dec B (mask_empty (n:=n)) then 1%Q else 0%Q) * c)%Q
                else 0%Q)
              (all_masks n)))
        (all_masks n)) == F U).
  fold inner.

  (* 1) For B ≠ empty, the B-term is 0 (so inner A is a B-pick) *)
  (* We'll do exactly the same “guarded” trick, but now in the INNER sum. *)
  eapply Qeq_trans.
  - (* rewrite each A-summand inner(A) into (if A=U then F A else 0) *)
    apply (@sumQ_map_ext (Mask n)
      inner
      (fun A : Mask n => if mask_eq_dec A U then F A else 0%Q)
      (all_masks n)).
    intros A HA.
    unfold inner.

    (* gB(B) is the mapped term in the inner sum for this fixed A *)
    set (gB :=
      fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (F A * (if mask_eq_dec B (mask_empty (n:=n)) then 1%Q else 0%Q) * c)%Q
        else 0%Q).

    (* show gB B = 0 when B ≠ empty *)
    assert (HgB0 : forall B : Mask n, B <> mask_empty -> gB B == 0%Q).
    {
      intros B Hne.
      unfold gB.
      destruct (mask_eq_dec (basis_mul_mask A B) U); simpl; [| apply Qeq_refl].
      destruct (mask_eq_dec B (mask_empty (n:=n))) as [Heq|Hneq].
      - exfalso; exact (Hne Heq).
      - ring.
    }

    (* guard the inner sum so sumQ_all_masks_pick applies *)
    set (E := mask_empty (n:=n)).
    set (guardB := fun B : Mask n => if mask_eq_dec B E then gB B else 0%Q).

    eapply Qeq_trans.
    + (* pointwise: gB B == guardB B *)
      apply (@sumQ_map_ext (Mask n) gB guardB (all_masks n)).
      intros B HB.
      unfold guardB.
      destruct (mask_eq_dec B E) as [Heq|Hneq].
      * subst B; apply Qeq_refl.
      * exact (HgB0 B Hneq).
    + (* pick B=E *)
      eapply Qeq_trans.
      * exact (@sumQ_all_masks_pick n gB E).
      * (* compute gB(empty) and show it equals if A=U then F A else 0 *)
        subst E.
        unfold gB.
        destruct (mask_eq_dec (mask_empty (n:=n)) (mask_empty (n:=n))) as [_|Hbad].
        2:{ exfalso; apply Hbad; reflexivity. }
        cbn.

        unfold basis_mul_mask.
        rewrite (mask_xor_empty_r (n:=n) A).
        destruct (mask_eq_dec A U) as [HeqAU|HneqAU]; simpl.
        -- (* A = U: reduce coeff(A,empty)=1 *)
           eapply Qeq_trans with (y := (F A * 1%Q * 1%Q)%Q).
           { apply Qmult_comp.
             - apply Qeq_refl.
             - exact (basis_mul_coeff_empty_r (n:=n) sq A).
           }
           ring.
        -- (* A ≠ U: both sides 0 *)
           ring.
  - (* outer pick over A *)
    exact (@sumQ_all_masks_pick n F U).
Qed.

(* ============================================================ *)
(* Closed form on basis blades                                  *)
(* ============================================================ *)

Lemma mv_gp_basis :
  forall n (sq : Vector.t Q n) (A B : Mask n) (U : Mask n),
    @mv_gp n sq (basis A) (basis B) U
    ==
    (if mask_eq_dec U (basis_mul_mask A B)
     then basis_mul_coeff sq A B
     else 0%Q).
Proof.
  intros n sq A B U.
  unfold mv_gp, basis.

  (* Step 1: rewrite the outer map so it is exactly in pick-shape over A' *)
  eapply Qeq_trans.
  - apply (@sumQ_map_ext (Mask n)
      (fun A' : Mask n =>
         sumQ
           (List.map
              (fun B' : Mask n =>
                 let c := basis_mul_coeff sq A' B' in
                 if mask_eq_dec (basis_mul_mask A' B') U
                 then ((if mask_eq_dec A' A then 1%Q else 0%Q)
                       * (if mask_eq_dec B' B then 1%Q else 0%Q) * c)%Q
                 else 0%Q)
              (all_masks n)))
      (fun A' : Mask n =>
         if mask_eq_dec A' A then
           sumQ
             (List.map
                (fun B' : Mask n =>
                   let c := basis_mul_coeff sq A B' in
                   if mask_eq_dec (basis_mul_mask A B') U
                   then ((1%Q)
                         * (if mask_eq_dec B' B then 1%Q else 0%Q) * c)%Q
                   else 0%Q)
                (all_masks n))
         else 0%Q)
      (all_masks n)).
    intros A' HA'.
    destruct (mask_eq_dec A' A) as [HeqA'|HneqA'].
    + subst A'. cbn. ring.
    + (* if A'<>A then (if A'=A then 1 else 0)=0, so entire inner sum is 0 *)
      cbn.
      (* show the inner sum is sum of zeros *)
      eapply Qeq_trans.
      * apply (@sumQ_map_ext (Mask n)
          (fun B' : Mask n =>
             let c := basis_mul_coeff sq A' B' in
             if mask_eq_dec (basis_mul_mask A' B') U
             then (0%Q * (if mask_eq_dec B' B then 1%Q else 0%Q) * c)%Q
             else 0%Q)
          (fun _ : Mask n => 0%Q)
          (all_masks n)).
        intros B' HB'.
        destruct (mask_eq_dec (basis_mul_mask A' B') U); cbn; ring.
      * exact (sumQ_map_const0 (A:=Mask n) (all_masks n)).
  - (* Step 2: pick A'=A in the outer sum *)
    eapply Qeq_trans.
    +
      set (innerA :=
        fun A' : Mask n =>
          sumQ
            (List.map
               (fun B' : Mask n =>
                  let c := basis_mul_coeff sq A' B' in
                  if mask_eq_dec (basis_mul_mask A' B') U
                  then ((if mask_eq_dec A' A then 1%Q else 0%Q)
                        * (if mask_eq_dec B' B then 1%Q else 0%Q) * c)%Q
                  else 0%Q)
               (all_masks n))).

      (* rewrite the current outer sum into the exact pick shape for innerA *)
      eapply Qeq_trans.
      * apply (@sumQ_map_ext (Mask n)
          (fun A' : Mask n =>
             if mask_eq_dec A' A
             then
               sumQ
                 (List.map
                    (fun B' : Mask n =>
                       let c := basis_mul_coeff sq A B' in
                       if mask_eq_dec (basis_mul_mask A B') U
                       then (1%Q * (if mask_eq_dec B' B then 1%Q else 0%Q) * c)%Q
                       else 0%Q)
                    (all_masks n))
             else 0%Q)
          (fun A' : Mask n =>
             if mask_eq_dec A' A then innerA A' else 0%Q)
          (all_masks n)).
        intros A' HA'.
        destruct (mask_eq_dec A' A) as [Heq|Hneq].
        ++ subst A'. unfold innerA.
         destruct (mask_eq_dec A A) as [_|Hbad].
         -- cbn. apply Qeq_refl.
         -- exfalso; apply Hbad; reflexivity.
        ++ cbn. apply Qeq_refl.
      * (* now apply pick *)
        exact (@sumQ_all_masks_pick n innerA A).
    + (* after pick, simplify innerA A *)
      cbn.
      destruct (mask_eq_dec A A) as [_|Hbad]; [|exfalso; apply Hbad; reflexivity].
      cbn.

  (* Step 3: rewrite inner B' sum into pick-shape over B' *)
  eapply Qeq_trans.
  * apply (@sumQ_map_ext (Mask n)
      (fun B' : Mask n =>
         let c := basis_mul_coeff sq A B' in
         if mask_eq_dec (basis_mul_mask A B') U
         then (1%Q * (if mask_eq_dec B' B then 1%Q else 0%Q) * c)%Q
         else 0%Q)
      (fun B' : Mask n =>
         if mask_eq_dec B' B then
           let c := basis_mul_coeff sq A B in
           if mask_eq_dec (basis_mul_mask A B) U
           then (1%Q * 1%Q * c)%Q
           else 0%Q
         else 0%Q)
      (all_masks n)).
    intros B' HB'.
    destruct (mask_eq_dec B' B) as [HeqB'|HneqB'].
    -- subst B'. cbn. ring.
    -- destruct (mask_eq_dec (basis_mul_mask A B') U); cbn; ring.
    * (* Step 4: pick B'=B *)
    set (innerB :=
      fun _ : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then (1%Q * 1%Q * c)%Q
        else 0%Q).

    (* rewrite into exact pick-shape that sumQ_all_masks_pick expects *)
    eapply Qeq_trans.
    -- apply (@sumQ_map_ext (Mask n)
         (fun B' : Mask n =>
            if mask_eq_dec B' B
            then let c := basis_mul_coeff sq A B in
                 if mask_eq_dec (basis_mul_mask A B) U
                 then (1%Q * 1%Q * c)%Q
                 else 0%Q
            else 0%Q)
         (fun B' : Mask n =>
            if mask_eq_dec B' B then innerB B' else 0%Q)
         (all_masks n)).
       intros B' HB'.
       destruct (mask_eq_dec B' B) as [->|Hneq]; cbn; apply Qeq_refl.
    
    -- eapply Qeq_trans.
        exact (@sumQ_all_masks_pick n innerB B).
        (* now prove innerB B == (if ... then coeff else 0) *)
        unfold innerB.
        (* simplify 1*1*c = c and reconcile the two eq_dec orientations *)
        destruct (mask_eq_dec (basis_mul_mask A B) U) as [HABU|HABU]; cbn.
        (* basis_mul_mask A B = U *)
        destruct (mask_eq_dec U (basis_mul_mask A B)) as [HU|HNU].
        ring.  (* (1*1*c)=c and RHS is coeff *)
        exfalso; apply HNU; symmetry; exact HABU.
        (* basis_mul_mask A B <> U *)
        destruct (mask_eq_dec U (basis_mul_mask A B)) as [HU|HNU].
        exfalso; apply HABU; symmetry; exact HU.
        ring.  (* both sides 0 *)
Qed.



(* ============================================================ *)
(* Singleton masks = generators e_i                               *)
(* ============================================================ *)

Fixpoint mask_single {n : nat} (i : Fin.t n) : Mask n :=
  match i with
  | Fin.F1 =>
      true :: Vector.const false _
  | Fin.FS j =>
      false :: mask_single j
  end.
Definition e (n : nat) (i : Fin.t n) : MV n := basis (mask_single i).

(* ============================================================ *)
(* Clifford relations on generators                               *)
(* ============================================================ *)

Lemma mask_xor_self :
  forall n (A : Mask n),
    mask_xor A A = mask_empty (n:=n).
Proof.
  induction n; intro A.
  - dependent destruction A. reflexivity.
  - dependent destruction A. cbn [mask_xor mask_empty].
    simpl. f_equal.
    + destruct h; reflexivity.
    + apply IHn.
Qed.

Lemma mask_xor_comm :
  forall n (A B : Mask n),
    mask_xor A B = mask_xor B A.
Proof.
  induction n; intros A B.
  - dependent destruction A; dependent destruction B; reflexivity.
  - dependent destruction A; dependent destruction B.
    cbn [mask_xor]. simpl. f_equal.
    + destruct h, h0; reflexivity.
    + apply IHn.
Qed.

Lemma metric_factor_single_square :
  forall n (sq : Vector.t Q n) (i : Fin.t n),
    metric_factor sq (mask_single i) (mask_single i) == Vector.nth sq i.
Proof.
  intros n sq i.
  induction i as [|n i IH].
    dependent destruction sq.
    cbn [mask_single].
    unfold metric_factor.
    simpl. simpl.

    change (List.fold_right Qmult 1%Q
      (List.map
        (fun '(sq_i, (ai, bi)) => if ai && bi then sq_i else 1%Q)
        (List.combine (Vector.to_list sq)
          (List.combine (Vector.to_list (Vector.const false n))
                        (Vector.to_list (Vector.const false n))))))
    with (metric_factor sqt (Vector.const false n) (Vector.const false n)).

    rewrite (@metric_factor_empty_l n sq (Vector.const false n)).
    cbn [Vector.nth].
    ring.
    
  - dependent destruction sq.

    cbn [mask_single].
    unfold metric_factor.
    simpl. simpl.

    change
      (List.fold_right Qmult 1%Q
         (List.map
            (fun '(sq_i, (ai, bi)) => if ai && bi then sq_i else 1%Q)
            (List.combine (Vector.to_list sq)
               (List.combine (Vector.to_list (mask_single i))
                             (Vector.to_list (mask_single i))))))
    with (metric_factor sqt (mask_single i) (mask_single i)).

    eapply Qeq_trans with
      (y := List.fold_right Qmult 1%Q
              (List.map (fun '(sq_i, (ai, bi)) => if ai && bi then sq_i else 1%Q)
                 (List.combine (Vector.to_list sq)
                    (List.combine (Vector.to_list (mask_single i))
                                  (Vector.to_list (mask_single i)))))).
    *  cbn [Vector.to_list].
       rewrite Qmult_1_l.
       reflexivity.
    *
      change (List.fold_right Qmult 1%Q
                (List.map (fun '(sq_i, (ai, bi)) => if ai && bi then sq_i else 1%Q)
                   (List.combine (Vector.to_list sq)
                      (List.combine (Vector.to_list (mask_single i))
                                    (Vector.to_list (mask_single i))))))
        with (metric_factor sq (mask_single i) (mask_single i)).
      exact (IH sq).
Qed.

Lemma swaps_parity_const_false :
  forall n,
    swaps_parity (Vector.const false n) (Vector.const false n) = false.
Proof.
  intro n.
  unfold swaps_parity.
  induction n as [|n IH].
  - cbn. reflexivity.
  - cbn [Vector.to_list Vector.const].
    simpl.

    (* name the tail list in the same shape IH uses *)
    set (tl := Vector.to_list (Vector.const false n)).

    (* force the goal to talk about tl, not the unfolded fixpoint *)
    change
      (fst
         (let '(s0, p0) :=
            List.fold_right
              (fun ab st : bool * bool =>
                 let '(ai, bi) := ab in
                 let '(s0, p0) := st in (xorb s0 (bi && p0), xorb ai p0))
              (false, false)
              (List.combine tl tl)
          in (xorb s0 false, p0)) = false).

    (* now remember a small term *)
    remember
      (List.fold_right
         (fun ab st : bool * bool =>
            let '(ai, bi) := ab in
            let '(s0, p0) := st in (xorb s0 (bi && p0), xorb ai p0))
         (false, false)
         (List.combine tl tl))
      as tail eqn:Htail.

    destruct tail as [s p]. cbn.

    (* goal becomes xorb s false = false *)
    rewrite xorb_false_r.  (* goal: s = false *)

    (* rewrite IH to the same statement using tl/tail *)
    unfold tl in Htail.
    rewrite <- Htail in IH.
    cbn in IH.

    exact IH.
Qed.

Lemma swaps_state_const_false :
  forall n,
    List.fold_right
      (fun ab st : bool * bool =>
         let '(ai, bi) := ab in
         let '(s, p) := st in (xorb s (bi && p), xorb ai p))
      (false, false)
      (List.combine (Vector.to_list (Vector.const false n))
                    (Vector.to_list (Vector.const false n)))
    = (false, false).
Proof.
  induction n as [|n IH].
  - cbn. reflexivity.
  - cbn [Vector.to_list Vector.const].
    simpl.

    set (tl := Vector.to_list (Vector.const false n)).

    remember
      (List.fold_right
         (fun ab st : bool * bool =>
            let '(ai, bi) := ab in
            let '(s, p) := st in (xorb s (bi && p), xorb ai p))
         (false, false)
         (List.combine tl tl))
      as tail eqn:Htail.

    (* goal is currently the head-step applied to the unfolded tail;
       fold it back to combine tl tl *)
    change ((let '(s0, p0) := 
      List.fold_right (fun ab st : bool * bool =>
                       let '(ai, bi) := ab in
                       let '(s0, p0) := st in
                       (xorb s0 (bi && p0), xorb ai p0)) (false, false)
                       (List.combine tl tl) in (xorb s0 false, p0)) = (false, false)).


    (* rewrite tail fold into (s,p), then compute *)
    rewrite <- Htail.
    destruct tail as [s p]. cbn.
    rewrite xorb_false_r.
    
    (* reduce IH to the same tail statement *)
    unfold tl in Htail.
    rewrite <- Htail in IH.
    exact IH.
Qed.

Lemma swaps_parity_single_self :
  forall n (i : Fin.t n),
    swaps_parity (mask_single i) (mask_single i) = false.
Proof.
  intros n i.
  induction i as [|n i IH].
  - (* i = F1 *)
    cbn [mask_single].            (* swaps_parity (true::const false _) (true::const false _) *)
    unfold swaps_parity.
    cbn [Vector.to_list].         (* to_list (true::v) *)
    cbn [Vector.to_list Vector.const].
    simpl.                        (* fold_right over (true,true)::tail *)
    
    (* fold the unfolded fixpoint back into Vector.to_list (Vector.const false n) *)
    set (tl := Vector.to_list (Vector.const false n)).
    
    change
      (fst
         (let '(s, p) :=
            List.fold_right
              (fun ab st : bool * bool =>
                 let '(ai, bi) := ab in
                 let '(s, p) := st in (xorb s (bi && p), xorb ai p))
              (false, false)
              (List.combine tl tl)
          in (xorb s p, if p then false else true)) = false).
          
    unfold tl.

    (* now it matches the lemma *)
    rewrite swaps_state_const_false.

    cbn.                          (* compute head step at (true,true) with tail=(false,false) *)
    reflexivity.

    - (* i = FS i *)
    cbn [mask_single].
    unfold swaps_parity.
    cbn [Vector.to_list]. 
    cbn [Vector.to_list Vector.const].
    simpl.

    (* Turn the tail fold into swaps_parity (mask_single i) (mask_single i) *)
    change
      (fst
         (let '(s, p) :=
            List.fold_right
              (fun ab st : bool * bool =>
                 let '(ai, bi) := ab in
                 let '(s, p) := st in (xorb s (bi && p), xorb ai p))
              (false, false)
              (List.combine (Vector.to_list (mask_single i))
                            (Vector.to_list (mask_single i)))
          in (xorb s false, p)) = false).

    (* Now simplify the let/fst: xorb s false = s *)
    remember
      (List.fold_right
         (fun ab st : bool * bool =>
            let '(ai, bi) := ab in
            let '(s, p) := st in (xorb s (bi && p), xorb ai p))
         (false, false)
         (List.combine (Vector.to_list (mask_single i))
                       (Vector.to_list (mask_single i))))
      as tail eqn:Htail.
    destruct tail as [s p]. cbn.
    rewrite xorb_false_r.

    (* Goal is now fst (s,p) = false; rewrite back to swaps_parity and use IH *)
    cbn.
    (* Use IH: swaps_parity (mask_single i) (mask_single i) = false *)
    unfold swaps_parity in IH.
    (* rewrite IH's fold into our s *)
    rewrite <- Htail in IH.
    cbn in IH.
    exact IH.
Qed.


Lemma e_square :
  forall n (sq : Vector.t Q n) (i : Fin.t n) (U : Mask n),
    (@mv_gp n sq (e i) (e i)) U
    ==
    (mv_scale (Vector.nth sq i) mv_one) U.
Proof.
  intros n sq i U.
  unfold mv_one.  (* don't unfold e; it's fine *)

  rewrite (@mv_gp_basis n sq (mask_single i) (mask_single i) U).
  unfold basis_mul_mask, basis_mul_coeff.
  rewrite (mask_xor_self (mask_single i)).

  destruct (mask_eq_dec U (mask_empty (n:=n))) as [HU|HUne].
  - subst U.
    rewrite swaps_parity_single_self.
    cbn [sgnQ].
    rewrite (@metric_factor_single_square n sq i).
    cbn.
    unfold basis; cbn.
    
    unfold mv_scale.
    cbn.  (* turns (fun T => if mask_eq_dec T mask_empty then 1 else 0) mask_empty into an if *)
    destruct (mask_eq_dec mask_empty mask_empty) as [_|H]; [|contradiction].
    cbn.
    ring.
  - unfold mv_scale, basis; cbn.
    destruct (mask_eq_dec U (mask_empty (n:=n))); [contradiction|].
    ring.
Qed.

Lemma metric_factor_single_disjoint :
  forall n (sq : Vector.t Q n) (i j : Fin.t n),
    i <> j ->
    metric_factor sq (mask_single i) (mask_single j) == 1%Q.
Proof.
  intros n sq i j Hij.
  revert sq j Hij.
  induction i as [|n i IH]; intros sq j Hij.
  - (* i = F1 *)
  dependent destruction sq. (* sq = h :: sqt, n is tail length *)
  dependent destruction j.

  + (* j = F1 *)
    exfalso. apply Hij. reflexivity.
  + (* j = FS j, with j : Fin.t n *)
    cbn [mask_single].
    unfold metric_factor.
    simpl. simpl.
    change (List.fold_right Qmult 1%Q
      (List.map (fun '(sq_i, (ai, bi)) => if ai && bi then sq_i else 1%Q)
        (List.combine (Vector.to_list sq)
          (List.combine (Vector.to_list (Vector.const false n))
                        (Vector.to_list (mask_single j))))))
    with (metric_factor sqt (Vector.const false n) (mask_single j)).
    rewrite (@metric_factor_empty_l n sq (mask_single j)).
    ring.


  - (* i = FS i *)
    dependent destruction sq.
    dependent destruction j.
    + (* j = F1 *)
      cbn [mask_single].
      unfold metric_factor.
      simpl. simpl.
      (* head overlap is false && true = false, so head factor is 1 *)
      change (List.fold_right Qmult 1%Q
        (List.map (fun '(sq_i,(ai,bi)) => if ai && bi then sq_i else 1%Q)
          (List.combine (Vector.to_list sq)
            (List.combine (Vector.to_list (mask_single i)) (Vector.to_list (Vector.const false n))))))
      with (metric_factor sqt (mask_single i) (Vector.const false n)).
      rewrite (@metric_factor_empty_r n sq (mask_single i)).
      ring.

    + (* j = FS j *)
      cbn [mask_single].
      unfold metric_factor.
      simpl. simpl.
      replace (List.fold_right Qmult 1%Q
          (List.map (fun '(sq_i,(ai,bi)) => if ai && bi then sq_i else 1%Q)
             (List.combine (Vector.to_list sq)
                (List.combine (Vector.to_list (mask_single i))
                              (Vector.to_list (mask_single j))))))
      with (metric_factor sq (mask_single i) (mask_single j)) by reflexivity.

      (* head contributes 1, so reduce to IH on tails *)
      eapply Qeq_trans with
        (y := List.fold_right Qmult 1%Q
                (List.map (fun '(sq_i, (ai, bi)) => if ai && bi then sq_i else 1%Q)
                  (List.combine (Vector.to_list sq)
                    (List.combine (Vector.to_list (mask_single i))
                                  (Vector.to_list (mask_single j)))))).
      * cbn [Vector.to_list]. rewrite Qmult_1_l. reflexivity.
      * change (List.fold_right Qmult 1%Q
                  (List.map (fun '(sq_i, (ai, bi)) => if ai && bi then sq_i else 1%Q)
                    (List.combine (Vector.to_list sq)
                      (List.combine (Vector.to_list (mask_single i))
                                    (Vector.to_list (mask_single j))))))
          with (metric_factor sq (mask_single i) (mask_single j)).
        (* Hij : FS i <> FS j  ->  i <> j *)
        apply (IH sq j).
        intro Heq. apply Hij. now f_equal.
Qed.

Lemma swaps_parity_empty_r_aux_snd :
  forall n (A : Mask n),
    snd
      (List.fold_right
         (fun ab st =>
            let '(ai, bi) := ab in
            let '(s, p) := st in
            (xorb s (andb bi p), xorb ai p))
         (false, false)
         (List.combine (Vector.to_list A)
                       (Vector.to_list (Vector.const false n))))
    = List.fold_right xorb false (Vector.to_list A).
Proof.
  induction n as [|n IH]; intro A.
  - dependent destruction A. simpl. reflexivity.
  - dependent destruction A. simpl.
    fold (Vector.to_list A).
    fold (Vector.to_list (Vector.const false n)).
    simpl.
    remember
      (List.fold_right
         (fun ab st : bool * bool =>
            let '(ai, bi) := ab in
            let '(s, p) := st in (xorb s (bi && p), xorb ai p))
         (false, false)
         (combine (to_list A) (to_list (const false n))))
      as tail eqn:Htail.
    destruct tail as [s p]. simpl.
    specialize (IH A).
    rewrite <- Htail in IH. simpl in IH.
    rewrite IH. reflexivity.
Qed.

Lemma parity_const_false :
  forall n,
    List.fold_right xorb false (Vector.to_list (Vector.const false n)) = false.
Proof.
  induction n as [|n IH].
  - simpl. reflexivity.
  - simpl.
    cbn [Vector.to_list].  (* this causes the tail to become a fixpoint *)

    (* >>> ADD THIS <<< fold the fixpoint tail back into Vector.to_list *)
    change
      ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
          match v with
          | [] => b
          | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
          end) n (Vector.const false n) []%list)
    with (Vector.to_list (Vector.const false n)).

    simpl.
    rewrite IH.
    simpl.
    reflexivity.
Qed.

Lemma to_list_cons :
  forall (A : Type) n (a : A) (v : Vector.t A n),
    Vector.to_list (Vector.cons A a n v) = (a :: Vector.to_list v)%list.
Proof.
  intros A n a v.
  reflexivity.
Qed.

Lemma mask_single_FS :
  forall n (j : Fin.t n),
    mask_single (Fin.FS j) = Vector.cons bool false n (mask_single j).
Proof.
  intros. reflexivity.
Qed.

Lemma parity_mask_single_true :
  forall n (j : Fin.t n),
    List.fold_right xorb false (Vector.to_list (mask_single j)) = true.
Proof.
  induction n as [|n IH]; intros j.
  - inversion j.
  - dependent destruction j.
    + cbn [mask_single]. simpl.
      change
        ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
            match v with
            | [] => b
            | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
            end) n (Vector.const false n) []%list)
      with (Vector.to_list (Vector.const false n)).
      rewrite parity_const_false.
      simpl. reflexivity.
    + (* j = FS j *)
      (* either just: cbn [mask_single]. rewrite to_list_cons. ... *)
      rewrite mask_single_FS.      (* include only if needed *)
      rewrite to_list_cons.
      simpl.
      apply IH.
Qed.

Lemma fst_let_pair :
  forall (t : bool * bool),
    fst (let '(s,p) := t in (xorb s false, p)) = fst t.
Proof.
  intros [s p]. simpl. now rewrite Bool.xorb_false_r.
Qed.


Lemma swaps_parity_single_swap_negb :
  forall n (i j : Fin.t n),
    i <> j ->
    swaps_parity (mask_single i) (mask_single j)
    =
    negb (swaps_parity (mask_single j) (mask_single i)).
Proof.
  intros n i j Hij.
  revert j Hij.
  induction i as [|n i IH]; intros j Hij.
  - (* i = F1 *)
    dependent destruction j.
    + exfalso. apply Hij. reflexivity.
    + (* j = FS j *)
      cbn [mask_single].
      unfold swaps_parity.
      cbn.

      (* tails *)
      remember
        (List.fold_right
           (fun ab st : bool * bool =>
              let '(ai, bi) := ab in
              let '(s, p) := st in (xorb s (andb bi p), xorb ai p))
           (false, false)
           (List.combine (Vector.to_list (Vector.const false n))
                         (Vector.to_list (mask_single j))))
        as tailL eqn:HtailL.
      remember
        (List.fold_right
           (fun ab st : bool * bool =>
              let '(ai, bi) := ab in
              let '(s, p) := st in (xorb s (andb bi p), xorb ai p))
           (false, false)
           (List.combine (Vector.to_list (mask_single j))
                         (Vector.to_list (Vector.const false n))))
        as tailR eqn:HtailR.

      destruct tailL as [sL pL]; destruct tailR as [sR pR]; cbn.

      (* fst tails are false *)
      assert (sL = false) as HsL.
      { pose proof (@swaps_parity_empty_l_aux n (mask_single j)) as Haux.
        rewrite <- HtailL in Haux.
        inversion Haux; reflexivity. }

      assert (sR = false) as HsR.
      { pose proof (@swaps_parity_empty_r_aux_fst n (mask_single j)) as Hr.
        rewrite <- HtailR in Hr.
        exact Hr. }

      subst sL sR.

      (* rewrite tails back into the goal; may need your fold-back 'change' tricks
         if to_list got unfolded by cbn somewhere *)
      (* fold back the first to_list (const false n) *)
      change
        ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
            match v with
            | [] => b
            | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
            end) n (Vector.const false n) []%list)
      with (Vector.to_list (Vector.const false n)).

      (* fold back the second to_list (mask_single j) *)
      change
        ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
            match v with
            | [] => b
            | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
            end) n (mask_single j) []%list)
      with (Vector.to_list (mask_single j)).

      rewrite <- HtailL.
      rewrite <- HtailR.
      cbn.

      (* From empty-left aux we can also get pL=false *)
      assert (pL = false) as HpL.
      { pose proof (@swaps_parity_empty_l_aux n (mask_single j)) as Haux.
        rewrite <- HtailL in Haux.
        inversion Haux; reflexivity. }
      subst pL.
      cbn.

      (* Now the remaining goal will be false = negb pR (like you saw before).
         Prove pR=true using the snd lemma + parity_mask_single_true. *)
      assert (pR = true) as HpR.
      { pose proof (@swaps_parity_empty_r_aux_snd n (mask_single j)) as Hsnd.
        (* Hsnd : snd (fold_right ... (combine (to_list (mask_single j)) (to_list (const false n))))
                 = fold_right xorb false (to_list (mask_single j)) *)
        rewrite <- HtailR in Hsnd.
        simpl in Hsnd.  (* snd (false,pR) = pR *)
        rewrite parity_mask_single_true in Hsnd.
        exact Hsnd. }

      rewrite HpR. cbn. reflexivity.

  - (* i = FS i *)
    dependent destruction j.
    + (* j = F1 *)
      (* This is the same statement as the previous case, with roles swapped.
         You can either redo the symmetric calculation, or just use the result
         from the previous case by appealing to the first branch IH on i=F1. *)

      (* easiest: reuse the already-proved base case by symmetry:
         show swaps_parity (mask_single (FS i)) (mask_single F1) = negb (...) *)
      cbn [mask_single].
      unfold swaps_parity.
      cbn.

      (* tails: now left tail is (mask_single i, const false), right tail is (const false, mask_single i) *)
      remember
        (List.fold_right
           (fun ab st : bool * bool =>
              let '(ai, bi) := ab in
              let '(s, p) := st in (xorb s (andb bi p), xorb ai p))
           (false, false)
           (List.combine (Vector.to_list (mask_single i))
                         (Vector.to_list (Vector.const false n))))
        as tailL eqn:HtailL.
      remember
        (List.fold_right
           (fun ab st : bool * bool =>
              let '(ai, bi) := ab in
              let '(s, p) := st in (xorb s (andb bi p), xorb ai p))
           (false, false)
           (List.combine (Vector.to_list (Vector.const false n))
                         (Vector.to_list (mask_single i))))
        as tailR eqn:HtailR.

      destruct tailL as [sL pL]; destruct tailR as [sR pR]; cbn.

      assert (sL = false) as HsL.
      { pose proof (@swaps_parity_empty_r_aux_fst n (mask_single i)) as Hr.
        rewrite <- HtailL in Hr. exact Hr. }
      assert (sR = false) as HsR.
      { pose proof (@swaps_parity_empty_l_aux n (mask_single i)) as Haux.
        rewrite <- HtailR in Haux.
        inversion Haux; reflexivity. }

      subst sL sR.
      (* fold back to_list (mask_single i) *)
      change
        ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
            match v with
            | [] => b
            | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
            end) n (mask_single i) []%list)
      with (Vector.to_list (mask_single i)).

      (* fold back to_list (const false n) *)
      change
        ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
            match v with
            | [] => b
            | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
            end) n (Vector.const false n) []%list)
      with (Vector.to_list (Vector.const false n)).

      rewrite <- HtailL.
      change
        ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
            match v with
            | [] => b
            | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
            end) n (Vector.const false n) []%list)
      with (Vector.to_list (Vector.const false n)).

      change
        ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
            match v with
            | [] => b
            | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
            end) n (mask_single i) []%list)
      with (Vector.to_list (mask_single i)).
      rewrite <- HtailR.
      cbn.

      (* Here you’ll get the “false = negb pL” shape; pL is parity of mask_single i, hence true. *)
      assert (pL = true) as HpL.
      { pose proof (@swaps_parity_empty_r_aux_snd n (mask_single i)) as Hsnd.
        rewrite <- HtailL in Hsnd.
        simpl in Hsnd.
        rewrite parity_mask_single_true in Hsnd.
        exact Hsnd. }
      rewrite HpL. cbn. reflexivity.

    + (* j = FS j *)
      (* reduce to tails; this should collapse directly to IH *)
      cbn [mask_single].
      unfold swaps_parity.
      cbn.

      (* fold the unfolded to_list fixpoints back so we can fold swaps_parity itself *)
      change
        ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
            match v with
            | [] => b
            | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
            end) n (mask_single i) []%list)
      with (Vector.to_list (mask_single i)).

      change
        ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
            match v with
            | [] => b
            | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
            end) n (mask_single j) []%list)
      with (Vector.to_list (mask_single j)).

      rewrite fst_let_pair.
      rewrite fst_let_pair.
      change (swaps_parity (n:=n) (mask_single i) (mask_single j) =
              negb (swaps_parity (n:=n) (mask_single j) (mask_single i))).
      apply IH; intro Heq; apply Hij; now f_equal.
Qed.

Lemma sgnQ_swaps_parity_flip :
  forall n (i j : Fin.t n),
    i <> j ->
    sgnQ (swaps_parity (mask_single i) (mask_single j))
    ==
    (-1)%Q * sgnQ (swaps_parity (mask_single j) (mask_single i)).
Proof.
  intros n i j Hij.
  rewrite (@swaps_parity_single_swap_negb n i j Hij).
  unfold sgnQ.
  destruct (swaps_parity (mask_single j) (mask_single i)); cbn; ring.
Qed.


Lemma e_anticomm :
  forall n (sq : Vector.t Q n) (i j : Fin.t n) (U : Mask n),
    i <> j ->
    (@mv_gp n sq (e i) (e j)) U
    ==
    mv_scale (-1)%Q (@mv_gp n sq (e j) (e i)) U.
Proof.
  intros n sq i j U Hij.
  unfold mv_scale.

  (* reduce both sides to basis coefficients *)
  rewrite (@mv_gp_basis n sq (mask_single i) (mask_single j) U).
  rewrite (@mv_gp_basis n sq (mask_single j) (mask_single i) U).
  unfold basis_mul_mask, basis_mul_coeff.

  (* xor is commutative *)
  rewrite (mask_xor_comm (mask_single i) (mask_single j)).

  destruct (mask_eq_dec U (mask_xor (mask_single j) (mask_single i))) as [HU|HUne].
  - subst U.
    (* now compare coefficients *)

    (* metric factors are 1 in both orders *)
    rewrite (@metric_factor_single_disjoint n sq i j Hij).
    rewrite (@metric_factor_single_disjoint n sq j i (fun H => Hij (eq_sym H))).

    (* now only the sgnQ terms differ *)
    rewrite (@sgnQ_swaps_parity_flip n i j Hij).

    ring.

  - (* outside the xor blade, both are zero *)
    cbn.
    ring.
Qed.

(*A. Sign side: swaps_parity cocycle*)

Lemma sgnQ_xorb :
  forall b1 b2,
    (sgnQ b1 * sgnQ b2)%Q == sgnQ (xorb b1 b2).
Proof.
  intros b1 b2. destruct b1, b2; unfold sgnQ; cbn; ring.
Qed.

Lemma sgnQ_negb :
  forall b, sgnQ (negb b) == (-1)%Q * sgnQ b.
Proof.
  intros b. destruct b; unfold sgnQ; cbn; ring.
Qed.


Lemma sumQ_fubini :
  forall (A B : Type) (la : list A) (lb : list B) (h : A -> B -> Q),
    sumQ (List.map (fun a => sumQ (List.map (fun b => h a b) lb)) la)
    ==
    sumQ (List.map (fun b => sumQ (List.map (fun a => h a b) la)) lb).
Proof.
  intros A B la.
  induction la as [|a tl IH]; intros lb h; simpl.
  - rewrite sumQ_map_const0. reflexivity.
  - eapply Qeq_trans.
    2: {
      apply (sumQ_map_ext (A:=B)
        (fun b => sumQ (List.map (fun a0 => h a0 b) (a :: tl)))
        (fun b => (h a b + sumQ (List.map (fun a0 => h a0 b) tl))%Q)
        lb).
      intros b Hb. simpl. reflexivity.
    }
    rewrite (sumQ_map_add (A:=B)
      (fun b => h a b)
      (fun b => sumQ (List.map (fun a0 => h a0 b) tl))
      lb).
    rewrite <- IH.
    ring.
Qed.

Definition parity_mask {n} (A : Mask n) : bool :=
  List.fold_right xorb false (Vector.to_list A).
  
Lemma parity_mask_cons :
  forall n (h:bool) (A:Mask n),
    parity_mask (h :: A) = xorb h (parity_mask A).
Proof.
  intros n h A.
  unfold parity_mask.
  cbn [Vector.to_list]. simpl.
  reflexivity.
Qed.

Lemma map2_cons :
  forall (A B C : Type) (g : A -> B -> C) n
         (a : A) (b : B) (va : Vector.t A n) (vb : Vector.t B n),
    Vector.map2 g (a :: va) (b :: vb) = (g a b) :: Vector.map2 g va vb.
Proof.
  intros A B C g n a b va vb.
  (* map2 is defined via rect2, so one cbn step solves it *)
  cbn [Vector.map2 Vector.rect2].
  reflexivity.
Qed.

Lemma mask_xor_cons :
  forall n (a b : bool) (A B : Mask n),
    mask_xor (a :: A) (b :: B) = (xorb a b) :: mask_xor A B.
Proof.
  intros n a b A B.
  unfold mask_xor.
  rewrite map2_cons.
  reflexivity.
Qed.

Lemma xorb_assoc : forall a b c, xorb a (xorb b c) = xorb (xorb a b) c.
Proof. intros a b c; destruct a, b, c; reflexivity. Qed.

Lemma xorb_comm : forall a b, xorb a b = xorb b a.
Proof. intros a b; destruct a, b; reflexivity. Qed.


Lemma parity_mask_xor :
  forall n (A B : Mask n),
    parity_mask (mask_xor A B) = xorb (parity_mask A) (parity_mask B).
Proof.
  induction n as [|n IH]; intros A B.
  - dependent destruction A; dependent destruction B.
    cbn [parity_mask mask_xor]. reflexivity.
  - dependent destruction A; dependent destruction B.
    (* rewrite RHS parities first *)
    rewrite parity_mask_cons.
    rewrite parity_mask_cons.
    (* rewrite LHS mask_xor into a cons, then parity_mask_cons applies *)
    rewrite mask_xor_cons.
    rewrite parity_mask_cons.
    (* tail *)
    rewrite IH.
    destruct h, h0; cbn;
    destruct (parity_mask A), (parity_mask B); cbn; reflexivity.
Qed.


Lemma swaps_state_snd :
  forall n (A B : Mask n),
    snd
      (List.fold_right
         (fun ab st : bool * bool =>
            let '(ai, bi) := ab in
            let '(s, p) := st in (xorb s (bi && p), xorb ai p))
         (false, false)
         (List.combine (Vector.to_list A) (Vector.to_list B)))
    = parity_mask A.
Proof.
  induction n as [|n IH]; intros A B.
  - dependent destruction A. dependent destruction B. cbn. reflexivity.
  - dependent destruction A. dependent destruction B. cbn.
    (* after cbn, the fold_right over combine becomes head :: tail *)
    simpl.
    (* unpack tail state *)
    remember
      (List.fold_right
         (fun ab st : bool * bool =>
            let '(ai, bi) := ab in
            let '(s, p) := st in (xorb s (bi && p), xorb ai p))
         (false, false)
         (combine (to_list A) (to_list B)))
      as tail eqn:Htail.
    destruct tail as [s p]. cbn.
    (* use IH on tails *)
    specialize (IH A B).
    rewrite <- Htail in IH. cbn in IH.
    unfold parity_mask in *. cbn in *.
    (* fold back the unfolded Vector.to_list A *)
    change
      ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
          match v with
          | [] => b
          | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
          end) n A []%list)
    with (Vector.to_list A).

    (* fold back the unfolded Vector.to_list B *)
    change
      ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b : list bool) {struct v} : list bool :=
          match v with
          | [] => b
          | Vector.cons _ a n1 w => (a :: fold_right_fix n1 w b)%list
          end) n B []%list)
    with (Vector.to_list B).

    rewrite <- Htail.
    cbn.
    now rewrite IH.
Qed.

Lemma swaps_parity_cons :
  forall n (a b : bool) (A B : Mask n),
    swaps_parity (Vector.cons _ a _ A) (Vector.cons _ b _ B)
    =
    xorb (swaps_parity A B) (b && parity_mask A).
Proof.
  intros n a b A B.
  unfold swaps_parity.
  cbn.
  (* unfold one step of fold_right on combine *)
  simpl.
  (* tail state *)
  remember
    (List.fold_right
       (fun ab st : bool * bool =>
          let '(ai, bi) := ab in
          let '(s, p) := st in (xorb s (bi && p), xorb ai p))
       (false, false)
       (combine (to_list A) (to_list B)))
    as tail eqn:Htail.
  destruct tail as [s p]. cbn.

  (* identify s = swaps_parity A B and p = parity_mask A *)
  assert (s = swaps_parity A B).
  { unfold swaps_parity. rewrite <- Htail. reflexivity. }
  assert (p = parity_mask A).
  { pose proof (@swaps_state_snd n A B) as Hp.
    rewrite <- Htail in Hp. exact Hp. }

    subst s p.

    (* fold the unfolded to_list fixpoints back so Htail matches *)
    change
      ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b0 : list bool) {struct v} : list bool :=
          match v with
          | [] => b0
          | Vector.cons _ a0 n1 w => (a0 :: fold_right_fix n1 w b0)%list
          end) n A []%list)
    with (Vector.to_list A).

    change
      ((fix fold_right_fix (n0 : nat) (v : Vector.t bool n0) (b0 : list bool) {struct v} : list bool :=
          match v with
          | [] => b0
          | Vector.cons _ a0 n1 w => (a0 :: fold_right_fix n1 w b0)%list
          end) n B []%list)
    with (Vector.to_list B).

    (* now the fold_right subterm matches Htail *)
    rewrite <- Htail.
    cbn.
    reflexivity.
Qed.

Lemma swaps_parity_cocycle :
  forall n (A B C : Mask n),
    xorb (swaps_parity A B) (swaps_parity (mask_xor A B) C)
    =
    xorb (swaps_parity B C) (swaps_parity A (mask_xor B C)).
Proof.
  induction n as [|n IH]; intros A B C.
  - dependent destruction A. dependent destruction B. dependent destruction C.
    cbn. reflexivity.
  - dependent destruction A. dependent destruction B. dependent destruction C.

    (* Expand the outer cons/cons swaps_parity *)
    repeat rewrite swaps_parity_cons.

    (* Put mask_xor into cons form so swaps_parity_cons applies again *)
    repeat rewrite mask_xor_cons.

    (* Expand the remaining swaps_parity *)
    repeat rewrite swaps_parity_cons.

    (* Push parity through xor *)
    repeat rewrite parity_mask_xor.

    (* At this point your goal is essentially the one you pasted:
       S ⊕ (X ⊕ (T ⊕ Y)) = U ⊕ (X' ⊕ (V ⊕ Y'))
       where S,T,U,V are tail swaps_parity terms. *)

    (* Name the relevant pieces to make rewriting predictable *)
    set (S := swaps_parity A B).
    set (T := swaps_parity (mask_xor A B) C).
    set (U := swaps_parity B C).
    set (V := swaps_parity A (mask_xor B C)).

    set (X  := h0 && parity_mask A).
    set (Y  := h1 && xorb (parity_mask A) (parity_mask B)).
    set (X' := h1 && parity_mask B).
    set (Y' := xorb h0 h1 && parity_mask A).
    
    (* normalize both sides to right-associated form so later regrouping is predictable *)
    rewrite <- xorb_assoc.  (* (S ⊕ X) ⊕ (T ⊕ Y)  ->  S ⊕ (X ⊕ (T ⊕ Y)) *)
    rewrite <- xorb_assoc.  (* (U ⊕ X') ⊕ (V ⊕ Y') -> U ⊕ (X' ⊕ (V ⊕ Y')) *)


    (* Regroup LHS into (S⊕T) ⊕ (X⊕Y) *)
    rewrite (xorb_assoc X T Y).          (* X ⊕ (T ⊕ Y) -> (X ⊕ T) ⊕ Y *)
    rewrite (xorb_comm X T).             (* (X ⊕ T) -> (T ⊕ X) *)
    rewrite <- (xorb_assoc T X Y).       (* (T ⊕ X) ⊕ Y -> T ⊕ (X ⊕ Y) *)
    rewrite (xorb_assoc S T (xorb X Y)). (* S ⊕ (T ⊕ ...) -> (S ⊕ T) ⊕ ... *)

    (* Regroup RHS into (U⊕V) ⊕ (X'⊕Y') *)
    rewrite (xorb_assoc X' V Y').
    rewrite (xorb_comm X' V).
    rewrite <- (xorb_assoc V X' Y').
    rewrite (xorb_assoc U V (xorb X' Y')).
    
    subst S T U V.

    (* Now we can use IH to turn the left (swaps_parity A B ⊕ swaps_parity (A⊕B) C)
       into (swaps_parity B C ⊕ swaps_parity A (B⊕C)) so both sides share the same left-xor. *)
    rewrite (IH A B C).

    (* cancel common left-xor without needing negb_inj *)
    assert (xorb_cancel_l_bool : forall a b c : bool, xorb a b = xorb a c -> b = c).
    { intros a b c; destruct a; cbn; intro H.
      - (* a = true : goal is negb b = negb c, just case-split b,c *)
        destruct b, c; cbn in H; try discriminate; reflexivity.
      - (* a = false *)
        exact H.
    }

    apply (xorb_cancel_l_bool (xorb (swaps_parity B C) (swaps_parity A (mask_xor B C)))).

    (* Now only the head/parity identity remains *)
    subst X Y X' Y'.
    destruct h, h0, h1; cbn;
    destruct (parity_mask A), (parity_mask B); cbn; reflexivity.
Qed.

(*B. Metric side: metric_factor cocycle*)

Lemma metric_factor_cons :
  forall n (h : Q) (sq : Vector.t Q n) (a b : bool) (A B : Mask n),
    metric_factor (Vector.cons Q h n sq)
                  (Vector.cons bool a n A)
                  (Vector.cons bool b n B)
    ==
    (if andb a b then h else 1%Q) * metric_factor sq A B.
Proof.
  intros n h sq a b A B.
  unfold metric_factor.
  repeat rewrite to_list_cons.
  cbn.
  reflexivity.
Qed.

Lemma metric_factor_cocycle :
  forall n (sq : Vector.t Q n) (A B C : Mask n),
    (metric_factor sq A B * metric_factor sq (mask_xor A B) C)%Q
    ==
    (metric_factor sq B C * metric_factor sq A (mask_xor B C))%Q.
Proof.
  induction n as [|n IH]; intros sq A B C.
  - dependent destruction sq.
    dependent destruction A; dependent destruction B; dependent destruction C.
    cbn [metric_factor mask_xor]. cbn. reflexivity.
  - dependent destruction sq.
    dependent destruction A; dependent destruction B; dependent destruction C.

    (* expose the xor heads so metric_factor_cons applies cleanly *)
    rewrite mask_xor_cons.
    rewrite mask_xor_cons.

    (* peel metric_factor one step on each occurrence *)
    rewrite (@metric_factor_cons n h sq h0 h1 A B).
    rewrite (@metric_factor_cons n h sq (xorb h0 h1) h2 (mask_xor A B) C).
    rewrite (@metric_factor_cons n h sq h1 h2 B C).
    rewrite (@metric_factor_cons n h sq h0 (xorb h1 h2) A (mask_xor B C)).

    (* Now everything is “head scalars” times tail metric_factors.
       Factor to isolate the IH subterm. *)
    set (H1 := if h0 && h1 then h else 1%Q).
    set (H2 := if xorb h0 h1 && h2 then h else 1%Q).
    set (H3 := if h1 && h2 then h else 1%Q).
    set (H4 := if h0 && xorb h1 h2 then h else 1%Q).

    set (mAB := metric_factor sq A B).
    set (mX  := metric_factor sq (mask_xor A B) C).
    set (mBC := metric_factor sq B C).
    set (mA  := metric_factor sq A (mask_xor B C)).

    (* turn (H1*mAB)*(H2*mX) into (H1*H2)*(mAB*mX), and similarly on RHS *)
    setoid_replace ((H1 * mAB) * (H2 * mX))%Q with ((H1 * H2) * (mAB * mX))%Q by ring.
    setoid_replace ((H3 * mBC) * (H4 * mA))%Q with ((H3 * H4) * (mBC * mA))%Q by ring.

    (* now IH matches exactly on (mAB*mX) *)
    subst mAB mX mBC mA.
    rewrite (IH sq A B C).

    (* remaining goal is purely about the head booleans *)
    subst H1 H2 H3 H4.
    destruct h0, h1, h2; cbn; ring.
Qed.

Lemma basis_mul_assoc_coeff :
  forall n sq (A B C : Mask n),
    (basis_mul_coeff sq A B * basis_mul_coeff sq (basis_mul_mask A B) C)%Q
    ==
    (basis_mul_coeff sq B C * basis_mul_coeff sq A (basis_mul_mask B C))%Q.
Proof.
  intros n sq A B C.
  unfold basis_mul_coeff, basis_mul_mask.
  cbn.

  (* Expand basis_mul_mask = mask_xor if that's your definition *)
  (* If basis_mul_mask is already mask_xor, this does nothing; otherwise keep it. *)

  (* Reassociate/commute so the sgnQ factors are adjacent on each side *)
  (* LHS: s1 * m1 * (s2 * m2)  ==>  (s1*s2) * (m1*m2) *)
  set (s1 := sgnQ (swaps_parity A B)).
  set (s2 := sgnQ (swaps_parity (mask_xor A B) C)).
  set (m1 := metric_factor sq A B).
  set (m2 := metric_factor sq (mask_xor A B) C).

  set (t1 := sgnQ (swaps_parity B C)).
  set (t2 := sgnQ (swaps_parity A (mask_xor B C))).
  set (n1 := metric_factor sq B C).
  set (n2 := metric_factor sq A (mask_xor B C)).

  (* Now rewrite the whole goal in these names so "ring" can rearrange cleanly under == *)
  change ((s1 * m1 * (s2 * m2))%Q == (t1 * n1 * (t2 * n2))%Q).

  (* Turn each side into (s1*s2)*(m1*m2) form *)
  setoid_replace (s1 * m1 * (s2 * m2))%Q with ((s1 * s2) * (m1 * m2))%Q by ring.
  setoid_replace (t1 * n1 * (t2 * n2))%Q with ((t1 * t2) * (n1 * n2))%Q by ring.

  (* Now apply the two cocycles *)
  (* 1) sign cocycle: (sgnQ p)*(sgnQ q) = sgnQ (xorb p q) and then swaps_parity_cocycle *)
  (* Use sgnQ_xorb only once per side after we have adjacency. *)
  subst s1 s2 t1 t2 m1 m2 n1 n2.

  (* Rewrite each adjacent sign product into sgnQ(xorb ...) *)
  repeat rewrite sgnQ_xorb.

  (* Reduce to showing the xorb arguments match and the metric products match *)
  (* We'll use swaps_parity_cocycle and metric_factor_cocycle. *)
  (* The signs are now: sgnQ (xorb (swaps_parity A B) (swaps_parity (mask_xor A B) C)) etc. *)
  (* So we rewrite inside with swaps_parity_cocycle *)
  rewrite swaps_parity_cocycle.

  (* Metric part is exactly metric_factor_cocycle *)
  rewrite metric_factor_cocycle.

  ring.
Qed.

Lemma mv_gp_assoc :
  forall n sq (F G H : MV n) (U : Mask n),
    mv_gp n sq (mv_gp n sq F G) H U
    ==
    mv_gp n sq F (mv_gp n sq G H) U.
Proof.
  (* Expand coefficients; use basis_mul_assoc_coeff inside the triple sums;
     ext is already “baked in” since we're proving pointwise at U. *)
Admitted.
