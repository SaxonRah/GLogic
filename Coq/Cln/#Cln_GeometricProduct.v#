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

  (* Let the “outer” summand be a function of A *)
  set (outer :=
    fun A : Mask n =>
      sumQ (List.map (fun B : Mask n =>
        let c := basis_mul_coeff sq A B in
        if mask_eq_dec (basis_mul_mask A B) U
        then ((if mask_eq_dec A (mask_empty (n:=n)) then 1%Q else 0%Q) * F B * c)%Q
        else 0%Q
      ) (all_masks n))).

  (* Rewrite the whole thing into Σ_A outer(A) *)
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
  (* This is definitionally outer *)
  fold outer.

  (* Now use your “Kronecker pick” lemma on A, picking A = empty *)
  eapply Qeq_trans.
  2: {
    (* after picking A=empty we’ll show it equals F U *)
    (* so we keep going below *)
    exact (@sumQ_all_masks_pick n outer mask_empty).



    exact (@sumQ_all_masks_pick n outer (mask_empty (n:=n))).
  }

  (* We still owe: outer(empty) == F U *)
  unfold outer.
  (* Simplify mv_one at A=empty and basis_mul_coeff/mask at empty *)
  (* First, kill the (if A=empty then 1 else 0) by evaluating at empty *)
  destruct (mask_eq_dec (mask_empty (n:=n)) (mask_empty (n:=n))) as [_|Hbad].
  2:{ exfalso; apply Hbad; reflexivity. }
  cbn.

  (* Rewrite basis_mul_mask empty B = B *)
  apply sumQ_map_ext; intros B HB.
  rewrite (mask_xor_empty_l (n:=n) B).  (* basis_mul_mask is mask_xor *)
  rewrite (basis_mul_coeff_empty_l (n:=n) sq B).
  (* Now each term is: if B=U then F B else 0 *)
  destruct (mask_eq_dec B U) as [Heq|Hneq].
  - subst. ring.  (* 1*F U*1 = F U *)
  - ring.
Qed.


Lemma mv_gp_one_r :
  forall n sq (F : MV n),
    mv_gp n sq F mv_one = F.
Proof. Admitted.

(* ============================================================ *)
(* Closed form on basis blades                                    *)
(* ============================================================ *)

Lemma mv_gp_basis :
  forall n sq (A B : Mask n),
    mv_gp n sq (basis A) (basis B)
    =
    mv_scale (basis_mul_coeff n sq A B)
             (basis (basis_mul_mask A B)).
Proof.
  (* Unfold mv_gp and basis; only one (A,B) term survives by eq_dec. *)
Admitted.

Lemma mv_gp_basis :
  forall n sq (A B : Mask n),
    mv_gp n sq (basis A) (basis B)
    =
    mv_scale (basis_mul_coeff sq A B)
             (basis (basis_mul_mask A B)).
Proof.
Admitted.

(* ============================================================ *)
(* Singleton masks = generators e_i                               *)
(* ============================================================ *)

Fixpoint mask_single (n : nat) : Fin.t n -> Mask n :=
  match n with
  | 0 => fun i => match i with end
  | S k =>
      fun i =>
        match i with
        | Fin.F1 =>
            true :: Vector.const false k
        | Fin.FS j =>
            false :: mask_single k j
        end
  end.

Definition e (n : nat) (i : Fin.t n) : MV n := basis (mask_single n i).

(* ============================================================ *)
(* Clifford relations on generators                               *)
(* ============================================================ *)

Lemma e_square :
  forall n (sq : Vector.t Q n) (i : Fin.t n),
    mv_gp n sq (e n i) (e n i)
    =
    mv_scale (Vector.nth sq i) mv_one.
Proof.
  (* Use mv_gp_basis with A=B=single i.
     - xor = empty
     - swaps_parity(single,single)=0
     - metric_factor contributes sq_i once (since overlap at i) *)
Admitted.

Lemma e_anticomm :
  forall n (sq : Vector.t Q n) (i j : Fin.t n),
    i <> j ->
    mv_gp n sq (e n i) (e n j)
    =
    mv_scale (-1)%Q (mv_gp n sq (e n j) (e n i)).
Proof.
  (* Reduce with mv_gp_basis; show:
       xor(single i, single j) = xor(single j, single i)
     and metric_factor symmetric (no overlap),
     but swaps_parity flips by 1 when i≠j. *)
Admitted.

(* ============================================================ *)
(* Associativity                                                  *)
(* ============================================================ *)

(*
  The heart is associativity on basis blades:
     (e_A e_B) e_C = e_A (e_B e_C)

  which amounts to the 2-cocycle identity for:
     cocycle(A,B) := basis_mul_coeff n sq A B

  Once you have basis associativity, lift to MV by bilinearity.
*)

Lemma basis_mul_assoc_coeff :
  forall n sq (A B C : Mask n),
    (basis_mul_coeff n sq A B * basis_mul_coeff n sq (basis_mul_mask A B) C)%Q
    ==
    (basis_mul_coeff n sq B C * basis_mul_coeff n sq A (basis_mul_mask B C))%Q.
Proof.
  (* Prove by induction on n using the recurrences for swaps_parity and metric_factor.
     This is the “cocycle law”. *)
Admitted.

Lemma mv_gp_assoc :
  forall n sq (F G H : MV n),
    mv_gp n sq (mv_gp n sq F G) H
    =
    mv_gp n sq F (mv_gp n sq G H).
Proof.
  (* Expand coefficients; use basis_mul_assoc_coeff inside the triple sums;
     ext U; rearrange finite sums (all_masks complete/nodup already in your basis file). *)
Admitted.
