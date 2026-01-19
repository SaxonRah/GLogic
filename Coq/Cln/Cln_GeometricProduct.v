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

(* Scalar 1 (the empty blade) *)
Definition mv_one {n} : MV n := basis (mask_empty (n:=n)).

(* ============================================================ *)
(* Basic algebra laws: bilinear + identity                        *)
(* ============================================================ *)
From Coq Require Import FunctionalExtensionality.

Require Import QArith.
Require Import Qcanon.

Lemma Qmult_plus_distr_r_eq : forall x y z : Q,
  (x + y) * z = x * z + y * z.
Proof.
  intros x y z.
  (* Convert to canonical form where Qeq becomes = *)
  assert (H: (x + y) * z == x * z + y * z) by apply Qmult_plus_distr_r.
  (* Now we need Qeq to = conversion *)
  unfold Qeq in H.
  simpl in H.
  (* This is tricky - let's just compute both sides *)
  destruct x as [xn xd], y as [yn yd], z as [zn zd].
  unfold Qplus, Qmult in *.
  simpl in *.
  (* Now H tells us the numerators are equal when cross-multiplied *)
  (* We need to show the fractions are equal *)
  apply Qeq_eqR.
  exact H.
Qed.

Lemma Qmult_assoc_eq : forall x y z : Q,
  (x * y) * z = x * (y * z).
Proof.
  intros.
  setoid_replace ((x * y) * z) with (x * (y * z)).
  - reflexivity.
  - apply Qmult_assoc.
Qed.

Lemma mv_gp_add_l :
  forall n (sq : Vector.t Q n) (F1 F2 G : MV n),
    @mv_gp n sq (mv_add F1 F2) G
    =
    mv_add (@mv_gp n sq F1 G) (@mv_gp n sq F2 G).
Proof.
  intros n sq F1 F2 G.
  apply functional_extensionality; intro U.
  unfold mv_gp, mv_add.
  
  assert (H1: forall l,
    sumQ (List.map
      (fun A => sumQ (List.map
        (fun B => if mask_eq_dec (basis_mul_mask A B) U
                  then (F1 A + F2 A) * G B * basis_mul_coeff sq A B
                  else 0) (all_masks n))) l) =
    (sumQ (List.map
      (fun A => sumQ (List.map
        (fun B => if mask_eq_dec (basis_mul_mask A B) U
                  then F1 A * G B * basis_mul_coeff sq A B
                  else 0) (all_masks n))) l) +
     sumQ (List.map
      (fun A => sumQ (List.map
        (fun B => if mask_eq_dec (basis_mul_mask A B) U
                  then F2 A * G B * basis_mul_coeff sq A B
                  else 0) (all_masks n))) l))).
  {
    intro l.
    induction l as [|A tl IH]; simpl.
    - reflexivity.
    - rewrite IH. clear IH.
      assert (H2: forall l2,
        sumQ (List.map
          (fun B => if mask_eq_dec (basis_mul_mask A B) U
                    then (F1 A + F2 A) * G B * basis_mul_coeff sq A B
                    else 0) l2) =
        (sumQ (List.map
          (fun B => if mask_eq_dec (basis_mul_mask A B) U
                    then F1 A * G B * basis_mul_coeff sq A B
                    else 0) l2) +
         sumQ (List.map
          (fun B => if mask_eq_dec (basis_mul_mask A B) U
                    then F2 A * G B * basis_mul_coeff sq A B
                    else 0) l2))).
      {
        intro l2.
        induction l2 as [|B tl2 IH2]; simpl.
        - reflexivity.
        - rewrite IH2.
          destruct (mask_eq_dec (basis_mul_mask A B) U).
          + (* Case: masks equal, need to distribute and rearrange *)
            (* Goal: (F1 A + F2 A) * G B * c + (s1 + s2) = F1 A * G B * c + s1 + (F2 A * G B * c + s2) *)
            set (c := basis_mul_coeff sq A B).
            set (s1 := sumQ (List.map _ tl2)).
            set (s2 := sumQ (List.map _ tl2)).
            (* Manually build the proof *)
            assert (Hdist: (F1 A + F2 A) * G B * c = F1 A * G B * c + F2 A * G B * c).
            { unfold c.
              destruct (F1 A) as [n1 d1], (F2 A) as [n2 d2], (G B) as [ng dg], (basis_mul_coeff sq A B) as [nc dc].
              unfold Qplus, Qmult. simpl.
              f_equal.
              - (* Numerator equality *)
                rewrite !Pos2Z.inj_mul.
                nia.
              - (* Denominator equality *)
                reflexivity.
            }
            rewrite Hdist.
            (* Now: (x + y) + (s1 + s2) = x + s1 + (y + s2) *)
            (* This is just associativity and commutativity *)
            rewrite <- !Qplus_assoc.
            f_equal.
            rewrite !Qplus_assoc.
            rewrite (Qplus_comm (F2 A * G B * c) s1).
            rewrite <- !Qplus_assoc.
            reflexivity.
          + (* Case: masks not equal, both sides have 0 *)
            reflexivity.
      }
      rewrite H2.
      (* Outer rearrangement: s + (t1 + t2) = (s + t1) + t2 *)
      rewrite !Qplus_assoc.
      f_equal.
      rewrite <- !Qplus_assoc.
      f_equal.
      apply Qplus_comm.
  }
  apply (H1 (all_masks n)).
Qed.

Lemma mv_gp_add_r :
  forall n sq (F G1 G2 : MV n),
    mv_gp n sq F (mv_add G1 G2)
    =
    mv_add (mv_gp n sq F G1) (mv_gp n sq F G2).
Proof.
  intros n sq F G1 G2.
  apply functional_extensionality; intro U.
  unfold mv_gp, mv_add.
  
  apply sumQ_map_ext; intros A HA.
  rewrite <- sumQ_map_add.
  apply sumQ_map_ext; intros B HB.
  
  destruct (mask_eq_dec (basis_mul_mask A B) U).
  - ring.
  - ring.
Qed.

Lemma mv_gp_scale_l :
  forall n sq (k : Q) (F G : MV n),
    mv_gp n sq (mv_scale k F) G
    =
    mv_scale k (mv_gp n sq F G).
Proof.
  intros n sq k F G.
  apply functional_extensionality; intro U.
  unfold mv_gp, mv_scale.
  
  rewrite <- sumQ_map_scale_l.
  apply sumQ_map_ext; intros A HA.
  rewrite <- sumQ_map_scale_l.
  apply sumQ_map_ext; intros B HB.
  
  destruct (mask_eq_dec (basis_mul_mask A B) U).
  - ring.
  - ring.
Qed.

Lemma mv_gp_scale_r :
  forall n sq (k : Q) (F G : MV n),
    mv_gp n sq F (mv_scale k G)
    =
    mv_scale k (mv_gp n sq F G).
Proof.
  intros n sq k F G.
  apply functional_extensionality; intro U.
  unfold mv_gp, mv_scale.
  
  apply sumQ_map_ext; intros A HA.
  rewrite <- sumQ_map_scale_l.
  apply sumQ_map_ext; intros B HB.
  
  destruct (mask_eq_dec (basis_mul_mask A B) U).
  - ring.
  - ring.
Qed.

Lemma mv_gp_one_l :
  forall n sq (F : MV n),
    mv_gp n sq mv_one F = F.
Proof.
  (* Reduce to basis multiplication with empty mask:
       xor(empty,B)=B, swaps(empty,B)=0, metric(empty,B)=1 *)
Admitted.

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
