(*
  ============================================================
  File: Cln_BooleanEmbedding.v
  ============================================================

  Boolean embedding into Cl(n):

    Π(a)   = 2^{-n} Σ_S χ_S(a) e_S   ∈ Cl(n)
          = 2^{-n} ∏_{i=1}^n (1 + a_i e_i)  (fixed increasing order)

    Embed(f) = Σ_{a∈{±1}^n} f(a) Π(a) ∈ Cl(n)

  Correctness:
    Eval_s(Embed(f)) = f(s)  (as Q: 0 or 1)

  This is exactly “Boolean logic is geometric”: every boolean function
  is a single multivector in Cl(n), and evaluation is a geometric-linear
  functional on corners.
*)
Require Import Cln_Basis.
Require Import Cln_Multivector.
Require Import Coq.Program.Equality.

From Coq Require Import List.
From Coq Require Import Bool.
From Coq Require Import Arith.
From Coq Require Import QArith.
From Coq Require Import Vectors.Vector.
From Coq Require Import ZArith.
From Coq Require Import Ring.

From Coq Require Import Setoid.
From Coq Require Import Morphisms.

Import ListNotations.

From Coq Require Import QArith.Qring.
Open Scope Q_scope.


Set Implicit Arguments.

Lemma map_map :
  forall (A B C : Type) (f : B -> C) (g : A -> B) (l : list A),
    List.map f (List.map g l) = List.map (fun x => f (g x)) l.
Proof.
  intros A B C f g l.
  induction l as [|a tl IH]; simpl.
  - reflexivity.
  - rewrite IH. reflexivity.
Qed.

Lemma Qadd_factor_r :
  forall x k : Q, x + k * x == (1 + k) * x.
Proof.
  intros x k.
  apply Qeq_trans with ((1%Q * x) + (k * x))%Q.
  - apply Qplus_comp.
    + symmetry. apply Qmult_1_l.
    + reflexivity.
  - (* now show 1*x + k*x == (1+k)*x *)
    symmetry.
    apply Qmult_plus_distr_l.
Qed.

Lemma Qmul_assoc3 :
  forall a b c : Q, (a * b) * c == a * (b * c).
Proof.
  intros a b c.
  rewrite Qmult_assoc.
  reflexivity.
Qed.

(* ============================================================ *)
(* Boolean-to-Q                                                  *)
(* ============================================================ *)

Definition bQ (b : bool) : Q := if b then 1%Q else 0%Q.

(* ============================================================ *)
(* The projector Π(a)                                            *)
(* ============================================================ *)

Definition Pi {n} (a : Corner n) : MV n :=
  fun m => ((1%Q / pow2 n) * chi m a)%Q.

(* ============================================================ *)
(* Walsh orthogonality over all masks                            *)
(*   S(a,s) = Σ_m χ_m(a) χ_m(s) = 2^n if a=s else 0             *)
(* ============================================================ *)

Definition walsh_sum_masks {n} (a s : Corner n) : Q :=
  sumQ (List.map (fun m : Mask n => (chi' m a * chi' m s)%Q) (all_masks n)).

Lemma walsh_sum_masks_0 :
  forall (a s : Corner 0),
    walsh_sum_masks a s == 1%Q.
Proof.
  intros a s.
  unfold walsh_sum_masks.
  dependent destruction a; dependent destruction s.
  simpl. ring.
Qed.

Lemma walsh_sum_masks_S :
  forall n (a s : Corner (S n)),
    walsh_sum_masks a s ==
    ((1%Q + (sQ (Vector.hd a) * sQ (Vector.hd s))%Q)
       * walsh_sum_masks (Vector.tl a) (Vector.tl s))%Q.
Proof.
  intros n a s.
  unfold walsh_sum_masks.
  simpl (all_masks (S n)).
  set (ms := all_masks n).

  (* all_masks (S n) = map (false::) ms ++ map (true::) ms *)
  rewrite List.map_app.
  rewrite sumQ_app.

  (* false-head contribution *)
  assert (Hfalse :
    sumQ
      (List.map
         (fun m0 : Mask n =>
            (chi' (n := S n) (Vector.cons bool false n m0) a
             * chi' (n := S n) (Vector.cons bool false n m0) s)%Q)
         ms)
    ==
    walsh_sum_masks (Vector.tl a) (Vector.tl s)).
  {
    unfold walsh_sum_masks.
    apply sumQ_map_ext; intros m0 Hm0.
    rewrite (@chi_false_cons n m0 (Vector.tl a) (Vector.hd a)).
    rewrite (@chi_false_cons n m0 (Vector.tl s) (Vector.hd s)).
    simpl. ring.
  }

  (* true-head contribution *)
  assert (Htrue :
    sumQ
      (List.map
         (fun m0 : Mask n =>
            (chi' (n := S n) (Vector.cons bool true n m0) a
             * chi' (n := S n) (Vector.cons bool true n m0) s)%Q)
         ms)
    ==
    ((sQ (Vector.hd a) * sQ (Vector.hd s))%Q
       * walsh_sum_masks (Vector.tl a) (Vector.tl s))%Q).
  {
    unfold walsh_sum_masks.

    (* Step 1: rewrite each term using chi_true_cons *)
    apply Qeq_trans with
      (sumQ
         (List.map
            (fun m0 : Mask n =>
               ((sQ (Vector.hd a) * chi' (n:=n) m0 (Vector.tl a))
                *
                (sQ (Vector.hd s) * chi' (n:=n) m0 (Vector.tl s)))%Q)
            ms)).
    - apply sumQ_map_ext; intros m0 Hm0.
      rewrite (@chi_true_cons n m0 (Vector.tl a) (Vector.hd a)).
      rewrite (@chi_true_cons n m0 (Vector.tl s) (Vector.hd s)).
      simpl. ring.
    - (* Step 2: regroup to k * (chi' m0 tl a * chi' m0 tl s) *)
      apply Qeq_trans with
        (sumQ
           (List.map
              (fun m0 : Mask n =>
                 ((sQ (Vector.hd a) * sQ (Vector.hd s))%Q
                    * (chi' (n:=n) m0 (Vector.tl a) * chi' (n:=n) m0 (Vector.tl s))%Q)%Q)
              ms)).
      + apply sumQ_map_ext; intros m0 Hm0; simpl; ring.
      + (* Step 3: factor k out using sumQ_map_scale_l *)
        unfold ms.
        rewrite <- (@sumQ_map_scale_l (Mask n)
                  (sQ (Vector.hd a) * sQ (Vector.hd s))%Q
                  (fun m0 : Mask n =>
                     (chi' (n:=n) m0 (Vector.tl a) * chi' (n:=n) m0 (Vector.tl s))%Q)
                  (all_masks n)).
        reflexivity.
  }

  (* Normalize nested maps so Hfalse/Htrue match *)
  assert (HmapF :
    List.map (fun m : Mask (S n) => chi' m a * chi' m s)
             (List.map (fun m0 : Vector.t bool n => Vector.cons bool false n m0) ms)
    =
    List.map (fun m0 : Vector.t bool n =>
                chi' (Vector.cons bool false n m0) a
                * chi' (Vector.cons bool false n m0) s)
             ms).
  {
    exact (@map_map (Vector.t bool n) (Mask (S n)) Q
            (fun m : Mask (S n) => chi' m a * chi' m s)
            (fun m0 : Vector.t bool n => Vector.cons bool false n m0)
            ms).
  }

  assert (HmapT :
    List.map (fun m : Mask (S n) => chi' m a * chi' m s)
             (List.map (fun m0 : Vector.t bool n => Vector.cons bool true n m0) ms)
    =
    List.map (fun m0 : Vector.t bool n =>
                chi' (Vector.cons bool true n m0) a
                * chi' (Vector.cons bool true n m0) s)
             ms).
  {
    exact (@map_map (Vector.t bool n) (Mask (S n)) Q
            (fun m : Mask (S n) => chi' m a * chi' m s)
            (fun m0 : Vector.t bool n => Vector.cons bool true n m0)
            ms).
  }

  (* Force goal to the exact lambda shapes above (binder-name stability) *)
  change (
    sumQ (List.map (fun m : Mask (S n) => chi' m a * chi' m s)
           (List.map (fun m0 : Vector.t bool n => Vector.cons bool false n m0) ms))
    +
    sumQ (List.map (fun m : Mask (S n) => chi' m a * chi' m s)
           (List.map (fun m0 : Vector.t bool n => Vector.cons bool true n m0) ms))
    ==
    (1 + sQ (Vector.hd a) * sQ (Vector.hd s))
      * sumQ (List.map (fun m : Mask n => chi' m (Vector.tl a) * chi' m (Vector.tl s)) ms)
  ).

  rewrite HmapF, HmapT.
  simpl.
  rewrite Hfalse, Htrue.

  (* turn RHS sum into walsh_sum_masks (tl a) (tl s) *)
  unfold walsh_sum_masks.
  fold ms.
  change (sumQ (List.map (fun m : Mask n => chi' m (Vector.tl a) * chi' m (Vector.tl s)) ms))
    with (walsh_sum_masks (Vector.tl a) (Vector.tl s)).

  (* finish: X + k*X == (1+k)*X *)
  apply Qadd_factor_r.
Qed.

Lemma walsh_sum_masks_closed :
  forall n (a s : Corner n),
    walsh_sum_masks a s == (if corner_eqb a s then pow2 n else 0%Q).
Proof.
  induction n; intros a s.
  - (* n = 0 *)
    dependent destruction a; dependent destruction s.
    simpl. reflexivity.
  - (* n = S n *)
    rewrite walsh_sum_masks_S.
    specialize (IHn (Vector.tl a) (Vector.tl s)).

    destruct (sign_eq_dec (Vector.hd a) (Vector.hd s)) as [Heq|Hneq].
    + (* heads equal *)
      replace (sQ (Vector.hd a) * sQ (Vector.hd s))%Q with 1%Q.
      2: { rewrite Heq. destruct (Vector.hd s); reflexivity. }
      
      destruct (corner_eqb (Vector.tl a) (Vector.tl s)) eqn:Htail.
      * (* tails equal -> a = s *)
        apply corner_eqb_eq in Htail.
        assert (Hall : a = s).
        { dependent destruction a; dependent destruction s; simpl in *.
          subst. f_equal. }
        rewrite Hall, corner_eqb_refl.
        simpl (pow2 (S n)).
        rewrite Htail in IHn.
        rewrite IHn.
        simpl. ring.
      * (* tails differ -> a <> s *)
        dependent destruction a; dependent destruction s; simpl in *.
        rewrite andb_false_intro2 by exact Htail.
        rewrite IHn.
        ring.
    + (* heads differ -> (1 + sQ(hd a)*sQ(hd s)) = 0 *)
      dependent destruction a; dependent destruction s; simpl.
      rewrite andb_false_intro1 by 
        (destruct h, h0; simpl in *; try reflexivity; contradiction).
      destruct h, h0; simpl in *; try contradiction; ring.
Qed.

(* ============================================================ *)
(* Π-delta lemma: Eval_s(Π(a)) = 1 if a=s else 0                 *)
(* ============================================================ *)

Lemma Pi_delta :
  forall n (a s : Corner n),
    eval (Pi a) s == (if corner_eqb a s then 1%Q else 0%Q).
Proof.
  intros n a s.
  unfold eval, Pi.
  change (chi) with (@chi') in *.
  
  (* First, rewrite the goal to the right form *)
  apply Qeq_trans with 
    (y := sumQ (List.map (fun m : Mask n => ((1%Q / pow2 n)%Q * (chi' m a * chi' m s))%Q) (all_masks n))).
  - (* Show the two forms are equal by associativity *)
    apply sumQ_map_ext; intros m Hm; simpl; ring.
  - (* Now apply the factoring lemma - FORWARD direction *)
    rewrite sumQ_map_scale_l.
    apply Qeq_trans with (y := (1%Q / pow2 n)%Q * walsh_sum_masks a s).
    + unfold walsh_sum_masks.
      reflexivity.
    + rewrite walsh_sum_masks_closed.
      destruct (corner_eqb a s) eqn:Heq.
      * (* equals *)
        simpl. field. apply pow2_nonzero.
      * (* not equals *)
        simpl. ring.
Qed.

(* ============================================================ *)
(* Embedding                                                     *)
(* ============================================================ *)

Definition embed {n} (f : Corner n -> bool) : MV n :=
  fun m => sumQ (List.map (fun a => (bQ (f a) * Pi a m)%Q) (all_corners n)).

Theorem embed_correct :
  forall n (f : Corner n -> bool) (s : Corner n),
    eval (embed f) s == bQ (f s).
Proof.
  intros n f s.
  unfold embed.
  unfold eval.
  
  (* Step 1: swap sums by induction over corners list *)
  assert (Hswap :
    eval (fun m => sumQ (List.map (fun a => (bQ (f a) * Pi a m)%Q) (all_corners n))) s
    ==
    sumQ (List.map (fun a => eval (mv_scale (bQ (f a)) (Pi a)) s) (all_corners n))).
  {
    specialize (@eval_sum_over_corners n (all_corners n) 
      (fun a m => (bQ (f a) * Pi a m)%Q) s).
    intro H0.
    refine (Qeq_trans _ _ _ H0 _).
    apply sumQ_map_ext; intros a Ha.
    unfold eval, mv_scale, Pi.
    apply sumQ_map_ext; intros m Hm.
    simpl. ring.
  }

  rewrite Hswap.
  apply Qeq_trans with
    (sumQ (List.map (fun a => (bQ (f a) * eval (Pi a) s)%Q) (all_corners n))).
  - apply sumQ_map_ext; intros a Ha.
    rewrite eval_scale. ring.
  - (* Use Pi_delta *)
    apply Qeq_trans with
      (sumQ (List.map (fun a => (bQ (f a) * (if corner_eqb a s then 1%Q else 0%Q))%Q) (all_corners n))).
    + apply sumQ_map_ext; intros a Ha.
      rewrite Pi_delta. reflexivity.
    + (* Collapse the sum *)
      set (cs := all_corners n).
      assert (Hnd : List.NoDup cs) by (subst cs; apply all_corners_nodup).
      assert (Hin : List.In s cs) by (subst cs; apply all_corners_complete).
      clearbody cs.
      revert Hnd Hin.
      induction cs as [|a tl IH]; intros Hnd Hin; simpl in *.
      { contradiction. }
      inversion Hnd as [|a' tl' Hnotin Hnd_tl]; subst a' tl'.
      simpl in Hin. destruct Hin as [Hs|Hin_tl].
      * subst a.
        rewrite corner_eqb_refl.
        assert (Hrest :
          sumQ (List.map (fun x : Corner n => (bQ (f x) * (if corner_eqb x s then 1%Q else 0%Q))%Q) tl)
          == 0%Q).
        {
          apply Qeq_trans with (sumQ (List.map (fun x => 0%Q) tl)).
          - apply sumQ_map_ext; intros x Hinx.
            destruct (corner_eqb x s) eqn:Heq.
            + apply corner_eqb_eq in Heq. subst x. contradiction.
            + ring.
          - apply sumQ_map_zero.
        }
        rewrite Hrest. ring.
      * destruct (corner_eqb a s) eqn:Heq.
        { apply corner_eqb_eq in Heq. subst a. contradiction. }
        specialize (IH Hnd_tl Hin_tl).
        rewrite IH. ring.
Qed.
