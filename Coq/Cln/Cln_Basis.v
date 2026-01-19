(*
  ============================================================
  File: Cln_Basis.v
  ============================================================

  Basic n-dimensional “hypercube” infrastructure:

    - Sign = {+1, -1}
    - Corner n = length-n vector of Sign   (points in {±1}^n)
    - Mask   n = length-n vector of bool   (basis-blade index / subset)

    - Enumerations all_corners n, all_masks n
    - Decidable equality and boolean equality tests
    - Facts: completeness + NoDup for enumerations
*)

From Coq Require Import List.
From Coq Require Import Bool.
From Coq Require Import Arith.
From Coq Require Import Vectors.Vector.
Import ListNotations.
Import VectorNotations.

Require Import Coq.Program.Equality.

Set Implicit Arguments.

(* ============================================================ *)
(* Signs and corners                                             *)
(* ============================================================ *)

Inductive Sign : Type := Pos | Neg.

Definition sign_eqb (a b : Sign) : bool :=
  match a, b with
  | Pos, Pos => true
  | Neg, Neg => true
  | _, _ => false
  end.

Lemma sign_eqb_refl : forall s, sign_eqb s s = true.
Proof. destruct s; reflexivity. Qed.

Lemma sign_eqb_eq : forall a b, sign_eqb a b = true -> a = b.
Proof. destruct a, b; simpl; intros H; try discriminate; reflexivity. Qed.

Lemma sign_eq_dec : forall (a b : Sign), {a = b} + {a <> b}.
Proof. decide equality. Qed.

Definition Corner (n : nat) : Type := Vector.t Sign n.
Definition Mask   (n : nat) : Type := Vector.t bool n.

Definition corner0 : Corner 0 := Vector.nil Sign.

(* Decidable equality on vectors (Corners and Masks) *)
Fixpoint corner_eq_dec {n : nat} (x y : Corner n) : {x = y} + {x <> y}.
Proof.
  destruct n as [|n'].
  - dependent destruction x.
    dependent destruction y.
    left; reflexivity.
  - dependent destruction x. (* h : Sign, x : Corner n' *)
    dependent destruction y. (* h0 : Sign, y : Corner n' *)
    destruct (sign_eq_dec h h0) as [Hh|Hh].
    + subst h0.
      destruct (corner_eq_dec n' x y) as [Ht|Ht].
      * subst y. left; reflexivity.
      * right.
        intro Heq.
        (* turn h::x = h::y into x = y *)
        inversion Heq as [H0].
        (* H0 is the annoying existT equality; eliminate it *)
        dependent destruction H0.
        (* now goal is False and we have x=y *)
        apply Ht; reflexivity.
    + right.
      intro Heq.
      inversion Heq.
      apply Hh; assumption.
Defined.

Fixpoint mask_eq_dec {n : nat} (x y : Mask n)   : {x = y} + {x <> y}.
Proof.
  destruct n as [|n'].
  - dependent destruction x.
    dependent destruction y.
    left; reflexivity.
  - dependent destruction x. (* h : bool, x : Mask n' *)
    dependent destruction y. (* h0 : bool, y : Mask n' *)
    destruct (Bool.bool_dec h h0) as [Hh|Hh].
    + subst h0.
      destruct (mask_eq_dec n' x y) as [Ht|Ht].
      * subst y. left; reflexivity.
      * right.
        intro Heq.
        inversion Heq as [H0].
        dependent destruction H0.
        apply Ht; reflexivity.
    + right.
      intro Heq.
      inversion Heq.
      apply Hh; assumption.
Defined.


Fixpoint corner_eqb {n} : Corner n -> Corner n -> bool :=
  match n with
  | 0 => fun _ _ => true
  | S n' =>
      fun a b =>
        andb (sign_eqb (Vector.hd a) (Vector.hd b))
             (corner_eqb (Vector.tl a) (Vector.tl b))
  end.

Lemma corner_eqb_refl : forall n (c : Corner n), corner_eqb c c = true.
Proof.
  induction n; intros c.
  - dependent destruction c; reflexivity.
  - dependent destruction c; simpl.
    rewrite sign_eqb_refl, IHn; reflexivity.
Qed.

Lemma corner_eqb_eq : forall n (a b : Corner n),
  corner_eqb a b = true -> a = b.
Proof.
  induction n; intros a b H.
  - dependent destruction a; dependent destruction b; reflexivity.
  - dependent destruction a; dependent destruction b; simpl in H.
    apply andb_true_iff in H as [Hh Ht].
    apply sign_eqb_eq in Hh.
    specialize (IHn _ _ Ht).
    subst.
    reflexivity.
Qed.

(* ============================================================ *)
(* Enumerations                                                  *)
(* ============================================================ *)

Fixpoint all_masks (n : nat) : list (Mask n) :=
  match n with
  | 0 => [ Vector.nil bool ]
  | S n' =>
      let ms := all_masks n' in
      (List.map (fun m => false :: m) ms)
        ++
      (List.map (fun m => true :: m) ms)
  end.

Fixpoint all_corners (n : nat) : list (Corner n) :=
  match n with
  | 0 => [ Vector.nil Sign ]
  | S n' =>
      let cs := all_corners n' in
      (List.map (fun c => Pos :: c) cs)
        ++
      (List.map (fun c => Neg :: c) cs)
  end.

(* ============================================================ *)
(* List lemmas used repeatedly                                   *)
(* ============================================================ *)

Lemma in_map_iff' :
  forall (A B : Type) (f : A -> B) (x : B) (l : list A),
    List.In x (List.map f l) <-> exists y, List.In y l /\ f y = x.
Proof.
  intros A B f x l; split; intro H.
  - apply List.in_map_iff in H.
    destruct H as [y [Hy1 Hy2]].
    exists y; split; auto.
  - destruct H as [y [Hy1 Hy2]].
    subst.
    apply List.in_map.
    exact Hy1.
Qed.

(* Injectivity of cons on vectors *)
Lemma cons_inj_sign :
  forall n (s1 s2 : Sign) (t1 t2 : Corner n),
    (s1 :: t1 = s2 :: t2) -> s1 = s2 /\ t1 = t2.
Proof.
  intros n s1 s2 t1 t2 H.
  dependent destruction H.
  split; reflexivity.
Qed.

Lemma cons_inj_bool :
  forall n (b1 b2 : bool) (t1 t2 : Mask n),
    (b1 :: t1 = b2 :: t2) -> b1 = b2 /\ t1 = t2.
Proof.
  intros n b1 b2 t1 t2 H.
  dependent destruction H.
  split; reflexivity.
Qed.

(* ============================================================ *)
(* Completeness and NoDup of all_masks                            *)
(* ============================================================ *)

Lemma NoDup_map_inj :
  forall (A B : Type) (f : A -> B) (l : list A),
    (forall x y, f x = f y -> x = y) ->
    List.NoDup l ->
    List.NoDup (List.map f l).
Proof.
  intros A B f l Hinj Hnd.
  induction Hnd as [|a l Hnotin Hnd IH]; simpl.
  - constructor.
  - constructor.
    + intro HIn.
      (* use stable lemma in_map_iff' *)
      apply (proj1 (in_map_iff' f (f a) l)) in HIn.
      destruct HIn as [x [HxIn Hfx]].
      (* HxIn : In x l, Hfx : f x = f a *)
      assert (x = a) by (apply Hinj; exact Hfx).
      subst x.
      contradiction.
    + exact IH.
Qed.


Lemma all_masks_complete :
  forall n (m : Mask n), List.In m (all_masks n).
Proof.
  induction n; intros m.
  - dependent destruction m. simpl. left. reflexivity.
  - dependent destruction m.
    simpl.
    destruct h.
    + (* head = true -> in right half *)
      apply List.in_or_app. right.
      apply List.in_map. apply IHn.
    + (* head = false -> in left half *)
      apply List.in_or_app. left.
      apply List.in_map. apply IHn.
Qed.

Lemma all_masks_nodup :
  forall n, List.NoDup (all_masks n).
Proof.
  induction n.
  - simpl. constructor.
    + intro H. inversion H.
    + constructor.
  - simpl.
    set (ms := all_masks n).
    assert (Hnd : List.NoDup ms) by apply IHn.

    (* NoDup of each half using injective map *)
    assert (HndF : List.NoDup (List.map (fun m => false :: m) ms)).
    {
      apply NoDup_map_inj.
      - intros x y Hxy.
        apply cons_inj_bool in Hxy as [_ Ht]. exact Ht.
      - exact Hnd.
    }

    assert (HndT : List.NoDup (List.map (fun m => true :: m) ms)).
    {
      apply NoDup_map_inj.
      - intros x y Hxy.
        apply cons_inj_bool in Hxy as [_ Ht]. exact Ht.
      - exact Hnd.
    }

    (* Disjointness: nothing in left half is in right half *)
    assert (Hdis : forall x,
        List.In x (List.map (fun m => false :: m) ms) ->
        ~ List.In x (List.map (fun m => true :: m) ms)).
    {
      intros x Hx Hin.
      (* use stable lemma in_map_iff' *)
      apply (proj1 (in_map_iff' (fun m => false :: m) x ms)) in Hx.
      destruct Hx as [m1 [Hm1 HxEq]].
      apply (proj1 (in_map_iff' (fun m => true :: m) x ms)) in Hin.
      destruct Hin as [m2 [Hm2 HinEq]].
      subst x.
      (* now HinEq : true::m2 = false::m1, contradiction *)
      inversion HinEq.
    }

    (* Prove NoDup of concatenation *)
    apply List.NoDup_app.
    + exact HndF.
    + exact HndT.
    + exact Hdis.
Qed.


(* ============================================================ *)
(* Completeness and NoDup of all_corners                           *)
(* ============================================================ *)

Lemma all_corners_complete :
  forall n (c : Corner n), List.In c (all_corners n).
Proof.
  induction n; intros c.
  - dependent destruction c. simpl. left. reflexivity.
  - dependent destruction c.
    simpl.
    destruct h.
    + (* head = Pos -> in left half *)
      apply List.in_or_app. left.
      apply List.in_map. apply IHn.
    + (* head = Neg -> in right half *)
      apply List.in_or_app. right.
      apply List.in_map. apply IHn.
Qed.

Lemma all_corners_nodup :
  forall n, List.NoDup (all_corners n).
Proof.
  induction n.
  - simpl. constructor.
    + intro H. inversion H.
    + constructor.
  - simpl.
    set (cs := all_corners n).
    assert (Hnd : List.NoDup cs) by apply IHn.

    (* NoDup of each half using injective map *)
    assert (HndP : List.NoDup (List.map (fun c => Pos :: c) cs)).
    {
      apply NoDup_map_inj.
      - intros x y Hxy.
        apply cons_inj_sign in Hxy as [_ Ht]. exact Ht.
      - exact Hnd.
    }

    assert (HndN : List.NoDup (List.map (fun c => Neg :: c) cs)).
    {
      apply NoDup_map_inj.
      - intros x y Hxy.
        apply cons_inj_sign in Hxy as [_ Ht]. exact Ht.
      - exact Hnd.
    }

    (* Disjointness: nothing in left half is in right half *)
    assert (Hdis : forall x,
        List.In x (List.map (fun c => Pos :: c) cs) ->
        ~ List.In x (List.map (fun c => Neg :: c) cs)).
    {
      intros x Hx Hin.
      (* use stable lemma in_map_iff' *)
      apply (proj1 (in_map_iff' (fun c => Pos :: c) x cs)) in Hx.
      destruct Hx as [c1 [Hc1 HxEq]].
      apply (proj1 (in_map_iff' (fun c => Neg :: c) x cs)) in Hin.
      destruct Hin as [c2 [Hc2 HinEq]].
      subst x.
      (* now HinEq : Neg::c2 = Pos::c1, contradiction *)
      inversion HinEq.
    }

    (* Prove NoDup of concatenation *)
    apply List.NoDup_app.
    + exact HndP.
    + exact HndN.
    + exact Hdis.
Qed.


