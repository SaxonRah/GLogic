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


(*
  ============================================================
  File: Cln_Multivector.v
  ============================================================

  Multivectors in Cl(n) (coefficient model) over rationals:

    MV n := Mask n -> Q    (coefficients on blades e_S)

  Also includes:
    * basic linear structure ( 0, +, scalar * )
    * finite sums over enumerations
    * evaluation functional Eval_s(F) = Σ_S F_S χ_S(s)

  This file is purely linear-algebraic, but the indexing is
  explicitly by GA basis blades (masks).
*)

From Coq Require Import List.
From Coq Require Import Bool.
From Coq Require Import Arith.
From Coq Require Import QArith.
From Coq Require Import Vectors.Vector.
From Coq Require Import Setoid.
From Coq Require Import Morphisms.
From Coq Require Import Ring.

Import ListNotations.

From Coq Require Import QArith.Qring.
Open Scope Q_scope.


Set Implicit Arguments.

(* ============================================================ *)
(* Multivectors as coefficient functions                          *)
(* ============================================================ *)

Definition MV (n : nat) : Type := Mask n -> Q.

Definition mv_zero {n} : MV n := fun _ => 0%Q.
Definition mv_add  {n} (F G : MV n) : MV n := fun m => (F m + G m)%Q.
Definition mv_scale {n} (k : Q) (F : MV n) : MV n := fun m => (k * F m)%Q.

Infix "⊕" := mv_add (at level 50, left associativity).

(* Basis blade e_S as Kronecker delta coefficient function *)
Definition basis {n} (S : Mask n) : MV n :=
  fun T => if mask_eq_dec T S then 1%Q else 0%Q.

(* ============================================================ *)
(* Finite sums over lists                                         *)
(* ============================================================ *)

Fixpoint sumQ (xs : list Q) : Q :=
  match xs with
  | List.nil => 0%Q
  | List.cons x tl => (x + sumQ tl)%Q
  end.

Lemma sumQ_app : forall xs ys,
  sumQ (xs ++ ys) == (sumQ xs + sumQ ys)%Q.
Proof.
  induction xs; intros ys; simpl.
  - ring.
  - rewrite IHxs. ring.
Qed.

Lemma sumQ_map_const0 :
  forall (A : Type) (l : list A),
    sumQ (List.map (fun _ : A => 0%Q) l) == 0%Q.
Proof.
  intros A l.
  induction l as [|a tl IH]; simpl.
  - apply Qeq_refl.
  - (* 0 + sumQ(...) == 0 *)
    rewrite IH.
    (* 0 + 0 == 0 *)
    rewrite Qplus_0_l.
    apply Qeq_refl.
Qed.

Lemma sumQ_map_ext :
  forall (A : Type) (f g : A -> Q) (l : list A),
    (forall x, List.In x l -> f x == g x) ->
    sumQ (List.map f l) == sumQ (List.map g l).
Proof.
  intros A f g l.
  induction l as [|a tl IH]; intros H; simpl.
  - reflexivity.
  - (* head *)
    apply Qplus_comp.
    + apply H. left; reflexivity.
    + apply IH. intros x Hx. apply H. right; exact Hx.
Qed.

Lemma sumQ_map_add :
  forall (A : Type) (f g : A -> Q) (l : list A),
    sumQ (List.map (fun x => (f x + g x)%Q) l) ==
    (sumQ (List.map f l) + sumQ (List.map g l))%Q.
Proof.
  induction l as [|a tl IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

Lemma sumQ_map_scale_l :
  forall (A : Type) (k : Q) (f : A -> Q) (l : list A),
    sumQ (List.map (fun x => (k * f x)%Q) l) ==
    (k * sumQ (List.map f l))%Q.
Proof.
  induction l as [|a tl IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

(* ============================================================ *)
(* Characters χ_S(s): product of selected coordinates              *)
(* ============================================================ *)

Definition sQ (s : Sign) : Q :=
  match s with
  | Pos => 1%Q
  | Neg => (-1)%Q
  end.

Lemma sQ_sq1 : forall s, (sQ s * sQ s)%Q == 1%Q.
Proof. destruct s; simpl; ring. Qed.

Lemma sQ_mul_eqb :
  forall a b,
    (sQ a * sQ b)%Q == (if sign_eqb a b then 1%Q else (-1)%Q).
Proof.
  destruct a, b; simpl; reflexivity.
Qed.

Fixpoint chi (n : nat) : Mask n -> Corner n -> Q :=
  match n with
  | O => fun _ _ => 1%Q
  | S n' =>
      fun m s =>
        let mh := Vector.hd m in
        let mt := Vector.tl m in
        let sh := Vector.hd s in
        let st := Vector.tl s in
        ((if mh then sQ sh else 1%Q) * (@chi n' mt st))%Q
  end.

(* ============================================================ *)
(* Chi lemmas (use chi')                                         *)
(* ============================================================ *)

Definition chi' {n : nat} : Mask n -> Corner n -> Q := @chi n.

Lemma chi_false_cons :
  forall n (m : Mask n) (s : Corner n) (h : Sign),
    chi' (n := S n) (Vector.cons bool false n m)
                   (Vector.cons Sign h n s)
    ==
    chi' (n := n) m s.
Proof.
  intros n m s h.
  unfold chi', chi. simpl.
  ring.
Qed.

Lemma chi_true_cons :
  forall n (m : Mask n) (s : Corner n) (h : Sign),
    chi' (n := S n) (Vector.cons bool true n m)
                   (Vector.cons Sign h n s)
    ==
    (sQ h * chi' (n := n) m s)%Q.
Proof.
  intros n m s h.
  unfold chi', chi. simpl.
  ring.
Qed.


(* ============================================================ *)
(* pow2                                                          *)
(* ============================================================ *)

Fixpoint pow2 (n : nat) : Q :=
  match n with
  | O => 1%Q
  | S k => (2%Q * pow2 k)%Q
  end.

Lemma pow2_nonzero : forall n, ~(pow2 n == 0%Q).
Proof.
  induction n; simpl.
  - (* pow2 0 = 1 *)
    intro H; discriminate.
  - intro H.
    apply Qmult_integral in H.
    destruct H as [H2|Hk].
    + discriminate.
    + apply IHn; exact Hk.
Qed.

(* ============================================================ *)
(* Evaluation functional                                          *)
(* ============================================================ *)

Definition eval {n} (F : MV n) (s : Corner n) : Q :=
  sumQ (List.map (fun m => (F m * chi' m s)%Q) (all_masks n)).

Lemma eval_add : forall n (F G : MV n) (s : Corner n),
  eval (mv_add F G) s == (eval F s + eval G s)%Q.
Proof.
  intros n F G s.
  unfold eval, mv_add.
  rewrite <- sumQ_map_add.
  apply sumQ_map_ext; intros m Hm; simpl; ring.
Qed.

Lemma eval_scale : forall n (k : Q) (F : MV n) (s : Corner n),
  eval (mv_scale k F) s == (k * eval F s)%Q.
Proof.
  intros n k F s.
  unfold eval, mv_scale.
  rewrite <- sumQ_map_scale_l.
  apply sumQ_map_ext; intros m Hm; simpl; ring.
Qed.

(* ============================================================ *)
(* Swap finite sums over corners (Fubini)                         *)
(* ============================================================ *)

(* Helper: sum of a list of zeros is zero *)
Lemma sumQ_map_zero :
  forall (A : Type) (l : list A),
    sumQ (List.map (fun _ => 0%Q) l) == 0%Q.
Proof.
  intros A l.
  induction l as [|a tl IH]; simpl.
  - reflexivity.
  - rewrite IH. simpl. reflexivity.
Qed.

Lemma eval_sum_over_corners :
  forall n (cs : list (Corner n)) (H : Corner n -> MV n) (s : Corner n),
    eval (fun m => sumQ (List.map (fun a => H a m) cs)) s ==
    sumQ (List.map (fun a => eval (H a) s) cs).
Proof.
  intros n cs.
  induction cs as [|a tl IH]; intros H s; simpl.
  - (* cs = [] *)
    unfold eval. simpl.
    (* LHS: sumQ (map (fun m => 0 * chi) all_masks) == 0 *)
    apply Qeq_trans with (sumQ (List.map (fun _ : Mask n => 0%Q) (all_masks n))).
    + apply sumQ_map_ext; intros m Hm; simpl; reflexivity.
    + apply sumQ_map_zero.
  - (* cs = a :: tl *)
    unfold eval. simpl.
    (* Distribute the multiplication across the pointwise sum *)
    apply Qeq_trans with
      (sumQ (List.map
               (fun m : Mask n =>
                  ((H a m + sumQ (List.map (fun a0 : Corner n => H a0 m) tl)) * chi' m s)%Q)
               (all_masks n))).
    + reflexivity.
    + (* Split into two sums using sumQ_map_add after rewriting the mapped function *)
      apply Qeq_trans with
        (sumQ (List.map (fun m : Mask n => (H a m * chi' m s)%Q) (all_masks n)) +
         sumQ (List.map (fun m : Mask n =>
                           (sumQ (List.map (fun a0 : Corner n => H a0 m) tl) * chi' m s)%Q)
                        (all_masks n)))%Q.
      * (* use sumQ_map_add *)
        rewrite <- sumQ_map_add.
        apply sumQ_map_ext; intros m Hm; simpl.
        (* (x+y)*z = x*z + y*z *)
        ring.
      * (* Now identify the first term as eval(H a) s and the second via IH *)
        rewrite <- (IH H s).
        unfold eval. simpl.
        ring.
Qed.


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

Lemma mask0_eta : forall (C : Mask 0), C = [].
Proof.
  intro C.
  dependent destruction C.
  reflexivity.
Qed.


Lemma mask_xor_assoc :
  forall n (A B C : Mask n),
    mask_xor (mask_xor A B) C = mask_xor A (mask_xor B C).
Proof.
  induction n as [|n IH]; intros A B C.
  - (* n = 0 *)
    rewrite (mask0_eta A).
    rewrite (mask0_eta B).
    rewrite (mask0_eta C).
    cbn [mask_xor]. reflexivity.
  - (* n = S n *)
    dependent destruction A.
    dependent destruction B.
    dependent destruction C.
    rewrite !mask_xor_cons.
    f_equal.
    + rewrite <- xorb_assoc; reflexivity.
    + apply IH.
Qed.

Lemma mask_xor_cancel_l :
  forall n (X Y : Mask n),
    mask_xor X Y = X -> Y = mask_empty (n:=n).
Proof.
  intros n X Y H.
  (* Left-xor both sides by X *)
  assert (H' : mask_xor X (mask_xor X Y) = mask_xor X X).
  { now rewrite H. }
  (* Reassociate the LHS: X ⊕ (X ⊕ Y) = (X ⊕ X) ⊕ Y *)
  rewrite <- (@mask_xor_assoc n X X Y) in H'.
  (* Simplify X ⊕ X to empty on both sides (RHS and inside LHS) *)
  rewrite (@mask_xor_self n X) in H'.
  (* Now H' is: mask_xor empty Y = empty, so reduce to Y = empty *)
  rewrite (@mask_xor_empty_l n Y) in H'.
  exact H'.
Qed.

(*

## Updated Phase Outline (current state of `Cln_Full.v`)

### Phase 0 — Core infrastructure - (YES)

**Goal:** have a clean finite model of masks, enumeration, and finite sums over `Q`.

What's already done in `Cln_Full.v`:

* `Mask n` as `Vector.t bool n`, plus:

  * `mask_empty`, `mask_single`, `mask_xor`, `mask_eq_dec`
  * key algebra on masks: **`mask_xor_self`**, **`mask_xor_empty_l/r`**, and you also added

    * **`mask_xor_assoc`** - (YES)
    * **`mask_xor_cancel_l`** - (YES) (useful for uniqueness-ish arguments)
* `all_masks n` enumeration, with:

  * **`all_masks_complete`** - (YES)
  * **`all_masks_nodup`** - (YES)
* Rational finite-sum layer:

  * `sumQ`, `sumQ_app`
  * map interaction lemmas:

    * **`sumQ_map_ext`**, **`sumQ_map_add`**, **`sumQ_map_scale_l`**, **`sumQ_map_zero`**, **`sumQ_map_const0`**
  * "Kronecker-pick" lemma:

    * **`sumQ_all_masks_pick`** - (YES)
  * 2D sum swap ("Fubini"):

    * **`sumQ_fubini`** - (YES)

This phase is *complete*.

---

### Phase 1 — Clifford algebra core in `Cln_Full.v` - (YES) except associativity

**Goal:** define multivectors, define geometric product, prove it's the intended algebra.

Already done:

* **Basis + multivectors**
  * `MV n := Mask n -> Q`
  * `basis : Mask n -> MV n`
  * `mv_add`, `mv_scale`, etc.
  
* **Geometric product definition** - (YES)
  * `mv_gp n sq F G : MV n` defined as the double-sum over
            `A,B ∈ all_masks n` with the `mask_eq_dec (basis_mul_mask A B) U` filter.
            
* **Basis multiplication machinery** - (YES)
  * `basis_mul_mask` and `basis_mul_coeff`
  * `swaps_parity` + `sgnQ`
  * `metric_factor`
  
* **Cocycle / associativity-at-the-coefficient-level** - (YES)
  * **`swaps_parity_cocycle`** - (YES)
  * **`metric_factor_cocycle`** - (YES)
  * **`basis_mul_assoc_coeff`** - (YES) (this is the thing you previously labelled "Tier 3: cocycle identity")
  
* **Everything in old "Tier 1 / Tier 2" is already handled in this combined file**
      (i.e., bilinearity/identity/basis closed form/Clifford relations).
      There are **no other `Admitted`** besides the final associativity lemma.

Remaining (the only blocker):

* **`mv_gp_assoc`** (NO) **(Admitted)**

  ```coq
  Lemma mv_gp_assoc :
    forall n (sq : Vector.t Q n) (F G H : MV n) (U : Mask n),
      mv_gp sq (mv_gp sq F G) H U
      ==
      mv_gp sq F (mv_gp sq G H) U.
  ```

So Phase 1 is "complete up to the final algebra law".

---

## What "Phase 1 completion" now really means

Once `mv_gp_assoc` is proved, you can legitimately treat `MV n` with `mv_add`, `mv_scale`, `mv_one`, and `mv_gp`
  as a **(rational) associative algebra**, and then everything you want to do later
  ("representation/computation equivalence" work) can be built on top of an actually-checked algebraic core.

---

## Updated next phases (post-`mv_gp_assoc`)

### Phase 2 — "Boolean embedding" interface (already present, but now you can build theorems cleanly)

**Status:** the embedding machinery exists; what's missing are *theorems about it* that rely on associativity.

* Prove the embedding's key algebraic invariants you'll use later:

  * `embed` linearity facts (if you model it that way)
  * injectivity (if true for your chosen embedding)
  * characterization of which multivectors are "Boolean-embeddable"

### Phase 3 — Representation-vs-composition theorems (your "composition failure")

This becomes clean once `⋆` is associative:

* "Not a subalgebra": there exist Boolean-embedded `F,G` such that `F⋆G` is not Boolean-embedded.
* Specific counterexamples (AND-square, etc.) become routine corollaries once the algebra laws are in place.

### Phase 4 — Structural / spectral analysis layer (Walsh/Fourier dual viewpoint)

This is where you connect to your "representation complexity" story:

* grade-support theorems, parity/top-grade behavior, etc.
* comparisons between "Fourier support" and "geometric-product support" measures

(But all of that is downstream; right now, `mv_gp_assoc` is the gate.)

---

## What remains to finish `mv_gp_assoc` (in terms of *your current file

This is not a new phase, just the concrete subplan for the last lemma, using lemmas you already have in `Cln_Full.v`:

1. **Unfold `mv_gp` twice** on each side.

   * LHS becomes a nested sum where `(mv_gp F G) X` is itself a double-sum.

2. **Use `sumQ_fubini` repeatedly** to flatten/reorder into a triple sum over `(A,B,C)` (conceptually: "sum over A,B,C of the unique term that lands in U").

3. **Eliminate the `if mask_eq_dec ... then ... else 0` filters** using `sumQ_all_masks_pick`.

   * This is the key move that turns "double sums with filters" into "single term with substituted index".

4. After reindexing, both sides reduce to the same triple-sum skeleton, differing only by:

   * the **mask-parenthesization**:
          `(A⊕B)⊕C` vs `A⊕(B⊕C)` [solve by **`mask_xor_assoc`**], and
   * the **coefficient-parenthesization**:
          `coeff(A,B)*coeff(A⊕B,C)` vs `coeff(B,C)*coeff(A,B⊕C)` [solve by **`basis_mul_assoc_coeff`**].

That's it. You already proved the two hard "local" identities (`mask_xor_assoc` and `basis_mul_assoc_coeff`)
    that make the global associativity proof go through.
*)

(* ============================================================ *)
(*  Associativity helpers: normalize each side to a triple sum    *)
(* ============================================================ *)

Local Opaque mask_eq_dec.

Lemma if_mask_eq_dec_sym :
  forall n (X Y : Mask n) (a b : Q),
    (if mask_eq_dec X Y then a else b)
    ==
    (if mask_eq_dec Y X then a else b).
Proof.
  intros n X Y a b.
  destruct (mask_eq_dec X Y) as [HXY|HXY].
  - (* X = Y *)
    destruct (mask_eq_dec Y X) as [HYX|HYX].
    + (* Y = X *)
      reflexivity.
    + (* Y <> X : contradiction *)
      exfalso; apply HYX; symmetry; exact HXY.
  - (* X <> Y *)
    destruct (mask_eq_dec Y X) as [HYX|HYX].
    + (* Y = X : contradiction *)
      exfalso; apply HXY; symmetry; exact HYX.
    + (* Y <> X *)
      reflexivity.
Qed.

Definition mv_gp_triple (n : nat) (sq : Vector.t Q n)
  (F G H : MV n) (U : Mask n) : Q :=
  sumQ (List.map (fun A =>
  sumQ (List.map (fun B =>
  sumQ (List.map (fun C =>
    if mask_eq_dec (mask_xor (mask_xor A B) C) U
    then (F A * G B * H C
          * basis_mul_coeff sq A B
          * basis_mul_coeff sq (mask_xor A B) C)%Q
    else 0%Q) (all_masks n))) (all_masks n))) (all_masks n)).

Lemma sumQ_map_scale_r :
  forall (A : Type) (k : Q) (f : A -> Q) (xs : list A),
    sumQ (List.map (fun x => (f x * k)%Q) xs)
    ==
    (sumQ (List.map f xs) * k)%Q.
Proof.
  intros A k f xs.
  induction xs as [|x xs IH]; cbn; ring_simplify; try ring.
  (* cbn gives: (f x * k) + sumQ(map (fun ...) xs) == (f x + sumQ(map f xs)) * k *)
  rewrite IH. ring.
Qed.

Lemma sumQ_map_push_const :
  forall (A : Type) (xs : list A) (k : Q) (f : A -> Q),
    sumQ (List.map (fun x => (k * f x)%Q) xs)
    ==
    (k * sumQ (List.map f xs))%Q.
Proof.
  intros A xs k f.
  induction xs as [|x xs IH].
  - cbn. ring.
  - cbn. (* goal: k*f x + sumQ(map ...) == k*(f x + sumQ(map ...)) *)
    rewrite IH. ring.
Qed.

Lemma sumQ_all_masks_pick_eq :
  forall n (f : Mask n -> Q) (U : Mask n),
    sumQ (List.map (fun X => if mask_eq_dec X U then f X else 0%Q) (all_masks n))
    == f U.
Proof.
  intros n f U.
  apply sumQ_all_masks_pick.
Qed.

Lemma sumQ_all_masks_pick_eq_xor :
  forall n (f : Mask n -> Q) (A B : Mask n),
    sumQ (List.map (fun X => if mask_eq_dec X (mask_xor A B) then f X else 0%Q) (all_masks n))
    == f (mask_xor A B).
Proof.
  intros n f A B.
  apply sumQ_all_masks_pick_eq.
Qed.

Lemma mv_gp_assoc_LHS_quad :
  forall n (sq : Vector.t Q n) (F G H : MV n) (U : Mask n),
    mv_gp sq (mv_gp sq F G) H U
    ==
    sumQ (List.map (fun X : Mask n =>
      sumQ (List.map (fun C : Mask n =>
        if mask_eq_dec (mask_xor X C) U
        then
          (sumQ (List.map (fun A : Mask n =>
             sumQ (List.map (fun B : Mask n =>
               if mask_eq_dec (mask_xor A B) X
               then (F A * G B * basis_mul_coeff sq A B)%Q
               else 0%Q) (all_masks n))) (all_masks n))
           * H C * basis_mul_coeff sq X C)%Q
        else 0%Q
      ) (all_masks n))
    ) (all_masks n)).
Proof.
  intros n sq F G H U.
  unfold mv_gp.
  (* after unfolding, basis_mul_mask is definitional mask_xor *)
  cbn [basis_mul_mask].
  reflexivity.
Qed.

Definition K_LHS {n} (sq : Vector.t Q n) (F G H : MV n) (U : Mask n)
  (X C A B : Mask n) : Q :=
  if mask_eq_dec (mask_xor X C) U then
    if mask_eq_dec (mask_xor A B) X then
      (F A * G B * H C
       * basis_mul_coeff sq A B
       * basis_mul_coeff sq X C)%Q
    else 0%Q
  else 0%Q.
  
Lemma mv_gp_assoc_LHS_quad_kernel :
  forall n (sq : Vector.t Q n) (F G H : MV n) (U : Mask n),
    mv_gp sq (mv_gp sq F G) H U
    ==
    sumQ (List.map (fun X : Mask n =>
      sumQ (List.map (fun C : Mask n =>
        sumQ (List.map (fun A : Mask n =>
          sumQ (List.map (fun B : Mask n =>
            K_LHS sq F G H U X C A B) (all_masks n)))
        (all_masks n)))
      (all_masks n)))
    (all_masks n)).
Proof.
  intros n sq F G H U.
  unfold mv_gp.

  (* rename outer indices by extensionality: A->X, B->C *)
  eapply Qeq_trans.
  - apply (sumQ_map_ext (A:=Mask n)); intros X HX.
    apply (sumQ_map_ext (A:=Mask n)); intros C HC.
    apply Qeq_refl.
  - (* work pointwise in X,C *)
    apply (sumQ_map_ext (A:=Mask n)); intros X HX.
    apply (sumQ_map_ext (A:=Mask n)); intros C HC.

    (* expose the outer filter on X,C and normalize basis_mul_mask *)
    cbn [basis_mul_mask].

    destruct (mask_eq_dec (mask_xor X C) U) as [Hhit|Hmiss].
    + (* hit: U = X⊕C *)
      subst U.

      (* kill the trivial if that remained from the original definition form *)
      change (basis_mul_mask X C) with (mask_xor X C).
      destruct (mask_eq_dec (mask_xor X C) (mask_xor X C)) as [_|Hbad]; [|contradiction].
      cbn.

      (* normalize inner filter basis_mul_mask A B -> mask_xor A B *)
      cbn [basis_mul_mask].

      set (k := (H C * basis_mul_coeff sq X C)%Q).

      (* rewrite sumAB * H C * coeffXC into sumAB * k *)
      setoid_replace
        ((sumQ
            (List.map
               (fun A : Mask n =>
                sumQ
                  (List.map
                     (fun B : Mask n =>
                      if mask_eq_dec (mask_xor A B) X
                      then (F A * G B * basis_mul_coeff sq A B)%Q
                      else 0%Q)
                     (all_masks n))) (all_masks n)) * H C * basis_mul_coeff sq X C)%Q)
      with
        ((sumQ
            (List.map
               (fun A : Mask n =>
                sumQ
                  (List.map
                     (fun B : Mask n =>
                      if mask_eq_dec (mask_xor A B) X
                      then (F A * G B * basis_mul_coeff sq A B)%Q
                      else 0%Q)
                     (all_masks n))) (all_masks n)) * k)%Q)
      by (unfold k; ring).

      (* push k into outer A-sum *)
      rewrite <- (@sumQ_map_scale_r (Mask n) k
        (fun A : Mask n =>
           sumQ (List.map (fun B : Mask n =>
             if mask_eq_dec (mask_xor A B) X
             then (F A * G B * basis_mul_coeff sq A B)%Q
             else 0%Q) (all_masks n)))
        (all_masks n)).

      (* push k into each inner B-sum *)
      apply (sumQ_map_ext (A:=Mask n)); intros A HA.
      rewrite <- (@sumQ_map_scale_r (Mask n) k
        (fun B : Mask n =>
           if mask_eq_dec (mask_xor A B) X
           then (F A * G B * basis_mul_coeff sq A B)%Q
           else 0%Q)
        (all_masks n)).

      (* now prove pointwise equality with K_LHS *)
      apply (sumQ_map_ext (A:=Mask n)); intros B HB.
      unfold K_LHS.

      (* outer test in K_LHS is reflexive after subst U *)
      destruct (mask_eq_dec (mask_xor X C) (mask_xor X C)) as [_|Hbad2]; [|contradiction].
      cbn.

      destruct (mask_eq_dec (mask_xor A B) X) as [Hab|Hab].
      * (* inner hit: both sides are the same product *)
        unfold k.
        ring.
      * (* inner miss: both sides are 0 *)
        ring.

    + (* miss: outer filter false *)
      (* LHS: the outer indicator is false *)
      (* miss branch: mask_xor X C <> U *)
      destruct (mask_eq_dec (mask_xor X C) U) as [Heq|Hne]; [contradiction|].
      destruct (mask_eq_dec (basis_mul_mask X C) U) as [Heq|Hneq].

      * (* then-branch: derive contradiction from Hne *)
        exfalso.
        apply Hne.  (* your hypothesis: mask_xor X C <> U *)
        (* convert Heq : basis_mul_mask X C = U into mask_xor X C = U *)
        cbn [basis_mul_mask] in Heq.
        exact Heq.
      * (* else-branch: LHS reduces to 0 *)
        cbn.

        unfold K_LHS.

        (* show the outer if is false using Hmiss *)
        destruct (mask_eq_dec (mask_xor X C) U) as [Heq|Hneq'].
        -- exfalso. exact (Hmiss Heq).
        
        --
          (* remaining goal: 0 == (double sum of zeros) *)
          assert (sumQ_zeros :
                    forall (l : list (Mask n)),
                      sumQ (List.map (fun _ : Mask n => 0%Q) l) == 0%Q).
          { intro l; induction l as [|x tl IH]; cbn.
            - ring.
            - rewrite IH; ring.
          }

          set (As := all_masks n).

          (* inner constant: sumQ (map (fun _ => 0) As) == 0 *)
          pose proof (sumQ_zeros As) as Hin.

          (* Replace the outer sum of constants by a sum of zeros, using sumQ_map_ext *)
          eapply Qeq_trans.
          2: {
            (* now goal will be: 0 == sumQ (map (fun _ => 0) As) *)
            apply Qeq_sym.
            (* goal is: 0 == sumQ (map (fun _ => inner) As) *)
            (* now: sumQ (map (fun _ => inner) As) == 0 *)

            (* Pick the intermediate explicitly, so no ?y evar appears. *)
            eapply Qeq_trans with
              (y := sumQ (List.map (fun _ : Mask n => 0%Q) As)).

            - (* outer constant-sum == sum of zeros *)
              apply (sumQ_map_ext (A := Mask n)); intros _ _.
              exact Hin.

            - (* sum of zeros == 0 *)
              exact (sumQ_zeros As).
          }
          reflexivity.
Qed.

Lemma mv_gp_assoc_LHS_triple :
  forall n (sq : Vector.t Q n) (F G H : MV n) (U : Mask n),
    mv_gp sq (mv_gp sq F G) H U
    ==
    sumQ (List.map (fun A =>
    sumQ (List.map (fun B =>
    sumQ (List.map (fun C =>
      if mask_eq_dec (mask_xor (mask_xor A B) C) U
      then (F A * G B * H C
            * basis_mul_coeff sq A B
            * basis_mul_coeff sq (mask_xor A B) C)%Q
      else 0%Q) (all_masks n))) (all_masks n))) (all_masks n)).
Proof.
Admitted.

Lemma mv_gp_assoc_RHS_triple :
  forall n (sq : Vector.t Q n) (F G H : MV n) (U : Mask n),
    mv_gp sq F (mv_gp sq G H) U
    ==
    sumQ (List.map (fun A =>
    sumQ (List.map (fun B =>
    sumQ (List.map (fun C =>
      if mask_eq_dec (mask_xor A (mask_xor B C)) U
      then (F A * G B * H C
            * basis_mul_coeff sq B C
            * basis_mul_coeff sq A (mask_xor B C))%Q
      else 0%Q) (all_masks n))) (all_masks n))) (all_masks n)).
Proof.
Admitted.

Lemma mv_gp_assoc :
  forall n (sq : Vector.t Q n) (F G H : MV n) (U : Mask n),
    mv_gp sq (mv_gp sq F G) H U
    ==
    mv_gp sq F (mv_gp sq G H) U.
Proof.
  intros n sq F G H U.
  eapply Qeq_trans.
  - apply mv_gp_assoc_LHS_triple.
  - eapply Qeq_trans.
    2: { symmetry. apply mv_gp_assoc_RHS_triple. }
    apply (sumQ_map_ext (A:=Mask n)); intros A HA.
    apply (sumQ_map_ext (A:=Mask n)); intros B HB.
    apply (sumQ_map_ext (A:=Mask n)); intros C HC.
    rewrite mask_xor_assoc.
    destruct (mask_eq_dec (mask_xor A (mask_xor B C)) U); [|reflexivity].
    (* Now both sides are products differing only in the coeff pair *)
    assert (Hc : (basis_mul_coeff sq A B * basis_mul_coeff sq (mask_xor A B) C)%Q
                 == (basis_mul_coeff sq B C * basis_mul_coeff sq A (mask_xor B C))%Q).
    { pose proof (basis_mul_assoc_coeff sq A B C) as H0.
      unfold basis_mul_mask in H0. exact H0. }
    apply Qeq_trans with (y := (F A * G B * H C * (basis_mul_coeff sq A B * basis_mul_coeff sq (mask_xor A B) C))%Q).
    { ring. }
    apply Qeq_trans with (y := (F A * G B * H C * (basis_mul_coeff sq B C * basis_mul_coeff sq A (mask_xor B C)))%Q).
    { apply Qmult_comp. reflexivity. exact Hc. }
    ring.
Qed.