(*
  ============================================================
  RepComp_P.v
  ============================================================

  This file makes "Representation = Computation for P" precise
  and clean in Coq.

  It is NOT a P vs NP file.

  Theorem (schema):
    InP F  <->  InRP F

  Where:
    - InP: there exists a poly-time algorithm deciding the family F
    - InRP: there exists a uniform family of polynomial-size representations
            evaluable in polynomial time, representing F.

  Key point:
    This equivalence is proved once you supply TWO compilers/bridges:

      (A) eval_as_alg : RepFamily -> Alg
          (evaluation of representation yields an algorithm)

      (B) compile : Alg -> RepFamily
          (compilation of an algorithm yields a representation family)

  Later:
    - instantiate Rep with uniform circuits (standard)
    - or attempt instantiation with GA/Walsh once you build compile.

  ============================================================
*)

From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Arith.Arith.
From Coq Require Import Lia.

Import ListNotations.
Open Scope nat_scope.

(* ============================================================ *)
(* PART 0: Boolean function families                            *)
(* ============================================================ *)

Definition BoolFn (n : nat) : Type := list bool -> bool.
Definition FnFamily : Type := forall n : nat, BoolFn n.

Definition wf (n:nat) (x:list bool) : Prop := length x = n.

(* ============================================================ *)
(* PART 1: Polynomial bounds                                     *)
(* ============================================================ *)

(*
  We use a simple (loose) definition of "poly" sufficient for
  structuring proofs. You can swap it later for your preferred one.
*)

Definition poly (p : nat -> nat) : Prop :=
  exists k c, k > 0 /\ forall n, p n <= c * (n ^ k + 1).

Lemma poly_plus1 : forall p, poly p -> poly (fun n => p n + 1).
Proof.
  intros p [k [c [Hk Hp]]].
  exists k (c + 1). split; [assumption|].
  intro n.
  specialize (Hp n).
  (* p n <= c*(n^k+1)  -> p n + 1 <= (c+1)*(n^k+1) *)
  nia.
Qed.

(* ============================================================ *)
(* PART 2: Computation model for P (abstract Alg)                *)
(* ============================================================ *)

Parameter Alg : Type.
Parameter run  : forall n:nat, Alg -> list bool -> bool.
Parameter time : forall n:nat, Alg -> list bool -> nat.

Definition polytime (A:Alg) : Prop :=
  exists p, poly p /\ forall n x, wf n x -> time n A x <= p n.

Definition computes (A:Alg) (F:FnFamily) : Prop :=
  forall n x, wf n x -> run n A x = F n x.

Definition InP (F:FnFamily) : Prop :=
  exists A, polytime A /\ computes A F.

(* ============================================================ *)
(* PART 3: Representation language                               *)
(* ============================================================ *)

Class RepLang (Rep : nat -> Type) := {
  rep_eval      : forall n, Rep n -> list bool -> bool;
  rep_eval_time : forall n, Rep n -> list bool -> nat;
  rep_size      : forall n, Rep n -> nat
}.

Definition represents {Rep:nat->Type} `{RepLang Rep}
  (n:nat) (r:Rep n) (f:BoolFn n) : Prop :=
  forall x, wf n x -> rep_eval n r x = f x.

Definition RepFamily (Rep:nat->Type) : Type := forall n, Rep n.

Section RP_Definitions.
Context {Rep:nat->Type}.
Context `{RepLang Rep}.

Definition poly_size (Rfam : RepFamily Rep) : Prop :=
  exists p, poly p /\ forall n, rep_size n (Rfam n) <= p n.

Definition poly_eval (Rfam : RepFamily Rep) : Prop :=
  exists p, poly p /\ forall n x, wf n x -> rep_eval_time n (Rfam n) x <= p n.

Definition InRP (F:FnFamily) : Prop :=
  exists Rfam,
    poly_size Rfam /\
    poly_eval Rfam /\
    forall n, represents n (Rfam n) (F n).

End RP_Definitions.

(* ============================================================ *)
(* PART 4: Bridges (compiler + evaluator-as-algorithm)           *)
(* ============================================================ *)

Section Bridges.
Context {Rep:nat->Type}.
Context `{RepLang Rep}.

(*
  Bridge 1: A single algorithm that evaluates the representation family.
  This gives the direction InRP -> InP.

  NOTE: This is where "uniformity" lives: eval_as_alg must use only
        the family Rfam and the input x (whose length gives n).
*)
Parameter eval_as_alg : RepFamily Rep -> Alg.

Axiom eval_as_alg_correct :
  forall (Rfam : RepFamily Rep) n x,
    wf n x ->
    run n (eval_as_alg Rfam) x = rep_eval n (Rfam n) x.

Axiom eval_as_alg_time :
  forall (Rfam : RepFamily Rep) n x,
    wf n x ->
    time n (eval_as_alg Rfam) x <= rep_eval_time n (Rfam n) x + 1.

(*
  Bridge 2: Compile an algorithm into a representation family.
  This gives the direction InP -> InRP.
*)
Parameter compile : Alg -> RepFamily Rep.

Axiom compile_correct :
  forall (A:Alg) n x,
    wf n x ->
    rep_eval n (compile A n) x = run n A x.

Axiom compile_size_poly :
  forall (A:Alg),
    polytime A ->
    exists p, poly p /\ forall n, rep_size n (compile A n) <= p n.

Axiom compile_eval_poly :
  forall (A:Alg),
    polytime A ->
    exists p, poly p /\ forall n x, wf n x ->
      rep_eval_time n (compile A n) x <= p n.

(* ------------------ Theorems ------------------ *)

Theorem InRP_implies_InP :
  forall (F:FnFamily),
    InRP (Rep:=Rep) F ->
    InP F.
Proof.
  intros F [Rfam [Hsz [Hev Hrep]]].
  exists (eval_as_alg Rfam).
  split.
  - (* polytime *)
    destruct Hev as [p [Hp Hpbd]].
    exists (fun n => p n + 1).
    split.
    + apply poly_plus1. exact Hp.
    + intros n x Hwf.
      specialize (eval_as_alg_time Rfam n x Hwf).
      specialize (Hpbd n x Hwf).
      lia.
  - (* computes *)
    intros n x Hwf.
    rewrite eval_as_alg_correct by assumption.
    unfold represents in Hrep.
    apply Hrep; assumption.
Qed.

Theorem InP_implies_InRP :
  forall (F:FnFamily),
    InP F ->
    InRP (Rep:=Rep) F.
Proof.
  intros F [A [Hpoly Hcomp]].
  exists (compile A).
  split.
  - (* poly_size *)
    destruct (compile_size_poly A Hpoly) as [p [Hp Hbd]].
    exists p. split; [exact Hp|].
    intro n. apply Hbd.
  - split.
    + (* poly_eval *)
      destruct (compile_eval_poly A Hpoly) as [p [Hp Hbd]].
      exists p. split; [exact Hp|].
      intros n x Hwf. apply Hbd; assumption.
    + (* represents *)
      intro n. unfold represents.
      intros x Hwf.
      rewrite compile_correct by assumption.
      apply Hcomp; assumption.
Qed.

Theorem Representation_equals_Computation_for_P :
  forall (F:FnFamily),
    InP F <-> InRP (Rep:=Rep) F.
Proof.
  intro F. split.
  - apply InP_implies_InRP.
  - apply InRP_implies_InP.
Qed.

End Bridges.

(*
  ============================================================
  End RepComp_P.v

  How you use it:

    - Choose RepLang (circuits is easiest).
    - Provide eval_as_alg and compile + their axioms/lemmas.
    - Then you get InP <-> InRP for that RepLang.

  For GA/Walsh:
    Rep n := MV n
    rep_eval := your Boolean evaluator via embed_correct
    rep_size := your chosen size measure
    compile := (hard) your "polytime => small GA rep" thesis
  ============================================================
*)
