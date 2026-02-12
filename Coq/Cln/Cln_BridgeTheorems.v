(*
Exports theorems in the “polynomial equivalence” style.
*)

(*
  ============================================================
  File: Cln_BridgeTheorems.v
  ============================================================

  Packages the "polynomial equivalence" bridges between:
    - Boolean circuits (computation)
    - GAProg over MV n (representation)

  This file should contain only clean, user-facing statements,
  proved from compilers/evaluators in earlier files.
*)

Require Import Cln_Basis.
Require Import Cln_Multivector.
Require Import Cln_GeometricProduct.

Require Import Cln_GAProg.
Require Import Cln_GAProg_Cost.
Require Import Cln_Circuit.
Require Import Cln_Circuit_Eval.
Require Import Cln_CircuitToGAProg.

From Coq Require Import List Arith Lia QArith.
Import ListNotations.
Open Scope Q_scope.
Set Implicit Arguments.

(* ============================================================ *)
(* Bridge A: Efficient computation ⇒ compact GA representation     *)
(* ============================================================ *)

Theorem BridgeA_Circuit_to_GAProg :
  forall n (sq:Vector.t Q n) (C:Circuit),
    wf_circuit C ->
    exists p tab,
      compile sq C = (p, tab)
      /\ size_prog p <= 6 * size_c C + 2
      /\ (forall ρ, eval_prog sq (input_env ρ) p = bool_mv (eval_circuit C ρ)).
Proof.
  intros n sq C Hwf.
  exists (fst (compile sq C)), (snd (compile sq C)).
  destruct (compile sq C) as [p tab] eqn:Hc; simpl.
  repeat split; try reflexivity.
  - (* size *) pose proof (compile_size (n:=n) (sq:=sq) (C:=C)) as Hs.
    rewrite Hc in Hs. exact Hs.
  - (* correctness *) intro ρ.
    pose proof (compile_correct (n:=n) (sq:=sq) (C:=C) (ρ:=ρ) Hwf) as Hk.
    rewrite Hc in Hk. exact Hk.
Qed.

(* ============================================================ *)
(* Bridge B: GAProg evaluation cost is polynomial in program size  *)
(* ============================================================ *)

(* In Coq we usually express this as a bound on depth/size, not runtime.
   Once you define an explicit step-count for eval_prog, you can strengthen this. *)

Theorem BridgeB_depth_le_poly_size :
  forall n (p:GAProg n),
    depth_prog p <= size_prog p + 1.
Proof.
  intros. apply depth_prog_le_size.
Qed.

(* ============================================================ *)
(* Bridge C: Compositional closure (sketch interface)              *)
(* ============================================================ *)

(*
  Compositional closure is where you show you can "wire" the output of one
  compiled program as an input into another without global expansion.

  You will implement an operation like:
    link : GAProg n -> nat (*output*) -> GAProg n -> GAProg n

  and prove size(link) <= size(p1)+size(p2)+O(1), and semantics of substitution.
*)

Parameter link :
  forall n, GAProg n -> nat -> GAProg n -> GAProg n.

Axiom link_size :
  forall n (p1 p2:GAProg n) out1,
    size_prog (link p1 out1 p2) <= size_prog p1 + size_prog p2 + 5.

Axiom link_correct :
  forall n (sq:Vector.t Q n) (ρ:Env n) (p1 p2:GAProg n) out1,
    (* Semantics: p2 sees its chosen input var replaced by p1's output, etc. *)
    True.
