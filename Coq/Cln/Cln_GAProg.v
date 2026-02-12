(*

Depends on: Cln_Multivector, Cln_GeometricProduct

Define a program language whose semantics is MV n:

Inductive GAExpr (n:nat) : Type :=
| EZero  : GAExpr n
| EOne   : GAExpr n
| EVar   : nat -> GAExpr n              (* external inputs as multivectors *)
| EAdd   : nat -> nat -> GAExpr n
| EScale : Q -> nat -> GAExpr n
| EGp    : nat -> nat -> GAExpr n.      (* geometric product of two nodes *)

Record GAProg (n:nat) : Type := {
  defs : list (GAExpr n);
  out  : nat;
}.


Semantics (DAG evaluation):
    environment: ρ : nat -> MV n (inputs)
    evaluation returns MV n
    wf_prog ensures indices < current node

Minimal lemmas:
    prefix-extension preserves eval of old nodes
    evaluation total under wf_prog

*)

Require Import Cln_Basis.
Require Import Cln_Multivector.
Require Import Cln_GeometricProduct.

From Coq Require Import List Arith Lia QArith.
Import ListNotations.
Open Scope Q_scope.
Set Implicit Arguments.

(*
  ============================================================
  Cln_GAProg.v

  A sharing-aware straight-line program (DAG) language whose
  semantics builds multivectors MV n using:
    - mv_add
    - mv_scale
    - mv_gp n sq   (geometric product)
    - constants mv_zero, mv_one
  ============================================================
*)

(* ------------------------------------------------------------ *)
(* Expression nodes reference earlier nodes by nat indices.       *)
(* The program is: defs : list (GAExpr n), out : nat              *)
(* ------------------------------------------------------------ *)

Inductive GAExpr (n : nat) : Type :=
| EZero  : GAExpr n
| EOne   : GAExpr n
| EVar   : nat -> GAExpr n                  (* external multivector inputs *)
| EAdd   : nat -> nat -> GAExpr n
| EScale : Q -> nat -> GAExpr n
| EGp    : nat -> nat -> GAExpr n.          (* geometric product of two nodes *)

Record GAProg (n : nat) : Type := {
  defs : list (GAExpr n);
  out  : nat;
}.

(* Well-formedness: every node only references earlier indices,
   and out is a valid index. *)
Fixpoint wf_defs {n} (k : nat) (ds : list (GAExpr n)) : Prop :=
  match ds with
  | [] => True
  | d :: tl =>
      (match d with
       | EZero => True
       | EOne => True
       | EVar _ => True
       | EAdd i j => i < k /\ j < k
       | EScale _ i => i < k
       | EGp i j => i < k /\ j < k
       end)
      /\ wf_defs (S k) tl
  end.

Definition wf_prog {n} (p : GAProg n) : Prop :=
  wf_defs 0 p.(defs) /\ p.(out) < length p.(defs).

(* ------------------------------------------------------------ *)
(* Semantics                                                     *)
(* ------------------------------------------------------------ *)

Section Semantics.
  Context {n : nat}.
  Variable sq : Vector.t Q n.

  (* External environment: variables map to multivectors. *)
  Variable env : nat -> MV n.

  (* Evaluate a node index, using defs as a DAG.
     NOTE: skeleton uses nth_error + default mv_zero; wf will ensure no default used. *)
  Fixpoint eval_idx (fuel : nat) (ds : list (GAExpr n)) (i : nat) : MV n :=
    match fuel with
    | 0 => mv_zero
    | S fuel' =>
        match nth_error ds i with
        | None => mv_zero
        | Some e =>
            match e with
            | EZero => mv_zero
            | EOne => mv_one
            | EVar x => env x
            | EAdd a b =>
                mv_add (eval_idx fuel' ds a) (eval_idx fuel' ds b)
            | EScale k a =>
                mv_scale k (eval_idx fuel' ds a)
            | EGp a b =>
                mv_gp n sq (eval_idx fuel' ds a) (eval_idx fuel' ds b)
            end
        end
    end.

  Definition eval_prog (p : GAProg n) : MV n :=
    (* enough fuel: length defs is safe upper bound if wf ensures acyclic by index *)
    eval_idx (S (length p.(defs))) p.(defs) p.(out).

End Semantics.

(* ------------------------------------------------------------ *)
(* Structural lemmas (skeleton stubs)                             *)
(* ------------------------------------------------------------ *)

Section Lemmas.
  Context {n : nat}.
  Variable sq : Vector.t Q n.
  Variable env : nat -> MV n.

  Lemma wf_defs_monotone :
    forall k ds,
      wf_defs k ds ->
      wf_defs (S k) ds.
  Proof.
    (* handy when appending/offsetting; fill later *)
  Admitted.

  (* Prefix-extension preservation: if ds is a prefix of ds',
     then evaluation of indices < length ds is preserved. *)
  Lemma eval_idx_prefix :
    forall fuel ds ds' i,
      (exists suf, ds' = ds ++ suf) ->
      i < length ds ->
      eval_idx (n:=n) sq env fuel ds i = eval_idx (n:=n) sq env fuel ds' i.
  Proof.
  Admitted.

  Lemma eval_prog_ext :
    forall (p : GAProg n) (env1 env2 : nat -> MV n),
      (forall x, env1 x = env2 x) ->
      eval_prog (n:=n) sq env1 p = eval_prog (n:=n) sq env2 p.
  Proof.
  Admitted.

End Lemmas.


(*
  ============================================================
  File: Cln_GAProg.v
  ============================================================

  A sharing-aware straight-line program (DAG) language whose
  semantics builds multivectors (MV n) using:
    - mv_add, mv_scale
    - geometric product mv_gp n sq
    - constants mv_zero, mv_one

  This is the representation language you will measure size/depth on.
*)

Require Import Cln_Basis.
Require Import Cln_Multivector.
Require Import Cln_GeometricProduct.

From Coq Require Import List Arith Lia QArith.
Import ListNotations.

Open Scope Q_scope.
Set Implicit Arguments.

(* ============================================================ *)
(* Expressions over previously-defined nodes                      *)
(* ============================================================ *)

Inductive GAExpr (n:nat) : Type :=
| EZero  : GAExpr n
| EOne   : GAExpr n
| EVar   : nat -> GAExpr n                      (* external multivector input *)
| EAdd   : nat -> nat -> GAExpr n
| EScale : Q -> nat -> GAExpr n
| EGp    : nat -> nat -> GAExpr n.              (* geometric product *)

Record GAProg (n:nat) : Type := {
  defs : list (GAExpr n);
  out  : nat
}.

(* Well-formedness: every node only references earlier nodes. *)
Fixpoint wf_defs {n} (k:nat) (ds:list (GAExpr n)) : Prop :=
  match ds with
  | [] => True
  | d :: tl =>
      (match d with
       | EZero | EOne => True
       | EVar _ => True
       | EAdd i j => i < k /\ j < k
       | EScale _ i => i < k
       | EGp i j => i < k /\ j < k
       end) /\
      wf_defs (S k) tl
  end.

Definition wf_prog {n} (p:GAProg n) : Prop :=
  wf_defs 0 p.(defs) /\ p.(out) < length p.(defs).

(* ============================================================ *)
(* Semantics                                                      *)
(* ============================================================ *)

Definition Env (n:nat) : Type := nat -> MV n.

(* Evaluate a single expression given already-evaluated node values. *)
Definition eval_expr {n} (sq : Vector.t Q n) (ρ : Env n) (vals : list (MV n)) (e : GAExpr n) : MV n :=
  match e with
  | EZero => mv_zero
  | EOne  => mv_one
  | EVar x => ρ x
  | EAdd i j =>
      mv_add (nth i vals mv_zero) (nth j vals mv_zero)
  | EScale k i =>
      mv_scale k (nth i vals mv_zero)
  | EGp i j =>
      mv_gp n sq (nth i vals mv_zero) (nth j vals mv_zero)
  end.

Fixpoint eval_defs {n} (sq : Vector.t Q n) (ρ : Env n) (ds : list (GAExpr n)) : list (MV n) :=
  match ds with
  | [] => []
  | d :: tl =>
      let vals := eval_defs sq ρ tl in
      (* Build from tail to head is awkward; prefer forward fold in proofs.
         We keep this definition minimal; a forward version is in Cost file. *)
      (* NOTE: This reverse recursion is intentionally a stub. *)
      (* TODO: replace with forward fold (see Cln_GAProg_Cost.v) *)
      (eval_expr sq ρ vals d) :: vals
  end.

(* A forward (prefix) evaluator that matches wf_defs indices. *)
Fixpoint eval_defs_fwd {n} (sq : Vector.t Q n) (ρ : Env n) (ds : list (GAExpr n)) (acc : list (MV n)) : list (MV n) :=
  match ds with
  | [] => acc
  | d :: tl =>
      let v := eval_expr sq ρ acc d in
      eval_defs_fwd sq ρ tl (acc ++ [v])
  end.

Definition eval_prog {n} (sq : Vector.t Q n) (ρ : Env n) (p : GAProg n) : MV n :=
  let vals := eval_defs_fwd sq ρ p.(defs) [] in
  nth p.(out) vals mv_zero.

(* ============================================================ *)
(* Basic lemmas you will use everywhere                           *)
(* ============================================================ *)

Lemma wf_defs_index_lt :
  forall n k (ds:list (GAExpr n)) i j,
    wf_defs k ds ->
    In (EAdd i j) ds ->
    i < k + length ds /\ j < k + length ds.
Proof. Admitted.

Lemma eval_defs_fwd_length :
  forall n (sq:Vector.t Q n) (ρ:Env n) ds acc,
    length (eval_defs_fwd sq ρ ds acc) = length acc + length ds.
Proof.
  induction ds; intros; simpl.
  - lia.
  - specialize (IHds acc ++ [eval_expr sq ρ acc a]).
    (* keep it simple; this lemma is mainly for indexing *)
    admit.
Admitted.

Lemma eval_defs_fwd_prefix :
  forall n (sq:Vector.t Q n) (ρ:Env n) ds acc1 acc2,
    eval_defs_fwd sq ρ ds (acc1 ++ acc2)
    =
    (eval_defs_fwd sq ρ ds acc1) ++ acc2.
Proof. Admitted.