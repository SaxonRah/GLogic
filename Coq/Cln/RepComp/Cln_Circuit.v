(*
Boolean circuit with sharing, same as earlier plan.
*)

From Coq Require Import List Arith Lia Bool.
Import ListNotations.
Set Implicit Arguments.

(*
  ============================================================
  Cln_Circuit.v

  A DAG circuit language with explicit sharing (gate list + out).
  Indices refer to earlier gates.
  ============================================================
*)

Inductive Gate : Type :=
| GConst : bool -> Gate
| GInput : nat -> Gate
| GNot   : nat -> Gate
| GAnd   : nat -> nat -> Gate
| GXor   : nat -> nat -> Gate
| GOr    : nat -> nat -> Gate.

Record Circuit : Type := {
  gates : list Gate;
  out   : nat;
}.

Fixpoint wf_gates (k : nat) (gs : list Gate) : Prop :=
  match gs with
  | [] => True
  | g :: tl =>
      (match g with
       | GConst _ => True
       | GInput _ => True
       | GNot i => i < k
       | GAnd i j => i < k /\ j < k
       | GXor i j => i < k /\ j < k
       | GOr  i j => i < k /\ j < k
       end)
      /\ wf_gates (S k) tl
  end.

Definition wf_circuit (C : Circuit) : Prop :=
  wf_gates 0 C.(gates) /\ C.(out) < length C.(gates).

Definition size_c (C : Circuit) : nat := length C.(gates).

(* Depth (skeleton): you can implement similarly to GAProg depth. *)
Parameter depth_c : Circuit -> nat.

(*
  ============================================================
  File: Cln_Circuit.v
  ============================================================

  A sharing-aware Boolean circuit DAG:
    - gates are listed in topological order
    - each gate references earlier indices
*)

From Coq Require Import List Bool Arith Lia.
Import ListNotations.
Set Implicit Arguments.

Inductive Gate : Type :=
| GConst : bool -> Gate
| GInput : nat -> Gate
| GNot   : nat -> Gate
| GAnd   : nat -> nat -> Gate
| GXor   : nat -> nat -> Gate
| GOr    : nat -> nat -> Gate.

Record Circuit : Type := {
  gates : list Gate;
  out   : nat
}.

Fixpoint wf_gates (k:nat) (gs:list Gate) : Prop :=
  match gs with
  | [] => True
  | g::tl =>
      (match g with
       | GConst _ => True
       | GInput _ => True
       | GNot i => i < k
       | GAnd i j => i < k /\ j < k
       | GXor i j => i < k /\ j < k
       | GOr  i j => i < k /\ j < k
       end) /\
      wf_gates (S k) tl
  end.

Definition wf_circuit (C:Circuit) : Prop :=
  wf_gates 0 C.(gates) /\ C.(out) < length C.(gates).

Definition size_c (C:Circuit) : nat := length C.(gates).