(*
Define:
    size_prog := length defs
    depth_prog via dependency recursion on indices
    lemmas: append/prefix doesn’t increase depths for old nodes, etc.

*)

Require Import Cln_GAProg.

From Coq Require Import List Arith Lia.
Import ListNotations.
Set Implicit Arguments.

(*
  ============================================================
  Cln_GAProg_Cost.v

  Basic cost measures on GAProg:
    - size: number of defs
    - depth: dependency depth of out
  plus standard lemmas needed by compilers.
  ============================================================
*)

Section Cost.
  Context {n : nat}.

  Definition size_prog (p : GAProg n) : nat := length p.(defs).

  (* Dependency list for a node expression (indices it references). *)
  Definition deps (e : GAExpr n) : list nat :=
    match e with
    | EZero => []
    | EOne => []
    | EVar _ => []
    | EAdd i j => [i; j]
    | EScale _ i => [i]
    | EGp i j => [i; j]
    end.

  (* Compute node depth given a depth table for previous nodes. *)
  Definition depth_step (tab : list nat) (e : GAExpr n) : nat :=
    match deps e with
    | [] => 1
    | ds =>
        1 + fold_right Nat.max 0 (map (fun i => nth i tab 0) ds)
    end.

  Fixpoint depth_table (ds : list (GAExpr n)) : list nat :=
    match ds with
    | [] => []
    | e :: tl =>
        let tab := depth_table tl in
        (* NOTE: this is a reverse build; for proofs you may prefer forward build.
           Skeleton keeps it simple; you can refactor later. *)
        (depth_step tab e) :: tab
    end.

  (* A simpler forward version is often nicer; leave as TODO. *)
  Parameter depth_prog : GAProg n -> nat.

  (* ---------------------------------------------------------- *)
  (* Lemmas you will want (skeleton stubs)                       *)
  (* ---------------------------------------------------------- *)

  Lemma size_prog_app :
    forall (p : GAProg n) (extra : list (GAExpr n)) out',
      size_prog {| defs := p.(defs) ++ extra; out := out' |}
      = size_prog p + length extra.
  Proof.
    intros; unfold size_prog; simpl; rewrite app_length; lia.
  Qed.

  Lemma depth_prog_le_size :
    forall (p : GAProg n),
      wf_prog p ->
      depth_prog p <= size_prog p + 1.
  Proof.
  Admitted.

End Cost.


(*
  ============================================================
  File: Cln_GAProg_Cost.v
  ============================================================

  Size/depth measures and structural lemmas for GAProg.
  Kept separate so compiler proofs stay clean.
*)

Require Import Cln_Basis.
Require Import Cln_Multivector.
Require Import Cln_GeometricProduct.
Require Import Cln_GAProg.

From Coq Require Import List Arith Lia QArith.
Import ListNotations.

Open Scope Q_scope.
Set Implicit Arguments.

Definition size_prog {n} (p:GAProg n) : nat := length p.(defs).

(* Depth of a node index in a program (dependency longest-path). *)
Fixpoint max_list (xs:list nat) : nat :=
  match xs with
  | [] => 0
  | x::tl => Nat.max x (max_list tl)
  end.

Fixpoint deps_of {n} (e:GAExpr n) : list nat :=
  match e with
  | EZero | EOne => []
  | EVar _ => []
  | EAdd i j => [i;j]
  | EScale _ i => [i]
  | EGp i j => [i;j]
  end.

(* A simple depth function computed on defs by forward scan. *)
Fixpoint depth_scan {n} (ds:list (GAExpr n)) (depths:list nat) : list nat :=
  match ds with
  | [] => depths
  | d::tl =>
      let dep_depths := List.map (fun i => nth i depths 0) (deps_of d) in
      let di := S (max_list dep_depths) in
      depth_scan tl (depths ++ [di])
  end.

Definition depth_prog {n} (p:GAProg n) : nat :=
  let depths := depth_scan p.(defs) [] in
  nth p.(out) depths 0.

Lemma depth_scan_length :
  forall n (ds:list (GAExpr n)) depths,
    length (depth_scan ds depths) = length depths + length ds.
Proof.
  induction ds; intros; simpl; try lia.
  rewrite IHds.
  (* length (depths ++ [di]) = length depths + 1 *)
  simpl. rewrite app_length. simpl. lia.
Qed.

Lemma depth_prog_le_size :
  forall n (p:GAProg n),
    depth_prog p <= size_prog p + 1.
Proof. Admitted.

(* Prefix/pasting lemma for the evaluator:
   appending extra nodes at the end doesn't change earlier node values. *)
Lemma eval_prog_prefix_invariant :
  forall n (sq:Vector.t Q n) (ρ:Env n) (ds1 ds2:list (GAExpr n)) out,
    out < length ds1 ->
    eval_prog sq ρ {| defs := ds1; out := out |}
    =
    eval_prog sq ρ {| defs := ds1 ++ ds2; out := out |}.
Proof. Admitted.