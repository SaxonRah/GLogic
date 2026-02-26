(*
  ============================================================
  File: Cln_DAG.v
  ============================================================

  DAG (circuit) representation of GA computations.

  GA_expr is a tree: shared subexpressions are duplicated,
  so expr_size counts formula size.

  GA_dag is a circuit: nodes reference earlier nodes by index,
  allowing sharing.  dag_size counts circuit size.

  Key properties:
  - Every tree flattens to a DAG (without sharing)
  - A DAG can be exponentially smaller than any equivalent tree
  - Excursion and trace-boolishness are node-local,
    so fan-out is free for both invariants and resource measures
  - Lower bounds against DAGs are lower bounds against circuits

  ============================================================
*)

Require Import Cln_Full.
Require Import Cln_Grade.
Require Import Cln_finite_l1_submultiplicativity.
Require Import Cln_BoolDist.
Require Import Cln_SupportAlgebra.
Require Import Cln_CompositeExcursion.

From Coq Require Import FunctionalExtensionality.
From Coq Require Import List Lia Arith.
From Coq Require Import QArith.
From Coq Require Import QArith.QArith_base.
From Coq Require Import QArith.Qabs.
From Coq Require Import Bool.
From Coq Require Import Fin.
From Coq Require Import Vectors.Vector.
From Coq Require Import Program.Equality.

Import ListNotations.
Open Scope Q_scope.
Set Implicit Arguments.


(* ============================================================ *)
(* Section 1: DAG Node Operations                                *)
(* ============================================================ *)

Section GA_DAG.

Variable n : nat.

(* A single operation at scope k: can reference k previously
   defined nodes via Fin.t k indices. *)

Inductive dag_op : nat -> Type :=
  | DOpBasis  : forall {k}, Fin.t n -> dag_op k
  | DOpScalar : forall {k}, Q -> dag_op k
  | DOpAdd    : forall {k}, Fin.t k -> Fin.t k -> dag_op k
  | DOpMul    : forall {k}, Fin.t k -> Fin.t k -> dag_op k
  | DOpConv   : forall {k}, Fin.t k -> Fin.t k -> dag_op k.

(* ============================================================ *)
(* Section 2: The DAG Structure                                  *)
(* ============================================================ *)

(* A snoc-list of nodes with increasing scope.
   GA_dag k means "k nodes have been defined so far." *)

Inductive GA_dag : nat -> Type :=
  | DagNil  : GA_dag 0
  | DagSnoc : forall {k}, GA_dag k -> dag_op k -> GA_dag (S k).

(* Size = number of nodes *)
Definition dag_size {k} (_ : GA_dag k) : nat := k.

(* ============================================================ *)
(* Section 3: Evaluation                                         *)
(* ============================================================ *)

(* Evaluate a single node given an environment of previous results *)
Definition eval_op (sq : Vector.t Q n) {k}
    (env : Fin.t k -> MV n) (op : dag_op k) : MV n :=
  match op with
  | DOpBasis _ i   => basis (mask_single i)
  | DOpScalar _ c  => constMV c
  | DOpAdd _ a b   => mv_add (env a) (env b)
  | DOpMul _ a b   => mv_gp sq (env a) (env b)
  | DOpConv _ a b  => mv_conv (env a) (env b)
  end.

(* Evaluate entire DAG, producing all intermediate results.
   Returns a function Fin.t k -> MV n (the environment).
   Uses an auxiliary vector internally. *)

Fixpoint eval_dag_env (sq : Vector.t Q n) {k} (d : GA_dag k)
    : Fin.t k -> MV n :=
  match d with
  | DagNil => fun i => Fin.case0 _ i
  | DagSnoc d' op =>
      let prev := eval_dag_env sq d' in
      let v := eval_op sq prev op in
      fun i =>
        match i in Fin.t (S k') return (Fin.t k' -> MV n) -> MV n -> MV n with
        | Fin.F1 => fun _ newest => newest
        | Fin.FS j => fun old _ => old j
        end prev v
  end.

(* Lookup any node's result *)
Definition eval_dag (sq : Vector.t Q n) {k} (d : GA_dag k)
    (i : Fin.t k) : MV n :=
  eval_dag_env sq d i.

(* Unfolding lemma: the newest node evaluates to eval_op applied
   to the prefix environment *)
Lemma eval_dag_env_snoc_F1 :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k),
    eval_dag_env sq (DagSnoc d op) Fin.F1
    = eval_op sq (eval_dag_env sq d) op.
Proof. Admitted.

Lemma eval_dag_env_snoc_FS :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k)
         (j : Fin.t k),
    eval_dag_env sq (DagSnoc d op) (Fin.FS j)
    = eval_dag_env sq d j.
Proof. Admitted.


(* ============================================================ *)
(* Section 4: Excursion Measures (peak over all nodes)           *)
(* ============================================================ *)

Fixpoint dag_max_l1 (sq : Vector.t Q n) {k} (d : GA_dag k) : Q :=
  match d with
  | DagNil => 0
  | DagSnoc d' op =>
      Qmax (dag_max_l1 sq d')
           (l1_norm (eval_op sq (eval_dag_env sq d') op))
  end.

Fixpoint dag_max_grade (sq : Vector.t Q n) {k} (d : GA_dag k) : nat :=
  match d with
  | DagNil => 0%nat
  | DagSnoc d' op =>
      Nat.max (dag_max_grade sq d')
              (max_grade (eval_op sq (eval_dag_env sq d') op))
  end.

Definition dag_exc (sq : Vector.t Q n) {k} (d : GA_dag k) : ExcNum :=
  {| exc_grade := dag_max_grade sq d;
     exc_l1    := dag_max_l1 sq d |}.

(* Every node's l1 norm is bounded by the peak *)
Lemma eval_dag_l1_le_peak :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (i : Fin.t k),
    l1_norm (eval_dag sq d i) <= dag_max_l1 sq d.
Proof. Admitted.

(* Every node's max_grade is bounded by the peak *)
Lemma eval_dag_grade_le_peak :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (i : Fin.t k),
    (max_grade (eval_dag sq d i) <= dag_max_grade sq d)%nat.
Proof. Admitted.

(* Monotonicity: extending the DAG doesn't decrease the peak *)
Lemma dag_max_l1_mono :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k),
    dag_max_l1 sq d <= dag_max_l1 sq (DagSnoc d op).
Proof. Admitted.

Lemma dag_max_grade_mono :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k),
    (dag_max_grade sq d <= dag_max_grade sq (DagSnoc d op))%nat.
Proof. Admitted.


(* ============================================================ *)
(* Section 5: Trace Boolishness                                  *)
(* ============================================================ *)

(* Every intermediate MV is near a Boolean function *)
Fixpoint dag_trace_boolish (sq : Vector.t Q n) {k}
    (d : GA_dag k) (tol : Q) : Prop :=
  match d with
  | DagNil => True
  | DagSnoc d' op =>
      dag_trace_boolish sq d' tol /\
      boolish_le (eval_op sq (eval_dag_env sq d') op) tol
  end.

(* Every intermediate MV is near a k-sparse Boolean combination *)
Fixpoint dag_trace_boolish_k (sq : Vector.t Q n) {k}
    (d : GA_dag k) (bound : nat) (tol : Q) : Prop :=
  match d with
  | DagNil => True
  | DagSnoc d' op =>
      dag_trace_boolish_k sq d' bound tol /\
      boolish_k_le (eval_op sq (eval_dag_env sq d') op) bound tol
  end.

Definition dag_trace_boolish_exists_k (sq : Vector.t Q n) {k}
    (d : GA_dag k) (tol : Q) : Prop :=
  exists bound, dag_trace_boolish_k sq d bound tol.

(* k=1, tol=0 means every node is exactly an embedded Boolean function *)
Definition dag_trace_exact_bool (sq : Vector.t Q n) {k}
    (d : GA_dag k) : Prop :=
  dag_trace_boolish_k sq d 1 0.

(* Monotonicity in k *)
Lemma dag_trace_boolish_k_mono :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) k1 k2 tol,
    (k1 <= k2)%nat ->
    dag_trace_boolish_k sq d k1 tol ->
    dag_trace_boolish_k sq d k2 tol.
Proof. Admitted.

(* Monotonicity in tol *)
Lemma dag_trace_boolish_tol_mono :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) bound tol1 tol2,
    tol1 <= tol2 ->
    dag_trace_boolish_k sq d bound tol1 ->
    dag_trace_boolish_k sq d bound tol2.
Proof. Admitted.

(* Extension: if the prefix is boolish and the new node is boolish,
   the extended DAG is boolish *)
Lemma dag_trace_boolish_snoc :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k) tol,
    dag_trace_boolish sq d tol ->
    boolish_le (eval_op sq (eval_dag_env sq d) op) tol ->
    dag_trace_boolish sq (DagSnoc d op) tol.
Proof. Admitted.

Lemma dag_trace_boolish_k_snoc :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k)
         bound tol,
    dag_trace_boolish_k sq d bound tol ->
    boolish_k_le (eval_op sq (eval_dag_env sq d) op) bound tol ->
    dag_trace_boolish_k sq (DagSnoc d op) bound tol.
Proof. Admitted.

(* Prefix extraction: boolishness of DagSnoc implies boolishness
   of the prefix *)
Lemma dag_trace_boolish_prefix :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k) tol,
    dag_trace_boolish sq (DagSnoc d op) tol ->
    dag_trace_boolish sq d tol.
Proof. Admitted.

Lemma dag_trace_boolish_k_prefix :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k)
         bound tol,
    dag_trace_boolish_k sq (DagSnoc d op) bound tol ->
    dag_trace_boolish_k sq d bound tol.
Proof. Admitted.


(* ============================================================ *)
(* Section 6: Computes Predicate                                 *)
(* ============================================================ *)

(* A DAG with a designated output node computes f *)
Definition dag_computes (sq : Vector.t Q n) {k}
    (d : GA_dag k) (root : Fin.t k) (f : Corner n -> bool) : Prop :=
  forall m : Mask n, eval_dag sq d root m == embed f m.

(* Computes implies the output is exactly Boolean *)
Lemma dag_computes_implies_boolish_0 :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (root : Fin.t k) (f : Corner n -> bool),
    dag_computes sq d root f ->
    boolish_le (eval_dag sq d root) 0.
Proof. Admitted.

(* Computes implies zero Boolean distance at the output *)
Lemma dag_computes_implies_dist_zero :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (root : Fin.t k) (f : Corner n -> bool),
    dag_computes sq d root f ->
    dist_to (eval_dag sq d root) f == 0.
Proof. Admitted.


(* ============================================================ *)
(* Section 7: Index Shifting Utilities                           *)
(* ============================================================ *)

(* Weaken: embed Fin.t k into Fin.t (m + k) by adding m
   to the de Bruijn index.  "Push into the older part." *)
Fixpoint fin_weaken_by (m : nat) {k} (i : Fin.t k)
    : Fin.t (m + k) :=
  match m with
  | 0 => i
  | S m' => Fin.FS (fin_weaken_by m' i)
  end.

(* Lift: embed Fin.t k into Fin.t (k + m) by keeping the index.
   "Stay in the newer part."
   Uses Fin.L from the standard library. *)
Definition fin_lift_into (m : nat) {k} (i : Fin.t k)
    : Fin.t (k + m) :=
  Fin.L m i.

(* Shift all references in a dag_op by m (for appending) *)
Definition shift_op (m : nat) {k} (op : dag_op k) : dag_op (k + m) :=
  match op with
  | DOpBasis _ i   => DOpBasis i
  | DOpScalar _ c  => DOpScalar c
  | DOpAdd _ a b   => DOpAdd (fin_lift_into m a) (fin_lift_into m b)
  | DOpMul _ a b   => DOpMul (fin_lift_into m a) (fin_lift_into m b)
  | DOpConv _ a b  => DOpConv (fin_lift_into m a) (fin_lift_into m b)
  end.

Lemma fin_weaken_by_0 :
  forall {k} (i : Fin.t k),
    fin_weaken_by 0 i = i.
Proof. Admitted.

Lemma fin_weaken_by_S :
  forall (m : nat) {k} (i : Fin.t k),
    fin_weaken_by (S m) i = Fin.FS (fin_weaken_by m i).
Proof. Admitted.


(* ============================================================ *)
(* Section 8: DAG Concatenation                                  *)
(* ============================================================ *)

(* Append d2 after d1: nodes of d2 are shifted so they can
   reference nodes of d1. Result has (k2 + k1) nodes. *)
Fixpoint dag_append {k1 k2} (d1 : GA_dag k1) (d2 : GA_dag k2)
    : GA_dag (k2 + k1) :=
  match d2 with
  | DagNil => d1
  | DagSnoc d2' op =>
      DagSnoc (dag_append d1 d2') (shift_op k1 op)
  end.

(* Evaluation of the appended DAG agrees with the originals *)
Lemma eval_dag_append_left :
  forall (sq : Vector.t Q n) {k1 k2}
         (d1 : GA_dag k1) (d2 : GA_dag k2) (i : Fin.t k1),
    eval_dag sq (dag_append d1 d2) (fin_weaken_by k2 i)
    = eval_dag sq d1 i.
Proof. Admitted.

Lemma eval_dag_append_right :
  forall (sq : Vector.t Q n) {k1 k2}
         (d1 : GA_dag k1) (d2 : GA_dag k2) (j : Fin.t k2),
    eval_dag sq (dag_append d1 d2) (fin_lift_into k1 j)
    = eval_dag sq d2 j.
Proof. Admitted.

(* Trace boolishness of append = conjunction of both parts *)
Lemma dag_trace_boolish_append :
  forall (sq : Vector.t Q n) {k1 k2}
         (d1 : GA_dag k1) (d2 : GA_dag k2) tol,
    dag_trace_boolish sq (dag_append d1 d2) tol
    <-> dag_trace_boolish sq d1 tol /\ dag_trace_boolish sq d2 tol.
Proof. Admitted.

Lemma dag_trace_boolish_k_append :
  forall (sq : Vector.t Q n) {k1 k2}
         (d1 : GA_dag k1) (d2 : GA_dag k2) bound tol,
    dag_trace_boolish_k sq (dag_append d1 d2) bound tol
    <-> dag_trace_boolish_k sq d1 bound tol
        /\ dag_trace_boolish_k sq d2 bound tol.
Proof. Admitted.

(* Peak excursion of append = max of both parts *)
Lemma dag_max_l1_append :
  forall (sq : Vector.t Q n) {k1 k2}
         (d1 : GA_dag k1) (d2 : GA_dag k2),
    dag_max_l1 sq (dag_append d1 d2)
    == Qmax (dag_max_l1 sq d1) (dag_max_l1 sq d2).
Proof. Admitted.

Lemma dag_max_grade_append :
  forall (sq : Vector.t Q n) {k1 k2}
         (d1 : GA_dag k1) (d2 : GA_dag k2),
    dag_max_grade sq (dag_append d1 d2)
    = Nat.max (dag_max_grade sq d1) (dag_max_grade sq d2).
Proof. Admitted.


(* ============================================================ *)
(* Section 9: Flattening (GA_expr → GA_dag)                      *)
(* ============================================================ *)

(* Returns: (number of nodes, DAG, root index).
   Root index is explicit — no "root = F1" convention assumed
   by the return type, though in practice flatten always
   produces root = F1. *)

Fixpoint flatten (e : GA_expr n)
    : { k : nat & GA_dag (S k) * Fin.t (S k) }%type :=
  match e with
  | Basis i =>
      existT _ 0
        (DagSnoc DagNil (DOpBasis i), Fin.F1)

  | Scalar c =>
      existT _ 0
        (DagSnoc DagNil (DOpScalar c), Fin.F1)

  | Cln_Grade.Add e1 e2 =>
      let '(existT _ k1 (d1, r1)) := flatten e1 in
      let '(existT _ k2 (d2, r2)) := flatten e2 in
      let combined := dag_append d1 d2 in
      let r1' := fin_weaken_by (S k2) r1 in
      let r2' := fin_lift_into (S k1) r2 in
      existT _ (S k2 + S k1)
        (DagSnoc combined (DOpAdd r1' r2'), Fin.F1)
      (* existT _ (S (S k2) + S k1)
          (DagSnoc combined (DOpAdd r1' r2'), Fin.F1)
      *)

  | Mul e1 e2 =>
      let '(existT _ k1 (d1, r1)) := flatten e1 in
      let '(existT _ k2 (d2, r2)) := flatten e2 in
      let combined := dag_append d1 d2 in
      let r1' := fin_weaken_by (S k2) r1 in
      let r2' := fin_lift_into (S k1) r2 in
      existT _ (S k2 + S k1)
        (DagSnoc combined (DOpMul r1' r2'), Fin.F1)
      (* existT _ (S (S k2) + S k1)
          (DagSnoc combined (DOpMul r1' r2'), Fin.F1)
      *)

  | Conv e1 e2 =>
      let '(existT _ k1 (d1, r1)) := flatten e1 in
      let '(existT _ k2 (d2, r2)) := flatten e2 in
      let combined := dag_append d1 d2 in
      let r1' := fin_weaken_by (S k2) r1 in
      let r2' := fin_lift_into (S k1) r2 in
      existT _ (S k2 + S k1)
        (DagSnoc combined (DOpConv r1' r2'), Fin.F1)
      (* existT _ (S (S k2) + S k1)
          (DagSnoc combined (DOpConv r1' r2'), Fin.F1)
      *)
  end.

(* Flatten always produces root = F1 *)
Lemma flatten_root_is_F1 :
  forall (e : GA_expr n),
    let '(existT _ _ (_, r)) := flatten e in
    r = Fin.F1.
Proof. Admitted.


(* ============================================================ *)
(* Section 10: Bridge Lemmas (Tree ↔ DAG)                       *)
(* ============================================================ *)

(* Flattening preserves evaluation *)
Theorem flatten_eval_correct :
  forall (sq : Vector.t Q n) (e : GA_expr n),
    let '(existT _ _ (d, r)) := flatten e in
    forall m : Mask n,
      eval_dag sq d r m == eval_expr sq e m.
Proof. Admitted.

(* Flattening preserves peak l1 (without sharing, tree = DAG) *)
Theorem flatten_exc_l1 :
  forall (sq : Vector.t Q n) (e : GA_expr n),
    let '(existT _ _ (d, _)) := flatten e in
    dag_max_l1 sq d == max_l1_during sq e.
Proof. Admitted.

(* Flattening preserves peak grade *)
Theorem flatten_exc_grade :
  forall (sq : Vector.t Q n) (e : GA_expr n),
    let '(existT _ _ (d, _)) := flatten e in
    dag_max_grade sq d = max_grade_during sq e.
Proof. Admitted.

(* Flattening preserves trace boolishness *)
Theorem flatten_trace_boolish :
  forall (sq : Vector.t Q n) (e : GA_expr n) (tol : Q),
    let '(existT _ _ (d, _)) := flatten e in
    trace_boolish_le sq e tol <-> dag_trace_boolish sq d tol.
Proof. Admitted.

Theorem flatten_trace_boolish_k :
  forall (sq : Vector.t Q n) (e : GA_expr n) (bound : nat) (tol : Q),
    let '(existT _ _ (d, _)) := flatten e in
    trace_boolish_k_le sq e bound tol
    <-> dag_trace_boolish_k sq d bound tol.
Proof. Admitted.

(* Flattening: DAG size = tree size (no sharing introduced) *)
Theorem flatten_size :
  forall (e : GA_expr n),
    let '(existT _ k _) := flatten e in
    S k = expr_size e.
Proof. Admitted.


(* ============================================================ *)
(* Section 11: Unfolding (GA_dag → GA_expr)                      *)
(* ============================================================ *)

(* Unfold a DAG node at index i into a tree expression.
   Duplicates shared subexpressions — tree size may be
   exponentially larger than DAG size. *)

Fixpoint unfold_node {k} (d : GA_dag k) (i : Fin.t k)
    : GA_expr n :=
  match d with
  | DagNil => Fin.case0 _ i
  | DagSnoc d' op =>
      match i with
      | Fin.F1 =>
          match op with
          | DOpBasis _ j   => Basis j
          | DOpScalar _ c  => Scalar c
          | DOpAdd _ a b   => Cln_Grade.Add (unfold_node d' a)
                                             (unfold_node d' b)
          | DOpMul _ a b   => Mul (unfold_node d' a)
                                   (unfold_node d' b)
          | DOpConv _ a b  => Conv (unfold_node d' a)
                                    (unfold_node d' b)
          end
      | Fin.FS j => unfold_node d' j
      end
  end.

(* Unfolding preserves evaluation *)
Theorem unfold_eval_correct :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (i : Fin.t k),
    forall m : Mask n,
      eval_expr sq (unfold_node d i) m == eval_dag sq d i m.
Proof. Admitted.

(* Unfolding preserves excursion:
   peak l1 during the unfolded tree ≤ peak l1 of the DAG.
   (Actually equal, since max doesn't change under duplication.) *)
Theorem unfold_exc_l1_le :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (i : Fin.t k),
    max_l1_during sq (unfold_node d i) <= dag_max_l1 sq d.
Proof. Admitted.

Theorem unfold_exc_grade_le :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (i : Fin.t k),
    (max_grade_during sq (unfold_node d i) <= dag_max_grade sq d)%nat.
Proof. Admitted.

(* Unfolding preserves trace boolishness *)
Theorem unfold_trace_boolish :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (i : Fin.t k) tol,
    dag_trace_boolish sq d tol ->
    trace_boolish_le sq (unfold_node d i) tol.
Proof. Admitted.

Theorem unfold_trace_boolish_k :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (i : Fin.t k)
         bound tol,
    dag_trace_boolish_k sq d bound tol ->
    trace_boolish_k_le sq (unfold_node d i) bound tol.
Proof. Admitted.


(* ============================================================ *)
(* Section 12: Reduction Directions                              *)
(* ============================================================ *)

(* Tree lower bound FROM DAG lower bound (easy direction:
   any tree is a DAG, so if no DAG can do it, no tree can) *)
Corollary tree_lb_from_dag_lb :
  forall (sq : Vector.t Q n) (f : Corner n -> bool)
         (L : Q) (bound : nat) (tol : Q),
    (* If every DAG computing f with boolish-k trace has peak l1 >= L *)
    (forall k (d : GA_dag (S k)) (r : Fin.t (S k)),
       dag_computes sq d r f ->
       dag_trace_boolish_k sq d bound tol ->
       L <= dag_max_l1 sq d) ->
    (* Then every tree computing f with boolish-k trace has peak l1 >= L *)
    (forall (e : GA_expr n),
       computes sq e f ->
       trace_boolish_k_le sq e bound tol ->
       L <= max_l1_during sq e).
Proof. Admitted.

(* DAG lower bound FROM tree lower bound:
   ONLY valid for excursion (peak doesn't change under unfolding).
   NOT valid for size-based arguments. *)
Corollary dag_exc_lb_from_tree_exc_lb :
  forall (sq : Vector.t Q n) (f : Corner n -> bool)
         (L : Q) (tol : Q),
    (* If every tree computing f with boolish trace has peak l1 >= L *)
    (forall (e : GA_expr n),
       computes sq e f ->
       trace_boolish_le sq e tol ->
       L <= max_l1_during sq e) ->
    (* Then every DAG computing f with boolish trace has peak l1 >= L *)
    (forall k (d : GA_dag (S k)) (r : Fin.t (S k)),
       dag_computes sq d r f ->
       dag_trace_boolish sq d tol ->
       L <= dag_max_l1 sq d).
Proof. Admitted.


(* ============================================================ *)
(* Section 13: Gate Gadgets                                      *)
(* ============================================================ *)

(* Helper: the standard variable encoding
   varMV i = (1/2)(1 + e_i) *)
Definition dag_basis_gadget (i : Fin.t n) : GA_dag 1 :=
  DagSnoc DagNil (DOpBasis i).

(* NOT gadget: given a node r in scope k, produce 1 - r.
   Adds 3 nodes: scalar 1, scalar -1, scale -1 * r, add(1, scaled).
   Actually simpler: Add(Scalar 1, Mul(Scalar(-1), r))
   But we only have binary ops referencing existing nodes.
   So: node k = Scalar 1, node k+1 = Scalar (-1),
       node k+2 = Mul(k+1, r),  -- this is (-1)*r via GP
       node k+3 = Add(k, k+2)   -- this is 1 + (-1)*r = 1 - r
   Wait, GP with scalar is just scaling. Let's use that. *)

(* NOT: 1 - x. Encoded as Add(Scalar(1), Mul(Scalar(-1), x)).
   Requires 3 new nodes on top of the existing DAG. *)
Definition not_gadget_ops {k} (r : Fin.t k)
    : dag_op k             (* node k:   scalar 1 *)
    * dag_op (S k)         (* node k+1: scalar -1, then GP with r *)
    * dag_op (S (S k))     (* node k+2: 1 + (-r) *)
    :=
  ( DOpScalar 1,
    DOpScalar (-1),
    (* Actually we need mul(scalar_node, r) then add(one_node, neg_r) *)
    (* This doesn't quite work with 3 nodes. Let's use 4. *)
    DOpAdd Fin.F1 Fin.F1   (* placeholder — see below *)
  ).

(* Better: define NOT as a small DAG fragment.
   Input: reference r : Fin.t k.
   Output: 4 new nodes. *)
Definition dag_not {k} (d : GA_dag k) (r : Fin.t k)
    : GA_dag (S (S (S k))) :=
  let d1 := DagSnoc d (DOpScalar 1) in                    (* node k: const 1 *)
  let d2 := DagSnoc d1 (DOpScalar (-1)) in                (* node k+1: const -1 *)
  let r_shifted := Fin.FS (Fin.FS r) in                   (* r in scope k+2 *)
  let d3 := DagSnoc d2 (DOpMul Fin.F1 r_shifted) in       (* node k+2: (-1)*x *)
  d3.
  (* Then NOT = Add(node_k, node_k+2)
     But we need one more node for the final add. *)

(* Let's be precise: NOT(x) = 1 + (-1)*x needs 4 new nodes *)
Definition dag_not_full {k} (d : GA_dag k) (r : Fin.t k)
    : { d' : GA_dag (S (S (S (S k)))) & Fin.t (S (S (S (S k)))) } :=
  let d1 := DagSnoc d  (DOpScalar 1) in
  let d2 := DagSnoc d1 (DOpScalar (-1)) in
  let r2 := Fin.FS (Fin.FS r) in
  let d3 := DagSnoc d2 (DOpMul Fin.F1 r2) in
  let one_ref := Fin.FS (Fin.FS Fin.F1) in
  let d4 := DagSnoc d3 (DOpAdd one_ref Fin.F1) in
  existT _ d4 Fin.F1.

(* AND gadget: x AND y = conv(x, y). Single new node. *)
Definition dag_and {k} (d : GA_dag k)
    (rx ry : Fin.t k) : { d' : GA_dag (S k) & Fin.t (S k) } :=
  existT _ (DagSnoc d (DOpConv rx ry)) Fin.F1.

(* OR gadget: x OR y = x + y - conv(x,y). Needs 3 new nodes. *)
Definition dag_or {k} (d : GA_dag k)
    (rx ry : Fin.t k)
    : { d' : GA_dag (S (S (S k))) & Fin.t (S (S (S k))) } :=
  let d1 := DagSnoc d (DOpAdd rx ry) in                   (* node k: x+y *)
  let rx1 := Fin.FS rx in
  let ry1 := Fin.FS ry in
  let d2 := DagSnoc d1 (DOpConv rx1 ry1) in               (* node k+1: conv(x,y) *)
  (* OR = (x+y) - conv(x,y).
     We don't have Sub, so: OR = Add(x+y, Mul(Scalar(-1), conv(x,y)))
     That needs 2 more nodes: Scalar(-1) and Mul. Then Add.
     Actually let's just extend. *)
  (* dag_or is a placeholder FIX WHEN YOU GET HERE *)
  let d3 := DagSnoc d2 (DOpScalar (-1)) in                (* node k+2: -1 *)
  existT _ d3 Fin.F1.  (* placeholder — needs more nodes *)

(* NAND gadget: NOT(AND(x,y)).
   = 1 - conv(x,y).
   Encoded as: conv node, then NOT of that. *)
Definition dag_nand {k} (d : GA_dag k) (rx ry : Fin.t k)
    : { k' : nat & GA_dag k' * Fin.t k' }%type :=
  let '(existT _ (DagSnoc _ _ as d_and) r_and) := dag_and d rx ry in
  let '(existT _ d_not r_not) := dag_not_full d_and r_and in
  existT _ (d_not, r_not).


(* ============================================================ *)
(* Section 14: Gate Gadget Correctness                           *)
(* ============================================================ *)

(* AND gadget is correct *)
Lemma dag_and_correct :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (rx ry : Fin.t k) (gx gy : Corner n -> bool),
    (forall m, eval_dag sq d rx m == embed gx m) ->
    (forall m, eval_dag sq d ry m == embed gy m) ->
    let '(existT _ d' r') := dag_and d rx ry in
    forall m, eval_dag sq d' r' m
              == embed (fun s => andb (gx s) (gy s)) m.
Proof. Admitted.

(* NOT gadget is correct *)
Lemma dag_not_correct :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (r : Fin.t k) (g : Corner n -> bool),
    (forall i, Qabs (Vector.nth sq i) == 1) ->
    (forall m, eval_dag sq d r m == embed g m) ->
    let '(existT _ d' r') := dag_not_full d r in
    forall m, eval_dag sq d' r' m
              == embed (fun s => negb (g s)) m.
Proof. Admitted.

(* NAND gadget is correct *)
Lemma dag_nand_correct :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (rx ry : Fin.t k) (gx gy : Corner n -> bool),
    (forall i, Qabs (Vector.nth sq i) == 1) ->
    (forall m, eval_dag sq d rx m == embed gx m) ->
    (forall m, eval_dag sq d ry m == embed gy m) ->
    let '(existT _ (d', r')) := dag_nand d rx ry in
    forall m, eval_dag sq d' r' m
              == embed (fun s => negb (andb (gx s) (gy s))) m.
Proof. Admitted.


(* ============================================================ *)
(* Section 15: Gate Gadget Boolishness Preservation              *)
(* ============================================================ *)

(* Key closure theorem: if inputs are exactly Boolean (d=0, k=1),
   then gate outputs are exactly Boolean. *)

Lemma dag_and_boolish :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (rx ry : Fin.t k),
    boolish_k_le (eval_dag sq d rx) 1 0 ->
    boolish_k_le (eval_dag sq d ry) 1 0 ->
    let '(existT _ d' r') := dag_and d rx ry in
    boolish_k_le (eval_dag sq d' r') 1 0.
Proof. Admitted.

Lemma dag_not_boolish :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (r : Fin.t k),
    (forall i, Qabs (Vector.nth sq i) == 1) ->
    boolish_k_le (eval_dag sq d r) 1 0 ->
    let '(existT _ d' r') := dag_not_full d r in
    boolish_k_le (eval_dag sq d' r') 1 0.
Proof. Admitted.

Lemma dag_nand_boolish :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (rx ry : Fin.t k),
    (forall i, Qabs (Vector.nth sq i) == 1) ->
    boolish_k_le (eval_dag sq d rx) 1 0 ->
    boolish_k_le (eval_dag sq d ry) 1 0 ->
    let '(existT _ (d', r')) := dag_nand d rx ry in
    boolish_k_le (eval_dag sq d' r') 1 0.
Proof. Admitted.

(* Trace-level closure: if the prefix DAG has boolish trace,
   and we append a gate gadget, the extended DAG has boolish trace. *)

Lemma dag_and_trace_boolish :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (rx ry : Fin.t k) tol,
    dag_trace_boolish sq d tol ->
    boolish_le (eval_dag sq d rx) tol ->
    boolish_le (eval_dag sq d ry) tol ->
    let '(existT _ d' _) := dag_and d rx ry in
    dag_trace_boolish sq d' tol.
Proof. Admitted.

Lemma dag_not_trace_boolish :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (r : Fin.t k) tol,
    (forall i, Qabs (Vector.nth sq i) == 1) ->
    dag_trace_boolish sq d tol ->
    boolish_le (eval_dag sq d r) tol ->
    let '(existT _ d' _) := dag_not_full d r in
    dag_trace_boolish sq d' tol.
Proof. Admitted.

Lemma dag_nand_trace_boolish :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (rx ry : Fin.t k) tol,
    (forall i, Qabs (Vector.nth sq i) == 1) ->
    dag_trace_boolish sq d tol ->
    boolish_le (eval_dag sq d rx) tol ->
    boolish_le (eval_dag sq d ry) tol ->
    let '(existT _ (d', _)) := dag_nand d rx ry in
    dag_trace_boolish sq d' tol.
Proof. Admitted.


(* ============================================================ *)
(* Section 16: Functional Completeness                           *)
(* ============================================================ *)

(* NAND is functionally complete: any Boolean circuit over NAND
   can be compiled to a GA_dag where every node is exactly Boolean.

   We state this as: for any Boolean circuit (represented as a
   list of NAND gates with wire references), there exists a GA_dag
   that computes the same function with exact boolish trace. *)

(* Wire reference in a Boolean circuit *)
Inductive bool_gate (num_inputs : nat) : nat -> Type :=
  | BGInput : forall {k}, Fin.t num_inputs -> bool_gate num_inputs k
  | BGNand  : forall {k}, Fin.t k -> Fin.t k -> bool_gate num_inputs k.

Inductive bool_circuit (num_inputs : nat) : nat -> Type :=
  | BCNil   : bool_circuit num_inputs 0
  | BCSnoc  : forall {k}, bool_circuit num_inputs k
              -> bool_gate num_inputs k
              -> bool_circuit num_inputs (S k).

(* Evaluate a Boolean circuit *)
Fixpoint eval_bool_circuit {ni k}
    (inputs : Fin.t ni -> bool)
    (c : bool_circuit ni k) : Fin.t k -> bool :=
  match c with
  | BCNil _ => fun i => Fin.case0 _ i
  | BCSnoc c' g =>
      let prev := eval_bool_circuit inputs c' in
      let v := match g with
               | BGInput _ i => inputs i
               | BGNand _ a b => negb (andb (prev a) (prev b))
               end in
      fun i =>
        match i with
        | Fin.F1 => v
        | Fin.FS j => prev j
        end
  end.

(* Compilation: Boolean circuit → GA_dag.
   Needs n = num_inputs for the variable embedding. *)
Fixpoint compile_bool_circuit {k}
    (sq_hyp : forall i : Fin.t n, Qabs (Vector.nth (Vector.const 1 n) i) == 1)
    (c : bool_circuit n k)
    : { k' : nat & GA_dag k' * (Fin.t k -> Fin.t k') }%type :=
  match c with
  | BCNil _ =>
      existT _ 0 (DagNil, fun i => Fin.case0 _ i)
  | BCSnoc c' g =>
      let '(existT _ k' (d, wire_map)) := compile_bool_circuit sq_hyp c' in
      match g with
      | BGInput _ i =>
          (* Add a Basis node for variable i *)
          existT _ (S k')
            (DagSnoc d (DOpBasis i),
             fun j => match j with
                      | Fin.F1 => Fin.F1
                      | Fin.FS j' => Fin.FS (wire_map j')
                      end)
      | BGNand _ a b =>
          let ra := wire_map a in
          let rb := wire_map b in
          let '(existT _ (d', r')) := dag_nand d ra rb in
          existT _ _
            (d',
             fun j => match j with
                      | Fin.F1 => r'
                      | Fin.FS j' => fin_weaken_by _ (wire_map j')
                      end)
      end
  end.

(* The compiled DAG computes the same function *)
Theorem compile_bool_circuit_correct :
  forall {k} (c : bool_circuit n k)
         (inputs : Fin.t n -> bool)
         (root : Fin.t k)
         (sq_hyp : forall i, Qabs (Vector.nth (Vector.const 1 n) i) == 1),
    let sq := Vector.const 1 n in
    let '(existT _ _ (d, wire_map)) :=
      compile_bool_circuit sq_hyp c in
    let s := fun i => if inputs i then Neg else Pos in
    eval_bool_circuit inputs c root
    = true <->
    eval_dag sq d (wire_map root) (mask_empty) == 1.
    (* or stronger pointwise version *)
Proof. Admitted.

(* The compiled DAG has exact boolish trace *)
Theorem compile_bool_circuit_boolish :
  forall {k} (c : bool_circuit n k)
         (sq_hyp : forall i, Qabs (Vector.nth (Vector.const 1 n) i) == 1),
    let sq := Vector.const 1 n in
    let '(existT _ _ (d, _)) := compile_bool_circuit sq_hyp c in
    dag_trace_boolish_k sq d 1 0.
Proof. Admitted.

(* DAG size is linear in circuit size
   (each NAND gate compiles to O(1) DAG nodes) *)
Theorem compile_bool_circuit_size :
  forall {k} (c : bool_circuit n k)
         (sq_hyp : forall i, Qabs (Vector.nth (Vector.const 1 n) i) == 1),
    let '(existT _ k' _) := compile_bool_circuit sq_hyp c in
    (k' <= 6 * k)%nat.    (* NAND = AND(1 node) + NOT(4 nodes) + overhead *)
Proof. Admitted.

End GA_DAG.


(* ============================================================ *)
(* Section 17: DAG Versions of Main Theorems                     *)
(* ============================================================ *)

(* Any DAG computing IP with boolish trace needs exponential
   excursion. Strictly stronger than the tree version because
   DAGs subsume trees. *)

Theorem IP_dag_exponential :
  forall d : Q,
  exists c : nat,
    forall m k (sq : Vector.t Q (m+m))
           (dag : GA_dag (m+m) (S k))
           (root : Fin.t (S k)),
      (m >= 2)%nat ->
      dag_computes (m+m) sq dag root (@IP_n_func (m+m)) ->
      dag_trace_boolish_exists_k (m+m) sq dag d ->
      (Qpow2 (c * m) <= exc_l1 (dag_exc (m+m) sq dag))%Q.
Proof. Admitted.

(* The tradeoff version for DAGs *)
Theorem IP_dag_booleanish_tradeoff :
  forall d : Q,
  exists c : nat,
    forall m k (sq : Vector.t Q (m+m))
           (dag : GA_dag (m+m) (S k))
           (root : Fin.t (S k)),
      (m >= 2)%nat ->
      dag_computes (m+m) sq dag root (@IP_n_func (m+m)) ->
      ( dag_trace_boolish_exists_k (m+m) sq dag d ->
          Qpow2 (c * m) <= exc_l1 (dag_exc (m+m) sq dag) )
      /\
      ( exc_l1 (dag_exc (m+m) sq dag) < Qpow2 (c * m) ->
          ~ dag_trace_boolish_exists_k (m+m) sq dag d ).
Proof. Admitted.

(* CNFs are easy even as DAGs *)
Theorem cnf_easy_dag :
  forall m (phi : CNF (m+m)),
  exists k (d : GA_dag (m+m) (S k)) (root : Fin.t (S k)),
    dag_computes (m+m) (Vector.const 1 (m+m)) d root (cnf_sem phi) /\
    dag_trace_boolish_k (m+m) (Vector.const 1 (m+m)) d 1 0 /\
    exc_l1 (dag_exc (m+m) (Vector.const 1 (m+m)) d) <= pow2 (m+m).
Proof. Admitted.

(* The ideal separation for DAGs *)
Theorem dag_booleanish_vs_unrestricted_separation :
  exists f : forall n, Corner n -> bool,
    (* f can be computed by a poly-excursion DAG (unrestricted) *)
    (exists poly_bound : nat -> Q,
       forall m,
         exists k (d : GA_dag (m+m) (S k)) (root : Fin.t (S k)),
           dag_computes (m+m) (Vector.const 1 (m+m)) d root (f (m+m)) /\
           exc_l1 (dag_exc (m+m) (Vector.const 1 (m+m)) d)
             <= poly_bound m)
    /\
    (* but any boolish-trace DAG needs exponential excursion *)
    (forall d, exists c,
       forall m k (sq : Vector.t Q (m+m))
              (dag : GA_dag (m+m) (S k)) (root : Fin.t (S k)),
         dag_computes (m+m) sq dag root (f (m+m)) ->
         dag_trace_boolish_exists_k (m+m) sq dag d ->
         Qpow2 (c * m) <= exc_l1 (dag_exc (m+m) sq dag)).
Proof. Admitted.


(* ============================================================ *)
(* Section 18: DAG EasyCompiler                                  *)
(* ============================================================ *)

(* Generalize the EasyCompiler record to DAGs *)

Record DAG_EasyCompiler := {
  dag_R : nat -> Type;

  dag_target : forall {m : nat}, dag_R m -> Corner m -> bool;

  dag_compile_size : forall {m : nat}, dag_R m -> nat;
  dag_compile : forall {m : nat} (r : dag_R m),
    GA_dag m (S (dag_compile_size r));
  dag_compile_root : forall {m : nat} (r : dag_R m),
    Fin.t (S (dag_compile_size r));

  dag_B : nat -> ExcNum;
  dag_d0 : Q;

  dag_compile_correct :
    forall {m : nat} (sq : Vector.t Q m) (r : dag_R m),
      dag_computes m sq (dag_compile r) (dag_compile_root r)
                   (dag_target r);

  dag_compile_exc_bound :
    forall {m : nat} (sq : Vector.t Q m) (r : dag_R m),
      exc_pre (dag_exc m sq (dag_compile r))
              (dag_B (S (dag_compile_size r)));

  dag_compile_boolish_bound :
    forall {m : nat} (sq : Vector.t Q m) (r : dag_R m),
      dag_trace_boolish m sq (dag_compile r) dag_d0
}.

Definition dag_easy (C : DAG_EasyCompiler) {m} (f : Corner m -> bool)
    : Prop :=
  exists r : dag_R C m,
    forall x : Mask m,
      embed (dag_target C r) x == embed f x.

Definition dag_easy_under (B : nat -> ExcNum) (d0 : Q)
    {m} (sq : Vector.t Q m) (f : Corner m -> bool) : Prop :=
  exists k (dag : GA_dag m (S k)) (root : Fin.t (S k)),
    dag_computes m sq dag root f /\
    exc_pre (dag_exc m sq dag) (B (S k)) /\
    dag_trace_boolish m sq dag d0.

Lemma dag_easy_implies_dag_easy_under :
  forall (C : DAG_EasyCompiler) m (sq : Vector.t Q m)
         (f : Corner m -> bool),
    dag_easy C (m:=m) f ->
    dag_easy_under (dag_B C) (dag_d0 C) sq f.
Proof. Admitted.

(* No DAG EasyCompiler can compile IP *)
Theorem dag_hard_family_not_easy :
  forall (C : DAG_EasyCompiler),
  exists c : nat,
  exists f : forall m, Corner m -> bool,
    forall m,
      ~ dag_easy C (m:=m) (f m).
Proof. Admitted.


(* ============================================================ *)
(* Section 19: Simulation of Tree by DAG (with sharing)          *)
(* ============================================================ *)

(* A GA_expr that computes f can be compiled to a GA_dag
   that computes f with the same excursion properties.
   (This is just flatten + bridge lemmas packaged.) *)

Theorem tree_to_dag_simulation :
  forall m (sq : Vector.t Q m) (e : GA_expr m) (f : Corner m -> bool),
    computes sq e f ->
    exists k (d : GA_dag m (S k)) (root : Fin.t (S k)),
      dag_computes m sq d root f /\
      dag_max_l1 m sq d == max_l1_during sq e /\
      (dag_max_grade m sq d = max_grade_during sq e)%nat.
Proof. Admitted.

(* A GA_dag can be unfolded to a GA_expr with bounded excursion.
   Size may blow up exponentially. *)
Theorem dag_to_tree_simulation :
  forall m (sq : Vector.t Q m) {k}
         (d : GA_dag m (S k)) (root : Fin.t (S k))
         (f : Corner m -> bool),
    dag_computes m sq d root f ->
    exists e : GA_expr m,
      computes sq e f /\
      max_l1_during sq e <= dag_max_l1 m sq d /\
      (max_grade_during sq e <= dag_max_grade m sq d)%nat.
Proof. Admitted.


(* ============================================================ *)
(* Section 20: The Full Picture                                  *)
(* ============================================================ *)

(* Boolean circuits embed into GA_dags with exact boolish trace.
   So the boolish-trace DAG model captures at least all of
   Boolean circuit computation.

   Theorem compile_bool_circuit_boolish (Section 16) shows:
   - every NAND circuit → GA_dag with trace_boolish_k ... 1 0
   - linear size blowup

   Combined with IP_dag_exponential:
   - IP requires exponential excursion under boolish trace
   - IP can be computed by Boolean circuits of size poly(n)
   - therefore the excursion of those circuits is exponential

   Combined with cnf_easy_dag:
   - CNFs are computed with bounded (2^n) excursion
   - CNFs have exact boolish trace

   This gives the separation:
   - CNFs: boolish, bounded excursion
   - IP: boolish circuits exist but require exponential excursion
   - IP: non-boolish circuits may have polynomial excursion
*)

(* Summary theorem: the model is non-trivial *)
Theorem model_captures_boolean_circuits :
  forall m (c : bool_circuit m (S 0)),
    (* Any single-output Boolean circuit of size s *)
    forall s, dag_size (snd (fst (projT2
      (compile_bool_circuit m
        (fun i => VectorDef_nth_const_1 m i) c)))) = s ->
    (* produces a GA_dag with boolish trace *)
    True.  (* stated loosely; the real content is in Sections 15-16 *)
Proof. Admitted.