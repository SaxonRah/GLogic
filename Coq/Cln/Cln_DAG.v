(*

  ============================================================
  Plan
  ============================================================


# Phase 0 — Lock in the goalposts (so you don’t drift)

### Target end state of DAG setup

You want `Cln_DAG.v` to support **three layers** cleanly:

1. **Semantics layer**: evaluate every node; prove snoc/FS/F1 lemmas.
2. **Measurement layer**: node-wise max grade / max ℓ₁; plus “proper-node peak” (exclude root).
3. **Trace layer**: node-wise (boolish) trace predicates that quantify over *all nodes*, compatible with flatten/unfold.

Everything else (gadgets, CNF, IP) is downstream.

---

# Phase 1 — Make evaluation rock-solid (do this first)

## 1.1 Prove the snoc evaluation lemmas

These are the “assembly instructions” for every later proof:

* **Newest node evaluation**: value at `Fin.F1` after snoc
* **Old node evaluation**: value at `Fin.FS i` after snoc

You’ll use these constantly to show “adding a node preserves previous nodes” and to reason by recursion on the DAG.

**Deliverable**: lemmas like

* `eval_dag_env_snoc_F1`
* `eval_dag_env_snoc_FS`

(Names don’t matter; the pattern does.)

## 1.2 Prove append / composition lemmas

You likely have `dag_append` or similar. Prove:

* `eval_dag_env (append d1 d2)` agrees with `d1` on old indices
* the “shift” of indices is correct
* evaluation of nodes coming from the right side is evaluation under the extended environment

**Deliverable**:

* `eval_dag_append_left`
* `eval_dag_append_right`
* any index-shifting lemmas you need

> Rule: if an index-shift lemma feels annoying, prove it immediately—these are the core friction points.

---

# Phase 2 — Define the metrics you’ll actually use for lower bounds

## 2.1 Keep your current max metrics

You already have:

* `dag_max_l1`
* `dag_max_grade`

Keep them—they’re useful baselines.

## 2.2 Add the “proper-node peak” metric (exclude root)

This is the single most important upgrade.

Define:

* `dag_max_l1_except sq d root : Q` = max ℓ₁ over nodes `i` with `i ≠ root`.

Also define the grade version if you want:

* `dag_max_grade_except sq d root : nat`

**Deliverables (lemmas you’ll use constantly):**

1. `dag_max_l1_except_le : dag_max_l1_except ≤ dag_max_l1`
2. `dag_max_l1_except_spec`:

   * if `i ≠ root`, then `l1_norm(value i) ≤ dag_max_l1_except`
3. “snoc behavior”:

   * if root is newest, new node excluded
   * if root is old, new node included

> This is where the output-mass shortcut dies. Make this definition/lemmas airtight before moving on.

---

# Phase 3 — Port trace predicates cleanly to DAG

You likely already did much of this. The checklist:

## 3.1 Nodewise trace predicates quantify over all nodes

You want:

* `dag_trace_boolish_k` means: for every node `i`, `boolish_k_le (value i) k d` (or equivalent)
* `dag_trace_boolish_poly_size` means: there exists k bounded by poly(size_dag) s.t. trace holds

**Deliverables:**

* monotonicity: `k1 ≤ k2` implies `dag_trace_boolish_k k1 -> dag_trace_boolish_k k2`
* snoc closure: if trace holds on prefix and newest node satisfies property, trace holds on snoc
* append closure: trace for append if both parts satisfy (watch index-shifts)

## 3.2 Make “size of DAG” a first-class thing

You need:

* `size_dag : GA_dag k -> nat`
* `size_dag (snoc d op) = S (size_dag d)` (or similar)
* `size_dag (append d1 d2) = size_dag d1 + size_dag d2`

This matters later when you bind k by poly(size).

---

# Phase 4 — Tree ↔ DAG bridges (flatten/unfold) as “regression tests”

This phase is how you keep yourself from building a DAG semantics that subtly disagrees with the tree model.

## 4.1 Flatten (tree → DAG)

Prove (even if only for the core ops first):

* `eval_expr sq e = eval_dag_node sq (flatten e).root`
* `dag_max_l1` bounds tree `max_l1_during` and vice versa (whichever direction is true with your definitions)
* `trace_boolish` on tree implies `dag_trace_boolish` on flatten, and conversely for unfold

## 4.2 Unfold (DAG → tree) for soundness (optional but powerful)

If you have `unfold`, prove:

* `eval_dag root` equals `eval_expr (unfold d root)`
* `max_l1_during (unfold ...) ≥ dag_max_l1` (trees can duplicate, so peaks can only go up)
* similarly for `dag_max_l1_except` vs tree “proper subexpr peak” (if you have it)

> These bridge theorems are your guarantee that DAG work isn’t drifting away from the established semantics.

---

# Phase 5 — Establish “baseline results” in DAG world (quick wins)

These aren’t the “internal-dynamics” theorems yet, but they confirm your DAG machinery is usable.

## 5.1 DAG version of the output-mass bound (trivial but sanity)

Prove:

* If `dag_computes sq d root IP`, then `pow2(m-1) ≤ dag_max_l1 sq d`.

This should be 5–10 lines and confirms “computes” + eval works.

## 5.2 Prove a nontrivial lemma about `dag_max_l1_except` (root forcing)

At the root node, depending on op:

* Add: some input node has ℓ₁ ≥ L/2
* Mul/Conv: some input node has ℓ₁ ≥ sqrt(L)

This is still output-driven, but it forces you to use node semantics and index constraints, and it exercises the “except root” definition hard.

---

# Phase 6 — Decide your “boolish tooth” (do not skip this)

Before you attempt the real pre-peak tradeoff theorem, pick ONE of these and implement it:

### Option A (cleanest): coefficient budget

Strengthen boolish to include `Σ |c_i| ≤ B` and bind B polynomially.

### Option B: scalar-cost in size

Count rational magnitude/bitlength in `size_expr` / `size_dag`.

### Option C: restrict scalars

Allow only a controlled scalar set.

**Deliverable**: a definition where “trace-poly” actually constrains something that matters for ℓ₁.

Without this, you’ll likely spend weeks chasing a theorem that’s simply false or vacuous.

---

# Phase 7 — State the real target theorem (even if admitted for now)

Once `dag_max_l1_except` exists and your trace has teeth, put the flagship theorem in the file:

> If a DAG computes IP and satisfies poly trace, then some **proper internal node** has exponential ℓ₁.

Something like:

```coq
Conjecture IP_booleanish_forces_prepeak_DAG :
  forall d:Q, exists c:nat,
  forall m sq (D : GA_dag (m+m)) root,
    dag_computes sq D root IP ->
    dag_trace_boolish_poly_size sq D d ->
    Qpow2 (c*m) <= dag_max_l1_except sq D root.
```

This is the “program succeeds” checkpoint.

---

# What to work on *daily* while finishing `Cln_DAG.v`

Here’s a tight loop you can follow:

1. Prove one semantics lemma (snoc/append/index shift).
2. Immediately add the corresponding metric lemma (how max changes under snoc/append).
3. Immediately add the corresponding trace lemma (how trace changes under snoc/append).
4. Run a regression proof: flatten/unfold lemma that uses the new facts.

This prevents “1k lines of infrastructure” from becoming a tangled pile.

---

# What not to do yet

* Don’t build CNF/IP gadgets on DAG until Phase 1–4 are stable.
* Don’t chase `prepeak` lower bounds before you’ve chosen the “boolish tooth.”
* Don’t let placeholder gadgets (`dag_or` etc.) leak into correctness claims.

*)


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

Require Import Coq.micromega.Lra.

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

(*
Inductive dag_op : nat -> Type :=
  | DOpBasis  : forall {k}, Fin.t n -> dag_op k
  | DOpScalar : forall {k}, Q -> dag_op k
  | DOpAdd    : forall {k}, Fin.t k -> Fin.t k -> dag_op k
  | DOpMul    : forall {k}, Fin.t k -> Fin.t k -> dag_op k
  | DOpConv   : forall {k}, Fin.t k -> Fin.t k -> dag_op k.
*)

Inductive dag_op (k : nat) : Type :=
  | DOpBasis  : Fin.t n -> dag_op k
  | DOpScalar : Q -> dag_op k
  | DOpAdd    : Fin.t k -> Fin.t k -> dag_op k
  | DOpMul    : Fin.t k -> Fin.t k -> dag_op k
  | DOpConv   : Fin.t k -> Fin.t k -> dag_op k.

Arguments DOpBasis  {k} _.
Arguments DOpScalar {k} _.
Arguments DOpAdd    {k} _ _.
Arguments DOpMul    {k} _ _.
Arguments DOpConv   {k} _ _.

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
  | DOpBasis i    => basis (mask_single i)
  | DOpScalar c   => constMV c
  | DOpAdd a b    => mv_add (env a) (env b)
  | DOpMul a b    => mv_gp sq (env a) (env b)
  | DOpConv a b   => mv_conv (env a) (env b)
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
Proof. reflexivity. Qed.

Lemma eval_dag_env_snoc_FS :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k)
         (j : Fin.t k),
    eval_dag_env sq (DagSnoc d op) (Fin.FS j)
    = eval_dag_env sq d j.
Proof. reflexivity. Qed.


(* ============================================================ *)
(* Section 4: Excursion Measures (peak over all nodes)           *)
(* ============================================================ *)

Fixpoint dag_max_l1 (sq : Vector.t Q n) {k} (d : GA_dag k) : Q :=
  match d with
  | DagNil => 0
  | DagSnoc d' op =>
      let prev_env := eval_dag_env sq d' in
      let newest   := eval_op sq prev_env op in
      Qmax (dag_max_l1 sq d') (l1_norm newest)
  end.

(* The dag_max_l1 has the same "output-mass shortcut" problem. Right now:

  dag_max_l1 sq d takes the max over all nodes, including the output node.

  dag_computes pins the output node to embed(f).

  so for IP, dag_max_l1 ≥ ‖embed(IP)‖₁ is immediate, and again doesn’t look inside.

  So even in DAG-land, as long as your headline measure is “max over all nodes” including the output,
  the booleanish trace hypotheses remain dead weight for IP.

  This is not a failure; it just means the measure needs the same “pre-peak” upgrade in the DAG world.

  ---
  
  The one change that makes DAG + boolish + internal dynamics click is a "proper-node peak" (exclude the designated root) 

  In DAGs we even have a clean interface for this because the output is explicit (root : Fin.t k).

  If we define:

    dag_max_l1_except sq d root = max ℓ₁ over all nodes i ≠ root.

  This is the DAG analogue of the max_l1_subexpr idea,
  and it is much cleaner than the tree version because
  you don’t need “proper subexpression” bookkeeping—just inequality of indices.

  Once you have that, the output-mass shortcut is gone by construction.

*)

Fixpoint dag_max_l1_except (sq : Vector.t Q n) {k}
  (d : GA_dag k) : Fin.t k -> Q :=
  match d in GA_dag k0 return Fin.t k0 -> Q with
  | DagNil =>
      fun _ => 0

  | @DagSnoc k' d' op =>
      fun root0 : Fin.t (S k') =>
        let prev_env := eval_dag_env sq d' in
        let newest   := eval_op sq prev_env op in

        (* compute peak over prefix, excluding the lifted root if it lies in prefix *)
        let peak_prev :=
          (match root0 in Fin.t (S k'') return (Fin.t k'' -> Q) -> Q with
           | Fin.F1 =>
               fun _rec => dag_max_l1 sq d'
           | Fin.FS r' =>
               fun rec  => rec r'
           end) (dag_max_l1_except sq d')
        in

        (* include newest only if root0 is not the newest *)
        match root0 with
        | Fin.F1    => peak_prev
        | Fin.FS _  => Qmax peak_prev (l1_norm newest)
        end
  end.

(* 

There’s some index plumbing (root in prefix vs root = newest),
but it’s straightforward because of your Fin.F1/Fin.FS split.

This is exactly the kind of thing that is already set up to handle (already planed to prove eval_dag_env_snoc_F1/FS).

If this is defined, you can state the internal theorem you actually want:

  If a DAG computes IP and has poly-trace boolish (plus a tooth like coefficient-budget or scalar-cost),
    then dag_max_l1_except is exponential.

That’s “look inside computation” in the circuit model.

*)

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
Proof.
  intros sq k d. unfold eval_dag.
  induction d as [| k' d' IH op]; intros i.
  - inversion i.
  - dependent destruction i.
    + (* F1: newest node *)
      simpl. apply Qmax_r.
    + (* FS i: in prefix *)
      simpl.
      eapply Qle_trans.
      * exact (IH i).
      * apply Qmax_l.
Qed.

(* Every node's max_grade is bounded by the peak *)
Lemma eval_dag_grade_le_peak :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (i : Fin.t k),
    (max_grade (eval_dag sq d i) <= dag_max_grade sq d)%nat.
Proof.
  intros sq k d. unfold eval_dag.
  induction d as [| k' d' IH op]; intros i.
  - inversion i.
  - dependent destruction i.
    + simpl. apply Nat.le_max_r.
    + simpl. eapply Nat.le_trans.
      * exact (IH i).
      * apply Nat.le_max_l.
Qed.

(* Monotonicity: extending the DAG doesn't decrease the peak *)
Lemma dag_max_l1_mono :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k),
    dag_max_l1 sq d <= dag_max_l1 sq (DagSnoc d op).
Proof.
  intros. simpl. apply Qmax_l.
Qed.

Lemma dag_max_grade_mono :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k),
    (dag_max_grade sq d <= dag_max_grade sq (DagSnoc d op))%nat.
Proof.
  intros. simpl. apply Nat.le_max_l.
Qed.


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
Proof.
  intros sq k d k1 k2 tol Hle.
  induction d as [| k' d' IH op]; simpl.
  - trivial.
  - intros [Hd Hop]. split.
    + exact (IH Hd).
    + eapply boolish_k_le_mono; eassumption.
Qed.

(* Monotonicity in tol *)
Lemma dag_trace_boolish_tol_mono :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) bound tol1 tol2,
    tol1 <= tol2 ->
    dag_trace_boolish_k sq d bound tol1 ->
    dag_trace_boolish_k sq d bound tol2.
Proof.
  intros sq k d bound tol1 tol2 Hle.
  induction d as [| k' d' IH op]; simpl.
  - trivial.
  - intros [Hd Hop]. split.
    + exact (IH Hd).
    + eapply boolish_k_le_tol_mono; eassumption.
Qed.

(* Extension: if the prefix is boolish and the new node is boolish,
   the extended DAG is boolish *)
Lemma dag_trace_boolish_snoc :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k) tol,
    dag_trace_boolish sq d tol ->
    boolish_le (eval_op sq (eval_dag_env sq d) op) tol ->
    dag_trace_boolish sq (DagSnoc d op) tol.
Proof.
  intros. simpl. split; assumption.
Qed.

Lemma dag_trace_boolish_k_snoc :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k)
         bound tol,
    dag_trace_boolish_k sq d bound tol ->
    boolish_k_le (eval_op sq (eval_dag_env sq d) op) bound tol ->
    dag_trace_boolish_k sq (DagSnoc d op) bound tol.
Proof.
  intros. simpl. split; assumption.
Qed.

(* Prefix extraction: boolishness of DagSnoc implies boolishness
   of the prefix *)
Lemma dag_trace_boolish_prefix :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k) tol,
    dag_trace_boolish sq (DagSnoc d op) tol ->
    dag_trace_boolish sq d tol.
Proof.
  intros sq k d op tol [Hd _]. exact Hd.
Qed.

Lemma dag_trace_boolish_k_prefix :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k) (op : dag_op k)
         bound tol,
    dag_trace_boolish_k sq (DagSnoc d op) bound tol ->
    dag_trace_boolish_k sq d bound tol.
Proof.
  intros sq k d op bound tol [Hd _]. exact Hd.
Qed.


(* ============================================================ *)
(* Section 6: Computes Predicate                                 *)
(* ============================================================ *)

(* A DAG with a designated output node computes f *)
Definition dag_computes (sq : Vector.t Q n) {k}
    (d : GA_dag k) (root : Fin.t k) (f : Corner n -> bool) : Prop :=
  forall m : Mask n, eval_dag sq d root m == embed f m.

(* Computes implies zero Boolean distance at the output *)
Lemma dag_computes_implies_dist_zero :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (root : Fin.t k) (f : Corner n -> bool),
    dag_computes sq d root f ->
    dist_to (eval_dag sq d root) f == 0.
Proof.
  intros sq k d root f Hcomp.
  unfold dist_to, bool_dist_wrt.
  
  assert (Hext : forall m, mv_sub (eval_dag sq d root) (embed f) m == 0).
  { intro m.
    unfold mv_sub.
    (* goal: eval_dag ... m - embed f m == 0 *)
    rewrite (Hcomp m).
    ring.
  }

  (* l1_norm of the zero function is 0 *)
  eapply Qeq_trans.
  - apply l1_norm_ext. intro m. apply Hext.
  - apply l1_norm_zero.
Qed.

(* Computes implies the output is exactly Boolean *)
Lemma dag_computes_implies_boolish_0 :
  forall (sq : Vector.t Q n) {k} (d : GA_dag k)
         (root : Fin.t k) (f : Corner n -> bool),
    dag_computes sq d root f ->
    boolish_le (eval_dag sq d root) 0.
Proof.
  intros sq k d root f Hcomp.
  exists f.
  pose proof (dag_computes_implies_dist_zero
                (sq:=sq) (d:=d) (root:=root) (f:=f) Hcomp) as Hz.
  unfold dist_to in Hz.
  rewrite Hz.
  apply Qle_refl.
Qed.

(* ============================================================ *)
(* Section 7: Index Shifting Utilities                           *)
(* ============================================================ *)

(* Weaken: embed Fin.t k into Fin.t (m + k) by adding m
   to the de Bruijn index.  "Push into the older part." *)
Fixpoint fin_weaken_by (m : nat) {k} (i : Fin.t k)
    : Fin.t (m + k) :=
  match m with
  | 0%nat => i
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
  | DOpBasis i    => DOpBasis (k:=k+m) i
  | DOpScalar c   => DOpScalar (k:=k+m) c
  | DOpAdd a b    => DOpAdd (k:=k+m) (fin_lift_into m a) (fin_lift_into m b)
  | DOpMul a b    => DOpMul (k:=k+m) (fin_lift_into m a) (fin_lift_into m b)
  | DOpConv a b   => DOpConv (k:=k+m) (fin_lift_into m a) (fin_lift_into m b)
  end.


Lemma fin_weaken_by_0 :
  forall {k} (i : Fin.t k),
    fin_weaken_by 0 i = i.
Proof. reflexivity. Qed.

Lemma fin_weaken_by_S :
  forall (m : nat) {k} (i : Fin.t k),
    fin_weaken_by (S m) i = Fin.FS (fin_weaken_by m i).
Proof. reflexivity. Qed.


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
      existT _ 0%nat
        (DagSnoc DagNil (DOpBasis i), Fin.F1)

  | Scalar c =>
      existT _ 0%nat
        (DagSnoc DagNil (DOpScalar c), Fin.F1)

  | Cln_Grade.Add e1 e2 =>
      let '(existT _ k1 (d1, r1)) := flatten e1 in
      let '(existT _ k2 (d2, r2)) := flatten e2 in
      let combined := dag_append d1 d2 in
      let r1' := fin_weaken_by (S k2) r1 in
      let r2' := fin_lift_into (S k1) r2 in
      existT _ ((S k2 + S k1)%nat)
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
      existT _ ((S k2 + S k1)%nat)
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
      existT _ ((S k2 + S k1)%nat)
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
Proof.
  induction e; simpl; try reflexivity;
    destruct (flatten e1) as [k1 [d1 r1]];
    destruct (flatten e2) as [k2 [d2 r2]];
    reflexivity.
Qed.


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

Fixpoint unfold_node' {k} (d : GA_dag k) : Fin.t k -> GA_expr n :=
  match d in GA_dag k0 return Fin.t k0 -> GA_expr n with
  | DagNil =>
      fun i => Fin.case0 _ i

  | @DagSnoc k' d' op =>
      fun i : Fin.t (S k') =>
        (match i in Fin.t (S k'')
               return (Fin.t k'' -> GA_expr n) -> GA_expr n with
         | Fin.F1 =>
             fun _rec =>
               match op with
               | DOpBasis j   => Basis j
               | DOpScalar c  => Scalar c
               | DOpAdd a b   => Cln_Grade.Add (unfold_node' d' a)
                                               (unfold_node' d' b)
               | DOpMul a b   => Mul (unfold_node' d' a)
                                     (unfold_node' d' b)
               | DOpConv a b  => Conv (unfold_node' d' a)
                                      (unfold_node' d' b)
               end
         | Fin.FS j =>
             fun rec => rec j
         end) (unfold_node' d')
  end.

Definition unfold_node {k} (d : GA_dag k) (i : Fin.t k) : GA_expr n :=
  unfold_node' d i.

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

(* OR gadget: x OR y = x + y - conv(x,y)
   Implemented as: sum := x+y; and := conv(x,y);
   minus1 := -1; neg_and := minus1 * and; or := sum + neg_and.
   Adds 5 nodes total. *)
Definition dag_or {k} (d : GA_dag k) (rx ry : Fin.t k)
  : { d' : GA_dag (S (S (S (S (S k))))) & Fin.t (S (S (S (S (S k))))) } :=
  let d1 := DagSnoc d (DOpAdd rx ry) in                         (* +1 node: sum *)
  let rx1 := Fin.FS rx in
  let ry1 := Fin.FS ry in
  let d2 := DagSnoc d1 (DOpConv rx1 ry1) in                     (* +1 node: and *)

  (* after 2 snocs, old nodes are shifted by FS∘FS *)
  let sum_ref := Fin.FS Fin.F1 in                               (* node from d1 *)
  let and_ref := Fin.F1 in                                      (* newest in d2 *)

  let d3 := DagSnoc d2 (DOpScalar (-1)) in                      (* +1 node: minus1 *)
  let and_ref3 := Fin.FS and_ref in                             (* shift and into d3 *)
  let d4 := DagSnoc d3 (DOpMul Fin.F1 and_ref3) in              (* +1 node: neg_and *)
  let sum_ref4 := Fin.FS (Fin.FS sum_ref) in                    (* shift sum into d4 *)
  let neg_and_ref := Fin.F1 in                                  (* newest in d4 *)

  let d5 := DagSnoc d4 (DOpAdd sum_ref4 neg_and_ref) in          (* +1 node: or *)
  existT _ d5 Fin.F1.

(* NAND gadget: NOT(AND(x,y)).
   = 1 - conv(x,y).
   Encoded as: conv node, then NOT of that. *)
Definition dag_nand {k} (d : GA_dag k) (rx ry : Fin.t k)
  : { d' : GA_dag (S (S (S (S (S k))))) & Fin.t (S (S (S (S (S k))))) } :=
  let '(existT _ d_and r_and) := dag_and d rx ry in
  let '(existT _ d_not r_not) := dag_not_full d_and r_and in
  existT _ d_not r_not.


Lemma dag_or_size :
  forall {k} (d : GA_dag k) rx ry d' r',
    dag_or d rx ry = existT _ d' r' ->
    dag_size d' = S (S (S (S (S k)))).
Proof.
Admitted.

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
    forall m,
      let '(existT _ d' r') := dag_nand d rx ry in
      eval_dag sq d' r' m
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
    let '(existT _ d' r') := dag_nand d rx ry in
    boolish_k_le (eval_dag sq d' r') 1 0.
Proof.
Admitted.

Definition dag_nand_d {k} (d : GA_dag k) rx ry :=
  projT1 (dag_nand d rx ry).

Definition dag_nand_r {k} (d : GA_dag k) rx ry :=
  projT2 (dag_nand d rx ry).

(* "boolish_k_le (eval_dag sq (dag_nand_d d rx ry) (dag_nand_r d rx ry)) 1 0." *)

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
    dag_trace_boolish
      sq
      (let '(existT _ d' _) := dag_nand d rx ry in d')
      tol.
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

Arguments BGInput {num_inputs k} _.
Arguments BGNand  {num_inputs k} _ _.

Inductive bool_circuit (num_inputs : nat) : nat -> Type :=
  | BCNil   : bool_circuit num_inputs 0
  | BCSnoc  : forall {k}, bool_circuit num_inputs k
              -> bool_gate num_inputs k
              -> bool_circuit num_inputs (S k).

(* Evaluate a Boolean circuit *)
Fixpoint eval_bool_circuit {ni k}
    (inputs : Fin.t ni -> bool)
    (c : bool_circuit ni k) : Fin.t k -> bool :=
  match c in bool_circuit _ k0 return Fin.t k0 -> bool with
  | BCNil _ =>
      fun i => Fin.case0 _ i

  | @BCSnoc _ k' c' g =>
      let prev := eval_bool_circuit inputs c' in
      let v :=
        (match g in bool_gate _ k0 return (Fin.t k0 -> bool) -> bool with
         | BGInput i   => fun _prev => inputs i
         | BGNand a b  => fun prev0 => negb (andb (prev0 a) (prev0 b))
         end) prev
      in
      fun i : Fin.t (S k') =>
        (match i in Fin.t (S k'') return (Fin.t k'' -> bool) -> bool with
         | Fin.F1    => fun _prev => v
         | Fin.FS j  => fun prev0 => prev0 j
         end) prev
  end.

Fixpoint fin_snoc_weaken (m : nat) {k} (i : Fin.t k) : Fin.t (Nat.iter m S k) :=
  match m with
  | O => i
  | S m' => Fin.FS (fin_snoc_weaken m' i)
  end.


(* Compilation: Boolean circuit → GA_dag.
   Needs n = num_inputs for the variable embedding. *)
Fixpoint compile_bool_circuit {k}
    (sq_hyp : forall i : Fin.t n,
        Qabs (Vector.nth (Vector.const 1 n) i) == 1)
    (c : bool_circuit n k)
  : { k' : nat & GA_dag k' * (Fin.t k -> Fin.t k') }%type :=
  match c in bool_circuit _ k0
        return { k' : nat & GA_dag k' * (Fin.t k0 -> Fin.t k') }%type with
  | BCNil _ =>
      existT _ O (DagNil, fun i => Fin.case0 _ i)

  | @BCSnoc _ k' c' g =>
      let '(existT _ kd (d, wire_map)) := compile_bool_circuit sq_hyp c' in
      match g in bool_gate _ k0
            return (k0 = k' ->
                    { k'' : nat & GA_dag k'' * (Fin.t (S k') -> Fin.t k'') }%type) with
      
      | BGInput i =>
          fun _ =>
            existT _ (S kd)
              ( DagSnoc d (DOpBasis i)
              , fun j : Fin.t (S k') =>
                  (match j in Fin.t (S k'')
                         return (Fin.t k'' -> Fin.t kd) -> Fin.t (S kd) with
                   | Fin.F1    => fun _wm => Fin.F1
                   | Fin.FS j' => fun wm  => Fin.FS (wm j')
                   end) wire_map )

      | BGNand a b =>
          fun H =>
            (* Here: a b : Fin.t k0, and H : k0 = k'. We can transport them to Fin.t k' *)
            let a' : Fin.t k' := eq_rect _ Fin.t a _ H in
            let b' : Fin.t k' := eq_rect _ Fin.t b _ H in
            let ra := wire_map a' in
            let rb := wire_map b' in
            let '(existT _ d' r') := dag_nand d ra rb in
            existT _ _
              ( d'
              , fun j : Fin.t (S k') =>
                  (* dependent match on j, and lift old wires by 5 *)
                  (match j in Fin.t (S k'')
                         return (Fin.t k'' -> Fin.t kd) -> Fin.t _ with
                   | Fin.F1    => fun _wm => r'
                   | Fin.FS j' => fun wm  => fin_snoc_weaken 5%nat (wm j')
                   end) wire_map )
      end eq_refl
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
      dag_computes (n := (m+m)%nat) sq dag root (@IP_n_func ((m+m)%nat)) ->
      dag_trace_boolish_exists_k (n := (m+m)%nat) sq dag d ->
      (Qpow2 ((c * m)%nat) <= exc_l1 (dag_exc (n := (m+m)%nat) sq dag))%Q.
Proof. Admitted.


(* The tradeoff version for DAGs *)
Theorem IP_dag_booleanish_tradeoff :
  forall d : Q,
  exists c : nat,
    forall m k (sq : Vector.t Q (m+m))
           (dag : GA_dag (m+m) (S k))
           (root : Fin.t (S k)),
      (m >= 2)%nat ->
      dag_computes (n := (m+m)%nat) sq dag root (@IP_n_func ((m+m)%nat)) ->
      ( dag_trace_boolish_exists_k (n := (m+m)%nat) sq dag d ->
          (Qpow2 ((c * m)%nat) <= exc_l1 (dag_exc (n := (m+m)%nat) sq dag))%Q )
      /\
      ( (exc_l1 (dag_exc (n := (m+m)%nat) sq dag) < Qpow2 ((c * m)%nat))%Q ->
          ~ dag_trace_boolish_exists_k (n := (m+m)%nat) sq dag d ).
Proof. Admitted.

(* CNFs are easy even as DAGs *)
Theorem cnf_easy_dag :
  forall m (phi : CNF (m+m)),
  exists k (d : GA_dag (m+m) (S k)) (root : Fin.t (S k)),
    dag_computes (n := (m+m)%nat) (Vector.const 1 (m+m)) d root (cnf_sem phi) /\
    dag_trace_boolish_k (n := (m+m)%nat) (Vector.const 1 (m+m)) d 1 0 /\
    (exc_l1 (dag_exc (n := (m+m)%nat) (Vector.const 1 (m+m)) d) <= pow2 (m+m))%Q.
Proof. Admitted.

(* The ideal separation for DAGs *)
Theorem dag_booleanish_vs_unrestricted_separation :
  exists f : forall n, Corner n -> bool,
    (* f can be computed by a poly-excursion DAG (unrestricted) *)
    (exists poly_bound : nat -> Q,
       forall m,
         exists k (d : GA_dag (m+m) (S k)) (root : Fin.t (S k)),
           dag_computes (n := (m+m)%nat) (Vector.const 1 (m+m)) d root (f (m+m)%nat) /\
           (exc_l1 (dag_exc (n := (m+m)%nat) (Vector.const 1 (m+m)) d)
              <= poly_bound m)%Q)
    /\
    (* but any boolish-trace DAG needs exponential excursion *)
    (forall d, exists c,
       forall m k (sq : Vector.t Q (m+m))
              (dag : GA_dag (m+m) (S k)) (root : Fin.t (S k)),
         dag_computes (n := (m+m)%nat) sq dag root (f (m+m)%nat) ->
         dag_trace_boolish_exists_k (n := (m+m)%nat) sq dag d ->
         (Qpow2 ((c * m)%nat) <= exc_l1 (dag_exc (n := (m+m)%nat) sq dag))%Q).
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
      dag_computes (n := m) sq (dag_compile r) (dag_compile_root r)
                   (dag_target r);

  dag_compile_exc_bound :
    forall {m : nat} (sq : Vector.t Q m) (r : dag_R m),
      exc_pre (dag_exc (n := m) sq (dag_compile r))
              (dag_B (S (dag_compile_size r)));

  dag_compile_boolish_bound :
    forall {m : nat} (sq : Vector.t Q m) (r : dag_R m),
      dag_trace_boolish (n := m) sq (dag_compile r) dag_d0
}.

Definition dag_easy (C : DAG_EasyCompiler) {m} (f : Corner m -> bool)
    : Prop :=
  exists r : dag_R C m,
    forall x : Mask m,
      embed (dag_target C r) x == embed f x.

Definition dag_easy_under (B : nat -> ExcNum) (d0 : Q)
    {m} (sq : Vector.t Q m) (f : Corner m -> bool) : Prop :=
  exists k (dag : GA_dag m (S k)) (root : Fin.t (S k)),
    dag_computes (n := m) sq dag root f /\
    exc_pre (dag_exc (n := m) sq dag) (B (S k)) /\
    dag_trace_boolish (n := m) sq dag d0.

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
      dag_computes (n := m) sq d root f /\
      dag_max_l1 (n := m) sq d == max_l1_during sq e /\
      (dag_max_grade (n := m) sq d = max_grade_during sq e)%nat.
Proof. Admitted.

(* A GA_dag can be unfolded to a GA_expr with bounded excursion.
   Size may blow up exponentially. *)
Theorem dag_to_tree_simulation :
  forall m (sq : Vector.t Q m) {k}
         (d : GA_dag m (S k)) (root : Fin.t (S k))
         (f : Corner m -> bool),
    dag_computes (n := m) sq d root f ->
    exists e : GA_expr m,
      computes sq e f /\
      max_l1_during sq e <= dag_max_l1 (n := m) sq d /\
      (max_grade_during sq e <= dag_max_grade (n := m) sq d)%nat.
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
Require Import Coq.Vectors.Vector.
Require Import Coq.Vectors.Fin.
Require Import Coq.QArith.QArith.
Require Import Coq.micromega.Lia.   (* for lia *)

Import VectorNotations.

(* Local helper: nth (const a) = a *)
Lemma my_Vector_nth_const :
  forall (A : Type) (a : A) n,
    forall i : Fin.t n,
      Vector.nth (Vector.const a n) i = a.
Proof.
  intros A a n i.
  revert n i.
  (* now n and i are both generalized so IH matches exactly *)
  fix IH 1.
  intros n i.
  destruct n as [|n].
  - inversion i.
  - dependent destruction i; simpl.
    + reflexivity.
    + apply IH.
Qed.

(* Local helper: 0 <= 1 in Q (in case Qle_0_1 isn't available) *)
Lemma my_Qle_0_1 : (0 <= (1:Q))%Q.
Proof.
  unfold Qle; simpl; lia.
Qed.

(* This is the replacement for your missing VectorDef_nth_const_1 *)
Lemma VectorDef_nth_const_1_abs :
  forall m (i : Fin.t m),
    Qabs (Vector.nth (Vector.const (1:Q) m) i) == 1.
Proof.
  intros m i.
  rewrite (@my_Vector_nth_const Q (1:Q) m i).
  rewrite Qabs_pos; [reflexivity | exact my_Qle_0_1 ].
Qed.

Theorem model_captures_boolean_circuits :
  forall m (c : bool_circuit m (S 0)),
    forall s,
      dag_size (fst (projT2
        (compile_bool_circuit (n := m)
          (fun i : Fin.t m => @VectorDef_nth_const_1_abs m i) c))) = s ->
    True.
Proof. Admitted.

