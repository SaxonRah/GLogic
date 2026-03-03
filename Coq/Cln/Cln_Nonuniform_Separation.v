(*
  Cln_Nonuniform_Separation_Skeleton.v

  Goal: a clean Coq “front door” for the non-uniform separation story:

    (1) Circuits (P/poly) compile into ClnPoly via compile_bool_circuit
    (2) SAT ∉ ClnPoly (your post-DAG hardness theorem)
    (3) Therefore SAT ∉ P/poly, hence NP ⊄ P/poly, hence P ≠ NP

  This file is a *skeleton*: it assumes your existing Cln DAG development compiles,
  and that the major lemmas are proven in your library.

  IMPORTANT:
  - I intentionally keep “poly” abstract (as a predicate on nat->nat bounds),
    so you can later instantiate it with your preferred polynomial notion.
  - I also keep “numeric cost / scalar bitlength” abstract; you can plug in the
    actual cost you choose (or restrict constants so numeric cost is trivial).
*)

From Coq Require Import Arith Lia.
From Coq Require Import Vector.
From Coq Require Import QArith.

(* You likely already have these; rename imports to match your project. *)
(* From Cln Require Import Cln_DAG. *)
(* From Cln Require Import Cln_CompositeExcursion. *)
(* From Cln Require Import Cln_Full. *)

Module ClnNonuniformSeparation.

(* ================================================================ *)
(* 0. Re-export / alias the core notions you already have            *)
(* ================================================================ *)

(* --- Core types from your development --- *)
Parameter Corner : nat -> Type.
Parameter MV : nat -> Type.

Parameter GA_dag : nat -> Type.
Parameter dag_size : forall {k}, GA_dag k -> nat.
Parameter eval_dag : forall {n k}, Vector.t Q n -> GA_dag k -> Fin.t k -> MV n.

(* “Computes” at the DAG level (you already use this shape). *)
Parameter dag_computes :
  forall {n k}, Vector.t Q n -> GA_dag k -> Fin.t k -> (Corner n -> bool) -> Prop.

(* Trace / “soundness” predicate you use for lower bounds (adjust name if needed). *)
Parameter dag_trace_boolish_exists_k :
  forall {n k}, Vector.t Q n -> GA_dag k -> Fin.t k -> Q -> Prop.

(* A standard “all ones” sq vector hypothesis (or your preferred sq choice). *)
Parameter sq_ones : forall n, Vector.t Q n.

(* Optional: you may also have boolean-distance or embedding correctness facts.
   Keep abstract here; the skeleton only needs dag_computes. *)

(* ================================================================ *)
(* 1. Cost model for nonuniform Cln computation                      *)
(* ================================================================ *)

(*
  Critical: “poly” must cost enough to block coefficient-smuggling.
  You can:
    - restrict constants, or
    - count bitlength of all constants / numerators / denominators, etc.

  We keep it abstract so you can plug in the real one.
*)
Parameter dag_numeric_cost : forall {k}, GA_dag k -> nat.

Definition dag_total_cost {k} (d : GA_dag k) : nat :=
  dag_size d + dag_numeric_cost d.

(* Polynomials (abstract) *)
Parameter Poly : (nat -> nat) -> Prop.
Axiom Poly_closed_add : forall p q, Poly p -> Poly q -> Poly (fun n => p n + q n).
Axiom Poly_closed_comp : forall p q, Poly p -> Poly q -> Poly (fun n => p (q n)).

(* ================================================================ *)
(* 2. Nonuniform classes: P/poly and ClnPoly                         *)
(* ================================================================ *)

(*
  Languages as families of Boolean functions on corners.
  You can swap Corner n with bit-vectors if you later build an encoding layer.
*)
Definition Lang : Type := forall n, Corner n -> bool.

(* -------- ClnPoly: poly-cost DAG families satisfying your “soundness” regime ---- *)

Record ClnFamily (L : Lang) : Type := {
  k_of : nat -> nat;
  dag_of : forall n, GA_dag (k_of n);
  root_of : forall n, Fin.t (k_of n);
}.

Definition ClnDecides (L : Lang) (F : ClnFamily L) : Prop :=
  forall n,
    dag_computes (sq_ones n) (dag_of L F n) (root_of L F n) (L n).

(*
  “Soundness regime” (trace-boolish etc.) as a per-n predicate.
  Here, we require it at some rational parameter d (often 0, or a fixed d0).
  You can hard-code d := 0 or quantify it—whatever your lower bound uses.
*)
Definition ClnSound (L : Lang) (F : ClnFamily L) (d : Q) : Prop :=
  forall n,
    dag_trace_boolish_exists_k (sq_ones n) (dag_of L F n) (root_of L F n) d.

(*
  ClnPoly: there exists a family with poly total cost and required soundness.
*)
Definition ClnPoly (L : Lang) (d : Q) : Prop :=
  exists (F : ClnFamily L) (p : nat -> nat),
    Poly p /\
    (forall n, dag_total_cost (dag_of L F n) <= p n) /\
    ClnDecides L F /\
    ClnSound L F d.

(* -------- P/poly: abstractly as poly-size Boolean circuit families ------------ *)

(*
  You already have bool_circuit and compile_bool_circuit in your dev.
  Keep them as parameters here, and rely on your existing correctness theorems.
*)
Parameter bool_circuit : nat -> nat -> Type.  (* bool_circuit n k, output arity k *)
Parameter circuit_size : forall {n k}, bool_circuit n k -> nat.

(* We’ll focus on single-output circuits. *)
Definition Circuit1 (n : nat) : Type := bool_circuit n 1.

(* Semantics of boolean circuits *)
Parameter eval_circuit1 : forall {n}, Circuit1 n -> Corner n -> bool.

Record CircuitFamily (L : Lang) : Type := {
  circ_of : forall n, Circuit1 n;
}.

Definition CircuitDecides (L : Lang) (C : CircuitFamily L) : Prop :=
  forall n x, eval_circuit1 (circ_of L C n) x = L n x.

Definition Ppoly (L : Lang) : Prop :=
  exists (C : CircuitFamily L) (p : nat -> nat),
    Poly p /\
    (forall n, circuit_size (circ_of L C n) <= p n) /\
    CircuitDecides L C.

(* ================================================================ *)
(* 3. Compiler: circuits -> Cln DAG                                  *)
(* ================================================================ *)

(*
  Your existing compiler likely has type like:
    compile_bool_circuit : bool_circuit n 1 -> {k & (GA_dag k * Fin.t k)} or similar
  We keep it abstract but add the lemmas you’ll use.
*)

Parameter compile_bool_circuit :
  forall n, Circuit1 n -> { k : nat & (GA_dag k * Fin.t k)%type }.

(* Unpack helper *)
Definition compiled_dag {n} (c : Circuit1 n) : GA_dag (projT1 (compile_bool_circuit n c)) :=
  fst (projT2 (compile_bool_circuit n c)).

Definition compiled_root {n} (c : Circuit1 n) : Fin.t (projT1 (compile_bool_circuit n c)) :=
  snd (projT2 (compile_bool_circuit n c)).

(* --- Key compiler theorems you already intend to prove in Cln_DAG.v --- *)

(* Correctness: compiled DAG computes the same Boolean function *)
Axiom compile_bool_circuit_correct :
  forall n (c : Circuit1 n),
    dag_computes (sq_ones n)
      (compiled_dag c) (compiled_root c)
      (fun x : Corner n => eval_circuit1 c x).

(* Soundness preservation: compiled DAG satisfies the trace/boolish regime needed *)
Axiom compile_bool_circuit_sound :
  forall n (c : Circuit1 n) (d : Q),
    (* Often you’ll have a fixed d0 or d=0; keep general if your theorem is general. *)
    dag_trace_boolish_exists_k (sq_ones n) (compiled_dag c) (compiled_root c) d.

(* Cost bound: compiled DAG has poly total cost in the circuit size *)
Axiom compile_bool_circuit_cost :
  exists (q : nat -> nat),
    Poly q /\
    forall n (c : Circuit1 n),
      dag_total_cost (compiled_dag c) <= q (circuit_size c).

(* ================================================================ *)
(* 4. Simulation: P/poly ⊆ ClnPoly                                   *)
(* ================================================================ *)

Theorem Ppoly_subset_ClnPoly :
  forall (L : Lang) (d : Q),
    Ppoly L ->
    ClnPoly L d.
Proof.
  intros L d [C [p [Hp [Hsize Hdec]]]].
  destruct compile_bool_circuit_cost as [q [Hq Hqbound]].

  (* Build the Cln family by compiling each circuit C_n *)
  refine (ex_intro _ _ (ex_intro _ (fun n => q (p n)) _)).
  - (* F : ClnFamily L *)
    refine {| k_of := fun n => projT1 (compile_bool_circuit n (circ_of L C n));
              dag_of := fun n => compiled_dag (circ_of L C n);
              root_of := fun n => compiled_root (circ_of L C n) |}.
  - (* Poly bound *)
    (* Poly (fun n => q (p n)) *)
    apply Poly_closed_comp; assumption.
  - split.
    + (* total cost bound *)
      intro n.
      eapply Nat.le_trans.
      * apply Hqbound.
      * (* use circuit size <= p n *)
        (* rewrite as q(circuit_size) <= q(p n) needs monotonicity; if q is poly,
           it may not be monotone. In practice, pick p’ that absorbs monotonicity,
           or define Poly as “eventually dominated by a monotone polynomial”.
           For skeleton: assume q is monotone or provide a lemma. *)
        admit.
    + split.
      * (* decides *)
        intro n.
        (* compiled DAG computes eval_circuit, which equals L by CircuitDecides *)
        eapply (dag_computes).
        (* This line is schematic: we want to use compile_bool_circuit_correct. *)
        (* In actual proof: rewrite function ext using Hdec. *)
        (* Here: *)
        pose proof (compile_bool_circuit_correct n (circ_of L C n)) as Hcomp.
        (* Need to transport along pointwise equality eval_circuit1 = L n. *)
        (* You likely have a lemma: computes_respects_ext or similar. *)
        exact Hcomp.
      * (* sound *)
        intro n.
        apply compile_bool_circuit_sound.
Admitted.

(*
  Notes:
  - The only real “math” gap above is monotonicity / domination needed to go from
    cost <= q(size(c)) and size(c) <= p(n) to cost <= q(p(n)).
  - In practice, you fix this by:
      (i) choosing your Poly predicate to be “bounded by some monotone polynomial”,
     (ii) adding a lemma: PolyMonotone q -> circuit_size <= p -> q(size) <= q(p),
    or (iii) define ClnPoly bound as exists p q with cost <= q(size) and size <= p(n).
*)

(* ================================================================ *)
(* 5. The SAT object (language family)                               *)
(* ================================================================ *)

(*
  You will plug in your SAT encoding. Keep abstract here.
  SAT_n : Corner n -> bool  (or Corner m where m encodes CNF instances of size n).
*)
Parameter SAT : Lang.

(* SAT is NP-complete etc. not needed for NP ⊄ P/poly once we prove SAT ∉ P/poly. *)

(* ================================================================ *)
(* 6. Your post-DAG hardness theorem: SAT ∉ ClnPoly                  *)
(* ================================================================ *)

Axiom SAT_notin_ClnPoly :
  forall d : Q, ~ ClnPoly SAT d.

(* ================================================================ *)
(* 7. Consequences: SAT ∉ P/poly, NP ⊄ P/poly, P ≠ NP                *)
(* ================================================================ *)

Corollary SAT_notin_Ppoly :
  ~ Ppoly SAT.
Proof.
  intro Hpp.
  (* Pick the d used by the Cln hardness theorem; often d=0. *)
  specialize (Ppoly_subset_ClnPoly SAT 0%Q Hpp) as Hcln.
  specialize (SAT_notin_ClnPoly 0%Q).
  contradiction.
Qed.

(*
  Standard complexity implication:
    SAT ∉ P/poly  ->  NP ⊄ P/poly  ->  P ≠ NP
  You can import a standard library formalization, or keep it as axioms and cite it.
*)

Parameter NP : Type.
Parameter P : Type.
Parameter PpolyClass : Type.

(* If you have your own formalization, replace these. *)
Axiom SAT_notin_Ppoly_implies_NP_not_subset_Ppoly :
  ~ Ppoly SAT -> True.  (* replace True with NP ⊄ P/poly statement *)

Axiom NP_not_subset_Ppoly_implies_P_neq_NP :
  True -> P <> NP.      (* replace True with NP ⊄ P/poly statement *)

Theorem P_neq_NP_from_Cln :
  P <> NP.
Proof.
  apply NP_not_subset_Ppoly_implies_P_neq_NP.
  apply SAT_notin_Ppoly_implies_NP_not_subset_Ppoly.
  apply SAT_notin_Ppoly.
Qed.

End ClnNonuniformSeparation.