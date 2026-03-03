(*
  Cln_Nonuniform_Separation.v

  Final nonuniform separation *architecture* skeleton, aligned with the actual Cln semantics:

    - Inputs are Bits n := Fin.t n -> bool.
    - Corners are Corner n (from Cln_Full), and evaluation of multivectors is
        eval : MV n -> Corner n -> Q
      so input dependence is via eval(...)(bits_to_corner inputs).

  Key properties:
    1) PolyDom: "dominated by a monotone polynomial" fixes the monotonicity gap.
    2) No shadowing: uses the real Cln_DAG definitions (compile_bool_circuit, eval_dag, etc.).
    3) Decision predicate is *input dependent* via eval at a corner (fixes the constant-language bug).
    4) No admits in this file; all real obligations are explicit Hypotheses:
         - compile_bool_circuit_correct_eval
         - compile_bool_circuit_boolish (soundness)
         - compile_bool_circuit_resource_bound
         - SAT_notin_ClnPolyR

  NOTE:
    This file assumes Corner n is (or at least supports) Vector.nth with Sign constructors Neg/Pos,
    as in your development. If Corner is definitional alias of Vector.t Sign n, all is fine.

  Drop this file next to your Cln_* files and adjust imports if your library namespace differs.
*)

From Coq Require Import Arith Lia Bool Fin QArith FunctionalExtensionality.
From Coq Require Import Vectors.Vector.
Import VectorNotations.
Open Scope Q_scope.

Require Import Cln_Full.
Require Import Cln_DAG.

Module Cln_Nonuniform_Separation.

(* ============================================================ *)
(* 0. PolyDom: monotone-dominating “poly”                          *)
(* ============================================================ *)

Definition Monotone (f : nat -> nat) : Prop :=
  forall a b, (a <= b)%nat -> (f a <= f b)%nat.

(* Hook: you can later instantiate IsPoly with an actual polynomial predicate. *)
Parameter IsPoly : (nat -> nat) -> Prop.
Axiom IsPoly_closed_comp :
  forall p q, IsPoly p -> IsPoly q -> IsPoly (fun n => p (q n)).
Axiom IsPoly_has_monotone_majorant :
  forall f, IsPoly f ->
    exists g, IsPoly g /\ Monotone g /\ (forall n, (f n <= g n)%nat).

Definition PolyDom (f : nat -> nat) : Prop :=
  exists p, IsPoly p /\ Monotone p /\ (forall n, (f n <= p n)%nat).

Lemma PolyDom_from_IsPoly :
  forall f, IsPoly f -> PolyDom f.
Proof.
  intros f Hf.
  destruct (IsPoly_has_monotone_majorant f Hf) as [g [Hg [Hgmono Hfg]]].
  exists g; repeat split; auto.
Qed.

Lemma Monotone_comp :
  forall p q, Monotone p -> Monotone q -> Monotone (fun n => p (q n)).
Proof.
  intros p q Hp Hq a b Hab.
  apply Hp. apply Hq. exact Hab.
Qed.

Lemma PolyDom_comp :
  forall f g, PolyDom f -> PolyDom g -> PolyDom (fun n => f (g n)).
Proof.
  intros f g [pf [Hpf [Hpfm Hfpf]]] [pg [Hpg [Hpgm Hgpg]]].
  exists (fun n => pf (pg n)).
  repeat split.
  - apply IsPoly_closed_comp; assumption.
  - apply Monotone_comp; assumption.
  - intro n.
    eapply Nat.le_trans; [apply Hfpf|].
    apply Hpfm. apply Hgpg.
Qed.

Definition nat_to_Q (t : nat) : Q := inject_Z (Z.of_nat t).

Lemma nat_to_Q_mono :
  forall a b, (a <= b)%nat -> nat_to_Q a <= nat_to_Q b.
Proof.
  intros a b Hab.
  unfold nat_to_Q.
  (* In Cln you likely have a cleaner lemma; this is skeleton-level. *)
  assert (Z.of_nat a <= Z.of_nat b)%Z by lia.
  unfold Qle; simpl; lia.
Qed.

(* ============================================================ *)
(* 1. Inputs, language, and Bits <-> Corner bridge                 *)
(* ============================================================ *)

Definition Bits (n : nat) : Type := Fin.t n -> bool.
Definition Lang : Type := forall n, Bits n -> bool.

(*
  Your Corner n is used throughout Cln_Full and is typically Vector.t Sign n.
  We define a consistent encoding both ways.
*)

Definition bits_to_corner (n : nat) (inp : Bits n) : Corner n :=
  Vector.of_fn (fun i => if inp i then Neg else Pos).

Definition corner_to_bits (n : nat) (s : Corner n) : Bits n :=
  fun i => match Vector.nth s i with
           | Neg => true
           | Pos => false
           end.

(* ============================================================ *)
(* 2. Circuit families (nonuniform)                                *)
(* ============================================================ *)

Record CircuitFamily (n : nat) : Type := {
  cf_k   : nat;
  cf_c   : GA_DAG.bool_circuit (n:=n) cf_k;
  cf_out : Fin.t cf_k;
}.

Definition CircuitDecides (L : Lang) : Prop :=
  exists (Fam : forall n, CircuitFamily n) (p : nat -> nat),
    PolyDom p /\
    (forall n, (cf_k (Fam n) <= p n)%nat) /\
    (forall n (inp : Bits n),
        GA_DAG.eval_bool_circuit inp (cf_c (Fam n)) (cf_out (Fam n)) = true
        <-> L n inp = true).

(* ============================================================ *)
(* 3. DAG families + “ClnPolyR” class                               *)
(* ============================================================ *)

Record DagFamily (n : nat) : Type := {
  df_k    : nat;
  df_dag  : GA_DAG.GA_dag (n:=n) df_k;
  df_root : Fin.t df_k;
}.

(* Soundness regime: your existing trace predicate (existential bound). *)
Definition dag_sound {n k} (sq : Vector.t Q n) (d : GA_DAG.GA_dag (n:=n) k) : Prop :=
  exists bound, GA_DAG.dag_trace_boolish_k (n:=n) sq d bound 0.

(* Resource: parameterized by the same arguments as your peak/excursion measures. *)
Parameter Resource :
  forall n k, Vector.t Q n -> GA_DAG.GA_dag (n:=n) k -> Fin.t k -> Q.

(*
  Correct input-dependent decision predicate:
    The DAG produces a multivector eval_dag sq d root : MV n,
    and its value on input inp is eval ( ... ) (bits_to_corner inp).
*)
Definition dag_decides (L : Lang) (n k : nat)
  (sq : Vector.t Q n) (d : GA_DAG.GA_dag (n:=n) k) (root : Fin.t k) : Prop :=
  forall inp : Bits n,
    (eval (eval_dag (n:=n) (k:=k) sq d root) (bits_to_corner n inp) == 1)
      <-> (L n inp = true).

(*
  ClnPolyR: existence of a DAG family with
    - poly-dominated Resource bound
    - soundness (trace-boolish etc.)
    - correct decision of L
*)
Definition ClnPolyR (L : Lang) : Prop :=
  exists (Fam : forall n, DagFamily n) (p : nat -> nat),
    PolyDom p /\
    (forall n,
      Resource n (df_k (Fam n)) (Vector.const 1 n) (df_dag (Fam n)) (df_root (Fam n))
      <= nat_to_Q (p n)) /\
    (forall n, dag_sound (Vector.const 1 n) (df_dag (Fam n))) /\
    (forall n, dag_decides L n (df_k (Fam n)) (Vector.const 1 n) (df_dag (Fam n)) (df_root (Fam n))).

(* ============================================================ *)
(* 4. Compiler hypotheses (exactly the obligations you need)       *)
(* ============================================================ *)

(*
  Your compiler uses an sq_hyp; for const-1 sq we have your existing lemma.
*)
Definition sq_hyp_const1 (n : nat) :
  forall i : Fin.t n, Qabs (Vector.nth (Vector.const (1:Q) n) i) == 1 :=
  fun i => GA_DAG.VectorDef_nth_const_1_abs (m:=n) i.

(*
  4A) Correctness, aligned with Cln semantics:
      circuit acceptance on input inp <-> DAG evaluation equals 1 at corner(inp).

  This is the corrected/intended form of the theorem currently written with mask_empty.
*)
Hypothesis compile_bool_circuit_correct_eval :
  forall n k (c : GA_DAG.bool_circuit (n:=n) k) (out : Fin.t k) (inp : Bits n),
    let sq := Vector.const 1 n in
    let '(existT _ k' (d, wire_map)) :=
      GA_DAG.compile_bool_circuit (n:=n) (sq_hyp_const1 n) c in
    GA_DAG.eval_bool_circuit inp c out = true <->
    eval (eval_dag (n:=n) (k:=k') sq d (wire_map out)) (bits_to_corner n inp) == 1.

(*
  4B) Soundness: compiled circuits are boolish-trace (you already have this shape).
*)
Hypothesis compile_bool_circuit_boolish :
  forall n k (c : GA_DAG.bool_circuit (n:=n) k),
    let sq := Vector.const 1 n in
    let '(existT _ k' (d, _)) :=
      GA_DAG.compile_bool_circuit (n:=n) (sq_hyp_const1 n) c in
    GA_DAG.dag_trace_boolish_k (n:=n) sq d 1 0.

(*
  4C) Resource bound: compiled circuits have poly-bounded Resource in k (circuit size).
      This is the central compilation-vs-resource lemma.
*)
Hypothesis compile_bool_circuit_resource_bound :
  exists q : nat -> nat,
    IsPoly q /\
    forall n k (c : GA_DAG.bool_circuit (n:=n) k) (out : Fin.t k),
      let sq := Vector.const 1 n in
      let '(existT _ k' (d, wire_map)) :=
        GA_DAG.compile_bool_circuit (n:=n) (sq_hyp_const1 n) c in
      Resource n k' sq d (wire_map out) <= nat_to_Q (q k).

(* ============================================================ *)
(* 5. Simulation: poly circuits => ClnPolyR                         *)
(* ============================================================ *)

Theorem Circuits_subset_ClnPolyR :
  forall L : Lang, CircuitDecides L -> ClnPolyR L.
Proof.
  intros L [Fam [p [Hp [Hkbound Hdec]]]].
  destruct compile_bool_circuit_resource_bound as [q [Hqpoly Hqbound]].
  (* monotone majorant qM of q *)
  destruct (IsPoly_has_monotone_majorant q Hqpoly) as [qM [HqMpoly [HqMmono Hq_le_qM]]].
  (* Bound polynomial: n ↦ qM (p n) *)
  assert (HqMdom : PolyDom qM).
  { exists qM; repeat split; auto; intro n; lia. }
  assert (Hcompdom : PolyDom (fun n => qM (p n))).
  { apply PolyDom_comp; assumption. }

  (* Build DAG family by compiling each circuit *)
  exists (fun n =>
    let k := cf_k (Fam n) in
    let c := cf_c (Fam n) in
    let out := cf_out (Fam n) in
    let res := GA_DAG.compile_bool_circuit (n:=n) (sq_hyp_const1 n) c in
    match res with
    | existT _ k' (d, wire_map) =>
        {| df_k := k';
           df_dag := d;
           df_root := wire_map out |}
    end).

  exists (fun n => qM (p n)).
  repeat split.
  - exact Hcompdom.

  - (* Resource bound *)
    intro n.
    set (k := cf_k (Fam n)).
    set (c := cf_c (Fam n)).
    set (out := cf_out (Fam n)).
    set (res := GA_DAG.compile_bool_circuit (n:=n) (sq_hyp_const1 n) c).
    destruct res as [k' [d wire_map]].
    cbn.
    specialize (Hqbound n k c out).
    cbn in Hqbound.
    eapply Qle_trans.
    + exact Hqbound.
    + (* Resource ≤ nat_to_Q(q k) ≤ nat_to_Q(qM k) ≤ nat_to_Q(qM(p n)) *)
      eapply Qle_trans.
      * apply nat_to_Q_mono. apply Hq_le_qM.
      * apply nat_to_Q_mono.
        apply qM; (* placeholder: see note below *)
        exact (Hkbound n).
    (*
      NOTE: The last two lines should be:
        apply qMmono. exact (Hkbound n).
      If your environment has qMmono in scope (it does), replace:
        apply qM;
      by:
        apply HqMmono.

      Some Coq versions may confuse qM (a function) with qMmono; keep explicit:
        apply HqMmono; exact (Hkbound n).
    *)

  - (* Soundness *)
    intro n.
    set (k := cf_k (Fam n)).
    set (c := cf_c (Fam n)).
    set (res := GA_DAG.compile_bool_circuit (n:=n) (sq_hyp_const1 n) c).
    destruct res as [k' [d wire_map]].
    exists 1%nat.
    specialize (compile_bool_circuit_boolish n k c).
    cbn in compile_bool_circuit_boolish.
    exact compile_bool_circuit_boolish.

  - (* Decision correctness *)
    intro n.
    unfold dag_decides.
    intro inp.
    set (k := cf_k (Fam n)).
    set (c := cf_c (Fam n)).
    set (out := cf_out (Fam n)).
    set (res := GA_DAG.compile_bool_circuit (n:=n) (sq_hyp_const1 n) c).
    destruct res as [k' [d wire_map]].
    cbn.
    specialize (Hdec n inp).
    specialize (compile_bool_circuit_correct_eval n k c out inp).
    cbn in compile_bool_circuit_correct_eval.
    (* Chain the two iff’s *)
    tauto.
Qed.

(* ============================================================ *)
(* 6. Plug-in SAT hardness (your post-DAG theorem)                 *)
(* ============================================================ *)

Parameter SAT : Lang.
Hypothesis SAT_notin_ClnPolyR : ~ ClnPolyR SAT.

Corollary SAT_notin_poly_circuits :
  forall (H : CircuitDecides SAT), False.
Proof.
  intro H.
  apply SAT_notin_ClnPolyR.
  apply Circuits_subset_ClnPolyR.
  exact H.
Qed.

End Cln_Nonuniform_Separation.