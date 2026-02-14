(*
compiler: builds one GAProg node per gate

correctness theorem:

Theorem compile_correct :
  forall C ρ,
    wf_circuit C ->
    eval_prog (compile C) (encode_inputs ρ)
    =
    mv_scale (if eval_circuit C ρ then 1 else 0) mv_one.


size linear

depth linear-ish

This is the backbone “computation → representation” result.
*)

Require Import Cln_Basis.
Require Import Cln_Multivector.
Require Import Cln_GeometricProduct.

Require Import Cln_GAProg.
Require Import Cln_GAProg_Cost.
Require Import Cln_Circuit.
Require Import Cln_Circuit_Eval.

From Coq Require Import List Arith Lia Bool QArith.
Import ListNotations.
Open Scope Q_scope.
Set Implicit Arguments.

(*
  ============================================================
  Cln_CircuitToGAProg.v

  Compiler: Circuit -> GAProg producing *scalar* multivectors
  (multiples of mv_one), using MV operations (add/scale/gp).

  Encoding:
    false ↦ 0 * mv_one
    true  ↦ 1 * mv_one

  Gate identities over {0,1}:
    NOT x  = 1 - x
    AND    = x*y
    XOR    = x + y - 2xy
    OR     = x + y - xy
  ============================================================
*)

Section Compile.
  Context {n : nat}.
  Variable sq : Vector.t Q n.

  (* scalar encoding *)
  Definition bQ (b : bool) : Q := if b then 1%Q else 0%Q.
  Definition as_scalar (k : Q) : MV n := mv_scale k mv_one.

  (* You will want this lemma once mv_gp_one_l/r and bilinearity are proven. *)
  Lemma gp_scalars :
    forall k l : Q,
      mv_gp n sq (as_scalar k) (as_scalar l)
      =
      as_scalar (k * l).
  Proof.
  Admitted.

  (* We compile each gate into a GAExpr node.
     Strategy: maintain a list of defs in the same order as gates,
     so gate index = GA node index. *)

  Definition compile_gate (g : Gate) : GAExpr n :=
    match g with
    | GConst b => EScale (bQ b) 0  (* placeholder; replaced during compile with EOne/EZero nodes *)
    | GInput x => EVar x
    | GNot a   => (* 1 - a *) EAdd (*TODO*) 0 0
    | GAnd a b => EGp a b
    | GXor a b => (* a + b - 2ab *) EAdd (*TODO*) 0 0
    | GOr  a b => (* a + b - ab *) EAdd (*TODO*) 0 0
    end.

  (* A better approach: normalize with EZero/EOne directly.
     Skeleton compiler will build EZero/EOne at indices 0 and 1, then gates after. *)

  (* Build a program:
       defs = [EZero; EOne] ++ map compile_gate gates
       out  = 2 + C.out
     And adjust indices in each gate by +2. *)

  Fixpoint shift_gate (g : Gate) : Gate :=
    match g with
    | GConst b => GConst b
    | GInput x => GInput x
    | GNot a => GNot (2 + a)
    | GAnd a b => GAnd (2 + a) (2 + b)
    | GXor a b => GXor (2 + a) (2 + b)
    | GOr  a b => GOr  (2 + a) (2 + b)
    end.

  Definition compile_gate2 (g : Gate) : GAExpr n :=
    match g with
    | GConst b =>
        if b then EOne else EZero
    | GInput x =>
        EVar x
    | GNot a =>
        (* 1 - a = 1 + (-1)*a *)
        EAdd 1 ( (* node = (-1)*a *) 0 ) (* placeholder; will be expanded by helper below *)
    | GAnd a b =>
        EGp a b
    | GXor a b =>
        (* a + b - 2ab *)
        EAdd 0 0 (* placeholder *)
    | GOr a b =>
        (* a + b - ab *)
        EAdd 0 0 (* placeholder *)
    end.

  (*
    To keep the skeleton minimal, we define a small macro-expander that,
    when compiling XOR/OR/NOT, appends intermediate nodes (scale/gp/add)
    rather than trying to cram everything into one node.
  *)

  (* A compilation state: current defs list. *)
  Definition St := list (GAExpr n).

  Definition emit (e : GAExpr n) (st : St) : (nat * St) :=
    let idx := length st in
    (idx, st ++ [e]).

  (* compile a shifted gate, returning index of its output node *)
  Definition compile_gate_st (g : Gate) (st : St) : (nat * St) :=
    match g with
    | GConst b =>
        if b then (1, st) else (0, st)   (* indices of EZero/EOne assumed *)
    | GInput x =>
        emit (EVar x) st
    | GNot a =>
        (* t := (-1)*a; out := 1 + t *)
        let '(t, st1) := emit (EScale (-1)%Q a) st in
        emit (EAdd 1 t) st1
    | GAnd a b =>
        emit (EGp a b) st
    | GXor a b =>
        (* ab := a⋆b; twoab := 2*ab; s := a+b; out := s + (-twoab) *)
        let '(ab, st1) := emit (EGp a b) st in
        let '(twoab, st2) := emit (EScale 2%Q ab) st1 in
        let '(s, st3) := emit (EAdd a b) st2 in
        let '(neg, st4) := emit (EScale (-1)%Q twoab) st3 in
        emit (EAdd s neg) st4
    | GOr a b =>
        (* ab := a⋆b; s := a+b; negab := (-1)*ab; out := s + negab *)
        let '(ab, st1) := emit (EGp a b) st in
        let '(s, st2) := emit (EAdd a b) st1 in
        let '(negab, st3) := emit (EScale (-1)%Q ab) st2 in
        emit (EAdd s negab) st3
    end.

  Fixpoint compile_gates (gs : list Gate) (st : St) : (list nat * St) :=
    match gs with
    | [] => ([], st)
    | g :: tl =>
        let '(idx, st1) := compile_gate_st g st in
        let '(idxs, st2) := compile_gates tl st1 in
        (idx :: idxs, st2)
    end.

  Definition compile (C : Circuit) : GAProg n :=
    let gs2 := map shift_gate C.(gates) in
    let st0 : St := [EZero; EOne] in
    let '(idxs, stF) := compile_gates gs2 st0 in
    {| defs := stF;
       out  := (* output is last index produced for the shifted out gate *)
               (* fallback: 2 + out, but actual index depends on expansion.
                  We'll compute by nth_error from idxs. *)
               match nth_error idxs C.(out) with
               | Some k => k
               | None => 0
               end |}.

  (* ---------------------------------------------------------- *)
  (* Correctness theorem: compiled program evaluates to scalar   *)
  (* ---------------------------------------------------------- *)

  Definition env_of_rho (rho : nat -> bool) : nat -> MV n :=
    fun x => as_scalar (bQ (rho x)).

  (* Main statement: compiled output is (bQ(eval_circuit)) * mv_one *)
  Theorem compile_correct :
    forall (C : Circuit) (rho : nat -> bool),
      wf_circuit C ->
      eval_prog (n:=n) sq (env_of_rho rho) (compile C)
      =
      as_scalar (bQ (eval_circuit rho C)).
  Proof.
  Admitted.

  Theorem compile_size_linear :
    forall C,
      size_prog (compile C) <= 6 * size_c C + 2.
  Proof.
  Admitted.

  Theorem compile_depth_poly :
    forall C,
      wf_circuit C ->
      depth_prog (compile C) <= 5 * depth_c C + 10.
  Proof.
  Admitted.

End Compile.

(*
  ============================================================
  File: Cln_CircuitToGAProg.v
  ============================================================

  Compiles Boolean circuits into GAProg over MV n, using only
  scalar-in-mv_one encodings.

  Encoding:
    false ↦ 0⋅mv_one
    true  ↦ 1⋅mv_one

  Gates compiled using identities valid on {0,1} scalars:
    NOT x = 1 - x
    AND x y = x*y
    XOR x y = x + y - 2xy
    OR  x y = x + y - xy
*)

Require Import Cln_Basis.
Require Import Cln_Multivector.
Require Import Cln_GeometricProduct.
Require Import Cln_GAProg.
Require Import Cln_GAProg_Cost.

Require Import Cln_Circuit.
Require Import Cln_Circuit_Eval.

From Coq Require Import List Bool Arith Lia QArith.
Import ListNotations.
Open Scope Q_scope.
Set Implicit Arguments.

(* ============================================================ *)
(* Boolean-as-scalar-in-MV encoding                               *)
(* ============================================================ *)

Definition b2Q (b:bool) : Q := if b then 1%Q else 0%Q.

Definition bool_mv {n} (b:bool) : MV n := mv_scale (b2Q b) mv_one.

Definition input_env {n} (ρ:BEnv) : Env n :=
  fun x => bool_mv (ρ x).

(* Useful derived MV operations on scalar-encoded values *)
Definition mv_sub {n} (F G:MV n) : MV n := mv_add F (mv_scale (-1)%Q G).

(* ============================================================ *)
(* Gate compilation into GAProg nodes                             *)
(* ============================================================ *)

(* We build one GAExpr node per gate, so indices align. *)

Definition compile_gate {n} (sq:Vector.t Q n) (g:Gate) : GAExpr n :=
  match g with
  | GConst b => EScale (b2Q b) 1 (* placeholder, patched by compile() *)
  | GInput x => EVar x
  | GNot i   =>
      (* 1 - x *)
      EAdd 1 (S i)  (* placeholder, patched by compile() *)
  | GAnd i j =>
      EGp i j
  | GXor i j =>
      (* x + y - 2xy *)
      (* We will synthesize as: (x + y) + (-2)⋅(x⋆y) using extra nodes in compile() *)
      EAdd i j (* placeholder *)
  | GOr i j =>
      (* x + y - xy *)
      EAdd i j (* placeholder *)
  end.

(*
  Because GAProg nodes reference earlier nodes only, some gates (XOR/OR/NOT/CONST)
  need a small number of extra helper nodes. We therefore compile with a
  "builder" that appends the necessary helper expressions and returns the index
  of the value node for each gate.
*)

Record BuilderState (n:nat) : Type := {
  b_defs : list (GAExpr n)
}.

Definition b_len {n} (st:BuilderState n) : nat := length st.(b_defs).

Definition b_emit {n} (e:GAExpr n) (st:BuilderState n) : BuilderState n * nat :=
  let idx := b_len st in
  ({| b_defs := st.(b_defs) ++ [e] |}, idx).

(* Emit constant 0/1 as MV scalars *)
Definition emit_zero {n} (st:BuilderState n) : BuilderState n * nat :=
  b_emit (EZero (n:=n)) st.

Definition emit_one {n} (st:BuilderState n) : BuilderState n * nat :=
  b_emit (EOne (n:=n)) st.

Definition emit_const {n} (b:bool) (st:BuilderState n) : BuilderState n * nat :=
  let '(st1, i1) := emit_one st in
  if b then (st1, i1)
  else emit_zero st.

Definition emit_add {n} (i j:nat) (st:BuilderState n) : BuilderState n * nat :=
  b_emit (EAdd (n:=n) i j) st.

Definition emit_scale {n} (k:Q) (i:nat) (st:BuilderState n) : BuilderState n * nat :=
  b_emit (EScale (n:=n) k i) st.

Definition emit_gp {n} (i j:nat) (st:BuilderState n) : BuilderState n * nat :=
  b_emit (EGp (n:=n) i j) st.

Definition emit_sub {n} (i j:nat) (st:BuilderState n) : BuilderState n * nat :=
  let '(st1, mj) := emit_scale (-1)%Q j st in
  emit_add i mj st1.

(* Compile a single gate, given that all earlier gate outputs are already nodes. *)
Definition compile_gate_build {n} (sq:Vector.t Q n) (g:Gate) (st:BuilderState n) : BuilderState n * nat :=
  match g with
  | GConst b =>
      emit_const b st

  | GInput x =>
      b_emit (EVar (n:=n) x) st

  | GNot i =>
      let '(st1, one) := emit_one st in
      let '(st2, xi)  := (st1, i) in
      emit_sub one xi st2

  | GAnd i j =>
      emit_gp i j st

  | GXor i j =>
      (* x + y - 2xy *)
      let '(st1, sxy) := emit_add i j st in
      let '(st2, xy)  := emit_gp i j st1 in
      let '(st3, two_xy) := emit_scale 2%Q xy st2 in
      emit_sub sxy two_xy st3

  | GOr i j =>
      (* x + y - xy *)
      let '(st1, sxy) := emit_add i j st in
      let '(st2, xy)  := emit_gp i j st1 in
      emit_sub sxy xy st2
  end.

Fixpoint compile_gates_build {n} (sq:Vector.t Q n) (gs:list Gate) (st:BuilderState n) : BuilderState n :=
  match gs with
  | [] => st
  | g::tl =>
      let '(st1, idx) := compile_gate_build sq g st in
      (* idx is the output node for this gate; by construction it's the last emitted.
         Subsequent gates can refer to gate indices by assuming: gate i ↦ node i.
         Therefore we must ensure: for each gate we emit exactly ONE node whose index equals gate index.
         The current builder emits multiple helper nodes for some gates, so we DO NOT preserve that alignment.
         Instead, we track a map gate_index ↦ node_index in the compiler in a later refinement.

         For now: we compile into a "flat" GAProg and separately provide a gate->node index table.
      *)
      compile_gates_build sq tl st1
  end.

(* ============================================================ *)
(* A practical compiler: returns program + a table mapping gate index → node index *)
(* ============================================================ *)

Fixpoint compile_gates_tab {n} (sq:Vector.t Q n) (gs:list Gate) (st:BuilderState n)
  : BuilderState n * list nat :=
  match gs with
  | [] => (st, [])
  | g::tl =>
      let '(st1, idx) := compile_gate_build sq g st in
      let '(st2, tab) := compile_gates_tab sq tl st1 in
      (st2, idx :: tab)
  end.

Definition tab_nth (tab:list nat) (i:nat) : nat := nth i tab 0.

Definition compile {n} (sq:Vector.t Q n) (C:Circuit) : GAProg n * list nat :=
  let st0 : BuilderState n := {| b_defs := [] |} in
  let '(stF, tab_rev) := compile_gates_tab sq C.(gates) st0 in
  let tab := rev tab_rev in
  let out_idx := tab_nth tab C.(out) in
  ({| defs := stF.(b_defs); out := out_idx |}, tab).

(* ============================================================ *)
(* Semantic connection helpers                                    *)
(* ============================================================ *)

(* A key lemma you will prove once mv_gp_one_l/r + bilinear are finished:
   geometric product on scalar-encoded values agrees with scalar multiplication. *)
Lemma gp_scalars :
  forall n (sq:Vector.t Q n) (k l:Q),
    mv_gp n sq (mv_scale k mv_one) (mv_scale l mv_one)
    =
    mv_scale (k*l)%Q mv_one.
Proof. Admitted.

(* ============================================================ *)
(* Correctness theorem (statement ready; proof after gp_scalars)   *)
(* ============================================================ *)

Theorem compile_correct :
  forall n (sq:Vector.t Q n) (C:Circuit) (ρ:BEnv),
    wf_circuit C ->
    let '(p, tab) := compile sq C in
    eval_prog sq (input_env ρ) p
    =
    bool_mv (eval_circuit C ρ).
Proof. Admitted.

(* Size bound: linear in (# gates) up to constant-factor helper nodes. *)
Theorem compile_size :
  forall n (sq:Vector.t Q n) (C:Circuit),
    let '(p, tab) := compile sq C in
    size_prog p <= 6 * size_c C + 2.
Proof. Admitted.

(*
  ============================================================
  File: Cln_CircuitToGAProg.v
  ============================================================

  Compiles Boolean circuits into GAProg over MV n, using only
  scalar-in-mv_one encodings.

  Encoding:
    false ↦ 0⋅mv_one
    true  ↦ 1⋅mv_one

  Gates compiled using identities valid on {0,1} scalars:
    NOT x = 1 - x
    AND x y = x*y
    XOR x y = x + y - 2xy
    OR  x y = x + y - xy

  IMPORTANT:
    Because some gates need helper nodes, gate indices do NOT equal node indices.
    The compiler maintains a table tab : list nat mapping gate_index ↦ node_index.
*)

Require Import Cln_Basis.
Require Import Cln_Multivector.
Require Import Cln_GeometricProduct.
Require Import Cln_GAProg.
Require Import Cln_GAProg_Cost.

Require Import Cln_Circuit.
Require Import Cln_Circuit_Eval.

From Coq Require Import List Bool Arith Lia QArith.
Import ListNotations.
Open Scope Q_scope.
Set Implicit Arguments.

(* ============================================================ *)
(* Boolean-as-scalar-in-MV encoding                               *)
(* ============================================================ *)

Definition b2Q (b:bool) : Q := if b then 1%Q else 0%Q.

Definition bool_mv {n} (b:bool) : MV n := mv_scale (b2Q b) mv_one.

Definition input_env {n} (ρ:BEnv) : Env n :=
  fun x => bool_mv (ρ x).

(* derived subtraction: F - G *)
Definition mv_sub {n} (F G:MV n) : MV n := mv_add F (mv_scale (-1)%Q G).

(* ============================================================ *)
(* A small builder for GAProg                                     *)
(* ============================================================ *)

Record BuilderState (n:nat) : Type := { b_defs : list (GAExpr n) }.

Definition b_len {n} (st:BuilderState n) : nat := length st.(b_defs).

Definition b_emit {n} (e:GAExpr n) (st:BuilderState n) : BuilderState n * nat :=
  let idx := b_len st in
  ({| b_defs := st.(b_defs) ++ [e] |}, idx).

Definition emit_zero {n} (st:BuilderState n) : BuilderState n * nat :=
  b_emit (EZero (n:=n)) st.

Definition emit_one {n} (st:BuilderState n) : BuilderState n * nat :=
  b_emit (EOne (n:=n)) st.

Definition emit_const {n} (b:bool) (st:BuilderState n) : BuilderState n * nat :=
  let '(st1, one) := emit_one st in
  if b then (st1, one) else emit_zero st.

Definition emit_add {n} (i j:nat) (st:BuilderState n) : BuilderState n * nat :=
  b_emit (EAdd (n:=n) i j) st.

Definition emit_scale {n} (k:Q) (i:nat) (st:BuilderState n) : BuilderState n * nat :=
  b_emit (EScale (n:=n) k i) st.

Definition emit_gp {n} (i j:nat) (st:BuilderState n) : BuilderState n * nat :=
  b_emit (EGp (n:=n) i j) st.

Definition emit_sub {n} (i j:nat) (st:BuilderState n) : BuilderState n * nat :=
  let '(st1, mj) := emit_scale (-1)%Q j st in
  emit_add i mj st1.

(* Gate-index table helpers *)
Definition tab_nth (tab:list nat) (i:nat) : nat := nth i tab 0.

(* ============================================================ *)
(* Compile one gate, given a mapping for earlier gates             *)
(* ============================================================ *)

Definition compile_gate_build {n} (sq:Vector.t Q n) (tab_prev:list nat) (g:Gate)
  (st:BuilderState n) : BuilderState n * nat :=
  match g with
  | GConst b =>
      emit_const b st

  | GInput x =>
      b_emit (EVar (n:=n) x) st

  | GNot i =>
      let xi := tab_nth tab_prev i in
      let '(st1, one) := emit_one st in
      emit_sub one xi st1

  | GAnd i j =>
      let xi := tab_nth tab_prev i in
      let yj := tab_nth tab_prev j in
      emit_gp xi yj st

  | GXor i j =>
      (* x + y - 2xy *)
      let xi := tab_nth tab_prev i in
      let yj := tab_nth tab_prev j in
      let '(st1, sxy) := emit_add xi yj st in
      let '(st2, xy)  := emit_gp xi yj st1 in
      let '(st3, two_xy) := emit_scale 2%Q xy st2 in
      emit_sub sxy two_xy st3

  | GOr i j =>
      (* x + y - xy *)
      let xi := tab_nth tab_prev i in
      let yj := tab_nth tab_prev j in
      let '(st1, sxy) := emit_add xi yj st in
      let '(st2, xy)  := emit_gp xi yj st1 in
      emit_sub sxy xy st2
  end.

Fixpoint compile_gates_tab {n} (sq:Vector.t Q n) (gs:list Gate)
  (st:BuilderState n) (tab:list nat)
  : BuilderState n * list nat :=
  match gs with
  | [] => (st, tab)
  | g::tl =>
      let '(st1, idx) := compile_gate_build sq tab g st in
      compile_gates_tab sq tl st1 (tab ++ [idx])
  end.

Definition compile {n} (sq:Vector.t Q n) (C:Circuit) : GAProg n * list nat :=
  let st0 : BuilderState n := {| b_defs := [] |} in
  let '(stF, tab) := compile_gates_tab sq C.(gates) st0 [] in
  let out_idx := tab_nth tab C.(out) in
  ({| defs := stF.(b_defs); out := out_idx |}, tab).

(* ============================================================ *)
(* Key scalar lemma (prove after mv_gp_one_l/r + bilinear)         *)
(* ============================================================ *)

Lemma gp_scalars :
  forall n (sq:Vector.t Q n) (k l:Q),
    mv_gp n sq (mv_scale k mv_one) (mv_scale l mv_one)
    =
    mv_scale (k*l)%Q mv_one.
Proof. Admitted.

(* ============================================================ *)
(* Correctness + size theorems (ready for proofs)                  *)
(* ============================================================ *)

Theorem compile_correct :
  forall n (sq:Vector.t Q n) (C:Circuit) (ρ:BEnv),
    wf_circuit C ->
    let '(p, tab) := compile sq C in
    eval_prog sq (input_env ρ) p
    =
    bool_mv (eval_circuit C ρ).
Proof. Admitted.

Theorem compile_size :
  forall n (sq:Vector.t Q n) (C:Circuit),
    let '(p, tab) := compile sq C in
    size_prog p <= 6 * size_c C + 2.
Proof. Admitted.