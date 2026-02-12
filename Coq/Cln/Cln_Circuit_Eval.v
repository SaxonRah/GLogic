(*
Standard boolean semantics.
*)

(*
The compiler that matters: Circuit -> GAProg using your embedding

You have two routes:

Route D1 (fastest): compile boolean circuits to Q-valued multivectors using your existing Boolean embedding
    This uses your already-proved Embed : (Corner n -> bool) -> MV n-style machinery (or something equivalent) if it’s stable for general n.
    But for circuits, we want uniform compilation, so usually we compile gate-by-gate, not “embed the whole function at once.”

Route D2 (better & uniform): represent a boolean value as a scalar multivector

Pick an encoding:
    false ↦ 0 * mv_one
    true ↦ 1 * mv_one

Then compile gate semantics into multivector algebra operations that preserve scalarness.

For example:
    NOT: ¬x = 1 - x (in {0,1})
    AND: x ∧ y = x*y (works in {0,1})
    XOR: x ⊕ y = x + y - 2xy

All of these are scalar arithmetic, so they sit inside the mv_one subalgebra.

That means your GAProg only needs:
    EOne, EZero
    EScale, EAdd
    and a scalar multiplication operator. You can use EGp because on scalars mv_gp agrees with scalar multiplication if you prove:
    mv_gp n sq (k⋅mv_one) (l⋅mv_one) = (k*l)⋅mv_one

So you’ll add a lemma:

Lemma gp_scalars :
  forall n sq k l,
    mv_gp n sq (mv_scale k mv_one) (mv_scale l mv_one)
    = mv_scale (k*l) mv_one.


This uses mv_gp_one_l/r + bilinearity.

*)

Require Import Cln_Circuit.

From Coq Require Import List Arith Lia Bool.
Import ListNotations.
Set Implicit Arguments.

(*
  ============================================================
  Cln_Circuit_Eval.v

  Boolean semantics for Circuit (with sharing).
  ============================================================
*)

Section Eval.
  Variable rho : nat -> bool.  (* external inputs *)

  Fixpoint eval_gate_list (fuel : nat) (gs : list Gate) (i : nat) : bool :=
    match fuel with
    | 0 => false
    | S fuel' =>
        match nth_error gs i with
        | None => false
        | Some g =>
            match g with
            | GConst b => b
            | GInput x => rho x
            | GNot a => negb (eval_gate_list fuel' gs a)
            | GAnd a b => andb (eval_gate_list fuel' gs a) (eval_gate_list fuel' gs b)
            | GXor a b => xorb (eval_gate_list fuel' gs a) (eval_gate_list fuel' gs b)
            | GOr  a b => orb  (eval_gate_list fuel' gs a) (eval_gate_list fuel' gs b)
            end
        end
    end.

  Definition eval_circuit (C : Circuit) : bool :=
    eval_gate_list (S (length C.(gates))) C.(gates) C.(out).

End Eval.

(* Lemmas (skeleton stubs) *)
Lemma eval_circuit_ext :
  forall C rho1 rho2,
    (forall x, rho1 x = rho2 x) ->
    eval_circuit rho1 C = eval_circuit rho2 C.
Proof.
Admitted.


(*
  ============================================================
  File: Cln_Circuit_Eval.v
  ============================================================

  Boolean semantics for Circuit.
*)

Require Import Cln_Circuit.

From Coq Require Import List Bool Arith Lia.
Import ListNotations.
Set Implicit Arguments.

Definition BEnv : Type := nat -> bool.

Definition eval_gate (ρ:BEnv) (vals:list bool) (g:Gate) : bool :=
  match g with
  | GConst b => b
  | GInput x => ρ x
  | GNot i   => negb (nth i vals false)
  | GAnd i j => andb (nth i vals false) (nth j vals false)
  | GXor i j => xorb (nth i vals false) (nth j vals false)
  | GOr  i j => orb  (nth i vals false) (nth j vals false)
  end.

Fixpoint eval_gates_fwd (ρ:BEnv) (gs:list Gate) (acc:list bool) : list bool :=
  match gs with
  | [] => acc
  | g::tl =>
      let v := eval_gate ρ acc g in
      eval_gates_fwd ρ tl (acc ++ [v])
  end.

Definition eval_circuit (C:Circuit) (ρ:BEnv) : bool :=
  let vals := eval_gates_fwd ρ C.(gates) [] in
  nth C.(out) vals false.

Lemma eval_gates_fwd_length :
  forall ρ gs acc,
    length (eval_gates_fwd ρ gs acc) = length acc + length gs.
Proof.
  induction gs; intros; simpl; try lia.
  rewrite IHgs.
  rewrite app_length. simpl. lia.
Qed.