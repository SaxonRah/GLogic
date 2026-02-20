From Stdlib Require Import List Arith Lia.
Import ListNotations.

From TraceGeometry Require Import TraceGeometryCore.

Declare Scope tg_scope.
Delimit Scope tg_scope with tg.
Open Scope tg_scope.

Module Type TRACE_GEOMETRY_STACK_MACHINE (TG : TRACE_GEOMETRY_CORE).
  Import TG.
  Open Scope tg_scope.

  (*** Machine state ***)
  Definition Stack := list A.

  (* Observe the "current value" of a state (top of stack, or zero). *)
  Definition obs (st : Stack) : A :=
    match st with
    | [] => zero
    | x :: _ => x
    end.

  (*** Instruction set ***)
  Inductive Instr : Type :=
  | IPush  (c : A)
  | IAdd
  | IConv
  | IGp
  | IGauge (g : Gauge)
  | IDup
  | ISwap
  | IPop
  .

  (*** Small-step execution on stacks ***)
  Inductive sstep : Stack -> Stack -> Prop :=
  | step_push : forall st c,
      sstep st (c :: st)

  | step_add : forall st x y,
      sstep (x :: y :: st) ((x + y)%tg :: st)

  | step_conv : forall st x y,
      sstep (x :: y :: st) ((x ⊙ y)%tg :: st)

  | step_gp : forall st x y,
      sstep (x :: y :: st) ((x ⋆ y)%tg :: st)

  | step_gauge : forall st g x,
      sstep (x :: st) (gauge g x :: st)

  | step_dup : forall st x,
      sstep (x :: st) (x :: x :: st)

  | step_swap : forall st x y,
      sstep (x :: y :: st) (y :: x :: st)

  | step_pop : forall st x,
      sstep (x :: st) st
  .

  (*** Deterministic exec function (optional) ***)
  Definition exec1 (i : Instr) (st : Stack) : option Stack :=
    match i, st with
    | IPush c, st => Some (c :: st)
    | IAdd, x :: y :: st => Some ((x + y)%tg :: st)
    | IConv, x :: y :: st => Some ((x ⊙ y)%tg :: st)
    | IGp, x :: y :: st => Some ((x ⋆ y)%tg :: st)
    | IGauge g, x :: st => Some (gauge g x :: st)
    | IDup, x :: st => Some (x :: x :: st)
    | ISwap, x :: y :: st => Some (y :: x :: st)
    | IPop, _ :: st => Some st
    | _, _ => None
    end.

  (*** Program semantics as traces of stacks ***)
  Definition Prog := list Instr.

  Fixpoint run_from (st : Stack) (p : Prog) : option (list Stack) :=
    match p with
    | [] => Some [st]
    | i :: tl =>
        match exec1 i st, run_from (match exec1 i st with Some st' => st' | None => st end) tl with
        | Some st', Some tr => Some (st :: tr)
        | _, _ => None
        end
    end.

  (* A relational “runs” is usually nicer than option-based exec, but both are fine. *)
  Inductive runs : Stack -> Prog -> list Stack -> Prop :=
  | runs_nil : forall st, runs st [] [st]
  | runs_cons :
      forall st i st' tl tr,
        sstep st st' ->
        runs st' tl tr ->
        runs st (i :: tl) (st :: tr).

  Definition cost (p : Prog) : nat := length p.

  Definition output (st0 : Stack) (p : Prog) : A :=
    (* output = obs of last state, defaulting safely *)
    match (fix last_stack (st : Stack) (p : Prog) : Stack :=
             match p with
             | [] => st
             | i :: tl =>
                 match exec1 i st with
                 | Some st' => last_stack st' tl
                 | None => st
                 end
             end) st0 p with
    | stf => obs stf
    end.

  (*** Bridge to TG.step / wf_trace on A by observing traces ***)

  (* You get to decide how machine steps relate to TG.step on observed values.
     This is the key axiom that says: the machine’s operational step respects
     the axiomatic “step” relation of the geometry. *)
  Axiom obs_step_sound :
    forall st st', sstep st st' -> step (obs st) (obs st').

End TRACE_GEOMETRY_STACK_MACHINE.
