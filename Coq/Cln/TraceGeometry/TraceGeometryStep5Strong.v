Unset Universe Polymorphism.
Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

From Stdlib Require Import List Arith Lia.
Import ListNotations.

From TraceGeometry Require Import TraceGeometryCore.
From TraceGeometry Require Import TraceGeometryExplosion.
From TraceGeometry Require Import TraceGeometryMachine.

Declare Scope tg_scope.
Delimit Scope tg_scope with tg.
Open Scope tg_scope.

(************************************************************)
(* Step 5 (strong): count only “effective gp-mix” events     *)
(*                                                          *)
(* Roadmap ref: TODO_TraceGeometry.md §5                     *)
(*                                                          *)
(* Strongest useful variant for LB proofs:                   *)
(*   - track an accumulator bound M on the *current top*.    *)
(*   - count an IGp as “effective” only when:                *)
(*        top has split ≥ M   AND   second has split ≥ b      *)
(*     so the accumulator can be proven to grow by ×b.        *)
(*                                                          *)
(* This avoids “count every gp” (too weak) and avoids         *)
(* “count gp where both are ≥b” (can give only b^(2^t)-style   *)
(* guarantees without extra discipline).                      *)
(************************************************************)

Module TraceGeometryStep5Strong
  (F  : TRACE_GEOMETRY_FULL)
  (SM : TRACE_GEOMETRY_STACK_MACHINE(F.TG)).

  Module TG := F.TG.
  Module EX := F.EX.
  Module GR := F.GR.
  Module SP := GR.S.

  Import TG EX GR SP.
  Open Scope tg_scope.

  Definition Stack := SM.Stack.
  Definition Instr := SM.Instr.
  Definition Prog  := SM.Prog.

  (************************************************************)
  (* Split-richness predicate                                 *)
  (************************************************************)

  Definition HasSplitAtLeast (x : A) (m : nat) : Prop :=
    exists S : Split x,
      (length (split_parts S) >= m)%nat.

  Definition Rich (b : nat) (x : A) : Prop := HasSplitAtLeast x b.

  (************************************************************)
  (* Instruction-indexed stepping (ties Instr to stack step). *)
  (************************************************************)

  Inductive istep : Instr -> Stack -> Stack -> Prop :=
  | istep_push : forall st c,
      istep (SM.IPush c) st (c :: st)
  | istep_add : forall st x y,
      istep SM.IAdd (x :: y :: st) ((x + y)%tg :: st)
  | istep_conv : forall st x y,
      istep SM.IConv (x :: y :: st) ((x ⊙ y)%tg :: st)
  | istep_gp : forall st x y,
      istep SM.IGp (x :: y :: st) ((x ⋆ y)%tg :: st)
  | istep_gauge : forall st g x,
      istep (SM.IGauge g) (x :: st) (gauge g x :: st)
  | istep_dup : forall st x,
      istep SM.IDup (x :: st) (x :: x :: st)
  | istep_swap : forall st x y,
      istep SM.ISwap (x :: y :: st) (y :: x :: st)
  | istep_pop : forall st x,
      istep SM.IPop (x :: st) st.

  Inductive runs_i : Stack -> Prog -> list Stack -> Prop :=
  | runs_i_nil : forall st, runs_i st [] [st]
  | runs_i_cons : forall st i st' tl tr,
      istep i st st' ->
      runs_i st' tl tr ->
      runs_i st (i :: tl) (st :: tr).

  (************************************************************)
  (* One-step effective gp lemma: accumulator × b growth.     *)
  (************************************************************)

  Lemma effective_gp_step_multiplies :
    forall (b M : nat) (x y : A) (SX : Split x) (SY : Split y),
      (length (split_parts SX) >= M)%nat ->
      (length (split_parts SY) >= b)%nat ->
      HasSplitAtLeast (x ⋆ y)%tg (M * b).
  Proof.
    intros b M x y SX SY Hx Hy.
    destruct (split_growth_under_gp (x:=x) (y:=y) SX SY) as [SZ Hlen].
    exists SZ.
    nia.
  Qed.

  (************************************************************)
  (* Effective-run relation: tracks (t, M) along the trace.   *)
  (************************************************************)

  Inductive runs_eff (b : nat) :
      nat (* M_in *) -> Stack -> Prog ->
      nat (* t_effective *) -> nat (* M_out *) ->
      list Stack -> Prop :=
  | runs_eff_nil : forall M st,
      runs_eff b M st [] 0%nat M [st]

  | runs_eff_cons_other :
      forall M st i st' tl t M' tr,
        i <> SM.IGp ->
        istep i st st' ->
        runs_eff b M st' tl t M' tr ->
        runs_eff b M st (i :: tl) t M' (st :: tr)

  | runs_eff_cons_gp_ineff :
      forall M st st' tl t M' tr,
        istep SM.IGp st st' ->
        runs_eff b M st' tl t M' tr ->
        runs_eff b M st (SM.IGp :: tl) t M' (st :: tr)

  | runs_eff_cons_gp_eff :
      forall M st st' x y tail tl t M' tr,
        st = x :: y :: tail ->
        st' = (x ⋆ y)%tg :: tail ->
        (* effectiveness witness: top has ≥M, second has ≥b *)
        (exists SX : Split x,
           (length (split_parts SX) >= M)%nat) ->
        (exists SY : Split y,
           (length (split_parts SY) >= b)%nat) ->
        runs_eff b (M * b) st' tl t M' tr ->
        runs_eff b M st (SM.IGp :: tl) (S t) M' (st :: tr).

  (************************************************************)
  (* The extra hypothesis you typically prove in Step 3/6:    *)
  (* non-effective steps cannot reduce the accumulator split   *)
  (* lower bound on the observed value.                        *)
  (*                                                          *)
  (* If you have stronger facts (e.g. Parts is monotone under  *)
  (* all instructions except gp), you can instantiate this     *)
  (* immediately.                                               *)
  (************************************************************)

  Parameter istep_preserves_obs_split_lb :
    forall (M : nat) (i : Instr) (st st' : Stack),
      i <> SM.IGp ->
      istep i st st' ->
      HasSplitAtLeast (SM.obs st) M ->
      HasSplitAtLeast (SM.obs st') M.

  Parameter istep_gp_ineff_preserves_obs_split_lb :
    forall (b M : nat) (st st' : Stack),
      istep SM.IGp st st' ->
      (* “ineffective” case: you may prove this from your invariant, or
         (later) rework runs_eff to require an explicit reason it is ineffect. *)
      HasSplitAtLeast (SM.obs st) M ->
      HasSplitAtLeast (SM.obs st') M.

  (************************************************************)
  (* Iteration theorem: M grows by ×b exactly t times.         *)
  (************************************************************)

  Fixpoint pow (b n : nat) : nat :=
    match n with
    | 0 => 1%nat
    | S k => (b * pow b k)%nat
    end.

    Lemma last_cons_default_irrel :
      forall (A : Type) (x : A) (xs : list A) (d1 d2 : A),
        List.last (x :: xs) d1 = List.last (x :: xs) d2.
    Proof.
      intros A x xs.
      induction xs as [|y ys IH]; intros d1 d2; simpl.
      - reflexivity.
      - (* now goal is about last (y :: ys) d1 vs d2 *)
        destruct ys as [|z zs]; simpl.
        + reflexivity.
        + (* last (y :: z :: zs) d1 = last (y :: z :: zs) d2 *)
          (* reduce to IH instantiated with x:=y and xs:=z::zs *)
          apply (IH d1 d2).
    Qed.
    
    Lemma last_nonempty_default_irrel :
      forall (A : Type) (l : list A) (d1 d2 : A),
        l <> [] -> List.last l d1 = List.last l d2.
    Proof.
      intros A l.
      destruct l as [|x xs]; intros d1 d2 H; [contradiction|].
      now apply last_cons_default_irrel.
    Qed.

  Lemma last_of_nonempty :
    forall (A : Type) (x : A) (xs : list A) (d : A),
      List.last (x :: xs) d = List.last (x :: xs) x.
  Proof.
    intros A x xs d.
    apply last_cons_default_irrel.
  Qed.
  
  Lemma last_match_form :
    forall (A : Type) (tr : list A) (d : A),
      List.last tr d =
      match tr with
      | [] => d
      | _ :: _ => List.last tr d
      end.
  Proof.
    intros A tr d; destruct tr; simpl; reflexivity.
  Qed.

  Lemma runs_eff_trace_nonempty :
    forall b M st p t M' tr,
      runs_eff b M st p t M' tr ->
      tr <> [].
  Proof.
    intros b M st p t M' tr Hr.
    induction Hr.
    - (* nil/base case *)
      (* this should close by computation if base trace is e.g. [st] *)
      discriminate.
    - (* other step *)
      (* this should close if the constructor makes the trace a cons, e.g. st :: tr *)
      discriminate.
    - (* gp ineff step *)
      discriminate.
    - (* gp eff step *)
      discriminate.
  Qed.

  Theorem runs_eff_obs_explosion :
    forall b M st p t M' tr,
      runs_eff b M st p t M' tr ->
      HasSplitAtLeast (SM.obs st) M ->
      HasSplitAtLeast (SM.obs (List.last tr st)) (M * pow b t).
  Proof.
    intros b M st p t M' tr Hr Hstart.
    induction Hr; simpl in *.
    -
      simpl.
      rewrite Nat.mul_1_r.
      exact Hstart.
    -
      assert (Hst' : HasSplitAtLeast (SM.obs st') M).
      { eapply istep_preserves_obs_split_lb; eauto. }
      assert (Hne : tr <> []).
      { eapply runs_eff_trace_nonempty; eauto. }
      rewrite <- (@last_match_form Stack tr st).
      rewrite (@last_nonempty_default_irrel Stack tr st st' Hne).
      exact (IHHr Hst').
    -
      assert (Hst' : HasSplitAtLeast (SM.obs st') M).
      { eapply istep_gp_ineff_preserves_obs_split_lb; eauto. }
      assert (Hne : tr <> []).
      { eapply runs_eff_trace_nonempty; eauto. }
      rewrite <- (@last_match_form Stack tr st).
      rewrite (@last_nonempty_default_irrel Stack tr st st' Hne).
      exact (IHHr Hst').
    -
      assert (Hst' : HasSplitAtLeast (SM.obs (x ⋆ y :: tail)) (M * b)).
      {
        simpl.
        destruct H1 as [SX HSX].
        destruct H2 as [SY HSY].
        eapply effective_gp_step_multiplies; eauto.
      }
      subst st'.
      pose proof (IHHr Hst') as IHlast.
      assert (Hne : tr <> []).
      { eapply runs_eff_trace_nonempty; eauto. }
      subst st.
      rewrite <- (@last_match_form Stack tr (x :: y :: tail)).
      rewrite (@last_nonempty_default_irrel Stack tr (x :: y :: tail) (x ⋆ y :: tail) Hne).
      replace (M * (b * pow b t))%nat with ((M * b) * pow b t)%nat by nia.
      exact IHlast.
  Qed.

  (************************************************************)
  (* Meta: effective count bounded by length p (hence cost).   *)
  (************************************************************)

  Lemma runs_eff_t_le_cost :
    forall b M st p t M' tr,
      runs_eff b M st p t M' tr ->
      (t <= length p)%nat.
  Proof.
    intros b M st p t M' tr Hr.
    induction Hr; simpl; try lia.
  Qed.

End TraceGeometryStep5Strong.
