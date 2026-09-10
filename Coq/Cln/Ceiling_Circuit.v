(* ================================================================= *)
(*  Ceiling_Circuit.v                                                *)
(*                                                                   *)
(*  The circuit-size ceiling for submultiplicative measures.         *)
(*                                                                   *)
(*  Circuits are straight-line programs: wires 0..w-1 are inputs,    *)
(*  and each gate reads earlier wires by index (fan-out allowed).    *)
(*  Same measure axioms as Ceiling_Formula.v:                        *)
(*                                                                   *)
(*      0 <= mu v,   mu(input) <= C,   mu(u a) <= mu a,              *)
(*      mu(o a b) <= mu a * mu b.                                    *)
(*                                                                   *)
(*  Fan-out is what changes the picture.  In a formula, exponents    *)
(*  add up over disjoint subtrees; in a circuit a wire can be used   *)
(*  twice, so the exponent can double at every gate.                 *)
(*                                                                   *)
(*  Main results.                                                    *)
(*                                                                   *)
(*    circuit_bound     every wire of an s-gate circuit has          *)
(*                      mu <= C ^ (2^s)                              *)
(*    certifies_sound   if C^(2^k) < mu f, every circuit computing   *)
(*                      f has more than k gates                      *)
(*    circuit_ceiling   if mu <= C^K on the whole value space, every *)
(*                      certified bound satisfies 2^k < K,           *)
(*    circuit_ceiling_log   hence k <= log2 K                        *)
(*    circuit_tight     repeated squaring attains C^(2^s) exactly    *)
(*                      in a model of the axioms, so no argument     *)
(*                      from these axioms alone can do better        *)
(*                                                                   *)
(*  Instance.  Fourier l1 on n = 2m variables: C = 2, K = m = n/2.   *)
(*  The l1 method certifies at most log2(n/2) gates for circuits.    *)
(*                                                                   *)
(*  Dependencies: Coq standard library only.  No Axioms, no Admits.  *)
(* ================================================================= *)

From Coq Require Import QArith Qabs Lia List.
Import ListNotations.
Open Scope Q_scope.

(* ----------------------------------------------------------------- *)
(* 1. Natural-number powers in Q  (same as Ceiling_Formula.v, kept   *)
(*    here so this file stands alone)                                *)
(* ----------------------------------------------------------------- *)

Fixpoint qpow (c : Q) (k : nat) : Q :=
  match k with
  | O => 1
  | S k' => c * qpow c k'
  end.

Lemma qpow_1 : forall c, qpow c 1 == c.
Proof. intro c. simpl. ring. Qed.

Lemma qpow_add : forall c a b, qpow c (a + b) == qpow c a * qpow c b.
Proof.
  intros c a b. induction a as [|a IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

Lemma qpow_nonneg : forall c k, 0 <= c -> 0 <= qpow c k.
Proof.
  intros c k Hc. induction k as [|k IH]; simpl.
  - discriminate.
  - apply Qmult_le_0_compat; assumption.
Qed.

Lemma qpow_step_le : forall c k, 1 <= c -> qpow c k <= qpow c (S k).
Proof.
  intros c k Hc. simpl.
  pose proof (qpow_nonneg c k (Qle_trans _ _ _ (Qle_bool_imp_le 0 1 eq_refl) Hc)) as Hp.
  setoid_replace (qpow c k) with (1 * qpow c k) at 1 by ring.
  apply Qmult_le_compat_r; assumption.
Qed.

Lemma qpow_mono : forall c a b, 1 <= c -> (a <= b)%nat -> qpow c a <= qpow c b.
Proof.
  intros c a b Hc Hab. induction Hab.
  - apply Qle_refl.
  - eapply Qle_trans; [exact IHHab | apply qpow_step_le; exact Hc].
Qed.

Lemma qpow_lt_exponent :
  forall c a b, 1 <= c -> qpow c a < qpow c b -> (a < b)%nat.
Proof.
  intros c a b Hc Hlt.
  destruct (Nat.lt_ge_cases a b) as [H|H]; [exact H|].
  exfalso. apply (Qlt_not_le _ _ Hlt). apply qpow_mono; assumption.
Qed.

(* ----------------------------------------------------------------- *)
(* 2. Circuits as straight-line programs                             *)
(* ----------------------------------------------------------------- *)

Section Gates.
  Variable UOp BOp : Type.

  Inductive gate : Type :=
  | GU : UOp -> nat -> gate            (* unary gate reading wire i        *)
  | GB : BOp -> nat -> nat -> gate.    (* binary gate reading wires i, j   *)

  Definition circuit := list gate.

  (* A gate is well-formed at width w if it only reads existing wires. *)
  Definition gate_ok (w : nat) (g : gate) : Prop :=
    match g with
    | GU _ i   => (i < w)%nat
    | GB _ i j => (i < w)%nat /\ (j < w)%nat
    end.

  Fixpoint wf (w : nat) (c : circuit) : Prop :=
    match c with
    | []      => True
    | g :: c' => gate_ok w g /\ wf (S w) c'
    end.

  (* Exponent bookkeeping: the bound on log_C mu, propagated through
     the gates exactly as submultiplicativity dictates. *)
  Definition gate_exp (es : list nat) (g : gate) : nat :=
    match g with
    | GU _ i   => nth i es 0%nat
    | GB _ i j => (nth i es 0 + nth j es 0)%nat
    end.

  Fixpoint run_e (es : list nat) (c : circuit) : list nat :=
    match c with
    | []      => es
    | g :: c' => run_e (es ++ [gate_exp es g]) c'
    end.

  Lemma run_e_app : forall c1 c2 es, run_e es (c1 ++ c2) = run_e (run_e es c1) c2.
  Proof. induction c1; intros; simpl; auto. Qed.

  Lemma run_e_length : forall c es, length (run_e es c) = (length es + length c)%nat.
  Proof.
    induction c as [|g c IH]; intro es; simpl.
    - lia.
    - rewrite IH, length_app. simpl. lia.
  Qed.
End Gates.

Arguments GU {UOp BOp} _ _.
Arguments GB {UOp BOp} _ _ _.
Arguments gate_ok {UOp BOp} _ _.
Arguments wf {UOp BOp} _ _.
Arguments gate_exp {UOp BOp} _ _.
Arguments run_e {UOp BOp} _ _.

(* ----------------------------------------------------------------- *)
(* 3. The doubling lemma (pure combinatorics, no measure yet)        *)
(*                                                                   *)
(*    If all starting exponents are <= B, then after s gates all     *)
(*    exponents are <= B * 2^s.  This is where fan-out bites.        *)
(* ----------------------------------------------------------------- *)

Lemma nth_le_of_Forall :
  forall (l : list nat) B i, Forall (fun e => (e <= B)%nat) l ->
  (nth i l 0 <= B)%nat.
Proof.
  intros l B i H.
  destruct (Nat.lt_ge_cases i (length l)) as [Hi|Hi].
  - rewrite Forall_forall in H. apply H. apply nth_In. exact Hi.
  - rewrite nth_overflow by exact Hi. lia.
Qed.

Lemma Forall_le_weaken :
  forall (l : list nat) B B', (B <= B')%nat ->
  Forall (fun e => (e <= B)%nat) l -> Forall (fun e => (e <= B')%nat) l.
Proof.
  intros l B B' HB H. eapply Forall_impl; [| exact H]. intros e He; simpl in *; lia.
Qed.

Theorem doubling :
  forall UOp BOp (c : circuit UOp BOp) es B,
    Forall (fun e => (e <= B)%nat) es ->
    Forall (fun e => (e <= B * 2 ^ length c)%nat) (run_e es c).
Proof.
  intros UOp BOp c. induction c as [|g c IH]; intros es B HB; simpl.
  - eapply Forall_le_weaken; [| exact HB]. lia.
  - (* after one gate every exponent is <= 2B; then apply IH with 2B *)
    eapply Forall_impl; [| apply (IH _ (2 * B)%nat)].
    { intros e He. simpl in He. lia. }
    apply Forall_app. split.
    + eapply Forall_le_weaken; [| exact HB]. lia.
    + constructor; [| constructor].
      destruct g as [u i | o i j]; simpl.
      * pose proof (nth_le_of_Forall es B i HB). lia.
      * pose proof (nth_le_of_Forall es B i HB).
        pose proof (nth_le_of_Forall es B j HB). lia.
Qed.

(* ----------------------------------------------------------------- *)
(* 4. The measure, and the propagation bound on circuits             *)
(* ----------------------------------------------------------------- *)

Section CircuitCeiling.

  Variable V   : Type.
  Variable UOp : Type.
  Variable BOp : Type.

  Variable uop_val : UOp -> V -> V.
  Variable bop_val : BOp -> V -> V -> V.

  (* Default value for nth.  Never read on well-formed circuits. *)
  Variable v0 : V.

  Variable mu : V -> Q.
  Variable C  : Q.

  Hypothesis C_ge_1     : 1 <= C.
  Hypothesis mu_nonneg  : forall v, 0 <= mu v.
  Hypothesis mu_unary   : forall u a, mu (uop_val u a) <= mu a.
  Hypothesis mu_submult : forall o a b, mu (bop_val o a b) <= mu a * mu b.

  Definition gate_val (vals : list V) (g : gate UOp BOp) : V :=
    match g with
    | GU u i   => uop_val u (nth i vals v0)
    | GB o i j => bop_val o (nth i vals v0) (nth j vals v0)
    end.

  (* Evaluate a circuit: returns the values on all wires, inputs first. *)
  Fixpoint run (vals : list V) (c : circuit UOp BOp) : list V :=
    match c with
    | []      => vals
    | g :: c' => run (vals ++ [gate_val vals g]) c'
    end.

  (* R v e  :=  "value v is within the propagated bound C^e". *)
  Definition R (v : V) (e : nat) : Prop := mu v <= qpow C e.

  Lemma Forall2_nth_R :
    forall vals es i, Forall2 R vals es -> (i < length vals)%nat ->
    R (nth i vals v0) (nth i es 0%nat).
  Proof.
    intros vals es i H. revert i.
    induction H as [|v e vals es Hve H IH]; intros i Hi; simpl in *.
    - lia.
    - destruct i; [exact Hve | apply IH; lia].
  Qed.

  Lemma gate_step :
    forall vals es g, Forall2 R vals es -> gate_ok (length vals) g ->
    R (gate_val vals g) (gate_exp es g).
  Proof.
    intros vals es g H Hok. unfold R.
    destruct g as [u i | o i j]; simpl in *.
    - eapply Qle_trans; [apply mu_unary|]. apply Forall2_nth_R; assumption.
    - destruct Hok as [Hi Hj].
      rewrite qpow_add.
      eapply Qle_trans; [apply mu_submult|].
      apply Qmult_le_compat_nonneg; split; auto;
        apply Forall2_nth_R; assumption.
  Qed.

  Lemma run_R :
    forall c vals es, Forall2 R vals es -> wf (length vals) c ->
    Forall2 R (run vals c) (run_e es c).
  Proof.
    induction c as [|g c IH]; intros vals es H Hwf; simpl in *.
    - exact H.
    - destruct Hwf as [Hok Hwf].
      apply IH.
      + apply Forall2_app; [exact H|].
        constructor; [apply gate_step; assumption | constructor].
      + rewrite length_app. simpl. rewrite Nat.add_1_r. exact Hwf.
  Qed.

  Lemma inputs_R :
    forall ins, Forall (fun v => mu v <= C) ins ->
    Forall2 R ins (repeat 1%nat (length ins)).
  Proof.
    induction ins as [|v ins IH]; intro H; simpl.
    - constructor.
    - inversion H; subst. constructor.
      + unfold R. rewrite qpow_1. assumption.
      + apply IH. assumption.
  Qed.

  Lemma Forall_repeat_le : forall k, Forall (fun e => (e <= 1)%nat) (repeat 1%nat k).
  Proof. induction k; simpl; constructor; auto. Qed.

  Lemma Forall2_Forall_bound :
    forall vals es B, Forall2 R vals es -> Forall (fun e => (e <= B)%nat) es ->
    Forall (fun v => mu v <= qpow C B) vals.
  Proof.
    intros vals es B H. induction H as [|v e vals es Hve H IH]; intro HB.
    - constructor.
    - inversion HB; subst. constructor.
      + eapply Qle_trans; [exact Hve | apply qpow_mono; assumption].
      + apply IH. assumption.
  Qed.

  (* --------------------------------------------------------------- *)
  (* 5. The propagation bound: mu <= C^(2^s) on every wire           *)
  (* --------------------------------------------------------------- *)

  Theorem circuit_bound :
    forall ins c,
      Forall (fun v => mu v <= C) ins ->
      wf (length ins) c ->
      Forall (fun v => mu v <= qpow C (2 ^ length c)) (run ins c).
  Proof.
    intros ins c Hins Hwf.
    apply Forall2_Forall_bound with (es := run_e (repeat 1%nat (length ins)) c).
    - apply run_R; [apply inputs_R|]; assumption.
    - replace (2 ^ length c)%nat with (1 * 2 ^ length c)%nat by lia.
      apply doubling. apply Forall_repeat_le.
  Qed.

  (* --------------------------------------------------------------- *)
  (* 6. The lower-bound method and its soundness                     *)
  (* --------------------------------------------------------------- *)

  Definition certifies (f : V) (k : nat) : Prop := qpow C (2 ^ k) < mu f.

  Theorem certifies_sound :
    forall f k ins c,
      certifies f k ->
      Forall (fun v => mu v <= C) ins ->
      wf (length ins) c ->
      In f (run ins c) ->
      (k < length c)%nat.
  Proof.
    intros f k ins c Hcert Hins Hwf Hin.
    pose proof (circuit_bound ins c Hins Hwf) as Hb.
    rewrite Forall_forall in Hb.
    assert (Hexp : (2 ^ k < 2 ^ length c)%nat).
    { apply (qpow_lt_exponent C); [exact C_ge_1|].
      eapply Qlt_le_trans; [exact Hcert | apply Hb; exact Hin]. }
    apply (Nat.pow_lt_mono_r_iff 2); [lia | exact Hexp].
  Qed.

  (* --------------------------------------------------------------- *)
  (* 7. THE CEILING                                                  *)
  (* --------------------------------------------------------------- *)

  Theorem circuit_ceiling :
    forall K, (forall v, mu v <= qpow C K) ->
    forall f k, certifies f k -> (2 ^ k < K)%nat.
  Proof.
    intros K Hmax f k Hcert.
    apply (qpow_lt_exponent C); [exact C_ge_1|].
    eapply Qlt_le_trans; [exact Hcert | apply Hmax].
  Qed.

  (* The same statement in logarithmic form: k <= log2 K.
     For Fourier l1 on n = 2m variables (K = m) this reads
     k <= log2 (n/2). *)
  Corollary circuit_ceiling_log :
    forall K, (forall v, mu v <= qpow C K) ->
    forall f k, certifies f k -> (k <= Nat.log2 K)%nat.
  Proof.
    intros K Hmax f k Hcert.
    pose proof (circuit_ceiling K Hmax f k Hcert) as H.
    apply Nat.log2_le_pow2; lia.
  Qed.

End CircuitCeiling.

(* ----------------------------------------------------------------- *)
(* 8. Tightness: repeated squaring attains C^(2^s)                   *)
(*                                                                   *)
(*    Model: V = Q, mu = Qabs, one binary gate = multiplication,     *)
(*    no unary gates, one input with value C.  The circuit           *)
(*       w_{t+1} := w_t * w_t      (t = 0 .. s-1)                    *)
(*    has s gates and its last wire has |w_s| = C^(2^s) exactly.     *)
(*    So circuit_bound is attained: no argument that uses only the   *)
(*    submultiplicative axioms can certify more than circuit_bound   *)
(*    does.                                                          *)
(*                                                                   *)
(*    Note what this does NOT say.  For Boolean AND, f /\ f = f, so  *)
(*    a real Boolean squaring chain does not blow up.  The point is  *)
(*    that the axioms cannot see that: a proof that reasons only     *)
(*    through submultiplicativity is stuck with 2^s, and therefore   *)
(*    with the log2 K ceiling.  Beating it requires using some       *)
(*    property of mu beyond the axioms.                              *)
(* ----------------------------------------------------------------- *)

Section Tightness.

  Variable C : Q.
  Hypothesis C_ge_1 : 1 <= C.

  Definition t_uop (u : Empty_set) (a : Q) : Q := match u with end.
  Definition t_bop (_ : unit) (a b : Q) : Q := a * b.

  Lemma t_C_nonneg : 0 <= C.
  Proof. eapply Qle_trans; [| exact C_ge_1]. discriminate. Qed.

  (* The model satisfies every hypothesis of CircuitCeiling. *)
  Lemma t_mu_nonneg : forall v, 0 <= Qabs v.
  Proof. apply Qabs_nonneg. Qed.
  Lemma t_mu_unary : forall u a, Qabs (t_uop u a) <= Qabs a.
  Proof. intros u; destruct u. Qed.
  Lemma t_mu_submult : forall o a b, Qabs (t_bop o a b) <= Qabs a * Qabs b.
  Proof. intros. unfold t_bop. rewrite Qabs_Qmult. apply Qle_refl. Qed.
  Lemma t_mu_input : Forall (fun v => Qabs v <= C) [C].
  Proof. constructor; [rewrite Qabs_pos by exact t_C_nonneg; apply Qle_refl | constructor]. Qed.

  (* The squaring chain, gate t reads wire t twice. *)
  Definition sq_chain_from (a s : nat) : circuit Empty_set unit :=
    map (fun t => GB tt t t) (seq a s).

  Definition sq_chain (s : nat) := sq_chain_from 0 s.

  Lemma sq_chain_length : forall s, length (sq_chain s) = s.
  Proof. intro s. unfold sq_chain, sq_chain_from. rewrite length_map, length_seq. reflexivity. Qed.

  Lemma sq_chain_from_wf : forall s a, wf (S a) (sq_chain_from a s).
  Proof.
    induction s as [|s IH]; intro a; simpl; [exact I|].
    split; [split; lia | apply IH].
  Qed.

  Lemma sq_chain_wf : forall s, wf 1 (sq_chain s).
  Proof. intro s. apply sq_chain_from_wf. Qed.

  Lemma run_e_single :
    forall (es : list nat) i j,
      run_e es [@GB Empty_set unit tt i j] = es ++ [(nth i es 0 + nth j es 0)%nat].
  Proof. reflexivity. Qed.

  Lemma nth_pow2_seq :
    forall s k, (s < k)%nat -> nth s (map (fun t => 2 ^ t)%nat (seq 0 k)) 0%nat = (2 ^ s)%nat.
  Proof.
    intros s k Hs.
    rewrite (nth_indep _ 0%nat (2 ^ 0)%nat) by (rewrite length_map, length_seq; exact Hs).
    rewrite map_nth, seq_nth by exact Hs. reflexivity.
  Qed.

  (* Exponent track of the chain: [2^0; 2^1; ...; 2^s]. *)
  Lemma sq_chain_exponents :
    forall s, run_e [1%nat] (sq_chain s) = map (fun t => 2 ^ t)%nat (seq 0 (S s)).
  Proof.
    induction s as [|s IH].
    - reflexivity.
    - unfold sq_chain, sq_chain_from in *.
      rewrite (seq_S s 0), map_app, run_e_app, IH.
      rewrite (seq_S (S s) 0), map_app.
      change (map (fun t => @GB Empty_set unit tt t t) [(0 + s)%nat])
        with [@GB Empty_set unit tt (0 + s) (0 + s)].
      change (map (fun t => (2 ^ t)%nat) [(0 + S s)%nat]) with [(2 ^ (0 + S s))%nat].
      rewrite run_e_single. f_equal. f_equal.
      rewrite nth_pow2_seq by lia.
      simpl. lia.
  Qed.

  (* In the model, mu equals the propagated bound exactly on every wire. *)
  Definition Req (v : Q) (e : nat) : Prop := Qabs v == qpow C e.

  Lemma Forall2_nth_Req :
    forall vals es i, Forall2 Req vals es -> (i < length vals)%nat ->
    Req (nth i vals 0) (nth i es 0%nat).
  Proof.
    intros vals es i H. revert i.
    induction H as [|v e vals es Hve H IH]; intros i Hi; simpl in *.
    - lia.
    - destruct i; [exact Hve | apply IH; lia].
  Qed.

  Lemma run_Req :
    forall c vals es, Forall2 Req vals es -> wf (length vals) c ->
    Forall2 Req (run Q Empty_set unit t_uop t_bop 0 vals c) (run_e es c).
  Proof.
    induction c as [|g c IH]; intros vals es H Hwf; simpl in *.
    - exact H.
    - destruct Hwf as [Hok Hwf]. apply IH.
      + apply Forall2_app; [exact H|]. constructor; [| constructor].
        destruct g as [u i | o i j]; [destruct u|].
        destruct Hok as [Hi Hj]. unfold Req. cbn [gate_val gate_exp]. unfold t_bop.
        rewrite Qabs_Qmult, qpow_add.
        pose proof (Forall2_nth_Req vals es i H Hi) as Ei.
        pose proof (Forall2_nth_Req vals es j H Hj) as Ej.
        unfold Req in Ei, Ej. rewrite Ei, Ej. reflexivity.
      + rewrite length_app. simpl. rewrite Nat.add_1_r. exact Hwf.
  Qed.

  Theorem circuit_tight :
    forall s,
      Qabs (nth s (run Q Empty_set unit t_uop t_bop 0 [C] (sq_chain s)) 0)
      == qpow C (2 ^ length (sq_chain s)).
  Proof.
    intro s.
    assert (Hstart : Forall2 Req [C] [1%nat]).
    { constructor; [| constructor]. unfold Req. rewrite qpow_1.
      apply Qabs_pos. exact t_C_nonneg. }
    pose proof (run_Req (sq_chain s) [C] [1%nat] Hstart (sq_chain_wf s)) as H.
    rewrite sq_chain_exponents in H.
    rewrite sq_chain_length.
    assert (Hlen : (s < length (run Q Empty_set unit t_uop t_bop 0%Q [C] (sq_chain s)))%nat).
    { apply Forall2_length in H. rewrite H, length_map, length_seq. lia. }
    pose proof (Forall2_nth_Req _ _ s H Hlen) as Hs.
    unfold Req in Hs. rewrite Hs.
    rewrite nth_pow2_seq by lia. reflexivity.
  Qed.

  (* And the chain is a legitimate input to the ceiling theorems:
     the propagated bound there is exactly what the model attains. *)
  Corollary circuit_bound_attained :
    forall s,
      Forall (fun v => Qabs v <= qpow C (2 ^ length (sq_chain s)))
             (run Q Empty_set unit t_uop t_bop 0 [C] (sq_chain s))
    /\
      Qabs (nth s (run Q Empty_set unit t_uop t_bop 0 [C] (sq_chain s)) 0)
      == qpow C (2 ^ length (sq_chain s)).
  Proof.
    intro s. split.
    - apply (circuit_bound Q Empty_set unit t_uop t_bop 0 Qabs C C_ge_1 t_mu_nonneg
               t_mu_unary t_mu_submult [C] (sq_chain s) t_mu_input (sq_chain_wf s)).
    - apply circuit_tight.
  Qed.

End Tightness.

Print Assumptions circuit_ceiling_log.
Print Assumptions circuit_tight.
