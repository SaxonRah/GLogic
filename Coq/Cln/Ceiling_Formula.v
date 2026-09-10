(* ================================================================= *)
(*  Ceiling_Formula.v                                                *)
(*                                                                   *)
(*  The formula-size ceiling for submultiplicative measures.         *)
(*                                                                   *)
(*  Setting.  Values V (Boolean functions, multivectors, ...) are    *)
(*  built by formulas: leaves, unary gates, binary gates.  A measure *)
(*  mu : V -> Q is assumed to satisfy                                *)
(*                                                                   *)
(*      0 <= mu v                        (nonnegative)               *)
(*      mu (leaf) <= C                   (bounded generators)        *)
(*      mu (u a)  <= mu a                (unary gates don't expand)  *)
(*      mu (o a b) <= mu a * mu b        (submultiplicative)         *)
(*                                                                   *)
(*  Main results.                                                    *)
(*                                                                   *)
(*    formula_bound      mu (eval phi) <= C ^ (leaves phi)           *)
(*    certifies_sound    if C^k < mu f, every formula for f has      *)
(*                       more than k leaves  (the lower-bound        *)
(*                       method, proven sound)                       *)
(*    formula_ceiling    if mu <= C^K on the whole value space,      *)
(*                       every bound the method certifies is < K     *)
(*    formula_tight      the propagated bound is attained in a       *)
(*                       concrete model of the axioms, so no         *)
(*                       argument using only these axioms can        *)
(*                       certify more                                *)
(*                                                                   *)
(*  Instance.  Fourier l1 on n = 2m variables, +-1 convention:       *)
(*  C = 2 (l1 of a literal is 1, of a 2-input AND is 2), and         *)
(*  l1 <= 2^(n/2) by Cauchy-Schwarz + Parseval, so K = m = n/2.      *)
(*  formula_ceiling then says the l1 method certifies at most n/2    *)
(*  leaves.  IP = XOR_i (x_i /\ y_i) meets this exactly.             *)
(*                                                                   *)
(*  Dependencies: Coq standard library only.  No Axioms, no Admits.  *)
(* ================================================================= *)

From Coq Require Import QArith Qabs Lia.
Open Scope Q_scope.

(* ----------------------------------------------------------------- *)
(* 1. Natural-number powers in Q                                     *)
(*    (own definition, to stay clear of Qpower's Z exponents)        *)
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

Lemma qpow_ge_1 : forall c k, 1 <= c -> 1 <= qpow c k.
Proof.
  intros c k Hc. induction k as [|k IH]; simpl.
  - apply Qle_refl.
  - setoid_replace 1 with (1 * 1) by ring.
    apply Qmult_le_compat_nonneg; split; try assumption; discriminate.
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

(* Strict-inequality contrapositive of monotonicity: the workhorse of
   every "certified bound is capped" argument below. *)
Lemma qpow_lt_exponent :
  forall c a b, 1 <= c -> qpow c a < qpow c b -> (a < b)%nat.
Proof.
  intros c a b Hc Hlt.
  destruct (Nat.lt_ge_cases a b) as [H|H]; [exact H|].
  exfalso. apply (Qlt_not_le _ _ Hlt). apply qpow_mono; assumption.
Qed.

(* ----------------------------------------------------------------- *)
(* 2. Formulas and the submultiplicative measure                     *)
(* ----------------------------------------------------------------- *)

Section FormulaCeiling.

  Variable V    : Type.   (* semantic values                       *)
  Variable Leaf : Type.   (* leaf labels: literals, constants      *)
  Variable UOp  : Type.   (* unary gate labels, e.g. NOT           *)
  Variable BOp  : Type.   (* binary gate labels, e.g. AND, XOR     *)

  Variable leaf_val : Leaf -> V.
  Variable uop_val  : UOp -> V -> V.
  Variable bop_val  : BOp -> V -> V -> V.

  Variable mu : V -> Q.
  Variable C  : Q.

  Hypothesis C_ge_1      : 1 <= C.
  Hypothesis mu_nonneg   : forall v, 0 <= mu v.
  Hypothesis mu_leaf     : forall l, mu (leaf_val l) <= C.
  Hypothesis mu_unary    : forall u a, mu (uop_val u a) <= mu a.
  Hypothesis mu_submult  : forall o a b, mu (bop_val o a b) <= mu a * mu b.

  Inductive formula : Type :=
  | FLeaf  : Leaf -> formula
  | FUnary : UOp -> formula -> formula
  | FBin   : BOp -> formula -> formula -> formula.

  Fixpoint eval (phi : formula) : V :=
    match phi with
    | FLeaf l      => leaf_val l
    | FUnary u a   => uop_val u (eval a)
    | FBin o a b   => bop_val o (eval a) (eval b)
    end.

  (* Formula size = number of leaves (the standard measure;
     binary gates = leaves - 1). *)
  Fixpoint leaves (phi : formula) : nat :=
    match phi with
    | FLeaf _      => 1
    | FUnary _ a   => leaves a
    | FBin _ a b   => leaves a + leaves b
    end.

  Fixpoint bin_gates (phi : formula) : nat :=
    match phi with
    | FLeaf _      => 0
    | FUnary _ a   => bin_gates a
    | FBin _ a b   => S (bin_gates a + bin_gates b)
    end.

  Lemma leaves_bin_gates : forall phi, leaves phi = S (bin_gates phi).
  Proof. induction phi; simpl; lia. Qed.

  (* --------------------------------------------------------------- *)
  (* 3. The propagation bound                                        *)
  (* --------------------------------------------------------------- *)

  Theorem formula_bound : forall phi, mu (eval phi) <= qpow C (leaves phi).
  Proof.
    induction phi as [l | u a IH | o a IHa b IHb]; simpl.
    - rewrite Qmult_1_r. apply mu_leaf.
    - eapply Qle_trans; [apply mu_unary | exact IH].
    - rewrite qpow_add.
      eapply Qle_trans; [apply mu_submult|].
      apply Qmult_le_compat_nonneg; split; auto.
  Qed.

  (* --------------------------------------------------------------- *)
  (* 4. The lower-bound method, and its soundness                    *)
  (*                                                                 *)
  (*    "mu certifies k for f" means C^k < mu f.  Soundness: then     *)
  (*    every formula computing f has more than k leaves.            *)
  (* --------------------------------------------------------------- *)

  Definition certifies (f : V) (k : nat) : Prop := qpow C k < mu f.

  Theorem certifies_sound :
    forall f k phi, certifies f k -> eval phi = f -> (k < leaves phi)%nat.
  Proof.
    intros f k phi Hcert Heval.
    apply (qpow_lt_exponent C); [exact C_ge_1|].
    eapply Qlt_le_trans; [exact Hcert|].
    rewrite <- Heval. apply formula_bound.
  Qed.

  (* --------------------------------------------------------------- *)
  (* 5. THE CEILING                                                  *)
  (*                                                                 *)
  (*    If the measure never exceeds C^K on the value space, then     *)
  (*    every lower bound the method can certify, for any f at all,  *)
  (*    is below K.  The cap depends only on the range of mu, not    *)
  (*    on f and not on how clever the rest of the proof is.         *)
  (* --------------------------------------------------------------- *)

  Theorem formula_ceiling :
    forall K, (forall v, mu v <= qpow C K) ->
    forall f k, certifies f k -> (k < K)%nat.
  Proof.
    intros K Hmax f k Hcert.
    apply (qpow_lt_exponent C); [exact C_ge_1|].
    eapply Qlt_le_trans; [exact Hcert | apply Hmax].
  Qed.

End FormulaCeiling.

Arguments FLeaf  {Leaf UOp BOp} _.
Arguments FUnary {Leaf UOp BOp} _ _.
Arguments FBin   {Leaf UOp BOp} _ _ _.

(* ----------------------------------------------------------------- *)
(* 6. Tightness: the propagation bound cannot be improved using      *)
(*    these axioms alone.                                            *)
(*                                                                   *)
(*    Model: V = Q, mu = Qabs, one leaf with value C, one binary     *)
(*    gate = multiplication.  All hypotheses of the section hold,    *)
(*    and a balanced product of k leaves has mu exactly C^k.  So     *)
(*    formula_bound is attained, and any proof that uses only        *)
(*    {nonneg, bounded leaves, submultiplicativity} cannot certify   *)
(*    anything formula_bound doesn't.  (IP is the Fourier-l1         *)
(*    realization of this model: disjoint XOR multiplies l1          *)
(*    exactly, and each x_i /\ y_i has l1 = 2.)                      *)
(* ----------------------------------------------------------------- *)

Section Tightness.

  Variable C : Q.
  Hypothesis C_ge_1 : 1 <= C.

  Definition t_leaf (_ : unit) : Q := C.
  Definition t_uop  (_ : Empty_set) (a : Q) : Q := a.
  Definition t_bop  (_ : unit) (a b : Q) : Q := a * b.

  Lemma t_C_nonneg : 0 <= C.
  Proof. eapply Qle_trans; [| exact C_ge_1]. discriminate. Qed.

  Lemma t_mu_nonneg : forall v, 0 <= Qabs v.
  Proof. apply Qabs_nonneg. Qed.

  Lemma t_mu_leaf : forall l, Qabs (t_leaf l) <= C.
  Proof.
    intro l. unfold t_leaf. rewrite Qabs_pos by exact t_C_nonneg. apply Qle_refl.
  Qed.

  Lemma t_mu_unary : forall u a, Qabs (t_uop u a) <= Qabs a.
  Proof. intros u; destruct u. Qed.

  Lemma t_mu_submult : forall o a b, Qabs (t_bop o a b) <= Qabs a * Qabs b.
  Proof. intros o a b. unfold t_bop. rewrite Qabs_Qmult. apply Qle_refl. Qed.

  (* Left-comb product of k+1 leaves. *)
  Fixpoint comb (k : nat) : formula unit Empty_set unit :=
    match k with
    | O    => FLeaf tt
    | S k' => FBin tt (comb k') (FLeaf tt)
    end.

  Lemma comb_leaves : forall k, leaves _ _ _ (comb k) = S k.
  Proof. induction k; simpl; [reflexivity | rewrite IHk; lia]. Qed.

  Lemma comb_eval :
    forall k, eval Q unit Empty_set unit t_leaf t_uop t_bop (comb k) == qpow C (S k).
  Proof.
    induction k as [|k IH]; simpl.
    - unfold t_leaf. ring.
    - unfold t_bop at 1. rewrite IH. unfold t_leaf. simpl. ring.
  Qed.

  (* The bound mu(eval phi) <= C^(leaves phi) holds with equality. *)
  Theorem formula_tight :
    forall k,
      Qabs (eval Q unit Empty_set unit t_leaf t_uop t_bop (comb k))
      == qpow C (leaves _ _ _ (comb k)).
  Proof.
    intro k. rewrite comb_leaves, comb_eval.
    apply Qabs_pos. apply qpow_nonneg. exact t_C_nonneg.
  Qed.

End Tightness.

Print Assumptions formula_ceiling.
Print Assumptions formula_tight.
