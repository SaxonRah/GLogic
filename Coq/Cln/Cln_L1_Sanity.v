Require Import Cln_Full Cln_Grade Cln_SupportAlgebra Cln_L1_Ceiling_Instance.
Require Ceiling_Formula.
Require Import Coq.Program.Equality.
From Coq Require Import Lia.

(* IP on 2 variables: x0 /\ x1 in the Pos=true convention, i.e.
   AND(NOT lit0, NOT lit1).  A genuine 2-leaf formula. *)
Definition ip2 : formula 1 :=
  Ceiling_Formula.FBin GAnd
    (Ceiling_Formula.FUnary tt (Ceiling_Formula.FLeaf (Some Fin.F1)))
    (Ceiling_Formula.FUnary tt (Ceiling_Formula.FLeaf (Some (Fin.FS Fin.F1)))).

Lemma ip2_correct : forall s, feval 1 ip2 s = @IP_n_func (1 + 1) s.
Proof.
  intro s. dependent destruction s. dependent destruction s. dependent destruction s.
  destruct h, h0; reflexivity.
Qed.

Lemma ip2_leaves : leaves 1 ip2 = 2%nat.
Proof. reflexivity. Qed.

(* The end-to-end corollary applies to it: 1 < 2 leaves. *)
Example ip2_lower_bound : (1 < leaves 1 ip2)%nat.
Proof. exact (IP_needs_more_than_m_leaves 1 ip2 ip2_correct). Qed.

Check l1_method_best_is_m.
Check l1_method_sound.
Check l1_pm_le_pow2.
