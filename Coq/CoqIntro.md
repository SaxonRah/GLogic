# Coq for Programmers

## The Big Idea

Coq is a **programming language where the type system is so powerful that types can express logical propositions** - and a program that typechecks is a proof that the proposition is true.

This is the **Curry–Howard correspondence**:
- propositions = types
- proofs = programs
- constructing a proof = writing a program inhabiting a type

Think of it this way:

In C, when you write:
```c
int add(int a, int b);
```

the type signature is a *weak* claim:
> "give me two ints, I'll return an int."

In Coq, a type can say something closer to:
> "give me two numbers, I'll return a number that is exactly their sum, **and here is a compile-time guarantee** of that."

### A concrete "strong spec" example (dependent type)

Coq can return a value *paired with a proof about it*:

```coq
(* Return n along with a proof that n = a + b *)
Definition add_checked (a b : nat) : { n : nat | n = a + b } :=
  exist _ (a + b) eq_refl.
```

Python/C don't have dependent types, but you can mimic the *shape* at runtime:

```python
def add_checked(a: int, b: int):
    n = a + b
    assert n == a + b
    return (n, "proof: asserted")  # runtime check, not compile-time
```

```c
#include <assert.h>

typedef struct {
    int n;
} AddChecked;

AddChecked add_checked(int a, int b) {
    AddChecked out = { .n = a + b };
    assert(out.n == a + b);   // runtime check, not compile-time
    return out;
}
```

**Key difference:** Coq's certificate is checked by the kernel at compile time; Python/C checks are runtime and can be skipped/removed/bypassed.

---

## Two Language Layers (important)

Coq feels like "one thing", but it's effectively two layers:

1. **Gallina**: the pure functional programming language (definitions, functions, data)
2. **Proof mode** (tactics / Ltac): an interactive scripting layer for building proof terms

You can always write proofs as pure terms, but tactics are the ergonomic interface.

---

## Universes: `Type` / `Set` / `Prop`

Think of these as "kinds" or "universes":

* `Set` / `Type` = the universe of **data types** (like `int`, `string`, `list`) → computation-land
* `Prop` = the universe of **logical propositions** (like "x = y", "∀n, n + 0 = n") → proof-land

### Extraction fact programmers care about

* Values in `Prop` are **erased at extraction time** (no runtime cost)
* Computation lives in `Type`/`Set`

---

## Coq is a *Total* Programming Language (the hidden rule)

Coq forbids:

* non-terminating general recursion
* partial functions without proof obligations
* "undefined behavior" / exceptions as a normal escape hatch

Every definitional function must terminate. This is why Curry–Howard works cleanly: proofs/programs are total.

---

# Core Keywords → Programming Equivalents

## `Definition` - `let` / `const`

Bind a name to a value (or define a non-recursive function).

### Coq

```coq
Definition x := 5.
Definition add a b := a + b.
```

### Python

```python
x = 5
def add(a, b): return a + b
```

### C

```c
int x = 5;
int add(int a, int b) { return a + b; }
```

---

## `Fixpoint` - recursive function (with termination)

Coq requires recursion to be structurally decreasing (or otherwise justified).

### Coq

```coq
Fixpoint factorial (n : nat) : nat :=
  match n with
  | 0 => 1
  | S n' => n * factorial n'
  end.
```

### Python

```python
def factorial(n: int) -> int:
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

### C

```c
int factorial(int n) {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}
```

**Difference:** Python/C allow non-termination; Coq rejects definitions that aren't provably terminating.

---

## `Inductive` - algebraic data types (enums / tagged unions)

Exactly like Rust enums / Haskell data / ML variants.

### Coq

```coq
Inductive bool :=
  | true
  | false.
```

### Python (Enum)

```python
from enum import Enum
class Bool(Enum):
    TRUE = 1
    FALSE = 2
```

### C

```c
typedef enum { TRUE, FALSE } Bool;
```

---

### `nat` - Peano naturals

A natural number is either:

* `O` (zero)
* `S n` (successor)

### Coq

```coq
Inductive nat :=
  | O
  | S (n : nat).
```

Programmer intuition: it's like a linked list of `+1`s.

### Python (conceptual)

```python
class Nat:
    def __init__(self, prev=None):
        self.prev = prev

O = Nat(None)
def S(n): return Nat(n)
```

### C (conceptual)

```c
typedef struct Nat {
    struct Nat* prev;
} Nat;
```

(In real C you'd almost never implement naturals this way; this is about matching the *constructor structure*.)

---

### `list` - generic List<T>

### Coq

```coq
Inductive list (A : Type) :=
  | nil
  | cons (x : A) (xs : list A).
```

### Python (linked-list style)

```python
class List:
    def __init__(self, head=None, tail=None):
        self.head = head
        self.tail = tail

nil = None
def cons(x, xs): return List(x, xs)
```

### C (linked list, fixed element type)

```c
typedef struct Node {
    int value;
    struct Node* next;
} Node;
```

---

## `match ... with ... end` - pattern matching

Switch / match from ML/Rust/Haskell.

### Coq

```coq
match n with
| 0 => ...
| S n' => ...
end
```

### Python

```python
if n == 0:
    ...
else:
    ...
```

### C

```c
if (n == 0) { ... }
else { ... }
```

---

## `Theorem` / `Lemma` / `Proposition`

All the same internally: they declare a proposition (a type in `Prop`) that you must inhabit with a proof term.

### Coq

```coq
Theorem add_0_r : forall n : nat, n + 0 = n.
```

Programming mental model:

```python
def add_0_r(n: Nat) -> Proof[n + 0 == n]:
    ...
```

---

## `forall` - generics (and also "for all")

**As generics / polymorphism:**

### Coq

```coq
forall (A : Type), A -> A
```

### Python

```python
from typing import TypeVar
T = TypeVar("T")
def identity(x: T) -> T:
    return x
```

### C++ (closest)

```cpp
template<typename T>
T identity(T x) { return x; }
```

**In `Prop`,** `forall` is also literal math "for every value".

---

## `fun` - lambda

### Coq

```coq
fun x => x + 1
```

### Python

```python
lambda x: x + 1
```

### C (no real lambda in C; function pointer style)

```c
int inc(int x) { return x + 1; }
```

---

## `Proof ... Qed`

This is the "function body" for a theorem.

* `Proof.` enters interactive proof mode
* `Qed.` closes it and registers the term

Think of `Qed` as "compile and seal".

---

## `Admitted`

This is `TODO` / `unimplemented!()` / "trust me bro".

It allows the file to compile but makes the development **unsound**.

### Coq

```coq
Lemma foo : 2 + 2 = 5.
Admitted.
```

Equivalent spirit:

### Python

```python
def foo():
    raise NotImplementedError
```

### C

```c
int foo() {
    /* TODO */
    return 0; /* lies */
}
```

---

## `Require Import`

Like `import` / `#include`.

```coq
Require Import List.
Require Import Arith.
```

---

## `Section` / `End`

Scoping blocks like namespaces/modules (variables inside are local and generalized when the section ends).

---

## `Record` - struct

### Coq

```coq
Record Point := {
  x : nat;
  y : nat;
}.
```

### Python

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int
```

### C

```c
typedef struct {
    int x;
    int y;
} Point;
```

---

## `Class` / `Instance` - typeclasses (traits)

Like Haskell typeclasses / Rust traits (conceptually).

---

## `Notation` - custom syntax / operator overloading

Write `x + y` instead of `Nat.add x y`, etc.

---

# Tactics → Programming Analogues

When you enter proof mode (`Proof.`), you see a **goal** (the type you must inhabit).
Tactics are **commands that transform the goal** until nothing is left.

Think: interactive debugger, but instead of stepping through execution, you're stepping through **construction** of a term.

You'll see:

```
n : nat
==================
n + 0 = n
```

* context (top) = variables/hypotheses in scope
* goal (bottom) = what you must produce

---

## `intros` - accept arguments

### Coq

```coq
intros n.
```

Like adding function parameters.

---

## `exact` - return this term

### Coq

```coq
exact eq_refl.
```

Equivalent to:

```python
return value
```

---

## `apply` - reverse function application

If your goal is `B` and you have a lemma `A -> B`, then:

```coq
apply lemma.
```

changes the goal to `A`.

Programmer framing:

> "I know how to produce `B` if you give me an `A`."

---

## `simpl` - normalize / reduce (compile-time evaluation)

This is "run the definitional computation rules" (pure reduction).
Not runtime execution - more like compiler normalization.

---

## `unfold` - inline a definition

Like manual inlining.

---

## `rewrite` - substitution using equality

Given `H : a = b`, `rewrite H` replaces `a` with `b` in the goal.

`rewrite <- H` does it backwards.

---

## `reflexivity` - goal is syntactically identical

If both sides reduce to the same term, `reflexivity` closes it.

---

## `induction` - structural recursion / proof by induction

For `nat`, it generates:

* base case (`0`)
* step case (`S n'`) with induction hypothesis

Programmer mental model:

> You are defining a recursive function by specifying base and recursive cases.

---

## `destruct` - case split (pattern match) without induction hypothesis

Like a `switch` over constructors.

---

## `split` - prove both sides of `A /\ B`

Turns one goal into two.

Like constructing a tuple:

```python
return (proofA, proofB)
```

---

## `left` / `right` - choose a side of `A \/ B`

Like constructing an `Either`.

---

## `exists` - provide a witness for `exists x, P x`

You give the value, then prove it satisfies the property.

Dependent pair construction.

---

## `assumption` - goal already in context

Return a variable/hypothesis you already have.

---

## `contradiction` - impossible path

If your context contains inconsistent hypotheses, the goal closes (ex falso).

Like unreachable code.

---

## `auto` - automation / proof search

Tries a database of lemmas and basic reasoning steps.

Autocomplete for proofs.

---

## `omega` / `lia` - arithmetic solvers

Automated solvers for linear arithmetic goals.

Think: "SMT for integer linear arithmetic".

---

## `discriminate` - constructor mismatch

If you have something like `0 = S n`, Coq can close it.

Like: `None == Some(x)` is impossible.

---

## `inversion` - deep constructor reasoning

From `Some x = Some y`, deduce `x = y`.
From an inductive relation, inversion extracts the constraints implied by constructors.

---

## `assert` - introduce a local helper lemma

Like declaring a local helper in the middle of a function.

---

## `trivial` - very small automation

Weaker than `auto`.

---

# Logical Connectives → Type Equivalents

| Coq / Logic      | Programming Equivalent      | Explanation             |
| ---------------- | --------------------------- | ----------------------- |
| `A /\ B`         | `(A, B)` tuple / pair       | you have both           |
| `A \/ B`         | `Either<A, B>`              | you have one            |
| `A -> B`         | function type               | given A produce B       |
| `~A`             | `A -> Void`                 | A implies impossibility |
| `True`           | `()` unit                   | always constructible    |
| `False`          | `Never` / `Void`            | no constructors         |
| `exists x, P(x)` | dependent pair `{x, proof}` | value + certificate     |
| `forall x, P(x)` | generic function            | works for every input   |
| `x = y`          | equality proof              | evidence they're equal  |

---

# The Mental Model

Think of Coq like this:

1. **Define data types** (`Inductive`) like Rust/Haskell
2. **Write total functions** (`Definition`, `Fixpoint`)
3. **State properties** (`Theorem`) as extremely strong type signatures
4. **Implement the proof** using tactics (constructing a program term)
5. **Extract** verified computation to OCaml/Haskell/Scheme

The core insight:

> **If the type is expressive enough, typechecking is verification.**

Coq gives you a type system expressive enough to encode the claims you actually care about.

---

# What makes Coq feel alien (at first)

* no partial functions / no silent failure
* termination is enforced
* rewriting is the normal "state change"
* dependent types (value + proof) show up everywhere
* you may need to guide automation instead of "just running code"

---

# A practical milestone (recommended)

Instead of only proving:

* `n + 0 = n`

try verifying something programmers respect immediately:

* a parser
* a SAT evaluator for CNF
* a simplifier + correctness proof
* a small compiler pass (and show semantics preserved)

That's when Coq stops being "logic homework" and becomes "verified engineering".

---

# Rosetta Section - Same Artifact in Coq / Python / C

This section shows the *same conceptual program* written three ways so programmers can see what Coq is doing differently.

We'll implement:

1. Natural numbers
2. Addition
3. Factorial
4. A property: `n + 0 = n`
5. What that proof corresponds to computationally

The key thing to watch:

> Coq encodes *structure + correctness* at the type level.

Python/C encode *behavior only*.

---

# 1) Natural Numbers

## Coq (Peano naturals)

```coq
Inductive nat :=
  | O
  | S (n : nat).
````

Numbers are constructed like:

* `0` → `O`
* `1` → `S O`
* `2` → `S (S O)`

This representation enables structural recursion and induction.

---

## Python (conceptual structural encoding)

Python normally uses built-in ints, but here's the equivalent structure:

```python
class Nat:
    def __init__(self, prev=None):
        self.prev = prev

O = Nat()

def S(n):
    return Nat(n)
```

---

## C (conceptual)

```c
typedef struct Nat {
    struct Nat* prev;
} Nat;

Nat* O() {
    return NULL;
}

Nat* S(Nat* n) {
    Nat* out = malloc(sizeof(Nat));
    out->prev = n;
    return out;
}
```

(Not idiomatic C - this mirrors the mathematical structure.)

---

# 2) Addition

## Coq

```coq
Fixpoint add (a b : nat) : nat :=
  match a with
  | O => b
  | S a' => S (add a' b)
  end.
```

This is structural recursion over `a`.

---

## Python

```python
def add(a, b):
    if a.prev is None:
        return b
    return S(add(a.prev, b))
```

---

## C

```c
Nat* add(Nat* a, Nat* b) {
    if (a == NULL) return b;
    return S(add(a->prev, b));
}
```

---

# 3) Factorial

## Coq

```coq
Fixpoint mul (a b : nat) : nat :=
  match a with
  | O => O
  | S a' => add b (mul a' b)
  end.

Fixpoint factorial (n : nat) : nat :=
  match n with
  | O => S O
  | S n' => mul n (factorial n')
  end.
```

---

## Python

```python
def mul(a, b):
    if a.prev is None:
        return O
    return add(b, mul(a.prev, b))

def factorial(n):
    if n.prev is None:
        return S(O)
    return mul(n, factorial(n.prev))
```

---

## C

```c
Nat* mul(Nat* a, Nat* b) {
    if (a == NULL) return NULL;
    return add(b, mul(a->prev, b));
}

Nat* factorial(Nat* n) {
    if (n == NULL) return S(NULL);
    return mul(n, factorial(n->prev));
}
```

---

# 4) Property: `n + 0 = n`

This is where Coq diverges fundamentally.

In Python/C:

* you *test* this
* maybe with unit tests

In Coq:

* you **prove it once for all n**
* and the compiler verifies it

---

## Coq Theorem

```coq
Theorem add_0_r : forall n : nat, add n O = n.
Proof.
  induction n.
  - reflexivity.
  - simpl.
    rewrite IHn.
    reflexivity.
Qed.
```

Explanation:

* base case: `O + O = O`
* step case:

  * assume `add n O = n`
  * show `add (S n) O = S n`

This proof is a program that constructs evidence.

---

## Python equivalent mindset

You would write tests:

```python
assert add(O, O) == O
assert add(S(O), O) == S(O)
assert add(S(S(O)), O) == S(S(O))
```

But:

* only tests a finite subset
* not universal

---

## C equivalent mindset

```c
assert(equal(add(O(), O()), O()));
assert(equal(add(S(O()), O()), S(O())));
```

Still finite.

---

# 5) What the proof *is* computationally

This is the critical Curry–Howard insight.

The Coq proof:

```coq
induction n.
```

is structurally identical to writing a recursive function:

```coq
Fixpoint proof_add_0_r (n : nat) : add n O = n :=
  match n with
  | O => eq_refl
  | S n' =>
      (* recursive call *)
      (* transport equality through constructor *)
  end.
```

You are building a function that:

* takes `n`
* returns a proof term

So:

| Proof step  | Program interpretation   |
| ----------- | ------------------------ |
| induction   | recursion                |
| base case   | base branch              |
| IH          | recursive call           |
| rewrite     | substitution             |
| reflexivity | returning identity proof |

Proofs are programs.

---

# 6) The Extraction Difference

If we extract:

```coq
Definition add ...
Fixpoint factorial ...
```

they become executable OCaml/Haskell.

If we extract:

```coq
Theorem add_0_r ...
```

it disappears.

Because:

* proofs live in `Prop`
* `Prop` is erased

Runtime keeps computation; discards certificates.

---

# 7) What changed vs Python/C

## Python/C workflow

1. write program
2. test behavior
3. hope correctness generalizes

## Coq workflow

1. define data
2. define total functions
3. state properties
4. prove properties
5. extract executable code

Correctness becomes part of compilation.

---

# 8) The conceptual shift

Normal languages:

> types prevent crashes

Coq:

> types prevent logical errors

And with dependent types:

> specifications become part of the program itself

---

# 9) Why this matters beyond toy examples

The exact same pattern scales to:

* verified compilers
* SAT solvers with correctness proofs
* symbolic algebra systems
* cryptographic protocols
* geometric algebra implementations
* computational complexity models

Once formalized:

> every theorem becomes a machine-checked invariant of the system.

You're no longer relying on:

* intuition
* testing
* peer review alone

You get kernel-checked correctness.

---

> In Coq, you don't test programs - you construct them so incorrect ones are untypeable.

---

# Runnable Examples 
Below are **paste-and-run** mini "files" for **Coq**, plus matching **Python** and **C** "same idea" versions.

Everything is intentionally small and self-contained.

---

## Example 1: Addition + proof that `n + 0 = n`

### Coq (paste into CoqIDE as a single file)

```coq
(*
  File: Add0.v
  Paste into CoqIDE / VSCoq and run.
*)

From Coq Require Import Arith.

(* A tiny wrapper around Coq's built-in Nat.add, just so it's explicit. *)
Definition add (a b : nat) : nat := Nat.add a b.

Theorem add_0_r : forall n : nat, add n 0 = n.
Proof.
  intro n.
  unfold add.
  (* now the goal is Nat.add n 0 = n *)
  induction n as [|n IH].
  - simpl. reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.
```

### Python (same idea: compute + "property check")

```python
# file: add0.py

def add(a: int, b: int) -> int:
    return a + b

def check_add_0_r(limit: int = 1000) -> None:
    for n in range(limit):
        assert add(n, 0) == n

if __name__ == "__main__":
    check_add_0_r()
    print("checked add(n,0)=n for n < 1000")
```

### C (same idea)

```c
/* file: add0.c */
#include <assert.h>
#include <stdio.h>

int add(int a, int b) { return a + b; }

void check_add_0_r(int limit) {
    for (int n = 0; n < limit; n++) {
        assert(add(n, 0) == n);
    }
}

int main(void) {
    check_add_0_r(1000);
    puts("checked add(n,0)=n for n < 1000");
    return 0;
}
```

**Key contrast:** Python/C "verify" by testing finitely many inputs; Coq proves it for *all* `n`.

---

## Example 2: Lists + `map id = id` (a "real programmer" lemma)

This is super common in functional code: mapping identity over a list returns the same list.

### Coq (paste into CoqIDE)

```coq
(*
  File: MapId.v
  Paste into CoqIDE / VSCoq and run.
*)

From Coq Require Import List.
Import ListNotations.

Definition id {A : Type} (x : A) : A := x.

Theorem map_id :
  forall (A : Type) (xs : list A),
    map (@id A) xs = xs.
Proof.
  intros A xs.
  induction xs as [|x xs IH].
  - simpl. reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.
```

### Python (same idea, tested)

```python
# file: map_id.py

from typing import TypeVar, List, Callable
T = TypeVar("T")

def identity(x: T) -> T:
    return x

def map_list(f: Callable[[T], T], xs: List[T]) -> List[T]:
    return [f(x) for x in xs]

def check_map_id() -> None:
    test_cases = [
        [],
        [1],
        [1, 2, 3],
        ["a", "b"],
    ]
    for xs in test_cases:
        assert map_list(identity, xs) == xs

if __name__ == "__main__":
    check_map_id()
    print("checked map(identity, xs) == xs on sample cases")
```

### C (same *concept*, but C needs a concrete element type)

C doesn't have parametric polymorphism, so we show it for `int`.

```c
/* file: map_id_int.c */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*int_fn)(int);

int identity(int x) { return x; }

void map_int(int_fn f, const int* xs, int* out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = f(xs[i]);
}

int arrays_equal(const int* a, const int* b, size_t n) {
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}

int main(void) {
    int xs[] = {1,2,3,4};
    size_t n = sizeof(xs)/sizeof(xs[0]);
    int* out = malloc(n * sizeof(int));

    map_int(identity, xs, out, n);
    assert(arrays_equal(xs, out, n));

    free(out);
    puts("checked map(identity, xs) == xs for an int array");
    return 0;
}
```

---

## What these two examples teach (in programmer terms)

* In Coq, the **theorem statement is a type signature**, and the proof is the "implementation."
* `induction` is the same shape as **recursive code over a datatype**.
* In Python/C, you can only do finite checks unless you bring in heavy external machinery.

---

Below is a third **paste-and-run** example that shows the thing most programmers are missing until they see it end-to-end:

> a function that returns a value **plus a proof that it is correct**

This is where Coq stops looking like "typed ML" and starts looking like a **specification language you can execute**.

---

# Example 3: Value + Certificate (Dependent Types)

We'll implement:
* a function that adds two numbers
* it returns the result
* **and** a proof the result really equals `a + b`

In Python/C this becomes "return a value + runtime assertion."
In Coq it becomes "return a value + compile-time proof."

---

## Coq - Complete runnable file

Paste into CoqIDE / VSCoq as one file.

```coq
(*
  File: AddWithProof.v
  Demonstrates a function returning a value PLUS a proof.
*)

From Coq Require Import Arith.

(* Dependent pair:
   { n : nat | P n } means:
   - a value n
   - and a proof that P n holds
*)

Definition add_with_proof (a b : nat)
  : { n : nat | n = a + b }.
Proof.
  (* we choose the witness *)
  exists (a + b).
  (* now we must prove it equals a + b *)
  reflexivity.
Defined.

(*
  Use it:
*)

Compute (proj1_sig (add_with_proof 2 3)).
```

What this means:
* function returns `n`
* and a proof object that `n = a + b`
* proof checked by kernel
* erased at runtime

---

## Python equivalent (runtime certificate)

```python
# file: add_with_proof.py

from typing import Tuple

def add_with_proof(a: int, b: int) -> Tuple[int, str]:
    n = a + b
    assert n == a + b
    return (n, "proof: assertion succeeded")

if __name__ == "__main__":
    value, proof = add_with_proof(2, 3)
    print(value, proof)
```

This is **not** the same guarantee:
* assertion can be removed
* runtime only
* not universal

---

## C equivalent (runtime certificate)

```c
/* file: add_with_proof.c */
#include <assert.h>
#include <stdio.h>

typedef struct {
    int value;
} AddResult;

AddResult add_with_proof(int a, int b) {
    AddResult out;
    out.value = a + b;
    assert(out.value == a + b);
    return out;
}

int main(void) {
    AddResult r = add_with_proof(2, 3);
    printf("%d\n", r.value);
    return 0;
}
```

Again:
* runtime check
* not part of type system

---

# Why this example matters
This tiny Coq function demonstrates the core power:

## In Python/C
Types say:
> this returns an int

## In Coq
Types say:
> this returns an int that is provably equal to a + b

The guarantee is part of compilation.

---

# The deeper interpretation

`{ n : nat | P n }` is a **dependent pair**:

| Component | Meaning                    |
| --------- | -------------------------- |
| value     | computation                |
| proof     | certificate of correctness |

This pattern scales to:

* SAT solver returning assignment + proof it satisfies formula
* optimizer returning program + proof semantics preserved
* cryptographic function returning ciphertext + proof of invariant
* geometric algebra routines returning result + proof of algebraic law

You're no longer separating:

* code
* tests
* documentation

They become one artifact.

---

# What to internalize

Normal languages:
> types describe shape

Coq:
> types describe truth

And with dependent types:
> types describe *truth about specific values*

---

# Where to go next (natural progression)

Now that we have:

* runnable Coq files
* runnable Python/C parallels

The next "bridge" examples that make programmers fully click:

1. **Verified function**
   * write `reverse`
   * prove `reverse (reverse xs) = xs`

2. **Verified data structure**
   * binary tree
   * prove lookup correctness

3. **Verified SAT evaluator**
   * compute truth value
   * prove evaluation matches semantics

4. **Extraction**
   * extract Coq function → OCaml
   * run verified executable

That's when Coq stops being "interesting theory" and becomes:
> a practical language for writing software that cannot lie.
