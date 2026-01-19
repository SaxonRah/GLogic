# Mathematical Meaning of the Cl(2,0) Boolean Embedding

The core embedding is formally verified using the Coq proof assistant for the n=2 case. 
See [`Cl2_BooleanEmbedding.v`](Cl2_BooleanEmbedding.v) for the mechanized proof.

> This file (`Cl2_BooleanEmbedding.v`) proves that Boolean logic on two variables is exactly a geometric object, and that evaluating that object reproduces logic itself;  not approximately, but by necessity.

---

## 1. What problem this work addresses

Boolean logic is usually treated as *purely symbolic*: truth tables, formulas, SAT clauses.
Separately, geometry and algebra deal with *continuous* objects: vectors, inner products, correlations.

This work shows that **Boolean logic already lives inside a geometric algebra**;  not approximately, not heuristically, but **exactly**.

For the case of two Boolean variables, every Boolean function

```
f : {±1}² → {0,1}
```

is embedded into a 4-dimensional algebra (the Clifford algebra `Cl(2,0)`) such that:

> Evaluating the embedded object at a Boolean input returns the original Boolean value.

This is not a metaphor. It is an exact, invertible construction.

---

## 2. The geometric space

In `Cl(2,0)`, a general element looks like:

```
F = a0 + a1·e1 + a2·e2 + a12·e12
```

We interpret this **not** as a geometric object in physical space, but as a **multilinear polynomial**:

```
F̂(s1, s2) = a0 + a1·s1 + a2·s2 + a12·s1·s2
where s1, s2 ∈ {±1}
```

This evaluation rule is the bridge between geometry and logic.

---

## 3. Projectors: the key construction

For each Boolean input (a corner of the hypercube)

```
a = (a1, a2) ∈ {±1}²
```

we define a **projector**:

```
Π(a) = 1/4 · (1 + a1·e1 + a2·e2 + a1·a2·e12)
```

This object has a crucial property:

```
Π̂(a)(s) = 1   if s = a
         = 0   otherwise
```

So each projector behaves like a **Kronecker delta** over Boolean inputs.

This is the geometric heart of the construction.

---

## 4. Embedding a Boolean function

Given any Boolean function `f`, we form:

```
F_f = Σ_{a ∈ {±1}²} f(a) · Π(a)
```

This is a **geometric superposition** of projectors.

Because the projectors form a delta basis, evaluating this sum gives:

```
F̂_f(s) = f(s)
```

No approximation.
No learning.
No optimization.

The Boolean function is *exactly reconstructed*.

---

## 5. What the coefficients mean

Each coefficient in `F_f` has semantic meaning:

* `a0`: total truth mass
* `a1,a2`: variable bias terms
* `a12`: genuine two-variable interaction

For example, XOR *requires* a nonzero `a12`.
This is not interpretation;  it is algebraically forced.

Thus, **logical structure appears as geometric grade**.

---

## 6. What the Coq proof establishes

The formal proof verifies all of the above mechanically:

1. **Correct evaluation semantics**
   Multivectors evaluate as multilinear polynomials.

2. **Projector delta property**
   Each projector evaluates to 1 at its own corner and 0 elsewhere.

3. **Linearity**
   Evaluation respects addition and scalar multiplication.

4. **Exact reconstruction theorem**
   The embedding recovers the Boolean function on all inputs.

Every step is checked by Coq's kernel.
Nothing is assumed. Nothing is heuristic.

The embedding for n=2 has been formally verified in Coq (see `Cl2_BooleanEmbedding.v`). 
This proof mechanically checks that for all 2^4 = 16 Boolean functions on two variables, 
the embedding-evaluation round-trip is exact. No assumptions, no numerical tolerance, 
no trust in floating-point arithmetic—just pure type-checked mathematical certainty.

---

# How This Mirrors the Executable Python Proof

The companion Python code and the Coq proof are doing **the same mathematics**, but at different levels.

| Python / Executable Proof | Coq / Formal Proof        |
| ------------------------- | ------------------------- |
| Truth tables              | `Corner`, `corners`       |
| ±1 encoding               | `Sign`, `sQ`              |
| Boolean functions         | `f : Corner → bool`       |
| Fourier-like expansion    | Multivector coefficients  |
| Projector construction    | `Pi`                      |
| Sampling on inputs        | `eval`                    |
| Exhaustive testing        | Case splits               |
| Numerical verification    | Kernel-checked equalities |

### Key correspondence

* In Python, you **run all 4 inputs** and check the outputs match.
* In Coq, you **prove once and for all** that *for any* input, the output matches.

Python shows **that it works**.
Coq proves **why it must work**.

---

## Why both matter

* The Python version shows **computational reality** and scalability.
* The Coq version shows **mathematical inevitability**.

Together, they establish that this is not:

* a visualization trick,
* a numerical coincidence,
* or a heuristic embedding.

It is a **structural identity** between Boolean logic and geometry.

---

## Hypercube

We use the term **hypercube** because Boolean inputs correspond to the vertices of an `n`-dimensional hypercube, and the Clifford algebra `Cl(n,0)` has exactly one basis element per vertex.

Boolean inputs live on a hypercube.
Clifford algebra provides a way to place that hypercube *inside an algebra*.

> The hypercube is the Boolean space.
> Clifford algebra is its coordinate system.

### Why this matters conceptually

Calling it a hypercube is not cosmetic. It signals that:

* logical structure is **geometric**
* correlations correspond to **edges, faces, volumes**
* higher-grade terms correspond to **higher-dimensional faces**

> We use "hypercube" because Boolean logic is defined on hypercube vertices.
>
> `Cl(2,0)` is 4-dimensional because there are 4 vertices.
>
> The embedding is exact because the algebra has exactly the right size.

---

## SAT / 3SAT

In this framework, a 3-SAT instance can be seen as a collection of low-dimensional geometric constraints (faces) embedded inside a high-dimensional Boolean hypercube, with the Clifford algebra acting as the coordinate system that makes those constraints composable and analyzable.

In other words:
- A SAT problem is not a formula or a list of clauses.
- It is a collection of low-dimensional faces carved out of a Boolean hypercube,
with Clifford algebra providing the coordinates that keep those faces composable.
- Satisfiability means those constraints leave at least one vertex untouched.

You can think of each clause as casting a shadow that deletes certain vertices of the hypercube:
- Easy SAT: the shadows barely overlap
- Hard SAT: the shadows overlap chaotically
- UNSAT: the shadows cover the entire hypercube

SAT, geometrically:
- Variables define a Boolean hypercube
- Clauses carve out low-dimensional faces
- Clifford algebra keeps those faces composable
- SAT ⇔ at least one vertex survives

***This doesn’t make SAT easy, but it may be a better space to work in; because it preserves structure that CNF representations throw away.***

Many modern SAT solvers spend a significant portion of their computation reconstructing structural information that is not explicit in a raw CNF representation, such as variable interaction geometry, clause overlap, and locality. This reconstruction happens repeatedly during search through heuristics, conflict analysis, and graph updates, and it carries real computational cost. In this framework, that structural information is preserved by construction: clauses exist as geometric objects on faces of a Boolean hypercube, and their interactions remain explicit throughout the computation. While this does not change the worst-case complexity of SAT, it suggests a different cost tradeoff: by paying an upfront representational cost to embed the problem geometrically, one may reduce redundant structural inference during solving, potentially lowering practical computational load on typical instances.

---

A concrete visualization of **one 3-clause as a 3-cube carved out of a larger hypercube (n=4)**:

(`three_clause_cube_in_hypercube.png`)

### How to read the picture

* The full SAT instance lives on the **4D Boolean hypercube** `{±1}^4`.
* A **single 3-clause** depends on only 3 variables (say `x1, x2, x3`), so it lives on a **3D face** (a cube) of that 4D hypercube.
* The "extra" variable `x4` creates **two parallel 3-cubes**:

  * left cube = slice `x4 = +1`
  * right cube = slice `x4 = -1`

### What clause is shown

This plot uses the clause:

```
(x1 OR x2 OR x3)
```

under the `{±1}` convention:

* `+1` = True
* `-1` = False

This clause is false only when:

```
(x1, x2, x3) = (-1, -1, -1)
```

So in the picture:

* the **X marker** is the *one forbidden vertex* in each cube (the assignment that violates the clause),
* all the **circle markers** are the satisfying assignments for that clause (7 of 8 vertices).

### Why this matches your framework

In your embedding, each clause is a constraint that "removes weight" (or sets to 0) on exactly those vertices that violate it. For a 3-clause, that means "carving out" **one vertex in a 3D subcube**, repeated across all settings of the other variables.
