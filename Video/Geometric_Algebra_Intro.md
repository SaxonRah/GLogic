# Geometric Algebra for the Absolute Beginner

## Start With What You Know: Numbers

You know numbers: 1, 2, 3, -5, 3.14...

Numbers measure **size**. That's it. Just size.

```
3 apples
5 miles  
-2 degrees (below zero)
```

One dimension: bigger or smaller.

---

## Level Up: Vectors (Numbers With Direction)

A vector is a number that **also points somewhere**.

```
Regular number:     5
Vector:            5 →  (5 units to the right)
                   
Another vector:    3 ↑  (3 units up)
```

**Real world example:**
- "I drove 50 miles" ← number (just size)
- "I drove 50 miles north" ← vector (size AND direction)

Vectors live in 2D, 3D, or higher spaces. They have **magnitude** (how long) and **direction** (which way).

---

## The Problem: How Do You Multiply Vectors?

Multiplying numbers is easy: `3 × 5 = 15`

But what does this mean?
```
(5 →) × (3 ↑) = ???
```

**Traditional math gives you TWO different answers:**

### Option 1: Dot Product (·)
"How much do they point in the same direction?"
```
(5 →) · (3 ↑) = 0
```
Result: A **number** (they're perpendicular, so zero)

### Option 2: Cross Product (×)
"What's perpendicular to both?"
```
(5 →) × (3 ↑) = 15 ⊗
```
Result: A **new vector** pointing out of the page

**The problem:** You need TWO different multiplications! That's like having "plus" for horizontal and "mega-plus" for vertical. It's messy.

---

## Enter: Geometric Algebra (The Unified Multiplication)

**Big Idea:** There's actually ONE multiplication that does both, and more.

It's called the **geometric product**, and it works like this:

```
a · b = (dot product part) + (wedge product part)
        \___ number ___/     \____ new object ____/
```

Let me show you with our example:

```
(5 →) · (3 ↑) = 0 + (15 →↑)
                ↑     ↑
            parallel   perpendicular
            component  component
```

**The perpendicular part** `(15 →↑)` is called a **bivector**. It's not a vector—it's an **oriented area**.

---

## What's a Bivector? (The New Thing)

Think of a bivector as **the plane** swept out by two vectors:

```
    ↑ 3 units
    |
    |______→ 5 units
    
This creates a plane!
```

**Key properties:**
- It has **area** (magnitude): 15 square units
- It has **orientation** (which way it spins): counterclockwise
- It's **not** a vector (it's 2D, not 1D)

**Notation:** We write it as `e₁₂` (the plane made by basis vectors e₁ and e₂)

---

## The Big Picture: Grades

Geometric algebra has different "grades" of objects:

| Grade | Name | What It Is | Example |
|-------|------|------------|---------|
| **0** | Scalar | Just a number | `5` |
| **1** | Vector | Number with direction | `3·e₁` (3 units in the x direction) |
| **2** | Bivector | Oriented area/plane | `6·e₁₂` (area in the x-y plane) |
| **3** | Trivector | Oriented volume | `2·e₁₂₃` (volume in 3D space) |

**The pattern:**
- Grade 0: point (0D)
- Grade 1: line (1D)
- Grade 2: plane (2D)
- Grade 3: volume (3D)

And you can **add them all together** to make a **multivector**:

```
M = 5 + 3·e₁ + 2·e₂ + 6·e₁₂
    ↑   ↑      ↑      ↑
  scalar  x-dir  y-dir  area
```

---

## The Connection to Boolean Logic (Here's Where It Gets Cool)

Remember Boolean variables? TRUE/FALSE, 1/0?

**What if we represent them as directions?**

```
TRUE  = +1  (pointing forward)
FALSE = -1  (pointing backward)
```

Now each Boolean variable is a **basis vector** that can point forward (+1) or backward (-1).

### Example: Two Boolean Variables

```
P₁ = TRUE  → e₁ pointing at +1
P₂ = FALSE → e₂ pointing at -1
```

In geometric algebra, this is just:
```
state = (1)·e₁ + (-1)·e₂
```

---

## The Magic: AND as Geometric Multiplication

Here's where it clicks. Watch what happens when we multiply Boolean states geometrically:

### Traditional Boolean AND:
```
P₁ AND P₂:
  (TRUE, TRUE)   → TRUE
  (TRUE, FALSE)  → FALSE
  (FALSE, TRUE)  → FALSE
  (FALSE, FALSE) → FALSE
```

### Geometric Algebra Version:

When both are TRUE, we encode the state as:
```
Π(TRUE, TRUE) = [(1 + 1·e₁)/2] · [(1 + 1·e₂)/2]
```

Multiply this out geometrically:
```
= (1/4)·1 + (1/4)·e₁ + (1/4)·e₂ + (1/4)·e₁·e₂
          ↑           ↑            ↑            ↑
       scalar     P₁ bias      P₂ bias    CORRELATION!
```

**Breaking it down:**

| Component | Meaning | Value |
|-----------|---------|-------|
| `1` | "How true is this?" | 0.25 (25%) |
| `e₁` | "P₁ tends to be..." | +0.25 (slightly true) |
| `e₂` | "P₂ tends to be..." | +0.25 (slightly true) |
| `e₁₂` | **"How do P₁ and P₂ relate?"** | **+0.25 (they agree!)** |

That last component—the **bivector** `e₁₂`—is the correlation!

---

## Why Bivectors Encode Correlation

Think about what `e₁₂` means geometrically:

```
e₁₂ = e₁ · e₂
```

It's the **plane** formed by the P₁ axis and the P₂ axis. The bivector measures **how the two axes interact**.

**Positive bivector** (+0.25): Variables **agree** (both true or both false)
**Negative bivector** (-0.50): Variables **disagree** (opposite values)
**Zero bivector** (0.00): Variables are **independent**

### Visual Analogy:

Imagine spinning a wheel:
- Clockwise spin: positive correlation (they move together)
- Counterclockwise spin: negative correlation (they move opposite)
- No spin: no correlation (independent)

**The bivector is the spin!**

---

## Compare AND vs XOR Geometrically

### AND Operation
```
TRUE only when BOTH are true

Geometric form:
0.25·1 + 0.25·e₁ + 0.25·e₂ + 0.25·e₁₂
                              ↑
                      Correlation: +0.25
                      (variables AGREE)
```

### XOR Operation
```
TRUE when inputs DIFFER

Geometric form:
0.50·1 + 0.00·e₁ + 0.00·e₂ - 0.50·e₁₂
                              ↑
                      Correlation: -0.50
                      (variables DISAGREE)
```

**Same geometric structure, different correlations!**

Boolean logic sees: "Two totally different operations"
Geometric algebra sees: "Two points in the same space with different spin"

---

## The Simple Explanation

Here's the elevator pitch:

**Geometric Algebra = Numbers that can point in multiple directions at once**

**Boolean Logic = Special case where directions are ±1**

**Geometric Logic = Using geometric algebra to do Boolean logic**

**The Payoff:**
- Boolean operations become geometric multiplications
- Correlations appear as bivectors (planes/spins)
- All structure is preserved in the multivector
- One multiplication replaces all logic gates

---

## A Step-by-Step Example

Let's compute `P₁ AND P₂` from scratch:

**Step 1: Encode the variables**
```
P₁ can be +1 (true) or -1 (false)
P₂ can be +1 (true) or -1 (false)
```

**Step 2: Create quasi-projectors for each TRUE assignment**

For (P₁=TRUE, P₂=TRUE):
```
Π(+1,+1) = [(1 + (+1)·e₁)/2] · [(1 + (+1)·e₂)/2]
         = [1 + e₁] · [1 + e₂] / 4
         = [1 + e₁ + e₂ + e₁₂] / 4
         = 0.25 + 0.25·e₁ + 0.25·e₂ + 0.25·e₁₂
```

**Step 3: That's it! That's the formula!**

```
AND = 0.25 + 0.25·e₁ + 0.25·e₂ + 0.25·e₁₂
      ↑      ↑         ↑         ↑
    truth   P₁ bias   P₂ bias   correlation
```

**Step 4: Evaluate on any input**

For (P₁=TRUE, P₂=FALSE), substitute e₁=+1, e₂=-1:
```
= 0.25·(1) + 0.25·(+1) + 0.25·(-1) + 0.25·(+1)·(-1)
= 0.25 + 0.25 - 0.25 - 0.25
= 0.00
= FALSE ✓
```

---

## The Mental Model

Think of geometric algebra as **enhanced numbers**:

| Type | What You Get | Example |
|------|--------------|---------|
| **Regular number** | Size | `5` |
| **Vector** | Size + 1 direction | `5·e₁` |
| **Multivector** | Size + all directions + all planes + all volumes | `5 + 3·e₁ + 2·e₁₂` |

**Boolean logic** lives in this space as:
- **Scalars** = truth values
- **Vectors** = variable biases
- **Bivectors** = correlations
- **Trivectors** = 3-way interactions
- etc.

---

## Why This Matters

**Traditional Boolean:**
```
Input → Logic Gates → Output
        ↑
    (correlation lost)
```

**Geometric Boolean:**
```
Input → Geometric Product → Multivector
                            ↑
                    (output + correlation + structure)
```

You get **everything** Boolean logic gives you, PLUS the geometric structure it was hiding.

---

## The Key Insight in One Sentence

**Geometric algebra is what you get when you stop treating vectors as special and start treating them as numbers that multiply in a way that preserves direction.**

And it turns out Boolean logic is just the simplest case: numbers that point at +1 or -1.

---

## Summary: The Progression

1. **Numbers**: Size only
2. **Vectors**: Size + direction
3. **Geometric Algebra**: Unified multiplication that creates scalars, vectors, bivectors, trivectors...
4. **Boolean Variables**: Vectors pointing at ±1
5. **Boolean Operations**: Geometric products of these special vectors
6. **Correlation**: Falls out automatically in the bivector components

**Boolean logic has been geometric all along—we just weren't looking at the full space.**

---

## Try It Yourself (Mental Exercise)

Imagine two light switches:
- Each switch is ON (+1) or OFF (-1)
- Each switch is a direction in space

**Question:** When both switches are ON, are they:
- Moving in the **same direction**? (positive correlation)
- Moving in **opposite directions**? (negative correlation)
- Moving **independently**? (no correlation)

**Answer:** Same direction! They're both pointing at +1.

**That's what the bivector measures.** It's the "sameness" of their directions.

And that's correlation.

And that's what Boolean logic has been missing for 170 years.

---

**Geometric algebra isn't complicated—it's just unfamiliar.**

**Once you see it, it's obvious.**

**And once it's obvious, you can't believe we didn't see it sooner.**