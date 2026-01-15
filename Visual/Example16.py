"""
Demo 16: From Boolean to Geometric - A Mental Journey
Five stages of understanding
"""

from r200 import R200
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_components(mv):
    return np.array([mv[0], mv[1], mv[2], mv[3]])


def embed_formula(truth_table):
    result = R200(0, 0)
    for p1, p2 in truth_table:
        s1 = 1 if p1 else -1
        s2 = 1 if p2 else -1
        factor1 = (R200(1, 0) + R200(s1, 1)) * 0.5
        factor2 = (R200(1, 0) + R200(s2, 2)) * 0.5
        result = result + (factor1 * factor2)
    return result


print("=" * 80)
print("THE PARADIGM SHIFT: FROM BOOLEAN TO GEOMETRIC THINKING")
print("=" * 80)
print("\nA guided journey through 5 stages of understanding")
print("=" * 80)

# =============================================================================
# STAGE 1: Boolean Thinking - "Is it true?"
# =============================================================================
print("\n" + "=" * 80)
print("STAGE 1: BOOLEAN THINKING - 'Is it true?'")
print("=" * 80)

print("""
In Boolean logic, you ask:
  • Is this formula TRUE or FALSE?
  • Can I simplify this to fewer operations?
  • What's the truth table?

Example: P1 AND P2
""")

# Boolean evaluation
for p1, p2 in [(True, True), (True, False), (False, True), (False, False)]:
    result = p1 and p2
    print(f"  P1={p1:5}, P2={p2:5} → {result:5}")

print("""
This is EVALUATION thinking:
  • Plug in values
  • Get answer
  • That's it

Limitation: You lose all context once you get the answer.
""")

# =============================================================================
# STAGE 2: Probabilistic Thinking - "How often is it true?"
# =============================================================================
print("\n" + "=" * 80)
print("STAGE 2: PROBABILISTIC THINKING - 'How often is it true?'")
print("=" * 80)

print("""
First upgrade: Think about PROBABILITY, not just truth.

Example: P1 AND P2
  • True in 1 out of 4 cases
  • Probability: 25%
  • This is the SCALAR component
""")

and_formula = embed_formula([(True, True)])
and_comps = get_components(and_formula)

print(f"\nP1 ∧ P2:")
print(f"  Boolean view: true/false (depends on inputs)")
print(f"  Probabilistic view: true 25% of the time")
print(f"  Geometric scalar: {and_comps[0]:.3f}")

print("""
This is STATISTICAL thinking:
  • How likely is this?
  • What's the expected value?

Improvement: You keep information (probability)
Limitation: Still just ONE number
""")

# =============================================================================
# STAGE 3: Relational Thinking - "How do variables relate?"
# =============================================================================
print("\n" + "=" * 80)
print("STAGE 3: RELATIONAL THINKING - 'How do variables relate?'")
print("=" * 80)

print("""
Second upgrade: Think about RELATIONSHIPS between variables.

Key insight: Variables aren't independent!
  • Do P1 and P2 tend to be true TOGETHER?
  • Or do they OPPOSE each other?
  • Are they INDEPENDENT?

This is the BIVECTOR component (e12)
""")

formulas = {
    'AND': [(True, True)],
    'OR': [(True, True), (True, False), (False, True)],
    'XOR': [(True, False), (False, True)],
    'IFF': [(True, True), (False, False)],
}

print("\nComparing formulas by CORRELATION:")
print("-" * 80)

for name, sat in formulas.items():
    mv = embed_formula(sat)
    comps = get_components(mv)

    prob = comps[0]
    corr = comps[3]

    print(f"{name:6}: Probability={prob:.2%}, Correlation={corr:+.3f}", end="")

    if corr > 0.3:
        print(" ← Variables AGREE")
    elif corr < -0.3:
        print(" ← Variables DISAGREE")
    else:
        print(" ← Independent")

print("""
This is CORRELATION thinking:
  • What's the relationship structure?
  • How do parts interact?

Improvement: TWO numbers (probability + correlation)
Limitation: Still not seeing the full geometry
""")

# =============================================================================
# STAGE 4: Spatial Thinking - "Where is it in space?"
# =============================================================================
print("\n" + "=" * 80)
print("STAGE 4: SPATIAL THINKING - 'Where is it in space?'")
print("=" * 80)

print("""
Third upgrade: Think GEOMETRICALLY.

Key insight: Formulas are POINTS in a 4D space!
  • Axis 1 (scalar): How true?
  • Axis 2 (e1): P1 bias
  • Axis 3 (e2): P2 bias  
  • Axis 4 (e12): Correlation

Each formula has a POSITION in this space.
""")

# Visualize in 3D (project 4D to 3D)
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')

colors = {'AND': 'red', 'OR': 'blue', 'XOR': 'green', 'IFF': 'purple'}
for name, sat in formulas.items():
    mv = embed_formula(sat)
    comps = get_components(mv)

    # Plot in (scalar, e1, e2) space
    ax1.scatter(comps[0], comps[1], comps[2],
                s=300, c=colors[name], marker='o',
                edgecolors='black', linewidths=2, label=name)

    # Label
    ax1.text(comps[0], comps[1], comps[2], f'  {name}', fontsize=9)

ax1.set_xlabel('Scalar (Truth)')
ax1.set_ylabel('e₁ (P₁ bias)')
ax1.set_zlabel('e₂ (P₂ bias)')
ax1.set_title('Formulas as Points in 3D\n(4th dimension e₁₂ not shown)',
              fontweight='bold')
ax1.legend()

# 2D view: (scalar, e12)
ax2 = fig.add_subplot(122)

for name, sat in formulas.items():
    mv = embed_formula(sat)
    comps = get_components(mv)

    ax2.scatter(comps[0], comps[3],
                s=300, c=colors[name], marker='o',
                edgecolors='black', linewidths=2, label=name)

    ax2.annotate(name, (comps[0], comps[3]),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=10, fontweight='bold')

ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax2.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax2.set_xlabel('Scalar (Truth Probability)', fontsize=11)
ax2.set_ylabel('e₁₂ (Correlation)', fontsize=11)
ax2.set_title('Truth-Correlation Space', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('demo16_spatial_thinking.png', dpi=150)
print("\n✅ Saved visualization: demo16_spatial_thinking.png")

print("""
This is GEOMETRIC thinking:
  • Formulas are POINTS
  • Operations are MOVEMENTS
  • Distance = similarity
  • Angle = relationship type

Improvement: Full 4D vector representation
Key insight: Can measure DISTANCE between formulas!
""")

# Compute distances
print("\nDistances between formulas:")
print("-" * 80)

formula_pairs = [
    ('AND', 'OR'),
    ('AND', 'XOR'),
    ('XOR', 'IFF'),
]

for name1, name2 in formula_pairs:
    mv1 = embed_formula(formulas[name1])
    mv2 = embed_formula(formulas[name2])
    comps1 = get_components(mv1)
    comps2 = get_components(mv2)

    distance = np.linalg.norm(comps1 - comps2)

    print(f"  {name1:6} ↔ {name2:6}: {distance:.3f}")

print("""
Limitation: Still thinking of formulas as STATIC objects
""")

# =============================================================================
# STAGE 5: Dynamic Thinking - "How does it transform?"
# =============================================================================
print("\n" + "=" * 80)
print("STAGE 5: DYNAMIC THINKING - 'How does it transform?'")
print("=" * 80)

print("""
Final upgrade: Think about TRANSFORMATIONS.

Key insight: Operations are MOVEMENTS in space!
  • Geometric product = combination
  • Reverse = reflection
  • Sandwich = rotation
  • Interpolation = smooth path

Formulas aren't just points - they can FLOW into each other!
""")

# Show transformation
AND = embed_formula([(True, True)])
XOR = embed_formula([(True, False), (False, True)])

print("\nExample: Continuous transformation AND → XOR")
print("-" * 80)

trajectory = []
for t in np.linspace(0, 1, 6):
    t = float(t)
    # Linear interpolation
    interp = AND * (1 - t) + XOR * t
    comps = get_components(interp)
    trajectory.append(comps)

    print(f"t={t:.1f}: scalar={comps[0]:.3f}, e12={comps[3]:+.3f}")

print("""
This is DYNAMIC thinking:
  • Formulas can TRANSFORM
  • Operations are VERBS not NOUNS
  • Smooth transitions exist
  • Infinite intermediate states

This is the FULL geometric perspective!
""")

plt.show()

# =============================================================================
# SUMMARY: The Mental Shift
# =============================================================================
print("\n" + "=" * 80)
print("THE MENTAL SHIFT SUMMARIZED")
print("=" * 80)

summary = """
┌──────────┬────────────────────┬────────────────────┬──────────────────────┐
│ Stage    │ Question           │ What You See       │ Dimension            │
├──────────┼────────────────────┼────────────────────┼──────────────────────┤
│ Boolean  │ "Is it true?"      │ Binary answer      │ 0D (point: T/F)      │
├──────────┼────────────────────┼────────────────────┼──────────────────────┤
│ Prob.    │ "How often true?"  │ Single number      │ 1D (line: [0,1])     │
├──────────┼────────────────────┼────────────────────┼──────────────────────┤
│ Relation │ "How do vars       │ Probability +      │ 2D (plane)           │
│          │  relate?"          │ correlation        │                      │
├──────────┼────────────────────┼────────────────────┼──────────────────────┤
│ Spatial  │ "Where in space?"  │ Point in 4D space  │ 4D (full multivec)   │
├──────────┼────────────────────┼────────────────────┼──────────────────────┤
│ Dynamic  │ "How does it       │ Flowing geometry   │ 4D + time            │
│          │  transform?"       │                    │                      │
└──────────┴────────────────────┴────────────────────┴──────────────────────┘

THE CORE SHIFT:

Boolean Mind:               Geometric Mind:
--------------             -----------------
"True or false?"     →     "Where in space?"
Discrete (16 ops)    →     Continuous (∞ ops)
Evaluation           →     Transformation
Static               →     Dynamic
One answer           →     Full structure
Operations destroy   →     Operations preserve
context                    information
"""

print(summary)

print("\n" + "=" * 80)
print("EXERCISES TO BUILD GEOMETRIC INTUITION")
print("=" * 80)

exercises = """
1. VISUALIZATION EXERCISE
   Draw formulas as POINTS in (truth, correlation) space
   - AND is in bottom-right (low truth, positive correlation)
   - OR is in top-left (high truth, negative correlation)
   Practice until you can "see" formulas spatially

2. DISTANCE EXERCISE
   Before computing, ESTIMATE similarity:
   - AND vs NAND: Same correlation, opposite truth → moderate distance
   - XOR vs IFF: Opposite correlation, same truth → large distance
   - P1 vs P2: No correlation → orthogonal

3. TRANSFORMATION EXERCISE
   Pick two formulas. Imagine smoothly transforming one into the other.
   What intermediate states exist? Draw the path.

4. OPERATION EXERCISE
   Don't think "A * B computes result"
   Think "A * B combines geometric structures"
   - Product = merge + interaction
   - Inner = projection (similarity)
   - Outer = independent combination

5. BIVECTOR INTUITION
   Train yourself to "feel" correlation:
   - Positive e12 = "tends to agree"
   - Negative e12 = "tends to disagree"
   - Zero e12 = "independent"
   When you see a formula, immediately ask: "What's the correlation?"
"""

print(exercises)

print("\n" + "=" * 80)
print("THE 'AHA MOMENT' TRIGGERS")
print("=" * 80)

aha_moments = """
Watch for these insights that signal you're "getting it":

✨ Moment 1: "Wait, formulas are LOCATIONS not answers!"
   You stop thinking about evaluating and start thinking about positioning.

✨ Moment 2: "Operations are MOVEMENTS!"
   You see A*B as "combining two positions" not "computing result"

✨ Moment 3: "There's stuff BETWEEN the Boolean operations!"
   You realize the 16 Boolean ops are just SAMPLES of infinite space

✨ Moment 4: "Correlation is everywhere!"
   You can't unsee the bivector. Every formula screams its correlation.

✨ Moment 5: "Boolean logic is just the scalar projection!"
   You see Boolean as the SHADOW of something higher-dimensional

✨ Moment 6: "I can INTERPOLATE formulas!"
   The discrete becomes continuous. Mind = blown.

✨ Moment 7: "This is like complex numbers but for logic!"
   You map it to familiar math: i² = -1 like e12² = -1

When these click, you're thinking geometrically! 🎯
"""

print(aha_moments)