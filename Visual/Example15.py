"""
Demo 15: Practical Use Case - Query Similarity Detection
Using geometric operations for real problems
"""

from r200 import R200
import numpy as np


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
print("PRACTICAL USE CASE: DATABASE QUERY SIMILARITY")
print("=" * 80)
print("\nScenario: You have a database query cache")
print("Goal: Find if new query is similar to cached ones")
print("=" * 80)

# Represent queries as Boolean formulas
# P1 = "user is premium", P2 = "age > 18"

queries = {
    "Premium adults": [(True, True)],  # premium AND adult
    "All adults": [(True, True), (False, True)],  # P2 (age > 18)
    "Premium users": [(True, True), (True, False)],  # P1 (premium)
    "Free adults": [(False, True)],  # NOT premium AND adult
    "Anyone": [(True, True), (True, False), (False, True), (False, False)],  # tautology
}

# Embed all queries
embedded = {name: embed_formula(sat) for name, sat in queries.items()}

# New query arrives
new_query = embed_formula([(True, True)])  # Premium adults
print("\n🔍 New Query: 'Premium adults'")
print(f"   Embedded: {new_query}")

# Find similar queries using INNER PRODUCT
print("\n📊 Similarity Analysis (using Inner Product A | B):")
print("-" * 80)

similarities = []
for name, query_mv in embedded.items():
    # Compute inner product
    inner = new_query | query_mv
    similarity = get_components(inner)[0]  # Scalar component
    similarities.append((name, similarity, query_mv))

    print(f"{name:20}: similarity = {similarity:+.3f}", end="")
    if similarity > 0.9:
        print(" ← VERY SIMILAR! Use cached result!")
    elif similarity > 0.5:
        print(" ← Somewhat similar")
    elif similarity > 0:
        print(" ← Slightly related")
    else:
        print(" ← Unrelated")

# Find best match
best_match = max(similarities, key=lambda x: x[1])
print(f"\n🎯 Best match: '{best_match[0]}' (similarity: {best_match[1]:.3f})")

print("\n💡 BENEFIT:")
print("   Traditional: Would need to execute query to compare results")
print("   Geometric: Know similarity INSTANTLY from structure alone!")
print("   → Can reuse cached results for similar queries")
print("   → Massive performance improvement!")

print("\n" + "=" * 80)