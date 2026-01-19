import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

# Build a visualization of a single 3-clause embedded in a 4-variable hypercube.
# Variables x1,x2,x3 form the 3D subcube; x4 is the "extra dimension" -> two parallel cubes (x4=+1 and x4=-1).
# Clause example: (x1 OR x2 OR x3) under ±1 encoding where True=+1, False=-1.
# Unsatisfied assignment: x1=x2=x3=-1 (independent of x4).

def cube_vertices():
    # vertices for {±1}^3
    vs = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vs.append((x, y, z))
    return np.array(vs, dtype=float)

def cube_edges(vs):
    # edges connect vertices differing in exactly one coordinate
    edges = []
    for i in range(len(vs)):
        for j in range(i+1, len(vs)):
            if np.sum(vs[i] != vs[j]) == 1:
                edges.append((vs[i], vs[j]))
    return edges

V = cube_vertices()
E = cube_edges(V)

# Two slices for x4=+1 and x4=-1; we'll place them side-by-side by translating along X.
offset = 3.0
V_pos = V + np.array([0.0, 0.0, 0.0])
V_neg = V + np.array([offset, 0.0, 0.0])

# Forbidden vertex for clause (x1 OR x2 OR x3): (-1,-1,-1)
forbidden = np.array([-1.0, -1.0, -1.0])
idx_forbidden = np.where((V == forbidden).all(axis=1))[0][0]

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

# Draw edges for both cubes
for a, b in E:
    a1, b1 = a, b
    ax.plot([a1[0], b1[0]], [a1[1], b1[1]], [a1[2], b1[2]])
    a2, b2 = a + np.array([offset, 0.0, 0.0]), b + np.array([offset, 0.0, 0.0])
    ax.plot([a2[0], b2[0]], [a2[1], b2[1]], [a2[2], b2[2]])

# Plot allowed vertices (default marker style)
allowed_pos = np.delete(V_pos, idx_forbidden, axis=0)
allowed_neg = np.delete(V_neg, idx_forbidden, axis=0)
ax.scatter(allowed_pos[:,0], allowed_pos[:,1], allowed_pos[:,2], s=40, marker='o', label='satisfies clause')
ax.scatter(allowed_neg[:,0], allowed_neg[:,1], allowed_neg[:,2], s=40, marker='o')

# Plot forbidden vertices as a different marker (no explicit color)
fp = V_pos[idx_forbidden]
fn = V_neg[idx_forbidden]
ax.scatter([fp[0]], [fp[1]], [fp[2]], s=120, marker='x', label='violates clause')
ax.scatter([fn[0]], [fn[1]], [fn[2]], s=120, marker='x')

# Labels and annotations
ax.text(-0.8, 1.15, 1.15, "x4 = +1 slice", fontsize=10)
ax.text(offset-0.8, 1.15, 1.15, "x4 = -1 slice", fontsize=10)

ax.text(fp[0]-0.35, fp[1]-0.15, fp[2]-0.1, "(-1,-1,-1)", fontsize=9)
ax.text(fn[0]-0.35, fn[1]-0.15, fn[2]-0.1, "(-1,-1,-1)", fontsize=9)

ax.set_xlabel("x1 (and cube offset)")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("A single 3-clause as a 3-cube 'carved out' of a larger hypercube (n=4)")

# Set equal-ish aspect (manual for 3D)
ax.set_box_aspect((2.0, 1.0, 1.0))

ax.legend(loc='upper left')

plt.tight_layout()
out_path = "three_clause_cube_in_hypercube.png"
plt.savefig(out_path, dpi=200)

