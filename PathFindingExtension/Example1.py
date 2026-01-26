import math, heapq, time
from dataclasses import dataclass

@dataclass
class Cell:
    parent_i: int = 0
    parent_j: int = 0
    f: float = float("inf")
    g: float = float("inf")
    h: float = 0.0

def is_valid(row, col, ROW, COL):
    return 0 <= row < ROW and 0 <= col < COL

def is_unblocked(grid, row, col):
    return grid[row][col] == 1

def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

def trace_path(cell_details, dest):
    path = []
    row, col = dest
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        row, col = cell_details[row][col].parent_i, cell_details[row][col].parent_j
    path.append((row, col))
    path.reverse()
    return path

# -----------------------
# Baseline A*
# -----------------------
def a_star_baseline(grid, src, dest, diagonal=True):
    ROW, COL = len(grid), len(grid[0])

    if not is_valid(src[0], src[1], ROW, COL) or not is_valid(dest[0], dest[1], ROW, COL):
        return None, {"ok": False, "reason": "invalid", "expanded": 0}
    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
        return None, {"ok": False, "reason": "blocked", "expanded": 0}
    if is_destination(src[0], src[1], dest):
        return [tuple(src)], {"ok": True, "expanded": 0}

    closed = [[False]*COL for _ in range(ROW)]
    cells = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    i, j = src
    cells[i][j].f = cells[i][j].g = cells[i][j].h = 0.0
    cells[i][j].parent_i = i
    cells[i][j].parent_j = j

    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    dirs = [(0,1),(0,-1),(1,0),(-1,0)]
    if diagonal:
        dirs += [(1,1),(1,-1),(-1,1),(-1,-1)]

    expanded = 0

    while open_list:
        f, i, j = heapq.heappop(open_list)
        if closed[i][j]:
            continue
        closed[i][j] = True
        expanded += 1

        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if not is_valid(ni, nj, ROW, COL) or not is_unblocked(grid, ni, nj) or closed[ni][nj]:
                continue

            if is_destination(ni, nj, dest):
                cells[ni][nj].parent_i = i
                cells[ni][nj].parent_j = j
                return trace_path(cells, dest), {"ok": True, "expanded": expanded}

            g_new = cells[i][j].g + 1.0
            h_new = math.hypot(ni - dest[0], nj - dest[1])
            f_new = g_new + h_new

            if cells[ni][nj].f == float("inf") or cells[ni][nj].f > f_new:
                heapq.heappush(open_list, (f_new, ni, nj))
                c = cells[ni][nj]
                c.f, c.g, c.h = f_new, g_new, h_new
                c.parent_i, c.parent_j = i, j

    return None, {"ok": False, "reason": "no_path", "expanded": expanded}

# -----------------------
# GA-A* in Cl(2,0)
# MV = (scalar, e1, e2, e12)
# Position is a pure vector: (0, i, j, 0)
# Heuristic h = ||pos - goal|| computed via Clifford norm
# -----------------------
def mv_sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3])

def gp_cl20(A, B):
    a0,a1,a2,a12 = A
    b0,b1,b2,b12 = B
    # Cl(2,0): e1^2=+1, e2^2=+1, e12^2=-1
    c0  = a0*b0 + a1*b1 + a2*b2 - a12*b12
    c1  = a0*b1 + a1*b0 - a2*b12 + a12*b2
    c2  = a0*b2 + a2*b0 + a1*b12 - a12*b1
    c12 = a0*b12 + a12*b0 + a1*b2 - a2*b1
    return (c0,c1,c2,c12)

def reverse_cl20(A):
    # reverse flips the bivector sign
    return (A[0], A[1], A[2], -A[3])

def ga_norm(A):
    # ||A|| = sqrt( <A * ~A>_0 )
    P = gp_cl20(A, reverse_cl20(A))
    s = P[0]
    if s < 0:
        s = 0.0
    return math.sqrt(s)

def pos_to_mv(i, j):
    return (0.0, float(i), float(j), 0.0)

def a_star_ga(grid, src, dest, diagonal=True):
    ROW, COL = len(grid), len(grid[0])

    if not is_valid(src[0], src[1], ROW, COL) or not is_valid(dest[0], dest[1], ROW, COL):
        return None, {"ok": False, "reason": "invalid", "expanded": 0}
    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
        return None, {"ok": False, "reason": "blocked", "expanded": 0}
    if is_destination(src[0], src[1], dest):
        return [tuple(src)], {"ok": True, "expanded": 0}

    closed = [[False]*COL for _ in range(ROW)]
    cells = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    i, j = src
    cells[i][j].f = cells[i][j].g = cells[i][j].h = 0.0
    cells[i][j].parent_i = i
    cells[i][j].parent_j = j

    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    dirs = [(0,1),(0,-1),(1,0),(-1,0)]
    if diagonal:
        dirs += [(1,1),(1,-1),(-1,1),(-1,-1)]

    goal_mv = pos_to_mv(dest[0], dest[1])
    expanded = 0

    while open_list:
        f, i, j = heapq.heappop(open_list)
        if closed[i][j]:
            continue
        closed[i][j] = True
        expanded += 1

        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if not is_valid(ni, nj, ROW, COL) or not is_unblocked(grid, ni, nj) or closed[ni][nj]:
                continue

            if is_destination(ni, nj, dest):
                cells[ni][nj].parent_i = i
                cells[ni][nj].parent_j = j
                return trace_path(cells, dest), {"ok": True, "expanded": expanded}

            g_new = cells[i][j].g + 1.0
            d_mv = mv_sub(pos_to_mv(ni, nj), goal_mv)
            h_new = ga_norm(d_mv)
            f_new = g_new + h_new

            if cells[ni][nj].f == float("inf") or cells[ni][nj].f > f_new:
                heapq.heappush(open_list, (f_new, ni, nj))
                c = cells[ni][nj]
                c.f, c.g, c.h = f_new, g_new, h_new
                c.parent_i, c.parent_j = i, j

    return None, {"ok": False, "reason": "no_path", "expanded": expanded}

# -----------------------
# Your grid + benchmark
# -----------------------
grid = [
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]
]
src = (8, 0)
dest = (0, 0)

# correctness
p0, s0 = a_star_baseline(grid, src, dest)
p1, s1 = a_star_ga(grid, src, dest)
print("Baseline:", s0, "path_len:", None if p0 is None else len(p0))
print("GA-A*   :", s1, "path_len:", None if p1 is None else len(p1))
print("Same length?", (p0 is not None and p1 is not None and len(p0)==len(p1)))

def bench(fn, iters=5000):
    t0 = time.perf_counter()
    expanded_total = 0
    ok = 0
    for _ in range(iters):
        _, stats = fn(grid, src, dest)
        ok += 1 if stats.get("ok") else 0
        expanded_total += stats.get("expanded", 0)
    t1 = time.perf_counter()
    return {
        "iters": iters,
        "seconds": t1 - t0,
        "iters_per_sec": iters / (t1 - t0),
        "ok": ok,
        "avg_expanded": expanded_total / iters
    }

res_base = bench(a_star_baseline, iters=5000)
res_ga   = bench(a_star_ga, iters=5000)
print("GA-A* vs Baseline A* | Benchmark")
print("\n--- Speed (same grid, 5000 runs) ---")
print("Baseline:", res_base)
print("GA-A*   :", res_ga)
print("Slowdown factor (GA vs baseline):", res_base["iters_per_sec"] / res_ga["iters_per_sec"])
