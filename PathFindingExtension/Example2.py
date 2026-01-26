import math, heapq, time, random
from collections import deque, defaultdict

# ============================================================
# Basic utilities
# ============================================================

def neighbors4(i, j):
    return [(i, j+1),(i, j-1),(i+1, j),(i-1, j)]

def in_bounds(grid, i, j):
    return 0 <= i < len(grid) and 0 <= j < len(grid[0])

def passable(grid, i, j):
    return grid[i][j] == 1

def d2(a, b):
    # squared distance (no sqrt)
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

# Streaming Bresenham generator (no list creation)
def line_cells(a, b):
    (x0, y0), (x1, y1) = a, b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    if dx >= dy:
        err = dx >> 1
        while x != x1:
            yield (x, y)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        yield (x, y)
    else:
        err = dy >> 1
        while y != y1:
            yield (x, y)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        yield (x, y)

def has_los(grid, a, b):
    for (i, j) in line_cells(a, b):
        if not in_bounds(grid, i, j) or not passable(grid, i, j):
            return False
    return True

def reconstruct_path(came, end):
    path = [end]
    while came[path[-1]] != path[-1]:
        path.append(came[path[-1]])
    path.reverse()
    return path

def reconstruct_unrolled(came, seg, end):
    chain = []
    cur = end
    while True:
        chain.append(cur)
        if came[cur] == cur:
            break
        cur = came[cur]
    chain.reverse()

    out = [(chain[0][0], chain[0][1])]
    for node in chain[1:]:
        out.extend(seg[node])
    return out

# ============================================================
# Baseline A* (4-neighborhood, squared heuristic)
# ============================================================

def a_star(grid, src, dest):
    openpq = []
    came = {src: src}
    g = {src: 0}
    heapq.heappush(openpq, (d2(src, dest), 0, src))
    closed = set()
    expanded = 0

    while openpq:
        f, gc, u = heapq.heappop(openpq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == dest:
            return reconstruct_path(came, u), {"ok": True, "expanded": expanded}

        ui, uj = u
        for v in neighbors4(ui, uj):
            if not in_bounds(grid, v[0], v[1]) or not passable(grid, v[0], v[1]):
                continue
            ng = gc + 1
            if ng < g.get(v, 1<<60):
                g[v] = ng
                came[v] = u
                heapq.heappush(openpq, (ng + d2(v, dest), ng, v))

    return None, {"ok": False, "expanded": expanded}

# ============================================================
# Preprocessing: dist transform + gradient (cached)
# ============================================================

def obstacle_distance_transform(grid):
    R, C = len(grid), len(grid[0])
    dist = [[10**9]*C for _ in range(R)]
    q = deque()
    for i in range(R):
        for j in range(C):
            if grid[i][j] == 0:
                dist[i][j] = 0
                q.append((i,j))
    while q:
        i, j = q.popleft()
        nd = dist[i][j] + 1
        for ni, nj in neighbors4(i, j):
            if in_bounds(grid, ni, nj) and dist[ni][nj] > nd:
                dist[ni][nj] = nd
                q.append((ni,nj))
    return dist

def precompute_gradients(dist):
    R, C = len(dist), len(dist[0])
    gx = [[0]*C for _ in range(R)]
    gy = [[0]*C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            up    = dist[i-1][j] if i-1 >= 0 else dist[i][j]
            down  = dist[i+1][j] if i+1 < R else dist[i][j]
            left  = dist[i][j-1] if j-1 >= 0 else dist[i][j]
            right = dist[i][j+1] if j+1 < C else dist[i][j]
            gx[i][j] = (down - up)
            gy[i][j] = (right - left)
    return gx, gy

# ============================================================
# WBS (Optimized): LOS cache + block cache + streaming rays
# ============================================================

def first_block_on_ray(grid, a, b):
    first = True
    for cell in line_cells(a, b):
        if first:
            first = False
            continue
        i, j = cell
        if not in_bounds(grid, i, j) or not passable(grid, i, j):
            return cell
    return None

def pick_slide_successors_cached(grid, cur, goal, los_goal_cache, max_slide=40):
    """
    Slide left/right relative to goal direction; use los-to-goal cache.
    """
    ci, cj = cur
    gi, gj = goal
    di, dj = gi - ci, gj - cj

    # perpendicular-ish
    left  = (-dj, di)
    right = (dj, -di)

    def step_dir(vec):
        vi, vj = vec
        si = 0 if vi == 0 else (1 if vi > 0 else -1)
        sj = 0 if vj == 0 else (1 if vj > 0 else -1)
        if si == 0 and sj == 0:
            si, sj = 0, 1
        return si, sj

    succs = []
    for step in (step_dir(left), step_dir(right)):
        i, j = ci, cj
        best = None
        best_d = 1<<60
        for _ in range(max_slide):
            ni, nj = i + step[0], j + step[1]
            if not in_bounds(grid, ni, nj):
                break
            if passable(grid, ni, nj):
                key = (ni, nj)
                los = los_goal_cache.get(key)
                if los is None:
                    los = has_los(grid, key, goal)
                    los_goal_cache[key] = los
                if los:
                    return [(ni, nj)]
                dd = d2((ni, nj), goal)
                if dd < best_d:
                    best_d = dd
                    best = (ni, nj)
            i, j = ni, nj
        if best is not None:
            succs.append(best)

    # dedup
    out = []
    seen = set()
    for s in succs:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def wbs_optimized(grid, src, dest):
    los_goal_cache = {}      # (i,j)->bool for LOS to dest
    block_cache = {}         # (i,j)->first blocked cell on ray to dest (or None)

    openpq = []
    start = src
    came = {start: start}
    seg  = {start: []}
    g = {start: 0}
    heapq.heappush(openpq, (d2(start, dest), 0, start))
    closed = set()
    expanded = 0

    while openpq:
        f, gc, u = heapq.heappop(openpq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == dest:
            return reconstruct_unrolled(came, seg, u), {"ok": True, "expanded": expanded}

        # LOS-to-goal cached
        los = los_goal_cache.get(u)
        if los is None:
            los = has_los(grid, u, dest)
            los_goal_cache[u] = los

        # 1) if LOS: jump directly
        if los:
            v = dest
            # stream the segment (excluding u)
            jump_seg = []
            first = True
            for cell in line_cells(u, v):
                if first:
                    first = False
                    continue
                jump_seg.append(cell)
            ng = gc + len(jump_seg)
            if ng < g.get(v, 1<<60):
                g[v] = ng
                came[v] = u
                seg[v] = jump_seg
                heapq.heappush(openpq, (ng + 0, ng, v))
            continue

        # 2) otherwise bypass around first barrier (cached)
        blk = block_cache.get(u)
        if blk is None and u not in block_cache:
            blk = first_block_on_ray(grid, u, dest)
            block_cache[u] = blk

        if blk is not None:
            for v in pick_slide_successors_cached(grid, u, dest, los_goal_cache, max_slide=40):
                if passable(grid, v[0], v[1]):
                    step_seg = []
                    first = True
                    for cell in line_cells(u, v):
                        if first:
                            first = False
                            continue
                        step_seg.append(cell)
                    ng = gc + len(step_seg)
                    if ng < g.get(v, 1<<60):
                        g[v] = ng
                        came[v] = u
                        seg[v] = step_seg
                        heapq.heappush(openpq, (ng + d2(v, dest), ng, v))

        # 3) local fallback (completeness)
        ui, uj = u
        for v in neighbors4(ui, uj):
            if not in_bounds(grid, v[0], v[1]) or not passable(grid, v[0], v[1]):
                continue
            ng = gc + 1
            if ng < g.get(v, 1<<60):
                g[v] = ng
                came[v] = u
                seg[v] = [v]
                heapq.heappush(openpq, (ng + d2(v, dest), ng, v))

    return None, {"ok": False, "expanded": expanded}

# ============================================================
# RG-A* Optimized (bin steering, no trig/atan2 inner loop)
# ============================================================

def rg_astar_optimized(
    grid, src, dest,
    preprocess=None,          # (dist, gx, gy)
    bins=32,
    base_K=18,
    K_tight=6,
    tight_dist=2,
    open_dist=7,
    guided_offsets=(-2, 0, 2),      # in bins
    mini_cone_offsets=(-6, -3, 0, 3, 6),
    full_cone_span_bins=12,         # +/- span
    full_cone_samples=13,
    fallback_budget=6,
    stuck_before_full=2,            # lazy escalation threshold
    repulse_floor=3,
    w_goal=1.0,
    w_rep=1.2,
    turn_gain=5.0,                  # converts steering signal to bin delta
    max_turn_step=3,                # max bins change per step

    no_progress_patience=3,         # early break if we stop improving distance
    progress_dist_w=3.0,            # emphasize goal progress to reduce detours
    progress_clear_w=1.0,
    progress_len_w=0.10
):
    """
    A* over (cell, heading_bin) with macro expansions.
    - steering uses cross/dot between current heading and desired vector (goal + repulsion)
    - no atan2; minimal trig via LUT
    """

    if preprocess is None:
        dist = obstacle_distance_transform(grid)
        gx, gy = precompute_gradients(dist)
    else:
        dist, gx, gy = preprocess

    # Heading LUT (trig only once)
    cos_lut = [math.cos((k+0.5)*2*math.pi/bins - math.pi) for k in range(bins)]
    sin_lut = [math.sin((k+0.5)*2*math.pi/bins - math.pi) for k in range(bins)]

    goal = dest

    # ------------------------------------------------------------
    # NEW: precompute movement per heading bin (4-neighborhood)
    # ------------------------------------------------------------
    move_i = [0]*bins
    move_j = [0]*bins
    for h in range(bins):
        c = cos_lut[h]
        s = sin_lut[h]
        si = 1 if c > 0.25 else (-1 if c < -0.25 else 0)
        sj = 1 if s > 0.25 else (-1 if s < -0.25 else 0)
        if si != 0 and sj != 0:
            # drop weaker component (no abs() recomputation in loop later)
            if (c*c) >= (s*s):
                sj = 0
            else:
                si = 0
        move_i[h] = si
        move_j[h] = sj

    # ------------------------------------------------------------
    # NEW: precompute cardinal bins for O(1) fallback heading update
    # ------------------------------------------------------------
    def best_bin_for_vec(dx, dy):
        bestk = 0
        bestdot = -1e18
        for k in range(bins):
            dot = cos_lut[k]*dx + sin_lut[k]*dy
            if dot > bestdot:
                bestdot = dot
                bestk = k
        return bestk

    BIN_UP    = best_bin_for_vec(-1, 0)
    BIN_DOWN  = best_bin_for_vec( 1, 0)
    BIN_LEFT  = best_bin_for_vec( 0,-1)
    BIN_RIGHT = best_bin_for_vec( 0, 1)

    # initial heading bin: choose best align to goal vector
    dx0 = goal[0] - src[0]
    dy0 = goal[1] - src[1]
    bestk = 0
    bestdot = -1e18
    for k in range(bins):
        d = cos_lut[k]*dx0 + sin_lut[k]*dy0
        if d > bestdot:
            bestdot = d
            bestk = k

    start = (src[0], src[1], bestk)

    def heuristic(state):
        i, j, hb = state
        return d2((i, j), goal)

    def adaptive_K(i, j):
        clearance = dist[i][j]
        if clearance <= tight_dist:
            return K_tight
        if clearance >= open_dist:
            return base_K
        t = (clearance - tight_dist) / max(1, (open_dist - tight_dist))
        return int(K_tight + t * (base_K - K_tight))

    def integrate(state, start_hb, K):
        """
        Returns: end_state, seg_cells, ok, progress_score
        """
        i, j, hb = state
        h = start_hb
        ci, cj = i, j
        seg_cells = []

        start_d = d2((ci, cj), goal)
        best_d = start_d
        bad = 0

        min_clear = dist[ci][cj]

        for _ in range(K):
            dxh = cos_lut[h]
            dyh = sin_lut[h]

            vg_x = (goal[0] - ci)
            vg_y = (goal[1] - cj)

            vr_x = gx[ci][cj]
            vr_y = gy[ci][cj]

            near = repulse_floor - dist[ci][cj]
            if near < 0:
                near = 0
            rep_gain = 1.0 + 0.8 * near

            vx = w_goal * vg_x + (w_rep * rep_gain) * vr_x
            vy = w_goal * vg_y + (w_rep * rep_gain) * vr_y

            dot = dxh * vx + dyh * vy
            cross = dxh * vy - dyh * vx

            denom = 1.0 + (dot if dot >= 0 else -dot)
            turn_signal = cross / denom

            delta = int(round(turn_gain * turn_signal))
            if delta > max_turn_step:
                delta = max_turn_step
            elif delta < -max_turn_step:
                delta = -max_turn_step

            h = (h + delta) % bins

            # ------------------------------------------------
            # NEW: use precomputed move_i/move_j
            # ------------------------------------------------
            ni = ci + move_i[h]
            nj = cj + move_j[h]
            if not in_bounds(grid, ni, nj) or not passable(grid, ni, nj):
                return (ci, cj, h), seg_cells, False, -1e18

            ci, cj = ni, nj
            seg_cells.append((ci, cj))

            dc = dist[ci][cj]
            if dc < min_clear:
                min_clear = dc

            if (ci, cj) == goal:
                break

            # ------------------------------------------------
            # NEW: early break if macro step stops improving
            # ------------------------------------------------
            cur_d = d2((ci, cj), goal)
            if cur_d < best_d:
                best_d = cur_d
                bad = 0
            else:
                bad += 1
                if bad >= no_progress_patience:
                    break

        end_d = d2((ci, cj), goal)

        # NEW: rebalance progress to reduce detours and improve path length
        progress = (
            progress_dist_w * (start_d - end_d)
            + progress_clear_w * min_clear
            + progress_len_w * len(seg_cells)
        )
        return (ci, cj, h), seg_cells, True, progress

    def make_full_cone_offsets():
        if full_cone_samples <= 1:
            return [0]
        span = full_cone_span_bins
        offs = []
        for k in range(full_cone_samples):
            t = (k/(full_cone_samples-1))*2 - 1
            offs.append(int(round(t * span)))
        out = []
        seen = set()
        for o in offs:
            if o not in seen:
                out.append(o)
                seen.add(o)
        return out

    full_cone_offsets = make_full_cone_offsets()

    openpq = []
    came = {start: start}
    seg = {start: []}
    g = {start: 0}
    heapq.heappush(openpq, (heuristic(start), 0, start))
    closed = set()
    expanded = 0

    stuck_count = defaultdict(int)

    while openpq:
        f, gc, u = heapq.heappop(openpq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1

        ui, uj, hb = u
        if (ui, uj) == goal:
            return reconstruct_unrolled(came, seg, u), {"ok": True, "expanded": expanded}

        K = adaptive_K(ui, uj)

        candidates = []

        # 1) guided macro steps
        for off in guided_offsets:
            start_h = (hb + off) % bins
            v, seg_cells, ok, prog = integrate(u, start_h, K)
            if ok and seg_cells:
                candidates.append((v, seg_cells, len(seg_cells), prog))

        # 2) if guided failed, do mini-cone first
        if not candidates:
            stuck_count[u] += 1
            cone_set = mini_cone_offsets if stuck_count[u] < stuck_before_full else full_cone_offsets

            best_heap = []
            need = fallback_budget

            for off in cone_set:
                start_h = (hb + off) % bins
                v, seg_cells, ok, prog = integrate(u, start_h, K)
                if ok and seg_cells:
                    if len(best_heap) < need:
                        heapq.heappush(best_heap, (prog, v, seg_cells))
                    else:
                        if prog > best_heap[0][0]:
                            heapq.heapreplace(best_heap, (prog, v, seg_cells))

            best_heap.sort(reverse=True, key=lambda x: x[0])
            for prog, v, seg_cells in best_heap:
                candidates.append((v, seg_cells, len(seg_cells), prog))

        # 3) completeness fallback: micro steps, O(1) heading updates
        if not candidates:
            for vi, vj in neighbors4(ui, uj):
                if not in_bounds(grid, vi, vj) or not passable(grid, vi, vj):
                    continue
                if vi == ui - 1:
                    nh = BIN_UP
                elif vi == ui + 1:
                    nh = BIN_DOWN
                elif vj == uj - 1:
                    nh = BIN_LEFT
                else:
                    nh = BIN_RIGHT
                v = (vi, vj, nh)
                candidates.append((v, [(vi, vj)], 1, 0.0))

            # boxed: allow turn-in-place
            if not candidates:
                for delta in (-1, 1):
                    v = (ui, uj, (hb + delta) % bins)
                    candidates.append((v, [], 1, -0.1))

        for v, seg_cells, cost, prog in candidates:
            ng = gc + cost
            if ng < g.get(v, 1 << 60):
                g[v] = ng
                came[v] = u
                seg[v] = seg_cells
                heapq.heappush(openpq, (ng + heuristic(v), ng, v))

    return None, {"ok": False, "expanded": expanded}


# ============================================================
# Map generators (variadic N x M)
# ============================================================

def corridor_map(n, m, seed=0):
    rng = random.Random(seed)
    grid = [[0]*m for _ in range(n)]
    for i in range(1, n-1, 4):
        for j in range(1, m-1):
            grid[i][j] = 1
            if rng.random() < 0.10 and i+1 < n-1:
                grid[i+1][j] = 1
    for j in range(2, m-2, 6):
        for i in range(1, n-1):
            grid[i][j] = 1
            if rng.random() < 0.08 and j+1 < m-1:
                grid[i][j+1] = 1
    for _ in range(max(3, (n*m)//500)):
        h = rng.randint(3, 7)
        w = rng.randint(3, 9)
        i0 = rng.randint(1, n-h-1)
        j0 = rng.randint(1, m-w-1)
        for i in range(i0, i0+h):
            for j in range(j0, j0+w):
                grid[i][j] = 1
    return grid

def ensure_solvable(grid, src, dest, tries=60, seed=0):
    rng = random.Random(seed)
    n, m = len(grid), len(grid[0])
    for _ in range(tries):
        grid[src[0]][src[1]] = 1
        grid[dest[0]][dest[1]] = 1
        _, st = a_star(grid, src, dest)
        if st.get("ok"):
            return grid
        # punch holes
        for _ in range((n*m)//200 + 8):
            grid[rng.randrange(n)][rng.randrange(m)] = 1
    return grid

# ============================================================
# Benchmark helpers
# ============================================================

def run_one(label, fn):
    t0 = time.perf_counter()
    path, st = fn()
    t1 = time.perf_counter()
    return {
        "label": label,
        "ok": st.get("ok", False),
        "expanded": st.get("expanded"),
        "time_ms": (t1 - t0) * 1000.0,
        "path_len": None if path is None else len(path),
    }

def bench(label, fn, iters=30):
    t0 = time.perf_counter()
    exp = 0
    ok = 0
    plen = 0
    for _ in range(iters):
        path, st = fn()
        ok += 1 if st.get("ok") else 0
        exp += st.get("expanded", 0)
        plen += 0 if path is None else len(path)
    t1 = time.perf_counter()
    return {
        "label": label,
        "iters": iters,
        "ok": ok,
        "avg_expanded": exp/iters,
        "avg_path_len": plen/iters if ok else None,
        "ms_per_iter": (t1 - t0) * 1000.0 / iters,
        "iters_per_sec": iters / (t1 - t0),
    }

# ============================================================
# Demo: plug any N x M here
# ============================================================

if __name__ == "__main__":
    n, m = 800, 600
    src = (n-2, 1)
    dest = (1, m-2)

    grid = corridor_map(n, m, seed=7)
    grid = ensure_solvable(grid, src, dest, seed=7)

    # preprocess once (important!)
    dist = obstacle_distance_transform(grid)
    gx, gy = precompute_gradients(dist)
    prep = (dist, gx, gy)

    print(f"n x m map: {n} x {m}")

    # Single-run
    print("Single Run")
    print(run_one("A*", lambda: a_star(grid, src, dest)))
    print(run_one("WBS(opt)", lambda: wbs_optimized(grid, src, dest)))
    print(run_one("RG-A*(opt)", lambda: rg_astar_optimized(grid, src, dest, preprocess=prep)))

    # Benchmark
    print("Benchmark")
    print(bench("A*", lambda: a_star(grid, src, dest), iters=40))
    print(bench("WBS(opt)", lambda: wbs_optimized(grid, src, dest), iters=40))
    print(bench("RG-A*(opt)", lambda: rg_astar_optimized(grid, src, dest, preprocess=prep), iters=40))
