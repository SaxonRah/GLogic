import pygame
import math
import heapq
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================
# PATHFINDING CORE (from your optimized code)
# ============================================================

def neighbors4(i, j):
    return [(i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j)]


def in_bounds(grid, i, j):
    return 0 <= i < len(grid) and 0 <= j < len(grid[0])


def passable(grid, i, j):
    return grid[i][j] == 1


def d2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


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


# A*
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
            if ng < g.get(v, 1 << 60):
                g[v] = ng
                came[v] = u
                heapq.heappush(openpq, (ng + d2(v, dest), ng, v))

    return None, {"ok": False, "expanded": expanded}


# WBS
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


def pick_slide_successors(grid, cur, goal, los_cache, max_slide=30):
    ci, cj = cur
    gi, gj = goal
    di, dj = gi - ci, gj - cj
    left = (-dj, di)
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
        best_d = 1 << 60
        for _ in range(max_slide):
            ni, nj = i + step[0], j + step[1]
            if not in_bounds(grid, ni, nj):
                break
            if passable(grid, ni, nj):
                key = (ni, nj)
                los = los_cache.get(key)
                if los is None:
                    los = has_los(grid, key, goal)
                    los_cache[key] = los
                if los:
                    return [(ni, nj)]
                dd = d2((ni, nj), goal)
                if dd < best_d:
                    best_d = dd
                    best = (ni, nj)
            i, j = ni, nj
        if best is not None:
            succs.append(best)

    out, seen = [], set()
    for s in succs:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def wbs_optimized(grid, src, dest):
    los_cache = {}
    block_cache = {}

    openpq = []
    came = {src: src}
    seg = {src: []}
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
            return reconstruct_unrolled(came, seg, u), {"ok": True, "expanded": expanded}

        los = los_cache.get(u)
        if los is None:
            los = has_los(grid, u, dest)
            los_cache[u] = los

        if los:
            v = dest
            jump_seg = []
            first = True
            for cell in line_cells(u, v):
                if first:
                    first = False
                    continue
                jump_seg.append(cell)
            ng = gc + len(jump_seg)
            if ng < g.get(v, 1 << 60):
                g[v] = ng
                came[v] = u
                seg[v] = jump_seg
                heapq.heappush(openpq, (ng, ng, v))
            continue

        blk = block_cache.get(u)
        if blk is None and u not in block_cache:
            blk = first_block_on_ray(grid, u, dest)
            block_cache[u] = blk

        if blk is not None:
            for v in pick_slide_successors(grid, u, dest, los_cache, max_slide=30):
                if passable(grid, v[0], v[1]):
                    step_seg = []
                    first = True
                    for cell in line_cells(u, v):
                        if first:
                            first = False
                            continue
                        step_seg.append(cell)
                    ng = gc + len(step_seg)
                    if ng < g.get(v, 1 << 60):
                        g[v] = ng
                        came[v] = u
                        seg[v] = step_seg
                        heapq.heappush(openpq, (ng + d2(v, dest), ng, v))

        ui, uj = u
        for v in neighbors4(ui, uj):
            if not in_bounds(grid, v[0], v[1]) or not passable(grid, v[0], v[1]):
                continue
            ng = gc + 1
            if ng < g.get(v, 1 << 60):
                g[v] = ng
                came[v] = u
                seg[v] = [v]
                heapq.heappush(openpq, (ng + d2(v, dest), ng, v))

    return None, {"ok": False, "expanded": expanded}


# Preprocessing for RG-A*
def obstacle_distance_transform(grid):
    R, C = len(grid), len(grid[0])
    dist = [[10 ** 9] * C for _ in range(R)]
    q = deque()
    for i in range(R):
        for j in range(C):
            if grid[i][j] == 0:
                dist[i][j] = 0
                q.append((i, j))
    while q:
        i, j = q.popleft()
        nd = dist[i][j] + 1
        for ni, nj in neighbors4(i, j):
            if in_bounds(grid, ni, nj) and dist[ni][nj] > nd:
                dist[ni][nj] = nd
                q.append((ni, nj))
    return dist


def precompute_gradients(dist):
    R, C = len(dist), len(dist[0])
    gx = [[0] * C for _ in range(R)]
    gy = [[0] * C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            up = dist[i - 1][j] if i - 1 >= 0 else dist[i][j]
            down = dist[i + 1][j] if i + 1 < R else dist[i][j]
            left = dist[i][j - 1] if j - 1 >= 0 else dist[i][j]
            right = dist[i][j + 1] if j + 1 < C else dist[i][j]
            gx[i][j] = (down - up)
            gy[i][j] = (right - left)
    return gx, gy


# RG-A* (simplified for demo - key parts only)
def rg_astar_simple(grid, src, dest, preprocess=None):
    if preprocess is None:
        dist = obstacle_distance_transform(grid)
        gx, gy = precompute_gradients(dist)
    else:
        dist, gx, gy = preprocess

    bins = 24
    K = 12

    cos_lut = [math.cos((k + 0.5) * 2 * math.pi / bins - math.pi) for k in range(bins)]
    sin_lut = [math.sin((k + 0.5) * 2 * math.pi / bins - math.pi) for k in range(bins)]

    move_i = [0] * bins
    move_j = [0] * bins
    for h in range(bins):
        c, s = cos_lut[h], sin_lut[h]
        si = 1 if c > 0.25 else (-1 if c < -0.25 else 0)
        sj = 1 if s > 0.25 else (-1 if s < -0.25 else 0)
        if si != 0 and sj != 0:
            if c * c >= s * s:
                sj = 0
            else:
                si = 0
        move_i[h] = si
        move_j[h] = sj

    def best_bin(dx, dy):
        best_k, best_dot = 0, -1e18
        for k in range(bins):
            dot = cos_lut[k] * dx + sin_lut[k] * dy
            if dot > best_dot:
                best_dot = dot
                best_k = k
        return best_k

    init_bin = best_bin(dest[0] - src[0], dest[1] - src[1])
    start = (src[0], src[1], init_bin)

    def heuristic(state):
        return d2((state[0], state[1]), dest)

    def integrate(state, start_h):
        i, j, _ = state
        h = start_h
        ci, cj = i, j
        seg = []

        for _ in range(K):
            dxh, dyh = cos_lut[h], sin_lut[h]
            vg_x, vg_y = dest[0] - ci, dest[1] - cj
            vr_x, vr_y = gx[ci][cj], gy[ci][cj]

            near = max(0, 3 - dist[ci][cj])
            rep_gain = 1.0 + 0.8 * near

            vx = vg_x + 1.2 * rep_gain * vr_x
            vy = vg_y + 1.2 * rep_gain * vr_y

            cross = dxh * vy - dyh * vx
            dot = dxh * vx + dyh * vy
            turn = cross / (1.0 + abs(dot))
            delta = max(-2, min(2, int(round(4.0 * turn))))

            h = (h + delta) % bins

            ni, nj = ci + move_i[h], cj + move_j[h]
            if not in_bounds(grid, ni, nj) or not passable(grid, ni, nj):
                return (ci, cj, h), seg, False

            ci, cj = ni, nj
            seg.append((ci, cj))
            if (ci, cj) == dest:
                break

        return (ci, cj, h), seg, True

    openpq = []
    came = {start: start}
    seg_map = {start: []}
    g = {start: 0}
    heapq.heappush(openpq, (heuristic(start), 0, start))
    closed = set()
    expanded = 0

    while openpq:
        f, gc, u = heapq.heappop(openpq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1

        ui, uj, hb = u
        if (ui, uj) == dest:
            return reconstruct_unrolled(came, seg_map, u), {"ok": True, "expanded": expanded}

        candidates = []
        for off in [-1, 0, 1]:
            v, seg, ok = integrate(u, (hb + off) % bins)
            if ok and seg:
                candidates.append((v, seg, len(seg)))

        if not candidates:
            for vi, vj in neighbors4(ui, uj):
                if in_bounds(grid, vi, vj) and passable(grid, vi, vj):
                    nh = best_bin(vi - ui, vj - uj)
                    candidates.append(((vi, vj, nh), [(vi, vj)], 1))

        for v, seg, cost in candidates:
            ng = gc + cost
            if ng < g.get(v, 1 << 60):
                g[v] = ng
                came[v] = u
                seg_map[v] = seg
                heapq.heappush(openpq, (ng + heuristic(v), ng, v))

    return None, {"ok": False, "expanded": expanded}


# ============================================================
# PHYSICS SIMULATION
# ============================================================

@dataclass
class Vec2:
    x: float
    y: float

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self):
        l = self.length()
        if l < 0.001:
            return Vec2(0, 0)
        return Vec2(self.x / l, self.y / l)

    def tuple(self):
        return (self.x, self.y)


class Agent:
    def __init__(self, x, y, algo_name, color):
        self.pos = Vec2(x, y)
        self.vel = Vec2(0, 0)
        self.radius = 8
        self.max_speed = 120  # pixels per second
        self.max_force = 300  # acceleration
        self.algo_name = algo_name
        self.color = color
        self.path = []
        self.path_index = 0
        self.goal = None
        self.stats = {"expanded": 0, "time_ms": 0, "path_len": 0}
        self.stuck_timer = 0

    def seek_path(self, dt, cell_size):
        if not self.path or self.path_index >= len(self.path):
            self.vel = self.vel * 0.95  # Slow down
            return

        # Get target cell in world coords
        target_cell = self.path[self.path_index]
        target_world = Vec2(
            target_cell[1] * cell_size + cell_size / 2,
            target_cell[0] * cell_size + cell_size / 2
        )

        # Check if reached current waypoint
        if (self.pos - target_world).length() < cell_size * 0.5:
            self.path_index += 1
            self.stuck_timer = 0
            if self.path_index >= len(self.path):
                return
            target_cell = self.path[self.path_index]
            target_world = Vec2(
                target_cell[1] * cell_size + cell_size / 2,
                target_cell[0] * cell_size + cell_size / 2
            )

        # Steering behavior
        desired = (target_world - self.pos).normalized() * self.max_speed
        steering = desired - self.vel

        # Limit force
        if steering.length() > self.max_force:
            steering = steering.normalized() * self.max_force

        self.vel = self.vel + steering * dt

        # Limit speed
        if self.vel.length() > self.max_speed:
            self.vel = self.vel.normalized() * self.max_speed

    def update(self, dt, obstacles, cell_size):
        # Pathfinding steering
        self.seek_path(dt, cell_size)

        # Obstacle avoidance (local collision response)
        avoid = Vec2(0, 0)
        for obs in obstacles:
            obs_center = Vec2(obs.x + obs.width / 2, obs.y + obs.height / 2)
            diff = self.pos - obs_center
            dist = diff.length()
            min_dist = self.radius + max(obs.width, obs.height) * 0.7
            if dist < min_dist and dist > 0:
                avoid = avoid + diff.normalized() * (min_dist - dist) * 5

        self.vel = self.vel + avoid * dt

        # Update position
        self.pos = self.pos + self.vel * dt

        # Wall collision
        self.pos.x = max(self.radius, min(self.pos.x, 800 - self.radius))
        self.pos.y = max(self.radius, min(self.pos.y, 600 - self.radius))

    def draw(self, screen):
        # Draw agent
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

        # Draw velocity direction
        if self.vel.length() > 1:
            end = self.pos + self.vel.normalized() * (self.radius + 8)
            pygame.draw.line(screen, (255, 255, 255), self.pos.tuple(), end.tuple(), 2)


class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (80, 80, 80)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (120, 120, 120), (self.x, self.y, self.width, self.height), 2)


# ============================================================
# MAIN SIMULATION
# ============================================================

class PhysicsPathfindingSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Physics Pathfinding: A* vs WBS vs RG-A*")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Grid setup
        self.cell_size = 10
        self.grid_w = 800 // self.cell_size
        self.grid_h = 600 // self.cell_size
        self.grid = [[1] * self.grid_w for _ in range(self.grid_h)]

        # Obstacles
        self.obstacles = []
        self.create_demo_obstacles()
        self.update_grid()

        # Preprocessing for RG-A*
        self.rg_preprocess = None
        self.recompute_rg_preprocess()

        # Agents
        self.agents = []

        # UI
        self.selected_algo = "A*"
        self.show_grid = False
        self.show_paths = True
        self.paused = False

    def create_demo_obstacles(self):
        # Walls
        self.obstacles.append(Obstacle(150, 100, 20, 200))
        self.obstacles.append(Obstacle(300, 300, 20, 200))
        self.obstacles.append(Obstacle(450, 50, 20, 250))
        self.obstacles.append(Obstacle(600, 250, 20, 200))

        # Scattered boxes
        self.obstacles.append(Obstacle(250, 150, 40, 40))
        self.obstacles.append(Obstacle(400, 400, 50, 30))
        self.obstacles.append(Obstacle(550, 100, 30, 60))

    def update_grid(self):
        # Reset grid
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                self.grid[i][j] = 1

        # Mark obstacles
        for obs in self.obstacles:
            i_start = max(0, int(obs.y // self.cell_size))
            i_end = min(self.grid_h, int((obs.y + obs.height) // self.cell_size) + 1)
            j_start = max(0, int(obs.x // self.cell_size))
            j_end = min(self.grid_w, int((obs.x + obs.width) // self.cell_size) + 1)

            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    self.grid[i][j] = 0

    def recompute_rg_preprocess(self):
        dist = obstacle_distance_transform(self.grid)
        gx, gy = precompute_gradients(dist)
        self.rg_preprocess = (dist, gx, gy)

    def spawn_agent(self, x, y, algo_name):
        colors = {
            "A*": (100, 200, 255),
            "WBS": (255, 180, 100),
            "RG-A*": (150, 255, 150)
        }
        agent = Agent(x, y, algo_name, colors.get(algo_name, (255, 255, 255)))
        self.agents.append(agent)

    def compute_path(self, agent, goal_world):
        # Convert to grid coords
        start_grid = (
            int(agent.pos.y // self.cell_size),
            int(agent.pos.x // self.cell_size)
        )
        goal_grid = (
            int(goal_world[1] // self.cell_size),
            int(goal_world[0] // self.cell_size)
        )

        # Clamp to valid
        start_grid = (
            max(0, min(start_grid[0], self.grid_h - 1)),
            max(0, min(start_grid[1], self.grid_w - 1))
        )
        goal_grid = (
            max(0, min(goal_grid[0], self.grid_h - 1)),
            max(0, min(goal_grid[1], self.grid_w - 1))
        )

        # Check if blocked
        if not passable(self.grid, start_grid[0], start_grid[1]):
            return None
        if not passable(self.grid, goal_grid[0], goal_grid[1]):
            return None

        # Run pathfinding
        t0 = time.perf_counter()

        if agent.algo_name == "A*":
            path, stats = a_star(self.grid, start_grid, goal_grid)
        elif agent.algo_name == "WBS":
            path, stats = wbs_optimized(self.grid, start_grid, goal_grid)
        elif agent.algo_name == "RG-A*":
            path, stats = rg_astar_simple(self.grid, start_grid, goal_grid, self.rg_preprocess)
        else:
            return None

        t1 = time.perf_counter()

        if path:
            agent.path = path
            agent.path_index = 0
            agent.goal = goal_world
            agent.stats = {
                "expanded": stats.get("expanded", 0),
                "time_ms": (t1 - t0) * 1000,
                "path_len": len(path)
            }
            return True
        return False

    def draw_grid(self):
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                x = j * self.cell_size
                y = i * self.cell_size
                if self.grid[i][j] == 0:
                    pygame.draw.rect(self.screen, (60, 60, 60),
                                     (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (40, 40, 40),
                                 (x, y, self.cell_size, self.cell_size), 1)

    def draw_paths(self):
        for agent in self.agents:
            if not agent.path:
                continue

            # Draw path as lines
            points = []
            for cell in agent.path:
                px = cell[1] * self.cell_size + self.cell_size / 2
                py = cell[0] * self.cell_size + self.cell_size / 2
                points.append((px, py))

            if len(points) > 1:
                pygame.draw.lines(self.screen, agent.color, False, points, 2)

            # Draw goal
            if agent.goal:
                pygame.draw.circle(self.screen, agent.color,
                                   (int(agent.goal[0]), int(agent.goal[1])), 6, 2)

    def draw_ui(self):
        y_offset = 10

        # Title
        title = self.font.render("Physics Pathfinding Sim", True, (255, 255, 255))
        self.screen.blit(title, (10, y_offset))
        y_offset += 30

        # Controls
        controls = [
            "Click: Set goal for all agents",
            "1/2/3: Spawn A*/WBS/RG-A* agent at mouse",
            "C: Clear all agents",
            "G: Toggle grid",
            "P: Toggle paths",
            "SPACE: Pause"
        ]

        for text in controls:
            surf = self.small_font.render(text, True, (200, 200, 200))
            self.screen.blit(surf, (10, y_offset))
            y_offset += 20

        # Agent stats
        y_offset += 10
        for agent in self.agents:
            color_box = pygame.Surface((15, 15))
            color_box.fill(agent.color)
            self.screen.blit(color_box, (10, y_offset))

            stats_text = f"{agent.algo_name}: {agent.stats['expanded']}n, {agent.stats['time_ms']:.2f}ms, {agent.stats['path_len']}cells"
            surf = self.small_font.render(stats_text, True, (220, 220, 220))
            self.screen.blit(surf, (30, y_offset))
            y_offset += 20

        # Status
        if self.paused:
            pause_surf = self.font.render("PAUSED", True, (255, 100, 100))
            self.screen.blit(pause_surf, (700, 10))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    mx, my = pygame.mouse.get_pos()
                    self.spawn_agent(mx, my, "A*")
                elif event.key == pygame.K_2:
                    mx, my = pygame.mouse.get_pos()
                    self.spawn_agent(mx, my, "WBS")
                elif event.key == pygame.K_3:
                    mx, my = pygame.mouse.get_pos()
                    self.spawn_agent(mx, my, "RG-A*")
                elif event.key == pygame.K_c:
                    self.agents.clear()
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_p:
                    self.show_paths = not self.show_paths
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mx, my = pygame.mouse.get_pos()
                    for agent in self.agents:
                        self.compute_path(agent, (mx, my))

        return True

    def update(self, dt):
        if not self.paused:
            for agent in self.agents:
                agent.update(dt, self.obstacles, self.cell_size)

    def draw(self):
        self.screen.fill((30, 30, 30))

        if self.show_grid:
            self.draw_grid()

        for obs in self.obstacles:
            obs.draw(self.screen)

        if self.show_paths:
            self.draw_paths()

        for agent in self.agents:
            agent.draw(self.screen)

        self.draw_ui()

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS, dt in seconds

            running = self.handle_events()
            self.update(dt)
            self.draw()

        pygame.quit()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    sim = PhysicsPathfindingSim()
    sim.run()