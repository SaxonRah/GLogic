#!/usr/bin/env python3
"""
cln_viz_act3.py — Act 3: "Sharing turns trees into DAGs"

Animated 3Blue1Brown-style morph showing how an expression tree
compresses into a DAG when identical subtrees are shared.

Run:
  manim -pql cln_viz_act3.py Act3DAG
  manim -pqh cln_viz_act3.py Act3DAG
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple

from manim import (
    Scene, VGroup, Text, MathTex, Line, Arrow,
    FadeIn, FadeOut, Create, LaggedStart, MoveToTarget,
    Indicate, GrowFromCenter, AnimationGroup,
    UP, DOWN, LEFT, RIGHT, ORIGIN, config,
    RoundedRectangle, SurroundingRectangle,
    BLUE_D, BLUE_C, YELLOW, GOLD_C, GREY_B, GREY_C, WHITE,
    GREEN_C, RED_B, TEAL_C, MAROON_B, ORANGE,
    smooth, ArcBetweenPoints,
)
import numpy as np

# ─────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────
VAR_COLOR  = BLUE_C
OP_COLOR   = GOLD_C
SHARED_COL = MAROON_B
EDGE_COLOR = GREY_B
BG_COLOR   = "#1a1a2e"
HIGHLIGHT_COLORS = [MAROON_B, ORANGE, TEAL_C]  # for successive merge rounds

# ─────────────────────────────────────────────
# Tree data
# ─────────────────────────────────────────────
# Expression:
#   ((x₀ ⊕ x₁) & (x₀ ⊕ x₁)) | (((x₀ ⊕ x₁) & (x₀ ⊕ x₁)) | ((x₂ | ~x₃) & (x₂ | ~x₃)))
#
# 25 tree nodes, 11 DAG nodes after sharing.

TREE_NODES = {
    0:  {"label": "|",  "children": [1, 2]},
    1:  {"label": "&",  "children": [3, 4]},
    2:  {"label": "|",  "children": [5, 6]},
    3:  {"label": "⊕",  "children": [7, 8]},
    4:  {"label": "⊕",  "children": [9, 10]},
    5:  {"label": "&",  "children": [11, 12]},
    6:  {"label": "&",  "children": [13, 14]},
    7:  {"label": "x₀", "children": []},
    8:  {"label": "x₁", "children": []},
    9:  {"label": "x₀", "children": []},
    10: {"label": "x₁", "children": []},
    11: {"label": "⊕",  "children": [15, 16]},
    12: {"label": "⊕",  "children": [17, 18]},
    13: {"label": "|",  "children": [19, 20]},
    14: {"label": "|",  "children": [21, 22]},
    15: {"label": "x₀", "children": []},
    16: {"label": "x₁", "children": []},
    17: {"label": "x₀", "children": []},
    18: {"label": "x₁", "children": []},
    19: {"label": "x₂", "children": []},
    20: {"label": "¬",  "children": [23]},
    21: {"label": "x₂", "children": []},
    22: {"label": "¬",  "children": [24]},
    23: {"label": "x₃", "children": []},
    24: {"label": "x₃", "children": []},
}

# Parent map (computed)
PARENT_OF = {}
for _nid, _data in TREE_NODES.items():
    for _cid in _data["children"]:
        PARENT_OF[_cid] = _nid


def get_subtree_ids(root_id: int) -> List[int]:
    """All node IDs in subtree rooted at root_id (BFS order)."""
    result = []
    queue = [root_id]
    while queue:
        nid = queue.pop(0)
        result.append(nid)
        queue.extend(TREE_NODES[nid]["children"])
    return result


# ─────────────────────────────────────────────
# Merge plan
# ─────────────────────────────────────────────
# Round 1: merge four ⊕(x₀,x₁) subtrees → canonical=3, dups=[4,11,12]
# Round 2: merge two |(x₂,¬x₃) subtrees → canonical=13, dups=[14]
# Round 3: merge two & nodes (whose children both now point to shared ⊕) → canonical=1, dups=[5]

MERGE_ROUNDS = [
    {
        "label": "⊕(x₀, x₁)  ×4 → ×1",
        "color": HIGHLIGHT_COLORS[0],
        "canonical": 3,
        "duplicates": [4, 11, 12],
    },
    {
        "label": "|(x₂, ¬x₃)  ×2 → ×1",
        "color": HIGHLIGHT_COLORS[1],
        "canonical": 13,
        "duplicates": [14],
    },
    {
        "label": "&  ×2 → ×1",
        "color": HIGHLIGHT_COLORS[2],
        "canonical": 1,
        "duplicates": [5],
    },
]


# ─────────────────────────────────────────────
# Tree layout
# ─────────────────────────────────────────────

def compute_subtree_widths(root_id: int) -> Dict[int, float]:
    """Width of each node's subtree (leaves = 1.0)."""
    widths = {}

    def _walk(nid):
        kids = TREE_NODES[nid]["children"]
        if not kids:
            widths[nid] = 1.0
        else:
            for k in kids:
                _walk(k)
            widths[nid] = sum(widths[k] for k in kids) + 0.15 * (len(kids) - 1)
        return widths[nid]

    _walk(root_id)
    return widths


def compute_tree_positions(
    root_id: int,
    h_scale: float = 1.0,
    v_scale: float = 1.0,
    top_y: float = 3.2,
) -> Dict[int, Tuple[float, float]]:
    """Assign (x, y) positions via width-proportional layout."""
    widths = compute_subtree_widths(root_id)
    positions = {}

    def _place(nid, x_left, depth):
        w = widths[nid]
        my_x = x_left + w / 2.0
        my_y = top_y - depth * v_scale
        positions[nid] = (my_x * h_scale, my_y)

        kids = TREE_NODES[nid]["children"]
        cursor = x_left
        for k in kids:
            _place(k, cursor, depth + 1)
            cursor += widths[k] + 0.15

    _place(root_id, 0.0, 0)

    # Center horizontally: shift so root x = 0
    root_x = positions[root_id][0]
    for nid in positions:
        x, y = positions[nid]
        positions[nid] = (x - root_x, y)

    return positions


# ─────────────────────────────────────────────
# DAG final layout (hand-tuned for 11 nodes)
# ─────────────────────────────────────────────
DAG_POSITIONS = {
    0:  (0.0,   3.2),    # root |
    1:  (-2.5,  1.8),    # &  (shared)
    2:  (2.0,   1.8),    # |
    3:  (-2.5,  0.2),    # ⊕  (shared)
    6:  (2.0,   0.2),    # &
    7:  (-3.5, -1.3),    # x₀
    8:  (-1.5, -1.3),    # x₁
    13: (2.0,  -1.3),    # |  (shared)
    19: (1.0,  -2.6),    # x₂
    20: (3.0,  -2.6),    # ¬
    23: (3.0,  -3.7),    # x₃
}

# DAG edges (parent → child), including multi-parent edges
DAG_EDGES = [
    (0, 1), (0, 2),
    (2, 1), (2, 6),      # note: 2→1 is a cross-level DAG edge (sharing!)
    (1, 3),               # & has both children = ⊕ (we'll draw two edges)
    (6, 13),              # & has both children = | (we'll draw two edges)
    (3, 7), (3, 8),
    (13, 19), (13, 20),
    (20, 23),
]

# Nodes where both children point to the same node (needs two-edge visual)
DOUBLE_EDGES = {1: 3, 6: 13}  # parent: shared_child


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def fit_to_frame(mobj, pad: float = 0.88):
    fw = config.frame_width * pad
    fh = config.frame_height * pad
    if mobj.width > fw:
        mobj.scale(fw / mobj.width)
    if mobj.height > fh:
        mobj.scale(fh / mobj.height)
    return mobj


def is_op(nid: int) -> bool:
    return TREE_NODES[nid]["label"] in ("|", "&", "⊕", "¬")


def node_color(nid: int):
    return OP_COLOR if is_op(nid) else VAR_COLOR


def make_tree_node(nid: int, pos, font_size=20, scale=1.0):
    """Create a rounded rect + label for a tree node."""
    w = 0.55 * scale
    h = 0.4 * scale
    color = node_color(nid)
    rect = RoundedRectangle(
        width=w, height=h, corner_radius=0.08,
        stroke_color=color, stroke_width=2.0,
        fill_color=color, fill_opacity=0.10,
    )
    label = Text(TREE_NODES[nid]["label"], font_size=font_size, color=color)
    label.move_to(rect.get_center())
    group = VGroup(rect, label)
    group.move_to(np.array([pos[0], pos[1], 0]))
    return group


def make_edge(parent_pos, child_pos) -> Arrow:
    """Arrow from parent bottom to child top."""
    start = np.array([parent_pos[0], parent_pos[1] - 0.2, 0])
    end = np.array([child_pos[0], child_pos[1] + 0.2, 0])
    return Arrow(
        start, end, buff=0.0,
        stroke_width=1.8, color=EDGE_COLOR,
        max_tip_length_to_length_ratio=0.10,
    )


# ─────────────────────────────────────────────
# Act 3 Scene
# ─────────────────────────────────────────────

class Act3DAG(Scene):

    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Title card ──
        title = Text(
            "Sharing turns trees into DAGs",
            font_size=48, color=WHITE,
        ).move_to(ORIGIN)
        self.play(FadeIn(title, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(title, shift=UP * 0.3))
        self.wait(0.3)

        self.do_dag_morph()

    def do_dag_morph(self):
        on_screen = []

        # ─────────────────────────────────────
        # Phase 1: Show expression
        # ─────────────────────────────────────
        sec_label = Text("expression tree", font_size=34, color=GREY_C)
        sec_label.to_edge(UP, buff=0.25)
        self.play(FadeIn(sec_label))
        on_screen.append(sec_label)

        expr_tex = MathTex(
            r"((x_0 \oplus x_1) \mathbin{\&} (x_0 \oplus x_1))"
            r"\;|\;"
            r"(((x_0 \oplus x_1) \mathbin{\&} (x_0 \oplus x_1))"
            r"\;|\;"
            r"((x_2 | \neg x_3) \mathbin{\&} (x_2 | \neg x_3)))",
            font_size=26,
        )
        expr_tex.next_to(sec_label, DOWN, buff=0.25)
        fit_to_frame(expr_tex, pad=0.95)
        self.play(FadeIn(expr_tex))
        on_screen.append(expr_tex)
        self.wait(0.6)

        # ─────────────────────────────────────
        # Phase 2: Grow the full tree level by level
        # ─────────────────────────────────────

        # Compute layout
        positions = compute_tree_positions(0, h_scale=0.85, v_scale=1.05, top_y=2.8)

        # Build mobjects
        node_mobs: Dict[int, VGroup] = {}
        edge_mobs: Dict[Tuple[int, int], Arrow] = {}

        for nid in TREE_NODES:
            node_mobs[nid] = make_tree_node(nid, positions[nid])

        for nid, data in TREE_NODES.items():
            for cid in data["children"]:
                edge_mobs[(nid, cid)] = make_edge(positions[nid], positions[cid])

        # Fit everything to frame
        all_tree = VGroup(*node_mobs.values(), *edge_mobs.values())
        fit_to_frame(all_tree, pad=0.82)

        # We need to re-extract positions after fit_to_frame scaling
        # (the group was scaled/shifted as a unit, so mobject positions are correct)

        # Organize by BFS depth for level-by-level animation
        depth_of: Dict[int, int] = {}
        queue = [(0, 0)]
        while queue:
            nid, d = queue.pop(0)
            depth_of[nid] = d
            for cid in TREE_NODES[nid]["children"]:
                queue.append((cid, d + 1))

        max_depth = max(depth_of.values())
        levels: Dict[int, List[int]] = {}
        for nid, d in depth_of.items():
            levels.setdefault(d, []).append(nid)

        # Fade out expression, animate tree growth
        self.play(FadeOut(expr_tex, run_time=0.5))
        on_screen.remove(expr_tex)

        # Level 0: root
        self.play(GrowFromCenter(node_mobs[0], run_time=0.5))

        # Subsequent levels: nodes + edges from parent
        for d in range(1, max_depth + 1):
            layer_nodes = sorted(levels.get(d, []))
            node_anims = []
            edge_anims = []
            for nid in layer_nodes:
                node_anims.append(FadeIn(node_mobs[nid], shift=DOWN * 0.15))
                pid = PARENT_OF[nid]
                edge_anims.append(Create(edge_mobs[(pid, nid)]))

            self.play(
                LaggedStart(*edge_anims, lag_ratio=0.05),
                LaggedStart(*node_anims, lag_ratio=0.05),
                run_time=0.7 if d < 3 else 0.5,
            )

        on_screen.append(all_tree)
        self.wait(0.4)

        # ─────────────────────────────────────
        # Node counter
        # ─────────────────────────────────────
        total_nodes = len(TREE_NODES)
        counter = Text(f"Nodes: {total_nodes}", font_size=28, color=WHITE)
        counter.to_corner(DOWN + RIGHT, buff=0.5)
        self.play(FadeIn(counter))
        on_screen.append(counter)
        self.wait(0.3)

        # Track which nodes are still alive
        alive: Set[int] = set(TREE_NODES.keys())

        # ─────────────────────────────────────
        # Phase 3: Merge rounds
        # ─────────────────────────────────────

        for round_info in MERGE_ROUNDS:
            canon_root = round_info["canonical"]
            dup_roots = round_info["duplicates"]
            hi_color = round_info["color"]
            round_label = round_info["label"]

            # Gather all nodes in canonical + duplicate subtrees
            canon_ids = [nid for nid in get_subtree_ids(canon_root) if nid in alive]
            dup_id_sets = []
            for dr in dup_roots:
                ids = [nid for nid in get_subtree_ids(dr) if nid in alive]
                dup_id_sets.append(ids)

            all_highlight_ids = list(canon_ids)
            for ds in dup_id_sets:
                all_highlight_ids.extend(ds)

            # Annotation
            anno = Text(f"identical2: {round_label}", font_size=24, color=hi_color)
            anno.to_edge(DOWN, buff=0.35).shift(LEFT * 2.5)
            self.play(FadeIn(anno, run_time=0.3))

            # Highlight all copies simultaneously
            highlight_anims = []
            for nid in all_highlight_ids:
                if nid in alive:
                    highlight_anims.append(
                        node_mobs[nid][0].animate.set_stroke(color=hi_color, width=3.5)
                    )
                    highlight_anims.append(
                        node_mobs[nid][1].animate.set_color(hi_color)
                    )
            if highlight_anims:
                self.play(*highlight_anims, run_time=0.6)
            self.wait(0.4)

            # Merge: slide duplicates toward canonical, then fade out
            for dup_ids in dup_id_sets:
                dup_root_id = dup_ids[0]  # the root of this dup subtree

                # Compute offset from dup root to canonical root
                canon_center = node_mobs[canon_root].get_center()
                dup_center = node_mobs[dup_root_id].get_center()
                offset = canon_center - dup_center

                # Collect all mobjects in this duplicate subtree (nodes + internal edges)
                dup_mobjects = []
                for nid in dup_ids:
                    dup_mobjects.append(node_mobs[nid])
                    for cid in TREE_NODES[nid]["children"]:
                        if cid in alive and (nid, cid) in edge_mobs:
                            dup_mobjects.append(edge_mobs[(nid, cid)])

                # Also include the edge from parent to dup root
                if dup_root_id in PARENT_OF:
                    parent_edge_key = (PARENT_OF[dup_root_id], dup_root_id)
                    if parent_edge_key in edge_mobs:
                        dup_mobjects.append(edge_mobs[parent_edge_key])

                # Animate: shift toward canonical, then fade out
                shift_anims = [m.animate.shift(offset) for m in dup_mobjects]
                self.play(*shift_anims, run_time=0.8, rate_func=smooth)
                self.play(*[FadeOut(m, run_time=0.3) for m in dup_mobjects])

                # Mark as dead
                for nid in dup_ids:
                    alive.discard(nid)
                    for cid in TREE_NODES[nid]["children"]:
                        edge_key = (nid, cid)
                        if edge_key in edge_mobs:
                            pass  # already faded

                # Create new edge from dup's parent to canonical root
                if dup_root_id in PARENT_OF:
                    pid = PARENT_OF[dup_root_id]
                    if pid in alive:
                        new_edge = Arrow(
                            node_mobs[pid].get_bottom(),
                            node_mobs[canon_root].get_top(),
                            buff=0.05,
                            stroke_width=1.8,
                            color=hi_color,
                            max_tip_length_to_length_ratio=0.10,
                        )
                        new_edge_key = (pid, canon_root, dup_root_id)  # unique key
                        self.play(Create(new_edge), run_time=0.4)
                        on_screen.append(new_edge)

            # Update counter
            new_count = len(alive)
            new_counter = Text(f"Nodes: {new_count}", font_size=28, color=WHITE)
            new_counter.to_corner(DOWN + RIGHT, buff=0.5)
            self.play(
                FadeOut(counter, run_time=0.2),
                FadeIn(new_counter, run_time=0.3),
            )
            on_screen.remove(counter)
            counter = new_counter
            on_screen.append(counter)

            # Restore canonical nodes to normal color
            restore_anims = []
            for nid in canon_ids:
                if nid in alive:
                    c = node_color(nid)
                    restore_anims.append(
                        node_mobs[nid][0].animate.set_stroke(color=c, width=2.0)
                    )
                    restore_anims.append(
                        node_mobs[nid][1].animate.set_color(c)
                    )
            if restore_anims:
                self.play(*restore_anims, run_time=0.4)

            self.play(FadeOut(anno, run_time=0.3))
            self.wait(0.2)

        self.wait(0.5)

        # ─────────────────────────────────────
        # Phase 4: Reposition to clean DAG layout
        # ─────────────────────────────────────
        sec_label2 = Text("shared-subexpression DAG", font_size=34, color=GREY_C)
        sec_label2.to_edge(UP, buff=0.25)
        self.play(FadeOut(sec_label), FadeIn(sec_label2), run_time=0.5)
        on_screen.remove(sec_label)
        on_screen.append(sec_label2)

        # Fade out all surviving edges (they'll be redrawn cleanly)
        surviving_edge_keys = [
            (pid, cid) for (pid, cid) in edge_mobs
            if pid in alive and cid in alive
        ]
        edge_fadeouts = []
        for key in surviving_edge_keys:
            if edge_mobs[key].get_fill_opacity() >= 0 or True:
                edge_fadeouts.append(FadeOut(edge_mobs[key], run_time=0.3))
        if edge_fadeouts:
            self.play(*edge_fadeouts)

        # Move surviving nodes to DAG positions
        # Build a fresh group for scaling
        dag_node_group = VGroup(*[node_mobs[nid] for nid in alive])

        move_anims = []
        for nid in alive:
            if nid in DAG_POSITIONS:
                target_pos = np.array([DAG_POSITIONS[nid][0], DAG_POSITIONS[nid][1], 0])
                move_anims.append(node_mobs[nid].animate.move_to(target_pos))

        self.play(*move_anims, run_time=1.5, rate_func=smooth)
        self.wait(0.3)

        # Draw DAG edges (clean)
        dag_edge_mobs = VGroup()

        for (pid, cid) in DAG_EDGES:
            edge = Arrow(
                node_mobs[pid].get_bottom(),
                node_mobs[cid].get_top(),
                buff=0.05,
                stroke_width=2.0,
                color=EDGE_COLOR,
                max_tip_length_to_length_ratio=0.10,
            )
            dag_edge_mobs.add(edge)

        # Double-edges: where both children point to same node, add a second
        # curved arrow so the sharing is visible
        for parent_id, child_id in DOUBLE_EDGES.items():
            arc = ArcBetweenPoints(
                node_mobs[parent_id].get_bottom() + LEFT * 0.15,
                node_mobs[child_id].get_top() + LEFT * 0.15,
                angle=-0.5,
                stroke_width=2.0,
                color=SHARED_COL,
            )
            # Add a small tip manually or just use a line — keep it simple
            dag_edge_mobs.add(arc)

        self.play(
            LaggedStart(*[Create(e) for e in dag_edge_mobs],
                         lag_ratio=0.06, run_time=1.2)
        )
        on_screen.append(dag_edge_mobs)
        self.wait(0.4)

        # Highlight the multi-parent nodes
        shared_ids = [3, 13, 1]  # ⊕, |, &  — nodes with >1 incoming edge
        shared_note = Text(
            "shared nodes have multiple parents",
            font_size=24, color=SHARED_COL,
        )
        shared_note.to_edge(DOWN, buff=0.35).shift(LEFT * 2)
        self.play(FadeIn(shared_note, run_time=0.4))
        self.play(
            *[Indicate(node_mobs[nid], color=SHARED_COL, scale_factor=1.2)
              for nid in shared_ids if nid in alive],
            run_time=0.8,
        )
        on_screen.append(shared_note)
        self.wait(0.5)
        self.play(FadeOut(shared_note, run_time=0.4))
        on_screen.remove(shared_note)

        # ─────────────────────────────────────
        # Phase 5: Final comparison
        # ─────────────────────────────────────
        comparison = Text(
            f"Tree: {len(TREE_NODES)} nodes  →  DAG: {len(alive)} nodes",
            font_size=34, color=WHITE, weight="BOLD",
        )
        comparison.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(comparison, shift=UP * 0.2), run_time=0.6)
        on_screen.append(comparison)
        self.wait(2.0)

        # ─────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────
        # Gather everything still visible
        remaining = list(on_screen)
        for nid in alive:
            remaining.append(node_mobs[nid])
        remaining.append(counter)

        # Deduplicate (some may be in on_screen already via all_tree)
        seen = set()
        unique_remaining = []
        for m in remaining:
            mid = id(m)
            if mid not in seen:
                seen.add(mid)
                unique_remaining.append(m)

        self.play(*[FadeOut(m) for m in unique_remaining], run_time=0.8)
        self.wait(0.4)