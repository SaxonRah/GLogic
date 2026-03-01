#!/usr/bin/env python3
"""
cln_viz_act1.py — Act 1: "Bitstrings live on hypercubes"

Animated 3Blue1Brown-style morph from bitstrings to hypercube vertex colorings.

Run:
  manim -pql cln_viz_act1.py Act1Hypercube
  manim -pqh cln_viz_act1.py Act1Hypercube   # high quality
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict

from manim import (
    Scene, VGroup, Dot, Text, Line, FadeIn, FadeOut,
    Create, LaggedStart, MoveToTarget, AnimationGroup,
    Write, Unwrite, Flash, Indicate,
    UP, DOWN, LEFT, RIGHT, ORIGIN, config,
    RoundedRectangle, SurroundingRectangle,
    BLUE_D, BLUE_C, YELLOW, GOLD_C, GREY_B, GREY_C, WHITE,
    GREEN_C, RED_B, TEAL_C, MAROON_B,
    rate_functions, DEFAULT_FONT_SIZE,
    MathTex,
)
from manim import smooth  # rate function

# ─────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────
BIT_ZERO   = BLUE_D
BIT_ONE    = YELLOW
VAR_COLOR  = BLUE_C
CLAUSE_COL = TEAL_C
NEG_COLOR  = RED_B
POS_COLOR  = GREEN_C
OP_COLOR   = GOLD_C
SHARED_COL = MAROON_B
EDGE_COLOR = GREY_B
BG_COLOR   = "#1a1a2e"    # dark navy background

DOT_RADIUS = 0.18
EDGE_WIDTH = 2.0

# ─────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────

def all_corners(n: int) -> List[Tuple[int, ...]]:
    """Pos=0 then Neg=1 recursion — matches Coq all_corners."""
    if n == 0:
        return [()]
    prev = all_corners(n - 1)
    return [(0,) + c for c in prev] + [(1,) + c for c in prev]


def corner_label(c: Tuple[int, ...]) -> str:
    return "".join("+" if b == 0 else "−" for b in c)


def bit_color(v: int):
    return BIT_ONE if v == 1 else BIT_ZERO


def fit_to_frame(mobj, pad: float = 0.88):
    fw = config.frame_width * pad
    fh = config.frame_height * pad
    if mobj.width > fw:
        mobj.scale(fw / mobj.width)
    if mobj.height > fh:
        mobj.scale(fh / mobj.height)
    return mobj


# ─────────────────────────────────────────────
# Hypercube 2D projections
# ─────────────────────────────────────────────

def hypercube_pos_2d(corner: Tuple[int, ...], scale: float = 1.8) -> tuple:
    """
    Return (x, y) screen position for a hypercube corner.
    corner is a tuple of 0s and 1s (Pos/Neg), length n.

    n=2: literal square
    n=3: front/back square with perspective offset
    n=4: cube-in-cube (tesseract)
    """
    n = len(corner)

    if n == 2:
        c0, c1 = corner
        x = (2 * c0 - 1) * scale
        y = (2 * c1 - 1) * scale
        return (x, y)

    elif n == 3:
        c0, c1, c2 = corner
        # c1,c2 form the base square; c0 selects front(0) vs back(1)
        base_x = (2 * c1 - 1) * scale
        base_y = (2 * c2 - 1) * scale
        # back face: shrink and offset
        depth = c0
        shrink = 1.0 - 0.3 * depth
        off_x = 0.6 * scale * depth
        off_y = 0.45 * scale * depth
        return (base_x * shrink + off_x, base_y * shrink + off_y)

    elif n == 4:
        c0, c1, c2, c3 = corner
        # c1,c2,c3 form the inner cube; c0 selects outer(0) vs inner(1)
        inner = hypercube_pos_2d((c1, c2, c3), scale=scale)
        depth = c0
        shrink = 1.0 - 0.45 * depth
        off_x = 0.15 * scale * depth
        off_y = 0.12 * scale * depth
        return (inner[0] * shrink + off_x, inner[1] * shrink + off_y)

    else:
        # fallback: just use first two coords
        x = (2 * corner[0] - 1) * scale
        y = (2 * corner[1] - 1) * scale
        return (x, y)


def hypercube_edges(n: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Return pairs of corners that differ in exactly one coordinate."""
    corners = all_corners(n)
    edges = []
    for i, c1 in enumerate(corners):
        for j, c2 in enumerate(corners):
            if j <= i:
                continue
            diff = sum(1 for a, b in zip(c1, c2) if a != b)
            if diff == 1:
                edges.append((c1, c2))
    return edges


# ─────────────────────────────────────────────
# Act 1 Scene
# ─────────────────────────────────────────────

class Act1Hypercube(Scene):

    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Title card ──
        title = Text(
            "Bitstrings live on hypercubes",
            font_size=48, color=WHITE
        ).move_to(ORIGIN)
        self.play(FadeIn(title, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(title, shift=UP * 0.3))
        self.wait(0.3)

        # ── n=2 : detailed walkthrough ──
        self.do_n2()

        # ── n=3 : faster ──
        self.do_n3()

        # ── n=4 : quick payoff ──
        self.do_n4()

    # ─────────────────────────────────────
    # n = 2  (slow, explanatory)
    # ─────────────────────────────────────
    def do_n2(self):
        bits = "0101"
        n = 2
        corners = all_corners(n)
        vals = [int(b) for b in bits]
        scale = 1.5

        on_screen = []  # track everything we add

        # — Section label —
        sec_label = Text("n = 2 :  4 corners of a square", font_size=36, color=GREY_C)
        sec_label.to_edge(UP, buff=0.4)
        self.play(FadeIn(sec_label))
        on_screen.append(sec_label)
        self.wait(0.3)

        # — Show bitstring as colored digits —
        digits = VGroup()
        for ch in bits:
            v = int(ch)
            d = Text(ch, font_size=52, color=bit_color(v), weight="BOLD")
            digits.add(d)
        digits.arrange(RIGHT, buff=0.35)
        digits.move_to(UP * 1.8)

        self.play(LaggedStart(*[FadeIn(d, shift=DOWN * 0.2) for d in digits],
                               lag_ratio=0.12))
        on_screen.append(digits)
        self.wait(0.4)

        # — Corner address labels beneath each digit —
        addr_labels = VGroup()
        for i, c in enumerate(corners):
            lab = Text(corner_label(c), font_size=28, color=GREY_B)
            lab.next_to(digits[i], DOWN, buff=0.15)
            addr_labels.add(lab)

        self.play(LaggedStart(*[FadeIn(a, shift=DOWN * 0.1) for a in addr_labels],
                               lag_ratio=0.1))
        on_screen.append(addr_labels)
        self.wait(0.3)

        # — Explanation text —
        explain = Text(
            "each bit lives at a corner",
            font_size=28, color=GREY_C
        )
        explain.next_to(addr_labels, DOWN, buff=0.25)
        self.play(FadeIn(explain))
        on_screen.append(explain)
        self.wait(0.8)
        self.play(FadeOut(explain))
        on_screen.remove(explain)

        # — Build target dots at hypercube positions —
        center = DOWN * 0.8
        dots = {}       # corner -> Dot
        dot_labels = {} # corner -> Text (the corner label like ++)
        val_labels = {} # corner -> Text (the 0/1 value)

        for i, c in enumerate(corners):
            px, py = hypercube_pos_2d(c, scale=scale)
            pos = center + px * RIGHT + py * UP

            dot = Dot(pos, radius=DOT_RADIUS, color=bit_color(vals[i]))
            dot.set_z_index(2)
            dots[c] = dot

            clab = Text(corner_label(c), font_size=22, color=GREY_B)
            clab.next_to(dot, UP + LEFT, buff=0.12)
            dot_labels[c] = clab

            vlab = Text(str(vals[i]), font_size=30, color=bit_color(vals[i]),
                        weight="BOLD")
            vlab.move_to(dot.get_center())
            val_labels[c] = vlab

        # — Animate digits flying to their corner positions —
        # Set up targets: each digit flies to the dot position
        fly_anims = []
        for i, c in enumerate(corners):
            digits[i].generate_target()
            digits[i].target.move_to(dots[c].get_center())
            digits[i].target.set_color(bit_color(vals[i]))
            digits[i].target.scale(0.7)
            fly_anims.append(MoveToTarget(digits[i], rate_func=smooth))

        # Also fade out the address labels since they'll reappear at the vertices
        self.play(
            *fly_anims,
            FadeOut(addr_labels, run_time=0.6),
            run_time=1.8
        )
        on_screen.remove(addr_labels)
        self.wait(0.2)

        # — Now replace the flown digits with proper dots + labels —
        dot_group = VGroup(*dots.values())
        clab_group = VGroup(*dot_labels.values())
        vlab_group = VGroup(*val_labels.values())

        self.play(
            FadeOut(digits, run_time=0.3),
            FadeIn(dot_group, run_time=0.5),
            FadeIn(vlab_group, run_time=0.5),
        )
        on_screen.remove(digits)
        on_screen.extend([dot_group, vlab_group])
        self.wait(0.2)

        # — Draw edges of the square —
        edge_list = hypercube_edges(n)
        edge_lines = VGroup()
        for c1, c2 in edge_list:
            line = Line(
                dots[c1].get_center(), dots[c2].get_center(),
                stroke_width=EDGE_WIDTH, color=EDGE_COLOR
            )
            line.set_z_index(0)
            edge_lines.add(line)

        self.play(LaggedStart(*[Create(e) for e in edge_lines],
                               lag_ratio=0.15, run_time=1.2))
        on_screen.append(edge_lines)
        self.wait(0.3)

        # — Corner labels —
        self.play(FadeIn(clab_group, run_time=0.6))
        on_screen.append(clab_group)
        self.wait(0.5)

        # — Flash all "1" vertices —
        one_dots = [dots[c] for i, c in enumerate(corners) if vals[i] == 1]
        if one_dots:
            flash_note = Text("1-bits highlighted", font_size=26, color=YELLOW)
            flash_note.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(flash_note))
            on_screen.append(flash_note)

            self.play(*[Indicate(d, color=YELLOW, scale_factor=1.5) for d in one_dots],
                      run_time=1.0)
            self.wait(0.6)
            self.play(FadeOut(flash_note))
            on_screen.remove(flash_note)

        self.wait(0.5)

        # — Clean up everything —
        self.play(*[FadeOut(m) for m in on_screen], run_time=0.7)
        self.wait(0.3)

    # ─────────────────────────────────────
    # n = 3  (medium pace)
    # ─────────────────────────────────────
    def do_n3(self):
        bits = "00010100"
        n = 3
        corners = all_corners(n)
        vals = [int(b) for b in bits]
        scale = 1.4

        on_screen = []

        # — Section label —
        sec_label = Text("n = 3 :  8 corners of a cube", font_size=36, color=GREY_C)
        sec_label.to_edge(UP, buff=0.4)
        self.play(FadeIn(sec_label))
        on_screen.append(sec_label)

        # — Bitstring —
        digits = VGroup()
        for ch in bits:
            v = int(ch)
            d = Text(ch, font_size=44, color=bit_color(v), weight="BOLD")
            digits.add(d)
        digits.arrange(RIGHT, buff=0.22)
        digits.move_to(UP * 1.6)
        self.play(FadeIn(digits))
        on_screen.append(digits)
        self.wait(0.3)

        # — Brief corner labels —
        addr_labels = VGroup()
        for i, c in enumerate(corners):
            lab = Text(corner_label(c), font_size=18, color=GREY_B)
            lab.next_to(digits[i], DOWN, buff=0.1)
            addr_labels.add(lab)
        self.play(FadeIn(addr_labels, run_time=0.5))
        on_screen.append(addr_labels)
        self.wait(0.4)

        # — Build cube —
        center = DOWN * 0.6
        dots = {}
        val_labels = {}
        dot_labels = {}

        for i, c in enumerate(corners):
            px, py = hypercube_pos_2d(c, scale=scale)
            pos = center + px * RIGHT + py * UP
            dot = Dot(pos, radius=DOT_RADIUS, color=bit_color(vals[i]))
            dot.set_z_index(2)
            dots[c] = dot

            vlab = Text(str(vals[i]), font_size=26, color=bit_color(vals[i]),
                        weight="BOLD")
            vlab.move_to(dot.get_center())
            val_labels[c] = vlab

            clab = Text(corner_label(c), font_size=16, color=GREY_B)
            clab.next_to(dot, UP + LEFT, buff=0.08)
            dot_labels[c] = clab

        # — Fly digits to positions —
        fly_anims = []
        for i, c in enumerate(corners):
            digits[i].generate_target()
            digits[i].target.move_to(dots[c].get_center())
            digits[i].target.set_color(bit_color(vals[i]))
            digits[i].target.scale(0.6)
            fly_anims.append(MoveToTarget(digits[i], rate_func=smooth))

        self.play(*fly_anims, FadeOut(addr_labels, run_time=0.5), run_time=1.5)
        on_screen.remove(addr_labels)

        # — Replace with dots —
        dot_group = VGroup(*dots.values())
        vlab_group = VGroup(*val_labels.values())
        clab_group = VGroup(*dot_labels.values())

        self.play(
            FadeOut(digits, run_time=0.3),
            FadeIn(dot_group, run_time=0.4),
            FadeIn(vlab_group, run_time=0.4),
        )
        on_screen.remove(digits)
        on_screen.extend([dot_group, vlab_group])

        # — Edges —
        edge_list = hypercube_edges(n)
        edge_lines = VGroup()
        for c1, c2 in edge_list:
            line = Line(
                dots[c1].get_center(), dots[c2].get_center(),
                stroke_width=EDGE_WIDTH, color=EDGE_COLOR
            )
            line.set_z_index(0)
            edge_lines.add(line)

        self.play(LaggedStart(*[Create(e) for e in edge_lines],
                               lag_ratio=0.08, run_time=1.0))
        on_screen.append(edge_lines)

        # — Corner labels (small) —
        self.play(FadeIn(clab_group, run_time=0.4))
        on_screen.append(clab_group)
        self.wait(0.3)

        # — Flash 1-bits —
        one_dots = [dots[c] for i, c in enumerate(corners) if vals[i] == 1]
        if one_dots:
            self.play(*[Indicate(d, color=YELLOW, scale_factor=1.4) for d in one_dots],
                      run_time=0.8)
        self.wait(0.6)

        # — Annotation: front vs back face —
        front_note = Text("front face: first coord = +", font_size=22, color=GREY_C)
        back_note = Text("back face: first coord = −", font_size=22, color=GREY_C)
        notes = VGroup(front_note, back_note).arrange(DOWN, buff=0.15)
        notes.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(notes, run_time=0.5))
        on_screen.append(notes)
        self.wait(1.0)

        # — Clean up —
        self.play(*[FadeOut(m) for m in on_screen], run_time=0.7)
        self.wait(0.3)

    # ─────────────────────────────────────
    # n = 4  (fast, visual payoff)
    # ─────────────────────────────────────
    def do_n4(self):
        bits = "0001001000110100"
        n = 4
        corners = all_corners(n)
        vals = [int(b) for b in bits]
        scale = 1.6

        on_screen = []

        # — Section label —
        sec_label = Text("n = 4 :  16 corners of a tesseract", font_size=36, color=GREY_C)
        sec_label.to_edge(UP, buff=0.4)
        self.play(FadeIn(sec_label))
        on_screen.append(sec_label)

        # — Bitstring (compact) —
        digits = VGroup()
        for ch in bits:
            v = int(ch)
            d = Text(ch, font_size=36, color=bit_color(v), weight="BOLD")
            digits.add(d)
        digits.arrange(RIGHT, buff=0.12)
        digits.move_to(UP * 1.5)
        fit_to_frame(digits, pad=0.85)
        self.play(FadeIn(digits))
        on_screen.append(digits)
        self.wait(0.3)

        # — Build tesseract —
        center = DOWN * 0.5
        dots = {}
        val_labels = {}

        for i, c in enumerate(corners):
            px, py = hypercube_pos_2d(c, scale=scale)
            pos = center + px * RIGHT + py * UP
            dot = Dot(pos, radius=DOT_RADIUS * 0.85, color=bit_color(vals[i]))
            dot.set_z_index(2)
            dots[c] = dot

            vlab = Text(str(vals[i]), font_size=22, color=bit_color(vals[i]),
                        weight="BOLD")
            vlab.move_to(dot.get_center())
            val_labels[c] = vlab

        # — Fly digits —
        fly_anims = []
        for i, c in enumerate(corners):
            digits[i].generate_target()
            digits[i].target.move_to(dots[c].get_center())
            digits[i].target.set_color(bit_color(vals[i]))
            digits[i].target.scale(0.5)
            fly_anims.append(MoveToTarget(digits[i], rate_func=smooth))

        self.play(*fly_anims, run_time=1.8)

        dot_group = VGroup(*dots.values())
        vlab_group = VGroup(*val_labels.values())

        self.play(
            FadeOut(digits, run_time=0.3),
            FadeIn(dot_group, run_time=0.4),
            FadeIn(vlab_group, run_time=0.4),
        )
        on_screen.remove(digits)
        on_screen.extend([dot_group, vlab_group])

        # — Edges: draw outer cube, inner cube, then cross edges —
        all_edges = hypercube_edges(n)

        # Classify edges by which coordinate differs
        outer_edges = VGroup()  # differ in coords 1,2,3 with c0=0
        inner_edges = VGroup()  # differ in coords 1,2,3 with c0=1
        cross_edges = VGroup()  # differ in coord 0

        for c1, c2 in all_edges:
            line = Line(
                dots[c1].get_center(), dots[c2].get_center(),
                stroke_width=EDGE_WIDTH, color=EDGE_COLOR
            )
            line.set_z_index(0)

            # find which coord differs
            diff_idx = next(k for k in range(n) if c1[k] != c2[k])
            if diff_idx == 0:
                cross_edges.add(line)
            elif c1[0] == 0:
                outer_edges.add(line)
            else:
                inner_edges.add(line)

        # Draw in stages for clarity
        self.play(LaggedStart(*[Create(e) for e in outer_edges],
                               lag_ratio=0.06, run_time=0.8))
        on_screen.append(outer_edges)

        # Inner edges slightly more transparent
        for e in inner_edges:
            e.set_opacity(0.5)
        self.play(LaggedStart(*[Create(e) for e in inner_edges],
                               lag_ratio=0.06, run_time=0.8))
        on_screen.append(inner_edges)

        # Cross edges dashed-ish (thinner)
        for e in cross_edges:
            e.set_stroke(width=1.2, opacity=0.4)
        self.play(LaggedStart(*[Create(e) for e in cross_edges],
                               lag_ratio=0.04, run_time=0.6))
        on_screen.append(cross_edges)
        self.wait(0.3)

        # — Flash 1-bits —
        one_dots = [dots[c] for i, c in enumerate(corners) if vals[i] == 1]
        if one_dots:
            flash_note = Text("1-bits form a substructure", font_size=26, color=YELLOW)
            flash_note.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(flash_note))
            on_screen.append(flash_note)

            self.play(*[Indicate(d, color=YELLOW, scale_factor=1.4) for d in one_dots],
                      run_time=1.0)
            self.wait(0.8)

        # — Annotation —
        anno = Text("outer cube = first coord +    inner cube = first coord −",
                     font_size=22, color=GREY_C)
        anno.to_edge(DOWN, buff=0.3)
        # check if flash_note is still there — place above it
        if one_dots:
            anno.next_to(flash_note, UP, buff=0.15)
        self.play(FadeIn(anno, run_time=0.4))
        on_screen.append(anno)
        self.wait(1.2)

        # — Node count annotation —
        count_text = Text(
            f"2⁴ = 16 vertices,  {len(all_edges)} edges",
            font_size=24, color=GREY_C
        )
        count_text.next_to(sec_label, DOWN, buff=0.2)
        self.play(FadeIn(count_text, run_time=0.4))
        on_screen.append(count_text)
        self.wait(1.0)

        # — Clean up —
        self.play(*[FadeOut(m) for m in on_screen], run_time=0.8)
        self.wait(0.4)