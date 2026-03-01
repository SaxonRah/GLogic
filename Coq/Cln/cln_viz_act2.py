#!/usr/bin/env python3
"""
cln_viz_act2.py — Act 2: "CNF formulas are bipartite graphs"

Animated 3Blue1Brown-style morph from CNF formula to clause-variable
incidence graph, with a satisfying-assignment check at the end.

Run:
  manim -pql cln_viz_act2.py Act2CNF
  manim -pqh cln_viz_act2.py Act2CNF
"""

from __future__ import annotations
from typing import List, Tuple, Dict

from manim import (
    Scene, VGroup, Dot, Text, MathTex, Line, DashedLine,
    FadeIn, FadeOut, Create, LaggedStart, MoveToTarget,
    Write, Flash, Indicate, GrowFromCenter,
    UP, DOWN, LEFT, RIGHT, ORIGIN, config,
    RoundedRectangle, SurroundingRectangle,
    BLUE_D, BLUE_C, YELLOW, GOLD_C, GREY_B, GREY_C, WHITE,
    GREEN_C, GREEN_D, RED_B, RED_C, TEAL_C, TEAL_D,
    MAROON_B, PURPLE, ORANGE,
    smooth,
)

# ─────────────────────────────────────────────
# Color palette (shared with Act 1)
# ─────────────────────────────────────────────
BIT_ZERO   = BLUE_D
BIT_ONE    = YELLOW
VAR_COLOR  = BLUE_C
NEG_COLOR  = RED_B
POS_COLOR  = GREEN_C
EDGE_COLOR = GREY_B
BG_COLOR   = "#1a1a2e"

# Clause-specific colors
CLAUSE_COLORS = [TEAL_C, GREEN_D, PURPLE, GOLD_C]

EDGE_POS_COLOR = GREEN_C
EDGE_NEG_COLOR = RED_B

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


def make_clause_node(index: int, color, pos) -> VGroup:
    """Create a clause node: rounded rect + C_i label."""
    rect = RoundedRectangle(
        width=1.1, height=0.7, corner_radius=0.15,
        stroke_color=color, stroke_width=2.5,
        fill_color=color, fill_opacity=0.12,
    )
    label = MathTex(f"C_{index}", font_size=30, color=color)
    label.move_to(rect.get_center())
    group = VGroup(rect, label)
    group.move_to(pos)
    return group


def make_var_node(var_name: str, pos) -> VGroup:
    """Create a variable node: rounded rect + label."""
    rect = RoundedRectangle(
        width=1.1, height=0.7, corner_radius=0.15,
        stroke_color=VAR_COLOR, stroke_width=2.5,
        fill_color=VAR_COLOR, fill_opacity=0.12,
    )
    label = MathTex(var_name, font_size=30, color=VAR_COLOR)
    label.move_to(rect.get_center())
    group = VGroup(rect, label)
    group.move_to(pos)
    return group


# ─────────────────────────────────────────────
# Act 2 Scene
# ─────────────────────────────────────────────

class Act2CNF(Scene):

    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Title card ──
        title = Text(
            "CNF formulas are bipartite graphs",
            font_size=48, color=WHITE,
        ).move_to(ORIGIN)
        self.play(FadeIn(title, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(title, shift=UP * 0.3))
        self.wait(0.3)

        self.do_cnf_morph()

    def do_cnf_morph(self):
        # ─────────────────────────────────────
        # Data
        # ─────────────────────────────────────
        clauses: List[List[Tuple[str, bool]]] = [
            [("x_0", True),  ("x_1", False), ("x_2", True)],   # C0
            [("x_0", False), ("x_3", True)],                     # C1
            [("x_1", True),  ("x_2", False), ("x_3", True)],   # C2
            [("x_3", False), ("x_2", True)],                     # C3
        ]
        vars_list = ["x_0", "x_1", "x_2", "x_3"]
        assignment = {"x_0": True, "x_1": False, "x_2": True, "x_3": True}

        # Clause content strings (for subtitles later)
        clause_content_tex = [
            r"x_0 \lor \neg x_1 \lor x_2",
            r"\neg x_0 \lor x_3",
            r"x_1 \lor \neg x_2 \lor x_3",
            r"\neg x_3 \lor x_2",
        ]

        # Layout positions
        clause_x = -4.2
        var_x = 4.2
        # 4 items, centered vertically around y=0.3 (slight upward shift)
        def clause_y(i):
            return 1.8 - i * 1.2 + 0.3
        def var_y(i):
            return 1.8 - i * 1.2 + 0.3

        on_screen = []  # strict object-permanence tracker

        # ─────────────────────────────────────
        # Phase 1: Show the formula
        # ─────────────────────────────────────
        sec_label = Text("CNF formula", font_size=34, color=GREY_C)
        sec_label.to_edge(UP, buff=0.35)
        self.play(FadeIn(sec_label))
        on_screen.append(sec_label)

        # Build formula as colored clause groups with grey ∧ between them
        formula_parts = []
        clause_texs = []  # just the clause MathTex objects
        conj_texs = []     # just the ∧ objects

        for i, ct in enumerate(clause_content_tex):
            part = MathTex(rf"({ct})", font_size=32)
            part.set_color(CLAUSE_COLORS[i])
            clause_texs.append(part)
            formula_parts.append(part)
            if i < len(clause_content_tex) - 1:
                conj = MathTex(r"\land", font_size=32, color=GREY_B)
                conj_texs.append(conj)
                formula_parts.append(conj)

        formula_group = VGroup(*formula_parts)
        formula_group.arrange(RIGHT, buff=0.18)
        formula_group.next_to(sec_label, DOWN, buff=0.5)
        fit_to_frame(formula_group, pad=0.92)

        self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.15) for p in formula_parts],
                               lag_ratio=0.08, run_time=1.4))
        on_screen.append(formula_group)
        self.wait(0.6)

        # ─────────────────────────────────────
        # Phase 2: Clauses detach → become nodes
        # ─────────────────────────────────────
        # Update section label
        sec_label2 = Text("decompose into clauses", font_size=34, color=GREY_C)
        sec_label2.to_edge(UP, buff=0.35)

        # Create clause nodes at their target positions
        clause_nodes = []
        clause_node_group = VGroup()
        for i in range(len(clauses)):
            pos = clause_x * RIGHT + clause_y(i) * UP
            node = make_clause_node(i, CLAUSE_COLORS[i], pos)
            clause_nodes.append(node)
            clause_node_group.add(node)

        # Animate: conjunctions fade out, clause texts shrink and move left,
        # then clause nodes replace them
        move_anims = []
        for i, ct in enumerate(clause_texs):
            ct.generate_target()
            ct.target.move_to(clause_x * RIGHT + clause_y(i) * UP)
            ct.target.scale(0.55)
            move_anims.append(MoveToTarget(ct, rate_func=smooth))

        self.play(
            *[FadeOut(c, run_time=0.8) for c in conj_texs],
            *move_anims,
            FadeOut(sec_label), FadeIn(sec_label2),
            run_time=1.5,
        )
        on_screen.remove(formula_group)
        on_screen.append(sec_label2)
        # clause_texs are still on screen (shrunk, at left positions)

        # Swap clause texts for proper nodes
        self.play(
            *[FadeOut(ct, run_time=0.3) for ct in clause_texs],
            *[FadeIn(cn, run_time=0.5) for cn in clause_nodes],
            run_time=0.6,
        )
        on_screen.append(clause_node_group)
        self.wait(0.3)

        # Add small content subtitles below each clause node
        clause_subs = []
        clause_sub_group = VGroup()
        for i, ct_str in enumerate(clause_content_tex):
            sub = MathTex(ct_str, font_size=18, color=GREY_C)
            sub.next_to(clause_nodes[i], DOWN, buff=0.08)
            clause_subs.append(sub)
            clause_sub_group.add(sub)

        self.play(FadeIn(clause_sub_group, run_time=0.5))
        on_screen.append(clause_sub_group)
        self.wait(0.4)

        # ─────────────────────────────────────
        # Phase 3: Variables coalesce
        # ─────────────────────────────────────
        sec_label3 = Text("extract shared variables", font_size=34, color=GREY_C)
        sec_label3.to_edge(UP, buff=0.35)
        self.play(FadeOut(sec_label2), FadeIn(sec_label3), run_time=0.5)
        on_screen.remove(sec_label2)
        on_screen.append(sec_label3)

        var_nodes: Dict[str, VGroup] = {}
        var_node_group = VGroup()

        for vi, var in enumerate(vars_list):
            # Find which clauses contain this variable
            containing_clauses = []
            for ci, cl in enumerate(clauses):
                for (v, pol) in cl:
                    if v == var:
                        containing_clauses.append(ci)

            # Create small copies of the variable name at each containing clause
            copies = VGroup()
            for ci in containing_clauses:
                copy = MathTex(var, font_size=26, color=VAR_COLOR)
                copy.move_to(clause_nodes[ci].get_center())
                copies.add(copy)

            # Flash the containing clause nodes
            self.play(
                *[Indicate(clause_nodes[ci], color=VAR_COLOR, scale_factor=1.1)
                  for ci in containing_clauses],
                run_time=0.5,
            )

            # Show the copies emerging from clause nodes
            self.play(FadeIn(copies, run_time=0.3))

            # Set up target position for variable node
            target_pos = var_x * RIGHT + var_y(vi) * UP

            # Fly copies to the target position
            for copy in copies:
                copy.generate_target()
                copy.target.move_to(target_pos)
            self.play(
                *[MoveToTarget(c, rate_func=smooth) for c in copies],
                run_time=0.7,
            )

            # Create the variable node and swap in
            var_node = make_var_node(var, target_pos)
            var_nodes[var] = var_node
            var_node_group.add(var_node)

            self.play(
                FadeOut(copies, run_time=0.2),
                GrowFromCenter(var_node, run_time=0.4),
            )
            self.wait(0.15)

        on_screen.append(var_node_group)

        # Sharing annotation (brief)
        share_note = Text(
            "each variable connects to multiple clauses",
            font_size=24, color=GREY_C,
        )
        share_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(share_note, run_time=0.4))
        self.wait(0.6)
        self.play(FadeOut(share_note, run_time=0.4))

        # ─────────────────────────────────────
        # Phase 4: Draw edges
        # ─────────────────────────────────────
        sec_label4 = Text("connect literals to variables", font_size=34, color=GREY_C)
        sec_label4.to_edge(UP, buff=0.35)
        self.play(FadeOut(sec_label3), FadeIn(sec_label4), run_time=0.5)
        on_screen.remove(sec_label3)
        on_screen.append(sec_label4)

        # Build edges, stored by (clause_idx, var_name) for later use
        edge_mobjects: Dict[Tuple[int, str], Line] = {}
        all_edges_group = VGroup()

        # Draw edges clause by clause (staggered)
        for ci, cl in enumerate(clauses):
            clause_edges = []
            for (var, pol) in cl:
                start = clause_nodes[ci].get_right()
                end = var_nodes[var].get_left()

                if pol:
                    edge = Line(
                        start, end,
                        stroke_width=2.0, color=EDGE_POS_COLOR,
                    )
                else:
                    edge = DashedLine(
                        start, end,
                        stroke_width=2.0, color=EDGE_NEG_COLOR,
                        dash_length=0.12,
                    )

                edge.set_z_index(-1)
                edge_mobjects[(ci, var)] = edge
                all_edges_group.add(edge)
                clause_edges.append(edge)

            self.play(
                *[Create(e) for e in clause_edges],
                run_time=0.6,
            )

        on_screen.append(all_edges_group)
        self.wait(0.3)

        # Legend
        legend = VGroup()
        pos_sample = Line(LEFT * 0.4, RIGHT * 0.4, stroke_width=2.0, color=EDGE_POS_COLOR)
        pos_label = Text("positive literal", font_size=20, color=EDGE_POS_COLOR)
        pos_label.next_to(pos_sample, RIGHT, buff=0.15)
        neg_sample = DashedLine(LEFT * 0.4, RIGHT * 0.4, stroke_width=2.0,
                                color=EDGE_NEG_COLOR, dash_length=0.12)
        neg_label = Text("negative literal", font_size=20, color=EDGE_NEG_COLOR)
        neg_label.next_to(neg_sample, RIGHT, buff=0.15)
        legend.add(VGroup(pos_sample, pos_label))
        legend.add(VGroup(neg_sample, neg_label))
        legend.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        legend.to_edge(DOWN, buff=0.3)

        self.play(FadeIn(legend, run_time=0.5))
        on_screen.append(legend)
        self.wait(0.8)

        # Fade out legend and section label, keep graph
        self.play(FadeOut(legend), FadeOut(sec_label4), run_time=0.5)
        on_screen.remove(legend)
        on_screen.remove(sec_label4)

        # ─────────────────────────────────────
        # Phase 5: Assignment check
        # ─────────────────────────────────────
        check_label = Text("check a satisfying assignment", font_size=34, color=GREY_C)
        check_label.to_edge(UP, buff=0.35)
        self.play(FadeIn(check_label))
        on_screen.append(check_label)

        # Show assignment text
        assign_str = r"x_0 = \top,\; x_1 = \bot,\; x_2 = \top,\; x_3 = \top"
        assign_tex = MathTex(assign_str, font_size=30, color=WHITE)
        assign_tex.next_to(check_label, DOWN, buff=0.25)
        self.play(FadeIn(assign_tex, run_time=0.5))
        on_screen.append(assign_tex)
        self.wait(0.4)

        # Add T/F badges to variable nodes
        badges = {}
        badge_group = VGroup()
        for var in vars_list:
            val = assignment[var]
            badge_text = "T" if val else "F"
            badge_color = BIT_ONE if val else BIT_ZERO
            badge = Text(badge_text, font_size=22, color=badge_color, weight="BOLD")
            badge.next_to(var_nodes[var], RIGHT, buff=0.18)
            badges[var] = badge
            badge_group.add(badge)

        self.play(LaggedStart(*[FadeIn(b, shift=LEFT * 0.1) for b in badge_group],
                               lag_ratio=0.12, run_time=0.8))
        on_screen.append(badge_group)
        self.wait(0.3)

        # Check each clause
        check_marks = []
        check_group = VGroup()

        for ci, cl in enumerate(clauses):
            # Highlight clause node
            self.play(Indicate(clause_nodes[ci], color=WHITE, scale_factor=1.08),
                      run_time=0.3)

            clause_satisfied = False
            edge_anims = []
            for (var, pol) in cl:
                val = assignment[var]
                lit_val = val if pol else (not val)
                edge = edge_mobjects[(ci, var)]

                if lit_val:
                    # Satisfied: flash green, thicken
                    edge_anims.append(
                        edge.animate.set_color(GREEN_C).set_stroke(width=4.0)
                    )
                    clause_satisfied = True
                else:
                    # Not satisfied: flash red, dim
                    edge_anims.append(
                        edge.animate.set_color(RED_C).set_stroke(opacity=0.4)
                    )

            self.play(*edge_anims, run_time=0.5)

            # Mark clause as satisfied or not
            if clause_satisfied:
                # Green fill + checkmark
                self.play(
                    clause_nodes[ci][0].animate.set_fill(GREEN_C, opacity=0.3),
                    run_time=0.3,
                )
                check = Text("✓", font_size=22, color=GREEN_C)
                check.next_to(clause_nodes[ci], LEFT, buff=0.12)
            else:
                self.play(
                    clause_nodes[ci][0].animate.set_fill(RED_B, opacity=0.3),
                    run_time=0.3,
                )
                check = Text("✗", font_size=22, color=RED_B)
                check.next_to(clause_nodes[ci], LEFT, buff=0.12)

            self.play(FadeIn(check, run_time=0.2))
            check_marks.append(check)
            check_group.add(check)
            self.wait(0.15)

        on_screen.append(check_group)
        self.wait(0.3)

        # Final verdict
        all_sat = all(
            any(
                (assignment[v] if pol else not assignment[v])
                for (v, pol) in cl
            )
            for cl in clauses
        )

        if all_sat:
            verdict = Text("All clauses satisfied  →  SAT ✓", font_size=36,
                           color=GREEN_C, weight="BOLD")
        else:
            verdict = Text("Unsatisfied clause  →  UNSAT ✗", font_size=36,
                           color=RED_B, weight="BOLD")

        verdict.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(verdict, shift=UP * 0.2), run_time=0.6)
        on_screen.append(verdict)
        self.wait(1.5)

        # ─────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────
        # Also need to fade the clause subs which are still on screen
        self.play(*[FadeOut(m) for m in on_screen], run_time=0.8)
        self.wait(0.4)