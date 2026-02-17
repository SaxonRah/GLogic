"""
Hyper-Dual Algebra — A Geometric View of Computation and Complexity
20-minute cinematic exposition

Requires: manim (community edition)
Run:  manim -pqh hyper_dual.py FullFilm
      manim -pql hyper_dual.py Opening        # test individual scenes
"""

from manim import *
import numpy as np
import itertools

# ─── Color palette (ManimColor objects to avoid interpolate_color bugs) ───
SEMANTIC_BLUE = ManimColor("#4A90D9")
DYNAMIC_RED = ManimColor("#E74C3C")
GRADE_GOLD = ManimColor("#F1C40F")
STABILITY_GREEN = ManimColor("#2ECC71")
ACCENT_PURPLE = ManimColor("#9B59B6")
TRUTH_CYAN = ManimColor("#00CED1")
MOTION_ORANGE = ManimColor("#FF6B35")
BG_DARK = "#1a1a2e"  # string is fine for background_color


# ─── Utility ─────────────────────────────────────────────────
def narration_text(txt, **kwargs):
    """Consistent narration styling."""
    return Text(txt, font_size=24, color=GREY_A, line_spacing=1.3, **kwargs)


def theorem_box(tex, label_str, label_color, font_size=36):
    """Framed theorem with label underneath."""
    eq = MathTex(tex, font_size=font_size, color=WHITE)
    label = Text(label_str, font_size=20, color=label_color)
    label.next_to(eq, DOWN, buff=0.25)
    group = VGroup(eq, label)
    box = SurroundingRectangle(group, color=label_color, buff=0.3, corner_radius=0.1)
    box.set_fill(BG_DARK, opacity=0.85)
    return VGroup(box, eq, label)


def section_title(act_str, title_str, title_color=WHITE):
    """Consistent section header."""
    act = Text(act_str, font_size=26, color=GREY_B).to_edge(UP, buff=0.3)
    title = Text(title_str, font_size=40, color=title_color)
    title.next_to(act, DOWN, buff=0.15)
    return VGroup(act, title)


# ═══════════════════════════════════════════════════════════════
# OPENING — Two Worlds (0:00 – 2:00)
# ═══════════════════════════════════════════════════════════════

class Opening(Scene):
    """The map, the terrain, and the hyper-duality declaration."""

    def construct(self):
        self.camera.background_color = BG_DARK
        self._the_map()
        self._the_terrain()
        self._hyper_duality_declaration()

    # ── The Map ─────────────────────────────────────────
    def _the_map(self):
        grid = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": BLUE_E, "stroke_opacity": 0.25},
            axis_config={"stroke_color": BLUE_D, "stroke_opacity": 0.4},
        ).scale(0.8)

        map_tag = Text("THE MAP", font_size=34, color=SEMANTIC_BLUE).to_edge(UP, buff=0.4)

        start = Dot(grid.c2p(-2, -2), color=WHITE, radius=0.1)
        start_l = Text("Start", font_size=18, color=WHITE).next_to(start, DL, buff=0.1)
        grandma = Dot(grid.c2p(2, 2), color=GRADE_GOLD, radius=0.12)
        grandma_l = Text("Grandma's", font_size=18, color=GRADE_GOLD).next_to(grandma, UR, buff=0.1)

        east = Arrow(grid.c2p(-2, -2), grid.c2p(2, -2),
                     color=SEMANTIC_BLUE, stroke_width=3, buff=0.08)
        north = Arrow(grid.c2p(2, -2), grid.c2p(2, 2),
                      color=SEMANTIC_BLUE, stroke_width=3, buff=0.08)
        e_lab = Text("East", font_size=18, color=SEMANTIC_BLUE).next_to(east, DOWN, buff=0.08)
        n_lab = Text("North", font_size=18, color=SEMANTIC_BLUE).next_to(north, RIGHT, buff=0.08)

        self.play(Create(grid), Write(map_tag), run_time=1.5)
        self.play(FadeIn(start), FadeIn(start_l),
                  FadeIn(grandma), FadeIn(grandma_l), run_time=0.8)
        self.play(GrowArrow(east), FadeIn(e_lab), run_time=0.8)
        self.play(GrowArrow(north), FadeIn(n_lab), run_time=0.8)

        narr1 = narration_text("On this map, directions combine perfectly.")
        narr1.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(narr1))
        self.wait(1.2)
        self.play(FadeOut(narr1))

        narr2 = narration_text("Nothing bends. Nothing interferes. Everything is flat.")
        narr2.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(narr2))
        self.wait(1.5)
        self.play(FadeOut(narr2))

        # Untwisted product
        eq = MathTex(r"e_A \odot e_B = e_{A \oplus B}", font_size=38, color=SEMANTIC_BLUE)
        eq.to_edge(DOWN, buff=0.7)
        tag = Text("Untwisted convolution — the algebra of truth",
                    font_size=20, color=GREY_B)
        tag.next_to(eq, DOWN, buff=0.2)
        self.play(Write(eq), FadeIn(tag), run_time=1.5)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ── The Terrain ─────────────────────────────────────
    def _the_terrain(self):
        terrain_tag = Text("THE TERRAIN", font_size=34, color=DYNAMIC_RED)
        terrain_tag.to_edge(UP, buff=0.4)

        # 2D mountain impression — layered sine waves with gradient
        mountains = VGroup()
        configs = [
            (0.7, 0.0, -0.8, 0.0),
            (1.0, 1.0, -0.1, 0.33),
            (0.5, 2.2, 0.5, 0.66),
            (0.9, 0.7, 1.1, 1.0),
        ]
        for amp, phase, y_off, t in configs:
            wave = FunctionGraph(
                lambda x, a=amp, p=phase, yo=y_off: a * np.sin(x * 1.5 + p) + yo,
                x_range=[-5.5, 5.5],
                color=interpolate_color(SEMANTIC_BLUE, DYNAMIC_RED, t),
                stroke_width=2.5,
                fill_opacity=0.08,
            )
            mountains.add(wave)

        self.play(Write(terrain_tag),
                  LaggedStart(*[Create(m) for m in mountains], lag_ratio=0.15),
                  run_time=2)

        narr = narration_text(
            "Same coordinates. Same destinations.\nDifferent physics."
        )
        narr.to_edge(DOWN, buff=1.0)
        self.play(FadeIn(narr))
        self.wait(1.5)
        self.play(FadeOut(narr))

        # Twisted product
        eq = MathTex(
            r"e_A \star e_B = \omega(A,B)\, e_{A \oplus B}",
            font_size=38, color=DYNAMIC_RED,
        ).to_edge(DOWN, buff=0.7)
        tag = Text("Twisted multiplication — the algebra of motion",
                    font_size=20, color=GREY_B)
        tag.next_to(eq, DOWN, buff=0.2)
        self.play(Write(eq), FadeIn(tag), run_time=1.5)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ── Hyper-Duality Declaration ───────────────────────
    def _hyper_duality_declaration(self):
        title = Text("Hyper-Duality", font_size=52, color=WHITE)
        self.play(Write(title), run_time=1)
        self.play(title.animate.to_edge(UP, buff=0.5), run_time=0.7)

        narr = narration_text("We did not change the space.\nWe changed the multiplication rule.")
        narr.shift(UP * 0.8)
        self.play(FadeIn(narr))
        self.wait(2)
        self.play(FadeOut(narr))

        lines = VGroup(
            Text("Same representation.", font_size=28, color=GREY_A),
            Text("Two compatible multiplications.", font_size=28, color=GREY_A),
            Text("One interprets truth.", font_size=28, color=TRUTH_CYAN),
            Text("The other governs dynamics.", font_size=28, color=MOTION_ORANGE),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to(ORIGIN)

        for line in lines:
            self.play(FadeIn(line, shift=RIGHT * 0.3), run_time=0.6)
            self.wait(0.4)
        self.wait(2)

        # Side-by-side
        left = VGroup(
            MathTex(r"e_A \odot e_B = e_{A \oplus B}", font_size=30, color=SEMANTIC_BLUE),
            Text("Truth", font_size=22, color=SEMANTIC_BLUE),
        ).arrange(DOWN, buff=0.12)
        right = VGroup(
            MathTex(r"e_A \star e_B = \omega \cdot e_{A \oplus B}", font_size=30, color=DYNAMIC_RED),
            Text("Motion", font_size=22, color=DYNAMIC_RED),
        ).arrange(DOWN, buff=0.12)
        comp = VGroup(left, right).arrange(RIGHT, buff=1.8).shift(DOWN * 0.5)

        self.play(FadeOut(lines), FadeIn(comp), run_time=1)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# PART I — The Underlying Space (2:00 – 5:00)
# ═══════════════════════════════════════════════════════════════

class PartI_Space(Scene):
    """Formalize the representation: corners, masks, MV_n."""

    def construct(self):
        self.camera.background_color = BG_DARK

        header = section_title("PART I", "The Underlying Space")
        self.play(FadeIn(header), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(header))

        # ── Fix n ──
        fix = MathTex(r"\text{Fix } n.", font_size=36, color=WHITE)
        group = MathTex(r"G = (\mathbb{Z}_2)^n", font_size=38, color=WHITE)
        group.next_to(fix, DOWN, buff=0.5)
        self.play(Write(fix))
        self.play(Write(group), run_time=1)
        self.wait(1)
        self.play(FadeOut(fix), group.animate.to_edge(UP, buff=0.8))

        # Two pictures
        corners_box = VGroup(
            Text("Corners", font_size=26, color=TRUTH_CYAN),
            MathTex(r"\{\pm 1\}^n", font_size=32, color=TRUTH_CYAN),
        ).arrange(DOWN, buff=0.15)

        masks_box = VGroup(
            Text("Masks", font_size=26, color=MOTION_ORANGE),
            MathTex(r"A \subseteq [n]", font_size=32, color=MOTION_ORANGE),
        ).arrange(DOWN, buff=0.15)

        pictures = VGroup(corners_box, masks_box).arrange(RIGHT, buff=2.0)
        pictures.next_to(group, DOWN, buff=0.7)

        self.play(FadeIn(corners_box, shift=LEFT * 0.3),
                  FadeIn(masks_box, shift=RIGHT * 0.3), run_time=1.2)
        self.wait(1.5)

        # Hypercube visualization for n=3
        cube = self._build_hypercube().scale(0.9).shift(DOWN * 1.2)
        self.play(FadeIn(cube), run_time=1.5)
        self.wait(1)

        # MV_n definition
        mv_eq = MathTex(
            r"\mathrm{MV}_n = \big\{ F : \mathrm{Mask}_n \to \mathbb{Q} \big\}",
            font_size=34, color=WHITE,
        ).to_edge(DOWN, buff=0.7)
        self.play(Write(mv_eq), run_time=1.5)
        self.wait(1)

        expand = MathTex(
            r"F = \sum_A F(A)\, e_A",
            font_size=34, color=GREY_A,
        )
        expand.next_to(mv_eq, UP, buff=0.3)
        self.play(Write(expand))
        self.wait(1)

        tag = narration_text("This is the group algebra  Q[G].  Nothing exotic yet.")
        tag.to_edge(DOWN, buff=0.3)
        self.play(FadeOut(mv_eq), expand.animate.shift(DOWN * 0.3), FadeIn(tag))
        self.wait(2)

        novelty = narration_text(
            "The novelty: two monoidal structures on the same space."
        )
        novelty.to_edge(DOWN, buff=0.3)
        self.play(FadeOut(tag), FadeIn(novelty))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def _build_hypercube(self):
        """3-cube wireframe with labeled vertices."""
        verts = {}
        dots = VGroup()
        for bits in itertools.product([0, 1], repeat=3):
            pos = np.array([
                (bits[0] - 0.5) * 2.0,
                (bits[1] - 0.5) * 2.0,
                0,
            ]) + np.array([bits[2] * 0.6, bits[2] * 0.6, 0])
            d = Dot(pos, radius=0.06, color=TRUTH_CYAN)
            verts[bits] = pos
            dots.add(d)

        edges = VGroup()
        for b1 in itertools.product([0, 1], repeat=3):
            for i in range(3):
                b2 = list(b1)
                b2[i] = 1 - b2[i]
                b2 = tuple(b2)
                if b1 < b2:
                    line = Line(verts[b1], verts[b2],
                                stroke_width=1.2, color=GREY_D)
                    edges.add(line)
        return VGroup(edges, dots)


# ═══════════════════════════════════════════════════════════════
# PART II — The Untwisted Product: Semantics (5:00 – 8:00)
# ═══════════════════════════════════════════════════════════════

class PartII_Semantics(Scene):
    """Untwisted product, Walsh evaluation, embed_correct, Pi_delta."""

    def construct(self):
        self.camera.background_color = BG_DARK

        header = section_title("PART II", "The Untwisted Product: Semantics", SEMANTIC_BLUE)
        self.play(FadeIn(header), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(header))

        # ── Group algebra product ──
        prod = MathTex(r"e_A \odot e_B = e_{A \oplus B}", font_size=42, color=SEMANTIC_BLUE)
        props = VGroup(
            Text("Associative", font_size=22, color=GREY_B),
            Text("Commutative", font_size=22, color=GREY_B),
            Text("Pure XOR", font_size=22, color=GREY_B),
        ).arrange(RIGHT, buff=0.8).next_to(prod, DOWN, buff=0.4)

        self.play(Write(prod), run_time=1.2)
        self.play(FadeIn(props), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(props), prod.animate.to_edge(UP, buff=0.7))

        # ── Walsh characters ──
        chi_eq = MathTex(
            r"\chi_A(s) = (-1)^{\langle A, s \rangle}",
            font_size=36, color=WHITE,
        )
        self.play(Write(chi_eq), run_time=1.2)
        self.wait(1)

        eval_eq = MathTex(
            r"\mathrm{eval}(F)(s) = \sum_A F(A)\, \chi_A(s)",
            font_size=34, color=WHITE,
        ).next_to(chi_eq, DOWN, buff=0.5)
        self.play(Write(eval_eq), run_time=1.2)

        narr = narration_text("Turns multivectors into functions on corners.")
        narr.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(narr))
        self.wait(1.5)
        self.play(FadeOut(narr), FadeOut(chi_eq), FadeOut(eval_eq))

        # ── embed_correct ──
        thm = theorem_box(
            r"\mathrm{eval}\big(\mathrm{embed}(f)\big)(s) = f(s)",
            "embed_correct",
            TRUTH_CYAN,
            font_size=40,
        )
        self.play(FadeIn(thm), run_time=1.5)
        narr2 = narration_text("Embedding and evaluation recover Boolean truth exactly.")
        narr2.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(narr2))
        self.wait(2)
        self.play(FadeOut(narr2), FadeOut(thm))

        # ── Pi_delta ──
        # Spotlight visual
        verts = VGroup()
        n_pts = 8
        for i in range(n_pts):
            angle = i * TAU / n_pts
            pos = 1.5 * np.array([np.cos(angle), np.sin(angle), 0])
            color = GRADE_GOLD if i == 0 else GREY_D
            radius = 0.12 if i == 0 else 0.07
            verts.add(Dot(pos, color=color, radius=radius))
        verts.shift(LEFT * 2.5)

        # Glow around the selected vertex
        glow = Circle(radius=0.35, color=GRADE_GOLD, stroke_width=2,
                      fill_color=GRADE_GOLD, fill_opacity=0.15)
        glow.move_to(verts[0])

        pi_eq = MathTex(
            r"\mathrm{eval}\big(\Pi(a)\big)(s) = "
            r"\begin{cases} 1 & s=a \\ 0 & s \neq a \end{cases}",
            font_size=32, color=WHITE,
        ).shift(RIGHT * 2)

        pi_label = Text("Pi_delta — Fourier orthogonality", font_size=20, color=ACCENT_PURPLE)
        pi_label.next_to(pi_eq, DOWN, buff=0.3)

        self.play(FadeIn(verts), FadeIn(glow), run_time=1)
        self.play(Write(pi_eq), FadeIn(pi_label), run_time=1.5)
        self.wait(2)

        closing = narration_text("The untwisted algebra is semantically faithful.\nIt defines truth.")
        closing.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(closing))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# CONCEPTUAL PAUSE
# ═══════════════════════════════════════════════════════════════

class ConceptualPause(Scene):
    """Brief visual recap before deformation."""

    def construct(self):
        self.camera.background_color = BG_DARK

        items = VGroup(
            Text("So far:", font_size=32, color=WHITE),
            Text("The space is fixed.", font_size=26, color=GREY_A),
            Text("Untwisted multiplication governs semantics.", font_size=26, color=SEMANTIC_BLUE),
            Text("Evaluation is homomorphic under  ⊙.", font_size=26, color=TRUTH_CYAN),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to(UP * 0.5)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.6)
            self.wait(0.4)
        self.wait(1.5)

        now = Text("Now we introduce deformation.", font_size=32, color=DYNAMIC_RED)
        now.shift(DOWN * 1.5)
        self.play(Write(now), run_time=1)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# PART III — The Twisted Product: Dynamics (8:00 – 11:00)
# ═══════════════════════════════════════════════════════════════

class PartIII_Dynamics(Scene):
    """Twisted product, cocycle, deformation geometry."""

    def construct(self):
        self.camera.background_color = BG_DARK

        header = section_title("PART III", "The Twisted Product: Dynamics", DYNAMIC_RED)
        self.play(FadeIn(header), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(header))

        # ── Twisted product ──
        prod = MathTex(
            r"e_A \star e_B = \omega(A,B)\, e_{A \oplus B}",
            font_size=42, color=DYNAMIC_RED,
        )
        self.play(Write(prod), run_time=1.5)
        self.wait(1)
        self.play(prod.animate.to_edge(UP, buff=0.7))

        # omega explanation
        omega_items = VGroup(
            MathTex(r"\omega", font_size=32, color=DYNAMIC_RED),
            Text("= 2-cocycle:", font_size=24, color=GREY_A),
        ).arrange(RIGHT, buff=0.2).shift(UP * 0.3)

        details = VGroup(
            Text("swap parity sign", font_size=22, color=GREY_B),
            Text("metric factor from quadratic form", font_size=22, color=GREY_B),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).next_to(omega_items, DOWN, buff=0.4)

        self.play(FadeIn(omega_items), FadeIn(details), run_time=1.2)
        self.wait(1.5)

        narr = narration_text("Masks still XOR.  But coefficients twist.")
        narr.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(narr))
        self.wait(1.5)
        self.play(FadeOut(omega_items), FadeOut(details), FadeOut(narr))

        # ── Cocycle condition ──
        cocycle = MathTex(
            r"\omega(A,B)\,\omega(A \oplus B, C)"
            r"= \omega(B,C)\,\omega(A, B \oplus C)",
            font_size=32, color=WHITE,
        )
        cocycle_label = Text("Cocycle condition  ⟹  associativity", font_size=22, color=GREY_B)
        cocycle_label.next_to(cocycle, DOWN, buff=0.3)

        self.play(Write(cocycle), FadeIn(cocycle_label), run_time=2)
        self.wait(2)

        # ── "Not altered the space" ──
        self.play(FadeOut(cocycle), FadeOut(cocycle_label), FadeOut(prod))

        key = VGroup(
            Text("We have not altered the underlying vector space.", font_size=26, color=GREY_A),
            Text("We have altered the multiplication.", font_size=26, color=DYNAMIC_RED),
            Text("And this changes geometry.", font_size=26, color=WHITE),
        ).arrange(DOWN, buff=0.3).move_to(ORIGIN)

        for line in key:
            self.play(FadeIn(line, shift=UP * 0.15), run_time=0.7)
            self.wait(0.5)
        self.wait(2)

        # ── Deformation visual ──
        # Animate a flat grid warping
        flat_grid = VGroup()
        for i in range(-3, 4):
            h_line = Line(LEFT * 3 + UP * i * 0.5, RIGHT * 3 + UP * i * 0.5,
                          stroke_width=1, color=SEMANTIC_BLUE, stroke_opacity=0.4)
            v_line = Line(UP * 1.5 + RIGHT * i * 0.5, DOWN * 1.5 + RIGHT * i * 0.5,
                          stroke_width=1, color=SEMANTIC_BLUE, stroke_opacity=0.4)
            flat_grid.add(h_line, v_line)
        flat_grid.shift(DOWN * 0.5)

        def warp_point(p):
            x, y, z = p
            return np.array([
                x + 0.15 * np.sin(y * 2.5),
                y + 0.15 * np.cos(x * 2.5),
                z,
            ])

        warped_grid = flat_grid.copy()
        for mob in warped_grid:
            mob.set_color(DYNAMIC_RED)
            mob.set_stroke(opacity=0.5)
            mob.apply_function(warp_point)

        self.play(FadeOut(key), run_time=0.5)
        self.play(Create(flat_grid), run_time=1)
        self.wait(0.5)
        self.play(Transform(flat_grid, warped_grid), run_time=2.5)

        deform_text = narration_text("Cocycle deformation: flat → curved.")
        deform_text.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(deform_text))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# PART IV — Programs as Routes (11:00 – 13:00)
# ═══════════════════════════════════════════════════════════════

class PartIV_Routes(Scene):
    """Algorithms as trajectories through twisted geometry."""

    def construct(self):
        self.camera.background_color = BG_DARK

        header = section_title("PART IV", "Programs as Routes", GRADE_GOLD)
        self.play(FadeIn(header), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(header))

        # ── The split ──
        split_left = VGroup(
            Text("Truth", font_size=28, color=SEMANTIC_BLUE),
            MathTex(r"\odot", font_size=36, color=SEMANTIC_BLUE),
            Text("checks correctness", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.15)

        split_right = VGroup(
            Text("Motion", font_size=28, color=DYNAMIC_RED),
            MathTex(r"\star", font_size=36, color=DYNAMIC_RED),
            Text("executes programs", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.15)

        split = VGroup(split_left, split_right).arrange(RIGHT, buff=2.5).shift(UP * 1.5)
        divider = Line(UP * 0.5, DOWN * 0.5, color=GREY_D).move_to(
            midpoint(split_left.get_right(), split_right.get_left())
        )

        self.play(FadeIn(split), Create(divider), run_time=1.5)
        self.wait(1.5)

        # ── Trajectory ──
        traj_eq = MathTex(
            r"F_0 \to F_1 \to F_2 \to \cdots \to F_{\mathrm{final}}",
            font_size=34, color=GRADE_GOLD,
        )
        each_step = narration_text("Each step uses  ★.").next_to(traj_eq, DOWN, buff=0.3)

        self.play(Write(traj_eq), FadeIn(each_step), run_time=1.5)
        self.wait(1.5)

        # Animated dot traveling along a wavy path
        path = VMobject(color=GRADE_GOLD, stroke_width=2.5)
        pts = [LEFT * 4 + DOWN * 1.5]
        for k in range(8):
            x = -4 + k * 1.0
            y = -1.5 + 0.4 * np.sin(k * 1.3)
            pts.append(np.array([x + 0.5, y, 0]))
        pts.append(RIGHT * 4 + DOWN * 1.5)
        path.set_points_smoothly(pts)

        dot = Dot(color=GRADE_GOLD, radius=0.08)
        dot.move_to(path.get_start())

        self.play(Create(path), run_time=1)
        self.play(MoveAlongPath(dot, path), run_time=3, rate_func=smooth)

        # Final check
        final_eq = MathTex(
            r"\mathrm{eval}(F_{\mathrm{final}}) = f",
            font_size=34, color=TRUTH_CYAN,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(final_eq), run_time=1)
        self.wait(1)

        gps_note = narration_text(
            "Not blind search.  You know the target.\nBut you must move through twisted geometry."
        )
        gps_note.to_edge(DOWN, buff=0.3)
        self.play(FadeOut(final_eq), FadeIn(gps_note))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# PART V — Grade as Altitude (13:00 – 16:00)
# ═══════════════════════════════════════════════════════════════

class PartV_Grade(Scene):
    """Grade, subadditivity, boundedness, hardness."""

    def construct(self):
        self.camera.background_color = BG_DARK

        header = section_title("PART V", "Grade as Altitude", GRADE_GOLD)
        self.play(FadeIn(header), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(header))

        # ── Grade definition ──
        grade_def = MathTex(r"\mathrm{grade}(A) = |A|", font_size=38, color=GRADE_GOLD)
        max_grade = MathTex(
            r"\max\_\mathrm{grade}(F) = \max\{ \mathrm{grade}(A) \mid F(A) \neq 0 \}",
            font_size=32, color=WHITE,
        ).next_to(grade_def, DOWN, buff=0.5)

        self.play(Write(grade_def), run_time=1)
        self.play(Write(max_grade), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(max_grade), grade_def.animate.to_edge(UP, buff=0.7))

        # ── Altitude gauge ──
        gauge = Line(DOWN * 2, UP * 2, color=GREY_B, stroke_width=2).shift(LEFT * 4.5)
        ticks = VGroup()
        for i in range(6):
            y = -2 + i * 0.8
            tick = Line(LEFT * 0.08, RIGHT * 0.08, color=GREY_B).move_to(gauge.get_start() + UP * i * 0.8)
            lbl = Text(str(i), font_size=16, color=GREY_B).next_to(tick, LEFT, buff=0.08)
            ticks.add(VGroup(tick, lbl))
        alt_label = Text("grade", font_size=20, color=GRADE_GOLD).next_to(gauge, UP, buff=0.15)

        marker = Triangle(color=GRADE_GOLD, fill_opacity=1).scale(0.12)
        marker.move_to(ticks[0][0].get_right() + RIGHT * 0.15)

        self.play(Create(gauge), FadeIn(ticks), Write(alt_label), FadeIn(marker), run_time=1.2)

        # Climb animation
        for target in [1, 2, 4, 3, 5]:
            target_y = ticks[target][0].get_right() + RIGHT * 0.15
            self.play(marker.animate.move_to(target_y), run_time=0.4)
        self.wait(0.8)

        # ── Subadditivity ──
        ineq = MathTex(
            r"\mathrm{grade}(A \oplus B) \le \mathrm{grade}(A) + \mathrm{grade}(B)",
            font_size=32, color=WHITE,
        ).shift(RIGHT * 0.5)
        ineq_label = Text("grade_xor_le — altitude accumulates", font_size=20, color=GRADE_GOLD)
        ineq_label.next_to(ineq, DOWN, buff=0.25)

        self.play(Write(ineq), FadeIn(ineq_label), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(ineq), FadeOut(ineq_label))

        # ── Boundedness ──
        bars = VGroup()
        heights = [0.4, 1.0, 0.7, 1.8, 1.4, 0.3, 1.6, 0.8]
        for i, h in enumerate(heights):
            bar = Rectangle(width=0.35, height=h, color=SEMANTIC_BLUE,
                            fill_opacity=0.55, stroke_width=1)
            bar.move_to(LEFT * 2.5 + RIGHT * i * 0.65 + UP * (h / 2 - 1.5))
            bars.add(bar)

        ceiling_y = 0.3
        ceiling = DashedLine(LEFT * 3 + UP * ceiling_y, RIGHT * 3 + UP * ceiling_y,
                             color=DYNAMIC_RED, stroke_width=2)
        k_label = MathTex(r"k", font_size=26, color=DYNAMIC_RED).next_to(ceiling, RIGHT, buff=0.1)

        bound_eq = MathTex(r"\max\_\mathrm{grade}(F) \le k", font_size=32, color=WHITE)
        bound_eq.shift(DOWN * 2.2)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.08),
            run_time=1.5,
        )
        self.play(Create(ceiling), FadeIn(k_label), run_time=0.8)
        self.play(Write(bound_eq), run_time=1)
        self.wait(1.5)
        self.play(FadeOut(bars), FadeOut(ceiling), FadeOut(k_label), FadeOut(bound_eq))

        # ── Hardness ──
        self.play(FadeOut(gauge), FadeOut(ticks), FadeOut(alt_label),
                  FadeOut(marker), FadeOut(grade_def))

        hard_eq = MathTex(
            r"\mathrm{Hard}(f) = "
            r"\min_{\text{all trajectories}}\;"
            r"\max_{\text{intermediate states}}\;"
            r"\mathrm{grade}",
            font_size=30, color=ACCENT_PURPLE,
        ).shift(UP * 0.5)

        narr = narration_text(
            "Among all programs computing f,\n"
            "what is the smallest maximum altitude required?"
        )
        narr.next_to(hard_eq, DOWN, buff=0.5)

        closing = Text("Hardness becomes geometric.", font_size=28, color=GRADE_GOLD)
        closing.next_to(narr, DOWN, buff=0.4)

        self.play(Write(hard_eq), run_time=2)
        self.play(FadeIn(narr))
        self.wait(1.5)
        self.play(Write(closing), run_time=1)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# PART VI — Stability and Error (16:00 – 18:00)
# ═══════════════════════════════════════════════════════════════

class PartVI_Stability(Scene):
    """l1 norm, submultiplicativity, Boolean distance, error propagation."""

    def construct(self):
        self.camera.background_color = BG_DARK

        header = section_title("PART VI", "Stability and Error", STABILITY_GREEN)
        self.play(FadeIn(header), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(header))

        # ── l1 norm ──
        l1_def = MathTex(r"\|F\|_1 = \sum_A |F(A)|", font_size=38, color=STABILITY_GREEN)
        self.play(Write(l1_def), run_time=1.2)
        self.wait(1)
        self.play(l1_def.animate.to_edge(UP, buff=0.7))

        # ── Submultiplicativity ──
        submul = theorem_box(
            r"\|F \star G\|_1 \;\le\; \|F\|_1 \cdot \|G\|_1",
            "l1_gp_submultiplicative",
            STABILITY_GREEN,
            font_size=38,
        )
        self.play(FadeIn(submul), run_time=1.5)

        narr = narration_text("Dynamics is submultiplicative.  It does not explode.")
        narr.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(narr))
        self.wait(2)
        self.play(FadeOut(submul), FadeOut(narr))

        # ── Boolean distance ──
        dist = MathTex(
            r"\mathrm{bool\_dist}(F,g) = \|F - \mathrm{embed}(g)\|_1",
            font_size=34, color=ACCENT_PURPLE,
        ).shift(UP * 0.5)
        self.play(Write(dist), run_time=1.5)
        self.wait(1)

        # ── Error propagation ──
        err = MathTex(
            r"\|F \star G - f \star g\|"
            r"\;\le\;"
            r"\|F\| \cdot \mathrm{err}(G)"
            r"\;+\;"
            r"\mathrm{err}(F) \cdot \|g\|",
            font_size=28, color=WHITE,
        ).next_to(dist, DOWN, buff=0.6)

        err_label = Text("bool_dist_wrt_gp — bilinear error propagation",
                         font_size=18, color=ACCENT_PURPLE)
        err_label.next_to(err, DOWN, buff=0.25)

        self.play(Write(err), FadeIn(err_label), run_time=2)
        self.wait(1.5)

        closing = narration_text("Semantic deviation is controlled.\nThe twisted geometry is stable.")
        closing.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(closing))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# PART VII — Blind Hiker vs GPS (18:00 – 19:00)
# ═══════════════════════════════════════════════════════════════

class PartVII_BlindVsGPS(Scene):
    """Two metaphors, one answer."""

    def construct(self):
        self.camera.background_color = BG_DARK

        header = section_title("PART VII", "Blind Hiker vs GPS")
        self.play(FadeIn(header), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(header))

        # ── Blind Hiker ──
        blind_icon = VGroup(
            Circle(radius=0.4, color=GREY_A, stroke_width=2,
                   fill_color=GREY_A, fill_opacity=0.05),
            Text("?", font_size=32, color=GREY_A),
        )
        blind_icon[1].move_to(blind_icon[0])

        blind_col = VGroup(
            Text("Blind Hiker", font_size=26, color=GREY_A),
            blind_icon,
            Text("Feels slope", font_size=18, color=GREY_B),
            Text("Climbs blind", font_size=18, color=GREY_B),
            Text("No map", font_size=18, color=GREY_B),
        ).arrange(DOWN, buff=0.2)

        # ── GPS Model ──
        gps_icon = VGroup(
            Circle(radius=0.4, color=TRUTH_CYAN, stroke_width=2,
                   fill_color=TRUTH_CYAN, fill_opacity=0.05),
            Text("✓", font_size=32, color=TRUTH_CYAN),
        )
        gps_icon[1].move_to(gps_icon[0])

        gps_col = VGroup(
            Text("GPS + Terrain", font_size=26, color=TRUTH_CYAN),
            gps_icon,
            Text("Knows the target", font_size=18, color=TRUTH_CYAN),
            Text("Knows correctness", font_size=18, color=TRUTH_CYAN),
            Text("Obeys terrain physics", font_size=18, color=MOTION_ORANGE),
        ).arrange(DOWN, buff=0.2)

        vs_text = Text("vs", font_size=22, color=GREY_D)
        panels = VGroup(blind_col, vs_text, gps_col).arrange(RIGHT, buff=1.2)
        panels.shift(UP * 0.5)

        self.play(FadeIn(panels), run_time=2)
        self.wait(2)

        # Highlight
        hl = SurroundingRectangle(gps_col, color=TRUTH_CYAN, buff=0.25,
                                  corner_radius=0.1, stroke_width=2.5)
        verdict = Text("Hyper-dual algebra is the GPS model.",
                       font_size=26, color=TRUTH_CYAN)
        verdict.shift(DOWN * 2)
        sub = narration_text("You are not searching blindly.\nYou are constrained geometrically.")
        sub.next_to(verdict, DOWN, buff=0.2)

        self.play(Create(hl), Write(verdict), run_time=1.5)
        self.play(FadeIn(sub))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# CLOSING — The Geometric Obstruction (19:00 – 20:00)
# ═══════════════════════════════════════════════════════════════

class Closing(Scene):
    """Final synthesis and closing slogan."""

    def construct(self):
        self.camera.background_color = BG_DARK

        header = section_title("CLOSING", "The Geometric Obstruction", ACCENT_PURPLE)
        self.play(FadeIn(header), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(header))

        # ── Structural summary ──
        bullets = VGroup(
            Text("A fixed representation space.", font_size=26, color=GREY_A),
            Text("Two compatible monoidal structures.", font_size=26, color=GREY_A),
            Text("One defines semantics.", font_size=26, color=SEMANTIC_BLUE),
            Text("One defines dynamics.", font_size=26, color=DYNAMIC_RED),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).shift(UP * 0.5)

        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.6)
            self.wait(0.3)

        emerge = Text(
            "Complexity emerges from the invariant\n"
            "required to reconcile them.",
            font_size=26, color=ACCENT_PURPLE, line_spacing=1.3,
        )
        emerge.next_to(bullets, DOWN, buff=0.5)
        self.play(Write(emerge), run_time=1.5)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── Lower bounds vision ──
        vision = narration_text(
            "If that obstruction scales across computational classes,\n"
            "it suggests a geometric route toward lower bounds."
        )
        self.play(FadeIn(vision))
        self.wait(3)
        self.play(FadeOut(vision))

        # ── Final poetic close ──
        self.wait(0.5)
        closing_lines = [
            ("The map tells you what is true.", TRUTH_CYAN),
            ("The mountains determine what is possible.", MOTION_ORANGE),
            ("Complexity measures how high you must climb.", ACCENT_PURPLE),
        ]

        prev = None
        mobs = []
        for text, color in closing_lines:
            t = Text(text, font_size=30, color=color)
            if prev is None:
                t.shift(UP * 0.5)
            else:
                t.next_to(prev, DOWN, buff=0.35)
            self.play(FadeIn(t, shift=UP * 0.15), run_time=0.9)
            self.wait(0.8)
            prev = t
            mobs.append(t)

        self.wait(2)
        self.play(*[FadeOut(m) for m in mobs])

        # ── Title card ──
        self.wait(0.5)
        title = Text("Hyper-Dual Algebra", font_size=56, color=WHITE)
        self.play(Write(title), run_time=2)
        self.wait(3)
        self.play(FadeOut(title))


# ═══════════════════════════════════════════════════════════════
# FULL FILM — single continuous render
# ═══════════════════════════════════════════════════════════════

class FullFilm(Scene):
    """
    Renders the entire 20-minute exposition as one video.
    Run: manim -pqh hyper_dual.py FullFilm
    """

    def construct(self):
        self.camera.background_color = BG_DARK
        self._opening()
        self._part1_space()
        self._conceptual_pause()
        self._part3_dynamics()
        self._part4_routes()
        self._part5_grade()
        self._part6_stability()
        self._part7_blind_vs_gps()
        self._closing()

    # ────────────────────────────────────────────────────
    # OPENING (0:00–2:00)
    # ────────────────────────────────────────────────────
    def _opening(self):
        # ── Map ──
        grid = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": BLUE_E, "stroke_opacity": 0.25},
            axis_config={"stroke_color": BLUE_D, "stroke_opacity": 0.4},
        ).scale(0.8)
        tag = Text("THE MAP", font_size=34, color=SEMANTIC_BLUE).to_edge(UP, buff=0.4)

        start = Dot(grid.c2p(-2, -2), color=WHITE, radius=0.1)
        grandma = Dot(grid.c2p(2, 2), color=GRADE_GOLD, radius=0.12)
        east = Arrow(grid.c2p(-2, -2), grid.c2p(2, -2),
                     color=SEMANTIC_BLUE, stroke_width=3, buff=0.08)
        north = Arrow(grid.c2p(2, -2), grid.c2p(2, 2),
                      color=SEMANTIC_BLUE, stroke_width=3, buff=0.08)

        self.play(Create(grid), Write(tag), run_time=1.5)
        self.play(FadeIn(start), FadeIn(grandma))
        self.play(GrowArrow(east), run_time=0.7)
        self.play(GrowArrow(north), run_time=0.7)

        n1 = narration_text("On this map, directions combine perfectly.")
        n1.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(n1))
        self.wait(1)
        self.play(FadeOut(n1))

        eq_u = MathTex(r"e_A \odot e_B = e_{A \oplus B}", font_size=38, color=SEMANTIC_BLUE)
        eq_u.to_edge(DOWN, buff=0.7)
        self.play(Write(eq_u), run_time=1)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── Terrain ──
        t_tag = Text("THE TERRAIN", font_size=34, color=DYNAMIC_RED).to_edge(UP, buff=0.4)
        mountains = VGroup()
        for amp, phase, y_off, t in [
            (0.7, 0.0, -0.8, 0.0), (1.0, 1.0, -0.1, 0.33),
            (0.5, 2.2, 0.5, 0.66), (0.9, 0.7, 1.1, 1.0),
        ]:
            w = FunctionGraph(
                lambda x, a=amp, p=phase, yo=y_off: a * np.sin(x * 1.5 + p) + yo,
                x_range=[-5.5, 5.5],
                color=interpolate_color(SEMANTIC_BLUE, DYNAMIC_RED, t),
                stroke_width=2.5,
            )
            mountains.add(w)

        self.play(Write(t_tag),
                  LaggedStart(*[Create(m) for m in mountains], lag_ratio=0.15),
                  run_time=2)

        eq_t = MathTex(r"e_A \star e_B = \omega(A,B)\, e_{A \oplus B}",
                       font_size=38, color=DYNAMIC_RED).to_edge(DOWN, buff=0.7)
        self.play(Write(eq_t), run_time=1)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── Declaration ──
        title = Text("Hyper-Duality", font_size=52, color=WHITE)
        self.play(Write(title), run_time=1)
        self.play(title.animate.to_edge(UP, buff=0.5), run_time=0.7)

        n2 = narration_text("We did not change the space.\nWe changed the multiplication rule.")
        n2.shift(UP * 0.5)
        self.play(FadeIn(n2))
        self.wait(1.5)
        self.play(FadeOut(n2))

        lines = VGroup(
            Text("Same representation.", font_size=28, color=GREY_A),
            Text("Two compatible multiplications.", font_size=28, color=GREY_A),
            Text("One interprets truth.", font_size=28, color=TRUTH_CYAN),
            Text("The other governs dynamics.", font_size=28, color=MOTION_ORANGE),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(ORIGIN)

        for line in lines:
            self.play(FadeIn(line, shift=RIGHT * 0.3), run_time=0.5)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ────────────────────────────────────────────────────
    # PART I — Underlying Space (2:00–5:00)
    # ────────────────────────────────────────────────────
    def _part1_space(self):
        h = section_title("PART I", "The Underlying Space")
        self.play(FadeIn(h), run_time=0.8)
        self.wait(0.6)
        self.play(FadeOut(h))

        g_eq = MathTex(r"G = (\mathbb{Z}_2)^n", font_size=40, color=WHITE)
        self.play(Write(g_eq))
        self.wait(0.8)
        self.play(g_eq.animate.to_edge(UP, buff=0.7))

        corners = VGroup(
            Text("Corners", font_size=24, color=TRUTH_CYAN),
            MathTex(r"\{\pm 1\}^n", font_size=30, color=TRUTH_CYAN),
        ).arrange(DOWN, buff=0.1)
        masks = VGroup(
            Text("Masks", font_size=24, color=MOTION_ORANGE),
            MathTex(r"A \subseteq [n]", font_size=30, color=MOTION_ORANGE),
        ).arrange(DOWN, buff=0.1)
        pics = VGroup(corners, masks).arrange(RIGHT, buff=2.0)
        self.play(FadeIn(pics), run_time=1)
        self.wait(1)

        mv = MathTex(
            r"\mathrm{MV}_n = \big\{ F : \mathrm{Mask}_n \to \mathbb{Q} \big\}",
            font_size=32, color=WHITE,
        ).shift(DOWN * 0.5)
        expand = MathTex(r"F = \sum_A F(A)\, e_A", font_size=32, color=GREY_A)
        expand.next_to(mv, DOWN, buff=0.3)

        self.play(Write(mv), run_time=1)
        self.play(Write(expand), run_time=1)

        n3 = narration_text("The group algebra Q[G].  Nothing exotic yet.")
        n3.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(n3))
        self.wait(1.5)

        n4 = narration_text("The novelty: two monoidal structures on the same space.")
        n4.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(n3), FadeIn(n4))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ────────────────────────────────────────────────────
    # PART II (Semantics) — inlined as a continuation
    # We call this from _conceptual_pause intro
    # ────────────────────────────────────────────────────
    def _part2_semantics(self):
        h = section_title("PART II", "The Untwisted Product: Semantics", SEMANTIC_BLUE)
        self.play(FadeIn(h), run_time=0.8)
        self.wait(0.6)
        self.play(FadeOut(h))

        prod = MathTex(r"e_A \odot e_B = e_{A \oplus B}", font_size=42, color=SEMANTIC_BLUE)
        self.play(Write(prod), run_time=1)
        self.play(prod.animate.to_edge(UP, buff=0.7))

        chi = MathTex(r"\chi_A(s) = (-1)^{\langle A, s \rangle}", font_size=36, color=WHITE)
        ev = MathTex(
            r"\mathrm{eval}(F)(s) = \sum_A F(A)\, \chi_A(s)",
            font_size=32, color=WHITE,
        ).next_to(chi, DOWN, buff=0.4)
        self.play(Write(chi), run_time=1)
        self.play(Write(ev), run_time=1)
        self.wait(1.5)
        self.play(FadeOut(chi), FadeOut(ev))

        # embed_correct
        thm = theorem_box(
            r"\mathrm{eval}\big(\mathrm{embed}(f)\big)(s) = f(s)",
            "embed_correct", TRUTH_CYAN, 38,
        )
        self.play(FadeIn(thm), run_time=1.2)
        self.wait(2)
        self.play(FadeOut(thm))

        # Pi_delta
        pi_eq = MathTex(
            r"\mathrm{eval}\big(\Pi(a)\big)(s) = "
            r"\begin{cases} 1 & s=a \\ 0 & s \neq a \end{cases}",
            font_size=32, color=WHITE,
        )
        lbl = Text("Pi_delta — Fourier orthogonality", font_size=20, color=ACCENT_PURPLE)
        lbl.next_to(pi_eq, DOWN, buff=0.3)
        self.play(Write(pi_eq), FadeIn(lbl), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(pi_eq), FadeOut(lbl), FadeOut(prod))

    # ────────────────────────────────────────────────────
    # CONCEPTUAL PAUSE
    # ────────────────────────────────────────────────────
    def _conceptual_pause(self):
        self._part2_semantics()

        items = VGroup(
            Text("So far:", font_size=30, color=WHITE),
            Text("The space is fixed.", font_size=24, color=GREY_A),
            Text("Untwisted multiplication governs semantics.", font_size=24, color=SEMANTIC_BLUE),
            Text("Evaluation is homomorphic under  ⊙.", font_size=24, color=TRUTH_CYAN),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(UP * 0.5)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.5)
        self.wait(1.5)

        now = Text("Now we introduce deformation.", font_size=30, color=DYNAMIC_RED)
        now.shift(DOWN * 1.5)
        self.play(Write(now))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ────────────────────────────────────────────────────
    # PART III — Dynamics (8:00–11:00)
    # ────────────────────────────────────────────────────
    def _part3_dynamics(self):
        h = section_title("PART III", "The Twisted Product: Dynamics", DYNAMIC_RED)
        self.play(FadeIn(h), run_time=0.8)
        self.wait(0.6)
        self.play(FadeOut(h))

        prod = MathTex(
            r"e_A \star e_B = \omega(A,B)\, e_{A \oplus B}",
            font_size=42, color=DYNAMIC_RED,
        )
        self.play(Write(prod), run_time=1.2)
        self.play(prod.animate.to_edge(UP, buff=0.7))

        omega_info = VGroup(
            Text("ω = 2-cocycle:", font_size=22, color=GREY_A),
            Text("  swap parity sign", font_size=20, color=GREY_B),
            Text("  metric factor from quadratic form", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT).shift(UP * 0.2)
        self.play(FadeIn(omega_info), run_time=1)
        self.wait(1.5)
        self.play(FadeOut(omega_info))

        # Cocycle condition
        cocycle = MathTex(
            r"\omega(A,B)\,\omega(A \oplus B, C)"
            r"= \omega(B,C)\,\omega(A, B \oplus C)",
            font_size=30, color=WHITE,
        )
        cocycle_l = Text("Cocycle condition ⟹ associativity", font_size=20, color=GREY_B)
        cocycle_l.next_to(cocycle, DOWN, buff=0.25)
        self.play(Write(cocycle), FadeIn(cocycle_l), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(cocycle), FadeOut(cocycle_l), FadeOut(prod))

        # Grid deformation
        flat = VGroup()
        for i in range(-3, 4):
            flat.add(Line(LEFT * 3 + UP * i * 0.5, RIGHT * 3 + UP * i * 0.5,
                          stroke_width=1, color=SEMANTIC_BLUE, stroke_opacity=0.4))
            flat.add(Line(UP * 1.5 + RIGHT * i * 0.5, DOWN * 1.5 + RIGHT * i * 0.5,
                          stroke_width=1, color=SEMANTIC_BLUE, stroke_opacity=0.4))

        def warp(p):
            x, y, z = p
            return np.array([x + 0.15 * np.sin(y * 2.5), y + 0.15 * np.cos(x * 2.5), z])

        warped = flat.copy()
        for mob in warped:
            mob.set_color(DYNAMIC_RED).set_stroke(opacity=0.5)
            mob.apply_function(warp)

        self.play(Create(flat), run_time=1)
        self.play(Transform(flat, warped), run_time=2.5)

        key = VGroup(
            Text("Same space.", font_size=26, color=GREY_A),
            Text("Altered multiplication.", font_size=26, color=DYNAMIC_RED),
            Text("Changed geometry.", font_size=26, color=WHITE),
        ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(key))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ────────────────────────────────────────────────────
    # PART IV — Routes (11:00–13:00)
    # ────────────────────────────────────────────────────
    def _part4_routes(self):
        h = section_title("PART IV", "Programs as Routes", GRADE_GOLD)
        self.play(FadeIn(h), run_time=0.8)
        self.wait(0.6)
        self.play(FadeOut(h))

        # Split diagram
        sl = VGroup(
            Text("Truth", font_size=26, color=SEMANTIC_BLUE),
            MathTex(r"\odot", font_size=34, color=SEMANTIC_BLUE),
            Text("checks correctness", font_size=18, color=GREY_B),
        ).arrange(DOWN, buff=0.12)
        sr = VGroup(
            Text("Motion", font_size=26, color=DYNAMIC_RED),
            MathTex(r"\star", font_size=34, color=DYNAMIC_RED),
            Text("executes programs", font_size=18, color=GREY_B),
        ).arrange(DOWN, buff=0.12)
        split = VGroup(sl, sr).arrange(RIGHT, buff=2.5).shift(UP * 1.5)
        self.play(FadeIn(split), run_time=1)
        self.wait(1.5)

        # Trajectory equation
        traj = MathTex(
            r"F_0 \to F_1 \to F_2 \to \cdots \to F_{\mathrm{final}}",
            font_size=34, color=GRADE_GOLD,
        )
        self.play(Write(traj), run_time=1.2)
        self.wait(1)

        # Animated dot on path
        path = VMobject(color=GRADE_GOLD, stroke_width=2.5)
        pts = []
        for k in range(9):
            pts.append(np.array([-4 + k, -1.5 + 0.4 * np.sin(k * 1.3), 0]))
        path.set_points_smoothly(pts)

        dot = Dot(color=GRADE_GOLD, radius=0.07).move_to(path.get_start())
        self.play(Create(path), run_time=0.8)
        self.play(MoveAlongPath(dot, path), run_time=2.5, rate_func=smooth)

        final = MathTex(r"\mathrm{eval}(F_{\mathrm{final}}) = f",
                        font_size=34, color=TRUTH_CYAN).to_edge(DOWN, buff=0.5)
        self.play(Write(final))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ────────────────────────────────────────────────────
    # PART V — Grade (13:00–16:00)
    # ────────────────────────────────────────────────────
    def _part5_grade(self):
        h = section_title("PART V", "Grade as Altitude", GRADE_GOLD)
        self.play(FadeIn(h), run_time=0.8)
        self.wait(0.6)
        self.play(FadeOut(h))

        g_def = MathTex(r"\mathrm{grade}(A) = |A|", font_size=38, color=GRADE_GOLD)
        self.play(Write(g_def))
        self.wait(1)
        self.play(g_def.animate.to_edge(UP, buff=0.7))

        # Gauge
        gauge = Line(DOWN * 2, UP * 2, color=GREY_B, stroke_width=2).shift(LEFT * 4.5)
        ticks = VGroup()
        for i in range(6):
            tick = Line(LEFT * 0.08, RIGHT * 0.08, color=GREY_B)
            tick.move_to(gauge.get_start() + UP * i * 0.8)
            lbl = Text(str(i), font_size=14, color=GREY_B).next_to(tick, LEFT, buff=0.06)
            ticks.add(VGroup(tick, lbl))

        marker = Triangle(color=GRADE_GOLD, fill_opacity=1).scale(0.1)
        marker.move_to(ticks[0][0].get_right() + RIGHT * 0.12)

        self.play(Create(gauge), FadeIn(ticks), FadeIn(marker), run_time=1)
        for t in [1, 3, 5, 4, 2]:
            self.play(marker.animate.move_to(ticks[t][0].get_right() + RIGHT * 0.12), run_time=0.35)

        # Subadditivity
        ineq = MathTex(
            r"\mathrm{grade}(A \oplus B) \le \mathrm{grade}(A) + \mathrm{grade}(B)",
            font_size=30, color=WHITE,
        ).shift(RIGHT * 0.5 + UP * 0.3)
        self.play(Write(ineq), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(ineq))

        # Hardness
        hard = MathTex(
            r"\mathrm{Hard}(f) = "
            r"\min_{\text{trajectories}}\;"
            r"\max_{\text{states}}\;"
            r"\mathrm{grade}",
            font_size=28, color=ACCENT_PURPLE,
        ).shift(RIGHT * 0.5)

        n5 = narration_text("Hardness = unavoidable altitude.")
        n5.to_edge(DOWN, buff=0.5)
        self.play(Write(hard), FadeIn(n5), run_time=1.5)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ────────────────────────────────────────────────────
    # PART VI — Stability (16:00–18:00)
    # ────────────────────────────────────────────────────
    def _part6_stability(self):
        h = section_title("PART VI", "Stability and Error", STABILITY_GREEN)
        self.play(FadeIn(h), run_time=0.8)
        self.wait(0.6)
        self.play(FadeOut(h))

        l1 = MathTex(r"\|F\|_1 = \sum_A |F(A)|", font_size=38, color=STABILITY_GREEN)
        self.play(Write(l1), run_time=1)
        self.play(l1.animate.to_edge(UP, buff=0.7))

        submul = theorem_box(
            r"\|F \star G\|_1 \;\le\; \|F\|_1 \cdot \|G\|_1",
            "l1_gp_submultiplicative", STABILITY_GREEN, 36,
        )
        self.play(FadeIn(submul), run_time=1.2)
        self.wait(2)
        self.play(FadeOut(submul))

        dist = MathTex(
            r"\mathrm{bool\_dist}(F,g) = \|F - \mathrm{embed}(g)\|_1",
            font_size=32, color=ACCENT_PURPLE,
        ).shift(UP * 0.3)
        self.play(Write(dist), run_time=1.2)

        err = MathTex(
            r"\|F \star G - f \star g\|"
            r"\le \|F\| \mathrm{err}(G) + \mathrm{err}(F) \|g\|",
            font_size=28, color=WHITE,
        ).next_to(dist, DOWN, buff=0.5)
        self.play(Write(err), run_time=1.5)

        n6 = narration_text("Semantic deviation is controlled.  The twisted geometry is stable.")
        n6.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(n6))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ────────────────────────────────────────────────────
    # PART VII — Blind vs GPS (18:00–19:00)
    # ────────────────────────────────────────────────────
    def _part7_blind_vs_gps(self):
        h = section_title("PART VII", "Blind Hiker vs GPS")
        self.play(FadeIn(h), run_time=0.8)
        self.wait(0.6)
        self.play(FadeOut(h))

        blind = VGroup(
            Text("Blind Hiker", font_size=26, color=GREY_A),
            Text("?", font_size=36, color=GREY_A),
            Text("Feels slope.  No map.", font_size=18, color=GREY_B),
        ).arrange(DOWN, buff=0.2)

        gps = VGroup(
            Text("GPS + Terrain", font_size=26, color=TRUTH_CYAN),
            Text("✓", font_size=36, color=TRUTH_CYAN),
            Text("Knows truth.  Obeys physics.", font_size=18, color=TRUTH_CYAN),
        ).arrange(DOWN, buff=0.2)

        vs = Text("vs", font_size=22, color=GREY_D)
        row = VGroup(blind, vs, gps).arrange(RIGHT, buff=1.2).shift(UP * 0.5)
        self.play(FadeIn(row), run_time=1.5)
        self.wait(1.5)

        hl = SurroundingRectangle(gps, color=TRUTH_CYAN, buff=0.2,
                                  corner_radius=0.1, stroke_width=2.5)
        verdict = Text("Hyper-dual algebra is the GPS model.",
                       font_size=26, color=TRUTH_CYAN).shift(DOWN * 1.5)
        self.play(Create(hl), Write(verdict), run_time=1.2)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ────────────────────────────────────────────────────
    # CLOSING (19:00–20:00)
    # ────────────────────────────────────────────────────
    def _closing(self):
        h = section_title("CLOSING", "The Geometric Obstruction", ACCENT_PURPLE)
        self.play(FadeIn(h), run_time=0.8)
        self.wait(0.6)
        self.play(FadeOut(h))

        bullets = VGroup(
            Text("A fixed representation space.", font_size=26, color=GREY_A),
            Text("Two compatible monoidal structures.", font_size=26, color=GREY_A),
            Text("One defines semantics.", font_size=26, color=SEMANTIC_BLUE),
            Text("One defines dynamics.", font_size=26, color=DYNAMIC_RED),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT).shift(UP * 0.5)

        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.5)
        self.wait(1)

        emerge = Text(
            "Complexity emerges from their geometric tension.",
            font_size=26, color=ACCENT_PURPLE,
        ).next_to(bullets, DOWN, buff=0.4)
        self.play(Write(emerge))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # Lower bounds
        vision = narration_text(
            "If that obstruction scales across computational classes,\n"
            "it suggests a geometric route toward lower bounds."
        )
        self.play(FadeIn(vision))
        self.wait(2.5)
        self.play(FadeOut(vision))

        # Poetic close
        self.wait(0.5)
        closing = [
            ("The map tells you what is true.", TRUTH_CYAN),
            ("The mountains determine what is possible.", MOTION_ORANGE),
            ("Complexity measures how high you must climb.", ACCENT_PURPLE),
        ]
        mobs = []
        prev = None
        for txt, col in closing:
            t = Text(txt, font_size=30, color=col)
            if prev is None:
                t.shift(UP * 0.5)
            else:
                t.next_to(prev, DOWN, buff=0.35)
            self.play(FadeIn(t, shift=UP * 0.15), run_time=0.8)
            self.wait(0.7)
            prev = t
            mobs.append(t)

        self.wait(2.5)
        self.play(*[FadeOut(m) for m in mobs])

        # Title card
        self.wait(0.5)
        hda = Text("Hyper-Dual Algebra", font_size=56, color=WHITE)
        self.play(Write(hda), run_time=2)
        self.wait(3)
        self.play(FadeOut(hda))

"""
manim -pqh HyperDualAlgebra.py FullFilm          # full 20-min video
manim -pql HyperDualAlgebra.py PartIII_Dynamics   # test one section
manim -pql HyperDualAlgebra.py PartV_Grade        # test grade/altitude
"""