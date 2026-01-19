from manim import *

# Run:
#
#   -pqh = preview, high quality
#       manim -pqh Manim\\glogic_cl2.py Cl2BooleanAsGeometry
#
#   -pql for fast iteration
#       manim -pql Manim\\glogic_cl2.py Cl2BooleanAsGeometry


class Cl2BooleanAsGeometry(Scene):
    def construct(self):
        self.camera.background_color = "#0b0f14"

        # ----------------------------
        # Title
        # ----------------------------
        title = Text("Cl(2,0): Boolean Logic as Geometry", font_size=44, weight=BOLD)
        subtitle = Text("An executable proof idea (visual form)", font_size=28)
        subtitle.next_to(title, DOWN, buff=0.35)
        subtitle.set_opacity(0.85)
        title_group = VGroup(title, subtitle).to_edge(UP)

        self.play(FadeIn(title_group, shift=UP), run_time=1.2)

        # ----------------------------
        # Boolean hypercube in 2D: points at (±1, ±1)
        # We'll draw it as a square with labeled vertices.
        # ----------------------------
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=6,
            y_length=6,
            background_line_style={"stroke_opacity": 0.2},
        ).shift(DOWN * 0.3)

        axes_label = Text("Boolean hypercube: {−1,+1}²", font_size=26).next_to(plane, UP, buff=0.25)
        axes_label.set_opacity(0.9)

        self.play(Create(plane), FadeIn(axes_label), run_time=1.2)

        # Corner points in sign coordinates (s1, s2)
        corners = {
            (+1, +1): plane.c2p( 1,  1),
            (+1, -1): plane.c2p( 1, -1),
            (-1, +1): plane.c2p(-1,  1),
            (-1, -1): plane.c2p(-1, -1),
        }

        dots = VGroup(*[
            Dot(pos, radius=0.07) for pos in corners.values()
        ])
        dots.set_color(WHITE)

        # Label mapping from signs to booleans (choose convention):
        # True -> +1, False -> -1
        # We'll label each corner with (P1,P2) in {T,F} and (s1,s2) in {+,-}.
        def tf(s):
            return "T" if s == +1 else "F"

        corner_labels = VGroup()
        for (s1, s2), pos in corners.items():
            lbl = VGroup(
                Text(f"({tf(s1)},{tf(s2)})", font_size=22),
                Text(f"({s1:+d},{s2:+d})".replace("+", "+"), font_size=18).set_opacity(0.75)
            ).arrange(DOWN, buff=0.08)
            lbl.next_to(pos, RIGHT if s1 < 0 else LEFT, buff=0.15)
            corner_labels.add(lbl)

        square = Polygon(
            corners[(-1, -1)],
            corners[(+1, -1)],
            corners[(+1, +1)],
            corners[(-1, +1)],
        )
        square.set_stroke(WHITE, opacity=0.6, width=2)
        square.set_fill(opacity=0)

        self.play(FadeIn(dots), Create(square), FadeIn(corner_labels), run_time=1.2)
        self.wait(0.4)

        # ----------------------------
        # Show the embedding idea: projectors Π(s)
        # Π(s) = ((1 + s1 e1)/2) ((1 + s2 e2)/2)
        # Expand to: (1/4)(1 + s1 e1 + s2 e2 + s1 s2 e12)
        # ----------------------------
        formula_title = Text("Quasi-projector for an assignment", font_size=28)
        formula_title.to_edge(RIGHT).shift(UP * 2.0 + LEFT * 0.5)

        pi_def = MathTex(
            r"\Pi(s_1,s_2)=",
            r"\prod_{i=1}^{2}\frac{1+s_i e_i}{2}"
        ).scale(0.85)
        pi_def.next_to(formula_title, DOWN, buff=0.25)

        pi_expanded = MathTex(
            r"\Pi(s_1,s_2)=",
            r"\frac{1}{4}\Big(1+s_1e_1+s_2e_2+s_1s_2e_{12}\Big)"
        ).scale(0.85)
        pi_expanded.next_to(pi_def, DOWN, buff=0.25)

        rhs_box = SurroundingRectangle(pi_expanded, buff=0.15).set_stroke(WHITE, opacity=0.25)

        self.play(FadeIn(formula_title, shift=RIGHT), Write(pi_def), run_time=1.2)
        self.play(TransformMatchingTex(pi_def.copy(), pi_expanded), run_time=1.0)
        self.play(Create(rhs_box), run_time=0.6)

        self.wait(0.4)

        # ----------------------------
        # Example: AND = only satisfying assignment (T,T) = (+1,+1)
        # So F_AND = Π(+1,+1) = 1/4(1 + e1 + e2 + e12)
        # ----------------------------
        example_title = Text("Example: AND", font_size=30, weight=BOLD)
        example_title.to_edge(RIGHT).shift(UP * 0.2 + LEFT * 0.5)

        and_tex = MathTex(
            r"F_{\wedge}=\Pi(+1,+1)=",
            r"\frac{1}{4}\left(1+e_1+e_2+e_{12}\right)"
        ).scale(0.9)
        and_tex.next_to(example_title, DOWN, buff=0.25)

        # Highlight the satisfying corner (+1,+1)
        highlight = Circle(radius=0.16).move_to(corners[(+1, +1)])
        highlight.set_stroke(YELLOW, width=5)
        highlight.set_fill(opacity=0)

        self.play(FadeIn(example_title, shift=RIGHT), Write(and_tex), run_time=1.2)
        self.play(Create(highlight), run_time=0.6)
        self.wait(0.3)

        # ----------------------------
        # Evaluate AND at each corner as polynomial sampling:
        # F(s1,s2)=a0 + a1 s1 + a2 s2 + a12 s1 s2
        # For AND: all coefficients = 1/4
        # values: (1,1)->1 ; others ->0
        # ----------------------------
        eval_title = Text("Evaluation = sampling on the hypercube", font_size=26)
        eval_title.to_edge(RIGHT).shift(DOWN * 1.2 + LEFT * 0.5)

        poly_tex = MathTex(
            r"F(s_1,s_2)=a_0+a_1s_1+a_2s_2+a_{12}s_1s_2"
        ).scale(0.78)
        poly_tex.next_to(eval_title, DOWN, buff=0.2)

        coeff_tex = MathTex(
            r"\text{AND: }a_0=a_1=a_2=a_{12}=\frac14"
        ).scale(0.78)
        coeff_tex.next_to(poly_tex, DOWN, buff=0.2)

        self.play(FadeIn(eval_title, shift=RIGHT), Write(poly_tex), Write(coeff_tex), run_time=1.2)
        self.wait(0.3)

        # Create value labels at corners and animate “probe” moving around
        probe = Dot(radius=0.09, color=YELLOW).move_to(corners[(+1, +1)])
        self.add(probe)

        # Precomputed AND values
        and_values = {
            (+1, +1): 1,
            (+1, -1): 0,
            (-1, +1): 0,
            (-1, -1): 0,
        }

        value_labels = VGroup()
        for (s1, s2), pos in corners.items():
            val = and_values[(s1, s2)]
            t = MathTex(str(val)).scale(0.9)
            t.next_to(pos, UP, buff=0.15)
            t.set_color(GREEN if val == 1 else RED)
            value_labels.add(t)

        self.play(FadeIn(value_labels), run_time=0.8)

        # Walk probe around corners
        order = [(+1,+1),(+1,-1),(-1,-1),(-1,+1),(+1,+1)]
        for key in order:
            self.play(probe.animate.move_to(corners[key]), run_time=0.5)
        self.wait(0.3)

        # ----------------------------
        # Optional kicker: XOR needs e12 term
        # XOR truth table: (T,F),(F,T)
        # F_xor = Π(+1,-1)+Π(-1,+1) = 1/2(1 - e12)
        # (vectors cancel, bivector remains)
        # ----------------------------
        kicker_title = Text("Kicker: XOR forces the bivector term", font_size=28, weight=BOLD)
        kicker_title.to_edge(RIGHT).shift(DOWN * 2.6 + LEFT * 0.5)

        xor_tex = MathTex(
            r"F_{\oplus}=\Pi(+1,-1)+\Pi(-1,+1)",
        ).scale(0.8)
        xor_tex.next_to(kicker_title, DOWN, buff=0.15)

        xor_simplify = MathTex(
            r"F_{\oplus}=\frac12\left(1 - e_{12}\right)"
        ).scale(0.85)
        xor_simplify.next_to(xor_tex, DOWN, buff=0.2)

        self.play(FadeIn(kicker_title, shift=RIGHT), Write(xor_tex), run_time=1.0)
        self.play(Write(xor_simplify), run_time=1.0)

        # Highlight the two satisfying XOR corners
        h1 = Circle(radius=0.16).move_to(corners[(+1, -1)]).set_stroke(BLUE, width=5)
        h2 = Circle(radius=0.16).move_to(corners[(-1, +1)]).set_stroke(BLUE, width=5)
        self.play(Create(h1), Create(h2), run_time=0.7)

        outro = Text("Boolean logic isn’t executed — it’s sampled.", font_size=30, slant=ITALIC)
        outro.to_edge(DOWN)
        outro.set_opacity(0.9)
        self.play(FadeIn(outro), run_time=0.9)
        self.wait(1.0)
