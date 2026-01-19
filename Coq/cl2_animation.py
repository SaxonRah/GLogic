from manim import *
import numpy as np

# Configuration
config.background_color = "#0e1117"
BOOLEAN_COLOR = "#4CAF50"
PROJECTOR_COLOR = "#FF9800"
FUNCTION_COLOR = "#2196F3"
HIGHLIGHT_COLOR = "#FFEB3B"


"""
# Logical operators
r"\land"      # ∧ (AND)
r"\lor"       # ∨ (OR)
r"\neg"       # ¬ (NOT)
r"\oplus"     # ⊕ (XOR)
r"\implies"   # ⇒ (IMPLIES)
r"\iff"       # ⇔ (IFF)

# Subscripts/superscripts
r"x_1"        # x₁
r"x^2"        # x²
r"x_{12}"     # x₁₂ (multi-character subscript)

# Combining them
r"x_1 \land x_2"           # x₁ ∧ x₂
r"\neg(x_1 \lor x_2)"      # ¬(x₁ ∨ x₂)
"""

class Cl2Animation(Scene):
    """Main animation following the step-by-step outline"""

    def construct(self):
        # Step 1: Boolean inputs
        self.step1_boolean_inputs()
        self.wait(2)
        self.clear()

        # Step 2: The question
        self.step2_the_question()
        self.wait(2)
        self.clear()

        # Step 3: Introduce the object
        self.step3_introduce_object()
        self.wait(2)
        self.clear()

        # Step 4: Evaluation rule
        self.step4_evaluation_rule()
        self.wait(2)
        self.clear()

        # Step 5: Projectors (the miracle)
        self.step5_projectors()
        self.wait(2)
        self.clear()

        # Step 6: Build any function
        self.step6_build_function()
        self.wait(2)
        self.clear()

        # Step 7: The reveal
        self.step7_reveal()
        self.wait(3)

    def step1_boolean_inputs(self):
        """Step 1: Show the four Boolean input cases"""

        title = Text("Two Boolean Variables", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Show encoding
        encoding = Text("Encode as ±1:", font_size=36)
        encoding.next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(encoding))
        self.wait()

        # Create the four corners as a square
        square_size = 2.5
        corners = VGroup()
        positions = [
            (square_size, square_size),  # (+1, +1)
            (square_size, -square_size),  # (+1, -1)
            (-square_size, square_size),  # (-1, +1)
            (-square_size, -square_size),  # (-1, -1)
        ]
        labels = [
            "(+1, +1)",
            "(+1, -1)",
            "(-1, +1)",
            "(-1, -1)"
        ]

        dots = []
        texts = []
        for pos, label in zip(positions, labels):
            dot = Dot(point=np.array([pos[0], pos[1], 0]),
                      radius=0.15, color=BOOLEAN_COLOR)
            text = Text(label, font_size=28, color=BOOLEAN_COLOR)
            text.next_to(dot, RIGHT if pos[0] > 0 else LEFT, buff=0.3)

            dots.append(dot)
            texts.append(text)
            corners.add(dot, text)

        # Draw square connecting them
        square = Polygon(
            positions[0] + (0,),
            positions[1] + (0,),
            positions[3] + (0,),
            positions[2] + (0,),
            stroke_color=BOOLEAN_COLOR,
            stroke_width=2,
            fill_opacity=0
        )

        self.play(Create(square))
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.2),
            LaggedStart(*[FadeIn(t) for t in texts], lag_ratio=0.2),
        )
        self.wait()

        # Emphasize "four cases"
        caption = Text("The Four Cases", font_size=32, color=HIGHLIGHT_COLOR)
        caption.to_edge(DOWN)
        self.play(FadeIn(caption))
        self.wait()

        # Highlight each corner
        for dot in dots:
            self.play(
                dot.animate.scale(1.5).set_color(HIGHLIGHT_COLOR),
                run_time=0.3
            )
            self.play(
                dot.animate.scale(1 / 1.5).set_color(BOOLEAN_COLOR),
                run_time=0.3
            )

    def step2_the_question(self):
        """Step 2: Pose the question"""

        question = Text(
            "Can we build a single object\n"
            "that knows what a Boolean function does\n"
            "on all four cases?",
            font_size=40,
            line_spacing=1.2
        ).set_color(HIGHLIGHT_COLOR)

        self.play(Write(question, run_time=3))
        self.wait()

        subtext = Text(
            "Not doing math yet — solving an information problem",
            font_size=28,
            color=GRAY
        )
        subtext.next_to(question, DOWN, buff=1)
        self.play(FadeIn(subtext))

    def step3_introduce_object(self):
        """Step 3: Show the 4-number container"""

        title = Text("A Container with Four Numbers", font_size=44)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Show the formula compactly
        formula = MathTex(
            "F", "=", "(", "a_0", ",", "a_1", ",", "a_2", ",", "a_{12}", ")",
            font_size=56
        )
        formula.move_to(ORIGIN)

        # Color each component
        formula[3].set_color(RED)
        formula[5].set_color(GREEN)
        formula[7].set_color(BLUE)
        formula[9].set_color(ORANGE)

        self.play(Write(formula))
        self.wait()

        # Add labels
        labels = VGroup(
            Text("scalar", font_size=24, color=RED),
            Text("x₁ term", font_size=24, color=GREEN),
            Text("x₂ term", font_size=24, color=BLUE),
            Text("interaction", font_size=24, color=ORANGE)
        )

        labels[0].next_to(formula[3], DOWN, buff=0.8)
        labels[1].next_to(formula[5], DOWN, buff=0.8)
        labels[2].next_to(formula[7], DOWN, buff=0.8)
        labels[3].next_to(formula[9], DOWN, buff=0.8)

        self.play(LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.3))
        self.wait()

        # Show as bars/sliders
        bars_group = VGroup()
        bar_width = 1.5
        bar_height = 0.3
        spacing = 0.8

        sample_values = [0.5, 0.3, -0.2, 0.4]
        colors = [RED, GREEN, BLUE, ORANGE]
        names = ["a₀", "a₁", "a₂", "a₁₂"]

        for i, (val, col, name) in enumerate(zip(sample_values, colors, names)):
            # Background bar
            bg_bar = Rectangle(
                width=bar_width,
                height=bar_height,
                stroke_color=WHITE,
                stroke_width=1,
                fill_opacity=0
            )
            # Value bar
            val_bar = Rectangle(
                width=bar_width * abs(val),
                height=bar_height,
                fill_color=col,
                fill_opacity=0.7,
                stroke_width=0
            )
            val_bar.align_to(bg_bar, LEFT if val > 0 else RIGHT)

            bar_label = Text(name, font_size=24, color=col)
            bar_label.next_to(bg_bar, LEFT, buff=0.3)

            bar_container = VGroup(bar_label, bg_bar, val_bar)
            bar_container.move_to(UP * 1.5 + DOWN * i * spacing)
            bars_group.add(bar_container)

        self.play(
            formula.animate.scale(0.7).to_edge(UP, buff=1.5),
            FadeOut(labels),
            FadeOut(title)
        )

        new_title = Text("Just a 4-number container", font_size=36)
        new_title.to_edge(UP)
        self.play(FadeIn(new_title))

        self.play(LaggedStart(*[FadeIn(b) for b in bars_group], lag_ratio=0.2))
        self.wait()

    def step4_evaluation_rule(self):
        """Step 4: Show the evaluation rule"""

        title = Text("One Rule: How to Evaluate", font_size=44)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Show the evaluation formula step by step
        subtitle = Text(
            "Given inputs (s₁, s₂), compute:",
            font_size=32,
            color=GRAY
        )
        subtitle.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(subtitle))
        self.wait()

        # Build formula line by line
        formula_lines = VGroup()

        line1 = MathTex("\\text{value}", "=", "a_0", font_size=44)
        line1[2].set_color(RED)

        line2 = MathTex("+", "a_1", "\\cdot", "s_1", font_size=44)
        line2[1].set_color(GREEN)
        line2[3].set_color(BOOLEAN_COLOR)

        line3 = MathTex("+", "a_2", "\\cdot", "s_2", font_size=44)
        line3[1].set_color(BLUE)
        line3[3].set_color(BOOLEAN_COLOR)

        line4 = MathTex("+", "a_{12}", "\\cdot", "s_1", "\\cdot", "s_2", font_size=44)
        line4[1].set_color(ORANGE)
        line4[3].set_color(BOOLEAN_COLOR)
        line4[5].set_color(BOOLEAN_COLOR)

        formula_lines.add(line1, line2, line3, line4)
        formula_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        formula_lines.move_to(ORIGIN)

        # Animate each line appearing
        self.play(Write(line1))
        self.wait(0.5)
        self.play(Write(line2))
        self.wait(0.5)
        self.play(Write(line3))
        self.wait(0.5)
        self.play(Write(line4))
        self.wait()

        # Box it
        box = SurroundingRectangle(formula_lines, color=HIGHLIGHT_COLOR, buff=0.3)
        self.play(Create(box))

        # Key insight
        insight = Text(
            "Just multilinear arithmetic. No tricks.",
            font_size=32,
            color=HIGHLIGHT_COLOR
        )
        insight.to_edge(DOWN)
        self.play(FadeIn(insight))
        self.wait()

    def step5_projectors(self):
        """Step 5: The miracle - projectors"""

        title = Text("The Key Insight: Projectors", font_size=44)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        explanation = Text(
            "Choose coefficients so the formula outputs:\n"
            "• 1 on exactly one input case\n"
            "• 0 on all other cases",
            font_size=30,
            line_spacing=1.3
        )
        explanation.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(explanation))
        self.wait()

        # Show example: projector for (+1, -1)
        target = Text("Target: (+1, -1)", font_size=36, color=PROJECTOR_COLOR)
        target.to_edge(LEFT).shift(UP * 0.5)
        self.play(FadeIn(target))

        # Show the coefficients
        coeffs = MathTex(
            "\\Pi_{(+1,-1)} = (",
            "\\frac{1}{4}", ",",
            "\\frac{1}{4}", ",",
            "-\\frac{1}{4}", ",",
            "-\\frac{1}{4}",
            ")",
            font_size=36
        )
        coeffs.next_to(target, DOWN, buff=0.5)
        self.play(Write(coeffs))
        self.wait()

        # Create evaluation table
        table_data = [
            ["Input", "Output"],
            ["(+1, +1)", "0"],
            ["(+1, -1)", "1"],
            ["(-1, +1)", "0"],
            ["(-1, -1)", "0"],
        ]

        table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_width": 1}
        )
        table.scale(0.6)
        table.next_to(coeffs, DOWN, buff=0.8)

        # Color the target row
        table.get_rows()[2].set_color(HIGHLIGHT_COLOR)

        self.play(Create(table))
        self.wait()

        # Animate evaluation for each row
        for i, row in enumerate(table.get_rows()[1:], start=1):
            self.play(row.animate.set_color(PROJECTOR_COLOR), run_time=0.5)
            self.wait(0.3)
            if i == 2:  # The (+1, -1) row
                self.play(row.animate.scale(1.2), run_time=0.3)
                self.play(row.animate.scale(1 / 1.2), run_time=0.3)
            self.play(row.animate.set_color(WHITE if i != 2 else HIGHLIGHT_COLOR), run_time=0.3)

        # Emphasize
        miracle = Text(
            "It's a delta function!",
            font_size=40,
            color=HIGHLIGHT_COLOR
        )
        miracle.to_edge(DOWN)
        self.play(FadeIn(miracle))
        self.wait()

    def step6_build_function(self):
        """Step 6: Build any Boolean function by combining"""

        title = Text("Build Any Boolean Function", font_size=44)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        explanation = Text(
            "Example: AND function\n"
            "True only when both inputs are +1",
            font_size=32,
            line_spacing=1.3
        )
        explanation.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(explanation))
        self.wait()

        # Show truth table
        truth_table_data = [
            ["x₁", "x₂", "AND"],
            ["+1", "+1", "1"],
            ["+1", "-1", "0"],
            ["-1", "+1", "0"],
            ["-1", "-1", "0"],
        ]

        truth_table = Table(
            truth_table_data,
            include_outer_lines=True
        )
        truth_table.scale(0.6)
        truth_table.to_edge(LEFT).shift(UP * 0.5)
        truth_table.get_rows()[1].set_color(BOOLEAN_COLOR)

        self.play(Create(truth_table))
        self.wait()

        # Show construction
        construction = VGroup()

        text1 = Text("We need the projector for (+1, +1):", font_size=28)
        text1.next_to(truth_table, DOWN, buff=0.8)

        formula1 = MathTex(
            "F_{AND} = 1 \\cdot \\Pi_{(+1,+1)} + 0 \\cdot \\Pi_{(+1,-1)} + ...",
            font_size=32
        )
        formula1.next_to(text1, DOWN, buff=0.3)

        simplified = MathTex(
            "F_{AND} = \\Pi_{(+1,+1)}",
            font_size=40,
            color=FUNCTION_COLOR
        )
        simplified.next_to(formula1, DOWN, buff=0.5)

        construction.add(text1, formula1, simplified)

        self.play(FadeIn(text1))
        self.wait()
        self.play(Write(formula1))
        self.wait()
        self.play(Write(simplified))

        # Emphasize
        box = SurroundingRectangle(simplified, color=HIGHLIGHT_COLOR, buff=0.2)
        self.play(Create(box))
        self.wait()

        conclusion = Text(
            "Any Boolean function = weighted sum of projectors",
            font_size=32,
            color=HIGHLIGHT_COLOR
        )
        conclusion.to_edge(DOWN)
        self.play(FadeIn(conclusion))
        self.wait()

    def step7_reveal(self):
        """Step 7: The reveal - it's Clifford algebra!"""

        # Clear and build suspense
        reveal_text = Text(
            "By the way...",
            font_size=48,
            color=GRAY
        )
        self.play(FadeIn(reveal_text))
        self.wait()
        self.play(FadeOut(reveal_text))

        # The reveal
        definition = VGroup()

        line1 = Text(
            "This 4-number object,",
            font_size=40
        )
        line2 = Text(
            "with this evaluation rule,",
            font_size=40
        )
        line3 = Text(
            "is exactly",
            font_size=40
        )
        line4 = Text(
            "Clifford Algebra Cl(2,0)",
            font_size=52,
            color=HIGHLIGHT_COLOR,
            weight=BOLD
        )

        definition.add(line1, line2, line3, line4)
        definition.arrange(DOWN, buff=0.4)
        definition.move_to(ORIGIN)

        self.play(
            LaggedStart(
                FadeIn(line1),
                FadeIn(line2),
                FadeIn(line3),
                Wait(0.5),
                Write(line4),
                lag_ratio=0.7
            ),
            run_time=4
        )
        self.wait()

        # Add the one-sentence summary
        summary = Text(
            '"Cl(2,0) is the space of all multilinear functions\n'
            'on two Boolean variables,\n'
            'written in symmetric, composable form."',
            font_size=28,
            line_spacing=1.3,
            slant=ITALIC
        )
        summary.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(summary))
        self.wait()

        # Final flourish
        check = Text("✓", font_size=100, color=GREEN)
        check.next_to(definition, RIGHT, buff=1.5)
        self.play(FadeIn(check, scale=2))
        self.wait()


# Additional scenes for deeper dives
class ProjectorVisualization(Scene):
    """Detailed visualization of how projectors work"""

    def construct(self):
        title = Text("Projectors: Delta Functions on the Hypercube", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        # Show all four projectors at once
        grid = VGroup()
        projector_data = [
            ((+1, +1), [1 / 4, 1 / 4, 1 / 4, 1 / 4]),
            ((+1, -1), [1 / 4, 1 / 4, -1 / 4, -1 / 4]),
            ((-1, +1), [1 / 4, -1 / 4, 1 / 4, -1 / 4]),
            ((-1, -1), [1 / 4, -1 / 4, -1 / 4, 1 / 4]),
        ]

        for i, (target, coeffs) in enumerate(projector_data):
            cell = self.create_projector_cell(target, coeffs)
            cell.move_to(
                LEFT * 3.5 * (i % 2) + UP * 2 * (i // 2) +
                RIGHT * 3.5 + DOWN * 1.5
            )
            grid.add(cell)

        self.play(LaggedStart(*[FadeIn(c) for c in grid], lag_ratio=0.3))
        self.wait()

    def create_projector_cell(self, target, coeffs):
        """Create a visualization cell for one projector"""
        cell = VGroup()

        # Target label
        target_text = Text(f"Π{target}", font_size=28, color=PROJECTOR_COLOR)

        # Coefficients
        coeff_text = Text(
            f"({coeffs[0]:.2f}, {coeffs[1]:.2f}, {coeffs[2]:.2f}, {coeffs[3]:.2f})",
            font_size=20
        )
        coeff_text.next_to(target_text, DOWN, buff=0.2)

        cell.add(target_text, coeff_text)
        return cell


class BooleanFunctionGallery(Scene):
    """Show various Boolean functions as multivectors"""

    def construct(self):
        title = Text("Boolean Functions as Multivectors", font_size=44)
        title.to_edge(UP)
        self.play(Write(title))

        # functions = [
        #     ("AND", "x₁ ∧ x₂", [0.25, 0.25, 0.25, 0.25]),
        #     ("OR", "x₁ ∨ x₂", [0.75, 0.25, 0.25, -0.25]),
        #     ("XOR", "x₁ ⊕ x₂", [0.5, 0, 0, -0.5]),
        #     ("NAND", "¬(x₁ ∧ x₂)", [0.75, -0.25, -0.25, -0.25]),
        # ]

        # Use proper LaTeX syntax instead of Unicode
        functions = [
            ("AND", r"x_1 \land x_2", [0.25, 0.25, 0.25, 0.25]),
            ("OR", r"x_1 \lor x_2", [0.75, 0.25, 0.25, -0.25]),
            ("XOR", r"x_1 \oplus x_2", [0.5, 0, 0, -0.5]),
            ("NAND", r"\neg(x_1 \land x_2)", [0.75, -0.25, -0.25, -0.25]),
        ]

        gallery = VGroup()
        for name, logic, coeffs in functions:
            card = self.create_function_card(name, logic, coeffs)
            gallery.add(card)

        gallery.arrange_in_grid(rows=2, cols=2, buff=1)
        gallery.next_to(title, DOWN, buff=1)

        self.play(LaggedStart(*[FadeIn(c) for c in gallery], lag_ratio=0.4))
        self.wait()

    def create_function_card(self, name, logic, coeffs):
        """Create a card showing one function"""
        card = VGroup()

        name_text = Text(name, font_size=32, weight=BOLD, color=FUNCTION_COLOR)
        logic_text = MathTex(logic, font_size=28)
        logic_text.next_to(name_text, DOWN, buff=0.2)

        # Coefficients as bars
        bars = VGroup()
        colors = [RED, GREEN, BLUE, ORANGE]
        for c, col in zip(coeffs, colors):
            bar = Rectangle(
                width=abs(c) * 2,
                height=0.2,
                fill_color=col,
                fill_opacity=0.7,
                stroke_width=0
            )
            bars.add(bar)

        bars.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        bars.next_to(logic_text, DOWN, buff=0.4)

        card.add(name_text, logic_text, bars)

        # Box it
        box = SurroundingRectangle(card, color=WHITE, buff=0.3)
        card.add(box)

        return card


# Render command helper
if __name__ == "__main__":
    """
    To render:

    manim -pql coq\cl2_animation.py Cl2Animation          # Main sequence
    manim -pql coq\cl2_animation.py ProjectorVisualization # Projectors deep dive
    manim -pql coq\cl2_animation.py BooleanFunctionGallery # Function gallery

    For high quality:
    manim -pqh cl2_animation.py Cl2Animation
    """
    pass