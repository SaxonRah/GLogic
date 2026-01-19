from manim import *
from boolean_hypercube import BooleanHypercube2D
from glogic_backend_cl2 import embed_cl2_from_mask, eval_mask_from_mv, cl2_coeffs_from_mv

class ExecutableCl2Demo(Scene):
    def construct(self):
        self.camera.background_color = "#0b0f14"

        cube = BooleanHypercube2D(size=6.2).shift(LEFT*2.4)
        self.play(FadeIn(cube), run_time=0.8)

        # AND mask=1, XOR mask=6 in our convention
        for name, mask in [("AND", 1), ("XOR", 6)]:
            F = embed_cl2_from_mask(mask)           # compute multivector from satisfying assignments
            implied = eval_mask_from_mv(F)          # compute truth table back from evaluation
            coeffs = cl2_coeffs_from_mv(F)          # extract coefficients

            label = Text(f"{name}   (mask {mask:04b}, eval→ {implied:04b})", font_size=30).to_edge(UP)
            self.play(FadeIn(label, shift=UP*0.2), run_time=0.4)

            # Drive the visualization from what evaluation says is true
            self.play(cube.animate_to_operation(implied, run_time=0.7))

            cube.clear_overlays()
            bar = cube.add_grade_bar(title="Cl(2) coeffs", components=coeffs)
            bar.to_edge(RIGHT).shift(DOWN*0.2)

            self.wait(1.0)
            self.play(FadeOut(label), run_time=0.3)
