from manim import *
from boolean_hypercube import BooleanHypercube2D

class Demo16Ops(Scene):
    def construct(self):
        self.camera.background_color = "#0b0f14"

        cube = BooleanHypercube2D(size=6.2, show_plane=True).shift(LEFT*2.3)
        title = Text("n=2 Boolean Hypercube — 16 Operations", font_size=34, weight=BOLD).to_edge(UP)

        self.play(FadeIn(title, shift=UP), FadeIn(cube), run_time=1.0)

        # Example masks (in order: TT, TF, FT, FF)
        # AND: only TT => 0001 (bit0=1) => mask=1
        # OR: all except FF => 0111 => mask=7
        # XOR: TF and FT => 0110 => mask=6
        # XNOR: TT and FF => 1001 => mask=9
        ops = [
            ("FALSE", 0),
            ("AND", 1),
            ("P1", 3),        # TT, TF true
            ("P2", 5),        # TT, FT true
            ("XOR", 6),
            ("OR", 7),
            ("XNOR", 9),
            ("NAND", 14),
            ("TRUE", 15),
        ]

        label = Text("", font_size=32).next_to(cube, RIGHT, buff=0.9).shift(UP*1.7)
        self.add(label)

        for name, mask in ops:
            new_label = Text(f"{name}   (mask={mask:04b})", font_size=32)
            new_label.move_to(label.get_center())
            self.play(Transform(label, new_label), run_time=0.35)
            self.play(cube.animate_to_operation(mask, run_time=0.7))
            self.wait(0.3)

        self.wait(0.8)
