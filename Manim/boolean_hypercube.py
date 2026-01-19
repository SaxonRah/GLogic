from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from manim import *


# -----------------------------
# Helpers
# -----------------------------

def tf_from_sign(s: int) -> str:
    return "T" if s == +1 else "F"

def sign_tuple_from_bools(p1: bool, p2: bool) -> Tuple[int, int]:
    return (+1 if p1 else -1, +1 if p2 else -1)

def bools_from_signs(s1: int, s2: int) -> Tuple[bool, bool]:
    return (s1 == +1, s2 == +1)

# Standard corner order for truth-masks:
# bit 0 -> (T,T) = (+1,+1)
# bit 1 -> (T,F) = (+1,-1)
# bit 2 -> (F,T) = (-1,+1)
# bit 3 -> (F,F) = (-1,-1)
CORNER_ORDER: List[Tuple[int, int]] = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]


def mask_to_satisfying(mask: int) -> Dict[Tuple[int, int], bool]:
    out = {}
    for i, corner in enumerate(CORNER_ORDER):
        out[corner] = bool((mask >> i) & 1)
    return out


def satisfying_to_mask(sat: Dict[Tuple[int, int], bool]) -> int:
    mask = 0
    for i, corner in enumerate(CORNER_ORDER):
        if sat.get(corner, False):
            mask |= (1 << i)
    return mask


def func_to_satisfying(func: Callable[[bool, bool], bool]) -> Dict[Tuple[int, int], bool]:
    sat = {}
    for (s1, s2) in CORNER_ORDER:
        p1, p2 = bools_from_signs(s1, s2)
        sat[(s1, s2)] = bool(func(p1, p2))
    return sat


@dataclass
class Hypercube2DStyle:
    # geometry
    square_opacity: float = 0.65
    grid_opacity: float = 0.18
    dot_radius: float = 0.075

    # colors (don’t lock yourself in; easy to change per video)
    dot_color: ManimColor = WHITE
    square_color: ManimColor = WHITE
    sat_color: ManimColor = GREEN
    unsat_color: ManimColor = RED
    highlight_color: ManimColor = YELLOW

    # text
    label_font_size: int = 22
    value_font_size: int = 32
    title_font_size: int = 30


class BooleanHypercube2D(VGroup):
    """
    Reusable visualization of the Boolean hypercube for n=2.

    Features:
      - 4 corners at (±1, ±1)
      - Labels for each vertex: (T/F) and (±1)
      - Show any of the 16 Boolean ops using either:
          * truth mask (0..15)
          * dict[(s1,s2)] -> bool
          * callable (p1,p2)->bool
      - Animated transitions between operations
      - Hooks for overlays: correlations, coefficients, attention, etc.
    """

    def __init__(
        self,
        *,
        size: float = 6.0,
        show_plane: bool = True,
        show_sign_labels: bool = True,
        show_tf_labels: bool = True,
        style: Hypercube2DStyle = Hypercube2DStyle(),
    ):
        super().__init__()
        self.style = style
        self.size = float(size)

        # Coordinate system
        self.plane: Optional[NumberPlane] = None
        if show_plane:
            self.plane = NumberPlane(
                x_range=[-2, 2, 1],
                y_range=[-2, 2, 1],
                x_length=self.size,
                y_length=self.size,
                background_line_style={"stroke_opacity": self.style.grid_opacity},
            )
            self.add(self.plane)

        # Corner positions in scene coords
        def c2p(x: float, y: float):
            return self.plane.c2p(x, y) if self.plane is not None else np.array([x, y, 0.0])

        self.corner_pos: Dict[Tuple[int, int], np.ndarray] = {
            (+1, +1): c2p( 1,  1),
            (+1, -1): c2p( 1, -1),
            (-1, +1): c2p(-1,  1),
            (-1, -1): c2p(-1, -1),
        }

        # Square
        self.square = Polygon(
            self.corner_pos[(-1, -1)],
            self.corner_pos[(+1, -1)],
            self.corner_pos[(+1, +1)],
            self.corner_pos[(-1, +1)],
        )
        self.square.set_stroke(self.style.square_color, opacity=self.style.square_opacity, width=2)
        self.square.set_fill(opacity=0)
        self.add(self.square)

        # Dots
        self.dots = {}
        for corner, pos in self.corner_pos.items():
            d = Dot(pos, radius=self.style.dot_radius, color=self.style.dot_color)
            self.dots[corner] = d
            self.add(d)

        # Labels (TF and/or signs)
        self.corner_labels = VGroup()
        for (s1, s2), pos in self.corner_pos.items():
            parts = []
            if show_tf_labels:
                parts.append(Text(f"({tf_from_sign(s1)},{tf_from_sign(s2)})", font_size=self.style.label_font_size))
            if show_sign_labels:
                parts.append(Text(f"({s1:+d},{s2:+d})", font_size=int(self.style.label_font_size * 0.85)).set_opacity(0.75))
            if parts:
                lbl = VGroup(*parts).arrange(DOWN, buff=0.06)
                # place label toward outside of square to reduce overlap
                lbl.next_to(pos, RIGHT if s1 < 0 else LEFT, buff=0.15)
                self.corner_labels.add(lbl)

        self.add(self.corner_labels)

        # Value markers for current operation
        self.value_labels = VGroup()
        self._value_label_map: Dict[Tuple[int, int], Mobject] = {}

        # Optional highlight ring (for “current vertex” / explanation)
        self.highlight_ring = Circle(radius=0.18).set_stroke(self.style.highlight_color, width=5).set_fill(opacity=0)
        self.highlight_ring.set_opacity(0)  # hidden by default
        self.add(self.highlight_ring)

        # Overlay container (for addons)
        self.overlays = VGroup()
        self.add(self.overlays)

        # Current op state
        self.current_satisfying: Dict[Tuple[int, int], bool] = {c: False for c in CORNER_ORDER}

    # -----------------------------
    # Core: set/show operations
    # -----------------------------

    def set_operation(
        self,
        op: Union[int, Dict[Tuple[int, int], bool], Callable[[bool, bool], bool]],
        *,
        show_values: bool = True,
        color_dots: bool = True,
    ) -> "BooleanHypercube2D":
        """
        Set without animation.
        op can be:
          - int mask 0..15 in CORNER_ORDER
          - dict[(s1,s2)] -> bool
          - callable (p1,p2)->bool
        """
        sat = self._normalize_op(op)
        self.current_satisfying = sat

        if color_dots:
            for corner, d in self.dots.items():
                d.set_color(self.style.sat_color if sat[corner] else self.style.unsat_color)

        if show_values:
            self._rebuild_value_labels(sat)

        return self

    def animate_to_operation(
        self,
        op: Union[int, Dict[Tuple[int, int], bool], Callable[[bool, bool], bool]],
        *,
        show_values: bool = True,
        color_dots: bool = True,
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """
        Returns an AnimationGroup that transitions the visualization to the new operation.
        Caller does: self.play(cube.animate_to_operation(...))
        """
        sat = self._normalize_op(op)

        anims: List[Animation] = []
        if color_dots:
            for corner, d in self.dots.items():
                target_color = self.style.sat_color if sat[corner] else self.style.unsat_color
                anims.append(d.animate.set_color(target_color))

        if show_values:
            # rebuild value labels as a target state and transform
            new_group, new_map = self._build_value_labels(sat)
            # If none yet, just fade in
            if len(self.value_labels) == 0:
                anims.append(FadeIn(new_group))
            else:
                anims.append(Transform(self.value_labels, new_group))
            # commit new structures
            self.value_labels = new_group
            self._value_label_map = new_map
            self.add(self.value_labels)

        self.current_satisfying = sat
        return AnimationGroup(*anims, lag_ratio=0.0, run_time=run_time)

    # -----------------------------
    # Highlight / explanation utilities
    # -----------------------------

    def highlight_corner(self, corner: Tuple[int, int]) -> "BooleanHypercube2D":
        self.highlight_ring.set_opacity(1.0)
        self.highlight_ring.move_to(self.corner_pos[corner])
        return self

    def hide_highlight(self) -> "BooleanHypercube2D":
        self.highlight_ring.set_opacity(0.0)
        return self

    # -----------------------------
    # Addon hooks (GeoSAT / GeoCPU / GeoAI)
    # -----------------------------

    def clear_overlays(self) -> "BooleanHypercube2D":
        self.overlays.submobjects = []
        return self

    def add_correlation_edge(
        self,
        corner_a: Tuple[int, int],
        corner_b: Tuple[int, int],
        *,
        label: Optional[str] = None,
        stroke_width: float = 6,
        opacity: float = 0.7,
    ) -> VGroup:
        """
        Generic overlay edge between any two corners.
        Useful for GeoSAT/structure visuals (correlation / constraints).
        """
        a = self.corner_pos[corner_a]
        b = self.corner_pos[corner_b]
        line = Line(a, b).set_stroke(WHITE, width=stroke_width, opacity=opacity)

        g = VGroup(line)
        if label:
            t = Text(label, font_size=22).set_opacity(0.85)
            t.move_to((a + b) / 2 + UP * 0.15)
            g.add(t)

        self.overlays.add(g)
        return g

    def add_grade_bar(
        self,
        *,
        title: str = "Grades",
        components: Dict[str, float],
        width: float = 3.2,
        height: float = 1.8,
    ) -> VGroup:
        """
        Simple reusable “component bars” overlay.
        Perfect for GeoCPU truncation (show scalar / e1 / e2 / e12 magnitudes),
        or for GeoAI (show contributions).
        """
        # Frame
        frame = RoundedRectangle(corner_radius=0.18, width=width, height=height)
        frame.set_stroke(WHITE, opacity=0.25, width=2)
        frame.set_fill(opacity=0.06)

        ttl = Text(title, font_size=22, weight=BOLD).move_to(frame.get_top() + DOWN * 0.22)

        # Normalize for display (keep sign visually by centering at 0)
        keys = list(components.keys())
        vals = [float(components[k]) for k in keys]
        max_abs = max([abs(v) for v in vals] + [1e-9])

        rows = VGroup()
        y0 = frame.get_center()[1] + 0.25
        dy = (height - 0.65) / max(1, len(keys))

        for i, (k, v) in enumerate(zip(keys, vals)):
            y = y0 - i * dy
            ktxt = Text(k, font_size=18).move_to([frame.get_left()[0] + 0.45, y, 0])
            # bar from midline
            mid_x = frame.get_center()[0] + 0.35
            bar_len = 1.2 * (v / max_abs)
            bar = Line([mid_x, y, 0], [mid_x + bar_len, y, 0]).set_stroke(WHITE, width=8, opacity=0.85)
            zero = Dot([mid_x, y, 0], radius=0.03, color=WHITE).set_opacity(0.6)
            vtxt = Text(f"{v:+.3f}", font_size=16).move_to([frame.get_right()[0] - 0.55, y, 0]).set_opacity(0.85)
            rows.add(VGroup(ktxt, bar, zero, vtxt))

        g = VGroup(frame, ttl, rows)
        self.overlays.add(g)
        return g

    # -----------------------------
    # Internals
    # -----------------------------

    def _normalize_op(
        self,
        op: Union[int, Dict[Tuple[int, int], bool], Callable[[bool, bool], bool]]
    ) -> Dict[Tuple[int, int], bool]:
        if isinstance(op, int):
            if op < 0 or op > 15:
                raise ValueError("Truth mask must be in [0,15] for n=2.")
            sat = mask_to_satisfying(op)
        elif isinstance(op, dict):
            sat = {c: bool(op.get(c, False)) for c in CORNER_ORDER}
        else:
            sat = func_to_satisfying(op)

        # ensure all corners exist
        for c in CORNER_ORDER:
            sat.setdefault(c, False)
        return sat

    def _build_value_labels(self, sat: Dict[Tuple[int, int], bool]) -> Tuple[VGroup, Dict[Tuple[int, int], Mobject]]:
        group = VGroup()
        mapping: Dict[Tuple[int, int], Mobject] = {}
        for corner, pos in self.corner_pos.items():
            val = 1 if sat[corner] else 0
            t = MathTex(str(val)).scale(1.0)
            t.set_color(self.style.sat_color if val == 1 else self.style.unsat_color)
            t.next_to(pos, UP, buff=0.16)
            group.add(t)
            mapping[corner] = t
        return group, mapping

    def _rebuild_value_labels(self, sat: Dict[Tuple[int, int], bool]) -> None:
        # remove old labels if any
        if len(self.value_labels) > 0:
            self.remove(self.value_labels)
        self.value_labels, self._value_label_map = self._build_value_labels(sat)
        self.add(self.value_labels)
