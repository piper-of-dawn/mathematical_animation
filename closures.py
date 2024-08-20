from ast import List
from re import T
from manim import (
    Text,
    Tex,
    Scene,
    Uncreate,
    RIGHT,
    Rotate,
    UP,
    Dot,
    PI,
    FadeOut,
    FadeIn,
    Create,
    RoundedRectangle,
    SVGMobject,
    always_rotate,
    rate_functions,
    VGroup,
    RIGHT,
    LEFT,
    NumberLine,
)
from colors import (
    AURORA_GREEN,
    FROST_TEAL,
    NORD_COLORS,
    NORD_BACKGROUND,
    NORD_FOREGROUND,
    AURORA_YELLOW,
    NORD_BLUE,
    FROST_AQUA,
    FROST_AZURE,
)
from loguru import logger
from manim_assets import get_text_box_with_icon
from manim_scripts import play_text_line_by_line

debug = logger.debug


def make_function_scope_rect(width=1, height=1, fill=AURORA_YELLOW):
    return RoundedRectangle(
        color=NORD_FOREGROUND,
        fill_opacity=1,
        fill_color=fill,
        stroke_width=0,
        height=height,
        width=width,
        corner_radius=0.2,
    )


class FunctionScopeScene(Scene):
    def construct(self):
        super().__init__()
        self.background_color = NORD_BACKGROUND
        self.text_color = NORD_FOREGROUND
        self.foreground_color = FROST_TEAL
        FONT_SIZE = 18
        Text.set_default(font="DM Mono", font_size=FONT_SIZE)
        self.camera.background_color = NORD_BACKGROUND
        text = "Rust’s closures are anonymous functions you can save in a variable or pass as arguments to other functions. You can create the closure in one place and then call the closure elsewhere to evaluate it in a different context. Unlike functions, closures can capture values from the scope in which they’re defined."
        play_text_line_by_line(self, text, [-4, 3, 0], width=80)
        play_text_line_by_line(
            self, "Closure is a callable", [-4, 2, 0], width=80, border_color=FROST_TEAL
        )
        play_text_line_by_line(
            self,
            "Closure can be assigned to a variable",
            [-4, 1.5, 0],
            width=80,
            border_color=AURORA_YELLOW,
        )
        play_text_line_by_line(
            self,
            "Closure can capture values",
            [-4, 1.5, 0],
            width=80,
            border_color=AURORA_YELLOW,
        )
        self.wait(10)
        gear = (
            SVGMobject("assets/gear.svg", fill_color=AURORA_GREEN)
            .move_to([0, 0, 0])
            .scale(0.50)
        )
        c = gear.get_center()
        c[1] += 1
        t = Tex("$f$", color=AURORA_GREEN, font_size=28).move_to(c)
        g = VGroup(t, gear)

        inp = get_text_box_with_icon(
            "x",
            "assets/object.svg",
            NORD_BACKGROUND,
            AURORA_YELLOW,
            width=0.85,
            height=0.8,
        ).move_to(gear.get_center() + LEFT * 2)
        debug(gear.get_center())

        out = get_text_box_with_icon(
            "f(x)",
            "assets/object.svg",
            NORD_BACKGROUND,
            AURORA_YELLOW,
            width=1.2,
            height=0.8,
        ).move_to(gear.get_center())

        # self.play(Rotate(gear, PI), run_time=5)
        always_rotate(gear, rate=PI / 2)
        self.add(gear)
        self.add(t)
        self.play(inp.animate.scale(0.5).shift(2 * RIGHT), run_time=5)
        self.play(
            FadeIn(out, scale=0.1),
            out.animate.shift(2 * RIGHT),
            run_time=5,
        )
        self.wait(10)
        self.play(FadeOut(gear), FadeOut(inp), FadeOut(t), run_time=5)

        l0 = NumberLine(
            x_range=[0, 100, 10],
            length=10,
            color=NORD_BLUE,
            include_numbers=True,
            label_direction=UP,
            label_constructor=Text,
            font_size=FONT_SIZE,
        )
        l0_pos = l0.get_center()
        l0_pos[1] = -2
        l1 = NumberLine(
            x_range=[0, 1, 0.2],
            length=5,
            color=NORD_BLUE,
            include_numbers=True,
            label_direction=UP,
            label_constructor=Text,
            font_size=FONT_SIZE,
        ).move_to(l0_pos)
        # dot = Dot(l0.number_to_point(85.3), color=FROST_AQUA)
        # debug()

        # self.add(g, input, output, closure, l0, l1)
        # moving_dots(self, [85.3, 70, 50, 30, 10], l0, l1)
        # self.play(dot.animate.move_to(l1.number_to_point(0.85)), run_time=2)
        # self.play(Create(input), run_time=2)
        # self.play(Create(output), run_time=2)
        # self.play(FadeOut(g), run_time=10)


def moving_dots(obj, lst: List, l0, l1):
    g = VGroup()
    dots = [Dot(l0.number_to_point(n), color=FROST_AQUA) for n in lst]
    debug(dots)
    obj.add(*dots)
    obj.play(
        *[
            dot.animate.move_to(l1.number_to_point(n / 100))
            for n, dot in zip(lst, dots)
        ],
        run_time=2,
    )
