from manim import Line, Rectangle, Axes, Text, UP, LEFT
from colors import NORD_FOREGROUND
import numpy as np


def get_cursor(text, height=0.2, width=0.1):
    return Rectangle(
        color=NORD_FOREGROUND,
        fill_color=NORD_FOREGROUND,
        height=height,
        width=width,
    ).move_to(text[0])


def get_left_border(coords, height=1, width=1, color=NORD_FOREGROUND):
    return Line(
        color=color,
        start=coords,
        end=np.array([coords[0], coords[1] + height, coords[2]]),
        stroke_width=width,
    ).move_to(coords, aligned_edge=UP)


def get_translucent_rectangle(
    coords=[0, 0, 0],
    height=1,
    width=1,
    color=NORD_FOREGROUND,
    stroke=NORD_FOREGROUND,
    opacity=0.05,
):
    return Rectangle(
        color=color,
        fill_color=color,
        fill_opacity=opacity,
        stroke_width=0,
        height=height,
        width=width,
    ).move_to(coords, aligned_edge=LEFT)


def get_axis(
    x_range,
    y_range,
    background_color=None,
    foreground_color=None,
    width=10,
    height=5,
    FONT_SIZE=14,
    round_off=2,
    x_round_off=2,
    y_round_off=2,
):
    ax = Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=width,
        y_length=height,
        tips=False,
        axis_config={
            "include_numbers": True,
            "longer_tick_multiple": 2,
            "label_constructor": Text,
            "numbers_with_elongated_ticks": range(0, 500, 100),
            "tick_size": 0.05,
            "decimal_number_config": {"num_decimal_places": round_off},
            "font_size": FONT_SIZE,
        },
        x_axis_config={
            "decimal_number_config": {"num_decimal_places": x_round_off},
        },
        y_axis_config={
            "decimal_number_config": {"num_decimal_places": y_round_off},
        },
        color="#E4F1FA",
    )
    if (background_color is not None) and (foreground_color is not None):
        ax.x_axis.stroke_color = background_color
        ax.x_axis.ticks.stroke_color = foreground_color
        ax.y_axis.stroke_color = background_color
        ax.y_axis.ticks.stroke_color = foreground_color
    return ax


from manim import (
    RoundedRectangle,
    SVGMobject,
    Text,
    VGroup,
    LEFT,
    RIGHT,
)


def get_text_box_with_icon(
    text,
    icon_path,
    color,
    rectangle_background_color,
    width,
    height,
    svg_fill_opacity=1,
) -> VGroup:
    r = RoundedRectangle(width=width, height=height, corner_radius=0.1, stroke_width=0)
    r.set_fill(rectangle_background_color, 1)

    adjustment = height / 7
    coords = r.get_corner(LEFT)
    coords[0] += adjustment
    svg = SVGMobject(icon_path)
    svg.set_stroke(color, 1)
    svg.scale_to_fit_height(height - height / 2)
    svg.set_fill(opacity=svg_fill_opacity)
    svg.move_to(coords, aligned_edge=LEFT)
    if isinstance(text, str):
        text = (
            Text(text, font="JetBrains Mono", font_size=15, color=color)
            .move_to(r.get_center())
            .move_to(svg.get_corner(RIGHT), aligned_edge=LEFT)
            .shift(RIGHT * adjustment)
        )
    else:
        text = (
            text.move_to(r.get_center())
            .move_to(svg.get_corner(RIGHT), aligned_edge=LEFT)
            .shift(RIGHT * adjustment)
        )
    return VGroup(r, text, svg)
