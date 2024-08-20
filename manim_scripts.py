from manim import Text, LEFT, Create
from textwrap import wrap
from loguru import logger

debug = logger.debug
from manim_assets import (
    get_cursor,
    get_left_border,
    get_axis,
    get_translucent_rectangle,
)
from colors import POIMANDRES_BACKGROUND, POIMANDRES_FOREGROUND
from typing_anim import AddTextLetterByLetterWithCursor


def play_text_line_by_line(
    manim_object,
    text,
    coords,
    border=True,
    wrap_text=True,
    font_size=12,
    run_time=5,
    width=50,
    color=None,
    border_color=POIMANDRES_FOREGROUND,
):
    if wrap_text:
        text_list = wrap(text, width)
    else:
        text_list = [text]

    if color is None:
        color = manim_object.text_color
    x, y, z = coords[0], coords[1], coords[2]
    for i, text in enumerate(text_list):
        coords[1] -= 1 / 4
        a_lot_of_text = Text(
            text,
            font_size=font_size,
            color=color,
        ).move_to(coords, aligned_edge=LEFT)
        x_t, y_t, _ = a_lot_of_text.get_corner(LEFT)
        if i == 0:
            if border:
                left_border = get_left_border(
                    [x_t - 1 / 8, y_t + 1 / 8, 0],
                    color=border_color,
                    height=0.25 * len(text_list),
                )
                manim_object.play(Create(left_border))
                r = get_translucent_rectangle(
                    coords=left_border.get_center(),
                    height=0.25 * len(text_list),
                    width=0.11 * width,
                    color="#E4F1FA",
                    stroke=manim_object.background_color,
                )
                manim_object.add(r)
        cursor = get_cursor(a_lot_of_text)

        manim_object.play(
            AddTextLetterByLetterWithCursor(
                a_lot_of_text, cursor, leave_cursor_on=False
            ),
            run_time=run_time,
        )
    return a_lot_of_text
