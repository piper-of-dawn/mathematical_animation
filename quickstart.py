from manim import (
    Scene,
    Circle,
    Group,
    Square,
    Create,
    UP,
    DOWN,
    Transform,
    FadeOut,
    RoundedRectangle,
    PI,
    Text,
    SVGMobject,
    VGroup,
    ValueTracker,
    PURPLE,
    GREY_A,
    Rectangle,
    LEFT,
    Axes,
    Line,
    always_redraw,
    Dot,
    Write,
    rate_functions


)
import manimpango

NORD_COLORS = [
    "#8FBCBB",
    "#88C0D0",
    "#81A1C1",
    "#5E81AC",
]


def create_textbox(color, string):
    length = len(string)
    width = 0.18 * length  # calculate the width of the box
    result = VGroup()  # create a VGroup
    box = RoundedRectangle(
        corner_radius=0.05,  # create a box
        height=0.5,
        width=width,
        color=color,
        stroke_width=0,
        fill_opacity=1,
    )
    # bottom_border = 
    text = Text(string, font="JetBrains Mono", font_size=15).move_to(
        box.get_center()
    )  # create text
    result.add(box, text)  # add both objects to the VGroup
    return result


def create_glow(vmobject, rad=1, col="#8FBCBB"):
    glow_group = VGroup()
    for idx in range(60):
        new_circle = Circle(radius=rad*(1.002**(idx**2))/400, stroke_opacity=0, fill_color=col,
        fill_opacity=0.2-idx/300).move_to(vmobject)
        glow_group.add(new_circle)
    return glow_group


    

from typing_anim import AddTextLetterByLetterWithCursor, RemoveTextLetterByLetterWithCursor, Blink
class TypingAnimation(Scene):
        
        def play_sq(self):
            svg = SVGMobject("assets/cpu.svg")
            svg_list = VGroup(*[svg.copy() for _ in range(5)])
            svg_list = [svg.copy() for _ in range(5)]
            for idx, svg in enumerate(svg_list):
                # svg.move_to([-5 + idx * 2, 0, 0])
                svg.set_stroke("#ECEFF4", 1)
                svg.scale(0.25)
                svg.set_fill(opacity=0)
            self.camera.background_color = "#141824"
            text = create_textbox(color="#4C566A", string="Hello world")
            rect_group = Group(*svg_list).arrange(buff=1)
            for rect in rect_group:
                coords = rect.get_center()
                self.play(rect.animate.move_to([coords[0]+1, coords[1], coords[2]]), rect.animate.scale(2.0),  rate_func=rate_functions.ease_in_out_elastic)  
            self.play(Create(text))
        def construct(self):
            text = Text("Typing", color=PURPLE).scale(1.5).to_edge(LEFT)
            cursor = Rectangle(
                color = GREY_A,
                fill_color = GREY_A,
                fill_opacity = 1.0,
                height = 1.1,
                width = 0.5,
            ).move_to(text[0]) # Position the cursor
            self.play(Blink(cursor, how_many_times=2))
            self.play(AddTextLetterByLetterWithCursor(text, cursor, leave_cursor_on=False)) # Turning off the cursor is important
            self.play(Blink(cursor, how_many_times=3))
            self.play(RemoveTextLetterByLetterWithCursor(text, cursor))
            self.play(Blink(cursor, how_many_times=2, ends_with_off=True))
            self.play_sq()
