from tqdm import tqdm
from manim import Scene, Axes, Text, Create, Rectangle, UP, LEFT
from colors import NORD_COLORS, NORD_BACKGROUND, NORD_FOREGROUND
import numpy as np
from dataclasses import dataclass
from typing_anim import (
    AddTextLetterByLetterWithCursor,
    RemoveTextLetterByLetterWithCursor,
    Blink,
)


import pickle

with open("price_walks.pkl", "rb") as f:
    price_walks = pickle.load(f)


@dataclass
class AxisConfig:
    foreground_color: str
    background_color: str
    x_vector: str
    y_vector: str
    x_label: str
    y_label: str
    width = 10
    height = 5


config = AxisConfig(
    foreground_color=NORD_FOREGROUND,
    background_color=NORD_BACKGROUND,
    x_vector=[0, 500, 1],
    y_vector=[0, 450, 1],
    x_label="Time",
    y_label="Price",
)


class Axis(Scene):
    def __init__(self, config=config):
        self.config = config
        self.x_vector = config.x_vector
        self.y_vector = config.y_vector
        self.foreground_color = config.foreground_color
        self.background_color = config.background_color
        self.x_label = config.x_label
        self.y_label = config.y_label
        self.n_ticks = config.width

    def _create_steps(self):
        self.x_step

    def _despine(self):
        self.ax.x_axis.stroke_color = self.background_color
        self.ax.x_axis.ticks.stroke_color = self.foreground_color
        self.ax.y_axis.stroke_color = self.background_color
        self.ax.y_axis.ticks.stroke_color = self.foreground_color

    def construct(self):
        FONT_SIZE = min([12 * (self.config.width / 8) + 4, 12])
        Text.set_default(font="SF Mono", font_size=FONT_SIZE)
        super().__init__()
        self.camera.background_color = self.background_color
        self.x_range = [
            np.min(self.x_vector),
            np.max(self.x_vector),
            (np.max(self.x_vector) - np.min(self.x_vector)) / self.n_ticks,
        ]
        self.y_range = [
            np.min(self.y_vector),
            np.max(self.y_vector),
            (np.max(self.y_vector) - np.min(self.y_vector)) / self.n_ticks,
        ]

        self.ax = Axes(
            x_range=self.x_range,
            y_range=self.x_range,
            x_length=self.config.width,
            y_length=self.config.height,
            tips=False,
            axis_config={
                "include_numbers": True,
                "longer_tick_multiple": 2,
                "label_constructor": Text,
                "numbers_with_elongated_ticks": range(0, 500, 100),
                "tick_size": 0.05,
                "decimal_number_config": {"num_decimal_places": 0},
                "font_size": FONT_SIZE,
            },
            color=NORD_COLORS[1],
        )
        title = Text(
            "Simulated Price Paths", font_size=18, color=NORD_FOREGROUND
        ).move_to([-3, 3.5, 0])
        x_label = Text("Number of Weeks →", font_size=FONT_SIZE)
        self._despine()
        y_label = Text("Simulated Price ↑", font_size=FONT_SIZE)
        labels = self.ax.get_axis_labels(x_label, y_label)

        # self.ax.x_axis.add_numbers(
        #     np.linspace(
        #         self.ax.x_axis.x_min, self.ax.x_axis.x_max, self.config.width + 1
        #     ),
        #     label_constructor=Text,
        #     font_size=FONT_SIZE,
        # )
        # self.ax.y_axis.add_numbers(
        #     np.linspace(
        #         self.ax.y_axis.x_min, self.ax.y_axis.x_max, self.config.height + 1
        #     ),
        #     label_constructor=Text,
        #     font_size=FONT_SIZE,
        # )
        cursor = Rectangle(
            color=NORD_FOREGROUND,
            fill_color=NORD_FOREGROUND,
            height=0.2,
            width=0.1,
        ).move_to(title[0])
        self.play(
            AddTextLetterByLetterWithCursor(title, cursor),
            Create(self.ax),
            Create(labels),
        )
        # Position the cursor
        for i, price_walk in tqdm(enumerate(price_walks[:5000])):
            graph = self.ax.plot(
                lambda x: price_walk[int(x)],
                x_range=[0, len(price_walk) - 1, 1],
                color=NORD_COLORS[i % len(NORD_COLORS)],
                stroke_width=2,
            )
            # )
            # self.play(Create(self.ax))
            # self.play(Create(labels))
            self.play(Create(graph), run_time=1 / (i + 1))
            # self.add(graph)
