from manim import Scene
from tqdm import tqdm
from loguru import logger

debug = logger.debug
from manim import (
    Scene,
    Axes,
    Text,
    Create,
    Rectangle,
    UP,
    LEFT,
    ITALIC,
    rate_functions,
    Line,
    RIGHT,
    FadeOut,
    MoveToTarget,
    LineJointType,
    ReplacementTransform,
    MoveAlongPath,
    DOWN,
    Tex,
)

from colors import NORD_COLORS, NORD_BACKGROUND, NORD_FOREGROUND
import numpy as np
import pandas as pd

from manim_assets import (
    get_cursor,
    get_left_border,
    get_axis,
    get_translucent_rectangle,
    get_text_box_with_icon,
)

from math import sin, pi
from colors import POIMANDRES_BACKGROUND, POIMANDRES_FOREGROUND
from manim_scripts import play_text_line_by_line


def get_stocks_data():
    df = pd.read_csv("NVDA.csv")
    x = df["UNIX"].to_numpy()
    y = df["Adj Close"].to_numpy()
    return x, y


from datetime import datetime, timedelta


def date_quantile(min_date: str, max_date: str, quantile: float) -> str:
    min_dt = datetime.strptime(min_date, "%Y-%m-%d")
    max_dt = datetime.strptime(max_date, "%Y-%m-%d")
    delta = max_dt - min_dt
    quantile_timedelta = timedelta(days=delta.days * quantile)
    result_date = min_dt + quantile_timedelta
    return result_date.strftime("%Y-%m-%d")


class SimpleScene(Scene):
    def __init__(self):
        self.text_color = POIMANDRES_FOREGROUND
        self.foreground_color = POIMANDRES_FOREGROUND
        self.title_color = "#89DCFE"
        self.background_color = POIMANDRES_BACKGROUND
        self.training_explanation = "Training data is used to train the parameters of the model. The loss function is minimized using the observations in the training set."
        self.validation_explanation = "Validation data is used to tune the hyperparameters of the model. The performance of the trained model is iteratively improved by measuring it on the validation set."
        self.testing_explanation = "Testing data is used to evaluate the performance of the model. The model is evaluated on the testing set to determine its generalization performance."

    def construct(self):
        super().__init__()
        FONT_SIZE = 28
        Text.set_default(font="SF Mono", font_size=FONT_SIZE)
        self.camera.background_color = self.background_color
        title = "NVIDIA CLOSING PRICE"
        title_text = play_text_line_by_line(
            self,
            title,
            [-4, 3.5, 0],
            border=False,
            wrap_text=False,
            font_size=16,
            run_time=1,
            color=self.title_color,
        )

        x, y = get_stocks_data()

        rescaled_x = [(n - x.min()) / (x.max() - x.min()) for n in x]
        ax = get_axis(
            x_range=[0, 1, 0.1],
            y_range=[y.min(), y.max(), 10],
            width=8,
            background_color=self.background_color,
            foreground_color=self.foreground_color,
            y_round_off=0,
        )
        x_label = Text("UNIX Time (Scaled between 0 and 1) →", font_size=12)
        y_label = Text("Adjusted Close Price ↑", font_size=12)
        labels = ax.get_axis_labels(x_label, y_label)
        line = ax.plot_line_graph(
            rescaled_x,
            y,
            add_vertex_dots=False,
            line_color="#5CE4C6",
        )

        all_points = line["line_graph"].get_points()
        training_threshold = int(0.70 * len(all_points))

        validation_threshold = int(0.80 * len(all_points))
        testing_threshold = int(1 * len(all_points))

        line = ax.plot_line_graph(
            rescaled_x,
            y,
            add_vertex_dots=False,
            line_color="#5CE4C6",
        )
        line1 = ax.plot_line_graph(
            [ax.point_to_coords(point)[0] for point in all_points[:training_threshold]],
            [ax.point_to_coords(point)[1] for point in all_points[:training_threshold]],
            add_vertex_dots=False,
            line_color="#5CE4C6",
            joint_type=LineJointType.ROUND,
        )
        line2 = ax.plot_line_graph(
            [
                point[0]
                for point in ax.point_to_coords(
                    all_points[training_threshold:validation_threshold]
                )
            ],
            [
                point[1]
                for point in ax.point_to_coords(
                    all_points[training_threshold:validation_threshold]
                )
            ],
            add_vertex_dots=False,
            line_color="#5CE4C6",
            joint_type=LineJointType.ROUND,
        )
        line3 = ax.plot_line_graph(
            [
                point[0]
                for point in ax.point_to_coords(
                    all_points[validation_threshold:testing_threshold]
                )
            ],
            [
                point[1]
                for point in ax.point_to_coords(
                    all_points[validation_threshold:testing_threshold]
                )
            ],
            add_vertex_dots=False,
            line_color="#5CE4C6",
            joint_type=LineJointType.ROUND,
        )
        # line2 = all_points[:100]
        ax.axes.remove(ax.x_axis)
        self.play(Create(ax), Create(labels), run_time=3)
        time_scaling = 1.0
        self.wait(2.0)
        self.play(
            Create(line1, rate_func=rate_functions.linear),
            run_time=7 / time_scaling,
        )
        self.play(
            Create(line2, rate_func=rate_functions.linear), run_time=1 / time_scaling
        )
        self.play(
            Create(line3, rate_func=rate_functions.linear), run_time=2 / time_scaling
        )

        line1.generate_target()

        line1.target.shift(2 * LEFT)

        line3.generate_target()
        line3.target.shift(0.9 * RIGHT)
        self.play(FadeOut(title_text))

        train_test_split = play_text_line_by_line(
            self,
            "PERFORMING THE TRAIN-VALIDATION-TEST SPLIT",
            title_text.get_corner(LEFT),
            wrap_text=False,
            border=False,
            font_size=16,
            run_time=3,
            color=self.title_color,
        )
        self.play(FadeOut(ax), FadeOut(labels))
        self.play(
            MoveToTarget(line1, rate_func=rate_functions.ease_in_out_expo),
            MoveToTarget(line3, rate_func=rate_functions.ease_in_out_expo),
            run_time=5,
        )
        training_text = play_text_line_by_line(
            self,
            "Training",
            line1.get_center() + (UP * 0.5) + LEFT,
            wrap_text=False,
            border=False,
            font_size=13,
            run_time=1,
        )
        validation_text = play_text_line_by_line(
            self,
            "Validation",
            line2.get_center() + (UP * 0.5) + LEFT,
            wrap_text=False,
            border=False,
            font_size=13,
            run_time=1,
        )
        testing_text = play_text_line_by_line(
            self,
            "Testing",
            line3.get_center() + (UP * 0.5) + (LEFT * 0.5),
            wrap_text=False,
            border=False,
            font_size=13,
            run_time=1,
        )
        training_explanation = play_text_line_by_line(
            self,
            self.training_explanation,
            line1.get_center() + (DOWN * 0.5) + LEFT,
            width=37,
            run_time=2,
        )
        validation_explanation = play_text_line_by_line(
            self,
            self.validation_explanation,
            line2.get_center() + (DOWN * 0.5) + LEFT,
            width=30,
            run_time=2,
        )
        testing_explanation = play_text_line_by_line(
            self,
            self.testing_explanation,
            line3.get_center() + (RIGHT * 0.5),
            width=22,
            run_time=2,
        )
        self.wait(10)
