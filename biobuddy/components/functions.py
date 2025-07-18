from typing import TypeAlias
import numpy as np


class SimmSpline:
    def __init__(self, x_points: np.ndarray, y_points: np.ndarray):
        self.x_points = x_points
        self.y_points = y_points

Functions: TypeAlias = SimmSpline