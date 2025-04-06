import numpy as np

from .marker_real import MarkerReal


class AxisReal:
    class Name:
        X = 0
        Y = 1
        Z = 2

    def __init__(self, name: Name, start: MarkerReal, end: MarkerReal):
        """
        Parameters
        ----------
        name:
            The AxisName of the Axis
        start:
            The initial Marker
        """
        self.name = name
        self.start_point = start
        self.end_point = end

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: Name):
        self._name = value

    @property
    def start_point(self) -> MarkerReal:
        return self._start_point

    @start_point.setter
    def start_point(self, value: MarkerReal):
        self._start_point = value

    @property
    def end_point(self) -> MarkerReal:
        return self._end_point

    @end_point.setter
    def end_point(self, value: MarkerReal):
        self._end_point = value

    def axis(self) -> np.ndarray:
        """
        Returns the axis vector
        """
        start = self.start_point.position
        end = self.end_point.position
        return end - start
