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
        if not isinstance(name, AxisReal.Name):
            raise ValueError("The name must be an AxisReal.Name")
        if not isinstance(start, MarkerReal):
            raise ValueError("The start must be a MarkerReal")
        if not isinstance(end, MarkerReal):
            raise ValueError("The end must be a MarkerReal")

        self.name = name
        self.start_point = start
        self.end_point = end

    def axis(self) -> np.ndarray:
        """
        Returns the axis vector
        """
        start = self.start_point.position
        end = self.end_point.position
        return end - start
