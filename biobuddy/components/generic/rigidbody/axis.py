from typing import Callable
import numpy as np

from ..rigidbody.marker import Marker
from ...real.biomechanical_model_real import BiomechanicalModelReal
from ...real.rigidbody.axis_real import AxisReal
from ....utils.protocols import Data
from ....utils.linear_algebra import RotoTransMatrix


class Axis:
    class Name(AxisReal.Name):
        """
        A copy of AxisReal.Name
        """

        pass

    def __init__(
        self,
        name: AxisReal.Name,
        start: Callable | str | Marker | None = None,
        end: Callable | str | Marker | None = None,
    ):
        """
        Defines an axis to create a SegmentCoordinateSystemReal. The axis is defined by a start and an end point.
        If neither start nor end is provided, the axis is defined as the global coordinate system.

        Parameters
        ----------
        name
            The AxisName of the Axis
        start
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the starting point of the axis.
            If a str is provided, the position of the corresponding marker is used
        end
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the end point of the axis.
            If a str is provided, the position of the corresponding marker is used
        """
        if start is None and end is not None or start is not None and end is None:
            raise ValueError("Both start and end must be provided or both must be None.")

        self.name = name
        self.start = start
        self.end = end

    @property
    def start(self) -> Marker:
        """
        The start point of the axis
        """
        return self._start

    @start.setter
    def start(self, value: Marker | str | None):
        """
        Setter for the start point of the axis
        """
        if value is None:
            value = lambda m, model: np.array([0.0, 0.0, 0.0])
        if isinstance(value, Marker):
            self._start = value
        elif isinstance(value, str):
            self._start = Marker(function=value, name=value)
        elif callable(value):
            self._start = Marker(function=value, name=f"start_{self.name}")
        else:
            raise RuntimeError("Start must be a Marker, a str, or a callable")

    @property
    def end(self) -> Marker:
        """
        The end point of the axis
        """
        return self._end

    @end.setter
    def end(self, value: Marker | str | None):
        """
        Setter for the end point of the axis
        """
        if value is None:
            value = lambda m, model: np.array([0.0 if i != self.name else 1.0 for i in range(3)])
        if isinstance(value, Marker):
            self._end = value
        elif isinstance(value, str):
            self._end = Marker(function=value, name=value)
        elif callable(value):
            self._end = Marker(function=value, name=f"end_{self.name}")
        else:
            raise RuntimeError("End must be a Marker, a str, or a callable")

    def to_axis(self, data: Data, model: BiomechanicalModelReal, scs: RotoTransMatrix) -> AxisReal:
        """
        Compute the axis from actual data
        Parameters
        ----------
        data
            The actual data
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        scs
            The SegmentCoordinateSystem that this axis is part of. It is useful to compute the axis in the context of
            the segment coordinate system
        """

        start = self.start.to_marker(data, model, scs)
        end = self.end.to_marker(data, model, scs)
        return AxisReal(self.name, start, end)
