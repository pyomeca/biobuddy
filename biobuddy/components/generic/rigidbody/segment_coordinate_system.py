from typing import Callable
import numpy as np

from .axis import Axis
from .marker import Marker
from ...real.biomechanical_model_real import BiomechanicalModelReal
from ...real.rigidbody.axis_real import AxisReal
from ...real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ....utils.protocols import Data
from ....utils.linear_algebra import RotoTransMatrixTimeSeries, RotoTransMatrix


class SegmentCoordinateSystem:
    def __init__(
        self,
        origin: Callable | str | Marker,
        first_axis: Axis = Axis(Axis.Name.X),
        second_axis: Axis = Axis(Axis.Name.Y),
        axis_to_keep: AxisReal.Name = Axis.Name.X,
    ):
        """
        Set the SegmentCoordinateSystemReal matrix of the segment. To compute the third axis, a first cross product of
        the first_axis with the second_axis is performed. All the axes are then normalized. Then, either the first or
        second axis (depending on [axis_to_keep]) is recomputed with a cross product to get an
        orthonormal system of axes. The system is finally moved to the origin

        Parameters
        ----------
        origin
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the
            origin of the reference frame.
            If a str is provided, the position of the corresponding marker is used
        first_axis
            The first axis defining the segment_coordinate_system
        second_axis
            The second axis defining the segment_coordinate_system
        axis_to_keep
            The Axis.Name of the axis to keep while recomputing the reference frame. It must be the same as either
            first_axis.name or second_axis.name
        """
        self.origin = origin
        self.first_axis = first_axis
        self.second_axis = second_axis
        self.axis_to_keep = axis_to_keep

    @property
    def origin(self) -> Marker:
        """
        The origin of the segment coordinate system
        """
        return self._origin

    @origin.setter
    def origin(self, value: Marker | str | Callable):
        """
        Setter for the origin of the segment coordinate system
        """
        if isinstance(value, str):
            value = Marker(name=value)
        elif isinstance(value, Marker):
            value = value
        elif callable(value):
            value = Marker(function=value)
        else:
            raise RuntimeError(f"The origin must be a Marker, a str or a Callable, not {type(value)}")
        self._origin = value

    def get_axes(
        self, data: Data, model: BiomechanicalModelReal, parent_scs: RotoTransMatrix
    ) -> tuple[AxisReal, AxisReal, AxisReal.Name]:

        # Find the two adjacent axes and reorder accordingly (assuming right-hand RT)
        if self.first_axis.name == self.second_axis.name:
            raise ValueError("The two axes cannot be the same axis")

        first_axis, second_axis = self.first_axis, self.second_axis
        if self.first_axis.name == AxisReal.Name.X:
            third_axis_name = AxisReal.Name.Y if self.second_axis.name == AxisReal.Name.Z else AxisReal.Name.Z
            if self.second_axis.name == AxisReal.Name.Z:
                first_axis, second_axis = self.second_axis, self.first_axis
        elif self.first_axis.name == AxisReal.Name.Y:
            third_axis_name = AxisReal.Name.Z if self.second_axis.name == AxisReal.Name.X else AxisReal.Name.X
            if self.second_axis.name == AxisReal.Name.X:
                first_axis, second_axis = self.second_axis, self.first_axis
        elif self.first_axis.name == AxisReal.Name.Z:
            third_axis_name = AxisReal.Name.X if self.second_axis.name == AxisReal.Name.Y else AxisReal.Name.Y
            if self.second_axis.name == AxisReal.Name.Y:
                first_axis, second_axis = self.second_axis, self.first_axis
        else:
            raise ValueError("first_axis should be an X, Y or Z axis")

        first_axis = first_axis.to_axis(data, model, parent_scs)
        second_axis = second_axis.to_axis(data, model, parent_scs)

        return first_axis, second_axis, third_axis_name

    def get_axes_vectors(
        self, first_axis: AxisReal, second_axis: AxisReal
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Compute the third axis and recompute one of the previous two
        first_axis_vector = first_axis.axis()[:3, :]
        second_axis_vector = second_axis.axis()[:3, :]
        third_axis_vector = np.cross(first_axis_vector, second_axis_vector, axis=0)
        if self.axis_to_keep == first_axis.name:
            second_axis_vector = np.cross(third_axis_vector, first_axis_vector, axis=0)
        elif self.axis_to_keep == second_axis.name:
            first_axis_vector = np.cross(second_axis_vector, third_axis_vector, axis=0)
        else:
            raise ValueError("Name of axis to keep should be one of the two axes")

        return first_axis_vector, second_axis_vector, third_axis_vector

    def get_scs_from_vectors(
        self,
        first_axis: AxisReal,
        second_axis: AxisReal,
        third_axis_name: AxisReal.Name,
        first_axis_vector: np.ndarray,
        second_axis_vector: np.ndarray,
        third_axis_vector: np.ndarray,
        origin: np.ndarray,
    ) -> RotoTransMatrix:
        # Dispatch the result into a matrix
        n_frames = max(first_axis_vector.shape[1], second_axis_vector.shape[1])
        rt = np.zeros((4, 4, n_frames))
        rt[:3, first_axis.name, :] = first_axis_vector / np.linalg.norm(first_axis_vector, axis=0)
        rt[:3, second_axis.name, :] = second_axis_vector / np.linalg.norm(second_axis_vector, axis=0)
        rt[:3, third_axis_name, :] = third_axis_vector / np.linalg.norm(third_axis_vector, axis=0)
        rt[:3, 3, :] = origin[:3, :]
        rt[3, 3, :] = 1
        all_scs = RotoTransMatrixTimeSeries(n_frames)
        all_scs.from_rt_matrix(rt)
        scs = all_scs.mean_homogenous_matrix()

        return scs

    def to_scs(
        self, data: Data, model: BiomechanicalModelReal, parent_scs: RotoTransMatrix
    ) -> SegmentCoordinateSystemReal:
        """
        This constructs a SegmentCoordinateSystemReal by evaluating the function that defines the marker to get an
        actual position

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        parent_scs
            The segment coordinate system in which the marker is defined. If None, the marker is assumed to be in the global
            coordinate system.
        """
        first_axis, second_axis, third_axis_name = self.get_axes(data, model, parent_scs)
        first_axis_vector, second_axis_vector, third_axis_vector = self.get_axes_vectors(first_axis, second_axis)
        origin = self.origin.to_marker(data, model, parent_scs).position
        scs = self.get_scs_from_vectors(
            first_axis, second_axis, third_axis_name, first_axis_vector, second_axis_vector, third_axis_vector, origin
        )

        return SegmentCoordinateSystemReal(scs=scs, is_scs_local=True)


class SegmentCoordinateSystemUtils:

    @staticmethod
    def mean_markers(marker_names: tuple[str, ...] | list[str]) -> Callable:
        """
        Compute the mean position of a set of markers

        Parameters
        ----------
        marker_names
            The names of the markers to compute the mean position from

        Returns
        -------
        A lambda function that can be called during the to_real process
        """
        return lambda m, bio: np.nanmean(np.nanmean(np.array([m[name] for name in marker_names]), axis=2), axis=0)
