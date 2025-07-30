from copy import deepcopy

# from typing import Self

import numpy as np

from .axis_real import AxisReal
from .marker_real import MarkerReal
from ....utils.aliases import Point, Points
from ....utils.linear_algebra import RotoTransMatrix, RotoTransMatrixTimeSeries, get_closest_rt_matrix


class SegmentCoordinateSystemReal:
    def __init__(
        self,
        scs: RotoTransMatrix = RotoTransMatrix(),
        is_scs_local: bool = False,
    ):
        """
        Parameters
        ----------
        scs
            The scs of the SegmentCoordinateSystemReal
        is_scs_local
            If the scs is already in local reference frame
        """
        self.scs = scs
        self.is_in_global = not is_scs_local

    @property
    def scs(self) -> RotoTransMatrix:
        return self._scs

    @scs.setter
    def scs(self, value: RotoTransMatrix):
        self._scs = value

    @property
    def is_in_global(self) -> bool:
        return self._is_in_global

    @is_in_global.setter
    def is_in_global(self, value: bool):
        self._is_in_global = value

    @property
    def is_in_local(self) -> bool:
        return not self._is_in_global

    @is_in_local.setter
    def is_in_local(self, value: bool):
        self._is_in_global = not value

    @staticmethod
    def from_markers(
        origin: MarkerReal,
        first_axis: AxisReal,
        second_axis: AxisReal,
        axis_to_keep: AxisReal.Name,
    ) -> "SegmentCoordinateSystemReal":
        """
        Parameters
        ----------
        origin
            The marker at the origin of the SegmentCoordinateSystemReal
        first_axis
            The first axis defining the segment_coordinate_system
        second_axis
            The second axis defining the segment_coordinate_system
        axis_to_keep
            The Axis.Name of the axis to keep while recomputing the reference frame. It must be the same as either
            first_axis.name or second_axis.name
        """

        # Find the two adjacent axes and reorder accordingly (assuming right-hand RT)
        if first_axis.name == second_axis.name:
            raise ValueError("The two axes cannot be the same axis")

        if first_axis.name == AxisReal.Name.X:
            third_axis_name = AxisReal.Name.Y if second_axis.name == AxisReal.Name.Z else AxisReal.Name.Z
            if second_axis.name == AxisReal.Name.Z:
                first_axis, second_axis = second_axis, first_axis
        elif first_axis.name == AxisReal.Name.Y:
            third_axis_name = AxisReal.Name.Z if second_axis.name == AxisReal.Name.X else AxisReal.Name.X
            if second_axis.name == AxisReal.Name.X:
                first_axis, second_axis = second_axis, first_axis
        elif first_axis.name == AxisReal.Name.Z:
            third_axis_name = AxisReal.Name.X if second_axis.name == AxisReal.Name.Y else AxisReal.Name.Y
            if second_axis.name == AxisReal.Name.Y:
                first_axis, second_axis = second_axis, first_axis
        else:
            raise ValueError("first_axis should be an X, Y or Z axis")

        # Compute the third axis and recompute one of the previous two
        first_axis_vector = first_axis.axis()[:3, :]
        second_axis_vector = second_axis.axis()[:3, :]
        third_axis_vector = np.cross(first_axis_vector, second_axis_vector, axis=0)
        if axis_to_keep == first_axis.name:
            second_axis_vector = np.cross(third_axis_vector, first_axis_vector, axis=0)
        elif axis_to_keep == second_axis.name:
            first_axis_vector = np.cross(second_axis_vector, third_axis_vector, axis=0)
        else:
            raise ValueError("Name of axis to keep should be one of the two axes")

        # Dispatch the result into a matrix
        n_frames = max(first_axis_vector.shape[1], second_axis_vector.shape[1])
        rt = np.zeros((4, 4, n_frames))
        rt[:3, first_axis.name, :] = first_axis_vector / np.linalg.norm(first_axis_vector, axis=0)
        rt[:3, second_axis.name, :] = second_axis_vector / np.linalg.norm(second_axis_vector, axis=0)
        rt[:3, third_axis_name, :] = third_axis_vector / np.linalg.norm(third_axis_vector, axis=0)
        rt[:3, 3, :] = origin.position[:3, :]
        rt[3, 3, :] = 1
        all_scs = RotoTransMatrixTimeSeries(n_frames)
        all_scs.from_rt_matrix(rt)
        scs = all_scs.mean_homogenous_matrix()

        return SegmentCoordinateSystemReal(scs=scs, is_scs_local=False)

    @staticmethod
    def from_rt_matrix(
        rt_matrix: np.ndarray,
        is_scs_local: bool = False,
    ) -> "SegmentCoordinateSystemReal":
        """
        Construct a SegmentCoordinateSystemReal from angles and translations

        Parameters
        ----------
        rt_matrix: np.ndarray
            The RT matrix
        is_scs_local
            If the scs is already in local reference frame
        """
        scs = RotoTransMatrix()
        scs.from_rt_matrix(rt_matrix)
        return SegmentCoordinateSystemReal(scs=scs, is_scs_local=is_scs_local)

    @staticmethod
    def from_euler_and_translation(
        angles: Points,
        angle_sequence: str,
        translation: Point,
        is_scs_local: bool = False,
    ) -> "SegmentCoordinateSystemReal":
        """
        Construct a SegmentCoordinateSystemReal from angles and translations

        Parameters
        ----------
        angles
            The actual angles
        angle_sequence
            The angle sequence of the angles
        translations
            The XYZ translations
        is_scs_local
            If the scs is already in local reference frame
        """
        scs = RotoTransMatrix()
        scs.from_euler_angles_and_translation(angles=angles, angle_sequence=angle_sequence, translation=translation)
        return SegmentCoordinateSystemReal(scs=scs, is_scs_local=is_scs_local)

    @property
    def inverse(self) -> "Self":
        out = deepcopy(self)
        out.scs = out.scs.inverse
        return out

    def to_biomod(self):

        out_string = ""
        closest_rt = get_closest_rt_matrix(self.scs.rt_matrix)
        out_string += f"\tRTinMatrix	1\n"
        out_string += f"\tRT\n"
        out_string += (
            f"\t\t{closest_rt[0, 0]:0.6f}\t{closest_rt[0, 1]:0.6f}\t{closest_rt[0, 2]:0.6f}\t{closest_rt[0, 3]:0.6f}\n"
        )
        out_string += (
            f"\t\t{closest_rt[1, 0]:0.6f}\t{closest_rt[1, 1]:0.6f}\t{closest_rt[1, 2]:0.6f}\t{closest_rt[1, 3]:0.6f}\n"
        )
        out_string += (
            f"\t\t{closest_rt[2, 0]:0.6f}\t{closest_rt[2, 1]:0.6f}\t{closest_rt[2, 2]:0.6f}\t{closest_rt[2, 3]:0.6f}\n"
        )
        out_string += (
            f"\t\t{closest_rt[3, 0]:0.6f}\t{closest_rt[3, 1]:0.6f}\t{closest_rt[3, 2]:0.6f}\t{closest_rt[3, 3]:0.6f}\n"
        )

        return out_string
