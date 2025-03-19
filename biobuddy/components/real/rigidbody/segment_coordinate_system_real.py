from typing import Self

import numpy as np

from .axis_real import AxisReal
from .marker_real import MarkerReal
from ....utils.aliases import Point, Points
from ....utils.linear_algebra import (
    euler_and_translation_to_matrix,
    mean_homogenous_matrix,
    to_euler,
    transpose_homogenous_matrix,
    multiply_homogeneous_matrix,
)


class SegmentCoordinateSystemReal:
    def __init__(
        self,
        scs: np.ndarray = np.identity(4),
        parent_scs: Self = None,
        is_scs_local: bool = False,
        rt_in_matrix: bool = None,
    ):
        """
        Parameters
        ----------
        scs
            The scs of the SegmentCoordinateSystemReal
        parent_scs
            The scs of the parent (is used when printing the model so SegmentCoordinateSystemReal
            is in parent's local reference frame
        is_scs_local
            If the scs is already in local reference frame
        """
        if scs.shape != (4, 4):
            raise ValueError("The scs must be a 4x4 matrix")
        self.scs = scs
        if len(self.scs.shape) == 2:
            self.scs = self.scs[:, :, np.newaxis]
        self.parent_scs = parent_scs
        self.is_in_global = not is_scs_local
        self.rt_in_matrix = rt_in_matrix

    @staticmethod
    def from_markers(
        origin: MarkerReal,
        first_axis: AxisReal,
        second_axis: AxisReal,
        axis_to_keep: AxisReal.Name,
        parent_scs: Self = None,
    ) -> Self:
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
        parent_scs
            The scs of the parent (is used when printing the model so SegmentCoordinateSystemReal
            is in parent's local reference frame
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
        n_frames = first_axis_vector.shape[1]
        rt = np.zeros((4, 4, n_frames))
        rt[:3, first_axis.name, :] = first_axis_vector / np.linalg.norm(first_axis_vector, axis=0)
        rt[:3, second_axis.name, :] = second_axis_vector / np.linalg.norm(second_axis_vector, axis=0)
        rt[:3, third_axis_name, :] = third_axis_vector / np.linalg.norm(third_axis_vector, axis=0)
        rt[:3, 3, :] = origin.position[:3, :]
        rt[3, 3, :] = 1

        return SegmentCoordinateSystemReal(scs=rt, parent_scs=parent_scs)

    @staticmethod
    def from_rt_matrix(
        rt_matrix: np.ndarray,
        parent_scs: Self = None,
    ) -> Self:
        """
        Construct a SegmentCoordinateSystemReal from angles and translations

        Parameters
        ----------
        rt_matrix: np.ndarray
            The RT matrix
        parent_scs
            The scs of the parent (is used when printing the model so SegmentCoordinateSystemReal
            is in parent's local reference frame
        """
        return SegmentCoordinateSystemReal(scs=rt_matrix, parent_scs=parent_scs, is_scs_local=True, rt_in_matrix=True)

    @staticmethod
    def from_euler_and_translation(
        angles: Points,
        angle_sequence: str,
        translations: Point,
        parent_scs: Self = None,
    ) -> Self:
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
        parent_scs
            The scs of the parent (is used when printing the model so SegmentCoordinateSystemReal
            is in parent's local reference frame
        """

        rt = euler_and_translation_to_matrix(angles=angles, angle_sequence=angle_sequence, translations=translations)
        return SegmentCoordinateSystemReal(scs=rt, parent_scs=parent_scs, is_scs_local=True, rt_in_matrix=False)

    def copy(self):
        return SegmentCoordinateSystemReal(scs=np.array(self.scs), parent_scs=self.parent_scs)

    @property
    def mean_scs(self) -> np.ndarray:
        """
        Computes the closest homogenous matrix that approximates all the scs

        Returns
        -------
        The mean homogenous matrix
        """
        return mean_homogenous_matrix(self.scs)

    @property
    def to_biomod(self):
        rt = self.scs
        if self.is_in_global:
            rt = self.parent_scs.transpose @ self.scs if self.parent_scs else np.identity(4)[:, :, np.newaxis]

        out_string = ""
        if self.rt_in_matrix:
            mean_rt = mean_homogenous_matrix(rt) if len(rt.shape) > 2 else rt
            out_string += f"\tRTinMatrix	1\n"
            out_string += f"\tRT\n"
            out_string += (
                f"\t\t{mean_rt[0, 0]:0.5f}\t{mean_rt[0, 1]:0.5f}\t{mean_rt[0, 2]:0.5f}\t{mean_rt[0, 3]:0.5f}\n"
            )
            out_string += (
                f"\t\t{mean_rt[1, 0]:0.5f}\t{mean_rt[1, 1]:0.5f}\t{mean_rt[1, 2]:0.5f}\t{mean_rt[1, 3]:0.5f}\n"
            )
            out_string += (
                f"\t\t{mean_rt[2, 0]:0.5f}\t{mean_rt[2, 1]:0.5f}\t{mean_rt[2, 2]:0.5f}\t{mean_rt[2, 3]:0.5f}\n"
            )
            out_string += (
                f"\t\t{mean_rt[3, 0]:0.5f}\t{mean_rt[3, 1]:0.5f}\t{mean_rt[3, 2]:0.5f}\t{mean_rt[3, 3]:0.5f}\n"
            )

        else:
            mean_rt = mean_homogenous_matrix(rt) if len(rt.shape) > 2 else rt
            sequence = "xyz"
            tx, ty, tz = mean_rt[0:3, 3]
            rx, ry, rz = to_euler(mean_rt[:, :, np.newaxis], sequence)
            out_string += f"\tRTinMatrix	0\n"
            out_string += f"\tRT\t{rx[0]:0.5f} {ry[0]:0.5f} {rz[0]:0.5f} {sequence} {tx:0.5f} {ty:0.5f} {tz:0.5f}"

        return out_string

    def __matmul__(self, other):
        if isinstance(other, SegmentCoordinateSystemReal):
            other = other.scs

        if not isinstance(other, np.ndarray):
            raise ValueError(
                "SCS multiplication must be performed against np.narray or SegmentCoordinateSystemReal classes"
            )

        return multiply_homogeneous_matrix(self=self, other=other)

    @property
    def transpose(self):
        out = self.copy()
        out.scs = transpose_homogenous_matrix(out.scs)
        return out
