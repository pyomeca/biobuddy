from copy import deepcopy
from lxml import etree
import numpy as np

from ....utils.aliases import Point, Points
from ....utils.linear_algebra import RotoTransMatrix, get_closest_rt_matrix
from ....utils.linear_algebra import rot2eul


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
        translation
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

    def to_urdf(self, origin: etree.Element):

        origin.set(
            "xyz", f"{self.scs.translation[0]:0.6f} {self.scs.translation[1]:0.6f} {self.scs.translation[2]:0.6f}"
        )
        rpy = self.scs.euler_angles("xyz")
        origin.set("rpy", f"{rpy[0]:0.6f} {rpy[1]:0.6f} {rpy[2]:0.6f}")

    def to_osim(self):
        """
        Note: In OpenSim, the SCS is written as part of the joint's PhysicalOffsetFrame,
        so this method returns the data in a format suitable for the joint writer.
        """
        closest_rt = get_closest_rt_matrix(self.scs.rt_matrix)

        translation = closest_rt[:3, 3]
        angles = rot2eul(closest_rt[:3, :3])

        return translation, angles
