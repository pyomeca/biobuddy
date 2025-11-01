from copy import deepcopy
import numpy as np

from ....utils.linear_algebra import mean_homogenous_matrix, transpose_homogenous_matrix
from ....utils.checks import check_name
from ....utils.linear_algebra import RotoTransMatrix


class InertialMeasurementUnitReal:
    def __init__(
        self,
        name: str,
        parent_name: str = None,
        scs: RotoTransMatrix = RotoTransMatrix(),
        is_technical: bool = True,
        is_anatomical: bool = False,
    ):
        """
        Parameters
        ----------
        name
            The name of the inertial measurement unit
        parent_name
            The name of the parent the inertial measurement unit is attached to
        scs
            The scs of the SegmentCoordinateSystemReal
        is_technical
            If the marker should be flagged as a technical imu
        is_anatomical
            If the marker should be flagged as an anatomical imu

        """
        self.name = name
        self.parent_name = check_name(parent_name)

        self.scs = scs
        self.is_technical = is_technical
        self.is_anatomical = is_anatomical

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def parent_name(self) -> str:
        return self._parent_name

    @parent_name.setter
    def parent_name(self, value: str):
        self._parent_name = value

    @property
    def scs(self) -> RotoTransMatrix:
        return self._scs

    @scs.setter
    def scs(self, value: RotoTransMatrix):
        self._scs = value

    @property
    def is_technical(self) -> bool:
        return self._is_technical

    @is_technical.setter
    def is_technical(self, value: bool):
        self._is_technical = value

    @property
    def is_anatomical(self) -> bool:
        return self._is_anatomical

    @is_anatomical.setter
    def is_anatomical(self, value: bool):
        self._is_anatomical = value

    def to_biomod(self):
        out_string = f"imu\t{self.name}\n"
        out_string += f"\tparent\t{self.parent_name}\n"

        mean_rt = (
            self.scs.rt_matrix
        )  # See if we want to do like SegmentCoordinateSystemReal and use mean_homogenous_matrix
        out_string += f"\tRTinMatrix	1\n"
        out_string += f"\tRT\n"
        out_string += f"\t\t{mean_rt[0, 0]:0.6f}\t{mean_rt[0, 1]:0.6f}\t{mean_rt[0, 2]:0.6f}\t{mean_rt[0, 3]:0.6f}\n"
        out_string += f"\t\t{mean_rt[1, 0]:0.6f}\t{mean_rt[1, 1]:0.6f}\t{mean_rt[1, 2]:0.6f}\t{mean_rt[1, 3]:0.6f}\n"
        out_string += f"\t\t{mean_rt[2, 0]:0.6f}\t{mean_rt[2, 1]:0.6f}\t{mean_rt[2, 2]:0.6f}\t{mean_rt[2, 3]:0.6f}\n"
        out_string += f"\t\t{mean_rt[3, 0]:0.6f}\t{mean_rt[3, 1]:0.6f}\t{mean_rt[3, 2]:0.6f}\t{mean_rt[3, 3]:0.6f}\n"

        out_string += f"\ttechnical\t{1 if self.is_technical else 0}\n"
        out_string += f"\tanatomical\t{1 if self.is_anatomical else 0}\n"
        out_string += "endimu\n"
        return out_string

    def to_osim(self):
        raise NotImplementedError("Writing .osim files with IMU is not possible yet.")
