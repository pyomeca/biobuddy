from typing import Self

import numpy as np

from ....utils.linear_algebra import mean_homogenous_matrix, transpose_homogenous_matrix


class InertialMeasurementUnitReal:
    def __init__(
        self,
        name: str,
        parent_name: str,
        scs: np.ndarray = np.identity(4),
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
        parent_scs
            The scs of the parent (is used when printing the model so SegmentCoordinateSystemReal
            is in parent's local reference frame
        is_scs_local
            If the scs is already in local reference frame
        is_technical
            If the marker should be flagged as a technical imu
        is_anatomical
            If the marker should be flagged as an anatomical imu

        """
        self.name = name
        self.parent_name = parent_name

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
    def scs(self) -> np.ndarray:
        return self._scs

    @scs.setter
    def scs(self, value: np.ndarray):
        if value.shape != (4, 4):
            raise ValueError("The scs must be a 4x4 matrix")
        self._scs = value
        if len(self._scs.shape) == 2:
            self._scs = self._scs[:, :, np.newaxis]

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

    def copy(self) -> Self:
        return InertialMeasurementUnitReal(name=self.name, parent_name=self.parent_name, scs=np.array(self.scs))

    def to_biomod(self):
        out_string = f"imu\t{self.name}\n"
        out_string += f"\tparent\t{self.parent_name}\n"

        rt = self.scs
        mean_rt = mean_homogenous_matrix(rt) if rt.shape[2] > 1 else rt[:, :, 0]
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

    @property
    def transpose(self) -> Self:
        out = self.copy()
        out.scs = transpose_homogenous_matrix(out.scs)
        return out
