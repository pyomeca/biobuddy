from typing import Self

import numpy as np

from ....utils.linear_algebra import mean_homogenous_matrix, transpose_homogenous_matrix


class InertialMeasurementUnitReal:
    def __init__(
        self,
        name: str,
        parent_name: str,
        scs: np.ndarray = np.identity(4),
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
        """
        self.name = name
        self.parent_name = parent_name

        if scs.shape != (4, 4):
            raise ValueError("The scs must be a 4x4 matrix")
        self.scs = scs
        if len(self.scs.shape) == 2:
            self.scs = self.scs[:, :, np.newaxis]

    def copy(self) -> Self:
        return InertialMeasurementUnitReal(name=self.name, parent_name=self.parent_name, scs=np.array(self.scs))

    @property
    def to_biomod(self):
        rt = self.scs
        if self.is_in_global:
            rt = self.parent_scs.transpose @ self.scs if self.parent_scs else np.identity(4)[:, :, np.newaxis]

        out_string = ""
        mean_rt = mean_homogenous_matrix(rt) if len(rt.shape) > 2 else rt
        out_string += f"\tRTinMatrix	1\n"
        out_string += f"\tRT\n"
        out_string += f"\t\t{mean_rt[0, 0]:0.5f}\t{mean_rt[0, 1]:0.5f}\t{mean_rt[0, 2]:0.5f}\t{mean_rt[0, 3]:0.5f}\n"
        out_string += f"\t\t{mean_rt[1, 0]:0.5f}\t{mean_rt[1, 1]:0.5f}\t{mean_rt[1, 2]:0.5f}\t{mean_rt[1, 3]:0.5f}\n"
        out_string += f"\t\t{mean_rt[2, 0]:0.5f}\t{mean_rt[2, 1]:0.5f}\t{mean_rt[2, 2]:0.5f}\t{mean_rt[2, 3]:0.5f}\n"
        out_string += f"\t\t{mean_rt[3, 0]:0.5f}\t{mean_rt[3, 1]:0.5f}\t{mean_rt[3, 2]:0.5f}\t{mean_rt[3, 3]:0.5f}\n"

        return out_string

    @property
    def transpose(self) -> Self:
        out = self.copy()
        out.scs = transpose_homogenous_matrix(out.scs)
        return out
