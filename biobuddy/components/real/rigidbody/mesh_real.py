from typing import Callable

import numpy as np

from .protocols import CoordinateSystemRealProtocol
from ..biomechanical_model_real import BiomechanicalModelReal
from ....utils.aliases import Point, point_to_array, Points, points_to_array
from ....utils.protocols import Data
from ....utils.linear_algebra import RotoTransMatrix


class MeshReal:
    def __init__(
        self,
        positions: Points = None,
    ):
        """
        Parameters
        ----------
        positions
            The 3d position of the all the mesh points
        """
        self.positions = positions

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @positions.setter
    def positions(self, value: Points):
        self._positions = points_to_array(points=value, name="positions")

    def add_positions(self, value: Points):
        self._positions = np.hstack((self._positions, points_to_array(points=value, name="positions")))

    @staticmethod
    def from_data(
        data: Data,
        model: BiomechanicalModelReal,
        functions: tuple[Callable[[dict[str, np.ndarray], BiomechanicalModelReal], Point], ...],
        parent_scs: CoordinateSystemRealProtocol = None,
    ):
        """
        This is a constructor for the MeshReal class. It evaluates the functions that defines the mesh to get
        actual positions

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        functions
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the mesh points
        parent_scs
            The segment coordinate system of the parent to transform the marker from global to local
        """

        # Get the position of the all the mesh points and do some sanity checks
        all_p = points_to_array(points=None, name="mesh_real")
        for f in functions:
            p = point_to_array(point=f(data.values, model), name="mesh function")
            p[3, :] = 1  # Do not trust user and make sure the last value is a perfect one
            projected_p = (parent_scs.scs.inverse if parent_scs is not None else RotoTransMatrix()) @ p
            if np.isnan(projected_p).all():
                raise RuntimeError(f"All the values for {f} returned nan which is not permitted")
            all_p = np.hstack((all_p, projected_p))

        return MeshReal(all_p)

    def to_biomod(self):
        # Do a sanity check
        if np.any(np.isnan(self.positions)):
            raise RuntimeError("The mesh contains nan values")

        out_string = ""
        for p in self.positions.T:
            out_string += f"\tmesh\t{p[0]:0.6f}\t{p[1]:0.6f}\t{p[2]:0.6f}\n"
        return out_string
