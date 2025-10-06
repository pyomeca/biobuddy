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

    def to_biomod(self):
        # Do a sanity check
        if np.any(np.isnan(self.positions)):
            raise RuntimeError("The mesh contains nan values")

        out_string = ""
        for p in self.positions.T:
            out_string += f"\tmesh\t{p[0]:0.6f}\t{p[1]:0.6f}\t{p[2]:0.6f}\n"
        return out_string

    def to_osim(self):
        raise NotImplementedError("Writing .osim files with meshes is not possible yet.")
