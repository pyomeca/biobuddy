from typing import Callable

import numpy as np

from .protocols import CoordinateSystemRealProtocol
from ..biomechanical_model_real import BiomechanicalModelReal
from ....utils.aliases import Point, point_to_array, Points, points_to_array
from ....utils.protocols import Data


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

        self.positions = points_to_array(name="positions", points=positions)

    @staticmethod
    def from_data(
        data: Data,
        functions: tuple[Callable[[dict[str, np.ndarray], BiomechanicalModelReal], Point], ...],
        kinematic_chain: BiomechanicalModelReal,
        parent_scs: CoordinateSystemRealProtocol = None,
    ):
        """
        This is a constructor for the MeshReal class. It evaluates the functions that defines the mesh to get
        actual positions

        Parameters
        ----------
        data
            The data to pick the data from
        functions
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the mesh points
        kinematic_chain
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        parent_scs
            The segment coordinate system of the parent to transform the marker from global to local
        """

        # Get the position of the all the mesh points and do some sanity checks
        all_p = []
        for f in functions:
            p = point_to_array(name="mesh function", point=f(data.values, kinematic_chain))
            p[3, :] = 1  # Do not trust user and make sure the last value is a perfect one
            projected_p = (parent_scs.transpose if parent_scs is not None else np.identity(4)) @ p
            if np.isnan(projected_p).all():
                raise RuntimeError(f"All the values for {f} returned nan which is not permitted")
            all_p.append(projected_p)

        return MeshReal(tuple(all_p))

    @property
    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = ""
        for position in self.positions:
            # Do a sanity check
            p = np.nanmean(position, axis=1)
            out_string += f"\tmesh\t{p[0]:0.4f}\t{p[1]:0.4f}\t{p[2]:0.4f}\n"
        return out_string
