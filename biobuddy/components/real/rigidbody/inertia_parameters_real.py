from typing import Callable

import numpy as np

from .protocols import CoordinateSystemRealProtocol
from ..biomechanical_model_real import BiomechanicalModelReal
from ....utils.aliases import Points, points_to_array
from ....utils.protocols import Data
from ....utils.linear_algebra import RotoTransMatrix


class InertiaParametersReal:
    def __init__(
        self,
        mass: float = None,
        center_of_mass: Points = None,
        inertia: Points = None,
    ):
        """
        Parameters
        ----------
        mass
            The mass of the segment with respect to the full body
        center_of_mass
            The position of the center of mass from the segment coordinate system on the main axis
        inertia
            The inertia xx, yy and zz parameters of the segment
        """
        self.mass = mass
        self.center_of_mass = center_of_mass
        self.inertia = inertia

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, value: float):
        self._mass = value

    @property
    def center_of_mass(self) -> np.ndarray:
        return self._center_of_mass

    @center_of_mass.setter
    def center_of_mass(self, value: Points):
        self._center_of_mass = points_to_array(points=value, name="center of mass")

    @property
    def inertia(self) -> np.ndarray:
        return self._inertia

    @inertia.setter
    def inertia(self, value: Points):
        if value.shape[1] == 0:
            return
        if value.shape[1] == 1:
            self._inertia = np.diag(value[:, 0])
        elif value.shape == (3, 3) or value.shape == (4, 4):
            self._inertia = points_to_array(points=value, name="inertia")
        else:
            raise RuntimeError(f"The inertia must be a np.ndarray of shape (3,), (4,), (3, 3), or (4, 4) not {value.shape}")

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        if self.mass is not None:
            out_string = f"\tmass\t{self.mass}\n"

        if np.any(self.center_of_mass):
            com = np.nanmean(self.center_of_mass, axis=1)[:3]
            out_string += f"\tCenterOfMass\t{com[0]:0.6f}\t{com[1]:0.6f}\t{com[2]:0.6f}\n"

        if np.any(self.inertia):
            out_string += f"\tinertia\n"
            out_string += f"\t\t{self.inertia[0, 0]:0.6f}\t{self.inertia[0, 1]:0.6f}\t{self.inertia[0, 2]:0.6f}\n"
            out_string += f"\t\t{self.inertia[1, 0]:0.6f}\t{self.inertia[1, 1]:0.6f}\t{self.inertia[1, 2]:0.6f}\n"
            out_string += f"\t\t{self.inertia[2, 0]:0.6f}\t{self.inertia[2, 1]:0.6f}\t{self.inertia[2, 2]:0.6f}\n"

        return out_string
