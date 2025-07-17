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
        self._inertia = points_to_array(points=value, name="inertia")
        if self.inertia.shape[1] == 0:
            return
        if self.inertia.shape[1] == 1:
            self._inertia = np.diag(self.inertia[:, 0])
        elif self.inertia.shape[1] != 3:
            raise RuntimeError(f"The inertia must be a np.ndarray of shape (3,) or (3, 3) not {self.inertia.shape}")

    @staticmethod
    def from_data(
        data: Data,
        model: BiomechanicalModelReal,
        relative_mass: Callable[[dict[str, np.ndarray], BiomechanicalModelReal], float],
        center_of_mass: Callable[[dict[str, np.ndarray], BiomechanicalModelReal], np.ndarray],
        inertia: Callable[[dict[str, np.ndarray], BiomechanicalModelReal], np.ndarray],
        parent_scs: CoordinateSystemRealProtocol = None,
    ):
        """
        This is a constructor for the InertiaParameterReal class.

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        relative_mass
            The callback function that returns the relative mass of the segment with respect to the full body
        center_of_mass
            The callback function that returns the position of the center of mass
            from the segment coordinate system on the main axis
        inertia
            The callback function that returns the inertia xx, yy and zz parameters of the segment
        parent_scs
            The segment coordinate system of the parent to transform the marker from global to local
        """

        mass = relative_mass(data.values, model)

        p = points_to_array(points=center_of_mass(data.values, model), name=f"center_of_mass function")
        p[3, :] = 1  # Do not trust user and make sure the last value is a perfect one
        com = (parent_scs.scs.inverse.rt_matrix if parent_scs is not None else RotoTransMatrix()) @ p
        if np.isnan(com).all():
            raise RuntimeError(f"All the values for {com} returned nan which is not permitted")
        inertia = points_to_array(points=inertia(data.values, model), name="inertia parameter function")

        return InertiaParametersReal(mass, com, inertia)

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
