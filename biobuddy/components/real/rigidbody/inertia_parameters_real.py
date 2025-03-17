from typing import Callable

import numpy as np

from .protocols import CoordinateSystemRealProtocol
from ..biomechanical_model_real import BiomechanicalModelReal
from ....utils.aliases import Points, points_to_array
from ....utils.protocols import Data


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
        self.center_of_mass = points_to_array(name="center of mass", points=center_of_mass)
        self.inertia = points_to_array(name="inertia", points=inertia)
        if self.inertia.shape[1] not in [1, 3]:
            raise RuntimeError(f"The inertia must be a np.ndarray of shape (3,) or (3, 3) not {inertia.shape}")

    @staticmethod
    def from_data(
        data: Data,
        relative_mass: Callable[[dict[str, np.ndarray], BiomechanicalModelReal], float],
        center_of_mass: Callable[[dict[str, np.ndarray], BiomechanicalModelReal], np.ndarray],
        inertia: Callable[[dict[str, np.ndarray], BiomechanicalModelReal], np.ndarray],
        kinematic_chain: BiomechanicalModelReal,
        parent_scs: CoordinateSystemRealProtocol = None,
    ):
        """
        This is a constructor for the InertiaParameterReal class.

        Parameters
        ----------
        data
            The data to pick the data from
        relative_mass
            The callback function that returns the relative mass of the segment with respect to the full body
        center_of_mass
            The callback function that returns the position of the center of mass
            from the segment coordinate system on the main axis
        inertia
            The callback function that returns the inertia xx, yy and zz parameters of the segment
        kinematic_chain
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        parent_scs
            The segment coordinate system of the parent to transform the marker from global to local
        """

        mass = relative_mass(data.values, kinematic_chain)

        p = points_to_array(name=f"center_of_mass function", points=center_of_mass(data.values, kinematic_chain))
        p[3, :] = 1  # Do not trust user and make sure the last value is a perfect one
        com = (parent_scs.transpose if parent_scs is not None else np.identity(4)) @ p
        if np.isnan(com).all():
            raise RuntimeError(f"All the values for {com} returned nan which is not permitted")
        inertia = points_to_array(name="inertia parameter function", points=inertia(data.values, kinematic_chain))

        return InertiaParametersReal(mass, com, inertia)

    @property
    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        com = np.nanmean(self.center_of_mass, axis=1)[:3]
        out_string = f"\tmass\t{self.mass}\n"
        out_string += f"\tCenterOfMass\t{com[0]:0.5f}\t{com[1]:0.5f}\t{com[2]:0.5f}\n"
        if self.inertia.shape[1] == 1:
            out_string += f"\tinertia_xxyyzz\t{self.inertia[0]}\t{self.inertia[1]}\t{self.inertia[2]}\n"
        elif self.inertia.shape[1] == 3:
            out_string += f"\tinertia\n"
            out_string += f"\t\t{self.inertia[0, 0]}\t{self.inertia[0, 1]}\t{self.inertia[0, 2]}\n"
            out_string += f"\t\t{self.inertia[1, 0]}\t{self.inertia[1, 1]}\t{self.inertia[1, 2]}\n"
            out_string += f"\t\t{self.inertia[2, 0]}\t{self.inertia[2, 1]}\t{self.inertia[2, 2]}\n"

        return out_string
