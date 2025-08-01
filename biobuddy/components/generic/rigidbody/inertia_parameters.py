from typing import Callable
import numpy as np

from ....utils.protocols import Data
from ....utils.aliases import points_to_array, inertia_to_array
from ....utils.linear_algebra import RotoTransMatrix


class InertiaParameters:
    def __init__(
        self,
        mass: Callable = None,
        center_of_mass: Callable = None,
        inertia: Callable = None,
        is_local: bool = True,
    ):
        """
        This is a pre-constructor for the InertiaParametersReal class. It allows to create a
        generic model by marker names

        Parameters
        ----------
        mass
            The callback function that returns the mass of the segment with respect to the full body
        center_of_mass
            The callback function that returns the position of the center of mass
            from the segment coordinate system on the main axis
        inertia
            The callback function that returns the inertia xx, yy and zz parameters of the segment
        is_local
            If True, the inertia parameters are expressed in the local segment coordinate system.
            If False, the inertia parameters are expressed in the global coordinate system.
            This is useful for the generic model where the inertia parameters are not yet computed
            and will be computed later when the segment coordinate system is known.
        """
        self.relative_mass = mass
        self.center_of_mass = center_of_mass
        self.inertia = inertia
        self.is_local = is_local

    @staticmethod
    def radii_of_gyration_to_inertia(
        mass: float, coef: tuple[float, float, float], start: np.ndarray, end: np.ndarray
    ) -> np.ndarray:
        """
        Computes the xx, yy and zz values of the matrix of inertia from the segment length. The radii of gyration used are
        'coef * length', where length is '||end - start||'

        Parameters
        ----------
        mass
            The mass of the segment
        coef
            The coefficient of the length of the segment that gives the radius of gyration about x, y and z
        start
            The starting point of the segment
        end
            The end point of the segment

        Returns
        -------
        The xx, yy, zz values of the matrix of inertia
        """

        if len(start.shape) == 1:
            start = start[:, np.newaxis]
        if len(end.shape) == 1:
            end = end[:, np.newaxis]

        length = np.nanmean(np.linalg.norm(end[:3, :] - start[:3, :], axis=0))
        r_2 = (np.array(coef) * length) ** 2
        return mass * r_2

    def to_inertia(
        self,
        data: Data,
        model: "BiomechanicalModelReal",
        scs: "SegmentCoordinateSystemReal",
    ) -> "InertiaParametersReal":
        """
        This constructs an InertiaParameterReal by evaluating the function that defines the contact to get an actual position

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        scs
            The segment coordinate system that the inertia parameters are expressed in. If the inertia parameters are
            expressed in the global coordinate system, this is not used.
        """
        from ...real.rigidbody.inertia_parameters_real import InertiaParametersReal

        if self.is_local:
            scs = RotoTransMatrix()
        elif scs is None:
            raise RuntimeError(
                "If you want to provide a global mesh, you must provide the segment's coordinate system."
            )

        # Mass
        if self.relative_mass is None:
            raise RuntimeError("To compute the inertia parameters, you must provide a mass function.")
        mass = self.relative_mass(data.values, model)

        # Center of mass
        if self.center_of_mass is None:
            raise RuntimeError("To compute the inertia parameters, you must provide a center of mass function.")
        com_p = points_to_array(points=self.center_of_mass(data.values, model), name=f"center_of_mass function")
        # Transform into local coordinates if needed
        com = scs.inverse @ com_p
        if np.isnan(com).all():
            raise RuntimeError(f"All the values for {com} returned nan which is not permitted")

        # Inertia
        if self.inertia is None:
            raise RuntimeError("To compute the inertia parameters, you must provide a inertia function.")
        inertia = inertia_to_array(inertia=self.inertia(data.values, model), name="inertia parameter function")
        # Do not transform inertia because it does not make any sens to express it elsewhere than at the CoM

        return InertiaParametersReal(mass, com, inertia)
