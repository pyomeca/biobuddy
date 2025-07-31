from typing import Callable
import numpy as np

from ....utils.protocols import Data
from ....utils.enums import Translations
from ....utils.checks import check_name
from ....utils.linear_algebra import RotoTransMatrix
from ....utils.aliases import points_to_array


class Contact:
    def __init__(
        self,
        name: str,
        function: Callable | str = None,
        parent_name: str = None,
        axis: Translations = None,
        is_local: bool = False,
    ):
        """
        Parameters
        ----------
        name
            The name of the new contact

        function
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the contact with.
        parent_name
            The name of the parent the contact is attached to
        axis
            The axis of the contact
        is_local
            Indicates whether the contact is defined in the local segment coordinate system.
            If True, the contact is defined in the local coordinate system of the parent segment.
            If False, the contact is defined in the global coordinate system.
        """
        self.name = name
        self.function = function
        self.parent_name = check_name(parent_name)
        self.axis = axis
        self.is_local = is_local

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def parent_name(self) -> str:
        return self._parent_name

    @parent_name.setter
    def parent_name(self, value: str) -> None:
        self._parent_name = value

    @property
    def function(self) -> Callable | str:
        return self._function

    @function.setter
    def function(self, value: Callable | str) -> None:
        if value is None:
            # Set the function to the name of the marker, so it can be used as a default
            value = self.name
        self._function = (lambda m, bio: np.nanmean(m[value], axis=1)) if isinstance(value, str) else value

    @property
    def axis(self) -> Translations:
        return self._axis

    @axis.setter
    def axis(self, value: Translations) -> None:
        self._axis = value

    def to_contact(self, data: Data, model: "BiomechanicalModelReal", scs: RotoTransMatrix) -> "ContactReal":
        """
        This constructs the ContactReal by evaluating the function that defines the contact to get an actual position

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The biomechanical model to which the contact belongs
        scs
            The segment coordinate system in which the mesh is defined. If None, the mesh is assumed to be in the global
            coordinate system.
        """
        from ...real.rigidbody.contact_real import ContactReal

        if self.function is None:
            raise RuntimeError("You must provide a position function to evaluate the Contact into a ContactReal.")

        if self.is_local:
            scs = RotoTransMatrix()
        elif scs is None:
            raise RuntimeError(
                "If you want to provide a global mesh, you must provide the segment's coordinate system."
            )

        # Get the position of the contact points and do some sanity checks
        p = np.nanmean(points_to_array(points=self.function(data.values, model), name="contact real function"), axis=1)
        projected_p = scs.inverse @ p
        if np.isnan(projected_p).all():
            raise RuntimeError(f"All the values for {self.function} returned nan which is not permitted")

        return ContactReal(self.name, self.parent_name, projected_p, self.axis)
