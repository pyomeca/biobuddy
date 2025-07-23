from typing import Callable
import numpy as np

from ...real.biomechanical_model_real import BiomechanicalModelReal
from ...real.rigidbody.contact_real import ContactReal
from ....utils.protocols import Data
from ....utils.translations import Translations
from ....utils.checks import check_name


class Contact:
    def __init__(
        self,
        name: str,
        function: Callable | str = None,
        parent_name: str = None,
        axis: Translations = None,
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
        """
        self.name = name
        self.function = function
        self.parent_name = check_name(parent_name)
        self.axis = axis

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

    def to_contact(self, data: Data, model: BiomechanicalModelReal) -> ContactReal:
        if self.function is None:
            raise RuntimeError("You must provide a position function to evaluate the Contact into a ContactReal.")
        return ContactReal.from_data(
            data,
            model,
            self.name,
            self.function,
            self.parent_name,
            self.axis,
        )
