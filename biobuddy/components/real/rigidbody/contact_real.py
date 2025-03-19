from typing import Callable

import numpy as np

from ....utils.aliases import Point, point_to_array, Points, points_to_array
from ....utils.protocols import Data
from ....utils.translations import Translations


class ContactReal:
    def __init__(
        self,
        name: str,
        parent_name: str,
        position: Point = None,
        axis: Translations = None,
    ):
        """
        Parameters
        ----------
        name
            The name of the new contact
        parent_name
            The name of the parent the contact is attached to
        position
            The 3d position of the contact
        axis
            The axis of the contact
        """
        self.name = name
        self.parent_name = parent_name
        self.position = position
        self.axis = axis

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def parent_name(self) -> str:
        return self._parent_name

    @parent_name.setter
    def parent_name(self, value: str):
        self._parent_name = value

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: Point):
        self._position = point_to_array(name="position", point=value)

    @property
    def axis(self) -> Translations:
        return self._axis

    @axis.setter
    def axis(self, value: Translations):
        self._axis = value

    @staticmethod
    def from_data(
        data: Data,
        name: str,
        function: Callable[[dict[str, np.ndarray]], Points],
        parent_name: str,
        axis: Translations = None,
    ):
        """
        This is a constructor for the Contact class. It evaluates the function that defines the contact to get an
        actual position

        Parameters
        ----------
        data
            The data to pick the data from
        name
            The name of the new contact
        function
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the contacts in the local joint coordinates.
        parent_name
            The name of the parent the contact is attached to
        axis
            The axis of the contact
        """

        # Get the position of the contact points and do some sanity checks
        p = points_to_array(name=f"contact real function", points=function(data.values))
        return ContactReal(name, parent_name, p, axis)

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"contact\t{self.name}\n"
        out_string += f"\tparent\t{self.parent_name}\n"
        out_string += f"\tposition\t{np.round(self.position[0], 4)}\t{np.round(self.position[1], 4)}\t{np.round(self.position[2], 4)}\n"
        out_string += f"\taxis\t{self.axis.value}\n"
        out_string += "endcontact\n"
        return out_string
