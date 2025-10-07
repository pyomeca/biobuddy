from typing import Callable

import numpy as np

from ....utils.aliases import Point, point_to_array, Points, points_to_array
from ....utils.protocols import Data
from ....utils.enums import Translations
from ....utils.checks import check_name


class ContactReal:
    def __init__(
        self,
        name: str,
        parent_name: str = None,
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
        self.parent_name = check_name(parent_name)
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
        self._position = points_to_array(points=value, name="position")

    @property
    def axis(self) -> Translations:
        return self._axis

    @axis.setter
    def axis(self, value: Translations):
        self._axis = value

    def to_biomod(self):
        if self.axis is None:
            raise RuntimeError("The axis of the contact must be defined before exporting to biomod.")
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"contact\t{self.name}\n"
        out_string += f"\tparent\t{self.parent_name}\n"
        out_string += f"\tposition\t{np.round(self.position[0, 0], 4)}\t{np.round(self.position[1, 0], 4)}\t{np.round(self.position[2, 0], 4)}\n"
        out_string += f"\taxis\t{self.axis.value}\n"
        out_string += "endcontact\n"
        return out_string

    def to_osim(self):
        raise NotImplementedError(
            "Writing contacts into a .osim fil is not implemented, yet. If you need this feature, please open an issue on GitHub."
        )
