from typing import Callable

import numpy as np

from ....utils.aliases import Points, points_to_array
from ....utils.protocols import Data
from ....utils.checks import check_name


class ViaPointReal:
    def __init__(
        self,
        name: str,
        parent_name: str,
        muscle_name: str,
        muscle_group: str,
        position: Points = None,
    ):
        """
        Parameters
        ----------
        name
            The name of the new via point
        parent_name
            The name of the parent the via point is attached to
        muscle_name
            The name of the muscle that passes through this via point
        muscle_group
            The muscle group the muscle belongs to
        position
            The 3d position of the via point in the local reference frame
        """
        self.name = name
        self.parent_name = check_name(parent_name)
        self.muscle_name = muscle_name
        self.muscle_group = muscle_group
        self.position = position

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
    def muscle_name(self) -> str:
        return self._muscle_name

    @muscle_name.setter
    def muscle_name(self, value: str) -> None:
        self._muscle_name = value

    @property
    def muscle_group(self) -> str:
        return self._muscle_group

    @muscle_group.setter
    def muscle_group(self, value: str) -> None:
        self._muscle_group = value

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: Points) -> None:
        self._position = points_to_array(points=value, name="viapoint")

    @staticmethod
    def from_data(
        data: Data,
        model: "BiomechanicalModel",
        name: str,
        parent_name: str,
        muscle_name: str,
        muscle_group: str,
        position_function: Callable[[dict[str, np.ndarray], "BiomechanicalModelReal"], Points],
    ):
        """
        This is a constructor for the Contact class. It evaluates the function that defines the contact to get an
        actual position

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        name
            The name of the new via point
        parent_name
            The name of the parent the via point is attached to
        muscle_name
            The name of the muscle that passes through this via point
        muscle_group
            The muscle group the muscle belongs to
        position_function
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the via point in the local joint coordinates.
        """

        # Get the position of the contact points and do some sanity checks
        position = points_to_array(points=position_function(data.values, model), name="viapoint function")
        return ViaPointReal(name, parent_name, muscle_name, muscle_group, position)

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"viapoint\t{self.name}\n"
        out_string += f"\tparent\t{self.parent_name}\n"
        out_string += f"\tmuscle\t{self.muscle_name}\n"
        out_string += f"\tmusclegroup\t{self.muscle_group}\n"
        out_string += f"\tposition\t{np.round(self.position[0, 0], 6)}\t{np.round(self.position[1, 0], 6)}\t{np.round(self.position[2, 0], 6)}\n"
        out_string += "endviapoint\n"
        return out_string
