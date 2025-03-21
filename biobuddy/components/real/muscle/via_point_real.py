from typing import Callable

import numpy as np


from ....utils.aliases import Points, points_to_array
from ....utils.protocols import Data


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
        self.parent_name = parent_name
        self.muscle_name = muscle_name
        self.muscle_group = muscle_group
        self.position = points_to_array(name="viapoint", points=position)

    @staticmethod
    def from_data(
        data: Data,
        name: str,
        parent_name: str,
        muscle_name: str,
        muscle_group: str,
        position_function: Callable[[dict[str, np.ndarray]], Points],
    ):
        """
        This is a constructor for the Contact class. It evaluates the function that defines the contact to get an
        actual position

        Parameters
        ----------
        data
            The data to pick the data from
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
        position = points_to_array(name="viapoint function", points=position_function(data.values))
        return ViaPointReal(name, parent_name, muscle_name, muscle_group, position)

    @property
    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"viapoint\t{self.name}\n"
        out_string += f"\tparent\t{self.parent_name}\n"
        out_string += f"\tmuscle\t{self.muscle_name}\n"
        out_string += f"\tmusclegroup\t{self.muscle_group}\n"
        out_string += f"\tposition\t{np.round(self.position[0], 4)}\t{np.round(self.position[1], 4)}\t{np.round(self.position[2], 4)}\n"
        out_string += "endviapoint\n"
        return out_string
