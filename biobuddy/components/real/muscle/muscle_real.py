from typing import Callable

import numpy as np
from enum import Enum

from .via_point_real import ViaPointReal
from ....utils.aliases import Points, point_to_array
from ....utils.protocols import Data


class MuscleType(Enum):
    HILL = "hill"
    HILL_THELEN = "hillethelen"
    HILL_DE_GROOTE = "hilldegroote"


class MuscleStateType(Enum):
    DEGROOTE = "degroote"
    DEFAULT = "default"
    BUCHANAN = "buchanan"


class MuscleReal:
    def __init__(
        self,
        name: str,
        muscle_type: MuscleType,
        state_type: MuscleStateType,
        muscle_group: str,
        origin_position: Points,
        insertion_position: Points,
        optimal_length: float,
        maximal_force: float,
        tendon_slack_length: float,
        pennation_angle: float,
        maximal_excitation: float,
        via_points: list[ViaPointReal] = None,
    ):
        """
        Parameters
        ----------
        name
            The name of the new contact
        muscle_type
            The type of the muscle
        state_type
            The state type of the muscle
        muscle_group
            The muscle group the muscle belongs to
        origin_position
            The origin position of the muscle in the local reference frame of the origin segment @pariterre: please confirm
        insertion_position
            The insertion position of the muscle the local reference frame of the insertion segment @pariterre: please confirm
        optimal_length
            The optimal length of the muscle
        maximal_force
            The maximal force of the muscle can reach
        tendon_slack_length
            The length of the tendon at rest
        pennation_angle
            The pennation angle of the muscle
        maximal_excitation
            The maximal excitation of the muscle (usually 1.0, since it is normalized)
        via_points
            The via points of the muscle
        """
        self.name = name
        self.muscle_type = muscle_type
        self.state_type = state_type
        self.muscle_group = muscle_group
        self.origin_position = point_to_array(name="origin position", point=origin_position)
        self.insertion_position = point_to_array(name="insertion position", point=insertion_position)
        self.optimal_length = optimal_length
        self.maximal_force = maximal_force
        self.tendon_slack_length = tendon_slack_length
        self.pennation_angle = pennation_angle
        self.maximal_excitation = maximal_excitation
        self.via_points = via_points
        # TODO: missing PCSA and  maxVelocity

    @staticmethod
    def from_data(
        data: Data,
        name: str,
        muscle_type: MuscleType,
        state_type: MuscleStateType,
        muscle_group: str,
        origin_position_function: Callable[[dict[str, np.ndarray]], Points],
        insertion_position_function: Callable[[dict[str, np.ndarray]], Points],
        optimal_length_function: Callable[[dict[str, np.ndarray]], float],
        maximal_force_function: Callable[[dict[str, np.ndarray]], float],
        tendon_slack_length_function: Callable[[dict[str, np.ndarray]], float],
        pennation_angle_function: Callable[[dict[str, np.ndarray]], float],
        maximal_excitation_function: Callable[[dict[str, np.ndarray]], float],
    ):
        """
        This is a constructor for the Muscle class. It evaluates the function that defines the muscle to get an
        actual position

        Parameters
        ----------
        data
            The data to pick the data from
        name
            The name of the muscle
        muscle_type
            The type of the muscle
        state_type
            The state type of the muscle
        muscle_group
            The muscle group the muscle belongs to
        origin_position_function
            The function (f(m) -> Points, where m is a dict of markers) that defines the origin position of the muscle
        insertion_position_function
            The function (f(m) -> Points, where m is a dict of markers) that defines the insertion position of the muscle
        optimal_length_function
            The function (f(m) -> float, where m is a dict of markers) that defines the optimal length of the muscle
        maximal_force_function
            The function (f(m) -> float, where m is a dict of markers) that defines the maximal force of the muscle
        tendon_slack_length_function
            The function (f(m) -> float, where m is a dict of markers) that defines the tendon slack length of the muscle
        pennation_angle_function
            The function (f(m) -> float, where m is a dict of markers) that defines the pennation angle of the muscle
        maximal_excitation_function
            The function that returns the maximal excitation of the muscle (usually 1.0, since it is normalized)
        """
        origin_position = points_to_array(name="muscle origin function", points=origin_position_function(data.values))
        insertion_position = points_to_array(
            name="muscle insertion function", points=insertion_position_function(data.values)
        )
        return MuscleReal(
            name,
            muscle_type,
            state_type,
            muscle_group,
            origin_position,
            insertion_position,
            optimal_length=optimal_length_function(data.values),
            maximal_force=maximal_force_function(data.values),
            tendon_slack_length=tendon_slack_length_function(data.values),
            pennation_angle=pennation_angle_function(data.values),
            maximal_excitation=maximal_excitation_function(data.values),
        )

    @property
    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"muscle\t{self.name}\n"
        out_string += f"\ttype\t{self.muscle_type.value}\n"
        out_string += f"\tstatetype\t{self.state_type.value}\n"
        out_string += f"\tmusclegroup\t{self.muscle_group}\n"
        out_string += f"\toriginposition\t{np.round(self.origin_position[0, 0], 4)}\t{np.round(self.origin_position[1, 0], 4)}\t{np.round(self.origin_position[2, 0], 4)}\n"
        out_string += f"\tinsertionposition\t{np.round(self.insertion_position[0, 0], 4)}\t{np.round(self.insertion_position[1, 0], 4)}\t{np.round(self.insertion_position[2, 0], 4)}\n"
        out_string += f"\toptimallength\t{self.optimal_length:0.4f}\n"
        out_string += f"\tmaximalforce\t{self.maximal_force:0.4f}\n"
        out_string += f"\ttendonslacklength\t{self.tendon_slack_length:0.4f}\n"
        out_string += f"\tpennationangle\t{self.pennation_angle:0.4f}\n"
        out_string += "endmuscle\n"
        return out_string
