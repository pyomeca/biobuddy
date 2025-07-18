from typing import Callable

import numpy as np
from enum import Enum

from ....utils.aliases import Points, point_to_array, points_to_array
from ....utils.protocols import Data


class MuscleType(Enum):
    HILL = "hill"
    HILL_THELEN = "hillthelen"
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
        origin_position: Points,  # TODO: should be a ViaPointReal (so that we allow for condition and movement)
        insertion_position: Points,  # TODO: should be a ViaPointReal (so that we allow for condition and movement)
        optimal_length: float = None,
        maximal_force: float = None,
        tendon_slack_length: float = None,
        pennation_angle: float = None,
        maximal_velocity: float = None,
        maximal_excitation: float = None,
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
            The origin position of the muscle in the local reference frame of the origin segment
        insertion_position
            The insertion position of the muscle the local reference frame of the insertion segment
        optimal_length
            The optimal length of the muscle
        maximal_force
            The maximal force of the muscle can reach
        tendon_slack_length
            The length of the tendon at rest
        pennation_angle
            The pennation angle of the muscle
        maximal_velocity
            The maximal contraction velocity of the muscle (a common value is 10 m/s)
        maximal_excitation
            The maximal excitation of the muscle (usually 1.0, since it is normalized)
        """
        if optimal_length is not None and optimal_length <= 0:
            raise ValueError("The optimal length of the muscle must be greater than 0.")
        if maximal_force is not None and maximal_force <= 0:
            raise ValueError("The maximal force of the muscle must be greater than 0.")
        if maximal_velocity is not None and maximal_velocity <= 0:
            raise ValueError("The maximal contraction velocity of the muscle must be greater than 0.")
        if maximal_excitation is not None and maximal_excitation <= 0:
            raise ValueError("The maximal excitation of the muscle must be greater than 0.")

        self.name = name
        self.muscle_type = muscle_type
        self.state_type = state_type
        self.muscle_group = muscle_group
        self.origin_position = origin_position
        self.insertion_position = insertion_position
        self.optimal_length = optimal_length
        self.maximal_force = maximal_force
        self.tendon_slack_length = tendon_slack_length
        self.pennation_angle = pennation_angle
        self.maximal_velocity = maximal_velocity
        self.maximal_excitation = maximal_excitation
        # TODO: missing PCSA and

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def muscle_type(self) -> MuscleType:
        return self._muscle_type

    @muscle_type.setter
    def muscle_type(self, value: MuscleType | str):
        if isinstance(value, str):
            value = MuscleType(value)
        self._muscle_type = value

    @property
    def state_type(self) -> MuscleStateType:
        return self._state_type

    @state_type.setter
    def state_type(self, value: MuscleStateType | str):
        if isinstance(value, str):
            value = MuscleStateType(value)
        self._state_type = value

    @property
    def muscle_group(self) -> str:
        return self._muscle_group

    @muscle_group.setter
    def muscle_group(self, value: str):
        self._muscle_group = value

    @property
    def origin_position(self) -> np.ndarray:
        return self._origin_position

    @origin_position.setter
    def origin_position(self, value: Points):
        self._origin_position = point_to_array(point=value, name="origin position")

    @property
    def insertion_position(self) -> np.ndarray:
        return self._insertion_position

    @insertion_position.setter
    def insertion_position(self, value: Points):
        self._insertion_position = point_to_array(point=value, name="insertion position")

    @property
    def optimal_length(self) -> float:
        return self._optimal_length

    @optimal_length.setter
    def optimal_length(self, value: float):
        self._optimal_length = value

    @property
    def maximal_force(self) -> float:
        return self._maximal_force

    @maximal_force.setter
    def maximal_force(self, value: float):
        self._maximal_force = value

    @property
    def tendon_slack_length(self) -> float:
        return self._tendon_slack_length

    @tendon_slack_length.setter
    def tendon_slack_length(self, value: float):
        self._tendon_slack_length = value

    @property
    def pennation_angle(self) -> float:
        return self._pennation_angle

    @pennation_angle.setter
    def pennation_angle(self, value: float):
        self._pennation_angle = value

    @property
    def maximal_velocity(self) -> float:
        return self._maximal_velocity

    @maximal_velocity.setter
    def maximal_velocity(self, value: float):
        self._maximal_velocity = value

    @property
    def maximal_excitation(self) -> float:
        return self._maximal_excitation

    @maximal_excitation.setter
    def maximal_excitation(self, value: float):
        self._maximal_excitation = value

    @staticmethod
    def from_data(
        data: Data,
        model: "BiomechanicalModel",
        name: str,
        muscle_type: MuscleType,
        state_type: MuscleStateType,
        muscle_group: str,
        origin_position_function: Callable[[dict[str, np.ndarray], "BiomechanicalModelReal"], Points],
        insertion_position_function: Callable[[dict[str, np.ndarray], "BiomechanicalModelReal"], Points],
        optimal_length_function: Callable[[dict[str, np.ndarray], "BiomechanicalModelReal"], Points],
        maximal_force_function: Callable[[dict[str, np.ndarray], "BiomechanicalModelReal"], Points],
        tendon_slack_length_function: Callable[[dict[str, np.ndarray], "BiomechanicalModelReal"], Points],
        pennation_angle_function: Callable[[dict[str, np.ndarray], "BiomechanicalModelReal"], Points],
        maximal_excitation: float,
    ):
        """
        This is a constructor for the Muscle class. It evaluates the function that defines the muscle to get an
        actual position

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
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
        maximal_excitation
            The maximal excitation of the muscle (usually 1.0, since it is normalized)
        """
        origin_position = points_to_array(
            points=origin_position_function(data.values, model), name="muscle origin function"
        )
        insertion_position = points_to_array(
            points=insertion_position_function(data.values, model), name="muscle insertion function"
        )
        return MuscleReal(
            name,
            muscle_type,
            state_type,
            muscle_group,
            origin_position,
            insertion_position,
            optimal_length=optimal_length_function(data.values, model),
            maximal_force=maximal_force_function(data.values, model),
            tendon_slack_length=tendon_slack_length_function(data.values, model),
            pennation_angle=pennation_angle_function(data.values, model),
            maximal_excitation=maximal_excitation,
        )

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"muscle\t{self.name}\n"
        out_string += f"\ttype\t{self.muscle_type.value}\n"
        out_string += f"\tstatetype\t{self.state_type.value}\n"
        out_string += f"\tmusclegroup\t{self.muscle_group}\n"
        out_string += f"\toriginposition\t{np.round(self.origin_position[0, 0], 4)}\t{np.round(self.origin_position[1, 0], 4)}\t{np.round(self.origin_position[2, 0], 4)}\n"
        out_string += f"\tinsertionposition\t{np.round(self.insertion_position[0, 0], 4)}\t{np.round(self.insertion_position[1, 0], 4)}\t{np.round(self.insertion_position[2, 0], 4)}\n"
        if isinstance(self.optimal_length, (float, int)):
            out_string += f"\toptimallength\t{self.optimal_length:0.4f}\n"
        out_string += f"\tmaximalforce\t{self.maximal_force:0.4f}\n"
        if isinstance(self.tendon_slack_length, (float, int)):
            out_string += f"\ttendonslacklength\t{self.tendon_slack_length:0.4f}\n"
        if isinstance(self.pennation_angle, (float, int)):
            out_string += f"\tpennationangle\t{self.pennation_angle:0.4f}\n"
        if isinstance(self.maximal_velocity, (float, int)):
            out_string += f"\tmaxvelocity\t{self.maximal_velocity:0.4f}\n"
        if isinstance(self.maximal_excitation, (float, int)):
            out_string += f"\tmaxexcitation\t{self.maximal_excitation:0.4f}\n"
        out_string += "endmuscle\n"
        return out_string
