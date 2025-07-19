from typing import Callable

from ..muscle.via_point import ViaPoint
from ....utils.protocols import Data
from ....utils.named_list import NamedList


class Muscle:
    def __init__(
        self,
        name: str,
        muscle_type: "MuscleType",
        state_type: "MuscleStateType",
        muscle_group: str,
        origin_position: ViaPoint,
        insertion_position: ViaPoint,
        optimal_length_function: Callable[[dict[str, float]], float],
        maximal_force_function: Callable[[dict[str, float]], float],
        tendon_slack_length_function: Callable[[dict[str, float]], float],
        pennation_angle_function: Callable[[dict[str, float]], float],
        maximal_excitation: float = None,
    ):
        """
        Parameters
        ----------
        name
            The name of the muscle
        muscle_type
            The type of the muscle
        """
        self.name = name
        self.muscle_type = muscle_type
        self.state_type = state_type
        self.muscle_group = muscle_group
        self.origin_position = origin_position
        self.insertion_position = insertion_position
        self.optimal_length_function = optimal_length_function
        self.maximal_force_function = maximal_force_function
        self.tendon_slack_length_function = tendon_slack_length_function
        self.pennation_angle_function = pennation_angle_function
        self.maximal_excitation = 1.0 if maximal_excitation is None else maximal_excitation

        self.via_points = NamedList[ViaPoint]()

    def add_via_point(self, via_point: ViaPoint) -> None:
        """
        Add a via point to the model

        Parameters
        ----------
        via_point
            The via point to add
        """
        if via_point.muscle_name is not None and via_point.muscle_name != self.name:
            raise ValueError(
                "The via points's muscle should be the same as the 'key'. Alternatively, via_point.muscle_name can be left undefined"
            )

        via_point.muscle_name = self.name
        self.via_points._append(via_point)

    def remove_via_point(self, via_point_name: str) -> None:
        """
        Remove a via point from the model

        Parameters
        ----------
        via_point_name
            The name of the via point to remove
        """
        self.via_points._remove(via_point_name)

    def to_muscle(self, data: Data, model) -> "MuscleReal":
        from ...real.muscle.muscle_real import MuscleReal

        return MuscleReal.from_data(
            data,
            model,
            self.name,
            self.muscle_type,
            self.state_type,
            self.muscle_group,
            self.origin_position,
            self.insertion_position,
            self.optimal_length_function,
            self.maximal_force_function,
            self.tendon_slack_length_function,
            self.pennation_angle_function,
            self.maximal_excitation,
        )
