from typing import Callable, Any

from ..muscle.via_point import ViaPoint
from ...muscle_utils import MuscleType, MuscleStateType, MuscleUtils
from ....utils.protocols import Data
from ....utils.named_list import NamedList
from ....utils.linear_algebra import RotoTransMatrix


class Muscle(MuscleUtils):
    def __init__(
        self,
        name: str,
        muscle_type: MuscleType,
        state_type: MuscleStateType,
        muscle_group: str,
        origin_position: ViaPoint,
        insertion_position: ViaPoint,
        optimal_length_function: Callable[[dict[str, Any], Any], float],
        maximal_force_function: Callable[[dict[str, Any], Any], float],
        tendon_slack_length_function: Callable[[dict[str, Any], Any], float],
        pennation_angle_function: Callable[[dict[str, Any], Any], float],
        maximal_velocity_function: Callable[[dict[str, Any], Any], float],
        maximal_excitation: float = None,
    ):
        """
        Parameters
        ----------
        name
            The name of the muscle
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
        optimal_length_function
            The function giving the optimal length of the muscle
        maximal_force_function
            The function giving the maximal force of the muscle can reach
        tendon_slack_length_function
            The function giving the length of the tendon at rest
        pennation_angle_function
            The function giving the pennation angle of the muscle
        maximal_velocity_function
            The function giving the maximal contraction velocity of the muscle (a common value is 10 m/s)
        maximal_excitation
            The maximal excitation of the muscle (usually 1.0, since it is normalized)
        """
        super().__init__()

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
        self.maximal_velocity_function = maximal_velocity_function
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
        if via_point.muscle_group is not None and via_point.muscle_group != self.muscle_group:
            raise ValueError(
                f"The via points's muscle group {via_point.muscle_group} should be the same as the muscle's name {self.muscle_group}. Alternatively, via_point.muscle_group can be left undefined"
            )

        via_point.muscle_name = self.name
        via_point.muscle_group = self.muscle_group
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
    def origin_position(self) -> ViaPoint:
        return self._origin_position

    @origin_position.setter
    def origin_position(self, value: ViaPoint):
        if value is None:
            self._origin_position = None
        else:
            if value.muscle_name is not None and value.muscle_name != self.name:
                raise ValueError(
                    f"The origin's muscle {value.muscle_name} should be the same as the muscle's name {self.name}. Alternatively, origin_position.muscle_name can be left undefined"
                )
            value.muscle_name = self.name
            if value.muscle_group is not None and value.muscle_group != self.muscle_group:
                raise ValueError(
                    f"The origin's muscle group {value.muscle_group} should be the same as the muscle's muscle group {self.muscle_group}. Alternatively, origin_position.muscle_group can be left undefined"
                )
            value.muscle_group = self.muscle_group
            self._origin_position = value

    @property
    def insertion_position(self) -> ViaPoint:
        return self._insertion_position

    @insertion_position.setter
    def insertion_position(self, value: ViaPoint):
        if value is None:
            self._insertion_position = None
        else:
            if value.muscle_name is not None and value.muscle_name != self.name:
                raise ValueError(
                    f"The insertion's muscle {value.muscle_name} should be the same as the muscle's name {self.name}. Alternatively, insertion_position.muscle_name can be left undefined"
                )
            value.muscle_name = self.name
            if value.muscle_group is not None and value.muscle_group != self.muscle_group:
                raise ValueError(
                    f"The insertion's muscle group {value.muscle_group} should be the same as the muscle's muscle group {self.muscle_group}. Alternatively, insertion_position.muscle_group can be left undefined"
                )
            value.muscle_group = self.muscle_group
            self._insertion_position = value

    @property
    def optimal_length_function(self) -> Callable[[dict[str, Any], Any], float]:
        return self._optimal_length_function

    @optimal_length_function.setter
    def optimal_length_function(self, value: Callable[[dict[str, Any], Any], float]):
        self._optimal_length_function = value

    @property
    def maximal_force_function(self) -> Callable[[dict[str, Any], Any], float]:
        return self._maximal_force_function

    @maximal_force_function.setter
    def maximal_force_function(self, value: Callable[[dict[str, Any], Any], float]):
        self._maximal_force_function = value

    @property
    def tendon_slack_length_function(self) -> Callable[[dict[str, Any], Any], float]:
        return self._tendon_slack_length_function

    @tendon_slack_length_function.setter
    def tendon_slack_length_function(self, value: Callable[[dict[str, Any], Any], float]):
        self._tendon_slack_length_function = value

    @property
    def pennation_angle_function(self) -> Callable[[dict[str, Any], Any], float]:
        return self._pennation_angle_function

    @pennation_angle_function.setter
    def pennation_angle_function(self, value: Callable[[dict[str, Any], Any], float]):
        self._pennation_angle_function = value

    @property
    def maximal_velocity_function(self) -> Callable[[dict[str, Any], Any], float]:
        return self._maximal_velocity_function

    @maximal_velocity_function.setter
    def maximal_velocity_function(self, value: Callable[[dict[str, Any], Any], float]):
        self._maximal_velocity_function = value

    @property
    def maximal_excitation(self) -> float:
        return self._maximal_excitation

    @maximal_excitation.setter
    def maximal_excitation(self, value: float):
        self._maximal_excitation = value

    def to_muscle(self, data: Data, model: "BiomechanicalModelReal", scs: RotoTransMatrix) -> "MuscleReal":
        """
        This constructs a MuscleReal by evaluating the function that defines the muscle to get an actual position

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        scs
            The segment coordinate system in which the muscle is defined. This is useful for the origin and insertion
            positions to be transformed correctly.
        """
        from ...real.muscle.muscle_real import MuscleReal

        origin_position = self.origin_position.to_via_point(data, model, scs)
        insertion_position = self.insertion_position.to_via_point(data, model, scs)
        muscle_real = MuscleReal(
            self.name,
            self.muscle_type,
            self.state_type,
            self.muscle_group,
            origin_position,
            insertion_position,
            optimal_length=self.optimal_length_function(model, data.values),
            maximal_force=self.maximal_force_function(model, data.values),
            tendon_slack_length=self.tendon_slack_length_function(model, data.values),
            pennation_angle=self.pennation_angle_function(model, data.values),
            maximal_excitation=self.maximal_excitation,
        )

        for via_point in self.via_points:
            via_point_real = via_point.to_via_point(data, model, scs)
            muscle_real.add_via_point(via_point_real)

        return muscle_real
