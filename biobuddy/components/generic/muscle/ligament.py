from typing import Callable, Any, TYPE_CHECKING

from ..muscle.via_point import ViaPoint
from ...ligament_utils import LigamentType
from ....utils.marker_data import MarkerData
from ....utils.named_list import NamedList
from ....utils.linear_algebra import RotoTransMatrix

if TYPE_CHECKING:
    from ...real.biomechanical_model_real import BiomechanicalModelReal
    from ...real.muscle.muscle_real import MuscleReal


class Muscle:
    def __init__(
        self,
        name: str,
        ligament_type: LigamentType,
        origin_position: ViaPoint,
        insertion_position: ViaPoint,
        maximal_force_function: Callable[[dict[str, Any], Any], float],
        ligament_slack_length_function: Callable[[dict[str, Any], Any], float],
        damping_function: Callable[[dict[str, Any], Any], float],
    ):
        """
        Parameters
        ----------
        name
            The name of the ligament
        ligament_type
            The type of the ligament
        origin_position
            The origin position of the ligament in the local reference frame of the origin segment
        insertion_position
            The insertion position of the ligament the local reference frame of the insertion segment
        maximal_force_function
            The function giving the maximal force of the ligament can reach
        ligament_slack_length_function
            The function giving the length of the ligament at rest
        damping_function
            The function giving the damping of the ligament
        """
        super().__init__()

        self.name = name
        self.ligament_type = ligament_type
        self.origin_position = origin_position
        self.insertion_position = insertion_position
        self.maximal_force_function = maximal_force_function
        self.ligament_slack_length_function = ligament_slack_length_function
        self.damping_function = damping_function

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def ligament_type(self) -> LigamentType:
        return self._ligament_type

    @ligament_type.setter
    def ligament_type(self, value: LigamentType | str):
        if isinstance(value, str):
            value = LigamentType(value)
        self._ligament_type = value

    @property
    def origin_position(self) -> ViaPoint:
        return self._origin_position

    @origin_position.setter
    def origin_position(self, value: ViaPoint):
        if value is None:
            self._origin_position = None
        else:
            self._origin_position = value

    @property
    def insertion_position(self) -> ViaPoint:
        return self._insertion_position

    @insertion_position.setter
    def insertion_position(self, value: ViaPoint):
        if value is None:
            self._insertion_position = None
        else:
            self._insertion_position = value

    @property
    def maximal_force_function(self) -> Callable[[dict[str, Any], Any], float]:
        return self._maximal_force_function

    @maximal_force_function.setter
    def maximal_force_function(self, value: Callable[[dict[str, Any], Any], float]):
        self._maximal_force_function = value

    @property
    def ligament_slack_length_function(self) -> Callable[[dict[str, Any], Any], float]:
        return self._ligament_slack_length_function

    @ligament_slack_length_function.setter
    def ligament_slack_length_function(self, value: Callable[[dict[str, Any], Any], float]):
        self._ligament_slack_length_function = value

    @property
    def damping_function(self) -> Callable[[dict[str, Any], Any], float]:
        return self._damping_function

    @damping_function.setter
    def damping_function(self, value: Callable[[dict[str, Any], Any], float]):
        self._damping_function = value

    def to_ligament(self, data: MarkerData, model: "BiomechanicalModelReal", scs: RotoTransMatrix) -> "LigamentReal":
        """
        This constructs a LigamentReal by evaluating the function that defines the ligament to get an actual position

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
        from ...real.muscle.ligament_real import LigamentReal

        origin_position = self.origin_position.to_via_point(data, model, scs)
        insertion_position = self.insertion_position.to_via_point(data, model, scs)
        ligament_real = LigamentReal(
            self.name,
            self.ligament_type,
            origin_position,
            insertion_position,
            maximal_force=self.maximal_force_function(model, data),
            ligament_slack_length=self.tendon_slack_length_function(model, data),
            damping=self.damping_function(model, data),
        )

        return ligament_real
