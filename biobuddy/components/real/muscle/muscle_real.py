import numpy as np
from lxml import etree

from .via_point_real import ViaPointReal
from ....utils.named_list import NamedList
from ...muscle_utils import MuscleType, MuscleStateType, MuscleUtils


class MuscleReal(MuscleUtils):
    def __init__(
        self,
        name: str,
        muscle_type: MuscleType,
        state_type: MuscleStateType,
        muscle_group: str,
        origin_position: ViaPointReal,
        insertion_position: ViaPointReal,
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
        super().__init__()

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

        self.via_points = NamedList[ViaPointReal]()

    def add_via_point(self, via_point: ViaPointReal) -> None:
        """
        Add a via point to the model

        Parameters
        ----------
        via_point
            The via point to add
        """
        if via_point.muscle_name is not None and via_point.muscle_name != self.name:
            raise ValueError(
                f"The via points's muscle {via_point.muscle_name} should be the same as the muscle's name {self.name}. Alternatively, via_point.muscle_name can be left undefined"
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
    def origin_position(self) -> ViaPointReal:
        return self._origin_position

    @origin_position.setter
    def origin_position(self, value: ViaPointReal):
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
    def insertion_position(self) -> ViaPointReal:
        return self._insertion_position

    @insertion_position.setter
    def insertion_position(self, value: ViaPointReal):
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
    def optimal_length(self) -> float:
        return self._optimal_length

    @optimal_length.setter
    def optimal_length(self, value: float):
        if value is not None and value <= 0:
            raise ValueError("The optimal length of the muscle must be greater than 0.")
        if isinstance(value, np.ndarray):
            if value.shape == (1,):
                value = value[0]
            else:
                raise ValueError("The optimal length must be a float.")
        self._optimal_length = value

    @property
    def maximal_force(self) -> float:
        return self._maximal_force

    @maximal_force.setter
    def maximal_force(self, value: float):
        if value is not None and value <= 0:
            raise ValueError("The maximal force of the muscle must be greater than 0.")
        if isinstance(value, np.ndarray):
            if value.shape == (1,):
                value = value[0]
            else:
                raise ValueError("The maximal force must be a float.")
        self._maximal_force = value

    @property
    def tendon_slack_length(self) -> float:
        return self._tendon_slack_length

    @tendon_slack_length.setter
    def tendon_slack_length(self, value: float):
        if value is not None and value <= 0:
            raise ValueError("The tendon slack length of the muscle must be greater than 0.")
        if isinstance(value, np.ndarray):
            if value.shape == (1,):
                value = value[0]
            else:
                raise ValueError("The tendon slack length must be a float.")
        self._tendon_slack_length = value

    @property
    def pennation_angle(self) -> float:
        return self._pennation_angle

    @pennation_angle.setter
    def pennation_angle(self, value: float):
        if isinstance(value, np.ndarray):
            if value.shape == (1,):
                value = value[0]
            else:
                raise ValueError("The optimal length must be a float.")
        self._pennation_angle = value

    @property
    def maximal_velocity(self) -> float:
        return self._maximal_velocity

    @maximal_velocity.setter
    def maximal_velocity(self, value: float):
        if value is not None and value <= 0:
            raise ValueError("The maximal contraction velocity of the muscle must be greater than 0.")
        if isinstance(value, np.ndarray):
            if value.shape == (1,):
                value = value[0]
            else:
                raise ValueError("The maximal velocity must be a float.")
        self._maximal_velocity = value

    @property
    def maximal_excitation(self) -> float:
        return self._maximal_excitation

    @maximal_excitation.setter
    def maximal_excitation(self, value: float):
        if value is not None and value <= 0:
            raise ValueError("The maximal excitation of the muscle must be greater than 0.")
        if isinstance(value, np.ndarray):
            if value.shape == (1,):
                value = value[0]
            else:
                raise ValueError("The maximal excitation must be a float.")
        self._maximal_excitation = value

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"muscle\t{self.name}\n"
        out_string += f"\ttype\t{self.muscle_type.value}\n"
        out_string += f"\tstatetype\t{self.state_type.value}\n"
        out_string += f"\tmusclegroup\t{self.muscle_group}\n"
        out_string += f"\toriginposition\t{np.round(self.origin_position.position[0, 0], 4)}\t{np.round(self.origin_position.position[1, 0], 4)}\t{np.round(self.origin_position.position[2, 0], 4)}\n"
        out_string += f"\tinsertionposition\t{np.round(self.insertion_position.position[0, 0], 4)}\t{np.round(self.insertion_position.position[1, 0], 4)}\t{np.round(self.insertion_position.position[2, 0], 4)}\n"
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
        out_string += "\n\n"

        out_string += "\n // ------ VIA POINTS ------\n"
        for via_point in self.via_points:
            out_string += via_point.to_biomod()

        return out_string

    def to_osim(self):
        """Generate OpenSim XML representation of the muscle"""

        # TODO: handle different muscle types than DeGrooteFregly2016Muscle
        muscle_elem = etree.Element("DeGrooteFregly2016Muscle", name=self.name)

        max_iso_force = etree.SubElement(muscle_elem, "max_isometric_force")
        max_iso_force.text = f"{self.maximal_force:.8f}" if self.maximal_force else "1000.0"

        opt_fiber_length = etree.SubElement(muscle_elem, "optimal_fiber_length")
        opt_fiber_length.text = f"{self.optimal_length:.8f}" if self.optimal_length else "0.1"

        tendon_slack = etree.SubElement(muscle_elem, "tendon_slack_length")
        tendon_slack.text = f"{self.tendon_slack_length:.8f}" if self.tendon_slack_length else "0.2"

        pennation = etree.SubElement(muscle_elem, "pennation_angle_at_optimal")
        pennation.text = f"{self.pennation_angle:.8f}" if self.pennation_angle else "0"

        max_velocity = etree.SubElement(muscle_elem, "max_contraction_velocity")
        max_velocity.text = f"{self.maximal_velocity:.8f}" if self.maximal_velocity else "10"

        # Geometry path
        geometry_path = etree.SubElement(muscle_elem, "GeometryPath", name="path")
        path_point_set = etree.SubElement(geometry_path, "PathPointSet")
        path_objects = etree.SubElement(path_point_set, "objects")

        # Origin
        origin_elem = self.origin_position.to_osim()
        if origin_elem is not None:
            origin_elem.set("name", f"{self.name}_origin")
            path_objects.append(origin_elem)

        # Via points
        for via_point in self.via_points:
            via_elem = via_point.to_osim()
            if via_elem is not None:
                path_objects.append(via_elem)

        # Insertion
        insertion_elem = self.insertion_position.to_osim()
        if insertion_elem is not None:
            insertion_elem.set("name", f"{self.name}_insertion")
            path_objects.append(insertion_elem)

        return muscle_elem
