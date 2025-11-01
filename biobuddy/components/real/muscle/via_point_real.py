import numpy as np
from lxml import etree

from ....utils.aliases import Points, points_to_array
from ....utils.checks import check_name
from ...via_point_utils import PathPointCondition, PathPointMovement


class ViaPointReal:
    def __init__(
        self,
        name: str,
        parent_name: str,
        muscle_name: str = None,
        muscle_group: str = None,
        position: Points = None,
        condition: PathPointCondition | None = None,
        movement: PathPointMovement | None = None,
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
        condition
            The condition that must be fulfilled for the via point to be active
        movement
            The movement that defines how the via point moves in the local reference frame
        """
        if position is not None and movement is not None:
            raise RuntimeError("You can only have either a position or a movement, not both.")
        if movement is not None and condition is not None:
            raise RuntimeError("You can only have either a condition or a movement, not both.")

        self.name = name
        self.parent_name = check_name(parent_name)
        self.muscle_name = muscle_name
        self.muscle_group = muscle_group
        self.position = position
        self.condition = condition
        self.movement = movement

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

    @property
    def condition(self) -> PathPointCondition:
        return self._condition

    @condition.setter
    def condition(self, value: PathPointCondition) -> None:
        self._condition = value

    @property
    def movement(self) -> PathPointMovement:
        return self._movement

    @movement.setter
    def movement(self, value: PathPointMovement) -> None:
        self._movement = value

    def to_biomod(self):
        """Define the print function, so it automatically formats things in the file properly."""
        if self.condition is not None:
            # To avoid this warning, it is possible to fix the via points using the BiomechanicalModelReal.fix_via_points(q)
            return f"\n// WARNING: biorbd doe not support conditional via points, so the via point {self.name} was ignored.\n"
        if self.movement is not None:
            # To avoid this warning, it is possible to fix the via points position using the BiomechanicalModelReal.fix_via_points(q)
            return (
                f"\n// WARNING: biorbd doe not support moving via points, so the via point {self.name} was ignored.\n"
            )
        out_string = f"viapoint\t{self.name}\n"
        out_string += f"\tparent\t{self.parent_name}\n"
        out_string += f"\tmuscle\t{self.muscle_name}\n"
        out_string += f"\tmusclegroup\t{self.muscle_group}\n"
        out_string += f"\tposition\t{np.round(self.position[0, 0], 6)}\t{np.round(self.position[1, 0], 6)}\t{np.round(self.position[2, 0], 6)}\n"
        out_string += "endviapoint\n"
        out_string += "\n\n"
        return out_string

    def to_osim(self):
        """Generate OpenSim XML representation of the via point (PathPoint element)"""
        if self.condition is not None or self.movement is not None:
            raise NotImplementedError(
                "Conditional and moving via points are not implemented yet. If you need this, please open an issue on GitHub."
            )

        path_point_elem = etree.Element("PathPoint", name=self.name)

        socket_parent = etree.SubElement(path_point_elem, "socket_parent_frame")
        socket_parent.text = f"bodyset/{self.parent_name}"

        location = etree.SubElement(path_point_elem, "location")
        p = self.position
        location.text = f"{p[0,0]:.8f} {p[1,0]:.8f} {p[2,0]:.8f}"

        if self.condition is not None:
            # To avoid this warning, it is possible to fix the via points using the BiomechanicalModelReal.fix_via_points(q). Otherwise, please open an issue on GitHub.
            raise NotImplementedError(
                f"Writing models with conditional muscle via points (muscle: {self.muscle_name}) to OpenSim format is not yet implemented."
            )
        if self.movement is not None:
            # To avoid this warning, it is possible to fix the via points position using the BiomechanicalModelReal.fix_via_points(q). Otherwise, please open an issue on GitHub.
            raise NotImplementedError(
                f"Writing models with conditional muscle via points (muscle: {self.muscle_name}) to OpenSim format is not yet implemented."
            )

        return path_point_elem
