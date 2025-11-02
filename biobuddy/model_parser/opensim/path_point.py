from lxml import etree

from ..utils_xml import find_in_tree, find_sub_elements_in_tree, match_tag
from .functions import spline_from_element, piece_wise_linear_from_element


class PathPointCondition:
    """
    Very similar to the BioBuddy version of this class, but it needs to be redeclared because of the joint_name logic.
    """

    def __init__(
        self,
        dof_name: str,
        joint_name: str,
        range_min: float,
        range_max: float,
    ):
        self.dof_name = dof_name
        self.joint_name = joint_name
        self.range_min = float(range_min)
        self.range_max = float(range_max)

    @staticmethod
    def from_element(element: etree.ElementTree) -> tuple["Self", str]:
        if len(find_in_tree(element, "socket_coordinate").split("/")) > 1:
            joint_name = find_in_tree(element, "socket_coordinate").split("/")[-2]
        else:
            joint_name = ""
        return (
            PathPointCondition(
                dof_name=find_in_tree(element, "socket_coordinate").split("/")[-1],
                joint_name=joint_name,
                range_min=find_in_tree(element, "range").split(" ")[0],
                range_max=find_in_tree(element, "range").split(" ")[1],
            ),
            "",
        )


class PathPointMovement:
    """
    Very similar to the BioBuddy version of this class, but it needs to be redeclared because of the joint_name logic.
    """

    def __init__(
        self,
        dof_names: list[str],
        joint_names: list[str],
        locations: list["Functions"],
    ):
        self.dof_names = dof_names
        self.joint_names = joint_names
        self.locations = locations

    @staticmethod
    def from_element(element: etree.ElementTree) -> tuple["Self", str]:
        warning = ""
        coordinate_elts = find_sub_elements_in_tree(
            element=element,
            parent_element_name=[],
            sub_element_names=["socket_x_coordinate", "socket_y_coordinate", "socket_z_coordinate"],
        )
        location_elts = find_sub_elements_in_tree(
            element=element, parent_element_name=[], sub_element_names=["x_location", "y_location", "z_location"]
        )
        dof_names = []
        locations = []
        joint_names = []
        moving_path_point = None
        for coord, loc in zip(coordinate_elts, location_elts):
            if match_tag(loc[0], "SimmSpline"):
                locations.append(spline_from_element(loc[0]))
                dof_names.append(coord.text.split("/")[-1])
                if len(coord.text.split("/")) > 1:
                    joint_names.append(coord.text.split("/")[-2])
                else:
                    joint_names.append("")
            elif match_tag(loc[0], "PiecewiseLinearFunction"):
                locations.append(piece_wise_linear_from_element(loc[0]))
                dof_names.append(coord.text.split("/")[-1])
                if len(coord.text.split("/")) > 1:
                    joint_names.append(coord.text.split("/")[-2])
                else:
                    joint_names.append("")
            else:
                warning += "Only SimmSpline and PiecewiseLinearFunction functions are supported for PathPointMovement locations."
        if warning == "":
            moving_path_point = PathPointMovement(dof_names=dof_names, locations=locations, joint_names=joint_names)
        return moving_path_point, warning


class PathPoint:
    def __init__(
        self,
        name: str,
        muscle: str,
        body: str,
        muscle_group: str,
        position: list,
        condition: PathPointCondition | None = None,
        movement: PathPointMovement | None = None,
    ):
        self.name = name
        self.muscle = muscle
        self.body = body
        self.muscle_group = muscle_group
        self.position = position
        self.condition = condition
        self.movement = movement

    @staticmethod
    def from_element(element: etree.ElementTree) -> "Self":
        return PathPoint(
            name=element.attrib["name"],
            muscle=None,  # is set in muscle.py
            body=find_in_tree(element, "socket_parent_frame").split("/")[-1],
            muscle_group=None,  # is set in muscle.py
            position=find_in_tree(element, "location"),
            condition=None,  # is set in muscle.py
            movement=None,  # is set in muscle.py
        )
