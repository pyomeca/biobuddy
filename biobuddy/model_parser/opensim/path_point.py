# from typing import Self

from lxml import etree

from .utils import find_in_tree, find_sub_elements_in_tree, match_tag
from .functions import spline_from_element
from ...components.via_point_utils import PathPointCondition, PathPointMovement


def condition_from_element(element: etree.ElementTree) -> PathPointCondition:
    return PathPointCondition(
        dof_name=find_in_tree(element, "socket_coordinate").split("/")[-1],
        range_min=find_in_tree(element, "range").split(" ")[0],
        range_max=find_in_tree(element, "range").split(" ")[1],
    )


def movement_from_element(element: etree.ElementTree) -> tuple[PathPointMovement, str]:
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
    moving_path_point = None
    if not find_in_tree(element, "SimmSpline"):
        warning += "Warning: No SimmSpline found in PathPointMovement locations. It will be ignored\n"
    for coord, loc in zip(coordinate_elts, location_elts):
        if match_tag(loc[0], "SimmSpline"):
            dof_names.append(coord.text.split("/")[-1])
            locations.append(spline_from_element(loc[0]))
        else:
            warning += "Only SimmSpline functions are supported for PathPointMovement locations."
    if warning == "":
        moving_path_point = PathPointMovement(
            dof_names=dof_names,
            locations=locations,
        )
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
        self.condition = (condition,)
        self.movement = (movement,)

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
