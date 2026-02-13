import numpy as np
from lxml import etree

from .functions import spline_from_element
from .path_point import PathPoint, PathPointMovement, PathPointCondition
from ..utils_xml import find_in_tree, find_sub_elements_in_tree, find_sub_element_in_tree
from ...components.ligament_utils import LigamentType
from ...components.real.force.ligament_real import LigamentReal
from ...components.real.force.via_point_real import ViaPointReal

"""
I have only seen OpenSim ligaments with a force-length function.
I will store this function for now, but I do not have any way to enforce it for now.
"""


def check_for_wrappings(element: etree.ElementTree, name: str) -> str:
    wrap = False
    warnings = ""
    if element.find("GeometryPath").find("PathWrapSet") is not None:
        try:
            wrap_tp = element.find("GeometryPath").find("PathWrapSet")[0].text
        except:
            wrap_tp = 0
        n_wrap = 0 if not wrap_tp else len(wrap_tp)
        wrap = n_wrap != 0
    if wrap:
        warnings += f"Some wrapping objects were present on the muscle {name} in the original file force set.\nWraping objects are not supported yet so they will be ignored."
    return warnings

def is_applied(element: etree.ElementTree, ignore_applied: bool) -> bool:
    applied = True
    if element.find("appliesForce") is not None and not ignore_applied:
        applied = element.find("appliesForce").text == "true"
    return applied


def get_ligament_from_element(
        element: etree.ElementTree,
        ignore_applied: bool,
) -> tuple[LigamentReal, str]:
    """
    TODO: Better handle ignore_applied parameter. LigamentReal should have a applied parameter, a remove_unapplied_ligament method, and we should remove unapplied ligament in to_biomod.
    """
    name = (element.attrib["name"]).split("/")[-1]
    warnings = check_for_wrappings(element, name)

    pcsa = find_in_tree(element, "pcsa_force")
    pcsa = float(pcsa) if pcsa else 1000.0

    ligament_slack_length = find_in_tree(element, "resting_length")
    if ligament_slack_length is None:
        warnings += f"The ligament {name} does not have a resting_length element. Please add this element in the original file force set."
    ligament_slack_length = float(ligament_slack_length) if ligament_slack_length else 0.2

    force_length_element = find_sub_element_in_tree(element, ["SimmSpline", "PiecewiseLinearFunction"])
    if force_length_element is not None:
        if len(force_length_element) > 1:
            warnings += f"The ligament {name} has {len(force_length_element)} force-length function, but only 1 is supported for now. Please report this issue to the developers."
            return None, warnings
        force_length_function = spline_from_element(force_length_element[0])

    path_points: list[PathPoint] = []
    via_points: list[PathPoint] = []
    path_point_elts = find_sub_elements_in_tree(
        element=element,
        parent_element_name=["GeometryPath", "PathPointSet", "objects"],
        sub_element_names=["PathPoint", "ConditionalPathPoint", "MovingPathPoint"],
    )

    if len(path_point_elts) != 2:
        raise RuntimeError(f"The ligament {name} has {len(path_point_elts)} path points, but only 2 are supported for now. "
                           f"Please report this issue to the developers.")

    for i_path_point, path_point_elt in enumerate(path_point_elts):
        via_point = PathPoint.from_element(path_point_elt)
        via_point.ligament = name

        # Condition
        condition = None
        warning = ""
        if path_point_elt.tag == "ConditionalPathPoint":
            condition, warning = PathPointCondition.from_element(path_point_elt)
            warning += f"\nThe ligament {name} has a conditional path point at {via_point.body} which is not supported for now. "
        if warning != "":
            warnings += warning
            if i_path_point == 0 or i_path_point == len(path_point_elts) - 1:
                # If there is a problem with the origin or insertion of a muscle, it is better to skip this muscle al together
                return None, None, warnings
        else:
            via_point.condition = condition

        # Movement
        movement = None
        warning = ""
        if path_point_elt.tag == "MovingPathPoint":
            movement, warning = PathPointMovement.from_element(path_point_elt)
            warning += f"\nThe ligament {name} has a moving path point at {via_point.body} which is not supported for now. "
        if warning != "":
            warnings += warning
            if i_path_point == 0 or i_path_point == len(path_point_elts) - 1:
                # If there is a problem with the origin or insertion of a muscle, it is better to skip this muscle al together
                return None, None, warnings
        else:
            via_point.movement = movement

        via_points.append(via_point)
        path_points.append(via_point)

    if not is_applied(element, ignore_applied):
        return None, ""
    else:

        if isinstance(path_points[0].movement, PathPointMovement):
            origin_pos = None
        else:
            origin_pos = np.array([float(v) for v in via_points[0].position.split()])

        if isinstance(path_points[-1].movement, PathPointMovement):
            insertion_pos = None
        else:
            insertion_pos = np.array([float(v) for v in via_points[-1].position.split()])

        origin_position = ViaPointReal(
            name=f"origin_{name}",
            parent_name=via_points[0].body,
            position=origin_pos,
            condition=via_points[0].condition,
            movement=via_points[0].movement,
        )
        insertion_position = ViaPointReal(
            name=f"insertion_{name}",
            parent_name=via_points[-1].body,
            position=insertion_pos,
            condition=via_points[-1].condition,
            movement=via_points[-1].movement,
        )

        ligament = LigamentReal(
            name=name,
            ligament_type=LigamentType.FUNCTION,  # I have only seen OpenSim ligaments with force-length function
            origin_position=origin_position,
            insertion_position=insertion_position,
            ligament_slack_length=ligament_slack_length,
            pcsa=pcsa,
            force_length_function=force_length_function,
        )

        return ligament, warnings
