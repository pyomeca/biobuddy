import numpy as np
from lxml import etree

from ..utils_xml import find_in_tree, find_sub_elements_in_tree
from .path_point import PathPoint, PathPointMovement, PathPointCondition
from ...components.real.muscle.muscle_real import MuscleReal
from ...components.real.muscle.via_point_real import ViaPointReal
from ...components.real.muscle.muscle_group_real import MuscleGroupReal
from ...components.muscle_utils import MuscleType, MuscleStateType


OPENSIM_MUSCLE_TYPE = {
    "Thelen2003Muscle": MuscleType.HILL_THELEN,
    "Millard2012EquilibriumMuscle": None,  # Not implemented
    "Schutte1993Muscle": None,  # Not implemented
    "DeGrooteFregly2016Muscle": MuscleType.HILL_DE_GROOTE,
    "Schutte1993Muscle_Deprecated": None,  # Not implemented
}


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


def check_for_unsupported_elements(element: etree.ElementTree, name: str) -> str:
    warnings = ""
    not_implemented_elements = [
        "FmaxTendonStrain",
        "FmaxMuscleStrain",
        "KshapeActive",
        "KshapePassive",
        "Af",
        "Flen",
        "activation_time_constant",
        "deactivation_time_constant",
    ]
    for elt_name in not_implemented_elements:
        if find_in_tree(element, elt_name):
            warnings += f"\nAn element {elt_name} was found in the muscle {name}, but this feature is not implemented yet so it will be ignored.\n"
    return warnings


def is_applied(element: etree.ElementTree, ignore_applied: bool) -> bool:
    applied = True
    if element.find("appliesForce") is not None and not ignore_applied:
        applied = element.find("appliesForce").text == "true"
    return applied


def get_muscle_from_element(
    element: etree.ElementTree, ignore_applied: bool, muscle_type: MuscleType = MuscleType.HILL_DE_GROOTE
) -> tuple[MuscleGroupReal, MuscleReal, str]:
    """
    TODO: Better handle ignore_applied parameter. MuscleReal should have a applied parameter, a remove_unapplied_muscle method, and we should remove unapplied muscles in to_biomod.
    """
    name = (element.attrib["name"]).split("/")[-1]
    warnings = check_for_wrappings(element, name) + check_for_unsupported_elements(element, name)
    # muscle_type = OPENSIM_MUSCLE_TYPE[element.tag]  # TODO: We should try to match OpenSim muscle types
    muscle_type = muscle_type

    maximal_force = find_in_tree(element, "max_isometric_force")
    maximal_force = float(maximal_force) if maximal_force else 1000.0

    optimal_length = find_in_tree(element, "optimal_fiber_length")
    optimal_length = float(optimal_length) if optimal_length else 0.1

    tendon_slack_length = find_in_tree(element, "tendon_slack_length")
    tendon_slack_length = float(tendon_slack_length) if tendon_slack_length else None

    pennation_angle = find_in_tree(element, "pennation_angle_at_optimal")
    pennation_angle = float(pennation_angle) if pennation_angle else 0.0

    maximal_velocity = find_in_tree(element, "max_contraction_velocity")
    maximal_velocity = float(maximal_velocity) if maximal_velocity else 10.0

    origin_or_insertion_problem = False
    path_points: list[PathPoint] = []
    via_points: list[PathPoint] = []
    path_point_elts = find_sub_elements_in_tree(
        element=element,
        parent_element_name=["GeometryPath", "PathPointSet", "objects"],
        sub_element_names=["PathPoint", "ConditionalPathPoint", "MovingPathPoint"],
    )
    for i_path_point, path_point_elt in enumerate(path_point_elts):
        via_point = PathPoint.from_element(path_point_elt)
        via_point.muscle = name

        # Condition
        condition = None
        warning = ""
        if path_point_elt.tag == "ConditionalPathPoint":
            condition, warning = PathPointCondition.from_element(path_point_elt)
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
        if warning != "":
            warnings += warning
            if i_path_point == 0 or i_path_point == len(path_point_elts) - 1:
                # If there is a problem with the origin or insertion of a muscle, it is better to skip this muscle al together
                return None, None, warnings
        else:
            via_point.movement = movement

        via_points.append(via_point)
        path_points.append(via_point)

    muscle_group_name = f"{path_points[0].body}_to_{path_points[-1].body}"
    try:
        muscle_group = MuscleGroupReal(
            name=muscle_group_name,
            origin_parent_name=path_points[0].body,
            insertion_parent_name=path_points[-1].body,
        )
    except Exception as e:
        # This error is raised when the origin and insertion parent names are the same which is accepted in OpenSim.
        warnings += (
            f"\nAn error occurred while creating the muscle group {muscle_group_name} for the muscle {name}: {e}\n"
        )
        return None, None, warnings

    for via_point in via_points:
        via_point.muscle_group = muscle_group_name

    if not is_applied(element, ignore_applied):
        return muscle_group, None, ""
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

        muscle = MuscleReal(
            name=name,
            muscle_type=muscle_type,
            state_type=MuscleStateType.DEGROOTE,  # TODO: make this configurable
            muscle_group=muscle_group_name,
            origin_position=origin_position,
            insertion_position=insertion_position,
            optimal_length=optimal_length,
            maximal_force=maximal_force,
            tendon_slack_length=tendon_slack_length,
            pennation_angle=pennation_angle,
            maximal_velocity=maximal_velocity,
            maximal_excitation=1.0,  # Default value since OpenSim does not handle maximal excitation?
        )

        for via_point in via_points[1:-1]:

            if via_point.movement is not None:
                position = None
            else:
                position = np.array([float(v) for v in via_point.position.split()])
            muscle.add_via_point(
                ViaPointReal(
                    name=via_point.name,
                    parent_name=via_point.body,
                    muscle_name=name,
                    muscle_group=muscle_group_name,
                    position=position,
                    condition=via_point.condition,
                    movement=via_point.movement,
                )
            )

        return muscle_group, muscle, warnings
