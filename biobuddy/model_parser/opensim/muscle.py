# from typing import Self

from lxml import etree

from .utils import find_in_tree
from .path_point import PathPoint


class Muscle:
    def __init__(
        self,
        name: str,
        via_point: list,
        origin: list,
        insersion: list,
        optimal_length: float,
        maximal_force: float,
        tendon_slack_length: float,
        pennation_angle: float,
        applied: bool,
        maximal_velocity: float,
        wrap: bool,
        group: list,
    ):
        self.name = name
        self.via_point = via_point
        self.origin = origin
        self.insersion = insersion
        self.optimal_length = optimal_length
        self.maximal_force = maximal_force
        self.tendon_slack_length = tendon_slack_length
        self.pennation_angle = pennation_angle
        self.applied = applied
        self.maximal_velocity = maximal_velocity
        self.wrap = wrap
        self.group = group

    @staticmethod
    def from_element(element: etree.ElementTree, ignore_applied: bool) -> "Self":
        name = (element.attrib["name"]).split("/")[-1]
        maximal_force = find_in_tree(element, "max_isometric_force")
        optimal_length = find_in_tree(element, "optimal_fiber_length")
        tendon_slack_length = find_in_tree(element, "tendon_slack_length")
        pennation_angle = find_in_tree(element, "pennation_angle_at_optimal")
        maximal_velocity = find_in_tree(element, "max_contraction_velocity")

        applied = False
        if element.find("appliesForce") is not None and not ignore_applied:
            applied = element.find("appliesForce").text == "true"

        # TODO: add type hints to general lists
        via_points = []
        for path_point_elt in element.find("GeometryPath").find("PathPointSet")[0].findall("PathPoint"):
            via_point = PathPoint.from_element(path_point_elt)
            via_point.muscle = name
            via_points.append(via_point)
        group = [via_points[0].body, via_points[-1].body]
        for i in range(len(via_points)):
            via_points[i].muscle_group = f"{group[0]}_to_{group[1]}"

        wrap = False
        if element.find("GeometryPath").find("PathWrapSet") is not None:
            try:
                wrap_tp = element.find("GeometryPath").find("PathWrapSet")[0].text
            except:
                wrap_tp = 0
            n_wrap = 0 if not wrap_tp else len(wrap_tp)
            wrap = n_wrap != 0

        insersion = via_points[-1].position
        origin = via_points[0].position
        via_point = via_points[1:-1]

        return Muscle(
            name=name,
            via_point=via_point,
            origin=origin,
            insersion=insersion,
            optimal_length=optimal_length,
            maximal_force=maximal_force,
            tendon_slack_length=tendon_slack_length,
            pennation_angle=pennation_angle,
            applied=applied,
            maximal_velocity=maximal_velocity,
            wrap=wrap,
            group=group,
        )
