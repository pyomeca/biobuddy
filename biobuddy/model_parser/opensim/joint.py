from typing import Self
from enum import Enum

from xml.etree import ElementTree
import numpy as np

from .coordinate import Coordinate
from .utils import find_in_tree
from .spatial_transform import SpatialTransform
from ...utils.linear_algebra import compute_matrix_rotation, rot2eul


class JointType(Enum):
    WELD_JOINT = "WeldJoint"
    CUSTOM_JOINT = "CustomJoint"


def _convert_offset_child(offset_child_rot, offset_child_trans):
    R = compute_matrix_rotation(offset_child_rot).T
    new_translation = -np.dot(R.T, offset_child_trans)
    new_rotation = -rot2eul(R)
    new_rotation_str = ""
    new_translation_str = ""
    for i in range(3):
        if i == 0:
            pass
        else:
            new_rotation_str += " "
            new_translation_str += " "
        new_rotation_str += str(new_rotation[i])
        new_translation_str += str(new_translation[i])
    return new_translation, new_rotation


class Joint:
    def __init__(
        self,
        parent: str,
        child: str,
        name: str,
        type: str,
        coordinates: list,
        parent_offset_trans: list,
        parent_offset_rot: list,
        child_offset_trans: list,
        child_offset_rot: list,
        child_body: str,
        parent_body: str,
        spatial_transform: list,
        implemented_joint: list,
        function,
    ):
        self.parent = parent
        self.child = child
        self.name = name
        self.type = type
        self.coordinates = coordinates
        self.parent_offset_trans = parent_offset_trans
        self.parent_offset_rot = parent_offset_rot
        self.child_offset_trans = child_offset_trans
        self.child_offset_rot = child_offset_rot
        self.child_body = child_body
        self.parent_body = parent_body
        self.spatial_transform = spatial_transform
        self.implemented_joint = implemented_joint
        self.function = function

    @staticmethod
    def from_element(element: ElementTree.Element, ignore_fixed: bool, ignore_clamped: bool) -> Self:
        tag = element.tag
        if tag not in [e.value for e in JointType]:
            raise RuntimeError(
                f"Joint type {tag} is not implemented yet." f"Allowed joint type are: {[e.value for e in JointType]}"
            )
        name = (element.attrib["name"]).split("/")[-1]
        parent = find_in_tree(element, "socket_parent_frame").split("/")[-1]
        child = find_in_tree(element, "socket_child_frame").split("/")[-1]

        coordinates = []
        spatial_transform = []
        function = False
        if element.find("coordinates") is not None:
            for coordinate in element.find("coordinates").findall("Coordinate"):
                coordinates.append(Coordinate.from_element(coordinate, ignore_fixed, ignore_clamped))

        if element.find("SpatialTransform") is not None:
            for i, transform in enumerate(element.find("SpatialTransform").findall("TransformAxis")):
                spat_transform = SpatialTransform.from_element(transform)
                if i < 3:
                    spat_transform.type = "rotation"
                else:
                    spat_transform.type = "translation"
                for coordinate in coordinates:
                    if coordinate.name == spat_transform.coordinate_name:
                        spat_transform.coordinate = coordinate
                function = spat_transform.function
                spatial_transform.append(spat_transform)

        for frame in element.find("frames").findall("PhysicalOffsetFrame"):
            if parent == frame.attrib["name"]:
                parent_body = frame.find("socket_parent").text.split("/")[-1]
                offset_rot = frame.find("orientation").text
                offset_trans = frame.find("translation").text
                parent_offset_rot = [float(i) for i in offset_rot.split(" ")]
                parent_offset_trans = [float(i) for i in offset_trans.split(" ")]
            elif child == frame.attrib["name"]:
                child_body = frame.find("socket_parent").text.split("/")[-1]
                offset_rot = frame.find("orientation").text
                offset_trans = frame.find("translation").text
                offset_trans = [float(i) for i in offset_trans.split(" ")]
                offset_rot = [-float(i) for i in offset_rot.split(" ")]
                child_offset_trans, child_offset_rot = _convert_offset_child(offset_rot, offset_trans)

        return Joint(
            parent=parent,
            child=child,
            name=name,
            type=tag,
            coordinates=coordinates,
            parent_offset_trans=parent_offset_trans,
            parent_offset_rot=parent_offset_rot,
            child_offset_trans=child_offset_trans,
            child_offset_rot=child_offset_rot,
            child_body=child_body,
            parent_body=parent_body,
            spatial_transform=spatial_transform,
            implemented_joint=[""],
            function=function,
        )
