from enum import Enum
from lxml import etree
import numpy as np

from .coordinate import Coordinate
from ..utils_xml import find_in_tree
from .spatial_transform import SpatialTransform
from ...utils.linear_algebra import compute_matrix_rotation, rot2eul


class JointType(Enum):
    WELD_JOINT = "WeldJoint"
    CUSTOM_JOINT = "CustomJoint"
    GROUND = "Ground"  # This is not an actual joint, but it creates an empty Joint object


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
    def from_element(element: etree.ElementTree, ignore_fixed: bool, ignore_clamped: bool) -> "Self":
        tag = element.tag
        if tag not in [e.value for e in JointType]:
            joint_types_str = ""
            for e in JointType:
                joint_types_str += e.value + " "
            raise RuntimeError(
                f"Joint type {tag} is not implemented yet. " f"Allowed joint type are: {joint_types_str}"
            )

        name = (element.attrib["name"]).split("/")[-1]
        parent_name = find_in_tree(element, "socket_parent_frame").split("/")[-1]
        child_name = find_in_tree(element, "socket_child_frame").split("/")[-1]

        coordinates = []
        spatial_transform = []
        function = False
        if element.find("coordinates") is not None:
            for coordinate in element.find("coordinates").findall("Coordinate"):
                coordinates.append(Coordinate.from_element(coordinate, parent_name, ignore_fixed, ignore_clamped))

        if element.find("SpatialTransform") is not None:
            for i, transform in enumerate(element.find("SpatialTransform").findall("TransformAxis")):
                spat_transform = SpatialTransform.from_element(transform)
                if i < 3:
                    spat_transform.type = "rotation"
                else:
                    spat_transform.type = "translation"
                for coordinate in coordinates:
                    if coordinate.name == spat_transform.coordinate_name:
                        # We extract the dof name and add the parent name as a prefix to ensure uniqueness
                        dof_name = f"{parent_name}_{coordinate.name}"
                        coordinate.name = dof_name
                        spat_transform.coordinate = coordinate
                function = spat_transform.function
                spatial_transform.append(spat_transform)

        parent_offset_trans = ["0", "0", "0"]
        parent_offset_rot = ["0", "0", "0"]
        child_offset_trans = ["0", "0", "0"]
        child_offset_rot = ["0", "0", "0"]
        child_body = (None,)
        parent_body = (None,)
        for frame in element.find("frames").findall("PhysicalOffsetFrame"):
            if parent_name == frame.attrib["name"]:
                parent_body = frame.find("socket_parent").text.split("/")[-1]
                offset_rot = frame.find("orientation").text
                offset_trans = frame.find("translation").text
                parent_offset_rot = [float(i) for i in offset_rot.split(" ")]
                parent_offset_trans = [float(i) for i in offset_trans.split(" ")]
            if child_name == frame.attrib["name"]:
                child_body = frame.find("socket_parent").text.split("/")[-1]
                offset_rot = frame.find("orientation").text
                offset_trans = frame.find("translation").text
                offset_trans = [float(i) for i in offset_trans.split(" ")]
                offset_rot = [-float(i) for i in offset_rot.split(" ")]
                child_offset_trans, child_offset_rot = _convert_offset_child(offset_rot, offset_trans)

        return Joint(
            parent=parent_name,
            child=child_name,
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
