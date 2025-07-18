"""
Note to the developers: This OpenSim parser is in an intermediary state. We have implemented temporary classes
(like body.py) to handle the features that are present in OpenSim but that are not implemented yet in BioBuddy.
There are other casses (like muscle.py) where all the features are implemented so we can directly use the BioBuddy classes.
The long term goal it to switch all the temporary classes to BioBuddy classes, but this will take time.
"""

from enum import Enum
from time import strftime

import numpy as np
from lxml import etree

from .utils import is_element_empty, match_tag
from .body import Body
from .joint import Joint
from .marker import Marker
from .muscle import get_muscle_from_element
from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.generic.muscle.muscle_group import MuscleGroup
from ...components.generic.rigidbody.range_of_motion import RangeOfMotion, Ranges
from ...components.real.muscle.muscle_real import MuscleReal, MuscleType, MuscleStateType
from ...components.real.rigidbody.segment_real import SegmentReal
from ...components.real.muscle.via_point_real import ViaPointReal
from ...components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ...components.real.rigidbody.marker_real import MarkerReal
from ...components.real.rigidbody.mesh_file_real import MeshFileReal
from ...components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ...utils.linear_algebra import (
    OrthoMatrix,
    compute_matrix_rotation,
    is_ortho_basis,
    ortho_norm_basis,
    get_closest_rt_matrix,
)
from ...utils.rotations import Rotations
from ...utils.translations import Translations


class ForceType(Enum):
    MUSCLE = "Muscle"


class Controller(Enum):
    NONE = None


def _get_file_version(model: etree.ElementTree) -> int:
    return int(model.getroot().attrib["Version"])


class OsimModelParser:
    def __init__(
        self,
        filepath: str,
        muscle_type: MuscleType,
        muscle_state_type: MuscleStateType,
        mesh_dir: str = None,
        print_warnings: bool = True,
        ignore_fixed_dof_tag: bool = False,
        ignore_clamped_dof_tag: bool = False,
        ignore_muscle_applied_tag: bool = False,
    ):
        """
        Reads and converts OpenSim model files (.osim) to a biomechanical model representation.

        Parameters
        ----------
        filepath : str
            Path to the OpenSim .osim file to read
        muscle_type: MuscleType
            The type of muscle to assume when interpreting the osim model
        muscle_state_type : MuscleStateType
            The muscle state type to assume when interpreting the osim model
        print_warnings : bool, optional
            Whether to print conversion warnings, default True
        mesh_dir : str, optional
            Directory containing mesh files

        Raises
        ------
        RuntimeError
            If file version is too old or units are not meters/newtons
        """
        # Initial attributes
        self.filepath = filepath
        self.muscle_type = muscle_type
        self.muscle_state_type = muscle_state_type
        self.mesh_dir = mesh_dir
        self.print_warnings = print_warnings
        self.ignore_fixed_dof_tag = ignore_fixed_dof_tag
        self.ignore_clamped_dof_tag = ignore_clamped_dof_tag
        self.ignore_muscle_applied_tag = ignore_muscle_applied_tag

        # Extended attributes
        self.model = etree.parse(filepath)
        file_version = _get_file_version(self.model)
        if file_version < 40000:
            raise RuntimeError(
                f".osim file version must be superior or equal to '40000' and you have: {file_version}."
                "To convert the osim file to the newest version please open and save your file in"
                "Opensim 4.0 or later."
            )

        self.gravity = np.array([0.0, 0.0, -9.81])
        self.ground_elt, self.default_elt, self.credit, self.publications = None, None, None, None
        self.bodyset_elt, self.jointset_elt, self.forceset_elt, self.markerset_elt = None, None, None, None
        self.controllerset_elt, self.constraintset_elt, self.contact_geometryset_elt = None, None, None
        self.componentset_elt, self.probeset_elt = None, None
        self.length_units, self.force_units = "meters", "newtons"

        self.parse_tags(self.model.getroot())

        self.bodies: list[Body] = []
        self.muscle_groups: list[MuscleGroup] = []
        self.muscles: list[MuscleReal] = []
        self.via_points: list[ViaPointReal] = []
        self.joints: list[Joint] = []
        self.markers: list[Marker] = []
        self.constraint_set = []  # Not implemented
        self.controller_set = []  # Not implemented
        self.prob_set = []  # Not implemented
        self.component_set = []  # Not implemented
        self.geometry_set: list[list[str]] = []

        self.header = ""
        self.warnings: list[str] = []

        # Create the biomechanical model
        self.biomechanical_model_real = BiomechanicalModelReal()
        self._read()

    def parse_tags(self, root):
        for element in root:
            if isinstance(element, etree._Comment):
                pass  # This line is a comment
            elif match_tag(element, "gravity"):
                self.gravity = np.array([float(i) for i in element.text.split(" ")])
            elif match_tag(element, "ground"):
                self.ground_elt = element
            elif match_tag(element, "defaults"):
                self.default_elt = element
            elif match_tag(element, "BodySet"):
                self.bodyset_elt = element
            elif match_tag(element, "JointSet"):
                self.jointset_elt = element
            elif match_tag(element, "ControllerSet"):
                self.controllerset_elt = element
            elif match_tag(element, "ConstraintSet"):
                self.constraintset_elt = element
            elif match_tag(element, "ForceSet"):
                self.forceset_elt = element
            elif match_tag(element, "MarkerSet"):
                self.markerset_elt = element
            elif match_tag(element, "ContactGeometrySet"):
                self.contact_geometryset_elt = element
            elif match_tag(element, "ComponentSet"):
                self.componentset_elt = element
            elif match_tag(element, "ProbeSet"):
                self.probeset_elt = element
            elif match_tag(element, "credits"):
                self.credit = element.text
            elif match_tag(element, "publications"):
                self.publications = element.text
            elif match_tag(element, "length_units"):
                self.length_units = element.text
                if self.length_units != "meters":
                    raise RuntimeError("Lengths units must be in meters.")
            elif match_tag(element, "force_units"):
                self.force_units = element.text
                if self.force_units != "N":
                    raise RuntimeError("Force units must be in newtons.")
            elif match_tag(element, "Model"):
                # Everything is encapsulated in the Model tag, so we recall the function with the interior
                self.parse_tags(element)
            else:
                raise RuntimeError(
                    f"Element {element.tag} not recognize. Please verify your xml file or send an issue"
                    f" in the github repository."
                )

    def to_real(self) -> BiomechanicalModelReal:
        return self.biomechanical_model_real

    def _get_body_mesh_list(self, body_set=None) -> list[list[str]]:
        """returns the list of vtp files included in the model"""
        body_mesh_list = []
        body_set = body_set if body_set else self.bodyset_elt[0]
        if is_element_empty(body_set):
            return None
        else:
            for element in body_set:
                mesh = Body.from_element(element).mesh
                body_mesh_list.append(mesh)
            return body_mesh_list

    def _get_marker_set(self) -> list[Marker]:
        markers = []
        if is_element_empty(self.markerset_elt):
            return []
        else:
            original_marker_names = []
            for element in self.markerset_elt[0]:
                marker = Marker.from_element(element)
                original_marker_names += [marker.name]
                markers.append(marker)
            return markers

    def _get_joint_set(self):
        joints = []
        if is_element_empty(self.jointset_elt):
            return []
        else:
            for element in self.jointset_elt[0]:
                joint = Joint.from_element(element, self.ignore_fixed_dof_tag, self.ignore_clamped_dof_tag)
                if joint.function:
                    self.warnings.append(
                        f"Some functions were present for the {joints[-1].name} joint. "
                        "This feature is not implemented in biorbd yet so it will be ignored."
                    )
                joints.append(joint)
            return joints

    def get_controller_set(self):
        if is_element_empty(self.controllerset_elt):
            self.controller_set = None
        else:
            self.warnings.append(
                "Some controllers were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_constraint_set(self):
        if is_element_empty(self.constraintset_elt):
            self.constraintset_elt = None
        else:
            self.warnings.append(
                "Some constraints were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_contact_geometry_set(self):
        if is_element_empty(self.contact_geometryset_elt):
            self.contact_geometryset_elt = None
        else:
            self.warnings.append(
                "Some contact geometry were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_component_set(self):
        if is_element_empty(self.componentset_elt):
            self.componentset_elt = None
        else:
            self.warnings.append(
                "Some additional components were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_probe_set(self):
        if is_element_empty(self.probeset_elt):
            self.probeset_elt = None
        else:
            self.warnings.append(
                "Some probes were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def _set_warnings(self):
        self.get_probe_set()
        self.get_component_set()
        self.get_contact_geometry_set()
        self.get_constraint_set()
        self.biomechanical_model_real.warnings = self.warnings

    def _set_header(self):
        out_string = ""
        out_string += f"\n// File extracted from {self.filepath} on the {strftime('%Y-%m-%d %H:%M')}\n"
        if self.publications:
            out_string += f"\n// Original file publication : {self.publications}\n"
        if self.credit:
            out_string += f"\n// Original file credit : {self.credit}\n"
        if self.force_units:
            out_string += f"\n// Force units : {self.force_units}\n"
        if self.length_units:
            out_string += f"\n// Length units : {self.length_units}\n"
        out_string += f"\n\ngravity\t{self.gravity[0]}\t{self.gravity[1]}\t{self.gravity[2]}"
        self.biomechanical_model_real.header = out_string

    def _set_ground(self):
        ground_set = self.ground_elt
        if not is_element_empty(ground_set):
            dof = Joint
            dof.child_offset_trans, dof.child_offset_rot = [0] * 3, [0] * 3
            self.write_dof(
                Body.from_element(ground_set),
                dof,
                self.mesh_dir,
                skip_virtual=True,
                parent="base",
            )
            for marker in self.markers:
                if marker.parent == "ground":
                    self.biomechanical_model_real.segments["ground"].add_marker(
                        MarkerReal(
                            name=marker.name,
                            parent_name="ground",
                            position=marker.position,
                        )
                    )

    def _set_segments(self):
        for dof in self.joints:
            for body in self.bodies:
                if body.socket_frame == dof.child_body:
                    self.write_dof(
                        body,
                        dof,
                        self.mesh_dir,
                    )

    def _add_markers_to_segments(self, markers: list[Marker]):
        # Add markers to their parent segments
        for marker in markers:
            parent_segment_name = marker.parent
            if parent_segment_name in self.biomechanical_model_real.segments.keys():
                # Convert position string to numpy array with proper float conversion
                position = np.array([float(v) for v in marker.position.split()] + [1.0])  # Add homogeneous coordinate

                # Create MarkerReal instance
                marker_real = MarkerReal(
                    name=marker.name,
                    parent_name=parent_segment_name,
                    position=position,
                    is_technical=True,
                    is_anatomical=False,
                )

                # Add to parent segment
                self.biomechanical_model_real.segments[parent_segment_name].add_marker(marker_real)
            else:
                self.warnings.append(
                    f"Marker {marker.name} references unknown parent segment {parent_segment_name}, skipping"
                )

    def _set_muscles(self):
        """Add the muscle components to the BiomechanicalModelReal."""
        if not self.muscle_groups and not self.muscles and not self.via_points:
            return

        for muscle_group in self.muscle_groups:
            # Add muscle group if it does not exist already
            muscle_group_name = muscle_group.name
            if muscle_group_name not in self.biomechanical_model_real.muscle_groups.keys():
                self.biomechanical_model_real.add_muscle_group(muscle_group)

        for muscle in self.muscles:
            # TODO: The muscle types should represent the opensim types instead of setting them to a fixed type
            muscle.muscle_type = self.muscle_type
            muscle.state_type = self.muscle_state_type
            self.biomechanical_model_real.add_muscle(muscle)

        for via_point in self.via_points:
            self.biomechanical_model_real.add_via_point(via_point)

    def write_dof(self, body, dof, mesh_dir=None, skip_virtual=False, parent=None):

        rotomatrix = OrthoMatrix([0, 0, 0])
        if not skip_virtual:
            parent = dof.parent_body.split("/")[-1]
            axis_offset = np.identity(3)
            # Parent offset
            body_name = body.name + "_parent_offset"
            offset = [dof.parent_offset_trans, dof.parent_offset_rot]
            self.write_virtual_segment(name=body_name, parent_name=parent, frame_offset=offset, rt_in_matrix=0)
            parent = body_name

            # Coordinates
            (
                translations,
                q_ranges_trans,
                trans_dof_names,
                default_value_trans,
                rotations,
                q_ranges_rot,
                rot_dof_names,
                default_value_rot,
            ) = self._get_transformation_parameters(dof.spatial_transform)

            trans_dof_names, rot_dof_names = np.array(trans_dof_names), np.array(rot_dof_names)
            dof_axis = np.array(["x", "y", "z"])

            # Translations
            if len(translations) != 0:
                body_name = body.name + "_translation"
                if is_ortho_basis(translations):
                    trans_axis = ""
                    effective_trans_dof_names = []
                    for idx in np.where(trans_dof_names != None)[0]:
                        trans_axis += dof_axis[idx]
                        effective_trans_dof_names += [trans_dof_names[idx]]
                    axis_offset = self.write_ortho_segment(
                        axis=translations,
                        axis_offset=axis_offset,
                        name=body_name,
                        parent=parent,
                        rt_in_matrix=1,
                        frame_offset=rotomatrix,
                        q_range=q_ranges_trans,
                        trans_dof=trans_axis,
                        dof_names=effective_trans_dof_names,
                    )
                    parent = body_name
                else:
                    raise RuntimeError("Non orthogonal translation vector not implemented yet.")

            # Rotations
            if len(rotations) != 0:
                if is_ortho_basis(rotations):
                    rot_axis = ""
                    effective_rot_dof_names = []
                    for idx in np.where(rot_dof_names != None)[0]:
                        rot_axis += dof_axis[idx]
                        effective_rot_dof_names += [rot_dof_names[idx]]
                    body_name = body.name + "_rotation_transform"
                    axis_offset = self.write_ortho_segment(
                        axis=rotations,
                        axis_offset=axis_offset,
                        name=body_name,
                        parent=parent,
                        rt_in_matrix=1,
                        frame_offset=rotomatrix,
                        q_range=q_ranges_rot,
                        rot_dof=rot_axis,
                        dof_names=effective_rot_dof_names,
                    )
                    parent = body_name
                else:
                    body_name = body.name
                    axis_offset, parent = self.write_non_ortho_rot_segment(
                        rotations,
                        axis_offset,
                        body_name,
                        parent,
                        frame_offset=rotomatrix,
                        rt_in_matrix=1,
                        spatial_transform=dof.spatial_transform,
                        q_ranges=q_ranges_rot,
                        default_values=default_value_rot,
                    )

            # segment to cancel axis effects
            rotomatrix.set_rotation_matrix(np.linalg.inv(axis_offset))

            if not rotomatrix.has_no_transformation():
                body_name = body.name + "_reset_axis"
                self.write_virtual_segment(
                    name=body_name,
                    parent_name=parent,
                    frame_offset=rotomatrix,
                    rt_in_matrix=1,
                )
                parent = body_name

        if parent is None:
            raise RuntimeError(
                f"You skipped virtual segment definition without define a parent." f" Please provide a parent name."
            )

        # True segment
        frame_offset = [dof.child_offset_trans, dof.child_offset_rot]

        body.mesh = body.mesh if len(body.mesh) != 0 else [None]
        body.mesh_color = body.mesh_color if len(body.mesh_color) != 0 else [None]
        body.mesh_scale_factor = body.mesh_scale_factor if len(body.mesh_scale_factor) != 0 else [None]

        axis_offset, parent = self.write_segments_with_a_geometry_only(body, parent, mesh_dir)

        self.write_true_segment(
            name=body.name,
            parent_name=parent,
            frame_offset=frame_offset,
            com=body.mass_center,
            mass=body.mass,
            inertia=body.inertia,
            mesh_file=f"{mesh_dir}/{body.mesh[0]}" if body.mesh[0] and mesh_dir is not None else None,
            mesh_color=body.mesh_color[0] if body.mesh[0] and mesh_dir is not None else None,
            mesh_scale=body.mesh_scale_factor[0] if body.mesh[0] and mesh_dir is not None else None,
            rt_in_matrix=0,
        )

    @staticmethod
    def get_scs_from_offset(rt_in_matrix, frame_offset):
        if rt_in_matrix == 0:
            frame_offset = frame_offset if frame_offset else [[0, 0, 0], [0, 0, 0]]
            segment_coordinate_system = SegmentCoordinateSystemReal.from_euler_and_translation(
                angles=np.array(frame_offset[1]),
                angle_sequence="xyz",
                translation=np.array(frame_offset[0]),
                is_scs_local=True,
            )
        else:
            frame_offset = frame_offset if frame_offset else OrthoMatrix([0, 0, 0])
            translation_vector = frame_offset.get_translation().tolist()
            rotation_matrix = frame_offset.get_rotation_matrix()
            rt_matrix = np.vstack((np.hstack((rotation_matrix, translation_vector)), np.array([0, 0, 0, 1])))
            # TODO: Charbie -> rt_matrix is not always orthogonal, should be handled differently !!!!!!!!
            segment_coordinate_system = SegmentCoordinateSystemReal.from_rt_matrix(
                rt_matrix=get_closest_rt_matrix(rt_matrix), is_scs_local=True
            )
        return segment_coordinate_system

    def write_ortho_segment(
        self,
        axis,
        axis_offset,
        name,
        parent,
        rt_in_matrix,
        frame_offset,
        q_range=None,
        trans_dof="",
        rot_dof="",
        dof_names: list[str] = None,
    ):
        x = axis[0]
        y = axis[1]
        z = axis[2]
        frame_offset.set_rotation_matrix(np.append(x, np.append(y, z)).reshape(3, 3).T)
        self.write_virtual_segment(
            name=name,
            parent_name=parent,
            frame_offset=frame_offset,
            q_range=q_range,
            rt_in_matrix=rt_in_matrix,
            trans_dof=trans_dof,
            rot_dof=rot_dof,
            dof_names=dof_names,
        )
        return axis_offset.dot(frame_offset.get_rotation_matrix())

    def write_non_ortho_rot_segment(
        self,
        axis,
        axis_offset,
        name,
        parent,
        rt_in_matrix,
        frame_offset,
        spatial_transform,
        q_ranges=None,
        default_values=None,
    ):
        default_values = [0, 0, 0] if not default_values else default_values
        axis_basis = []
        list_rot_dof = ["x", "y", "z"]
        count_dof_rot = 0
        q_range = None
        for i, axe in enumerate(axis):
            if len(axis_basis) == 0:
                axis_basis.append(ortho_norm_basis(axe, i))
                initial_rotation = compute_matrix_rotation([float(default_values[i]), 0, 0])
            elif len(axis_basis) == 1:
                axis_basis.append(np.linalg.inv(axis_basis[i - 1]).dot(ortho_norm_basis(axe, i)))
                initial_rotation = compute_matrix_rotation([0, float(default_values[i]), 0])
            else:
                axis_basis.append(
                    np.linalg.inv(axis_basis[i - 1]).dot(np.linalg.inv(axis_basis[i - 2])).dot(ortho_norm_basis(axe, i))
                )
                initial_rotation = compute_matrix_rotation([0, 0, float(default_values[i])])

            coordinate = spatial_transform[i].coordinate
            if coordinate is None:
                body_dof = name + f"_rotation_{i}"
                rot_dof = ""
            else:
                rot_dof = list_rot_dof[count_dof_rot] if not coordinate.locked else "//" + list_rot_dof[count_dof_rot]
                body_dof = name + "_" + spatial_transform[i].coordinate.name
                q_range = q_ranges[i]

            frame_offset.set_rotation_matrix(axis_basis[i].dot(initial_rotation))
            count_dof_rot += 1
            self.write_virtual_segment(
                name=body_dof,
                parent_name=parent,
                frame_offset=frame_offset,
                q_range=self.get_q_range(q_range),
                rt_in_matrix=rt_in_matrix,
                rot_dof=rot_dof,
            )
            axis_offset = axis_offset.dot(frame_offset.get_rotation_matrix())
            parent = body_dof
        return axis_offset, parent

    def write_true_segment(
        self,
        name,
        parent_name,
        frame_offset,
        com,
        mass,
        inertia,
        mesh_file=None,
        mesh_scale=None,
        mesh_color=None,
        rt_in_matrix=0,
    ):
        """
        True segments hold the inertia and markers, but do not have any DoFs.
        These segments are the last "segment" to be added.
        """
        inertia_parameters = None
        if inertia:
            [i11, i22, i33, i12, i13, i23] = inertia.split(" ")
            inertia_parameters = InertiaParametersReal(
                mass=float(mass),
                center_of_mass=np.array([float(c) for c in com.split(" ")]),
                inertia=np.array(
                    [
                        [float(i11), float(i12), float(i13)],
                        [float(i12), float(i22), float(i23)],
                        [float(i13), float(i23), float(i33)],
                    ]
                ),
            )
        self.biomechanical_model_real.add_segment(
            SegmentReal(
                name=name,
                parent_name=parent_name,
                inertia_parameters=inertia_parameters,
                segment_coordinate_system=self.get_scs_from_offset(rt_in_matrix, frame_offset),
                mesh_file=(
                    MeshFileReal(
                        mesh_file_name=mesh_file,
                        mesh_color=tuple(map(float, mesh_color.split())) if mesh_color else None,
                        mesh_scale=tuple(map(float, mesh_scale.split())) if mesh_scale else None,
                    )
                    if mesh_file
                    else None
                ),
            )
        )

    def write_virtual_segment(
        self,
        name,
        parent_name,
        frame_offset,
        q_range=None,
        rt_in_matrix=0,
        trans_dof="",
        rot_dof="",
        mesh_file=None,
        mesh_color=None,
        mesh_scale=None,
        dof_names: list[str] = None,
    ):
        """
        This function aims to add virtual segment to convert osim dof in biomod dof.
        """
        translations = getattr(Translations, trans_dof.upper(), Translations.NONE)
        rotations = getattr(Rotations, rot_dof.upper(), Rotations.NONE)

        self.biomechanical_model_real.add_segment(
            SegmentReal(
                name=name,
                parent_name=parent_name,
                translations=translations,
                rotations=rotations,
                dof_names=dof_names,
                q_ranges=(
                    self.get_q_range(q_range)
                    if (translations != Translations.NONE or rotations != Rotations.NONE)
                    else None
                ),
                qdot_ranges=None,  # OpenSim does not handle qdot ranges
                inertia_parameters=None,
                segment_coordinate_system=self.get_scs_from_offset(rt_in_matrix, frame_offset),
                mesh_file=(
                    MeshFileReal(
                        mesh_file_name=mesh_file,
                        mesh_color=tuple(map(float, mesh_color.split())) if mesh_color else None,
                        mesh_scale=tuple(map(float, mesh_scale.split())) if mesh_scale else None,
                    )
                    if mesh_file
                    else None
                ),
            )
        )

    def write_segments_with_a_geometry_only(self, body, parent, mesh_dir):
        parent_name = parent
        frame_offset = body.socket_frame
        for i, virt_body in enumerate(body.virtual_body):
            if i == 0:
                # ignore the first body as already printed as a true segment
                continue

            body_name = virt_body
            self.write_virtual_segment(
                name=body_name,
                parent_name=parent,
                frame_offset=body.mesh_offset[i],
                mesh_file=f"{mesh_dir}/{body.mesh[i]}" if mesh_dir is not None else None,
                mesh_color=body.mesh_color[i] if mesh_dir is not None else None,
                mesh_scale=body.mesh_scale_factor[i] if mesh_dir is not None else None,
                rt_in_matrix=1,
            )
            parent_name = body_name
            frame_offset = body.mesh_offset[i]
        return frame_offset, parent_name

    @staticmethod
    def get_q_range(q_range):
        if isinstance(q_range, RangeOfMotion) or q_range is None:
            return q_range
        elif isinstance(q_range, list) or isinstance(q_range, str):
            q_range = [q_range] if isinstance(q_range, str) else q_range
            min_bound = []
            max_bound = []
            for range in q_range:
                if range is None:
                    min_bound += [-2 * np.pi]
                    max_bound += [2 * np.pi]
                else:
                    if "// " in range:
                        range = range.replace("// ", "")
                    r = range.split(" ")
                    min_bound += [float(r[0])]
                    max_bound += [float(r[1])]
            q_range = RangeOfMotion(range_type=Ranges.Q, min_bound=min_bound, max_bound=max_bound)
            return q_range
        else:
            raise NotImplementedError(f"You have provided {q_range}, q_range type {type(q_range)} not implemented.")

    def _read(self):
        """Parse the OpenSim model file and populate the output model.

        Processes:
        - Joints and their coordinate ranges
        - Body segments with inertia properties
        - Markers and their parent segments
        - Mesh geometry references

        Raises
        ------
        RuntimeError
            If critical model components are missing or invalid

        Note
        ----
        Modifies the output_model object in place by adding segments, markers, etc.
        """

        # Read the .osim file
        self.muscle_groups, self.muscles, self.via_points = self._get_force_set()
        self.joints = self._get_joint_set()
        self.bodies = self._get_body_set()
        self.markers = self._get_marker_set()
        self.geometry_set = self._get_body_mesh_list()

        # Fill the biomechanical model
        self._set_header()
        self._set_ground()
        self._set_segments()
        self._add_markers_to_segments(self.markers)

        # Muscles
        self._set_muscles()

        # Warnings
        self._set_warnings()

    def _get_body_set(self, body_set: etree.ElementTree = None) -> list[Body]:
        bodies = []
        body_set = body_set if body_set else self.bodyset_elt[0]
        if is_element_empty(body_set):
            return None
        else:
            for element in body_set:
                bodies.append(Body.from_element(element))
            return bodies

    def _get_force_set(self) -> tuple[list[MuscleGroup], list[MuscleReal], list[ViaPointReal]]:
        muscle_groups = []
        muscles = []
        via_points = []
        original_muscle_names = []
        if is_element_empty(self.forceset_elt):
            return None
        else:
            for element in self.forceset_elt[0]:
                if "Muscle" in element.tag:
                    original_muscle_names += [(element.attrib["name"]).split("/")[-1]]
                    muscle_group, muscle, via_point, warnings = get_muscle_from_element(
                        element, self.ignore_muscle_applied_tag
                    )
                    muscle_groups += [muscle_group] if muscle_group is not None else []
                    muscles += [muscle] if muscle is not None else []
                    via_points += via_point if via_point is not None else []
                    if len(warnings) > 0:
                        self.warnings.append(warnings)

                elif "Force" in element.tag or "Actuator" in element.tag:
                    self.warnings.append(
                        f"Some {element.tag} were present in the original file force set. "
                        "Only muscles are supported so they will be ignored."
                    )

            return muscle_groups, muscles, via_points

    def _get_transformation_parameters(self, spatial_transform):
        translations = []
        rotations = []
        q_ranges_trans = []
        q_ranges_rot = []
        trans_dof_names = []
        default_value_trans = []
        default_value_rot = []
        rot_dof_names = []
        for transform in spatial_transform:
            q_range = None
            axis = [float(i.replace(",", ".")) for i in transform.axis.split(" ")]
            if transform.coordinate:
                if transform.coordinate.range:
                    q_range = transform.coordinate.range
                    if not transform.coordinate.clamped:
                        q_range = "// " + q_range
                else:
                    q_range = None
                value = transform.coordinate.default_value
                default_value = value if value else 0
                is_dof_tmp = None if transform.coordinate.locked else transform.coordinate.name
            else:
                is_dof_tmp = None
                default_value = 0
            if transform.type == "translation":
                translations.append(axis)
                q_ranges_trans.append(q_range)
                trans_dof_names.append(is_dof_tmp)
                default_value_trans.append(default_value)
            elif transform.type == "rotation":
                rotations.append(axis)
                q_ranges_rot.append(q_range)
                rot_dof_names.append(is_dof_tmp)
                default_value_rot.append(default_value)
            else:
                raise RuntimeError("Transform must be 'rotation' or 'translation'")
        return (
            translations,
            q_ranges_trans,
            trans_dof_names,
            default_value_trans,
            rotations,
            q_ranges_rot,
            rot_dof_names,
            default_value_rot,
        )
