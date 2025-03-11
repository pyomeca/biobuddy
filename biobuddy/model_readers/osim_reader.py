from xml.etree import ElementTree
from enum import Enum
from time import strftime

import numpy as np
from lxml import etree

from biobuddy.utils import find, OrthoMatrix, compute_matrix_rotation, rot2eul, is_ortho_basis, ortho_norm_basis
from biobuddy.components.segment_real import SegmentReal
from biobuddy.components.inertia_parameters_real import InertiaParametersReal
from biobuddy.components.rotations import Rotations
from biobuddy.components.translations import Translations
from biobuddy.components.segment_coordinate_system_real import SegmentCoordinateSystemReal
from biobuddy.components.marker_real import MarkerReal
from biobuddy.components.range_of_motion import RangeOfMotion, Ranges
from biobuddy.components.mesh_file_real import MeshFileReal
from biobuddy.components.muscle_real import MuscleReal, MuscleType, MuscleStateType
from biobuddy.components.muscle_group import MuscleGroup
from biobuddy.components.via_point_real import ViaPointReal
from biobuddy.biomechanical_model_real import BiomechanicalModelReal


class JointType(Enum):
    WELD_JOINT = "WeldJoint"
    CUSTOM_JOINT = "CustomJoint"


class ForceType(Enum):
    MUSCLE = "Muscle"


class Controller(Enum):
    NONE = None


class OsimReader:
    """Reads and converts OpenSim model files (.osim) to a biomechanical model representation.
    
    Parameters
    ----------
    osim_path : str
        Path to the OpenSim .osim file to read
    muscle_type: MuscleType
        The type of muscle to assume when interpreting the osim model
    muscle_state_type : MuscleStateType
        The muscle state type to assume when interpreting the osim model
    print_warnings : bool, optional
        Whether to print conversion warnings, default True
    mesh_dir : str, optional
        Directory containing mesh files, defaults to 'Geometry_cleaned'
    Raises
    ------
    RuntimeError
        If file version is too old or units are not meters/newtons
    """
    def __init__(self,
                 osim_path: str,
                 muscle_type: MuscleType,
                 muscle_state_type: MuscleStateType,
                 mesh_dir: str,
                 print_warnings: bool = True,
                 ):

        # Initial attributes
        self.osim_path = osim_path
        self.osim_model = etree.parse(self.osim_path)
        self.mesh_dir = mesh_dir
        self.muscle_type = muscle_type
        self.muscle_state_type = muscle_state_type

        # Extended attributes
        self.output_model = BiomechanicalModelReal()
        self.model = ElementTree.parse(self.osim_path)
        self.root = self.model.getroot()[0]
        self.print_warnings = print_warnings
        if self.get_file_version() < 40000:
            raise RuntimeError(
                f".osim file version must be superior or equal to '40000' and you have: {self.get_file_version()}."
                "To convert the osim file to the newest version please open and save your file in"
                "Opensim 4.0 or later."
            )

        self.gravity = np.array([0.0, 0.0, -9.81])
        self.output_model.gravity = np.array([0.0, 0.0, -9.81])
        self.ground_elt, self.default_elt, self.credit, self.publications = None, None, None, None
        self.bodyset_elt, self.jointset_elt, self.forceset_elt, self.markerset_elm = None, None, None, None
        self.controllerset_elt, self.constraintset_elt, self.contact_geometryset_elt = None, None, None
        self.componentset_elt, self.probeset_elt = None, None
        self.length_units, self.force_units = "meters", "newtons"

        for element in self.root:
            if element.tag == "gravity":
                gravity = [float(i) for i in element.text.split(' ')]
                self.gravity = np.array(gravity)
                self.output_model.gravity = np.array(gravity)
            elif element.tag == "Ground":
                self.ground_elt = element
            elif element.tag == "defaults":
                self.default_elt = element
            elif element.tag == "BodySet":
                self.bodyset_elt = element
            elif element.tag == "JointSet":
                self.jointset_elt = element
            elif element.tag == "ControllerSet":
                self.controllerset_elt = element
            elif element.tag == "ConstraintSet":
                self.constraintset_elt = element
            elif element.tag == "ForceSet":
                self.forceset_elt = element
            elif element.tag == "MarkerSet":
                self.markerset_elt = element
            elif element.tag == "ContactGeometrySet":
                self.contact_geometryset_elt = element
            elif element.tag == "ComponentSet":
                self.componentset_elt = element
            elif element.tag == "ProbeSet":
                self.probeset_elt = element
            elif element.tag == "credits":
                self.credit = element.text
            elif element.tag == "publications":
                self.publications = element.text
            elif element.tag == "length_units":
                self.length_units = element.text
                if self.length_units != "meters":
                    raise RuntimeError("Lengths units must be in meters.")
            elif element.tag == "force_units":
                self.force_units = element.text
                if self.force_units != "N":
                    raise RuntimeError("Force units must be in newtons.")
            else:
                raise RuntimeError(
                    f"Element {element.tag} not recognize. Please verify your xml file or send an issue"
                    f" in the github repository."
                )

        self.bodies = []
        self.forces = []
        self.joints = []
        self.markers = []
        self.constraint_set = []
        self.controller_set = []
        self.prob_set = []
        self.component_set = []
        self.geometry_set = []
        self.warnings = []


    @staticmethod
    def _is_element_empty(element):
        if element:
            if not element[0].text:
                return True
            else:
                return False
        else:
            return True

    def get_body_set(self, body_set=None):
        bodies = []
        body_set = body_set if body_set else self.bodyset_elt[0]
        if self._is_element_empty(body_set):
            return None
        else:
            for element in body_set:
                bodies.append(Body().get_body_attrib(element))
            return bodies

    def get_body_mesh_list(self, body_set=None) -> list[str]:
        """returns the list of vtp files included in the model"""
        body_mesh_list = []
        body_set = body_set if body_set else self.bodyset_elt[0]
        if self._is_element_empty(body_set):
            return None
        else:
            for element in body_set:
                body_mesh_list.extend(Body().get_body_attrib(element).mesh)
            return body_mesh_list

    def get_marker_set(self):
        markers = []
        if self._is_element_empty(self.markerset_elt):
            return None
        else:
            original_marker_names = []
            for element in self.markerset_elt[0]:
                marker = Marker().get_marker_attrib(element)
                original_marker_names += [marker.name]
                markers.append(marker)
            return markers

    def get_force_set(self, ignore_muscle_applied_tag=False):
        forces = []
        wrap = []
        original_muscle_names = []
        if self._is_element_empty(self.forceset_elt):
            return None
        else:
            for element in self.forceset_elt[0]:
                if "Muscle" in element.tag:
                    original_muscle_names += [(element.attrib["name"]).split("/")[-1]]
                    current_muscle = Muscle().get_muscle_attributes(element, ignore_muscle_applied_tag)
                    if current_muscle is not None:
                        forces.append(current_muscle)
                        if forces[-1].wrap:
                            wrap.append(forces[-1].name)
                elif "Force" in element.tag or "Actuator" in element.tag:
                    self.warnings.append(
                        f"Some {element.tag} were present in the original file force set. "
                        "Only muscles are supported so they will be ignored."
                    )
            if len(wrap) != 0:
                self.warnings.append(
                    f"Some wrapping objects were present on the muscles :{wrap} in the original file force set.\n"
                    "Only via point are supported in biomod so they will be ignored."
                )
            return forces

    def get_joint_set(self, ignore_fixed_dof_tag=False, ignore_clamped_dof_tag=False):
        joints = []
        if self._is_element_empty(self.forceset_elt):
            return None
        else:
            for element in self.jointset_elt[0]:
                joints.append(Joint().get_joint_attrib(element, ignore_fixed_dof_tag, ignore_clamped_dof_tag))
                if joints[-1].function:
                    self.warnings.append(
                        f"Some functions were present for the {joints[-1].name} joint. "
                        "This feature is not implemented in biorbd yet so it will be ignored."
                    )
            # joints = self._reorder_joints(joints)
            return joints

    def get_controller_set(self):
        if self._is_element_empty(self.controllerset_elt):
            self.controller_set = None
        else:
            self.warnings.append(
                "Some controllers were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_constraint_set(self):
        if self._is_element_empty(self.constraintset_elt):
            self.constraintset_elt = None
        else:
            self.warnings.append(
                "Some constraints were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_contact_geometry_set(self):
        if self._is_element_empty(self.contact_geometryset_elt):
            self.contact_geometryset_elt = None
        else:
            self.warnings.append(
                "Some contact geometry were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_component_set(self):
        if self._is_element_empty(self.componentset_elt):
            self.componentset_elt = None
        else:
            self.warnings.append(
                "Some additional components were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_probe_set(self):
        if self._is_element_empty(self.probeset_elt):
            self.probeset_elt = None
        else:
            self.warnings.append(
                "Some probes were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def set_warnings(self):
        self.get_probe_set()
        self.get_component_set()
        self.get_contact_geometry_set()
        self.get_constraint_set()
        self.output_model.warnings = self.warnings

    def get_file_version(self):
        return int(self.model.getroot().attrib["Version"])


    def set_header(self):
        out_string = ""
        out_string += f"\n// File extracted from {self.osim_path} on the {strftime('%Y-%m-%d %H:%M')}\n"
        if self.publications:
            out_string += f"\n// Original file publication : {self.publications}\n"
        if self.credit:
            out_string += f"\n// Original file credit : {self.credit}\n"
        if self.force_units:
            out_string += f"\n// Force units : {self.force_units}\n"
        if self.length_units:
            out_string += f"\n// Length units : {self.length_units}\n"
        out_string += f"\n\ngravity\t{self.gravity[0]}\t{self.gravity[1]}\t{self.gravity[2]}"
        self.output_model.header = out_string

    def set_ground(self):
        ground_set = self.ground_elt
        if not self._is_element_empty(ground_set):
            dof = Joint()
            dof.child_offset_trans, dof.child_offset_rot = [0] * 3, [0] * 3
            self.write_dof(
                Body().get_body_attrib(ground_set[0]),
                dof,
                self.mesh_dir,
                skip_virtual=True,
                parent="base",
            )
            for marker in self.markers:
                if marker.parent == "ground":
                    self.output_model.segments["ground"].add_marker(MarkerReal(
                        name=marker.name,
                        parent_name="ground",
                        position=marker.position,
                    ))

    def set_segments(self):
        for dof in self.joints:
            for body in self.bodies:
                if body.socket_frame == dof.child_body:
                    self.write_dof(
                        body,
                        dof,
                        self.mesh_dir,
                    )

    def add_markers_to_segments(self, markers):
        # Add markers to their parent segments
        for marker in markers:
            parent_segment_name = marker.parent
            if parent_segment_name in self.output_model.segments:
                # Convert position string to numpy array with proper float conversion
                position = np.array([float(v) for v in marker.position.split()] + [1.0])  # Add homogeneous coordinate

                # Create MarkerReal instance
                marker_real = MarkerReal(
                    name=marker.name,
                    parent_name=parent_segment_name,
                    position=position,
                    is_technical=True,
                    is_anatomical=False
                )

                # Add to parent segment
                self.output_model.segments[parent_segment_name].add_marker(marker_real)
            else:
                self.warnings.append(
                    f"Marker {marker.name} references unknown parent segment {parent_segment_name}, skipping"
                )

    def set_muscles(self):
        """Convert OpenSim muscles to BiomechanicalModelReal muscles."""
        if not self.forces:
            return

        for muscle in self.forces:
            try:
                # Add muscle group if it does not exist already
                muscle_group_name = f"{muscle.group[0]}_to_{muscle.group[1]}"
                if muscle_group_name not in self.output_model.muscle_groups:
                    self.output_model.muscle_groups[muscle_group_name] = MuscleGroup(name=muscle_group_name,
                                                                                     origin_parent_name=
                                                                                     muscle.group[0],
                                                                                     insertion_parent_name=
                                                                                     muscle.group[1])

                # Convert muscle properties
                muscle_real = MuscleReal(
                    name=muscle.name,
                    muscle_type=self.muscle_type,
                    state_type=self.muscle_state_type,
                    muscle_group=muscle_group_name,
                    origin_position=np.array([float(v) for v in muscle.origin.split()]),
                    insertion_position=np.array([float(v) for v in muscle.insersion.split()]),
                    optimal_length=float(muscle.optimal_length) if muscle.optimal_length else 0.1,
                    maximal_force=float(muscle.maximal_force) if muscle.maximal_force else 1000.0,
                    tendon_slack_length=float(muscle.tendon_slack_length) if muscle.tendon_slack_length else None,
                    pennation_angle=float(muscle.pennation_angle) if muscle.pennation_angle else 0.0,
                    maximal_excitation=1.0,  # Default value since OpenSim does not handle maximal excitation.
                )

                # Add via points if any
                for via_point in muscle.via_point:
                    via_real = ViaPointReal(
                        name=via_point.name,
                        parent_name=via_point.body,
                        muscle_name=muscle.name,
                        muscle_group=muscle_real.muscle_group,
                        position=np.array([float(v) for v in via_point.position.split()])
                    )
                    self.output_model.via_points[via_real.name] = via_real

                self.output_model.muscles[muscle.name] = muscle_real

            except Exception as e:
                self.warnings.append(
                    f"Failed to convert muscle {muscle.name}: {str(e)}. Muscle skipped."
                )

    def write_dof(self, body, dof, mesh_dir=None, skip_virtual=False, parent=None):
        rotomatrix = OrthoMatrix([0, 0, 0])
        if not skip_virtual:
            parent = dof.parent_body.split("/")[-1]
            axis_offset = np.identity(3)
            # Parent offset
            body_name = body.name + "_parent_offset"
            offset = [dof.parent_offset_trans, dof.parent_offset_rot]
            self.write_virtual_segment(
                name=body_name,
                parent_name=parent,
                frame_offset=offset,
                rt_in_matrix=0)

            parent = body_name
            # Coordinates
            (
                translations,
                q_ranges_trans,
                is_dof_trans,
                default_value_trans,
                rotations,
                q_ranges_rot,
                is_dof_rot,
                default_value_rot,
            ) = self._get_transformation_parameters(dof.spatial_transform)

            is_dof_trans, is_dof_rot = np.array(is_dof_trans), np.array(is_dof_rot)
            dof_axis = np.array(["x", "y", "z"])
            # if len(translations) != 0 or len(rotations) != 0 -> Segments to define transformation axis.\n")

            # Translations
            if len(translations) != 0:
                body_name = body.name + "_translation"
                if is_ortho_basis(translations):
                    trans_axis = ""
                    for idx in np.where(is_dof_trans != None)[0]:
                        trans_axis += dof_axis[idx]
                    axis_offset = self.write_ortho_segment(
                        axis=translations,
                        axis_offset=axis_offset,
                        name=body_name,
                        parent=parent,
                        rt_in_matrix=1,
                        frame_offset=rotomatrix,
                        q_range=q_ranges_trans,
                        trans_dof=trans_axis,
                    )
                    parent = body_name
                else:
                    raise RuntimeError("Non orthogonal translation vector not implemented yet.")

            # Rotations
            if len(rotations) != 0:
                if is_ortho_basis(rotations):
                    rot_axis = ""
                    for idx in np.where(is_dof_rot != None)[0]:
                        rot_axis += dof_axis[idx]
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

        body.mesh = body.mesh if len(body.mesh) != 0 else [None]
        body.mesh_color = body.mesh_color if len(body.mesh_color) != 0 else [None]
        body.mesh_scale_factor = body.mesh_scale_factor if len(body.mesh_scale_factor) != 0 else [None]

        self.write_true_segment(
            name=body.name,
            parent_name=parent,
            frame_offset=[dof.child_offset_trans, dof.child_offset_rot],
            com=body.mass_center,
            mass=body.mass,
            inertia=body.inertia,
            mesh_file=f"{mesh_dir}/{body.mesh[0]}" if body.mesh[0] else None,
            mesh_color=body.mesh_color[0],
            mesh_scale=body.mesh_scale_factor[0],
            rt_in_matrix=0,
        )
        self.write_segments_with_a_geometry_only(body, body.name, mesh_dir)

    @staticmethod
    def get_scs_from_offset(rt_in_matrix, frame_offset):
        if rt_in_matrix == 0:
            frame_offset = frame_offset if frame_offset else [[0, 0, 0], [0, 0, 0]]
            segment_coordinate_system = SegmentCoordinateSystemReal.from_euler_and_translation(
                angles=frame_offset[1],
                angle_sequence="xyz",
                translations=frame_offset[0],
            )
        else:
            frame_offset = frame_offset if frame_offset else OrthoMatrix([0, 0, 0])
            [[r14], [r24], [r34]] = frame_offset.get_translation().tolist()
            [r41, r42, r43, r44] = [0, 0, 0, 1]

            r11, r12, r13 = frame_offset.get_rotation_matrix()[0, :]
            r21, r22, r23 = frame_offset.get_rotation_matrix()[1, :]
            r31, r32, r33 = frame_offset.get_rotation_matrix()[2, :]
            segment_coordinate_system = SegmentCoordinateSystemReal.from_rt_matrix(
                rt_matrix = np.array([[r11, r12, r13, r14],
                                      [r21, r22, r23, r24],
                                      [r31, r32, r33, r34],
                                      [r41, r42, r43, r44]])
            )
        return segment_coordinate_system

    def write_ortho_segment(
        self, axis, axis_offset, name, parent, rt_in_matrix, frame_offset, q_range=None, trans_dof="", rot_dof=""
    ):
        x = axis[0]
        y = axis[1]
        z = axis[2]
        frame_offset.set_rotation_matrix(np.append(x, np.append(y, z)).reshape(3, 3).T)

        translations = getattr(Translations, ''.join(trans_dof), Translations.NONE)
        rotations = getattr(Rotations, ''.join(rot_dof), Rotations.NONE)
        segment_coordinate_system = self.get_scs_from_offset(rt_in_matrix, frame_offset)
        self.output_model.segments[name] = SegmentReal(
            name=name,
            parent_name=parent,
            translations=translations,
            rotations=rotations,
            q_ranges=self.get_q_range(q_range) if (translations != Translations.NONE and rotations != Rotations.NONE) else None,
            qdot_ranges=None,  # OpenSim does not handle qdot ranges
            inertia_parameters=None,  # TODO: Charbie -> verify this
            segment_coordinate_system=segment_coordinate_system,
            mesh_file=None,
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

            # TODO: Do not add a try here. If the you can know in advance the error, test it with a if.
            #  If you actually need a try, catch a specific error (`except ERRORNAME:` instead of `except:`)
            try:
                coordinate = spatial_transform[i].coordinate
                rot_dof = list_rot_dof[count_dof_rot] if not coordinate.locked else "//" + list_rot_dof[count_dof_rot]
                body_dof = name + "_" + spatial_transform[i].coordinate.name
                q_range = q_ranges[i]
            except:
                body_dof = name + f"_rotation_{i}"
                rot_dof = ""

            frame_offset.set_rotation_matrix(axis_basis[i].dot(initial_rotation))
            count_dof_rot += 1
            self.write_virtual_segment(
                name=body_dof,
                parent_name=parent,
                frame_offset=frame_offset,
                q_range=self.get_q_range(q_range),
                rt_in_matrix=rt_in_matrix,
                rot_dof=rot_dof
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
                inertia=np.array([[float(i11), float(i12), float(i13)], [float(i12), float(i22), float(i23)], [float(i13), float(i23), float(i33)]]),
                )
        self.output_model.segments[name] = SegmentReal(
            name=name,
            parent_name=parent_name,
            inertia_parameters=inertia_parameters,
            segment_coordinate_system=self.get_scs_from_offset(rt_in_matrix, frame_offset),
            mesh_file=MeshFileReal(
                mesh_file_name=mesh_file,
                mesh_color=tuple(map(float, mesh_color.split())) if mesh_color else None,
                mesh_scale=tuple(map(float, mesh_scale.split())) if mesh_scale else None,
            ) if mesh_file else None,
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
    ):
        """
        This function aims to add virtual segment to convert osim dof in biomod dof.
        """
        translations = getattr(Translations, trans_dof, Translations.NONE)
        rotations = getattr(Rotations, rot_dof, Rotations.NONE)

        self.output_model.segments[name] = SegmentReal(
            name=name,
            parent_name=parent_name,
            translations=translations,
            rotations=rotations,
            q_ranges=self.get_q_range(q_range) if (translations != Translations.NONE and rotations != Rotations.NONE) else None,
            qdot_ranges=None,  # OpenSim does not handle qdot ranges
            inertia_parameters=None,
            segment_coordinate_system=self.get_scs_from_offset(rt_in_matrix, frame_offset),
            mesh_file=MeshFileReal(
                mesh_file_name=mesh_file,
                mesh_color=tuple(map(float, mesh_color.split())) if mesh_color else None,
                mesh_scale=tuple(map(float, mesh_scale.split())) if mesh_scale else None,
            ) if mesh_file else None,
        )

    def write_segments_with_a_geometry_only(self, body, parent, mesh_dir):
        for i, virt_body in enumerate(body.virtual_body):
            if i == 0:
                # ignore the first body as already printed as a true segment
                continue

            body_name = virt_body
            self.write_virtual_segment(
                name=body_name,
                parent_name=parent,
                frame_offset=body.mesh_offset[i],
                mesh_file=f"{mesh_dir}/{body.mesh[i]}",
                mesh_color=body.mesh_color[i],
                mesh_scale=body.mesh_scale_factor[i],
                rt_in_matrix=1,
            )

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
                    min_bound += [-2*np.pi]
                    max_bound += [2*np.pi]
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

    @staticmethod
    def _get_transformation_parameters(spatial_transform):
        translations = []
        rotations = []
        q_ranges_trans = []
        q_ranges_rot = []
        is_dof_trans = []
        default_value_trans = []
        default_value_rot = []
        is_dof_rot = []
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
                is_dof_trans.append(is_dof_tmp)
                default_value_trans.append(default_value)
            elif transform.type == "rotation":
                rotations.append(axis)
                q_ranges_rot.append(q_range)
                is_dof_rot.append(is_dof_tmp)
                default_value_rot.append(default_value)
            else:
                raise RuntimeError("Transform must be 'rotation' or 'translation'")
        return (
            translations,
            q_ranges_trans,
            is_dof_trans,
            default_value_trans,
            rotations,
            q_ranges_rot,
            is_dof_rot,
            default_value_rot,
        )


    def read(self):
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
        self.forces = self.get_force_set(ignore_muscle_applied_tag=False)
        self.joints = self.get_joint_set(ignore_fixed_dof_tag=False, ignore_clamped_dof_tag=False)
        self.bodies = self.get_body_set()
        self.markers = self.get_marker_set()
        self.geometry_set = self.get_body_mesh_list()

        # Fill the biomechanical model
        self.set_header()
        self.set_ground()
        self.set_segments()
        self.add_markers_to_segments(self.markers)

        # Muscles
        self.set_muscles()

        # Warnings
        self.set_warnings()


class Body:
    def __init__(self):
        self.name = None
        self.mass = None
        self.inertia = None
        self.mass_center = None
        self.wrap = None
        self.socket_frame = None
        self.markers = []
        self.mesh = []
        self.mesh_color = []
        self.mesh_scale_factor = []
        self.mesh_offset = []
        self.virtual_body = []

    def get_body_attrib(self, element):
        self.name = (element.attrib["name"]).split("/")[-1]
        self.mass = find(element, "mass")
        self.inertia = find(element, "inertia")
        self.mass_center = find(element, "mass_center")
        geometry = element.find("FrameGeometry")
        self.socket_frame = self.name
        if geometry:
            self.socket_frame = geometry.find("socket_frame").text.split("/")[-1]
            if self.socket_frame == "..":
                self.socket_frame = self.name

        if element.find("WrapObjectSet") is not None:
            self.wrap = True if len(element.find("WrapObjectSet").text) != 0 else False

        if element.find("attached_geometry") is not None:
            mesh_list = element.find("attached_geometry").findall("Mesh")
            mesh_list = extend_mesh_list_with_extra_components(mesh_list, element)

            for mesh in mesh_list:
                self.mesh.append(mesh[0].find("mesh_file").text)
                self.virtual_body.append(mesh[0].attrib["name"])
                mesh_scale_factor = mesh[0].find("scale_factors")
                self.mesh_scale_factor.append(mesh_scale_factor.text if mesh_scale_factor is not None else None)
                if mesh[0].find("Appearance") is not None:
                    mesh_color = mesh[0].find("Appearance").find("color")
                    self.mesh_color.append(mesh_color.text if mesh_color is not None else None)
                self.mesh_offset.append(mesh[1])
        return self

    def return_segment_attrib(self):
        name = self.name
        mass = 1e-8 if not self.mass else float(self.mass)
        inertia = np.eye(3)
        if self.inertia:
            [i11, i22, i33, i12, i13, i23] = self.inertia.split(" ")
            inertia = np.array([[i11, i12, i13], [i12, i22, i23], [i13, i23, i33]])
        center_of_mass = np.zeros(3) if not self.mass_center else np.array([float(i) for i in self.mass_center.split(" ")])
        return name, mass, inertia, center_of_mass


def extend_mesh_list_with_extra_components(mesh_list, element) -> list[tuple[str, OrthoMatrix]]:
    """Convert mesh_list from list[str] to list[tuple(str, OrthoMatrix)] to include offset in some meshes"""
    mesh_list_and_offset = [(mesh, OrthoMatrix()) for mesh in mesh_list]

    if element.find("components"):
        frames = element.find("components").findall("PhysicalOffsetFrame")
        for frame in frames:
            if frame.find("attached_geometry") is not None:
                translation = frame.find("translation").text
                translation_tuple = tuple([float(t) for t in translation.split(" ")])
                tuple_mesh_and_offset = (
                    frame.find("attached_geometry").find("Mesh"),
                    OrthoMatrix(translation=translation_tuple),
                )
                mesh_list_and_offset.append(tuple_mesh_and_offset)

    return mesh_list_and_offset


class Joint:
    def __init__(self):
        self.parent = None
        self.child = None
        self.type = None
        self.name = None
        self.coordinates = []
        self.parent_offset_trans = []
        self.parent_offset_rot = []
        self.child_offset_trans = []
        self.child_offset_rot = []
        self.child_body = None
        self.parent_body = None
        self.spatial_transform = []
        self.implemented_joint = [""]
        self.function = False

    def get_joint_attrib(self, element, ignore_fixed, ignore_clamped):
        self.type = element.tag
        if self.type not in [e.value for e in JointType]:
            raise RuntimeError(
                f"Joint type {self.type} is not implemented yet."
                f"Allowed joint type are: {[e.value for e in JointType]}"
            )
        self.name = (element.attrib["name"]).split("/")[-1]
        self.parent = find(element, "socket_parent_frame").split("/")[-1]
        self.child = find(element, "socket_child_frame").split("/")[-1]

        if element.find("coordinates") is not None:
            for coordinate in element.find("coordinates").findall("Coordinate"):
                self.coordinates.append(Coordinate().get_coordinate_attrib(coordinate, ignore_fixed, ignore_clamped))

        if element.find("SpatialTransform") is not None:
            for i, transform in enumerate(element.find("SpatialTransform").findall("TransformAxis")):
                spat_transform = SpatialTransform().get_transform_attrib(transform)
                if i < 3:
                    spat_transform.type = "rotation"
                else:
                    spat_transform.type = "translation"
                for coordinate in self.coordinates:
                    if coordinate.name == spat_transform.coordinate_name:
                        spat_transform.coordinate = coordinate
                self.function = spat_transform.function
                self.spatial_transform.append(spat_transform)

        for frame in element.find("frames").findall("PhysicalOffsetFrame"):
            if self.parent == frame.attrib["name"]:
                self.parent_body = frame.find("socket_parent").text.split("/")[-1]
                offset_rot = frame.find("orientation").text
                offset_trans = frame.find("translation").text
                self.parent_offset_rot = [-float(i) for i in offset_rot.split(" ")]
                self.parent_offset_trans = [float(i) for i in offset_trans.split(" ")]
            elif self.child == frame.attrib["name"]:
                self.child_body = frame.find("socket_parent").text.split("/")[-1]
                offset_rot = frame.find("orientation").text
                offset_trans = frame.find("translation").text
                offset_trans = [float(i) for i in offset_trans.split(" ")]
                offset_rot = [-float(i) for i in offset_rot.split(" ")]
                self.child_offset_trans, self.child_offset_rot = self._convert_offset_child(offset_rot, offset_trans)
        return self

    @staticmethod
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


class Coordinate:
    def __init__(self):
        self.name = None
        self.type = None
        self.default_value = None
        self.range = []
        self.clamped = True
        self.locked = False

    def get_coordinate_attrib(self, element, ignore_fixed=False, ignore_clamped=False):
        self.name = (element.attrib["name"]).split("/")[-1]
        self.default_value = find(element, "default_value")
        self.range = find(element, "range")
        if not ignore_clamped:
            clamped = find(element, "clamped")
            self.clamped = clamped == "true" if clamped else False

        if not ignore_fixed:
            locked = find(element, "locked")
            self.locked = locked == "true" if locked else False
        return self


class SpatialTransform:
    def __init__(self):
        self.name = None
        self.type = None
        self.coordinate_name = None
        self.coordinate = []
        self.axis = None
        self.function = False

    def get_transform_attrib(self, element):
        self.name = (element.attrib["name"]).split("/")[-1]
        self.coordinate_name = find(element, "coordinates")
        self.axis = find(element, "axis")
        for elt in element[0]:
            if "Function" in elt.tag and len(elt.text) != 0:
                self.function = True
        return self

class Muscle:
    def __init__(self):
        self.name = None
        self.via_point = []
        self.type = None
        self.origin = None
        self.insersion = None
        self.optimal_length = None
        self.maximal_force = None
        self.tendon_slack_length = None
        self.pennation_angle = None
        self.applied = True
        self.maximal_velocity = None  # TODO: This is read but not used in the biomod yet (Charbie does not know what it is --')
        self.wrap = False
        self.group = None
        self.state_type = None


    def get_muscle_attributes(self, element, ignore_applied):
        name = (element.attrib["name"]).split("/")[-1]
        self.name = name
        self.maximal_force = find(element, "max_isometric_force")
        self.optimal_length = find(element, "optimal_fiber_length")
        self.tendon_slack_length = find(element, "tendon_slack_length")
        self.pennation_angle = find(element, "pennation_angle_at_optimal")
        self.maximal_velocity = find(element, "max_contraction_velocity")

        if element.find("appliesForce") is not None and not ignore_applied:
            self.applied = element.find("appliesForce").text == "true"

        for path_point_elt in element.find("GeometryPath").find("PathPointSet")[0].findall("PathPoint"):
            via_point = PathPoint().get_path_point_attrib(path_point_elt)
            via_point.muscle = self.name
            self.via_point.append(via_point)
        self.group = [self.via_point[0].body, self.via_point[-1].body]
        for i in range(len(self.via_point)):
            self.via_point[i].muscle_group = f"{self.group[0]}_to_{self.group[1]}"

        if element.find("GeometryPath").find("PathWrapSet") is not None:
            try:
                wrap = element.find("GeometryPath").find("PathWrapSet")[0].text
            except:
                wrap = 0
            n_wrap = 0 if not wrap else len(wrap)
            self.wrap = True if n_wrap != 0 else False

        self.insersion = self.via_point[-1].position
        self.origin = self.via_point[0].position
        self.via_point = self.via_point[1:-1]

        return self


class PathPoint:
    def __init__(self):
        self.name = None
        self.muscle = None
        self.body = None
        self.muscle_group = None
        self.position = None

    def get_path_point_attrib(self, element):
        self.name = element.attrib["name"]
        self.body = find(element, "socket_parent_frame").split("/")[-1]
        self.position = find(element, "location")
        return self


class Marker:
    def __init__(self):
        self.name = None
        self.parent = None
        self.position = None
        self.fixed = True

    def get_marker_attrib(self, element):
        self.name = (element.attrib["name"]).split("/")[-1]
        self.position = find(element, "location")
        self.parent = find(element, "socket_parent_frame").split("/")[-1]
        fixed = find(element, "fixed")
        self.fixed = fixed == "true" if fixed else None
        return self
