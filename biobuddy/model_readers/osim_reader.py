from xml.etree import ElementTree
from enum import Enum
from time import strftime

import numpy as np
from lxml import etree

from biobuddy.utils import find, OrthoMatrix, compute_matrix_rotation, rot2eul
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

        self.output_model.gravity = np.array([0.0, 0.0, -9.81])
        self.ground_elt, self.default_elt, self.credit, self.publications = None, None, None, None
        self.bodyset_elt, self.jointset_elt, self.forceset_elt, self.markerset_elm = None, None, None, None
        self.controllerset_elt, self.constraintset_elt, self.contact_geometryset_elt = None, None, None
        self.componentset_elt, self.probeset_elt = None, None
        self.length_units, self.force_units = "meters", "newtons"

        for element in self.root:
            if element.tag == "gravity":
                gravity = [float(i) for i in element.text.split(' ')]
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

    def set_segments(self, body_set=None):
        body_set = body_set if body_set else self.bodyset_elt[0]
        if self._is_element_empty(body_set):
            return None
        else:
            for element in body_set:
                body = Body().get_body_attrib(element)
                name, mass, inertia, center_of_mass = body.return_segment_attrib()
                
                # Create inertia parameters
                inertia_params = InertiaParametersReal(
                    mass=mass,
                    center_of_mass=center_of_mass,
                    inertia=inertia
                )
                
                # Find corresponding joint
                joint = next((j for j in self.joints if j.child_body == name), None)
                
                # Determine translations and rotations from joint
                translations = Translations.NONE
                rotations = Rotations.NONE
                if joint:
                    rotation_axes = []
                    translation_axes = []
                    for transform in joint.spatial_transform:
                        if transform.type == 'rotation':
                            axis = list(map(float, transform.axis.split()))
                            if axis[0] == 1.0:
                                rotation_axes.append('X')
                            elif axis[1] == 1.0:
                                rotation_axes.append('Y')
                            elif axis[2] == 1.0:
                                rotation_axes.append('Z')
                        elif transform.type == 'translation':
                            axis = list(map(float, transform.axis.split()))
                            if axis[0] == 1.0:
                                translation_axes.append('X')
                            elif axis[1] == 1.0:
                                translation_axes.append('Y')
                            elif axis[2] == 1.0:
                                translation_axes.append('Z')
                    
                    # Get rotations enum
                    if rotation_axes:
                        rotation_name = ''.join(rotation_axes)
                        rotations = getattr(Rotations, rotation_name, Rotations.NONE)
                    
                    # Get translations enum
                    if translation_axes:
                        translation_name = ''.join(translation_axes)
                        translations = getattr(Translations, translation_name, Translations.NONE)
                    else:
                        translations = None
                
                # Create segment coordinate system from joint offsets
                scs = SegmentCoordinateSystemReal()
                if joint:
                    try:
                        # Parse joint offset values
                        translation = joint.child_offset_trans
                        rotation = joint.child_offset_rot
                        angle_sequence = ''.join(rotation_axes).lower() if rotation_axes else 'xyz'
                        
                        scs = SegmentCoordinateSystemReal.from_euler_and_translation(
                            angles=rotation,
                            angle_sequence=angle_sequence,
                            translations=translation,
                        )
                    except Exception as e:
                        self.warnings.append(
                            f"Could not parse joint offsets for segment {name}: {str(e)}. Using default coordinate system."
                        )

                # Get coordinate ranges from joint
                q_ranges = []
                if joint:
                    for coord in joint.coordinates:
                        if coord.range:
                            min_val, max_val = map(float, coord.range.split())
                            q_ranges.append((min_val, max_val))

                # Create virtual parent chain first
                current_parent: str = body.socket_frame if body.socket_frame != name else ""
                virtual_names: list[str] = []

                # Create transformation virtual segments first in parent chain
                for i, virt_body in enumerate(body.virtual_body):
                    if i == 0:  # Skip first body as it's the main segment we'll add later
                        continue

                    virt_name = f"{name}_{virt_body}"
                    if i >= len(body.mesh_offset):
                        break  # For safety if offsets list is shorter

                    # Create virtual segment with transformation
                    mesh_offset = body.mesh_offset[i]
                    self.output_model.segments[virt_name] = SegmentReal(
                        name=virt_name,
                        parent_name=current_parent,
                        translations=Translations.NONE,
                        rotations=Rotations.NONE,
                        segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                            angles=rot2eul(mesh_offset.get_rotation_matrix()).reshape(3, ),
                            angle_sequence="xyz",
                            translations=mesh_offset.get_translation().reshape(3, ),
                        ),
                        mesh_file=None  # Virtual segments in parent chain don't carry meshes
                    )

                    virtual_names.append(virt_name)
                    current_parent = virt_name

                # Create main segment as child of last virtual parent
                self.output_model.segments[name] = SegmentReal(
                    name=name,
                    parent_name=current_parent,
                    translations=translations,
                    rotations=rotations,
                    q_ranges=RangeOfMotion(Ranges.Q, [r[0] for r in q_ranges], [r[1] for r in q_ranges]) if q_ranges else None,
                    qdot_ranges=None,
                    inertia_parameters=inertia_params,
                    segment_coordinate_system=scs,
                    mesh_file=MeshFileReal(
                        mesh_file_name=f"{self.mesh_dir}/{body.mesh[0]}" if body.mesh else None,
                        mesh_translation=tuple(map(float, body.mesh_offset[0].get_translation().flatten())) if body.mesh_offset else None,
                        mesh_rotation=tuple(map(float, rot2eul(body.mesh_offset[0].get_rotation_matrix()).flatten())) if body.mesh_offset else None,
                        mesh_color=tuple(map(float, body.mesh_color[0].split())) if body.mesh_color else None,
                        mesh_scale=tuple(map(float, body.mesh_scale_factor[0].split())) if body.mesh_scale_factor else None,
                    ) if body.mesh else None,
                )

                # Add geometry virtual segments as children of main segment
                for i, virt_body in reversed(list(enumerate(body.virtual_body))):
                    if i == 0:  # Skip first body as it's the main segment
                        continue

                    virt_name = f"{name}_{virt_body}_geom"
                    if i >= len(body.mesh):
                        break  # For safety if meshes list is shorter

                    self.output_model.segments[virt_name] = SegmentReal(
                        name=virt_name,
                        parent_name=name,
                        translations=Translations.NONE,
                        rotations=Rotations.NONE,
                        segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                            angles=[0, 0, 0],
                            angle_sequence="xyz",
                            translations=[0, 0, 0]
                        ),
                        mesh_file=MeshFileReal(
                            mesh_file_name=f"{self.mesh_dir}/{body.mesh[i-1]}",  # Use i-1 since main segment is index 0
                            mesh_translation=tuple(map(float, body.mesh_offset[i].get_translation().flatten())) if body.mesh_offset else None,
                            mesh_rotation=tuple(map(float, rot2eul(body.mesh_offset[i].get_rotation_matrix()).flatten())) if body.mesh_offset else None,
                            mesh_color=tuple(map(float, body.mesh_color[i-1].split())) if body.mesh_color else None,
                            mesh_scale=tuple(map(float, body.mesh_scale_factor[i-1].split())) if body.mesh_scale_factor else None,
                        )
                    )
            return

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
                                                                                    origin_parent_name=muscle.group[0],
                                                                                    insertion_parent_name=muscle.group[1])

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

    @staticmethod
    def _reorder_joints(joints: list):
        # TODO: This function is not actually called. Is it necessary?
        ordered_joints = [joints[0]]
        joints.pop(0)
        while len(joints) != 0:
            for o, ord_joint in enumerate(ordered_joints):
                idx = []
                for j, joint in enumerate(joints):
                    if joint.parent == ord_joint.child:
                        ordered_joints = ordered_joints + [joint]
                        idx.append(j)
                    elif ord_joint.parent == joint.child:
                        ordered_joints = [joint] + ordered_joints
                        idx.append(j)
                if len(idx) != 0:
                    joints.pop(idx[0])
                elif len(idx) > 1:
                    raise RuntimeError("Two segment can't have the same parent in a biomod.")
        return ordered_joints

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
        self.output_model.header = out_string


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

        self.joints = self.get_joint_set(ignore_fixed_dof_tag=False, ignore_clamped_dof_tag=False)
        self.markers = self.get_marker_set()
        self.geometry_set = self.get_body_mesh_list()
        self.forces = self.get_force_set(ignore_muscle_applied_tag=False)

        # Header
        self.set_header()

        # Segments
        self.set_segments(body_set=[self.ground_elt])
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
                self.parent_offset_rot = frame.find("orientation").text
                self.parent_offset_trans = frame.find("translation").text
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
        self.pcsa = None
        self.maximal_velocity = None
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
