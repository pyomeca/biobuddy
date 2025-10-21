from lxml import etree
import numpy as np

from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_real import SegmentReal
from ...components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ...components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ...components.real.rigidbody.mesh_file_real import MeshFileReal
from ...components.generic.rigidbody.range_of_motion import RangeOfMotion, Ranges
from ...utils.named_list import NamedList
from ...utils.linear_algebra import RotoTransMatrix
from ...utils.enums import Rotations, Translations
from ..abstract_model_parser import AbstractModelParser
from ..utils import read_float_vector
from .material import Material
from .utils import inertia_to_matrix


# # TODO: implement this function
# def add_joint_dynamics(joint, lines):
#     # Optional: Add joint dynamics info as comments
#     dynamics = joint.find("dynamics")
#     if dynamics is not None:
#         damping = dynamics.attrib.get("damping", None)
#         friction = dynamics.attrib.get("friction", None)
#         if damping is not None:
#             lines.append(f"    // damping {damping}")
#         if friction is not None:
#             lines.append(f"    // friction {friction}")
#     return lines


class UrdfModelParser(AbstractModelParser):
    def __init__(self, filepath: str):
        """
        Load the model from the filepath

        Parameters
        ----------
        filepath
            The path to the model to load
        """
        super().__init__(filepath)
        # Extended attributes
        parsed_file = etree.parse(filepath)
        self._check_version(parsed_file)
        self.model = parsed_file.getroot()
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.links_elt: dict[str, etree.Element] = {link.attrib["name"]: link for link in self.model.findall("link")}
        self.joints_elt: list[etree.Element] = self.model.findall("joint")

        # Create the biomechanical model
        self.biomechanical_model_real = BiomechanicalModelReal()
        self._read()

    def _check_version(self, parsed_file):
        """
        Check that the version of the URDF file is supported
        """
        version = float(parsed_file.docinfo.xml_version)
        if version != 1.0:
            raise NotImplementedError(
                f"The only file version tested yet is 1.0, you have {version}. If you encounter this error, please notify the developers by opening an issue on GitHub."
            )

    def _get_material_elts(self) -> NamedList[Material]:
        """
        Get all the material elements from the URDF file. This will be used later when referencing a material (color name).
        """
        material_elts = NamedList()
        for mat in self.model.findall("material"):
            material_elts._append(Material.from_element(mat))
        return material_elts

    @staticmethod
    def _get_inertia_parameters(link: etree.Element) -> InertiaParametersReal:
        """
        Read the inertia parameters from a link element
        """
        inertia = inertia_to_matrix()
        mass = None
        center_of_mass = np.array([0.0, 0.0, 0.0])

        inertial = link.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            if mass_elem is not None and "value" in mass_elem.attrib:
                mass = float(mass_elem.attrib["value"])
            origin = inertial.find("origin")
            if origin is not None:
                if "xyz" in origin.attrib:
                    center_of_mass = read_float_vector(origin.attrib["xyz"])
                if "rpy" in origin.attrib:
                    rpy = read_float_vector(origin.attrib["rpy"])
                    if np.any(rpy != [0.0, 0.0, 0.0]):
                        raise NotImplementedError(
                            "Inertial origin with non-zero rpy is not supported yet. If you encounter this error, please notify the developers by opening an issue on GitHub."
                        )
            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                ixx = float(inertia_elem.attrib.get("ixx", 0))
                iyy = float(inertia_elem.attrib.get("iyy", 0))
                izz = float(inertia_elem.attrib.get("izz", 0))
                ixy = float(inertia_elem.attrib.get("ixy", 0))
                ixz = float(inertia_elem.attrib.get("ixz", 0))
                iyz = float(inertia_elem.attrib.get("iyz", 0))
                inertia = inertia_to_matrix(ixx, iyy, izz, ixy, ixz, iyz)

        return InertiaParametersReal(
            mass=mass,
            center_of_mass=center_of_mass,
            inertia=inertia,
        )

    def _get_mesh_file(self, link: etree.Element) -> MeshFileReal | None:
        """
        Read the mesh file from a link element
        """
        visual = link.find("visual")
        mesh_file_name = None
        mesh_file_directory = None
        mesh_color = None
        mesh_rotation = None
        mesh_translation = None
        if visual is not None:
            # Get the file name
            geometry = visual.find("geometry")
            if geometry is not None:
                mesh_elem = geometry.find("mesh")
                if mesh_elem is not None and "filename" in mesh_elem.attrib:
                    mesh_file_name = mesh_elem.attrib["filename"].split("/")[-1]
                    mesh_file_directory = "/".join(mesh_elem.attrib["filename"].split("/")[:-1])
            # Get the color of the mesh
            material = visual.find("material")
            if material is not None:
                color_name = material.attrib["name"]
                mesh_color = self.material_elts[color_name].color
            # Get the mesh RT
            origin = visual.find("origin")
            mesh_rotation = origin.attrib["rpy"]
            if mesh_rotation is not None:
                mesh_rotation = read_float_vector(mesh_rotation)
            mesh_translation = origin.attrib["xyz"]
            if mesh_translation is not None:
                mesh_translation = read_float_vector(mesh_translation)

        if mesh_file_name is not None:
            return MeshFileReal(
                mesh_file_name=mesh_file_name,
                mesh_file_directory=mesh_file_directory,
                mesh_color=mesh_color,
                mesh_scale=None,
                mesh_rotation=mesh_rotation,
                mesh_translation=mesh_translation,
            )
        else:
            return None

    @staticmethod
    def _get_scs(joint) -> SegmentCoordinateSystemReal:
        """
        Read the segment coordinate system from a joint element and convert it into a RT matrix
        """
        origin = joint.find("origin")
        origin_translation = np.array([0, 0, 0])
        origin_rotation_angles = np.array([0, 0, 0])
        if origin is not None:
            if "xyz" in origin.attrib:
                origin_translation = read_float_vector(origin.attrib["xyz"])
            if "rpy" in origin.attrib:
                origin_rotation_angles = read_float_vector(origin.attrib["rpy"])
        scs = RotoTransMatrix()
        scs.from_euler_angles_and_translation(
            angle_sequence="xyz", angles=origin_rotation_angles, translation=origin_translation
        )
        return SegmentCoordinateSystemReal(scs=scs, is_scs_local=True)

    @staticmethod
    def _get_rotations_and_ranges(joint) -> tuple[Rotations, RangeOfMotion | None]:
        """
        Read the axis of rotations and ranges of motion from a joint element (fixed joints do not have DoFs)
        """
        if joint.attrib["type"] == "fixed":
            return Rotations.NONE, None
        elif joint.attrib["type"] != "revolute":
            raise NotImplementedError(
                f"Only revolute joints are supported for now, you have a joint of type {joint.attrib['type']}. If you encounter this error, please notify the developers by opening an issue on GitHub."
            )

        # get the axis of rotation
        axis_elem = joint.find("axis")
        axis = np.array([0, 0, 0])
        if axis_elem is not None:
            axis = read_float_vector(axis_elem.attrib["xyz"])

        # Get the associated rotation type
        axis_map = {0: "x", 1: "y", 2: "z"}
        rotations = Rotations.NONE
        axis_is_inverted = False
        for i, ax in enumerate(axis):
            if ax == 1.0:
                rotations = Rotations(axis_map[i])
                break
            elif ax == -1.0:
                rotations = Rotations(axis_map[i])
                axis_is_inverted = True
                break

        # Get the range of motion
        limit = joint.find("limit")
        lower = 0
        upper = 0
        if limit is not None:
            lower = float(limit.attrib.get("lower", 0))
            upper = float(limit.attrib.get("upper", 0))

        if axis_is_inverted:
            ranges = RangeOfMotion(
                range_type=Ranges.Q,
                min_bound=[-upper],
                max_bound=[-lower],
            )
        else:
            ranges = RangeOfMotion(
                range_type=Ranges.Q,
                min_bound=[lower],
                max_bound=[upper],
            )

        return rotations, ranges

    @staticmethod
    def get_parent(link_name: str, all_joints: dict[str, list[str]]) -> str | None:
        """
        Get the parent of this link (link_name)
        """
        parent_name = None
        for joint_keys, joint_links in all_joints.items():
            if link_name == joint_links[1]:
                parent_name = joint_links[0]
                break
        return parent_name

    @staticmethod
    def get_joint_idx(link_name: str, all_joints: dict[str, list[str]]) -> int | None:
        """
        Get the index of the joint associated with this link (link_name)
        """
        joint_idx = 0
        found_dof = False
        for joint_keys, joint_links in all_joints.items():
            if link_name == joint_links[1]:
                found_dof = True
                break
            joint_idx += 1
        return joint_idx if found_dof else None

    def _read(self):
        """
        Read the URDF file and fill the BiomechanicalModelReal
        """
        self.material_elts: NamedList[Material] = self._get_material_elts()

        # Store the joints info (parent/child)
        all_joints = {}
        for joint in self.joints_elt:
            all_joints[joint.attrib["name"]] = [joint.find("parent").attrib["link"], joint.find("child").attrib["link"]]

        # Read the links and get the appropriate joint
        for link_name in self.links_elt:
            parent_name = self.get_parent(link_name, all_joints)
            child_link = self.links_elt[link_name]

            rotations = Rotations.NONE
            ranges = None
            dof_names = None
            scs = SegmentCoordinateSystemReal()
            joint_idx = self.get_joint_idx(link_name, all_joints)
            if joint_idx is not None:
                joint = self.joints_elt[joint_idx]
                rotations, ranges = self._get_rotations_and_ranges(joint)
                dof_names = [joint.attrib["name"]] if rotations != Rotations.NONE else None
                scs = self._get_scs(joint)
            self.biomechanical_model_real.add_segment(
                SegmentReal(
                    name=link_name,
                    parent_name=parent_name if parent_name is not None else "root",
                    segment_coordinate_system=scs,
                    translations=Translations.NONE,
                    rotations=rotations,
                    dof_names=dof_names,
                    q_ranges=ranges,
                    qdot_ranges=None,
                    inertia_parameters=self._get_inertia_parameters(child_link),
                    mesh=None,
                    mesh_file=self._get_mesh_file(child_link),
                ),
            )

    def to_real(self) -> BiomechanicalModelReal:
        return self.biomechanical_model_real
