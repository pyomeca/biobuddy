import math
import xml.etree.ElementTree as ET
import numpy as np

from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_real import SegmentReal
from ...components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ...components.real.rigidbody.mesh_file_real import MeshFileReal
from ...utils.named_list import NamedList
from ..abstract_model_parser import AbstractModelParser
from .material import Material


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return R_z @ R_y @ R_x


def build_RT_matrix(xyz, rpy):
    R = rpy_to_matrix(*rpy)
    RT = np.eye(4)
    RT[0:3, 0:3] = R
    RT[0:3, 3] = xyz
    return RT


def format_RT_matrix(RT):
    lines = []
    for i in range(4):
        row = RT[i, :]
        lines.append("        " + "    ".join(f"{v:.16g}" for v in row))
    return "\n".join(lines)


def inertia_to_matrix(
        ixx: float = 0,
        iyy: float = 0,
        izz: float = 0,
        ixy: float = 0,
        ixz: float = 0,
        iyz: float = 0):
    # Format symmetric inertia matrix (3x3)
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def add_markers(segment_name, com, marker_outputs):
    marker_outputs.append(f"marker    {segment_name}_origin")
    marker_outputs.append(f"    parent    {segment_name}")
    marker_outputs.append("    position    0 0 0")
    marker_outputs.append("endmarker\n")
    marker_outputs.append(f"marker    {segment_name}_COM")
    marker_outputs.append(f"    parent    {segment_name}")
    marker_outputs.append(f"    position    {com[0]} {com[1]} {com[2]}")
    marker_outputs.append("endmarker\n")


def add_joint_dynamics(joint, lines):
    # Optional: Add joint dynamics info as comments
    dynamics = joint.find("dynamics")
    if dynamics is not None:
        damping = dynamics.attrib.get("damping", None)
        friction = dynamics.attrib.get("friction", None)
        if damping is not None:
            lines.append(f"    // damping {damping}")
        if friction is not None:
            lines.append(f"    // friction {friction}")
    return lines


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
        self.model = ET.parse(filepath).getroot()
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.links_elt: dict[str, ET.Element] = {link.attrib["name"]: link for link in self.model.findall("link")}
        self.joints_elt: list[ET.Element] = self.model.findall("joint")

        # Create the biomechanical model
        self.biomechanical_model_real = BiomechanicalModelReal()
        self._read()

    def _check_version(self):
        version = int(self.model.attrib["Version"])
        if version != 1.0:
            raise NotImplementedError(
                f"The only file version tested yet is 1.0, you have {version}. If you encounter this error, please notify the developers by opening an issue on GitHub.")

    def _get_material_elts(self) -> NamedList[Material]:
        material_elts = NamedList()
        for mat in self.model.findall("material"):
            material_elts._append(Material.from_element(mat))
        return material_elts

    @staticmethod
    def get_inertia_parameters(link: ET.Element) -> InertiaParametersReal:
        inertia = inertia_to_matrix()
        mass = None
        center_of_mass = np.array([0.0, 0.0, 0.0])

        inertial = link.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            if mass_elem is not None and "value" in mass_elem.attrib:
                mass = parse_float(mass_elem.attrib["value"], 0.0)
            origin = inertial.find("origin")
            if origin is not None and "xyz" in origin.attrib:
                center_of_mass = np.array([parse_float(x, 0.0) for x in origin.attrib["xyz"].split()])
            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                ixx = parse_float(inertia_elem.attrib.get("ixx", 0))
                iyy = parse_float(inertia_elem.attrib.get("iyy", 0))
                izz = parse_float(inertia_elem.attrib.get("izz", 0))
                ixy = parse_float(inertia_elem.attrib.get("ixy", 0))
                ixz = parse_float(inertia_elem.attrib.get("ixz", 0))
                iyz = parse_float(inertia_elem.attrib.get("iyz", 0))
                inertia = inertia_to_matrix(ixx, iyy, izz, ixy, ixz, iyz)

        return InertiaParametersReal(
            mass=mass,
            center_of_mass=center_of_mass,
            inertia=inertia,
        )

    def get_mesh_file(self, link: ET.Element) -> MeshFileReal | None:
        visual = link.find("visual")
        mesh_file_name = None
        mesh_color = None
        mesh_rotation = None
        mesh_translation = None
        if visual is not None:
            # Get the file name
            geometry = visual.find("geometry")
            if geometry is not None:
                mesh_elem = geometry.find("mesh")
                if mesh_elem is not None and "filename" in mesh_elem.attrib:
                    mesh_file_name = mesh_elem.attrib["filename"]
            # Get the color of the mesh
            material = visual.find("material")
            if material is not None:
                color_name = material.attrib["name"]
                mesh_color = self.material_elts[color_name].color
            # Get the mesh RT
            origin = visual.find("origin")
            mesh_rotation = origin.attrib["rpy"]
            if mesh_rotation is not None:
                mesh_rotation = np.array([float(elt) for elt in mesh_rotation.split()])
            mesh_translation = origin.attrib["xyz"]
            if mesh_translation is not None:
                mesh_translation = np.array([float(elt) for elt in mesh_translation.split()])

        if mesh_file_name is not None:
            return MeshFileReal(
                mesh_file_name=mesh_file_name,
                mesh_color=mesh_color,
                mesh_scale=None,
                mesh_rotation=mesh_rotation,
                mesh_translation=mesh_translation,
            )
        else:
            return None

    def _read(self):

        self._check_version()
        self.material_elts: NamedList[Material] = self._get_material_elts()

        all_links = set(self.links_elt.keys())
        child_links = {j.find("child").attrib["link"] for j in self.joints_elt}
        root_links = list(all_links - child_links)
        root_link = root_links[0] if root_links else list(all_links)[0]

        segment_outputs = []
        marker_outputs = []

        # Base segment (root link)
        base_link = self.links_elt[root_link]
        self.biomechanical_model_real.add_segment(
            SegmentReal(
                name=root_link,
                inertia_parameters=self.get_inertia_parameters(base_link),
                mesh_file=self.get_mesh_file(base_link),
            ),
        )

        add_markers(root_link, com, marker_outputs)

        previous_segment = root_link

        for joint in self.joints_elt:
            child = joint.find("child").attrib["link"]

            origin = joint.find("origin")
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin is not None:
                if "xyz" in origin.attrib:
                    xyz = [parse_float(x, 0.0) for x in origin.attrib["xyz"].split()]
                if "rpy" in origin.attrib:
                    rpy = [parse_float(r, 0.0) for r in origin.attrib["rpy"].split()]

            RT = build_RT_matrix(xyz, rpy)

            axis_elem = joint.find("axis")
            axis = [0, 0, 0]
            if axis_elem is not None:
                axis = [parse_float(a, 0) for a in axis_elem.attrib["xyz"].split()]

            axis_map = {0: "x", 1: "y", 2: "z"}
            axis_idx = 0
            for i, v in enumerate(axis):
                if abs(v) > 0.5:
                    axis_idx = i
                    break
            axis_str = axis_map.get(axis_idx, "z")

            limit = joint.find("limit")
            lower = 0
            upper = 0
            if limit is not None:
                lower = parse_float(limit.attrib.get("lower", 0))
                upper = parse_float(limit.attrib.get("upper", 0))

            # Rotation segment acting as the child's segment with translation in RT
            rot_segment_name = f"{child}"

            child_link = self.links_elt[child]

            rot_lines = [f"segment {rot_segment_name}"]
            rot_lines.append(f"    parent {previous_segment}")
            rot_lines.append("    RTinMatrix    1")
            rot_lines.append("    RT")
            rot_lines.append(format_RT_matrix(RT))
            rot_lines.append(f"    rotations {axis_str}")
            rot_lines.append(f"    ranges {lower} {upper}")

            rot_lines, mass, com = add_inertia_mass_com(child_link, rot_lines)
            rot_lines = add_mesh(child_link, rot_lines)

            rot_lines.append("endsegment\n")
            segment_outputs.append("\n".join(rot_lines))
            add_markers(rot_segment_name, com, marker_outputs)

            previous_segment = rot_segment_name

        # Add End Effector Marker
        marker_outputs.append(f"marker    {rot_segment_name}_EndEffector")
        marker_outputs.append(f"    parent    {previous_segment}")
        marker_outputs.append("    position    0 0 0.05")
        marker_outputs.append("endmarker\n")

        biomod_lines.extend(segment_outputs)
        biomod_lines.append("// Markers\n")
        biomod_lines.extend(marker_outputs)

        with open(biomod_file, "w") as f:
            f.write("\n".join(biomod_lines))

        print(f"Exported {biomod_file}")

    def to_real(self) -> BiomechanicalModelReal:
        return self.biomechanical_model_real

