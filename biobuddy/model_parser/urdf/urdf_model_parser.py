import math
import xml.etree.ElementTree as ET
import numpy as np

from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_real import SegmentReal
from ...utils.named_list import NamedList


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


def inertia_to_matrix(ixx, iyy, izz, ixy=0, ixz=0, iyz=0):
    # Format symmetric inertia matrix (3x3)
    return f"        {ixx} {ixy} {ixz}\n" f"        {ixy} {iyy} {iyz}\n" f"        {ixz} {iyz} {izz}"


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


def add_inertia_mass_com(link, lines):
    inertia_mat = "        0 0 0\n        0 0 0\n        0 0 0"
    mass = 0.0
    com = [0.0, 0.0, 0.0]

    inertial = link.find("inertial")
    if inertial is not None:
        mass_elem = inertial.find("mass")
        if mass_elem is not None and "value" in mass_elem.attrib:
            mass = parse_float(mass_elem.attrib["value"], 0.0)
        origin = inertial.find("origin")
        if origin is not None and "xyz" in origin.attrib:
            com = [parse_float(x, 0.0) for x in origin.attrib["xyz"].split()]
        inertia_elem = inertial.find("inertia")
        if inertia_elem is not None:
            ixx = parse_float(inertia_elem.attrib.get("ixx", 0))
            iyy = parse_float(inertia_elem.attrib.get("iyy", 0))
            izz = parse_float(inertia_elem.attrib.get("izz", 0))
            ixy = parse_float(inertia_elem.attrib.get("ixy", 0))
            ixz = parse_float(inertia_elem.attrib.get("ixz", 0))
            iyz = parse_float(inertia_elem.attrib.get("iyz", 0))
            inertia_mat = inertia_to_matrix(ixx, iyy, izz, ixy, ixz, iyz)

    lines.append(f"    mass {mass}")
    lines.append(f"    com    {com[0]} {com[1]} {com[2]}")
    lines.append("    inertia")
    lines.append(inertia_mat)
    return lines, mass, com


def add_mesh(link, lines):
    visual = link.find("visual")
    mesh = None
    if visual is not None:
        geometry = visual.find("geometry")
        if geometry is not None:
            mesh_elem = geometry.find("mesh")
            if mesh_elem is not None and "filename" in mesh_elem.attrib:
                mesh = mesh_elem.attrib["filename"]
    if mesh:
        lines.append(f"    meshfile {mesh}")
    return lines


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


class UrdfModelParser:
    def __init__(self, filepath: str):
        """
        Load the model from the filepath

        Parameters
        ----------
        filepath
            The path to the model to load
        """
        self.gravity = None
        self.segments = NamedList[SegmentReal]()

        # biomod_lines = []
        # biomod_lines.append("version 1")
        # biomod_lines.append("gravity 0 0 -9.81\n")
        biorbd_version = None
        gravity = [0, 0, -9.81]

        tree = ET.parse(filepath)
        root = tree.getroot()

        links = {link.attrib["name"]: link for link in root.findall("link")}
        joints = root.findall("joint")

        all_links = set(links.keys())
        child_links = {j.find("child").attrib["link"] for j in joints}
        root_links = list(all_links - child_links)
        root_link = root_links[0] if root_links else list(all_links)[0]

        segment_outputs = []
        marker_outputs = []

        # Base segment (root link)
        base_link = links[root_link]
        base_lines = [f"segment {root_link}"]
        base_lines, mass, com = add_inertia_mass_com(base_link, base_lines)
        base_lines = add_mesh(base_link, base_lines)
        base_lines.append("endsegment\n")
        segment_outputs.append("\n".join(base_lines))
        add_markers(root_link, com, marker_outputs)

        previous_segment = root_link

        for joint in joints:
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

            child_link = links[child]

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
        """
        Convert the model to a BiomechanicalModelReal
        """
