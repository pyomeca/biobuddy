from lxml import etree
import numpy as np

from ..utils_xml import find_in_tree
from ...utils.linear_algebra import RotoTransMatrix


def _extend_mesh_list_with_extra_components(
    mesh_list, element: etree.ElementTree
) -> list[tuple[etree.ElementTree, RotoTransMatrix]]:
    """Convert mesh_list from list[str] to list[tuple(str, RotoTransMatrix)] to include offset in some meshes"""
    mesh_list_and_offset = [(mesh, RotoTransMatrix()) for mesh in mesh_list]

    if element.find("components") is not None:
        frames = element.find("components").findall("PhysicalOffsetFrame")
        for frame in frames:
            if frame.find("attached_geometry") is not None:
                translation = frame.find("translation").text
                translation_array = np.array([float(t) for t in translation.split(" ")])
                mesh = frame.find("attached_geometry").find("Mesh")
                mesh_rt = RotoTransMatrix()
                mesh_rt.from_rotation_matrix_and_translation(
                    rotation_matrix=np.identity(3), translation=translation_array
                )
                mesh_list_and_offset += [(mesh, mesh_rt)]

    return mesh_list_and_offset


class Body:
    def __init__(
        self,
        name: str,
        mass: float,
        inertia: np.ndarray,
        mass_center: np.ndarray,
        wrap: bool,
        socket_frame: str,
        markers: list,
        mesh: list,
        mesh_color: list,
        mesh_scale_factor: list,
        mesh_offset: list,
        virtual_body: list,
    ):
        self.name = name
        self.mass = mass
        self.inertia = inertia
        self.mass_center = mass_center
        self.wrap = wrap
        self.socket_frame = socket_frame
        self.markers = markers
        self.mesh = mesh
        self.mesh_color = mesh_color
        self.mesh_scale_factor = mesh_scale_factor
        self.mesh_offset = mesh_offset
        self.virtual_body = virtual_body

    @staticmethod
    def from_element(element: etree.ElementTree) -> "Self":
        name = (element.attrib["name"]).split("/")[-1]
        mass = find_in_tree(element, "mass")
        inertia = find_in_tree(element, "inertia")
        mass_center = find_in_tree(element, "mass_center")
        geometry = element.find("FrameGeometry")
        socket_frame = name
        if geometry is not None:
            socket_frame = geometry.find("socket_frame").text.split("/")[-1]
            if socket_frame == "..":
                socket_frame = name

        wrap = False
        if element.find("WrapObjectSet") is not None:
            wrap = len(element.find("WrapObjectSet").text) != 0

        mesh = []
        virtual_body = []
        mesh_scale_factor = []
        mesh_color = []
        mesh_offset = []
        if element.find("attached_geometry") is not None:
            mesh_list = element.find("attached_geometry").findall("Mesh")
            mesh_list = _extend_mesh_list_with_extra_components(mesh_list, element)

            for mesh_tp in mesh_list:
                mesh.append(mesh_tp[0].find("mesh_file").text)
                virtual_body.append(mesh_tp[0].attrib["name"])
                mesh_scale_factor_tp = mesh_tp[0].find("scale_factors")
                mesh_scale_factor.append(mesh_scale_factor_tp.text if mesh_scale_factor_tp is not None else None)
                if mesh_tp[0].find("Appearance") is not None:
                    mesh_color_tp = mesh_tp[0].find("Appearance").find("color")
                    mesh_color.append(mesh_color_tp.text if mesh_color_tp is not None else None)
                else:
                    mesh_color.append(None)
                mesh_offset.append(mesh_tp[1])

        return Body(
            name=name,
            mass=mass,
            inertia=inertia,
            mass_center=mass_center,
            wrap=wrap,
            socket_frame=socket_frame,
            markers=[],
            mesh=mesh,
            mesh_color=mesh_color,
            mesh_scale_factor=mesh_scale_factor,
            mesh_offset=mesh_offset,
            virtual_body=virtual_body,
        )

    def to_attrib(self):
        name = self.name
        mass = 1e-8 if not self.mass else float(self.mass)
        inertia = np.identity(3)
        if self.inertia:
            [i11, i22, i33, i12, i13, i23] = self.inertia.split(" ")
            inertia = np.array([[i11, i12, i13], [i12, i22, i23], [i13, i23, i33]])
        center_of_mass = (
            np.zeros(3) if not self.mass_center else np.array([float(i) for i in self.mass_center.split(" ")])
        )
        return name, mass, inertia, center_of_mass
