from lxml import etree

import numpy as np

from ..biomechanical_model_real import BiomechanicalModelReal
from ....utils.protocols import Data
from ....utils.aliases import point_to_array
from ....utils.linear_algebra import RotoTransMatrix


class MeshFileReal:
    def __init__(
        self,
        mesh_file_name: str,
        mesh_file_directory: str,
        mesh_color: np.ndarray[float] = None,
        mesh_scale: np.ndarray[float] = None,
        mesh_rotation: np.ndarray[float] = None,
        mesh_translation: np.ndarray[float] = None,
    ):
        """
        Parameters
        ----------
        mesh_file_name
            The name of the mesh file (no path)
        mesh_file_directory
            The directory where the mesh file is located
        mesh_color
            The color the mesh should be displayed in (RGB)
        mesh_scale
            The scaling that must be applied to the mesh (XYZ)
        mesh_rotation
            The rotation that must be applied to the mesh (Euler angles: XYZ)
        mesh_translation
            The translation that must be applied to the mesh (XYZ)
        """
        self.mesh_file_name = mesh_file_name
        self.mesh_file_directory = mesh_file_directory
        self.mesh_color = mesh_color
        self.mesh_scale = mesh_scale
        self.mesh_rotation = mesh_rotation
        self.mesh_translation = mesh_translation

    @property
    def mesh_file_name(self) -> str:
        return self._mesh_file_name

    @mesh_file_name.setter
    def mesh_file_name(self, value: str):
        self._mesh_file_name = value

    @property
    def mesh_file_directory(self) -> str:
        return self._mesh_file_directory

    @mesh_file_directory.setter
    def mesh_file_directory(self, value: str):
        self._mesh_file_directory = value

    @property
    def mesh_color(self) -> np.ndarray[float]:
        return self._mesh_color

    @mesh_color.setter
    def mesh_color(self, value: np.ndarray[float]):
        mesh_color = None
        if value is not None:
            mesh_color = np.array(value)
            if mesh_color.shape == (3, 1):
                mesh_color = mesh_color.reshape((3,))
            elif mesh_color.shape != (3,):
                raise RuntimeError("The mesh_color must be a vector of dimension 3 (RGB)")
        self._mesh_color = mesh_color

    @property
    def mesh_scale(self) -> np.ndarray:
        if self._mesh_scale is None:
            return np.ones((4, 1))
        else:
            return self._mesh_scale

    @mesh_scale.setter
    def mesh_scale(self, value: np.ndarray[float]):
        if value is None:
            self._mesh_scale = None
        else:
            self._mesh_scale = point_to_array(value, "mesh_scale")

    @property
    def mesh_rotation(self) -> np.ndarray:
        if self._mesh_rotation is None:
            return np.zeros((4, 1))
        else:
            return self._mesh_rotation

    @mesh_rotation.setter
    def mesh_rotation(self, value: np.ndarray[float]):
        if value is None:
            self._mesh_rotation = None
        else:
            self._mesh_rotation = point_to_array(value, "mesh_rotation")

    @property
    def mesh_translation(self) -> np.ndarray:
        if self._mesh_translation is None:
            return np.zeros((4, 1))
        else:
            return self._mesh_translation

    @mesh_translation.setter
    def mesh_translation(self, value: np.ndarray[float]):
        if value is None:
            self._mesh_translation = None
        else:
            self._mesh_translation = point_to_array(value, "mesh_translation")

    @property
    def mesh_rt(self):
        mesh_rt = RotoTransMatrix()
        mesh_rt.from_euler_angles_and_translation("xyz", self.mesh_rotation[:3, 0], self.mesh_translation[:3, 0])
        return mesh_rt

    @mesh_rt.setter
    def mesh_rt(self, value):
        raise RuntimeError("The mesh_rt cannot be set directly, set the mesh_rotation and mesh_translation instead")

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = ""
        out_string += f"\tmeshfile\t{self.mesh_file_directory}/{self.mesh_file_name}\n"
        if self.mesh_color is not None:
            out_string += f"\tmeshcolor\t{self.mesh_color[0]}\t{self.mesh_color[1]}\t{self.mesh_color[2]}\n"
        out_string += f"\tmeshscale\t{self.mesh_scale[0, 0]}\t{self.mesh_scale[1, 0]}\t{self.mesh_scale[2, 0]}\n"
        if self.mesh_rotation is not None and self.mesh_translation is not None:
            out_string += f"\tmeshrt\t{self.mesh_rotation[0, 0]}\t{self.mesh_rotation[1, 0]}\t{self.mesh_rotation[2, 0]}\txyz\t{self.mesh_translation[0, 0]}\t{self.mesh_translation[1, 0]}\t{self.mesh_translation[2, 0]}\n"
        elif self.mesh_rotation is not None or self.mesh_translation is not None:
            raise RuntimeError("The mesh_rotation and mesh_translation must be both defined or both undefined")
        return out_string

    def to_urdf(self, material_elts: etree.Element, link: etree.Element):

        color_name = None

        # Add the materials from this segment
        if self.mesh_color is not None:
            color_name = f"material_{len(material_elts)}"
            material = etree.SubElement(
                material_elts,
                "material",
                name=color_name,
            )
            color = etree.SubElement(material, "color")
            color.set("rgba", f"{self.mesh_color[0]} {self.mesh_color[1]} {self.mesh_color[2]} 1")

        # Add the visual element to the link
        visual = etree.SubElement(link, "visual")
        geometry = etree.SubElement(visual, "geometry")
        mesh = etree.SubElement(geometry, "mesh")
        mesh.set("filename", f"{self.mesh_file_directory}/{self.mesh_file_name}")
        if np.any(self.mesh_scale != np.ones((4, 1))):
            raise NotImplementedError("Mesh scaling is not implemented yet for URDF export")
        origin = etree.SubElement(visual, "origin")
        origin.set(
            "xyz",
            f"{self.mesh_translation[0,0]} {self.mesh_translation[1,0]} {self.mesh_translation[2,0]}",
        )
        origin.set(
            "rpy",
            f"{self.mesh_rotation[0,0]} {self.mesh_rotation[1,0]} {self.mesh_rotation[2,0]}",
        )
        if color_name is not None:
            visual.set("material", color_name)
