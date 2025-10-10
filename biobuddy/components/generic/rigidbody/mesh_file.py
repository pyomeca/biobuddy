from typing import Callable
import numpy as np

from ....utils.protocols import Data


class MeshFile:
    def __init__(
        self,
        mesh_file_name: str,
        mesh_file_directory: str,
        mesh_color: np.ndarray[float] | list[float] | tuple[float] = None,
        scaling_function: Callable = None,
        rotation_function: Callable = None,
        translation_function: Callable = None,
    ):
        """
        This is a pre-constructor for the MeshFileReal class. It allows to create a generic model by marker names

        Parameters
        ----------
        mesh_file_name
            The name of the mesh file
        mesh_file_directory
            The directory where the mesh file is located
        mesh_color
            The color the mesh should be displayed in (RGB)
        scaling_function
            The function that defines the scaling of the mesh
        rotation_function
            The function that defines the rotation of the mesh
        translation_function
            The function that defines the translation of the mesh
        """
        self.mesh_file_name = mesh_file_name
        self.mesh_file_directory = mesh_file_directory
        self.mesh_color = mesh_color
        self.scaling_function = scaling_function
        self.rotation_function = rotation_function
        self.translation_function = translation_function

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
    def scaling_function(self) -> Callable:
        return self._scaling_function

    @scaling_function.setter
    def scaling_function(self, value: Callable):
        self._scaling_function = value

    @property
    def rotation_function(self) -> Callable:
        return self._rotation_function

    @rotation_function.setter
    def rotation_function(self, value: Callable):
        self._rotation_function = value

    @property
    def translation_function(self) -> Callable:
        return self._translation_function

    @translation_function.setter
    def translation_function(self, value: Callable):
        self._translation_function = value

    def to_mesh_file(
        self,
        data: Data,
        model: "BiomechanicalModelReal",
    ) -> "MeshFileReal":
        """
        This is constructs MeshFileReal by evaluating the functions that defines the mesh file to get actual
        characteristics of the mesh

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        scs
            The segment coordinate system to which the mesh file is attached
        """
        from ...real.rigidbody.mesh_file_real import MeshFileReal

        if self.scaling_function is None:
            mesh_scale = np.array([1, 1, 1])
        else:
            mesh_scale: np.ndarray = self.scaling_function(data.values, model)
            if not isinstance(mesh_scale, np.ndarray):
                raise RuntimeError(
                    f"The scaling_function {self.scaling_function} must return a vector of dimension 3 (XYZ)"
                )
            if mesh_scale.shape == (3, 1):
                mesh_scale = mesh_scale.reshape((3,))
            elif mesh_scale.shape != (3,):
                raise RuntimeError(
                    f"The scaling_function {self.scaling_function} must return a vector of dimension 3 (XYZ)"
                )

        if self.rotation_function is None:
            mesh_rotation = np.array([0, 0, 0])
        else:
            mesh_rotation: np.ndarray = self.rotation_function(data.values, model)
            if not isinstance(mesh_rotation, np.ndarray):
                raise RuntimeError(
                    f"The rotation_function {self.rotation_function} must return a vector of dimension 3 (XYZ)"
                )
            if mesh_rotation.shape == (3, 1):
                mesh_rotation = mesh_rotation.reshape((3,))
            elif mesh_rotation.shape != (3,):
                raise RuntimeError(
                    f"The rotation_function {self.rotation_function} must return a vector of dimension 3 (XYZ)"
                )

        if self.translation_function is None:
            mesh_translation = np.array([0, 0, 0])
        else:
            mesh_translation: np.ndarray = self.translation_function(data.values, model)
            if not isinstance(mesh_translation, np.ndarray):
                raise RuntimeError(
                    f"The translation_function {self.translation_function} must return a vector of dimension 3 (XYZ)"
                )
            if mesh_translation.shape == (3, 1):
                mesh_translation = mesh_translation.reshape((3,))
            elif mesh_translation.shape != (3,):
                raise RuntimeError(
                    f"The translation_function {self.translation_function} must return a vector of dimension 3 (XYZ)"
                )

        return MeshFileReal(
            mesh_file_name=self.mesh_file_name,
            mesh_file_directory=self.mesh_file_directory,
            mesh_color=self.mesh_color,
            mesh_scale=mesh_scale,
            mesh_rotation=mesh_rotation,
            mesh_translation=mesh_translation,
        )
