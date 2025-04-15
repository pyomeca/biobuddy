from enum import Enum
import logging
import os
from pathlib import Path

from .mesh import Mesh

_logger = logging.getLogger(__name__)


class MeshFormat(Enum):
    """
    Enum to define the mesh format to use
    """

    VTP = "vtp"


class MeshParser:
    """
    Convert vtp mesh to triangles mesh
    """

    def __init__(self, geometry_folder: str):
        self._geometry_filepaths: list[Path] = []
        available_formats = [available_format.value for available_format in MeshFormat]
        for filename in os.listdir(geometry_folder):
            filepath = Path(geometry_folder) / filename
            if filepath.suffix[1:] in available_formats:
                self._geometry_filepaths.append(filepath)

        self._meshes: list[Mesh] = []
        self._is_processed = False

    @property
    def meshes(self):
        return self._meshes

    @property
    def is_processed(self):
        return self._is_processed

    def process_meshes(self, fail_on_error: bool = True):
        _logger.info("Cleaning vtp file into triangles: ")

        for filepath in self._geometry_filepaths:
            _logger.info(f"Reading - \t{filepath}")

            with open(filepath, "r") as f:
                try:
                    if filepath.suffix[1:] == MeshFormat.VTP.value:
                        self._meshes.append(Mesh.from_vtp(filepath))
                    else:
                        raise ValueError(f"Unsupported format {filepath.suffix}")

                except Exception as e:
                    if fail_on_error:
                        raise e
                    else:
                        self._meshes.append(None)
                        _logger.info(f"\tError while processing {filepath.name}. Skipping")

        self._is_processed = True

    def write(self, folder: str, format: MeshFormat):
        if not self._is_processed:
            raise RuntimeError("The meshes have not been processed yet. Please run process_meshes first.")

        if not os.path.exists(folder):
            os.makedirs(folder)
        for filepath, mesh in zip(self._geometry_filepaths, self._meshes):
            if mesh is not None:
                _logger.info(f"Writing - \t{filepath.name}.")
                if format is MeshFormat.VTP:
                    mesh.to_vtp(filepath=Path(folder) / filepath.name)
                else:
                    raise ValueError(f"Unsupported format {format}")
