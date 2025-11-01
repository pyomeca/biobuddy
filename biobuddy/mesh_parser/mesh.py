from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh:
    """
    Class to store a mesh
    """

    polygons: np.ndarray
    nodes: np.ndarray
    normals: np.ndarray

    @staticmethod
    def from_vtp(filepath: str) -> "Self":
        from .vtp_utils import read_vtp

        return read_vtp(filepath)

    def to_vtp(self, filepath: str) -> None:
        from .vtp_utils import write_vtp

        return write_vtp(mesh=self, filepath=filepath)
