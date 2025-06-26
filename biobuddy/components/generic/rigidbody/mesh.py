from typing import Callable
import numpy as np

from ...real.biomechanical_model_real import BiomechanicalModelReal
from ...real.rigidbody.mesh_real import MeshReal
from ...real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ....utils.protocols import Data


class Mesh:
    def __init__(
        self,
        functions: tuple[Callable | str, ...],
    ):
        """
        This is a pre-constructor for the MeshReal class. It allows to create a generic model by marker names

        Parameters
        ----------
        functions
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the marker with.
            If a str is provided, the position of the corresponding marker is used
        """
        self.functions = []
        for f in functions:
            if isinstance(f, str):
                self.functions += [
                    lambda m, bio, name=f: m[name] if len(m[name].shape) == 1 else np.nanmean(m[name], axis=1)
                ]
            elif callable(f):
                self.functions += [f]
            else:
                raise TypeError(f"Expected a str or a callable, got {type(f)}")

    def to_mesh(
        self, data: Data, model: BiomechanicalModelReal, parent_scs: SegmentCoordinateSystemReal = None
    ) -> MeshReal:
        return MeshReal.from_data(
            data,
            model,
            self.functions,
            parent_scs,
        )
