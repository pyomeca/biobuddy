from typing import Callable

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
        self.functions = functions

    @property
    def functions(self) -> list[Callable | str]:
        return self._functions

    @functions.setter
    def functions(self, value: list[Callable | str]) -> None:
        functions_list = []
        if value is not None:
            for function in value:
                if not callable(function) and not isinstance(function, str):
                    raise TypeError(
                        f"Expected a callable or a string, got {type(function)} instead. "
                        "Please provide a valid function or marker name."
                    )
                if isinstance(function, str):
                    functions_list += [lambda m, bio: m[function]]
                else:
                    functions_list += [function]
        else:
            functions_list = None
        self._functions = functions_list

    def to_mesh(
        self, data: Data, model: BiomechanicalModelReal, parent_scs: SegmentCoordinateSystemReal = None
    ) -> MeshReal:
        if self.functions is None:
            raise RuntimeError("You must provide a position function to evaluate the Mesh into a MeshReal.")
        return MeshReal.from_data(
            data,
            model,
            self.functions,
            parent_scs,
        )
