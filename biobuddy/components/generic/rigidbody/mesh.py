from typing import Callable
import numpy as np

from ....utils.protocols import Data
from ....utils.linear_algebra import RotoTransMatrix
from ....utils.aliases import points_to_array


class Mesh:
    def __init__(
        self,
        functions: tuple[Callable | str, ...],
        is_local: bool = False,
    ):
        """
        This is a pre-constructor for the MeshReal class. It allows to create a generic model by marker names

        Parameters
        ----------
        functions
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the marker with.
            If a str is provided, the position of the corresponding marker is used
        is_local
            Indicates weather the mesh defined by the functions is defined in the local segment coordinate system.
        """
        self.functions = functions
        self.is_local = is_local

    @property
    def functions(self) -> list[Callable | str]:
        return self._functions

    @functions.setter
    def functions(self, value: list[Callable | str]) -> None:
        functions_list = []
        if value is not None:
            for function in value:
                if isinstance(function, str):
                    functions_list += [
                        lambda m, bio, name=function: (
                            m[name] if len(m[name].shape) == 1 else np.nanmean(m[name], axis=1)
                        )
                    ]
                elif callable(function):
                    functions_list += [function]
                else:
                    raise TypeError(
                        f"Expected a callable or a string, got {type(function)} instead. "
                        "Please provide a valid function or marker name."
                    )
        else:
            functions_list = None
        self._functions = functions_list

    def to_mesh(self, data: Data, model: "BiomechanicalModelReal", scs: RotoTransMatrix) -> "MeshReal":
        """
        This construct a MeshReal object by evaluating the functions that defines the mesh to get actual positions.

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        scs
            The segment coordinate system in which the mesh is defined. If None, the mesh is assumed to be in the global
            coordinate system.
        """
        from ...real.rigidbody.mesh_real import MeshReal

        if self.functions is None:
            raise RuntimeError("You must provide a position function to evaluate the Mesh into a MeshReal.")

        if self.is_local:
            scs = RotoTransMatrix()
        elif scs is None:
            raise RuntimeError(
                "If you want to provide a global mesh, you must provide the segment's coordinate system."
            )

        # Get the position of the all the mesh points and do some sanity checks
        all_p = points_to_array(points=None, name="mesh_real")
        for f in self.functions:
            p = np.nanmean(points_to_array(points=f(data.values, model), name="mesh function"), axis=1)
            projected_p = scs.inverse @ p
            if np.isnan(projected_p).all():
                raise RuntimeError(f"All the values for {f} returned nan which is not permitted")
            all_p = np.hstack((all_p, projected_p.reshape(4, 1)))

        return MeshReal(all_p)
