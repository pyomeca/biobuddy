from typing import Callable, TYPE_CHECKING

import numpy as np

from ....utils.aliases import points_to_array
from ....utils.marker_data import MarkerData
from ....utils.checks import check_name
from ....utils.linear_algebra import RotoTransMatrix

if TYPE_CHECKING:
    from ...real.biomechanical_model_real import BiomechanicalModelReal
    from ...real.muscle.via_point_real import ViaPointReal


class ViaPoint:
    def __init__(
        self,
        name: str,
        parent_name: str = None,
        muscle_name: str = None,
        muscle_group: str = None,
        position_function: Callable[[MarkerData, "BiomechanicalModelReal"], np.ndarray] | str = None,
        is_local: bool = True,
    ):
        """
        Parameters
        ----------
        name
            The name of the new via point
        parent_name
            The name of the parent the via point is attached to
        muscle_name
            The name of the muscle that passes through this via point
        muscle_group
            The muscle group the muscle belongs to
        position_function
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the via point with.
        is_local
            If True, the via point is defined in the local coordinate system of the parent segment.
            If False, the via point is defined in the global coordinate system.
            This parameter is not used in this class but may be useful for subclasses or future extensions.
        """
        self.name = name
        self.position_function = position_function
        self.parent_name = check_name(parent_name)
        self.muscle_name = muscle_name
        self.muscle_group = muscle_group
        self.is_local = is_local

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def parent_name(self) -> str:
        return self._parent_name

    @parent_name.setter
    def parent_name(self, value: str) -> None:
        self._parent_name = value

    @property
    def muscle_name(self) -> str:
        return self._muscle_name

    @muscle_name.setter
    def muscle_name(self, value: str) -> None:
        self._muscle_name = value

    @property
    def muscle_group(self) -> str:
        return self._muscle_group

    @muscle_group.setter
    def muscle_group(self, value: str) -> None:
        self._muscle_group = value

    @property
    def position_function(self) -> Callable[[MarkerData, "BiomechanicalModelReal"], np.ndarray] | str:
        return self._position_function

    @position_function.setter
    def position_function(self, value: Callable[[MarkerData, "BiomechanicalModelReal"], np.ndarray] | str) -> None:
        if isinstance(value, str):
            position_function = lambda m, bio: (
                m.get_position([value]) if len(m.get_position([value]).shape) == 1 else m.mean_marker_position(value)
            )
        elif callable(value):
            position_function = value
        elif value is None:
            position_function = None
        else:
            raise TypeError(
                f"Expected a callable or a string, got {type(value)} instead. "
                "Please provide a valid function or marker name."
            )
        self._position_function = position_function

    def to_via_point(self, data: MarkerData, model: "BiomechanicalModelReal", scs: RotoTransMatrix = None) -> "ViaPointReal":
        """
        This constructs a ViaPointReal by evaluating the function that defines the contact to get an actual position

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        scs
            The segment coordinate system in which the via point is defined. This is used to transform the position
            from the global coordinate system to the local coordinate system of the parent segment.
        """
        from ...real.muscle.via_point_real import ViaPointReal

        if self.position_function is None:
            raise RuntimeError("You must provide a position function to evaluate the ViaPoint into a ViaPointReal.")

        if self.is_local:
            # The scs has no effect (should be None)
            scs = RotoTransMatrix()
        else:
            # The scs must be provided when using global coordinates
            if scs is None:
                raise RuntimeError(
                    "If you want to provide a global mesh, you must provide the segment's coordinate system."
                )

        # Get the position of the contact points and do some sanity checks
        p = points_to_array(points=self.position_function(data, model), name="via point function")
        position = scs.inverse @ p
        if np.isnan(position).all():
            raise RuntimeError(f"All the values for {self.position_function} returned nan which is not permitted")

        return ViaPointReal(
            name=self.name,
            parent_name=self.parent_name,
            muscle_name=self.muscle_name,
            muscle_group=self.muscle_group,
            position=position,
            condition=None,  # Not implemented
            movement=None,  # Not implemented
        )
