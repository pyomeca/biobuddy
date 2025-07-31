from typing import Callable
import numpy as np

from ....utils.protocols import Data
from ....utils.checks import check_name
from ....utils.aliases import points_to_array
from ....utils.linear_algebra import RotoTransMatrix


class ViaPoint:
    def __init__(
        self,
        name: str,
        parent_name: str = None,
        muscle_name: str = None,
        muscle_group: str = None,
        position_function: Callable | str = None,
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
    def position_function(self) -> Callable | str:
        return self._position_function

    @position_function.setter
    def position_function(self, value: Callable | str) -> None:
        if value is not None:
            position_function = (lambda m, bio: m[value]) if isinstance(value, str) else value
        else:
            position_function = None
        self._position_function = position_function

    def to_via_point(self, data: Data, model: "BiomechanicalModelReal", scs: RotoTransMatrix) -> "ViaPointReal":
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
            scs = RotoTransMatrix()
        elif scs is None:
            raise RuntimeError(
                "If you want to provide a global mesh, you must provide the segment's coordinate system."
            )

        # Get the position of the contact points and do some sanity checks
        p = np.nanmean(
            points_to_array(points=self.position_function(data.values, model), name="via point function"), axis=1
        )
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
