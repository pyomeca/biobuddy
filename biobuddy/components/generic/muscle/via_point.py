from typing import Callable

from ...real.muscle.via_point_real import ViaPointReal
from ....utils.protocols import Data
from ....utils.checks import check_name


class ViaPoint:
    def __init__(
        self,
        name: str,
        parent_name: str = None,
        muscle_name: str = None,
        muscle_group: str = None,
        position_function: Callable | str = None,
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
        """
        self.name = name
        self.position_function = position_function
        self.parent_name = check_name(parent_name)
        self.muscle_name = muscle_name
        self.muscle_group = muscle_group

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

    def to_via_point(self, data: Data, model: "BiomechanicalModelReal") -> ViaPointReal:
        if self.position_function is None:
            raise RuntimeError("You must provide a position function to evaluate the ViaPoint into a ViaPointReal.")
        return ViaPointReal.from_data(
            data,
            model,
            self.name,
            self.parent_name,
            self.muscle_name,
            self.muscle_group,
            self.position_function,
        )
