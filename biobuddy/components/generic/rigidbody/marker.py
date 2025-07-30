from typing import Callable
import numpy as np

from ...real.biomechanical_model_real import BiomechanicalModelReal
from ...real.rigidbody.marker_real import MarkerReal
from ...real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ....utils.protocols import Data
from ....utils.checks import check_name


class Marker:
    def __init__(
        self,
        name: str = None,
        function: Callable | str = None,
        parent_name: str = None,
        is_technical: bool = True,
        is_anatomical: bool = False,
    ):
        """
        This is a pre-constructor for the Marker class. It allows to create a generic model by marker names

        Parameters
        ----------
        name
            The name of the new marker
        function
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the marker with.
            If a str is provided, the position of the corresponding marker is used
        parent_name
            The name of the parent the marker is attached to
        is_technical
            If the marker should be flagged as a technical marker
        is_anatomical
            If the marker should be flagged as an anatomical marker
        """
        self.name = name
        self.function = function
        self.parent_name = check_name(parent_name)
        self.is_technical = is_technical
        self.is_anatomical = is_anatomical

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
    def function(self) -> Callable | str:
        return self._function

    @function.setter
    def function(self, value: Callable | str) -> None:
        if value is None:
            # Set the function to the name of the marker, so it can be used as a default
            value = self.name

        if isinstance(value, str):
            self._function = lambda m, bio: m[value] if len(m[value].shape) == 1 else np.nanmean(m[value], axis=1)
        elif callable(value):
            self._function = value
        else:
            raise TypeError(
                f"Expected a callable or a string, got {type(value)} instead. "
                "Please provide a valid function or marker name."
            )

    @property
    def is_technical(self) -> bool:
        return self._is_technical

    @is_technical.setter
    def is_technical(self, value: bool) -> None:
        self._is_technical = value

    @property
    def is_anatomical(self) -> bool:
        return self._is_anatomical

    @is_anatomical.setter
    def is_anatomical(self, value: bool) -> None:
        self._is_anatomical = value

    def to_marker(self, data: Data, model: BiomechanicalModelReal) -> MarkerReal:
        return MarkerReal.from_data(
            data,
            model,
            self.name,
            self.function,
            self.parent_name,
            is_technical=self.is_technical,
            is_anatomical=self.is_anatomical,
        )
