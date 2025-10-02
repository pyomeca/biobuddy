from typing import Callable
import numpy as np

from ....utils.protocols import Data
from ....utils.checks import check_name
from ....utils.aliases import points_to_array
from ....utils.linear_algebra import RotoTransMatrix


class Marker:
    def __init__(
        self,
        name: str = None,
        function: Callable | str = None,
        parent_name: str = None,
        is_technical: bool = True,
        is_anatomical: bool = False,
        is_local: bool = False,
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
        is_local
            Indicates whether the marker is defined in the local segment coordinate system.
            If True, the marker is defined in the local coordinate system of the parent segment.
            If False, the marker is defined in the global coordinate system.
        """
        self.name = name
        self.function = function
        self.parent_name = check_name(parent_name)
        self.is_technical = is_technical
        self.is_anatomical = is_anatomical
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

    def to_marker(self, data: Data, model: "BiomechanicalModelReal", scs: RotoTransMatrix) -> "MarkerReal":
        """
        This constructs a MarkerReal by evaluating the function that defines the marker to get an actual position

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        scs
            The segment coordinate system in which the marker is defined. If None, the marker is assumed to be in the global
            coordinate system.
        """
        from ...real.rigidbody.marker_real import MarkerReal

        if self.is_local:
            scs = RotoTransMatrix()
        elif scs is None:
            raise RuntimeError(
                "If you want to provide a global mesh, you must provide the segment's coordinate system."
            )

        # Get the position of the markers and do some sanity checks
        position = points_to_array(points=self.function(data.values, model), name=f"marker function")
        marker_position = scs.inverse @ position

        if np.isnan(marker_position).all():
            raise RuntimeError(f"All the values for {self.function} returned nan which is not permitted")

        return MarkerReal(
            name=self.name,
            parent_name=self.parent_name,
            position=marker_position,
            is_technical=self.is_technical,
            is_anatomical=self.is_anatomical,
        )
