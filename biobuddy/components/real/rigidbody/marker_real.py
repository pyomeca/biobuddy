import numpy as np
from lxml import etree

from ....utils.aliases import Points, points_to_array, point_to_array
from ....utils.checks import check_name


class MarkerReal:
    def __init__(
        self,
        name: str,
        parent_name: str = None,
        position: Points = None,
        is_technical: bool = True,
        is_anatomical: bool = False,
    ):
        """
        Parameters
        ----------
        name
            The name of the new marker
        parent_name
            The name of the parent the marker is attached to
        position
            The 3d position of the marker. If multiple positions are given, the mean value is used
        is_technical
            If the marker should be flagged as a technical marker
        is_anatomical
            If the marker should be flagged as an anatomical marker
        """
        self.name = name
        self.parent_name = check_name(parent_name)
        self.position = position

        self.is_technical = is_technical
        self.is_anatomical = is_anatomical

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def parent_name(self) -> str:
        return self._parent_name

    @parent_name.setter
    def parent_name(self, value: str):
        self._parent_name = value

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: Points):
        self._position = points_to_array(points=value, name="marker")

    @property
    def is_technical(self) -> bool:
        return self._is_technical

    @is_technical.setter
    def is_technical(self, value: bool):
        self._is_technical = value

    @property
    def is_anatomical(self) -> bool:
        return self._is_anatomical

    @is_anatomical.setter
    def is_anatomical(self, value: bool):
        self._is_anatomical = value

    @property
    def mean_position(self) -> np.ndarray:
        """
        Get the mean value of the marker position
        """
        if len(self.position.shape) == 1:
            return self.position
        elif len(self.position.shape) == 2 and self.position.shape[0] == 4:
            return np.nanmean(self.position, axis=1)
        else:
            raise NotImplementedError(
                f"marker_real.position is of shape {self.position.shape}, but only shapes (4, ) or (4, nb_frames) are implemented."
            )

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"marker\t{self.name}\n"
        out_string += f"\tparent\t{self.parent_name}\n"

        p = self.mean_position
        out_string += f"\tposition\t{p[0]:0.8f}\t{p[1]:0.8f}\t{p[2]:0.8f}\n"
        out_string += f"\ttechnical\t{1 if self.is_technical else 0}\n"
        out_string += f"\tanatomical\t{1 if self.is_anatomical else 0}\n"
        out_string += "endmarker\n"
        return out_string

    def to_osim(self):
        """Generate OpenSim XML representation of the marker"""

        p = self.mean_position
        marker_elem = etree.Element("Marker", name=self.name)

        socket_parent = etree.SubElement(marker_elem, "socket_parent_frame")
        socket_parent.text = f"bodyset/{self.parent_name}"

        location = etree.SubElement(marker_elem, "location")
        location.text = f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}"

        fixed = etree.SubElement(marker_elem, "fixed")
        fixed.text = "false"

        return marker_elem

    def __add__(self, other: np.ndarray | tuple):
        if isinstance(other, tuple):
            other = np.array(other)

        if isinstance(other, np.ndarray):
            return MarkerReal(
                name=self.name, parent_name=self.parent_name, position=self.position + point_to_array(other)
            )
        elif isinstance(other, MarkerReal):
            return MarkerReal(name=self.name, parent_name=self.parent_name, position=self.position + other.position)
        else:
            raise NotImplementedError(f"The addition for {type(other)} is not implemented")

    def __sub__(self, other):
        if isinstance(other, tuple):
            other = np.array(other)

        if isinstance(other, np.ndarray):
            return MarkerReal(
                name=self.name, parent_name=self.parent_name, position=self.position - point_to_array(other)
            )
        elif isinstance(other, MarkerReal):
            return MarkerReal(name=self.name, parent_name=self.parent_name, position=self.position - other.position)
        else:
            raise NotImplementedError(f"The subtraction for {type(other)} is not implemented")
