import numpy as np
from lxml import etree

from ..utils import read_float_vector


class Material:
    def __init__(
        self,
        name: str,
        color: np.ndarray[float],
    ):
        self.name = name
        self.color = color

    @staticmethod
    def from_element(element: etree.Element) -> "Self":
        # TODO: implement the transparency with the A from RGBA

        name = element.attrib["name"]
        color_str = element.find("color").attrib["rgba"]
        color = read_float_vector(color_str)[:3]
        return Material(
            name=name,
            color=color,
        )
