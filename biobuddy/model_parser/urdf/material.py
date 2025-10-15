import numpy as np
import xml.etree.ElementTree as ET

from .utils import find_in_tree


class Material:
    def __init__(
        self,
        name: str,
        color: np.ndarray[float],
    ):
        self.name = name
        self.color = color

    @staticmethod
    def from_element(element: ET.Element) -> "Self":
        name = (element.attrib["name"]).split("/")[-1]
        color_str = find_in_tree(element, "color")
        color = np.array([float(elt) for elt in color_str.split(' ').strip])

        return Material(
            name=name,
            color=color,
        )