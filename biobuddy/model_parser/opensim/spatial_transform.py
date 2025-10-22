from lxml import etree

from ..utils_xml import find_in_tree


class SpatialTransform:
    def __init__(self, name: str, type: str, coordinate_name: str, coordinate: list, axis: str, function: bool):
        self.name = name
        self.type = type
        self.coordinate_name = coordinate_name
        self.coordinate = coordinate
        self.axis = axis
        self.function = function

    @staticmethod
    def from_element(element: etree.ElementTree) -> "Self":
        function = False
        for elt in element[0]:
            if "Function" in elt.tag and len(elt.text) != 0:
                function = True

        return SpatialTransform(
            name=(element.attrib["name"]).split("/")[-1],
            type=find_in_tree(element, "type"),
            coordinate_name=find_in_tree(element, "coordinates"),
            coordinate=find_in_tree(element, "coordinate"),
            axis=find_in_tree(element, "axis"),
            function=function,
        )
