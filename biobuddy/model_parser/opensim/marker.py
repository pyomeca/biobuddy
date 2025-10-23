from lxml import etree

from ..utils_xml import find_in_tree


class Marker:
    def __init__(self, name: str, parent: str, position: list, fixed: bool):
        self.name = name
        self.parent = parent
        self.position = position
        self.fixed = fixed

    @staticmethod
    def from_element(element: etree.ElementTree) -> "Self":
        return Marker(
            name=(element.attrib["name"]).split("/")[-1],
            parent=find_in_tree(element, "socket_parent_frame").split("/")[-1],
            position=find_in_tree(element, "location"),
            fixed=(find_in_tree(element, "fixed") == "true" if find_in_tree(element, "fixed") else None),
        )
