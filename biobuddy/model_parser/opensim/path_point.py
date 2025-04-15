# from typing import Self

from xml.etree import ElementTree

from .utils import find_in_tree


class PathPoint:
    def __init__(self, name: str, muscle: str, body: str, muscle_group: str, position: list):
        self.name = name
        self.muscle = muscle
        self.body = body
        self.muscle_group = muscle_group
        self.position = position

    @staticmethod
    def from_element(element: ElementTree.Element) -> "Self":
        return PathPoint(
            name=element.attrib["name"],
            muscle=None,
            body=find_in_tree(element, "socket_parent_frame").split("/")[-1],
            muscle_group=None,
            position=find_in_tree(element, "location"),
        )
