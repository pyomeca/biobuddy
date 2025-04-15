# from typing import Self

from xml.etree import ElementTree

from .utils import find_in_tree


class Coordinate:
    def __init__(self, name: str, default_value: float, range: list, clamped: bool, locked: bool):
        self.name = name
        self.default_value = default_value
        self.range = range
        self.clamped = clamped
        self.locked = locked

    @staticmethod
    def from_element(element: ElementTree.Element, ignore_fixed: bool = False, ignore_clamped: bool = False) -> "Self":

        if ignore_fixed:
            locked = False
        else:
            locked = find_in_tree(element, "locked") == "true"

        if ignore_clamped:
            clamped = False
        else:
            clamped = find_in_tree(element, "clamped") == "true"

        return Coordinate(
            name=(element.attrib["name"]).split("/")[-1],
            default_value=find_in_tree(element, "default_value"),
            range=find_in_tree(element, "range"),
            clamped=clamped,
            locked=locked,
        )
