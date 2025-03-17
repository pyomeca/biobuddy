from xml.etree import ElementTree


def find_in_tree(element: ElementTree.Element, string: str):
    if element.find(string) is not None:
        return element.find(string).text
    else:
        return None
