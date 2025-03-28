from xml.etree import ElementTree


def find_in_tree(element: ElementTree.Element, string: str):
    if element.find(string) is not None:
        return element.find(string).text
    else:
        return None

def _is_element_empty(element):
    if element:
        if not element[0].text:
            return True
        else:
            return False
    else:
        return True