from lxml import etree


def find_in_tree(element: etree.ElementTree, string: str):
    if element.find(string) is not None:
        return element.find(string).text
    else:
        return None


def is_element_empty(element):
    if element is not None:
        if not element[0].text:
            return True
        else:
            return False
    else:
        return True


def match_tag(element, tag_name: str):
    return element.tag.lower().strip() == tag_name.lower()


def match_text(element, text: str):
    return element.text.lower().strip() == text.lower()


def str_to_bool(text: str):
    return text.strip().lower() == "true"
