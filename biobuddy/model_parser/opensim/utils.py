from lxml import etree


def find_in_tree(element: etree.ElementTree, string: str):
    if element.find(string) is not None:
        return element.find(string).text
    else:
        return None


def get_etree_without_comments(element: etree.ElementTree) -> etree.ElementTree:
    element_list: list[etree.ElementTree] = []
    for sub_element in element:
        if not isinstance(sub_element, etree._Comment):
            element_list += [sub_element]
    return element_list


def find_sub_elements_in_tree(
    element: etree.ElementTree, parent_element_name: list[str] | str, sub_element_names: list[str] | str
) -> list[etree.ElementTree] | None:
    parent_element = element
    for parent_name in parent_element_name:
        parent_element = parent_element.find(parent_name)
    sub_elements: list[etree.ElementTree] = []
    for sub_elem in get_etree_without_comments(parent_element):
        for sub_element_name in sub_element_names:
            if match_tag(sub_elem, sub_element_name):
                sub_elements += [sub_elem]
    if len(sub_elements) == 0:
        return None
    else:
        return sub_elements


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
