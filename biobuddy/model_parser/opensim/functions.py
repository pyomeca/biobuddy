from lxml import etree
import numpy as np

from ...components.functions import SimmSpline, PiecewiseLinearFunction
from ..utils_xml import find_in_tree


def spline_from_element(element: etree.ElementTree) -> SimmSpline:
    x_points_list = find_in_tree(element, "x").split(" ")
    x_points_list[:] = [float(item) for item in x_points_list if item != ""]
    y_points_list = find_in_tree(element, "y").split(" ")
    y_points_list[:] = [float(item) for item in x_points_list if item != ""]
    return SimmSpline(
        x_points=np.array(x_points_list),
        y_points=np.array(y_points_list),
    )


def piece_wise_linear_from_element(element: etree.ElementTree) -> PiecewiseLinearFunction:
    x_points_list = find_in_tree(element, "x").split(" ")
    x_points_list[:] = [float(item) for item in x_points_list if item != ""]
    y_points_list = find_in_tree(element, "y").split(" ")
    y_points_list[:] = [float(item) for item in x_points_list if item != ""]
    return PiecewiseLinearFunction(
        x_points=np.array(x_points_list),
        y_points=np.array(y_points_list),
    )
