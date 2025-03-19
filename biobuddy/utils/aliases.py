from typing import TypeAlias, Iterable

import numpy as np

Point: TypeAlias = np.ndarray | Iterable[float]
Points: TypeAlias = np.ndarray | Iterable[Point]


def point_to_array(name: str, point: Point) -> np.ndarray:
    """
    Convert a point to a numpy array

    Parameters
    ----------
    point
        The point to convert

    Returns
    -------
    np.ndarray
        The point as a numpy array
    """
    if not isinstance(point, np.ndarray):
        point = np.array(point)

    error_message = f"The {name} must be a np.ndarray of shape (3,) or (4,), but received: {point.shape}"

    # Check the first dimension
    if len(point.shape) == 1:
        point = point[:, None]
    if point.shape[0] == 3:
        point = np.vstack((point, np.ones(point.shape[1])))
    if point.shape[0] != 4:
        raise RuntimeError(error_message)

    # Check the second dimension
    if len(point.shape) != 2:
        raise RuntimeError(error_message)
    if point.shape[1] != 1:
        raise RuntimeError(error_message)

    return point


def points_to_array(name: str, points: Points) -> np.ndarray:
    """
    Convert a list of points to a numpy array

    Parameters
    ----------
    name
        The name of the variable
    points
        The points to convert

    Returns
    -------
    np.ndarray
        The points as a numpy array
    """
    if isinstance(points, np.ndarray):
        if len(points.shape) == 1:
            points = points[:, None]

        if len(points.shape) != 2 or points.shape[0] not in (3, 4):
            raise RuntimeError(
                f"The {name} must be a np.ndarray of shape (3,), (3, x) (4,) or (4, x), but received: {points.shape}"
            )

        if points.shape[0] == 3:
            points = np.vstack((points, np.ones(points.shape[1])))

        return points

    return np.array([point_to_array(name=name, point=point) for point in points])
