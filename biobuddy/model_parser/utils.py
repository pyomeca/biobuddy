import numpy as np


def read_float(string: str) -> float:
    return float(string)


def read_float_vector(string: str) -> np.ndarray:
    return np.array([read_float(string_part) for string_part in string.split()])
