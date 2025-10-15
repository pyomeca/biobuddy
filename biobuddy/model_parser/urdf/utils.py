import numpy as np


def inertia_to_matrix(ixx: float = 0, iyy: float = 0, izz: float = 0, ixy: float = 0, ixz: float = 0, iyz: float = 0):
    # Format symmetric inertia matrix (3x3)
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
