from .aliases import Point, Points, point_to_array, points_to_array

import numpy as np


def euler_and_translation_to_matrix(
    angles: Points,
    angle_sequence: str,
    translations: Point,
) -> np.ndarray:
    """
    Construct a SegmentCoordinateSystemReal from angles and translations

    Parameters
    ----------
    angles
        The actual angles
    angle_sequence
        The angle sequence of the angles
    translations
        The XYZ translations
    parent_scs
        The scs of the parent (is used when printing the model so SegmentCoordinateSystemReal
        is in parent's local reference frame
    """

    angles = point_to_array(name="angles", point=angles).reshape((4,))
    translations = point_to_array(name="translations", point=translations).reshape((4,))

    matrix = {
        "x": lambda x: np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]),
        "y": lambda y: np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]),
        "z": lambda z: np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]),
    }
    rt = np.identity(4)
    for angle, axis in zip(angles, angle_sequence):
        rt[:3, :3] = rt[:3, :3] @ matrix[axis](angle)
    rt[:3, 3] = translations[:3]

    return rt


def mean_homogenous_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Computes the closest homogenous matrix that approximates all the homogenous matrices

    This is based on the dmuir answer on Stack Overflow
    https://stackoverflow.com/questions/51517466/what-is-the-correct-way-to-average-several-rotation-matrices

    Returns
    -------
    The mean homogenous matrix
    """
    mean_matrix = np.identity(4)

    # Perform an Arithmetic mean of each element
    arithmetic_mean_scs = np.nanmean(matrix, axis=2)
    mean_matrix[:3, 3] = arithmetic_mean_scs[:3, 3]

    # Get minimized rotation matrix from the svd decomposition
    u, s, v = np.linalg.svd(arithmetic_mean_scs[:3, :3])
    mean_matrix[:3, :3] = u @ v
    return mean_matrix


def to_euler(rt, sequence: str) -> np.ndarray:
    if sequence == "xyz":
        rx = np.arctan2(-rt[1, 2, :], rt[2, 2, :])
        ry = np.arcsin(rt[0, 2, :])
        rz = np.arctan2(-rt[0, 1, :], rt[0, 0, :])
    else:
        raise NotImplementedError("This sequence is not implemented yet")

    return np.array([rx, ry, rz])


def transpose_homogenous_matrix(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix).transpose((1, 0, 2))
    out[:3, 3, :] = np.einsum("ijk,jk->ik", -out[:3, :3, :], matrix[:3, 3, :])
    out[3, :3, :] = 0
    return out


def multiply_homogeneous_matrix(self: np.ndarray, other: np.ndarray) -> np.ndarray:
    if len(other.shape) == 3:  # If it is a RT @ RT
        return np.einsum("ijk,jlk->ilk", self.scs, other)
    elif len(other.shape) == 2:  # if it is a RT @ vector
        return np.einsum("ijk,jk->ik", self.scs, other)
    else:
        NotImplementedError("This multiplication is not implemented yet")


def norm2(v) -> np.ndarray:
    """Compute the squared norm of each row of the matrix v."""
    return np.sum(v**2, axis=1)


def compute_matrix_rotation(_rot_value) -> np.ndarray:
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(_rot_value[0]), -np.sin(_rot_value[0])],
            [0, np.sin(_rot_value[0]), np.cos(_rot_value[0])],
        ]
    )

    rot_y = np.array(
        [
            [np.cos(_rot_value[1]), 0, np.sin(_rot_value[1])],
            [0, 1, 0],
            [-np.sin(_rot_value[1]), 0, np.cos(_rot_value[1])],
        ]
    )

    rot_z = np.array(
        [
            [np.cos(_rot_value[2]), -np.sin(_rot_value[2]), 0],
            [np.sin(_rot_value[2]), np.cos(_rot_value[2]), 0],
            [0, 0, 1],
        ]
    )
    rot_matrix = np.dot(rot_z, np.dot(rot_y, rot_x))
    return rot_matrix


def rot2eul(rot) -> np.ndarray:
    beta = -np.arcsin(rot[2, 0])
    alpha = np.arctan2(rot[2, 1], rot[2, 2])
    gamma = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array((alpha, beta, gamma))


def get_closest_rotation_matrix(rt_matrix: np.ndarray) -> np.ndarray:
    """
    Projects a rotation matrix to the closest rotation matrix using Singular Value Decomposition (SVD).
    """
    if np.abs(np.sum(rt_matrix[:3, :3] ** 2) - 3.0) < 1e-6:
        return rt_matrix

    else:
        output_rt = np.eye(4)
        output_rt[:3, 3] = rt_matrix[:3, 3]

        u, _, vt = np.linalg.svd(rt_matrix[:3, :3])
        projected_rot_matrix = u @ vt

        # Ensure det(R) = +1
        if np.linalg.det(projected_rot_matrix) < 0:
            u[:, -1] *= -1
            projected_rot_matrix = u @ vt

        output_rt[:3, :3] = projected_rot_matrix
        return output_rt


def coord_sys(axis) -> tuple[list[np.ndarray], str]:
    # define orthonormal coordinate system with given z-axis
    [a, b, c] = axis
    if a == 0:
        if b == 0:
            if c == 0:
                return [[1, 0, 0], [0, 1, 0], [0, 0, 1]], ""
            else:
                return [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "z"
        else:
            if c == 0:
                return [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "y"
            else:
                y_temp = [0, -c / b, 1]
    else:
        if b == 0:
            if c == 0:
                return [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "x"
            else:
                y_temp = [-c / a, 0, 1]
        else:
            y_temp = [-b / a, 1, 0]
    z_temp = [a, b, c]
    x_temp = np.cross(y_temp, z_temp)
    norm_x_temp = np.linalg.norm(x_temp)
    norm_z_temp = np.linalg.norm(z_temp)
    x = [1 / norm_x_temp * x_el for x_el in x_temp]
    z = [1 / norm_z_temp * z_el for z_el in z_temp]
    y = [y_el for y_el in np.cross(z, x)]
    return [x, y, z], ""


def ortho_norm_basis(vector, idx) -> np.ndarray:
    # build an orthogonal basis fom a vector
    basis = []
    v = np.random.random(3)
    vector_norm = vector / np.linalg.norm(vector)
    z = np.cross(v, vector_norm)
    z_norm = z / np.linalg.norm(z)
    y = np.cross(vector_norm, z)
    y_norm = y / np.linalg.norm(y)
    if idx == 0:
        basis = np.append(vector_norm, np.append(y_norm, z_norm)).reshape(3, 3).T
        if np.linalg.det(basis) < 0:
            basis = np.append(vector_norm, np.append(y_norm, -z_norm)).reshape(3, 3).T
    elif idx == 1:
        basis = np.append(y_norm, np.append(vector_norm, z_norm)).reshape(3, 3).T
        if np.linalg.det(basis) < 0:
            basis = np.append(y_norm, np.append(vector_norm, -z_norm)).reshape(3, 3).T
    elif idx == 2:
        basis = np.append(z_norm, np.append(y_norm, vector_norm)).reshape(3, 3).T
        if np.linalg.det(basis) < 0:
            basis = np.append(-z_norm, np.append(y_norm, vector_norm)).reshape(3, 3).T
    return basis


def is_ortho_basis(basis) -> bool:
    return (
        False
        if np.dot(basis[0], basis[1]) != 0 or np.dot(basis[1], basis[2]) != 0 or np.dot(basis[0], basis[2]) != 0
        else True
    )

def rotation_matrix_from_euler(axis: str, angle: float) -> np.ndarray:
    if axis.upper() == "X":
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )
    elif axis.upper() == "Y":
        return np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
    elif axis.upper() == "Z":
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise RuntimeError(f"Axis {axis} not recognized. Please use X, Y, or Z.")


class OrthoMatrix:
    def __init__(self, translation=(0, 0, 0), rotation_1=(0, 0, 0), rotation_2=(0, 0, 0), rotation_3=(0, 0, 0)):
        self.trans = np.transpose(np.array([translation]))
        self.axe_1 = rotation_1  # axis of rotation for theta_1
        self.axe_2 = rotation_2  # axis of rotation for theta_2
        self.axe_3 = rotation_3  # axis of rotation for theta_3
        self.rot_1 = np.transpose(np.array(coord_sys(self.axe_1)[0]))  # rotation matrix for theta_1
        self.rot_2 = np.transpose(np.array(coord_sys(self.axe_2)[0]))  # rotation matrix for theta_2
        self.rot_3 = np.transpose(np.array(coord_sys(self.axe_3)[0]))  # rotation matrix for theta_3
        self.rotation_matrix = self.rot_3.dot(self.rot_2.dot(self.rot_1))  # rotation matrix for
        self.matrix = np.append(np.append(self.rotation_matrix, self.trans, axis=1), np.array([[0, 0, 0, 1]]), axis=0)

    def get_rotation_matrix(self):
        return self.rotation_matrix

    def set_rotation_matrix(self, rotation_matrix):
        self.rotation_matrix = rotation_matrix

    def get_translation(self):
        return self.trans

    def set_translation(self, trans):
        self.trans = trans

    def get_matrix(self):
        return np.append(np.append(self.rotation_matrix, self.trans, axis=1), np.array([[0, 0, 0, 1]]), axis=0)

    def transpose(self):
        self.rotation_matrix = np.transpose(self.rotation_matrix)
        self.trans = -self.rotation_matrix.dot(self.trans)
        return self.matrix

    def product(self, other):
        self.rotation_matrix = self.rotation_matrix.dot(other.get_rotation_matrix())
        self.trans = self.trans + other.get_translation()
        return self.matrix

    def get_axis(self):
        return coord_sys(self.axe_1)[1] + coord_sys(self.axe_2)[1] + coord_sys(self.axe_3)[1]

    def has_no_transformation(self):
        return np.all(self.get_matrix() == np.eye(4))


class RotoTransMatrix:
    def __init__(self):
        self.rt_matrix = None

    def from_rotation_and_translation(self, rotation_matrix: np.ndarray, translation: np.ndarray):
        rt_matrix = np.zeros((4, 4))
        rt_matrix[:3, :3] = rotation_matrix[:3, :3]
        rt_matrix[:3, 3] = translation
        rt_matrix[3, 3] = 1.0
        self._rt_matrix = rt_matrix

    @property
    def rt_matrix(self) -> np.ndarray:
        return self._rt_matrix

    @rt_matrix.setter
    def rt_matrix(self, value: np.ndarray):
        self._rt_matrix = value

    @property
    def translation(self) -> np.ndarray:
        return self._rt_matrix[:3, 3]

    @translation.setter
    def translation(self, value: np.ndarray):
        self._rt_matrix[:3, 3] = value

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self.rt_matrix[:3, :3]

    @rotation_matrix.setter
    def rotation_matrix(self, value: np.ndarray):
        self.rt_matrix[:3, :3] = value

    @property
    def inverse(self) -> np.ndarray:

        inverse_rotation_matrix = np.transpose(self.rotation_matrix)
        inverse_translation = -inverse_rotation_matrix.reshape(3, 3) @ self.translation

        rt_matrix = np.zeros((4, 4))
        rt_matrix[:3, :3] = inverse_rotation_matrix.reshape(3, 3)
        rt_matrix[:3, 3] = inverse_translation.reshape(
            3,
        )
        rt_matrix[3, 3] = 1.0

        return rt_matrix
