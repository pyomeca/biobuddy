from .aliases import point_to_array, Point

import numpy as np


# TODO: Charbie -> uniformization !!!! (angle_sequence: Rotations enum, RototransMatrix everywhere)


def rot_x_matrix(angle):
    """
    Rotation matrix around the x-axis
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def rot_y_matrix(angle):
    """
    Rotation matrix around the y-axis
    """
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def rot_z_matrix(angle):
    """
    Rotation matrix around the z-axis
    """
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def get_rotation_vector_from_sequence(sequence: str):
    if sequence == "x":
        rotation_vector = np.array([1, 0, 0])
    elif sequence == "y":
        rotation_vector = np.array([0, 1, 0])
    elif sequence == "z":
        rotation_vector = np.array([0, 0, 1])
    else:
        raise RuntimeError(f"Rotation sequence {sequence} not recognized. Please use 'x', 'y', or 'z'.")
    return rotation_vector


def get_sequence_from_rotation_vector(rotation_vector: np.ndarray):
    if np.all(np.abs(rotation_vector[:3] - np.array([1, 0, 0])) < 1e-6):
        sequence = "x"
    elif np.all(np.abs(rotation_vector[:3] - np.array([0, 1, 0])) < 1e-6):
        sequence = "y"
    elif np.all(np.abs(rotation_vector[:3] - np.array([0, 0, 1])) < 1e-6):
        sequence = "z"
    else:
        raise RuntimeError(
            f"Rotation vector {rotation_vector} not recognized. Please use np.array([1, 0, 0]), np.array([0, 1, 0]), or np.array([0, 0, 1])."
        )
    return sequence


def euler_and_translation_to_matrix(
    angles: np.ndarray,
    angle_sequence: str,
    translations: np.ndarray,
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
    """

    if len(angles.shape) > 1 and angles.shape[1] != 1:
        raise RuntimeError("The angles must be a vector of size (number or rotations x 1).")
    if len(angle_sequence) != angles.shape[0]:
        raise RuntimeError("The number of angles must be equal to the number of angles in the sequence.")
    if len(translations.shape) > 1 and translations.shape[1] != 1:
        raise RuntimeError("The translations must be a vector of size (number or translations x 1).")

    matrix = {
        "x": rot_x_matrix,
        "y": rot_y_matrix,
        "z": rot_z_matrix,
    }
    rt = np.identity(4)
    for angle, axis in zip(angles, angle_sequence):
        rt[:3, :3] = rt[:3, :3] @ matrix[axis](angle)
    rt[:3, 3] = translations[:3]

    return rt


def mean_homogenous_matrix(matrices: np.ndarray) -> np.ndarray:
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
    arithmetic_mean_scs = np.nanmean(matrices, axis=2)
    mean_matrix[:3, 3] = arithmetic_mean_scs[:3, 3]

    # Get minimized rotation matrix from the svd decomposition
    u, s, v = np.linalg.svd(arithmetic_mean_scs[:3, :3])
    mean_matrix[:3, :3] = u @ v
    return mean_matrix


def mean_unit_vector(vectors: np.ndarray) -> np.ndarray:
    """
    Computes the mean unit vector from a set of unit vectors
    """
    if vectors.shape[0] != 4 or len(vectors.shape) != 2:
        raise RuntimeError(
            "The vectors must be of shape (4, n). Only the three first components will be averaged (the last component is a 1)."
        )

    mean_vector = np.ones((4,))
    mean_vector[:3] = np.nanmean(vectors[:3, :], axis=1)
    mean_vector[:3] /= np.linalg.norm(mean_vector[:3])
    return mean_vector


def to_euler(rt, angle_sequence: str) -> np.ndarray:
    if angle_sequence == "xyz":
        rx = np.arctan2(-rt[1, 2], rt[2, 2])
        ry = np.arcsin(rt[0, 2])
        rz = np.arctan2(-rt[0, 1], rt[0, 0])
    else:
        raise NotImplementedError("This angle_sequence is not implemented yet")

    return np.array([rx, ry, rz])


def transpose_homogenous_matrix(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix).transpose((1, 0, 2))
    out[:3, 3, :] = np.einsum("ijk,jk->ik", -out[:3, :3, :], matrix[:3, 3, :])
    out[3, :3, :] = 0
    return out


def norm2(v) -> np.ndarray:
    """Compute the squared norm of each row of the matrix v."""
    return np.sum(v**2, axis=1)


def unit_vector(x):
    """Returns a unit vector"""
    return x / np.linalg.norm(x)


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


def get_closest_rt_matrix(rt_matrix: np.ndarray) -> np.ndarray:
    """
    Projects a rotation matrix to the closest rotation matrix using Singular Value Decomposition (SVD).
    """
    if np.abs(np.sum(rt_matrix[:3, :3] ** 2) - 3.0) > 0.1:
        raise RuntimeError(f"The rotation matrix {rt_matrix[:3, :3]} is far from SO(3).")
    if np.abs(np.linalg.norm(rt_matrix[3, :]) - 1) > 0.1:
        raise RuntimeError(f"Check rt matrix: the bottom line is {rt_matrix[3, :]} and should be [0, 0, 0, 1].")

    if np.abs(np.sum(rt_matrix[:3, :3] ** 2) - 3.0) < 1e-6:
        return rt_matrix

    else:
        output_rt = np.identity(4)
        output_rt[:3, 3] = rt_matrix[:3, 3]

        u, _, vt = np.linalg.svd(rt_matrix[:3, :3])
        projected_rot_matrix = u @ vt

        # Ensure det(R) = +1
        if np.linalg.det(projected_rot_matrix) < 0:
            u[:, -1] *= -1
            projected_rot_matrix = u @ vt

        output_rt[:3, :3] = projected_rot_matrix
        return output_rt


def quaternion_to_rotation_matrix(quat_scalar: float, quat_vector: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion to a 4x4 homogeneous rotation matrix.

    Parameters
    ----------
    quat_scalar
        The real part of the quaternion
    quat_vector: shape (3, )
    The imaginary vertor of the quaternion

    Returns
    -------
    rt_matrix : ndarray, shape (4, 4)
        Homogeneous transformation matrix (rotation only, no translation)
    """
    qw = quat_scalar
    qx, qy, qz = quat_vector

    rot_matrix = np.array(
        [
            [1.0 - 2.0 * qy**2 - 2.0 * qz**2, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw],
            [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx**2 - 2.0 * qz**2, 2.0 * qy * qz - 2.0 * qx * qw],
            [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx**2 - 2.0 * qy**2],
        ]
    )

    if np.abs(3 - np.linalg.norm(rot_matrix) ** 2) > 1e-6:
        raise RuntimeError("Something went wrong, the rotation matrix computed does not lie in SO(3).")

    return rot_matrix


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
        if np.abs(np.dot(basis[0], basis[1])) > 1e-8
        or np.abs(np.dot(basis[1], basis[2])) > 1e-8
        or np.abs(np.dot(basis[0], basis[2])) > 1e-8
        else True
    )


def get_rt_aligning_markers_in_global(
    markers_in_global: np.ndarray, local_centered: np.ndarray, local_centroid: np.ndarray
) -> np.ndarray:

    markers_in_global = markers_in_global[:3, :]
    local_centered = local_centered[:3, :]
    local_centroid = local_centroid[:3]

    global_centroid = np.mean(markers_in_global, axis=1, keepdims=True)
    global_centered = markers_in_global - global_centroid

    # Cross-covariance matrix
    H = global_centered @ local_centered.T

    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        # Reflection correction
        Vt[-1, :] *= -1
        rotation = U @ Vt

    translation = global_centroid.squeeze() - rotation @ local_centroid.squeeze()

    rt_matrix = np.identity(4)
    rt_matrix[:3, :3] = rotation[:3, :3]
    rt_matrix[:3, 3] = translation[:3]

    return rt_matrix


def point_from_global_to_local(point_in_global: np.ndarray, jcs_in_global: np.ndarray) -> np.ndarray:
    rt_matrix = RotoTransMatrix()
    rt_matrix.from_rt_matrix(jcs_in_global)
    return rt_matrix.inverse @ point_to_array(point=point_in_global)


def point_from_local_to_global(point_in_local: np.ndarray, jcs_in_global: np.ndarray) -> np.ndarray:
    return jcs_in_global @ point_in_local


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
        return np.all(self.get_matrix() == np.identity(4))


class RotoTransMatrix:
    def __init__(self):
        self._rt = np.identity(4)

    def __matmul__(self, other: "Self" | Point) -> "Self" | Point:
        if isinstance(other, RotoTransMatrix):
            # Matrix multiplication of two RotoTransMatrix objects gives a new RotoTransMatrix object
            mult_result = self.rt_matrix @ other.rt_matrix
            out = RotoTransMatrix()
            out.from_rt_matrix(mult_result)
        elif isinstance(other, np.ndarray):
            # Matrix multiplication of a RotoTransMatrix with a Point (np.array vector) gives a Point (np.array vector)
            if other.shape == (4, 4):
                raise ValueError(
                    "You seem to be trying to multiply two RotoTransMatrix objects. Please use RotoTransMatrix @ RotoTransMatrix instead."
                )
            out = self.rt_matrix @ point_to_array(point=other)
        else:
            raise NotImplementedError(
                f"The multiplication of RotoTransMatrix with {type(other)} is not implemented yet."
            )
        return out

    def from_rotation_matrix_and_translation(self, rotation_matrix: np.ndarray, translation: Point):
        if rotation_matrix.shape != (3, 3):
            raise ValueError(
                f"The rotation_matrix used to initialize a RotoTransMatrix should be of shape (3, 3). You have {rotation_matrix.shape}"
            )
        if translation.shape != (3,) and translation.shape != (4,):
            raise ValueError(
                f"The translation used to initialize a RotoTransMatrix should be of shape (3,) or (4,). You have {translation.shape}"
            )
        if np.abs(np.linalg.det(rotation_matrix) - 1.0) > 1e-6:
            raise ValueError(
                f"The rotation matrix provided {rotation_matrix} is not a valid rotation matrix (det = {np.linalg.det(rotation_matrix)}, and should be 1.0)."
            )

        rt_matrix = np.zeros((4, 4))
        rt_matrix[:3, :3] = rotation_matrix[:3, :3]
        rt_matrix[:3, 3] = translation[:3]
        rt_matrix[3, 3] = 1.0
        self._rt = rt_matrix

    def from_euler_angles_and_translation(self, angle_sequence: str, angles: np.ndarray, translation: np.ndarray):
        if len(angles.shape) > 1:
            raise ValueError(
                f"The angles used to initialize a RotoTransMatrix should be of shape (nb_angles, ). You have {angles.shape}"
            )
        if translation.shape != (3,) and translation.shape != (4,):
            raise ValueError(
                f"The translation used to initialize a RotoTransMatrix should be of shape (3,) or (4, ). You have {translation.shape}"
            )
        if len(angle_sequence) != angles.shape[0]:
            raise ValueError(
                f"The number of angles and the length of the angle_sequence must match. You have {angles.shape} and {angle_sequence}"
            )

        matrix = {
            "x": rot_x_matrix,
            "y": rot_y_matrix,
            "z": rot_z_matrix,
        }

        rt_matrix = np.identity(4)
        for angle, axis in zip(angles, angle_sequence):
            rt_matrix[:3, :3] = rt_matrix[:3, :3] @ matrix[axis](angle)
        rt_matrix[:3, 3] = translation[:3]
        self._rt = rt_matrix

    def from_rt_matrix(self, rt: np.ndarray):
        if rt.shape == (4, 4, 1):
            rt = rt[:, :, 0]
        elif rt.shape != (4, 4):
            raise ValueError(
                f"The rt used to initialize a RotoTransMatrix should be of shape (4, 4). You have {rt.shape}"
            )
        # # TODO: Charbie -> Uncomment when the bug in OpenSIm is fixed
        # if np.abs(np.linalg.det(rt[:3, :3]) - 1.0) > 1e-6:
        #     raise ValueError(f"The rotation matrix provided {rt} is not a valid rototranslation matrix (det of the rotation part = {np.linalg.det(rt[:3, :3])}, and should be 1.0).")
        self._rt = rt

    @property
    def rt_matrix(self) -> np.ndarray:
        return self._rt

    @property
    def translation(self) -> np.ndarray:
        return self._rt[:3, 3]

    @translation.setter
    def translation(self, trans: np.ndarray):
        self._rt[:3, 3] = trans

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self._rt[:3, :3]

    @rotation_matrix.setter
    def rotation_matrix(self, rot: np.ndarray):
        if rot.shape != (3, 3):
            raise ValueError(
                f"The rotation_matrix used to set a RotoTransMatrix should be of shape (3, 3). You have {rot.shape}"
            )
        if np.abs(np.linalg.det(rot) - 1.0) > 1e-6:
            raise ValueError(
                f"The rotation matrix provided {rot} is not a valid rotation matrix (det = {np.linalg.det(rot)}, and should be 1.0)."
            )
        self._rt[:3, :3] = rot

    def euler_angles(self, angle_sequence: str) -> np.ndarray:
        return to_euler(self.rotation_matrix, angle_sequence)

    @property
    def inverse(self) -> "Self":

        inverse_rotation_matrix = np.transpose(self.rotation_matrix)
        inverse_translation = -inverse_rotation_matrix.reshape(3, 3) @ self.translation

        rt_matrix = np.zeros((4, 4))
        rt_matrix[:3, :3] = inverse_rotation_matrix.reshape(3, 3)
        rt_matrix[:3, 3] = inverse_translation.reshape(
            3,
        )
        rt_matrix[3, 3] = 1.0

        out_inverse = RotoTransMatrix()
        out_inverse.from_rt_matrix(rt_matrix)
        return out_inverse


class RotoTransMatrixTimeSeries:
    """
    This class is a list of nb_frames RotoTranMatrix so that it is possible to define a RotoTranMatrix for each frame.
    """

    def __init__(self, nb_frames: int):
        self._rt_time_series = [RotoTransMatrix() for _ in range(nb_frames)]

    def __getitem__(self, index: int):
        return self._rt_time_series[index]

    def __setitem__(self, index: int, value: "RotoTransMatrix"):
        self._rt_time_series[index] = value

    def __len__(self):
        return len(self._rt_time_series)

    def from_rotation_matrix_and_translation(self, rotation_matrix: np.ndarray, translation: np.ndarray):
        if len(rotation_matrix.shape) != 3 or len(translation) != 3:
            raise ValueError(
                f"The rotation_matrix and translation used to initialize a RotoTransMatrixTimeSeries should be of shape (..., nb_frames). You have {rotation_matrix.shape} and {translation.shape}"
            )
        if rotation_matrix.shape[2] != len(self):
            raise ValueError(
                f"You must provide as many rotation matrices as the number of frames ({len(self)}). You have {rotation_matrix.shape[2]} rotation matrices."
            )

        rt_time_series = []
        for i_frame in range(rotation_matrix.shape[2]):
            rt_matrix = RotoTransMatrix()
            rt_matrix.from_rotation_matrix_and_translation(rotation_matrix[:, :, i_frame], translation[:, i_frame])
            rt_time_series += [rt_matrix]

        self._rt_time_series = rt_time_series

    def from_rt_matrix(self, rt: np.ndarray):
        if len(rt.shape) != 3:
            raise ValueError(
                f"The rt used to initialize a RotoTransMatrixTimeSeries should be of shape (..., nb_frames). You have {rt.shape}"
            )
        if rt.shape[2] != len(self):
            raise ValueError(
                f"You must provide as many rt matrices as the number of frames ({len(self)}). You have {rt.shape[2]} rotation matrices."
            )

        rt_time_series = []
        for i_frame in range(rt.shape[2]):
            rt_matrix = RotoTransMatrix()
            rt_matrix.from_rt_matrix(rt[:, :, i_frame])
            rt_time_series += [rt_matrix]

        self._rt_time_series = rt_time_series

    def mean_homogenous_matrix(self) -> RotoTransMatrix:
        """
        Computes the closest homogenous matrix that approximates all the homogenous matrices in the time series

        Returns
        -------
        The mean homogenous matrix
        """
        matrices = np.zeros((4, 4, len(self._rt_time_series)))
        for i_frame, rt in enumerate(self._rt_time_series):
            matrices[:, :, i_frame] = rt.rt_matrix
        mean_rt = mean_homogenous_matrix(matrices)
        out_rt = RotoTransMatrix()
        out_rt.from_rt_matrix(mean_rt)
        return out_rt

    def get_rt_matrix(self) -> np.ndarray:
        """
        Returns the RotoTransMatrix as a 3D numpy array of shape (4, 4, nb_frames)
        """
        rt_matrices = np.zeros((4, 4, len(self._rt_time_series)))
        for i_frame, rt in enumerate(self._rt_time_series):
            rt_matrices[:, :, i_frame] = rt.rt_matrix
        return rt_matrices
