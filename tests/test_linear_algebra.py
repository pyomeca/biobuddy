import numpy as np
import numpy.testing as npt
import pytest

import biorbd

from biobuddy import Rotations
from biobuddy.utils.linear_algebra import (
    rot_x_matrix,
    rot_y_matrix,
    rot_z_matrix,
    get_sequence_from_rotation_vector,
    get_vector_from_sequence,
    mean_homogenous_matrix,
    mean_unit_vector,
    to_euler,
    transpose_homogenous_matrix,
    norm2,
    unit_vector,
    compute_matrix_rotation,
    rot2eul,
    get_closest_rt_matrix,
    quaternion_to_rotation_matrix,
    coord_sys,
    ortho_norm_basis,
    is_ortho_basis,
    get_rt_aligning_markers_in_global,
    point_from_global_to_local,
    point_from_local_to_global,
    RotationMatrix,
    RotoTransMatrix,
    RotoTransMatrixTimeSeries,
    get_closest_rotation_matrix,
    local_rt_between_global_rts,
)


def test_rotation_matrices():
    """Test basic rotation matrices for known angles."""
    # Test identity (zero rotation)
    npt.assert_almost_equal(rot_x_matrix(0), np.eye(3))
    npt.assert_almost_equal(rot_y_matrix(0), np.eye(3))
    npt.assert_almost_equal(rot_z_matrix(0), np.eye(3))

    # Test 90 degree rotations
    angle_90 = np.pi / 2

    # X rotation by 90 degrees
    expected_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    npt.assert_almost_equal(rot_x_matrix(angle_90), expected_x)

    # Y rotation by 90 degrees
    expected_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    npt.assert_almost_equal(rot_y_matrix(angle_90), expected_y)

    # Z rotation by 90 degrees
    expected_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    npt.assert_almost_equal(rot_z_matrix(angle_90), expected_z)

    # Test 180 degree rotations
    angle_180 = np.pi

    # X rotation by 180 degrees
    expected_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    npt.assert_almost_equal(rot_x_matrix(angle_180), expected_x_180)

    # Y rotation by 180 degrees
    expected_y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    npt.assert_almost_equal(rot_y_matrix(angle_180), expected_y_180)

    # Z rotation by 180 degrees
    expected_z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    npt.assert_almost_equal(rot_z_matrix(angle_180), expected_z_180)

    # Test properties: all rotation matrices should be orthogonal
    for angle in [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
        rx = rot_x_matrix(angle)
        ry = rot_y_matrix(angle)
        rz = rot_z_matrix(angle)

        # Check orthogonality: R @ R.T = I
        npt.assert_almost_equal(rx @ rx.T, np.eye(3))
        npt.assert_almost_equal(ry @ ry.T, np.eye(3))
        npt.assert_almost_equal(rz @ rz.T, np.eye(3))

        # Check determinant = 1
        npt.assert_almost_equal(np.linalg.det(rx), 1.0)
        npt.assert_almost_equal(np.linalg.det(ry), 1.0)
        npt.assert_almost_equal(np.linalg.det(rz), 1.0)


def test_rotation_vector_sequences():
    """Test rotation vector and sequence conversion functions."""
    # Test get_vector_from_sequence
    npt.assert_almost_equal(get_vector_from_sequence("x"), np.array([1, 0, 0]))
    npt.assert_almost_equal(get_vector_from_sequence("y"), np.array([0, 1, 0]))
    npt.assert_almost_equal(get_vector_from_sequence("z"), np.array([0, 0, 1]))

    # Test invalid sequence
    with pytest.raises(RuntimeError, match="Rotation sequence .* not recognized"):
        get_vector_from_sequence("invalid")

    # Test get_sequence_from_rotation_vector
    assert get_sequence_from_rotation_vector(np.array([1, 0, 0])) == "x"
    assert get_sequence_from_rotation_vector(np.array([0, 1, 0])) == "y"
    assert get_sequence_from_rotation_vector(np.array([0, 0, 1])) == "z"

    # Test with 4D vector (should use first 3 components)
    assert get_sequence_from_rotation_vector(np.array([1, 0, 0, 5])) == "x"

    # Test invalid rotation vector
    with pytest.raises(RuntimeError, match="Rotation vector .* not recognized"):
        get_sequence_from_rotation_vector(np.array([1, 1, 0]))


def test_mean_homogenous_matrix():
    """Test mean homogeneous matrix computation."""
    # Create test matrices
    n_matrices = 5
    matrices = np.zeros((4, 4, n_matrices))

    for i in range(n_matrices):
        # Create slight variations of identity
        angle = 0.1 * i
        rot = rot_x_matrix(angle)
        trans = np.array([i, 0, 0])

        matrices[:3, :3, i] = rot
        matrices[:3, 3, i] = trans
        matrices[3, 3, i] = 1.0

    mean_matrix = mean_homogenous_matrix(matrices)

    # Check dimensions
    assert mean_matrix.shape == (4, 4)

    # Check homogeneous matrix structure
    npt.assert_almost_equal(mean_matrix[3, :], np.array([0, 0, 0, 1]))

    # Check that result is a valid transformation matrix
    npt.assert_almost_equal(mean_matrix[:3, :3] @ mean_matrix[:3, :3].T, np.eye(3), decimal=6)
    npt.assert_almost_equal(np.linalg.det(mean_matrix[:3, :3]), 1.0)

    # Test the translation values
    npt.assert_almost_equal(mean_matrix[:3, 3], np.array([2.0, 0.0, 0.0]))

    # Test the rotation values
    npt.assert_almost_equal(
        mean_matrix[:3, :3], np.array([[1.0, 0.0, 0.0], [0.0, 0.98006658, -0.19866933], [0.0, 0.19866933, 0.98006658]])
    )


def test_mean_unit_vector():
    """Test mean unit vector computation."""
    # Create test vectors
    n_vectors = 5
    vectors = np.zeros((4, n_vectors))

    for i in range(n_vectors):
        # Create unit vectors with slight variations
        angle = 0.1 * i
        vectors[:3, i] = np.array([np.cos(angle), np.sin(angle), 0])
        vectors[3, i] = 1.0

    mean_vector = mean_unit_vector(vectors)

    # Check dimensions
    assert mean_vector.shape == (4,)

    # Check that result is a unit vector
    npt.assert_almost_equal(np.linalg.norm(mean_vector[:3]), 1.0)

    # Check last component is 1
    assert mean_vector[3] == 1.0

    # Test error condition
    with pytest.raises(RuntimeError, match="The vectors must be of shape"):
        mean_unit_vector(np.array([[1, 2], [3, 4]]))

    # Test the mean vector value
    npt.assert_almost_equal(mean_vector, np.array([0.98006658, 0.19866933, 0.0, 1.0]))


def test_to_euler():
    """Test conversion from rotation matrix to Euler angles."""
    # Test with known angles
    angles = np.array([0.1, 0.2, 0.3])
    rt_matrix = RotoTransMatrix()
    rt_matrix.from_euler_angles_and_translation("xyz", angles, np.array([0, 0, 0]))
    rt = rt_matrix.rt_matrix

    # Check the rt matrix
    npt.assert_almost_equal(
        rt[:3, :3],
        np.array(
            [
                [0.93629336, -0.28962948, 0.19866933],
                [0.31299183, 0.94470249, -0.0978434],
                [-0.15934508, 0.153792, 0.97517033],
            ]
        ),
    )

    # Extract Euler angles
    extracted_angles = to_euler(rt, "xyz")

    # Check that we get back the original angles (within tolerance)
    npt.assert_almost_equal(extracted_angles, angles, decimal=5)

    # Test with identity matrix
    identity_rt = np.eye(4)
    zero_angles = to_euler(identity_rt, "xyz")
    npt.assert_almost_equal(zero_angles, np.array([0, 0, 0]), decimal=10)

    # Test error condition
    with pytest.raises(NotImplementedError, match="This angle_sequence is not implemented"):
        to_euler(np.eye(4), "zyx")


def test_norm2():
    """Test squared norm computation."""
    # Test with simple vectors
    v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    expected = np.array([1, 1, 1, 3])

    result = norm2(v)
    npt.assert_almost_equal(result, expected)

    # Test with single vector
    v_single = np.array([[3, 4]])
    expected_single = np.array([25])

    result_single = norm2(v_single)
    npt.assert_almost_equal(result_single, expected_single)


def test_unit_vector():
    """Test unit vector computation."""
    # Test with simple vector
    v = np.array([3, 4, 0])
    unit_v = unit_vector(v)

    # Check that result is unit vector
    npt.assert_almost_equal(np.linalg.norm(unit_v), 1.0)

    # Check direction is preserved
    npt.assert_almost_equal(unit_v, np.array([0.6, 0.8, 0]))

    # Test with already unit vector
    unit_input = np.array([1, 0, 0])
    result = unit_vector(unit_input)
    npt.assert_almost_equal(result, unit_input)


def test_compute_matrix_rotation():
    """Test rotation matrix computation from XYZ angles."""
    # Test with zero rotations
    rot_values = np.array([0, 0, 0])
    result = compute_matrix_rotation(rot_values)
    npt.assert_almost_equal(result, np.eye(3))

    # Test with known angles
    rot_values = np.array([np.pi / 2, 0, 0])
    result = compute_matrix_rotation(rot_values)
    expected = rot_x_matrix(np.pi / 2)
    npt.assert_almost_equal(result, expected)

    # Test properties
    rot_values = np.array([0.1, 0.2, 0.3])
    result = compute_matrix_rotation(rot_values)

    # Check orthogonality
    npt.assert_almost_equal(result @ result.T, np.eye(3))

    # Check determinant
    npt.assert_almost_equal(np.linalg.det(result), 1.0)


def test_rot2eul():
    """Test rotation matrix to Euler angles conversion."""
    # Test with known rotation matrix
    rot_values = np.array([0.1, 0.2, 0.3])
    rot_matrix = compute_matrix_rotation(rot_values)

    # Extract Euler angles
    extracted_angles = rot2eul(rot_matrix)

    # Check that we get back similar angles (rotation order may differ)
    # At least check that it's a valid conversion
    assert len(extracted_angles) == 3
    assert all(np.isfinite(extracted_angles))

    # Test with identity matrix
    identity_angles = rot2eul(np.eye(3))
    npt.assert_almost_equal(identity_angles, np.array([0, 0, 0]), decimal=10)


def test_get_closest_rt_matrix():
    """Test projection to closest rotation matrix."""
    # Test with already valid rotation matrix
    valid_rt = np.eye(4)
    result = get_closest_rt_matrix(valid_rt)
    npt.assert_almost_equal(result, valid_rt)

    # Test with slightly invalid rotation matrix
    invalid_rt = np.eye(4)
    invalid_rt[:3, :3] = np.array([[1.01, 0, 0], [0, 0.99, 0], [0, 0, 1.0]])

    result = get_closest_rt_matrix(invalid_rt)

    # Check that result is valid
    assert result.shape == (4, 4)
    npt.assert_almost_equal(result[3, :], np.array([0, 0, 0, 1]))
    npt.assert_almost_equal(result[:3, :3], np.eye(3), decimal=6)
    npt.assert_almost_equal(np.linalg.det(result[:3, :3]), 1.0)

    # Test with slightly invalid rotation matrix with inverted axis
    invalid_rt = np.eye(4)
    invalid_rt[:3, :3] = np.array([[0, 0, 1.0], [0, 0.99, 0], [1.01, 0, 0]])

    result = get_closest_rt_matrix(invalid_rt)

    # Check that result is valid
    assert result.shape == (4, 4)
    npt.assert_almost_equal(result[3, :], np.array([0, 0, 0, 1]))
    npt.assert_almost_equal(result[:3, :3], np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]), decimal=6)
    npt.assert_almost_equal(np.linalg.det(result[:3, :3]), 1.0)

    # Test with NaN values
    nan_rt = np.eye(4)
    nan_rt[0, 0] = np.nan
    result = get_closest_rt_matrix(nan_rt)
    assert np.all(np.isnan(result))

    # Test error conditions
    with pytest.raises(RuntimeError, match="far from SO\\(3\\)"):
        invalid_rt = np.eye(4)
        invalid_rt[:3, :3] = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        get_closest_rt_matrix(invalid_rt)

    with pytest.raises(RuntimeError, match="Check rt matrix"):
        invalid_rt = np.eye(4)
        invalid_rt[3, :] = np.array([1, 0, 0, 1])
        get_closest_rt_matrix(invalid_rt)


def test_quaternion_to_rotation_matrix():
    """Test quaternion to rotation matrix conversion."""
    # Test with identity quaternion
    quat_scalar = 1.0
    quat_vector = np.array([0, 0, 0])

    result = quaternion_to_rotation_matrix(quat_scalar, quat_vector)
    npt.assert_almost_equal(result, np.eye(3))

    # Test with 90-degree rotation around x-axis
    quat_scalar = np.cos(np.pi / 4)
    quat_vector = np.array([np.sin(np.pi / 4), 0, 0])

    result = quaternion_to_rotation_matrix(quat_scalar, quat_vector)
    expected = rot_x_matrix(np.pi / 2)
    npt.assert_almost_equal(result, expected, decimal=6)

    # Test properties
    quat_scalar = 0.5
    quat_vector = np.array([0.5, 0.5, 0.5])

    result = quaternion_to_rotation_matrix(quat_scalar, quat_vector)

    # Check orthogonality
    npt.assert_almost_equal(result @ result.T, np.eye(3), decimal=6)

    # Check determinant
    npt.assert_almost_equal(np.linalg.det(result), 1.0)


def test_coord_sys():
    """Test orthonormal coordinate system generation."""
    # Test with standard axes
    axes, label = coord_sys([0, 0, 1])
    expected_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    npt.assert_almost_equal(axes, expected_axes)
    assert label == "z"

    axes, label = coord_sys([0, 1, 0])
    expected_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    npt.assert_almost_equal(axes, expected_axes)
    assert label == "y"

    axes, label = coord_sys([1, 0, 0])
    expected_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    npt.assert_almost_equal(axes, expected_axes)
    assert label == "x"

    # Test with arbitrary axis
    axes, label = coord_sys([1, 1, 1])

    # Check that we get three vectors
    assert len(axes) == 3

    # Check orthogonality
    x, y, z = axes
    npt.assert_almost_equal(np.dot(x, y), 0, decimal=10)
    npt.assert_almost_equal(np.dot(y, z), 0, decimal=10)
    npt.assert_almost_equal(np.dot(x, z), 0, decimal=10)

    # Check unit vectors
    npt.assert_almost_equal(np.linalg.norm(x), 1.0)
    npt.assert_almost_equal(np.linalg.norm(y), 1.0)
    npt.assert_almost_equal(np.linalg.norm(z), 1.0)


def test_ortho_norm_basis():
    """Test orthonormal basis generation."""
    # Set random seed for reproducible results
    np.random.seed(42)

    # Test with standard vector
    vector = np.array([1, 0, 0])

    for idx in range(3):
        basis = ortho_norm_basis(vector, idx)

        # Check dimensions
        assert basis.shape == (3, 3)

        # Check orthogonality - all dot products should be zero
        for i in range(3):
            for j in range(i + 1, 3):
                npt.assert_almost_equal(np.dot(basis[i, :], basis[j, :]), 0, decimal=10)

        # Check unit vectors - all rows should have norm 1
        for i in range(3):
            npt.assert_almost_equal(np.linalg.norm(basis[i, :]), 1.0)

        # Check right-handed coordinate system - determinant should be 1
        npt.assert_almost_equal(np.linalg.det(basis), 1.0, decimal=10)

        # For idx=0, the input vector should be the first basis vector
        normalized_vector = vector / np.linalg.norm(vector)
        if idx == 0:
            npt.assert_almost_equal(basis[0, :], normalized_vector)

    # Test with different vector
    vector2 = np.array([0, 1, 0])
    basis2 = ortho_norm_basis(vector2, 0)

    # Check that we get an orthonormal basis
    assert basis2.shape == (3, 3)
    for i in range(3):
        for j in range(i + 1, 3):
            npt.assert_almost_equal(np.dot(basis2[i, :], basis2[j, :]), 0, decimal=10)
    for i in range(3):
        npt.assert_almost_equal(np.linalg.norm(basis2[i, :]), 1.0)
    npt.assert_almost_equal(np.linalg.det(basis2), 1.0, decimal=10)


def test_is_ortho_basis():
    """Test orthogonal basis checking."""
    # Test with orthogonal basis
    basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert is_ortho_basis(basis) == True

    # Test with non-orthogonal basis
    basis = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    assert is_ortho_basis(basis) == False

    # Test with another orthogonal basis
    basis = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2), 0], [-1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 1]])
    assert is_ortho_basis(basis) == True


def test_get_rt_aligning_markers_in_global():
    """Test alignment of markers between coordinate systems."""
    # Create simple test case
    # Local markers in a square
    local_centered = np.array([[1, -1, 1, -1], [1, 1, -1, -1], [0, 0, 0, 0]])

    local_centroid = np.array([0, 0, 0])

    # Global markers - same square but rotated and translated
    angle = np.pi / 4
    rot_matrix = rot_z_matrix(angle)
    translation = np.array([5, 10, 15])

    markers_in_global = rot_matrix @ local_centered + translation.reshape(3, 1)

    # Test alignment
    rt_matrix = get_rt_aligning_markers_in_global(markers_in_global, local_centered, local_centroid)

    # Check dimensions
    assert rt_matrix.shape == (4, 4)

    # Check homogeneous matrix structure
    npt.assert_almost_equal(rt_matrix[3, :], np.array([0, 0, 0, 1]))

    # Check that the transformation aligns the markers correctly
    transformed_local = rt_matrix[:3, :3] @ local_centered + rt_matrix[:3, 3:4]
    npt.assert_almost_equal(transformed_local, markers_in_global[:3, :], decimal=10)


def test_rototrans_matrix_class():
    """Test RotoTransMatrix class."""
    # Test initialization from rotation matrix and translation
    rotation_matrix = rot_x_matrix(np.pi / 4)
    translation = np.array([1, 2, 3])

    rt = RotoTransMatrix()
    rt.from_rotation_matrix_and_translation(rotation_matrix, translation)

    # Check properties
    npt.assert_almost_equal(rt.rotation_matrix, rotation_matrix)
    npt.assert_almost_equal(rt.translation, translation)

    # Check full matrix
    expected_rt = np.eye(4)
    expected_rt[:3, :3] = rotation_matrix
    expected_rt[:3, 3] = translation
    npt.assert_almost_equal(rt.rt_matrix, expected_rt)

    # Test initialization from Euler angles
    angles = np.array([np.pi / 6, np.pi / 4, np.pi / 3])
    angle_sequence = "xyz"

    rt2 = RotoTransMatrix()
    rt2.from_euler_angles_and_translation(angle_sequence, angles, translation)

    # Check dimensions
    assert rt2.rt_matrix.shape == (4, 4)
    npt.assert_almost_equal(rt2.translation, translation)

    # Test initialization from rt matrix
    rt_matrix = np.eye(4)
    rt_matrix[:3, :3] = rotation_matrix
    rt_matrix[:3, 3] = translation

    rt3 = RotoTransMatrix()
    rt3.from_rt_matrix(rt_matrix)

    npt.assert_almost_equal(rt3.rt_matrix, rt_matrix)

    # Test inverse
    inverse_rt = rt3.inverse.rt_matrix

    # Check that inverse is correct
    npt.assert_almost_equal(rt3.rt_matrix @ inverse_rt, np.eye(4))

    # Test Euler angles extraction
    angles_extracted = rt2.euler_angles("xyz")
    assert len(angles_extracted) == 3

    # Test the initialization with nan
    rt_matrix[0, 0] = np.nan
    rt4 = RotoTransMatrix()
    rt4.from_rt_matrix(rt_matrix)
    npt.assert_equal(rt4.rt_matrix, np.eye(4) * np.nan)

    # Test error conditions
    with pytest.raises(ValueError, match="should be of shape \\(3, 3\\)"):
        rt.from_rotation_matrix_and_translation(np.eye(2), translation)

    with pytest.raises(ValueError, match="should be of shape \\(3,\\)"):
        rt.from_rotation_matrix_and_translation(rotation_matrix, np.array([1, 2]))

    with pytest.raises(ValueError, match="should be of shape \\(nb_angles, \\)"):
        rt.from_euler_angles_and_translation("xyz", np.array([[1, 2, 3]]), translation)

    with pytest.raises(ValueError, match="should be of shape \\(3,\\)"):
        rt.from_euler_angles_and_translation("xyz", angles, np.array([1, 2]))

    with pytest.raises(ValueError, match="must match"):
        rt.from_euler_angles_and_translation("xyz", np.array([1, 2]), translation)

    with pytest.raises(ValueError, match="should be of shape \\(4, 4\\)"):
        rt.from_rt_matrix(np.eye(3))


def test_rototrans_matrix_time_series():
    """Test RotoTransMatrixTimeSeries class."""
    # Create test data
    n_frames = 5
    rotation_matrices = np.zeros((3, 3, n_frames))
    translations = np.zeros((3, n_frames))

    for i in range(n_frames):
        angle = i * 0.1
        rotation_matrices[:, :, i] = rot_x_matrix(angle)
        translations[:, i] = np.array([i, 0, 0])

    # Test initialization
    rt_series = RotoTransMatrixTimeSeries(n_frames)
    rt_series.from_rotation_matrix_and_translation(rotation_matrices, translations)

    # Check that we can access individual frames
    for i in range(n_frames):
        rt_frame = rt_series[i]
        npt.assert_almost_equal(rt_frame.rotation_matrix, rotation_matrices[:, :, i])
        npt.assert_almost_equal(rt_frame.translation, translations[:, i])

    # Test initialization from rt matrices
    rt_matrices = np.zeros((4, 4, n_frames))
    for i in range(n_frames):
        rt_matrices[:3, :3, i] = rotation_matrices[:, :, i]
        rt_matrices[:3, 3, i] = translations[:, i]
        rt_matrices[3, 3, i] = 1.0

    rt_series2 = RotoTransMatrixTimeSeries(n_frames)
    rt_series2.from_rt_matrix(rt_matrices)

    # Check that we get the same results
    for i in range(n_frames):
        rt_frame = rt_series2[i]
        npt.assert_almost_equal(rt_frame.rt_matrix, rt_matrices[:, :, i])

    # Test the initialization with nan
    rt_matrices[0, 0, 2] = np.nan
    rt_series3 = RotoTransMatrixTimeSeries(n_frames)
    rt_series3.from_rt_matrix(rt_matrices)
    npt.assert_equal(rt_series3[2].rt_matrix, np.eye(4) * np.nan)
    npt.assert_equal(rt_series3[3].rt_matrix, rt_matrices[:, :, 3])

    # Test mean_homogenous_matrix method
    mean_rt = rt_series.mean_homogenous_matrix()
    assert isinstance(mean_rt, RotoTransMatrix)

    # Test get_rt_matrix method
    rt_matrices_result = rt_series.get_rt_matrix()
    assert rt_matrices_result.shape == (4, 4, n_frames)
    for i in range(n_frames):
        if i != 2:  # Skip the NaN frame
            npt.assert_almost_equal(rt_matrices_result[:, :, i], rt_series[i].rt_matrix)

    # Test error conditions
    with pytest.raises(ValueError, match="should be of shape"):
        rt_series.from_rotation_matrix_and_translation(np.eye(3), np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="should be of shape"):
        rt_series.from_rt_matrix(np.eye(4))


def test_rt():

    np.random.seed(42)

    for angle_sequence in Rotations:
        if angle_sequence != Rotations.NONE:
            nb_angles = len(angle_sequence.value)
            angles = np.random.rand(nb_angles) * 2 * np.pi
            translations = np.random.rand(3)

            rt_biobuddy = RotoTransMatrix()
            rt_biobuddy.from_euler_angles_and_translation(
                angles=angles, angle_sequence=angle_sequence.value, translation=translations
            )

            rot_biobuddy = rt_biobuddy.rotation_matrix

            rotation_matrix_biorbd = biorbd.Rotation.fromEulerAngles(angles, angle_sequence.value)
            rot_biorbd = rotation_matrix_biorbd.to_array()

            npt.assert_almost_equal(
                rot_biobuddy,
                rot_biorbd,
            )
            npt.assert_almost_equal(translations, rt_biobuddy.translation)

            # --- Euler angles from rotation matrix --- #
            if angle_sequence == Rotations.XYZ:
                angles_biobuddy = rt_biobuddy.euler_angles(angle_sequence=angle_sequence.value)
                angles_biorbd = biorbd.Rotation.toEulerAngles(rotation_matrix_biorbd, angle_sequence.value).to_array()

                npt.assert_almost_equal(
                    angles_biobuddy,
                    angles_biorbd,
                )
            else:
                with pytest.raises(NotImplementedError, match="This angle_sequence is not implemented yet"):
                    angles_biobuddy = rt_biobuddy.euler_angles(angle_sequence=angle_sequence.value)


def test_negative_determinant_matrices():
    """Test handling of matrices with negative determinants."""
    # Test with negative determinant matrix for get_closest_rotation_matrix
    neg_det_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    result = get_closest_rotation_matrix(neg_det_rot)

    # Check that result has positive determinant
    npt.assert_almost_equal(np.linalg.det(result), 1.0)

    # Test with NaN values for get_closest_rotation_matrix
    nan_rot = np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = get_closest_rotation_matrix(nan_rot)
    assert np.all(np.isnan(result))

    # Test with negative determinant matrix for get_closest_rt_matrix
    neg_det_rt = np.eye(4)
    neg_det_rt[:3, :3] = neg_det_rot
    result_rt = get_closest_rt_matrix(neg_det_rt)

    # Check that result has positive determinant
    npt.assert_almost_equal(np.linalg.det(result_rt[:3, :3]), 1.0)


def test_rotation_matrix_class():
    """Test RotationMatrix class."""
    # Test initialization from rotation matrix
    rotation = rot_x_matrix(np.pi / 4)
    rot_obj = RotationMatrix()
    rot_obj.from_rotation_matrix(rotation)

    # Check properties
    npt.assert_almost_equal(rot_obj.rotation_matrix, rotation)

    # Test initialization from Euler angles
    angles = np.array([np.pi / 6, np.pi / 4, np.pi / 3])
    angle_sequence = "xyz"

    rot_obj2 = RotationMatrix()
    rot_obj2.from_euler_angles(angle_sequence, angles)
    npt.assert_almost_equal(
        rot_obj2.rotation_matrix,
        np.array(
            [
                [0.35355339, -0.61237244, 0.70710678],
                [0.9267767, 0.12682648, -0.35355339],
                [0.12682648, 0.78033009, 0.61237244],
            ]
        ),
    )
    revert_to_euler_angles = rot_obj2.euler_angles(angle_sequence)
    npt.assert_almost_equal(revert_to_euler_angles, angles, decimal=5)

    # Test rotation matrix setter
    new_rotation = rot_y_matrix(np.pi / 3)
    rot_obj.rotation_matrix = new_rotation
    npt.assert_almost_equal(rot_obj.rotation_matrix, new_rotation)

    # Test inverse
    inverse_rot = rot_obj.inverse

    # Check that inverse is correct
    npt.assert_almost_equal(rot_obj.rotation_matrix @ inverse_rot.rotation_matrix, np.eye(3))

    # Test matrix multiplication
    mult_result = rot_obj @ rot_obj2
    assert isinstance(mult_result, RotationMatrix)
    npt.assert_almost_equal(mult_result.rotation_matrix, rot_obj.rotation_matrix @ rot_obj2.rotation_matrix)

    # Test vector multiplication
    vector = np.array([1, 2, 3])
    vector_result = rot_obj @ vector
    npt.assert_almost_equal(vector_result[:, 0], rot_obj.rotation_matrix @ vector)

    # Test error conditions
    with pytest.raises(ValueError, match="should be of shape \\(3, 3\\)"):
        rot_obj.from_rotation_matrix(np.eye(2))

    with pytest.raises(ValueError, match="should be of shape \\(nb_angles, \\)"):
        rot_obj.from_euler_angles("xyz", np.array([[1, 2, 3]]))

    with pytest.raises(ValueError, match="must match"):
        rot_obj.from_euler_angles("xyz", np.array([1, 2]))

    with pytest.raises(NotImplementedError):
        rot_obj @ "invalid_type"

    with pytest.raises(ValueError):
        rot_obj @ np.eye(3)


def test_point_from_global_to_local():
    point_in_global = np.array([0.1, 0.1, 0.1])
    jcs_in_global = RotoTransMatrix()
    jcs_in_global.from_rt_matrix(
        np.array([[1.0, 0.0, 0.0, 0.1], [0.0, 0.0, -1.0, 0.1], [0.0, 1.0, 0.0, 0.1], [0.0, 0.0, 0.0, 1.0]])
    )

    point_in_local = point_from_global_to_local(point_in_global, jcs_in_global)
    npt.assert_almost_equal(point_in_local, np.array([[0.0], [0.0], [0.0], [1.0]]))


def test_transpose_homogenous_matrix():
    """Test transpose of homogeneous matrix for 3D arrays."""
    # Create test data - 3D array of homogeneous matrices
    n_frames = 3
    matrices = np.zeros((4, 4, n_frames))

    for i in range(n_frames):
        angle = i * 0.1
        rotation = rot_x_matrix(angle)
        translation = np.array([i, 0, 0])

        matrices[:3, :3, i] = rotation
        matrices[:3, 3, i] = translation
        matrices[3, 3, i] = 1.0

    # Test transpose
    transposed = transpose_homogenous_matrix(matrices)

    # Check dimensions
    assert transposed.shape == (4, 4, n_frames)

    # Check that each frame is properly transposed
    for i in range(n_frames):
        # For a homogeneous matrix, the transpose should be the inverse
        # RT^T = [R^T  -R^T*t]
        #        [0    1     ]
        original = matrices[:, :, i]
        trans = transposed[:, :, i]

        # Check that multiplying gives identity
        product = original @ trans
        npt.assert_almost_equal(product, np.eye(4), decimal=10)

        # Check structure
        npt.assert_almost_equal(trans[3, :3], np.zeros(3))
        npt.assert_almost_equal(trans[3, 3], 1.0)

    # Test the values for the second frame
    npt.assert_almost_equal(
        transposed[:, :, 1],
        np.array(
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 0.99500417, 0.09983342, 0.0],
                [0.0, -0.09983342, 0.99500417, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_point_transformations():
    """Test point coordinate transformations."""
    # Create a simple transformation matrix
    angle = np.pi / 4
    rotation = rot_z_matrix(angle)
    translation = np.array([1, 2, 3])

    rt_matrix = RotoTransMatrix()
    rt_matrix.from_rotation_matrix_and_translation(rotation, translation)

    # Test point
    point_global = np.array([5, 6, 7])

    # Transform to local coordinates
    point_local = point_from_global_to_local(point_global, rt_matrix)

    # Check dimensions
    assert point_local.shape == (4, 1)
    assert point_local[3, 0] == 1.0

    # Transform back to global coordinates
    point_global_back = point_from_local_to_global(point_local, rt_matrix)

    # Check that we get back the original point
    npt.assert_almost_equal(point_global_back[:3, 0], point_global)
    assert point_global_back[3, 0] == 1.0

    # Test with 4D point
    point_global_4d = np.array([5, 6, 7, 1])
    point_local_4d = point_from_global_to_local(point_global_4d, rt_matrix)
    npt.assert_almost_equal(point_local_4d, point_local)


def test_local_rt_between_global_rts():
    """Test computation of local RT between two global RTs."""
    # Create parent RT in global
    parent_rotation = rot_x_matrix(np.pi / 6)
    parent_translation = np.array([1, 2, 3])
    parent_rt = RotoTransMatrix()
    parent_rt.from_rotation_matrix_and_translation(parent_rotation, parent_translation)

    # Create child RT in global
    child_rotation = rot_z_matrix(np.pi / 4)
    child_translation = np.array([4, 5, 6])
    child_rt = RotoTransMatrix()
    child_rt.from_rotation_matrix_and_translation(child_rotation, child_translation)

    # Compute local RT
    local_rt = local_rt_between_global_rts(parent_rt, child_rt)

    # Check that local_rt transforms parent to child correctly
    computed_child_rt = parent_rt @ local_rt
    npt.assert_almost_equal(computed_child_rt.rt_matrix, child_rt.rt_matrix)

    # Test with identity matrices
    identity_rt = RotoTransMatrix()
    identity_rt.from_rt_matrix(np.eye(4))

    # Identity parent, arbitrary child
    local_rt1 = local_rt_between_global_rts(identity_rt, child_rt)
    npt.assert_almost_equal(local_rt1.rt_matrix, child_rt.rt_matrix)

    # Arbitrary parent, identity child
    local_rt2 = local_rt_between_global_rts(parent_rt, identity_rt)
    npt.assert_almost_equal(local_rt2.rt_matrix, parent_rt.inverse.rt_matrix)

    # Test error case with invalid input types
    with pytest.raises(TypeError, match="must be instances of RotoTransMatrix"):
        local_rt_between_global_rts(np.eye(4), child_rt)

    # Test with 4D point
    point_global_4d = np.array([5, 6, 7, 1])
    point_local_4d = point_from_global_to_local(point_global_4d, child_rt)
    npt.assert_almost_equal(point_local_4d, child_rt.inverse @ point_global_4d)


def test_roto_trans_matrix():

    # Test coord_sys with zero vector
    axes, label = coord_sys([0, 0, 0])
    expected_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    npt.assert_almost_equal(axes, expected_axes)
    assert label == ""

    # Test quaternion_to_rotation_matrix with invalid quaternion
    # Should raise error for non-unit quaternion that results in bad rotation
    quat_scalar = 2.0  # This will make the norm > 1
    quat_vector = np.array([1, 1, 1])

    # This should still work but might trigger the error check
    try:
        result = quaternion_to_rotation_matrix(quat_scalar, quat_vector)
        # If it doesn't raise an error, check that it's still a valid rotation matrix
        npt.assert_almost_equal(result @ result.T, np.eye(3), decimal=6)
    except RuntimeError as e:
        # Expected for invalid quaternions
        assert "rotation matrix computed does not lie in SO(3)" in str(e)

    # Test RotoTransMatrix with 3D rt matrix input
    rt_3d = np.eye(4).reshape(4, 4, 1)
    rt_obj = RotoTransMatrix()
    rt_obj.from_rt_matrix(rt_3d)
    npt.assert_almost_equal(rt_obj.rt_matrix, np.eye(4))

    # Test setters for RotoTransMatrix
    rt_obj2 = RotoTransMatrix()
    rt_obj2.from_rt_matrix(np.eye(4))

    # Test translation setter
    new_translation = np.array([1, 2, 3])
    rt_obj2.translation = new_translation
    npt.assert_almost_equal(rt_obj2.translation, new_translation)

    # Test rotation matrix setter
    new_rotation = rot_x_matrix(np.pi / 4)
    rt_obj2.rotation_matrix = new_rotation
    npt.assert_almost_equal(rt_obj2.rotation_matrix, new_rotation)

    # Test the rt_matrix
    rt_expected = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.70710678, -0.70710678, 2.0],
            [0.0, 0.70710678, 0.70710678, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    npt.assert_almost_equal(rt_obj2.rt_matrix, rt_expected)

    # Test the multiplication by a matrix
    expected_identity = rt_obj2.inverse @ rt_obj2
    npt.assert_almost_equal(expected_identity.rt_matrix, np.eye(4))
    npt.assert_almost_equal(rt_obj2.inverse.rt_matrix, np.linalg.inv(rt_expected))
    mult_res = rt_obj2 @ rt_obj2
    npt.assert_almost_equal(mult_res.rt_matrix, rt_expected @ rt_expected)

    # Test the multiplication by a vector
    point_4D = np.array([0.01, 0.2, 3.3, 1.0])
    mult_res = rt_obj2 @ point_4D
    npt.assert_almost_equal(
        mult_res.reshape(
            4,
        ),
        rt_expected @ point_4D,
    )
    point_3D = np.array([0.01, 0.2, 3.3])
    mult_res = rt_obj2 @ point_3D
    npt.assert_almost_equal(
        mult_res.reshape(
            4,
        ),
        rt_expected @ point_4D,
    )

    # Test is_identity method
    identity_rt = RotoTransMatrix()
    identity_rt.from_rt_matrix(np.eye(4))
    assert identity_rt.is_identity == True

    non_identity_rt = RotoTransMatrix()
    non_identity_rt.from_rt_matrix(rt_expected)
    assert non_identity_rt.is_identity == False
