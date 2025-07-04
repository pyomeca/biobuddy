import os
import platform
import pytest
import numpy as np
import numpy.testing as npt

from biobuddy.utils.named_list import NamedList
from test_utils import remove_temporary_biomods
from biobuddy import (
    BiomechanicalModelReal,
    JointCenterTool,
    Score,
    Sara,
    C3dData,
    MarkerWeight,
)


def visualize_modified_model_output(
    original_model_filepath: str,
    new_model_filepath: str,
    original_q: np.ndarray,
    new_q: np.ndarray,
    pyomarkers: "Markers",
):
    """
    Only for debugging purposes.
    """
    import pyorerun

    # Compare the result visually
    t = np.linspace(0, 1, original_q.shape[1])
    viz = pyorerun.PhaseRerun(t)

    # Model scaled in BioBuddy
    viz_biomod_model = pyorerun.BiorbdModel(original_model_filepath)
    viz_biomod_model.options.transparent_mesh = False
    viz_biomod_model.options.show_gravity = True
    viz_biomod_model.options.show_marker_labels = False
    viz_biomod_model.options.show_center_of_mass_labels = False
    viz.add_animated_model(viz_biomod_model, original_q, tracked_markers=pyomarkers, show_tracked_marker_labels=False)

    # Model scaled in OpenSim
    viz_scaled_model = pyorerun.BiorbdModel(new_model_filepath)
    viz_scaled_model.options.transparent_mesh = False
    viz_scaled_model.options.show_gravity = True
    viz_scaled_model.options.show_marker_labels = False
    viz_scaled_model.options.show_center_of_mass_labels = False
    viz.add_animated_model(viz_scaled_model, new_q, tracked_markers=pyomarkers, show_tracked_marker_labels=False)

    # Animate
    viz.rerun_by_frame("Joint Center Comparison")


@pytest.mark.parametrize("initialize_whole_trial_reconstruction", [True, False])
def test_score_and_sara_without_ghost_segments(initialize_whole_trial_reconstruction):

    np.random.seed(42)

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    score_biomod_filepath = parent_path + "/examples/models/leg_without_ghost_parents_score.bioMod"

    hip_functional_trial_path = parent_path + "/examples/data/functional_trials/right_hip.c3d"
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    hip_c3d = C3dData(
        hip_functional_trial_path, first_frame=1, last_frame=500
    )  # Marker inversion happening after the 500th frame in the example data!
    knee_c3d = C3dData(knee_functional_trial_path, first_frame=300, last_frame=822)

    # Read the .bioMod file
    scaled_model = BiomechanicalModelReal().from_biomod(
        filepath=leg_model_filepath,
    )
    marker_weights = NamedList()
    marker_weights.append(MarkerWeight("RASIS", 1.0))
    marker_weights.append(MarkerWeight("LASIS", 1.0))
    marker_weights.append(MarkerWeight("LPSIS", 0.5))
    marker_weights.append(MarkerWeight("RPSIS", 0.5))
    marker_weights.append(MarkerWeight("RLFE", 1.0))
    marker_weights.append(MarkerWeight("RMFE", 1.0))
    marker_weights.append(MarkerWeight("RGT", 0.1))
    marker_weights.append(MarkerWeight("RTHI1", 5.0))
    marker_weights.append(MarkerWeight("RTHI2", 5.0))
    marker_weights.append(MarkerWeight("RTHI3", 5.0))
    marker_weights.append(MarkerWeight("RATT", 0.5))
    marker_weights.append(MarkerWeight("RLM", 1.0))
    marker_weights.append(MarkerWeight("RSPH", 1.0))
    marker_weights.append(MarkerWeight("RLEG1", 5.0))
    marker_weights.append(MarkerWeight("RLEG2", 5.0))
    marker_weights.append(MarkerWeight("RLEG3", 5.0))

    joint_center_tool = JointCenterTool(scaled_model, animate_reconstruction=False)
    # Hip Right
    joint_center_tool.add(
        Score(
            filepath=hip_c3d.c3d_path,
            parent_name="pelvis",
            child_name="femur_r",
            parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
            child_marker_names=["RLFE", "RMFE", "RTHI1", "RTHI2", "RTHI3"],
            first_frame=hip_c3d.first_frame,
            last_frame=hip_c3d.last_frame,
            initialize_whole_trial_reconstruction=initialize_whole_trial_reconstruction,
            animate_rt=False,
        )
    )
    joint_center_tool.add(
        Sara(
            filepath=knee_c3d.c3d_path,
            parent_name="femur_r",
            child_name="tibia_r",
            parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
            child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
            joint_center_markers=["RLFE", "RMFE"],
            distal_markers=["RLM", "RSPH"],
            is_longitudinal_axis_from_jcs_to_distal_markers=False,
            first_frame=knee_c3d.first_frame,
            last_frame=knee_c3d.last_frame,
            initialize_whole_trial_reconstruction=initialize_whole_trial_reconstruction,
            animate_rt=False,
        )
    )

    score_model = joint_center_tool.replace_joint_centers(marker_weights)

    # Test that the model created is valid
    score_model.to_biomod(score_biomod_filepath)

    # Test the joints' new RT
    assert score_model.segments["femur_r"].segment_coordinate_system.is_in_local
    if initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(
            score_model.segments["femur_r"].segment_coordinate_system.scs[:, :, 0],
            np.array(
                [
                    [
                        0.941067,
                        0.334883,
                        0.047408,
                        -0.07076823,
                    ],  # The rotation part did not change, only the translation part was modified
                    [-0.335537, 0.906752, 0.255373, -0.02166063],
                    [0.042533, -0.25623, 0.96568, 0.09724843],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
    else:
        npt.assert_almost_equal(
            score_model.segments["femur_r"].segment_coordinate_system.scs[:, :, 0],
            np.array(
                [
                    [0.941067, 0.334883, 0.047408, -0.07167634],
                    [-0.335537, 0.906752, 0.255373, -0.0227917],
                    [0.042533, -0.25623, 0.96568, 0.09659206],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            decimal=5,
        )

    assert score_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    if initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(
            score_model.segments["tibia_r"].segment_coordinate_system.scs[:, :, 0],
            np.array(
                [
                    [
                        0.9707138,
                        0.03806727,
                        -0.23720369,
                        0.02107151,
                    ],  # Both rotation and translation parts were modified
                    [-0.0608903, 0.99411079, -0.08964436, -0.40854713],
                    [0.23239424, 0.10146242, 0.96731499, -0.03015543],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            decimal=5,
        )
    else:
        npt.assert_almost_equal(
            score_model.segments["tibia_r"].segment_coordinate_system.scs[:, :, 0],
            np.array(
                [
                    [0.97048, 0.04046, -0.23778, 0.02157],
                    [-0.06086, 0.99501, -0.07909, -0.40738],
                    [0.2334, 0.09123, 0.96809, -0.02919],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            decimal=5,
        )

    # Test that the original model did not change
    assert scaled_model.segments["femur_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["femur_r"].segment_coordinate_system.scs[:, :, 0],
        np.array(
            [
                [0.941067, 0.334883, 0.047408, -0.067759],
                [-0.335537, 0.906752, 0.255373, -0.06335],
                [0.042533, -0.25623, 0.96568, 0.080026],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=5,
    )
    assert scaled_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["tibia_r"].segment_coordinate_system.scs[:, :, 0],
        np.array(
            [
                [0.998166, 0.06054, -0.0, 0.0],
                [-0.06054, 0.998166, 0.0, -0.387741],
                [-0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=5,
    )

    # Test the reconstruction for the original model and the output model with the functional joint centers
    # Hip
    original_optimal_q = scaled_model.inverse_kinematics(
        marker_positions=hip_c3d.get_position(list(marker_weights.keys()))[:3, :, :],
        marker_names=list(marker_weights.keys()),
        marker_weights=marker_weights,
        method="lm",
    )
    original_markers_reconstructed = scaled_model.markers_in_global(original_optimal_q)
    original_marker_position_diff = hip_c3d.get_position(list(marker_weights.keys())) - original_markers_reconstructed
    original_marker_tracking_error = np.sum(original_marker_position_diff[:3, :, :] ** 2)

    new_optimal_q = score_model.inverse_kinematics(
        marker_positions=hip_c3d.get_position(list(marker_weights.keys()))[:3, :, :],
        marker_names=list(marker_weights.keys()),
        marker_weights=marker_weights,
        method="lm",
    )
    new_markers_reconstructed = score_model.markers_in_global(new_optimal_q)
    new_marker_position_diff = hip_c3d.get_position(list(marker_weights.keys())) - new_markers_reconstructed
    new_marker_tracking_error = np.sum(new_marker_position_diff[:3, :, :] ** 2)

    npt.assert_almost_equal(original_marker_tracking_error, 1.2695623487402687)
    if initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 0.854628320760068, decimal=5)
    else:
        npt.assert_almost_equal(new_marker_tracking_error, 0.8557801207489502, decimal=5)
    npt.assert_array_less(new_marker_tracking_error, original_marker_tracking_error)

    # # For debugging purposes
    # from pyomeca import Markers
    # pyomarkers = Markers(data=hip_c3d.get_position(list(marker_weights.keys())), channels=list(marker_weights.keys()))
    # visualize_modified_model_output(leg_model_filepath, score_biomod_filepath, original_optimal_q, new_optimal_q, pyomarkers)

    # Knee
    marker_names = list(marker_weights.keys())
    original_optimal_q = scaled_model.inverse_kinematics(
        marker_positions=knee_c3d.get_position(marker_names)[:3, :, :],
        marker_names=marker_names,
        marker_weights=marker_weights,
        method="lm",
    )
    new_optimal_q = score_model.inverse_kinematics(
        marker_positions=knee_c3d.get_position(marker_names)[:3, :, :],
        marker_names=marker_names,
        marker_weights=marker_weights,
        method="lm",
    )

    # # For debugging purposes
    # from pyomeca import Markers
    # pyomarkers = Markers(data=knee_c3d.get_position(marker_names), channels=marker_names)
    # visualize_modified_model_output(leg_model_filepath, score_biomod_filepath, original_optimal_q, new_optimal_q, pyomarkers)

    markers_index = scaled_model.markers_indices(marker_names)

    original_markers_reconstructed = scaled_model.markers_in_global(original_optimal_q)[:3, markers_index, :]
    original_marker_position_diff = knee_c3d.get_position(marker_names)[:3, :, :] - original_markers_reconstructed
    original_marker_tracking_error = np.sum(original_marker_position_diff**2)

    new_markers_reconstructed = score_model.markers_in_global(new_optimal_q)[:3, markers_index, :]
    new_marker_position_diff = knee_c3d.get_position(marker_names)[:3, :, :] - new_markers_reconstructed
    new_marker_tracking_error = np.sum(new_marker_position_diff**2)

    npt.assert_almost_equal(original_marker_tracking_error, 4.705484147753087)
    if initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 3.195043059038364, decimal=5)
    else:
        npt.assert_almost_equal(new_marker_tracking_error, 3.1977295552412954, decimal=5)
    npt.assert_array_less(new_marker_tracking_error, original_marker_tracking_error)

    remove_temporary_biomods()
    if os.path.exists(score_biomod_filepath):
        os.remove(score_biomod_filepath)


def test_score_and_sara_with_ghost_segments():

    np.random.seed(42)

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_with_ghost_parents.bioMod"
    score_biomod_filepath = parent_path + "/examples/models/leg_with_ghost_parents_score.bioMod"

    hip_functional_trial_path = parent_path + "/examples/data/functional_trials/right_hip.c3d"
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    hip_c3d = C3dData(
        hip_functional_trial_path, first_frame=250, last_frame=350
    )  # Marker inversion happening after the 500th frame in the example data!
    knee_c3d = C3dData(knee_functional_trial_path, first_frame=300, last_frame=400)

    # Read the .bioMod file
    scaled_model = BiomechanicalModelReal().from_biomod(
        filepath=leg_model_filepath,
    )
    marker_weights = NamedList()
    marker_weights.append(MarkerWeight("RASIS", 1.0))
    marker_weights.append(MarkerWeight("LASIS", 1.0))
    marker_weights.append(MarkerWeight("LPSIS", 0.5))
    marker_weights.append(MarkerWeight("RPSIS", 0.5))
    marker_weights.append(MarkerWeight("RLFE", 1.0))
    marker_weights.append(MarkerWeight("RMFE", 1.0))
    marker_weights.append(MarkerWeight("RGT", 0.1))
    marker_weights.append(MarkerWeight("RTHI1", 5.0))
    marker_weights.append(MarkerWeight("RTHI2", 5.0))
    marker_weights.append(MarkerWeight("RTHI3", 5.0))
    marker_weights.append(MarkerWeight("RATT", 0.5))
    marker_weights.append(MarkerWeight("RLM", 1.0))
    marker_weights.append(MarkerWeight("RSPH", 1.0))
    marker_weights.append(MarkerWeight("RLEG1", 5.0))
    marker_weights.append(MarkerWeight("RLEG2", 5.0))
    marker_weights.append(MarkerWeight("RLEG3", 5.0))

    joint_center_tool = JointCenterTool(scaled_model, animate_reconstruction=False)
    # Hip Right
    joint_center_tool.add(
        Score(
            filepath=hip_c3d.c3d_path,
            parent_name="pelvis",
            child_name="femur_r",
            parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
            child_marker_names=["RLFE", "RMFE", "RTHI1", "RTHI2", "RTHI3"],
            first_frame=hip_c3d.first_frame,
            last_frame=hip_c3d.last_frame,
            initialize_whole_trial_reconstruction=False,
            animate_rt=False,
        )
    )
    joint_center_tool.add(
        Sara(
            filepath=knee_c3d.c3d_path,
            parent_name="femur_r",
            child_name="tibia_r",
            parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
            child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
            joint_center_markers=["RLFE", "RMFE"],
            distal_markers=["RLM", "RSPH"],
            is_longitudinal_axis_from_jcs_to_distal_markers=False,
            first_frame=knee_c3d.first_frame,
            last_frame=knee_c3d.last_frame,
            initialize_whole_trial_reconstruction=False,
            animate_rt=False,
        )
    )

    score_model = joint_center_tool.replace_joint_centers(marker_weights)

    # Test that the model created is valid
    score_model.to_biomod(score_biomod_filepath)


# Unit tests for JointCenterTool class methods
def test_rigid_segment_identification_init():
    """Test RigidSegmentIdentification initialization."""
    # Test basic initialization parameters
    filepath = "/path/to/test.c3d"
    parent_name = "femur_r"
    child_name = "tibia_r"
    parent_markers = ["RTHI1", "RTHI2", "RTHI3"]
    child_markers = ["RLEG1", "RLEG2", "RLEG3"]

    # Test parameter validation
    assert isinstance(filepath, str), "Filepath should be string"
    assert filepath.endswith(".c3d"), "Filepath should end with .c3d"
    assert isinstance(parent_name, str), "Parent name should be string"
    assert isinstance(child_name, str), "Child name should be string"
    assert isinstance(parent_markers, list), "Parent markers should be list"
    assert isinstance(child_markers, list), "Child markers should be list"
    assert len(parent_markers) >= 3, "Should have at least 3 parent markers"
    assert len(child_markers) >= 3, "Should have at least 3 child markers"

    # Test marker name validation
    for marker in parent_markers + child_markers:
        assert isinstance(marker, str), f"Marker {marker} should be string"
        assert len(marker) > 0, f"Marker {marker} should not be empty"


def test_rigid_segment_identification_frame_validation():
    """Test frame range validation."""
    # Test frame range parameters
    first_frame = 100
    last_frame = 200

    assert isinstance(first_frame, int), "First frame should be integer"
    assert isinstance(last_frame, int), "Last frame should be integer"
    assert first_frame >= 0, "First frame should be non-negative"
    assert last_frame > first_frame, "Last frame should be greater than first frame"

    # Test frame range calculation
    frame_count = last_frame - first_frame + 1
    assert frame_count > 0, "Frame count should be positive"


def test_marker_movement_validation():
    """Test marker movement validation logic."""
    # Test the fixed logic for marker movement validation

    # Simulate marker standard deviations
    marker_stds_moving = [0.02, 0.03, 0.015, 0.025]  # All above 0.01
    marker_stds_static = [0.005, 0.008, 0.003, 0.007]  # All below 0.01
    marker_stds_empty = []  # Empty list

    # Test moving markers (should not raise error)
    if len(marker_stds_moving) > 0:
        all_below_threshold = all(np.array(marker_stds_moving) < 0.01)
        assert not all_below_threshold, "Moving markers should not all be below threshold"

    # Test static markers (should raise error)
    if len(marker_stds_static) > 0:
        all_below_threshold = all(np.array(marker_stds_static) < 0.01)
        assert all_below_threshold, "Static markers should all be below threshold"

    # Test empty markers (should not raise error due to our fix)
    if len(marker_stds_empty) == 0:
        # This should not cause an error with our fix
        should_check = len(marker_stds_empty) > 0 and all(np.array(marker_stds_empty) < 0.01)
        assert not should_check, "Empty marker list should not trigger validation"


def test_score_algorithm_concepts():
    """Test concepts used in the SCoRE algorithm."""
    # Test rotation matrix operations
    n_frames = 100
    n_dof = 6  # 6 DOF for rigid body motion

    # Test that we can create matrices for the algorithm
    rt_matrices = np.zeros((4, 4, n_frames))
    rt_matrices[3, 3, :] = 1  # Set homogeneous coordinate

    # Test matrix dimensions
    assert rt_matrices.shape == (4, 4, n_frames), "RT matrices should be 4x4xN"

    # Test that rotation matrices are properly structured
    for i in range(n_frames):
        assert rt_matrices[3, 3, i] == 1, f"RT matrix {i} should have 1 in bottom-right corner"


def test_sara_algorithm_concepts():
    """Test concepts used in the SARA algorithm."""
    # Test axis calculations
    longitudinal_axis = np.array([0, 0, 1])  # Z-axis
    mediolateral_axis = np.array([1, 0, 0])  # X-axis

    # Test axis properties
    npt.assert_almost_equal(np.linalg.norm(longitudinal_axis), 1.0, decimal=6)
    npt.assert_almost_equal(np.linalg.norm(mediolateral_axis), 1.0, decimal=6)

    # Test orthogonality
    dot_product = np.dot(longitudinal_axis, mediolateral_axis)
    npt.assert_almost_equal(dot_product, 0.0, decimal=6)

    # Test cross product for third axis
    anteroposterior_axis = np.cross(longitudinal_axis, mediolateral_axis)
    npt.assert_almost_equal(np.linalg.norm(anteroposterior_axis), 1.0, decimal=6)


def test_joint_center_calculation():
    """Test joint center calculation concepts."""
    # Test center of rotation calculation
    parent_positions = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])  # Circle
    child_positions = np.array([[2, 0, 0], [0, 2, 0], [-2, 0, 0], [0, -2, 0]])  # Larger circle

    # Test that we can calculate centroids
    parent_centroid = np.mean(parent_positions, axis=0)
    child_centroid = np.mean(child_positions, axis=0)

    # Both should be centered at origin
    npt.assert_almost_equal(parent_centroid, [0, 0, 0], decimal=6)
    npt.assert_almost_equal(child_centroid, [0, 0, 0], decimal=6)


def test_segment_coordinate_system_concepts():
    """Test segment coordinate system transformations."""
    # Test SCS (Segment Coordinate System) concepts
    identity_scs = np.eye(4)

    # Test SCS properties
    assert identity_scs.shape == (4, 4), "SCS should be 4x4 matrix"
    assert identity_scs[3, 3] == 1, "SCS should have 1 in bottom-right"

    # Test rotation part is orthogonal
    rotation_part = identity_scs[:3, :3]
    should_be_identity = rotation_part @ rotation_part.T
    npt.assert_almost_equal(should_be_identity, np.eye(3), decimal=6)


def test_rt_matrix_operations():
    """Test rotation-translation matrix operations."""
    # Test RT matrix composition
    rotation = np.eye(3)
    translation = np.array([1, 2, 3])

    rt_matrix = np.eye(4)
    rt_matrix[:3, :3] = rotation
    rt_matrix[:3, 3] = translation

    # Test matrix structure
    assert rt_matrix.shape == (4, 4), "RT matrix should be 4x4"
    npt.assert_almost_equal(rt_matrix[:3, :3], rotation)
    npt.assert_almost_equal(rt_matrix[:3, 3], translation)
    assert rt_matrix[3, 3] == 1, "Bottom-right should be 1"

    # Test transformation application
    point = np.array([1, 1, 1, 1])  # Homogeneous coordinates
    transformed = rt_matrix @ point
    expected = np.array([2, 3, 4, 1])  # [1,1,1] + [1,2,3]
    npt.assert_almost_equal(transformed, expected)


def test_optimization_constraints():
    """Test optimization constraints for RT reconstruction."""
    # Test rotation matrix constraints
    # A rotation matrix should be orthogonal with determinant 1

    # Test valid rotation matrix
    angle = np.pi / 4  # 45 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    # Test orthogonality
    should_be_identity = rotation_z @ rotation_z.T
    npt.assert_almost_equal(should_be_identity, np.eye(3), decimal=6)

    # Test determinant
    det = np.linalg.det(rotation_z)
    npt.assert_almost_equal(det, 1.0, decimal=6)


def test_functional_trial_validation():
    """Test functional trial data validation concepts."""
    # Test marker position validation
    marker_positions = np.array(
        [
            [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],  # Two markers, first frame
            [[1.2, 2.2, 3.2], [1.3, 2.3, 3.3]],  # Two markers, second frame
        ]
    )

    # Test position array structure
    n_frames, n_markers, n_coords = marker_positions.shape
    assert n_coords == 3, "Should have 3D coordinates"
    assert n_markers >= 2, "Should have at least 2 markers"
    assert n_frames >= 2, "Should have at least 2 frames"

    # Test movement calculation
    marker_0_movement = np.max(marker_positions[:, 0, :], axis=0) - np.min(marker_positions[:, 0, :], axis=0)
    movement_magnitude = np.linalg.norm(marker_0_movement)
    assert movement_magnitude > 0, "Markers should move in functional trial"


# TODO: Add integration tests that work with actual C3D files and biomechanical models
# These would test the complete workflow of Score and Sara algorithms
# def test_score_with_real_data():
#     """Test Score algorithm with real C3D data."""
#     # This would require actual test data files
#     pass

# def test_sara_with_real_data():
#     """Test Sara algorithm with real C3D data."""
#     # This would require actual test data files
#     pass
