import os
from copy import deepcopy
import pytest
import numpy as np
import numpy.testing as npt

from biobuddy.utils.named_list import NamedList
from test_utils import remove_temporary_biomods, create_simple_model, MockEmptyC3dData
from biobuddy import (
    BiomechanicalModelReal,
    JointCenterTool,
    Score,
    Sara,
    C3dData,
    MarkerWeight,
    RotoTransMatrix,
)
from biobuddy.model_modifiers.joint_center_tool import RigidSegmentIdentification


def visualize_modified_model_output(
    original_model_filepath: str,
    new_model_filepath: str,
    original_q: np.ndarray,
    new_q: np.ndarray,
    pyomarkers,
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
    viz.add_animated_model(viz_biomod_model, original_q, tracked_markers=pyomarkers)

    # Model scaled in OpenSim
    viz_scaled_model = pyorerun.BiorbdModel(new_model_filepath)
    viz_scaled_model.options.transparent_mesh = False
    viz_scaled_model.options.show_gravity = True
    viz_scaled_model.options.show_marker_labels = False
    viz_scaled_model.options.show_center_of_mass_labels = False
    viz.add_animated_model(viz_scaled_model, new_q, tracked_markers=pyomarkers)

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
            functional_c3d=hip_c3d,
            parent_name="pelvis",
            child_name="femur_r",
            parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
            child_marker_names=["RLFE", "RMFE", "RTHI1", "RTHI2", "RTHI3"],
            initialize_whole_trial_reconstruction=initialize_whole_trial_reconstruction,
            animate_rt=False,
        )
    )
    joint_center_tool.add(
        Sara(
            functional_c3d=knee_c3d,
            parent_name="femur_r",
            child_name="tibia_r",
            parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
            child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
            joint_center_markers=["RLFE", "RMFE"],
            distal_markers=["RLM", "RSPH"],
            is_longitudinal_axis_from_jcs_to_distal_markers=False,
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
            # Both rotation and translation parts were modified
            np.array(
                [
                    [0.97298983, 0.03790546, -0.22771465, 0.02107151],
                    [-0.0612039, 0.99348445, -0.09613912, -0.40854714],
                    [0.22258677, 0.10747942, 0.96897023, -0.03015542],
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
                    [0.97197658, 0.0418383, -0.23132462, 0.02157429],
                    [-0.06106338, 0.99519176, -0.07658086, -0.40738262],
                    [0.22700834, 0.08856027, 0.96985787, -0.02918892],
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

    npt.assert_almost_equal(original_marker_tracking_error, 1.2695623487402687, decimal=5)
    if initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 0.8533901218909357, decimal=5)
    else:
        npt.assert_almost_equal(new_marker_tracking_error, 0.8546461146170594, decimal=5)
    npt.assert_array_less(new_marker_tracking_error, original_marker_tracking_error)

    # # For debugging purposes
    # from pyorerun import PyoMarkers
    # pyomarkers = PyoMarkers(data=hip_c3d.get_position(list(marker_weights.keys())), channels=list(marker_weights.keys()), show_labels=False)
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
    # from pyorerun import PyoMarkers
    # pyomarkers = PyoMarkers(data=knee_c3d.get_position(marker_names), channels=marker_names, show_labels=False)
    # visualize_modified_model_output(leg_model_filepath, score_biomod_filepath, original_optimal_q, new_optimal_q, pyomarkers)

    markers_index = scaled_model.markers_indices(marker_names)

    original_markers_reconstructed = scaled_model.markers_in_global(original_optimal_q)[:3, markers_index, :]
    original_marker_position_diff = knee_c3d.get_position(marker_names)[:3, :, :] - original_markers_reconstructed
    original_marker_tracking_error = np.sum(original_marker_position_diff**2)

    new_markers_reconstructed = score_model.markers_in_global(new_optimal_q)[:3, markers_index, :]
    new_marker_position_diff = knee_c3d.get_position(marker_names)[:3, :, :] - new_markers_reconstructed
    new_marker_tracking_error = np.sum(new_marker_position_diff**2)

    npt.assert_almost_equal(original_marker_tracking_error, 4.705484147753087, decimal=5)
    if initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 3.1653655932067504, decimal=5)
    else:
        npt.assert_almost_equal(new_marker_tracking_error, 3.1621464045718777, decimal=5)
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
            functional_c3d=hip_c3d,
            parent_name="pelvis",
            child_name="femur_r",
            parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
            child_marker_names=["RLFE", "RMFE", "RTHI1", "RTHI2", "RTHI3"],
            initialize_whole_trial_reconstruction=False,
            animate_rt=False,
        )
    )
    joint_center_tool.add(
        Sara(
            functional_c3d=knee_c3d,
            parent_name="femur_r",
            child_name="tibia_r",
            parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
            child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
            joint_center_markers=["RLFE", "RMFE"],
            distal_markers=["RLM", "RSPH"],
            is_longitudinal_axis_from_jcs_to_distal_markers=False,
            initialize_whole_trial_reconstruction=False,
            animate_rt=False,
        )
    )

    score_model = joint_center_tool.replace_joint_centers(marker_weights)

    # Test that the model created is valid
    score_model.to_biomod(score_biomod_filepath)

    # Test the joints' new RT
    assert score_model.segments["femur_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["femur_r_parent_offset"].segment_coordinate_system.scs[:, :, 0],
        np.array(
            [
                [1.0, 0.0, 0.0, -0.0360743],
                [0.0, 1.0, 0.0, -0.03499795],
                [0.0, 0.0, 1.0, -0.01135702],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=5,
    )
    assert score_model.segments["femur_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["femur_r"].segment_coordinate_system.scs[:, :, 0],
        np.array(
            [
                [1.0, -0.0, 0.0, -0.0316847],
                [-0.0, 1.0, 0.0, -0.02835205],
                [0.0, 0.0, 1.0, 0.09138302],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=5,
    )

    assert score_model.segments["tibia_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["tibia_r_parent_offset"].segment_coordinate_system.scs[:, :, 0],
        np.array(
            [
                [0.9415693, 0.13609885, -0.30809797, 0.00539646],
                [-0.1586161, 0.98611693, -0.04913589, -0.38268824],
                [0.29713329, 0.09513415, 0.95008489, -0.00965356],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=5,
    )

    assert score_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["tibia_r"].segment_coordinate_system.scs[:, :, 0],
        np.array([[1.0, -0.0, 0.0, 0.0], [-0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )

    # Test that the original model did not change
    assert scaled_model.segments["femur_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["femur_r_parent_offset"].segment_coordinate_system.scs[:, :, 0],
        np.array(
            [[1.0, 0.0, 0.0, -0.067759], [0.0, 1.0, 0.0, -0.06335], [0.0, 0.0, 1.0, 0.080026], [0.0, 0.0, 0.0, 1.0]]
        ),
    )
    assert scaled_model.segments["femur_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["femur_r"].segment_coordinate_system.scs[:, :, 0],
        np.array([[1.0, -0.0, 0.0, 0.0], [-0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    assert scaled_model.segments["tibia_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["tibia_r_parent_offset"].segment_coordinate_system.scs[:, :, 0],
        np.array([[1.0, -0.0, 0.0, 0.0], [-0.0, 1.0, 0.0, -0.387741], [0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    assert scaled_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["tibia_r"].segment_coordinate_system.scs[:, :, 0],
        np.array([[1.0, -0.0, 0.0, 0.0], [-0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
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

    # The error is worse because it is a small test (for the tests to run quickly)
    npt.assert_almost_equal(original_marker_tracking_error, 0.28717883184190574, decimal=5)
    npt.assert_almost_equal(new_marker_tracking_error, 1.1144893588689808, decimal=5)

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

    markers_index = scaled_model.markers_indices(marker_names)

    original_markers_reconstructed = scaled_model.markers_in_global(original_optimal_q)[:3, markers_index, :]
    original_marker_position_diff = knee_c3d.get_position(marker_names)[:3, :, :] - original_markers_reconstructed
    original_marker_tracking_error = np.sum(original_marker_position_diff**2)

    new_markers_reconstructed = score_model.markers_in_global(new_optimal_q)[:3, markers_index, :]
    new_marker_position_diff = knee_c3d.get_position(marker_names)[:3, :, :] - new_markers_reconstructed
    new_marker_tracking_error = np.sum(new_marker_position_diff**2)

    # The error is worse because it is a unit test (for the tests to run quickly)
    npt.assert_almost_equal(original_marker_tracking_error, 9.674445375391658, decimal=5)
    npt.assert_almost_equal(new_marker_tracking_error, 9.853176510568787, decimal=5)

    remove_temporary_biomods()
    if os.path.exists(score_biomod_filepath):
        os.remove(score_biomod_filepath)


# Test Rigid Segment Identification:
def test_init_rigid_segment_identification():

    # Set up
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    c3d_data = C3dData(knee_functional_trial_path, first_frame=300, last_frame=400)
    mock_model = create_simple_model()

    # create JointCenterTool instance
    joint_center_tool = JointCenterTool(mock_model)

    # Create a test instance
    parent_name = "femur_r"
    child_name = "tibia_r"
    parent_marker_names = ["RGT", "RTHI1", "RTHI2", "RTHI3"]
    child_marker_names = ["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"]
    rsi = RigidSegmentIdentification(
        c3d_data,
        parent_name,
        child_name,
        parent_marker_names,
        child_marker_names,
    )

    # Test with valid names
    rsi._check_segment_names()  # Should not raise an error

    # Test with invalid names
    with pytest.raises(RuntimeError, match="The names _reset_axis are not allowed in the parent or child names. Please change the segment named parent_reset_axis from the Score configuration."):
        RigidSegmentIdentification(
            c3d_data,
            "parent_reset_axis",
            child_name,
            parent_marker_names,
            child_marker_names,
        )

    with pytest.raises(RuntimeError, match="The names _translation are not allowed in the parent or child names. Please change the segment named child_translation from the Score configuration."):
        RigidSegmentIdentification(
            c3d_data,
            parent_name,
            "child_translation",
            parent_marker_names,
            child_marker_names,
        )

    # Test with valid marker movement
    rsi._check_c3d_functional_trial_file()  # Should not raise an error

    # Test with no markers
    empty_c3d_data = deepcopy(c3d_data)
    empty_c3d_data.all_marker_positions = np.empty((4, empty_c3d_data.nb_markers, empty_c3d_data.nb_frames))
    rsi_no_markers = RigidSegmentIdentification(
        MockEmptyC3dData(),
        parent_name,
        child_name,
        parent_marker_names,
        child_marker_names,
    )
    with pytest.raises(RuntimeError, match="There are no markers in the functional trial"):
        rsi_no_markers._check_c3d_functional_trial_file()

    # Test with no movement
    c3d_data.std_marker_position.return_value = [0.005, 0.005]  # All < 0.01
    with pytest.raises(RuntimeError, match="are not moving in the functional trial"):
        rsi._check_c3d_functional_trial_file()

def test_marker_residual():
    # Create test data
    optimal_rt = np.eye(4).flatten()
    static_markers_in_local = np.ones((4, 2))  # 4D, 2 markers
    functional_markers_in_global = np.ones((4, 2))  # 4D, 2 markers

    # When RT is identity and markers match, residual should be 0
    residual = rsi.marker_residual(
        optimal_rt, static_markers_in_local, functional_markers_in_global
    )
    assert residual == 0

    # When markers don't match, residual should be positive
    functional_markers_in_global = np.ones((4, 2)) * 2
    residual = rsi.marker_residual(
        optimal_rt, static_markers_in_local, functional_markers_in_global
    )
    assert residual > 0

def test_get_good_frames():
    # Create test residuals
    residuals = np.array([1.0, 1.1, 1.2, 5.0, 1.3])  # One outlier at index 3
    nb_frames = len(residuals)

    # Test frame filtering
    valid_frames = rsi.get_good_frames(residuals, nb_frames)
    assert np.sum(valid_frames) == 4  # Should remove one frame
    assert not valid_frames[3]  # The outlier should be removed

def test_rt_constraints():
    # Test with a valid rotation matrix (orthonormal)
    rt_matrix = np.eye(4)
    constraints = rsi.rt_constraints(rt_matrix.flatten())
    assert np.allclose(constraints, np.zeros(6))

    # Test with an invalid rotation matrix
    rt_matrix = np.eye(4)
    rt_matrix[0, 0] = 2.0  # Make it non-orthonormal
    constraints = rsi.rt_constraints(rt_matrix.flatten())
    assert not np.allclose(constraints, np.zeros(6))

def test_check_optimal_rt_inputs():
    # Create valid test data
    markers = np.random.rand(3, 2, 10)  # 3D, 2 markers, 10 frames
    markers = np.vstack((markers, np.ones((1, 2, 10))))  # Add homogeneous coordinate
    static_markers = np.random.rand(3, 2)  # 3D, 2 markers
    static_markers = np.vstack((static_markers, np.ones((1, 2))))  # Add homogeneous coordinate
    marker_names = ["marker1", "marker2"]

    # Test with valid inputs
    result = rsi.check_optimal_rt_inputs(markers, static_markers, marker_names)
    assert result is not None
    assert len(result) == 3

    # Test with mismatched marker names
    with pytest.raises(RuntimeError, match="do not match the number of markers"):
        rsi.check_optimal_rt_inputs(markers, static_markers, ["marker1"])

    # Test with marker movement
    # Make markers move significantly between static and functional
    static_markers[0, 0] = 0
    markers[0, 0, :] = 1.0  # Large difference in position
    with pytest.raises(RuntimeError, match="seem to move during the functional trial"):
        rsi.check_optimal_rt_inputs(markers, static_markers, marker_names)

def test_remove_offset_from_optimal_rt():
    # Create a mock model
    model = MagicMock(spec=BiomechanicalModelReal)

    # Test without parent offset
    model.has_parent_offset.return_value = False
    rt_parent = np.random.rand(4, 4, 5)  # 4x4 RT matrices for 5 frames
    rt_child = np.random.rand(4, 4, 5)

    result_parent, result_child, _ = rsi.remove_offset_from_optimal_rt(model, rt_parent, rt_child)
    assert np.array_equal(result_parent, rt_parent)
    assert np.array_equal(result_child, rt_child)

    # Test with parent offset
    model.has_parent_offset.return_value = True
    offset_rt = RotoTransMatrix()
    offset_rt.from_rt_matrix(np.eye(4))
    model.rt_from_parent_offset_to_real_segment.return_value = offset_rt

    result_parent, result_child, _ = rsi.remove_offset_from_optimal_rt(model, rt_parent, rt_child)
    assert not np.array_equal(result_parent, rt_parent)  # Should be different with offset


# Test SCoRE
def setup_method():
    # Create a mock C3dData object
    c3d_data = MagicMock(spec=C3dData)
    c3d_data.marker_names = ["marker1", "marker2", "marker3", "marker4"]
    c3d_data.all_marker_positions = np.random.rand(3, 4, 10)  # 3D, 4 markers, 10 frames
    c3d_data.std_marker_position.return_value = [0.02, 0.03]  # Return values > 0.01 to pass check
    c3d_data.get_position.return_value = np.random.rand(4, 2, 10)  # 4D, 2 markers, 10 frames

    # Create a test instance
    parent_name = "parent"
    child_name = "child"
    parent_marker_names = ["marker1", "marker2"]
    child_marker_names = ["marker3", "marker4"]
    score = Score(
        c3d_data,
        parent_name,
        child_name,
        parent_marker_names,
        child_marker_names,
    )

def test_score_algorithm():
    # Create test RT matrices
    nb_frames = 5
    rt_parent = np.zeros((4, 4, nb_frames))
    rt_child = np.zeros((4, 4, nb_frames))

    # Set up RT matrices for a known center of rotation
    cor_parent_local = np.array([0.1, 0.2, 0.3])
    cor_child_local = np.array([0.4, 0.5, 0.6])

    for i in range(nb_frames):
        # Create rotation matrices (identity for simplicity)
        rt_parent[:3, :3, i] = np.eye(3)
        rt_child[:3, :3, i] = np.eye(3)

        # Set translations to place the CoR at the same global position
        rt_parent[:3, 3, i] = np.array([i*0.1, i*0.1, i*0.1])
        rt_child[:3, 3, i] = rt_parent[:3, 3, i] + cor_parent_local - cor_child_local

        # Set homogeneous coordinate
        rt_parent[3, 3, i] = 1
        rt_child[3, 3, i] = 1

    # Run the SCoRE algorithm
    cor_global, cor_parent, cor_child, _, _ = score._score_algorithm(
        rt_parent, rt_child, recursive_outlier_removal=False
    )

    # Check that the estimated CoR is close to the expected values
    assert np.allclose(cor_parent, cor_parent_local, atol=1e-2)
    assert np.allclose(cor_child, cor_child_local, atol=1e-2)


# Test SARA
def setup_method():
    # Create a mock C3dData object
    c3d_data = MagicMock(spec=C3dData)
    c3d_data.marker_names = ["marker1", "marker2", "marker3", "marker4", "marker5", "marker6"]
    c3d_data.all_marker_positions = np.random.rand(3, 6, 10)  # 3D, 6 markers, 10 frames
    c3d_data.std_marker_position.return_value = [0.02, 0.03]  # Return values > 0.01 to pass check
    c3d_data.get_position.return_value = np.random.rand(4, 2, 10)  # 4D, 2 markers, 10 frames

    # Create a test instance
    parent_name = "parent"
    child_name = "child"
    parent_marker_names = ["marker1", "marker2"]
    child_marker_names = ["marker3", "marker4"]
    joint_center_markers = ["marker5"]
    distal_markers = ["marker6"]
    sara = Sara(
        c3d_data,
        parent_name,
        child_name,
        parent_marker_names,
        child_marker_names,
        joint_center_markers,
        distal_markers,
        is_longitudinal_axis_from_jcs_to_distal_markers=True,
    )

def test_sara_algorithm():
    # Create test RT matrices for a known axis of rotation
    nb_frames = 5
    rt_parent = np.zeros((4, 4, nb_frames))
    rt_child = np.zeros((4, 4, nb_frames))

    # Define axis of rotation in local frames
    aor_parent_local = np.array([0, 0, 1])  # Z-axis
    aor_child_local = np.array([0, 0, 1])  # Z-axis

    for i in range(nb_frames):
        # Create rotation matrices (identity for simplicity)
        rt_parent[:3, :3, i] = np.eye(3)
        rt_child[:3, :3, i] = np.eye(3)

        # Set translations
        rt_parent[:3, 3, i] = np.array([0, 0, 0])
        rt_child[:3, 3, i] = np.array([0.1, 0.1, 0])  # Offset in X and Y, but not Z

        # Set homogeneous coordinate
        rt_parent[3, 3, i] = 1
        rt_child[3, 3, i] = 1

    # Run the SARA algorithm
    aor_global, _, _ = sara._sara_algorithm(
        rt_parent, rt_child, recursive_outlier_removal=False
    )

    # The estimated axis should be close to the Z-axis in global frame
    mean_aor = np.mean(aor_global, axis=1)
    mean_aor = mean_aor / np.linalg.norm(mean_aor)
    assert np.allclose(mean_aor, np.array([0, 0, 1]), atol=1e-2)

def test_longitudinal_axis():
    # Create a mock model
    model = MagicMock(spec=BiomechanicalModelReal)

    # Set up the forward kinematics result
    rt = RotoTransMatrix()
    rt.from_rt_matrix(np.eye(4))
    model.forward_kinematics.return_value = {parent_name: np.eye(4)}

    # Set up markers
    joint_center_marker = np.array([0, 0, 0, 1])
    distal_marker = np.array([0, 0, 1, 1])

    model.markers_indices.side_effect = lambda x: [0] if x == joint_center_markers else [1]
    model.markers_in_global.return_value = np.column_stack((joint_center_marker, distal_marker))

    # Test the longitudinal axis calculation
    joint_center, longitudinal_axis = sara._longitudinal_axis(model)

    # The joint center should be at the origin
    assert np.allclose(joint_center[:3], np.array([0, 0, 0]))

    # The longitudinal axis should point in the Z direction
    assert np.allclose(longitudinal_axis[:3, 0], np.array([0, 0, 1]))

    # Test with reversed direction
    sara.longitudinal_axis_sign = -1
    joint_center, longitudinal_axis = sara._longitudinal_axis(model)
    assert np.allclose(longitudinal_axis[:3, 0], np.array([0, 0, -1]))

def test_get_rotation_index():
    # Create a mock model with X rotation
    model = MagicMock(spec=BiomechanicalModelReal)
    model.segments = {
        child_name: MagicMock(rotations=MagicMock(value="x"))
    }

    # Test X rotation
    aor_index, perp_index, long_index = sara.get_rotation_index(model)
    assert aor_index == 0
    assert perp_index == 1
    assert long_index == 2

    # Test Z rotation
    model.segments[child_name].rotations.value = "z"
    aor_index, perp_index, long_index = sara.get_rotation_index(model)
    assert aor_index == 2
    assert perp_index == 0
    assert long_index == 1

    # Test Y rotation (should raise NotImplementedError)
    model.segments[child_name].rotations.value = "y"
    with pytest.raises(NotImplementedError):
        sara.get_rotation_index(model)

    # Test multiple rotations (should raise RuntimeError)
    model.segments[child_name].rotations.value = "xyz"
    with pytest.raises(RuntimeError):
        sara.get_rotation_index(model)


# Test Joint Center Tool
def test_init():
    # Create a mock model
    model = MagicMock(spec=BiomechanicalModelReal)
    model.segments = []

    # Test initialization
    jct = JointCenterTool(model)
    assert jct.original_model == model
    assert jct.animate_reconstruction is False
    assert len(jct.joint_center_tasks) == 0

def test_add():
    # Create a mock model
    model = MagicMock(spec=BiomechanicalModelReal)

    # Set up segments for parent-child relationship
    parent_segment = MagicMock()
    parent_segment.name = "parent"
    parent_segment.parent_name = "base"

    child_segment = MagicMock()
    child_segment.name = "child"
    child_segment.parent_name = "parent"

    model.segments = {"parent": parent_segment, "child": child_segment}
    model.get_chain_between_segments.return_value = ["parent", "child"]

    # Create a mock Score object
    c3d_data = MagicMock(spec=C3dData)
    score = Score(
        c3d_data,
        "parent",
        "child",
        ["marker1", "marker2"],
        ["marker3", "marker4"],
    )

    # Test adding a Score task
    jct = JointCenterTool(model)
    jct.add(score)
    assert len(jct.joint_center_tasks) == 1
    assert jct.joint_center_tasks[0] == score

    # Test adding an invalid task
    with pytest.raises(RuntimeError):
        jct.add("not a Score or Sara object")

    # Test adding a task with invalid parent-child relationship
    invalid_score = Score(
        c3d_data,
        "parent",
        "nonexistent",
        ["marker1", "marker2"],
        ["marker3", "marker4"],
    )
    model.get_chain_between_segments.side_effect = lambda p, c: []
    with pytest.raises(RuntimeError):
        jct.add(invalid_score)

