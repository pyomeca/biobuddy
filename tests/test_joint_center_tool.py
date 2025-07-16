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
    pyomarkers: "PyoMarkers",
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
            score_model.segments["femur_r"].segment_coordinate_system.scs.rt_matrix,
            # The rotation part did not change, only the translation part was modified
            np.array(
                [
                    [
                        0.941067,
                        0.334883,
                        0.047408,
                        -0.07076823,
                    ],
                    [-0.335537, 0.906752, 0.255373, -0.02166063],
                    [0.042533, -0.25623, 0.96568, 0.09724843],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            decimal=5,
        )
    else:
        npt.assert_almost_equal(
            score_model.segments["femur_r"].segment_coordinate_system.scs.rt_matrix,
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
            score_model.segments["tibia_r"].segment_coordinate_system.scs.rt_matrix,
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
            score_model.segments["tibia_r"].segment_coordinate_system.scs.rt_matrix,
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
        scaled_model.segments["femur_r"].segment_coordinate_system.scs.rt_matrix,
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
        scaled_model.segments["tibia_r"].segment_coordinate_system.scs.rt_matrix,
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
        npt.assert_almost_equal(new_marker_tracking_error, 3.1621464045718777, decimal=4)
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
        score_model.segments["femur_r_parent_offset"].segment_coordinate_system.scs.rt_matrix,
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
        score_model.segments["femur_r"].segment_coordinate_system.scs.rt_matrix,
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
        score_model.segments["tibia_r_parent_offset"].segment_coordinate_system.scs.rt_matrix,
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
        score_model.segments["tibia_r"].segment_coordinate_system.scs.rt_matrix,
        np.array([[1.0, -0.0, 0.0, 0.0], [-0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )

    # Test that the original model did not change
    assert scaled_model.segments["femur_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["femur_r_parent_offset"].segment_coordinate_system.scs.rt_matrix,
        np.array(
            [[1.0, 0.0, 0.0, -0.067759], [0.0, 1.0, 0.0, -0.06335], [0.0, 0.0, 1.0, 0.080026], [0.0, 0.0, 0.0, 1.0]]
        ),
    )
    assert scaled_model.segments["femur_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["femur_r"].segment_coordinate_system.scs.rt_matrix,
        np.array([[1.0, -0.0, 0.0, 0.0], [-0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    assert scaled_model.segments["tibia_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["tibia_r_parent_offset"].segment_coordinate_system.scs.rt_matrix,
        np.array([[1.0, -0.0, 0.0, 0.0], [-0.0, 1.0, 0.0, -0.387741], [0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    assert scaled_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["tibia_r"].segment_coordinate_system.scs.rt_matrix,
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
