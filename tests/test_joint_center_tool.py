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
                    [0.96988145, 0.03936066, -0.2403762, 0.02157451],
                    [-0.0607774, 0.99474922, -0.0823413, -0.40738561],
                    [0.23587303, 0.09447074, 0.96718105, -0.02918969],
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
        npt.assert_almost_equal(new_marker_tracking_error, 0.8563361750437755, decimal=5)
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
        npt.assert_almost_equal(new_marker_tracking_error, 3.2169738227469282, decimal=5)
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

    # Test the joints' new RT
    assert score_model.segments["femur_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["femur_r_parent_offset"].segment_coordinate_system.scs[:, :, 0],
        np.array(
            [
                [1.0, 0.0, 0.0, -0.03420629],
                [0.0, 1.0, 0.0, -0.03289557],
                [0.0, 0.0, 1.0, -0.01101487],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert score_model.segments["femur_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["femur_r"].segment_coordinate_system.scs[:, :, 0],
        np.array(
            [
                [1.0, -0.0, 0.0, -0.03355271],
                [-0.0, 1.0, 0.0, -0.03045443],
                [0.0, 0.0, 1.0, 0.09104087],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    assert score_model.segments["tibia_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["tibia_r_parent_offset"].segment_coordinate_system.scs[:, :, 0],
        np.array(
            [
                [0.94215046, 0.13837971, -0.30529259, 0.00540745],
                [-0.15878056, 0.986381, -0.04290986, -0.38266012],
                [0.29519695, 0.08890207, 0.95129132, -0.0098363],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
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

    # The error is worse because it is a unit test (for the tests to run quickly)
    npt.assert_almost_equal(original_marker_tracking_error, 0.2879320932283139)
    npt.assert_almost_equal(new_marker_tracking_error, 1.1078381510264328, decimal=5)

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
    npt.assert_almost_equal(original_marker_tracking_error, 9.956010278714874)
    npt.assert_almost_equal(new_marker_tracking_error, 10.517878851688314, decimal=5)

    remove_temporary_biomods()
    if os.path.exists(score_biomod_filepath):
        os.remove(score_biomod_filepath)
