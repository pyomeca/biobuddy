import os
import pytest
import numpy as np
import numpy.testing as npt

from biobuddy import (
    BiomechanicalModelReal,
    JointCenterTool,
    Score,
    Sara,
    C3dData,
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


@pytest.mark.parametrize(
    "rt_method",
    [
        "optimization",
        # "numerical",
    ],
)
@pytest.mark.parametrize("initialize_whole_trial_reconstruction", [True, False])
def test_score_and_sara_without_ghost_segments(rt_method, initialize_whole_trial_reconstruction):

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
    scaled_model = BiomechanicalModelReal.from_biomod(
        filepath=leg_model_filepath,
    )
    marker_weights = {
        "RASIS": 1.0,
        "LASIS": 1.0,
        "LPSIS": 0.5,
        "RPSIS": 0.5,
        "RLFE": 1.0,
        "RMFE": 1.0,
        "RGT": 0.1,
        "RTHI1": 5.0,
        "RTHI2": 5.0,
        "RTHI3": 5.0,
        "RATT": 0.5,
        "RLM": 1.0,
        "RSPH": 1.0,
        "RLEG1": 5.0,
        "RLEG2": 5.0,
        "RLEG3": 5.0,
    }

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
            method=rt_method,
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
            method=rt_method,
            initialize_whole_trial_reconstruction=initialize_whole_trial_reconstruction,
            animate_rt=False,
        )
    )

    score_model = joint_center_tool.replace_joint_centers(marker_weights)

    # Test that the model created is valid
    score_model.to_biomod(score_biomod_filepath)

    # Test the joints' new RT
    assert score_model.segments["femur_r"].segment_coordinate_system.is_in_local
    if rt_method == "optimization" and initialize_whole_trial_reconstruction:
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
    elif rt_method == "optimization" and not initialize_whole_trial_reconstruction:
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
        )
    else:
        raise RuntimeError("mmmmm")

    assert score_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    if rt_method == "optimization" and initialize_whole_trial_reconstruction:
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
        )
    elif rt_method == "optimization" and not initialize_whole_trial_reconstruction:
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
        )
    else:
        raise RuntimeError("mmmmm")

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
    if rt_method == "optimization" and initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 0.854628320760068, decimal=5)
    elif rt_method == "optimization" and not initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 0.8563361750437755)
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
    if rt_method == "optimization" and initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 3.195043059038364)
    elif rt_method == "optimization" and not initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 3.2169738227469282)
    npt.assert_array_less(new_marker_tracking_error, original_marker_tracking_error)


def test_score_and_sara_with_ghost_segments():

    from examples import replace_joint_centers_functionally as example

    example.main(False)
    # TODO !
