import os
import pytest
import numpy as np
import numpy.testing as npt

from biobuddy.utils.named_list import NamedList
from test_utils import remove_temporary_biomods, MockEmptyC3dData
from biobuddy import (
    BiomechanicalModelReal,
    JointCenterTool,
    Score,
    Sara,
    C3dData,
    MarkerWeight,
    Rotations,
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
            score_model.segments["femur_r"].segment_coordinate_system.scs.rt_matrix,
            # The rotation part did not change, only the translation part was modified
            np.array(
                [
                    [0.94106637, 0.33488294, 0.04740786, -0.07073665],
                    [-0.33553695, 0.90675222, 0.25537299, -0.02090582],
                    [0.04253287, -0.25623002, 0.96567962, 0.09795824],
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
                    [0.94106637, 0.33488294, 0.04740786, -0.07167729],
                    [-0.33553695, 0.90675222, 0.25537299, -0.02279122],
                    [0.04253287, -0.25623002, 0.96567962, 0.09659234],
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
                    [0.97291137, 0.03767885, -0.22808722, 0.02126479],
                    [-0.06119303, 0.99340983, -0.09691404, -0.40906061],
                    [0.22293247, 0.10824612, 0.96880539, -0.03103533],
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
                    [0.97197795, 0.04183997, -0.23131857, 0.02157546],
                    [-0.06106364, 0.99519203, -0.0765771, -0.407386],
                    [0.22700242, 0.08855641, 0.96985961, -0.02919023],
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
    original_optimal_q, _ = scaled_model.inverse_kinematics(
        marker_positions=hip_c3d.get_position(list(marker_weights.keys()))[:3, :, :],
        marker_names=list(marker_weights.keys()),
        marker_weights=marker_weights,
        method="lm",
    )
    original_markers_reconstructed = scaled_model.markers_in_global(original_optimal_q)
    original_marker_position_diff = hip_c3d.get_position(list(marker_weights.keys())) - original_markers_reconstructed
    original_marker_tracking_error = np.sum(original_marker_position_diff[:3, :, :] ** 2)

    new_optimal_q, _ = score_model.inverse_kinematics(
        marker_positions=hip_c3d.get_position(list(marker_weights.keys()))[:3, :, :],
        marker_names=list(marker_weights.keys()),
        marker_weights=marker_weights,
        method="lm",
    )
    new_markers_reconstructed = score_model.markers_in_global(new_optimal_q)
    new_marker_position_diff = hip_c3d.get_position(list(marker_weights.keys())) - new_markers_reconstructed
    new_marker_tracking_error = np.sum(new_marker_position_diff[:3, :, :] ** 2)

    npt.assert_almost_equal(original_marker_tracking_error, 1.2695623487402687, decimal=2)
    if initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 0.852078389733663, decimal=2)
    else:
        npt.assert_almost_equal(new_marker_tracking_error, 0.8546370633355006, decimal=2)
    npt.assert_array_less(new_marker_tracking_error, original_marker_tracking_error)

    # # For debugging purposes
    # from pyorerun import PyoMarkers
    # pyomarkers = PyoMarkers(data=hip_c3d.get_position(list(marker_weights.keys())), channels=list(marker_weights.keys()), show_labels=False)
    # visualize_modified_model_output(leg_model_filepath, score_biomod_filepath, original_optimal_q, new_optimal_q, pyomarkers)

    # Knee
    marker_names = list(marker_weights.keys())
    original_optimal_q, _ = scaled_model.inverse_kinematics(
        marker_positions=knee_c3d.get_position(marker_names)[:3, :, :],
        marker_names=marker_names,
        marker_weights=marker_weights,
        method="lm",
    )
    new_optimal_q, _ = score_model.inverse_kinematics(
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

    npt.assert_almost_equal(original_marker_tracking_error, 4.705350581055244, decimal=2)
    if initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 3.1482122493133398, decimal=2)
    else:
        npt.assert_almost_equal(new_marker_tracking_error, 3.162024226671943, decimal=2)
    npt.assert_array_less(new_marker_tracking_error, original_marker_tracking_error)

    # Test replace_joint_centers
    for muscle_group in scaled_model.muscle_groups:
        # Check that there are the same number of muscles
        assert (
            scaled_model.muscle_groups[muscle_group.name].muscle_names
            == score_model.muscle_groups[muscle_group.name].muscle_names
        )
        assert (
            scaled_model.muscle_groups[muscle_group.name].nb_muscles
            == score_model.muscle_groups[muscle_group.name].nb_muscles
        )

        for muscle in muscle_group.muscles:
            # Test that the origin and insertion have been updated locally
            origin_scaled = scaled_model.muscle_groups[muscle_group.name].muscles[muscle.name].origin_position.position
            insertion_scaled = (
                scaled_model.muscle_groups[muscle_group.name].muscles[muscle.name].insertion_position.position
            )
            origin_score = score_model.muscle_groups[muscle_group.name].muscles[muscle.name].origin_position.position
            insertion_score = (
                score_model.muscle_groups[muscle_group.name].muscles[muscle.name].insertion_position.position
            )
            if muscle_group.origin_parent_name == "pelvis":
                # pelvis did not move so should be the same
                assert np.all(origin_scaled == origin_score)
            else:
                assert np.any(origin_scaled != origin_score)
            assert np.any(insertion_scaled != insertion_score)
            # So that they stay at the same place in the global reference frame
            scaled_origin_in_global = scaled_model.muscle_origin_in_global(muscle.name)
            score_origin_in_global = score_model.muscle_origin_in_global(muscle.name)
            npt.assert_almost_equal(scaled_origin_in_global, score_origin_in_global, decimal=5)
            scaled_insertion_in_global = scaled_model.muscle_insertion_in_global(muscle.name)
            score_insertion_in_global = score_model.muscle_insertion_in_global(muscle.name)
            npt.assert_almost_equal(scaled_insertion_in_global, score_insertion_in_global, decimal=5)

            # Test the position of the via points
            via_points_scaled = scaled_model.via_points_in_global(muscle.name)
            via_points_score = score_model.via_points_in_global(muscle.name)
            npt.assert_almost_equal(via_points_scaled, via_points_score, decimal=5)

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
                [1.0, 0.0, 0.0, -0.0361026],
                [0.0, 1.0, 0.0, -0.03539462],
                [0.0, 0.0, 1.0, -0.01121598],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=3,
    )
    assert score_model.segments["femur_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["femur_r"].segment_coordinate_system.scs.rt_matrix,
        np.array(
            [
                [1.0, -0.0, 0.0, -0.0316564],
                [-0.0, 1.0, 0.0, -0.02795538],
                [0.0, 0.0, 1.0, 0.09124198],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=3,
    )

    assert score_model.segments["tibia_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["tibia_r_parent_offset"].segment_coordinate_system.scs.rt_matrix,
        np.array(
            [
                [-0.93782901, 0.16036525, 0.30783068, 0.00538483],
                [0.18406367, 0.98167428, 0.04935769, -0.38267316],
                [-0.2942742, 0.10294952, -0.95016003, -0.00960224],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=3,
    )

    assert score_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["tibia_r"].segment_coordinate_system.scs.rt_matrix,
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
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
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    assert scaled_model.segments["tibia_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["tibia_r_parent_offset"].segment_coordinate_system.scs.rt_matrix,
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -0.387741], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    assert scaled_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["tibia_r"].segment_coordinate_system.scs.rt_matrix,
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )

    # Test the reconstruction for the original model and the output model with the functional joint centers
    # Hip
    original_optimal_q, _ = scaled_model.inverse_kinematics(
        marker_positions=hip_c3d.get_position(list(marker_weights.keys()))[:3, :, :],
        marker_names=list(marker_weights.keys()),
        marker_weights=marker_weights,
        method="lm",
    )
    original_markers_reconstructed = scaled_model.markers_in_global(original_optimal_q)
    original_marker_position_diff = hip_c3d.get_position(list(marker_weights.keys())) - original_markers_reconstructed
    original_marker_tracking_error = np.sum(original_marker_position_diff[:3, :, :] ** 2)

    new_optimal_q, _ = score_model.inverse_kinematics(
        marker_positions=hip_c3d.get_position(list(marker_weights.keys()))[:3, :, :],
        marker_names=list(marker_weights.keys()),
        marker_weights=marker_weights,
        method="lm",
    )
    new_markers_reconstructed = score_model.markers_in_global(new_optimal_q)
    new_marker_position_diff = hip_c3d.get_position(list(marker_weights.keys())) - new_markers_reconstructed
    new_marker_tracking_error = np.sum(new_marker_position_diff[:3, :, :] ** 2)

    # The error is worse because it is a small test (for the tests to run quickly)
    npt.assert_almost_equal(original_marker_tracking_error, 8.822514366803158, decimal=2)
    npt.assert_almost_equal(new_marker_tracking_error, 10.571775620737359, decimal=2)

    # Knee
    marker_names = list(marker_weights.keys())
    original_optimal_q, _ = scaled_model.inverse_kinematics(
        marker_positions=knee_c3d.get_position(marker_names)[:3, :, :],
        marker_names=marker_names,
        marker_weights=marker_weights,
        method="lm",
    )
    new_optimal_q, _ = score_model.inverse_kinematics(
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
    npt.assert_almost_equal(original_marker_tracking_error, 9.065551601012801, decimal=2)
    npt.assert_almost_equal(new_marker_tracking_error, 9.532989125419526, decimal=2)

    # Test replace_joint_centers
    for muscle_group in scaled_model.muscle_groups:
        # Check that there are the same number of muscles
        assert (
            scaled_model.muscle_groups[muscle_group.name].muscle_names
            == score_model.muscle_groups[muscle_group.name].muscle_names
        )
        assert (
            scaled_model.muscle_groups[muscle_group.name].nb_muscles
            == score_model.muscle_groups[muscle_group.name].nb_muscles
        )

        for muscle in muscle_group.muscles:
            # Test that the origin and insertion have been updated locally
            origin_scaled = scaled_model.muscle_groups[muscle_group.name].muscles[muscle.name].origin_position.position
            insertion_scaled = (
                scaled_model.muscle_groups[muscle_group.name].muscles[muscle.name].insertion_position.position
            )
            origin_score = score_model.muscle_groups[muscle_group.name].muscles[muscle.name].origin_position.position
            insertion_score = (
                score_model.muscle_groups[muscle_group.name].muscles[muscle.name].insertion_position.position
            )
            if muscle_group.origin_parent_name == "pelvis":
                # pelvis did not move so should be the same
                assert np.all(origin_scaled == origin_score)
            else:
                assert np.any(origin_scaled != origin_score)
            assert np.any(insertion_scaled != insertion_score)
            # So that they stay at the same place in the global reference frame
            scaled_origin_in_global = scaled_model.muscle_origin_in_global(muscle.name)
            score_origin_in_global = score_model.muscle_origin_in_global(muscle.name)
            npt.assert_almost_equal(scaled_origin_in_global, score_origin_in_global, decimal=5)
            scaled_insertion_in_global = scaled_model.muscle_insertion_in_global(muscle.name)
            score_insertion_in_global = score_model.muscle_insertion_in_global(muscle.name)
            npt.assert_almost_equal(scaled_insertion_in_global, score_insertion_in_global, decimal=5)

            # Test the position of the via points
            via_points_scaled = scaled_model.via_points_in_global(muscle.name)
            via_points_score = score_model.via_points_in_global(muscle.name)
            npt.assert_almost_equal(via_points_scaled, via_points_score, decimal=5)

    # TODO: Test mesh files and contacts

    remove_temporary_biomods()
    if os.path.exists(score_biomod_filepath):
        os.remove(score_biomod_filepath)


# Test Rigid Segment Identification:
def test_init_rigid_segment_identification():

    # Set up
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    c3d_data = C3dData(knee_functional_trial_path, first_frame=300, last_frame=400)

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
    with pytest.raises(
        RuntimeError,
        match="The names _reset_axis are not allowed in the parent or child names. Please change the segment named parent_reset_axis from the Score configuration.",
    ):
        RigidSegmentIdentification(
            c3d_data,
            "parent_reset_axis",
            child_name,
            parent_marker_names,
            child_marker_names,
        )

    with pytest.raises(
        RuntimeError,
        match="The names _translation are not allowed in the parent or child names. Please change the segment named child_translation from the Score configuration.",
    ):
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
    with pytest.raises(RuntimeError, match=r"The marker position is empty \(shape: \(4, 1, 0\)\), cannot compute std."):
        rsi_no_markers = RigidSegmentIdentification(
            MockEmptyC3dData(),
            parent_name,
            child_name,
            parent_marker_names,
            child_marker_names,
        )

    # Test with no movement
    c3d_data.all_marker_positions = np.ones_like(c3d_data.all_marker_positions)
    with pytest.raises(
        RuntimeError,
        match=r"The markers \['RGT', 'RTHI1', 'RTHI2', 'RTHI3', 'RATT', 'RLM', 'RSPH', 'RLEG1', 'RLEG2', 'RLEG3'\] are not moving in the functional trial ",
    ):
        rsi = RigidSegmentIdentification(
            c3d_data,
            parent_name,
            child_name,
            parent_marker_names,
            child_marker_names,
        )


def test_marker_residual():

    # Set up
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    c3d_data = C3dData(knee_functional_trial_path, first_frame=300, last_frame=400)

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

    # Create test data
    optimal_rt = np.eye(4).flatten()
    static_markers_in_local = np.ones((4, 2))  # 4D, 2 markers
    functional_markers_in_global = np.ones((4, 2))  # 4D, 2 markers

    # When RT is identity and markers match, residual should be 0
    residual = rsi.marker_residual(optimal_rt, static_markers_in_local, functional_markers_in_global)
    assert residual == 0

    # When markers don't match, residual should be positive
    functional_markers_in_global = np.ones((4, 2)) * 2
    residual = rsi.marker_residual(optimal_rt, static_markers_in_local, functional_markers_in_global)
    assert residual > 0

    # Test get_good_frames
    # Create test residuals
    residuals = np.array([1.0, 1.1, 1.2, 5.0, 1.3])  # One outlier at index 3
    nb_frames = len(residuals)

    # Test frame filtering
    valid_frames = rsi.get_good_frames(residuals, nb_frames)
    assert np.sum(valid_frames) == 4  # Should remove one frame
    assert not valid_frames[3]  # The outlier should be removed

    # Test rt_constraints
    # Test with a valid rotation matrix (orthonormal)
    rt_matrix = np.eye(4)
    constraints = rsi.rt_constraints(rt_matrix.flatten())
    assert np.allclose(constraints, np.zeros(6))

    # Test with an invalid rotation matrix
    rt_matrix = np.eye(4)
    rt_matrix[0, 0] = 2.0  # Make it non-orthonormal
    constraints = rsi.rt_constraints(rt_matrix.flatten())
    assert not np.allclose(constraints, np.zeros(6))

    # Test check_optimal_rt_inputs
    # Create valid test data
    markers = np.random.rand(3, 2, 10) * 0.0001  # 3D, 2 markers, 10 frames
    markers = np.vstack((markers, np.ones((1, 2, 10))))  # Add homogeneous coordinate
    static_markers = np.random.rand(3, 2) * 0.0001  # 3D, 2 markers
    static_markers = np.vstack((static_markers, np.ones((1, 2))))  # Add homogeneous coordinate
    marker_names = ["marker1", "marker2"]

    # Test with valid inputs
    result = rsi.check_optimal_rt_inputs(markers, static_markers, marker_names)
    assert result is not None
    assert len(result) == 3
    assert result[0] == 2  # Number of markers
    assert result[1] == 10  # Number of frames
    npt.assert_almost_equal(result[2], np.zeros((3, 2)), decimal=3)  # Static centered

    # Test with mismatched marker names
    with pytest.raises(RuntimeError, match=r"The marker_names \['marker1'\] do not match the number of markers 2."):
        rsi.check_optimal_rt_inputs(markers, static_markers, ["marker1"])

    # Test with marker movement
    # Make markers move significantly between static and functional
    static_markers[0, 0] = 0
    markers[0, 0, :] = 1.0  # Large difference in position
    with pytest.raises(
        RuntimeError,
        match="The marker marker1 seem to move during the functional trial.The distance between the center and this marker is ",
    ):
        rsi.check_optimal_rt_inputs(markers, static_markers, marker_names)


# Test SARA
def test_longitudinal_axis():

    # Set up
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    knee_c3d = C3dData(knee_functional_trial_path, first_frame=300, last_frame=400)

    child_name = "tibia_r"
    parent_name = "femur_r"
    scaled_model = BiomechanicalModelReal().from_biomod(
        filepath=leg_model_filepath,
    )
    sara = Sara(
        functional_c3d=knee_c3d,
        parent_name=parent_name,
        child_name=child_name,
        parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
        child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
        joint_center_markers=["RLFE", "RMFE"],
        distal_markers=["RLM", "RSPH"],
        is_longitudinal_axis_from_jcs_to_distal_markers=False,
        initialize_whole_trial_reconstruction=False,
        animate_rt=False,
    )

    # Test the longitudinal axis calculation
    joint_center, longitudinal_axis = sara._longitudinal_axis(scaled_model)
    npt.assert_almost_equal(
        joint_center.reshape(
            4,
        ),
        np.array([0.00498378, -0.37616598, -0.00302045, 1.0]),
        decimal=6,
    )
    npt.assert_almost_equal(
        longitudinal_axis.reshape(
            4,
        ),
        np.array([0.06652103, 0.99764921, -0.01646222, 1.0]),
        decimal=6,
    )

    # TODO: test the other configurations when I have a model to test it correctly

    # Test get_rotation_index
    # Test Z rotation
    aor_index, perp_index, long_index = sara.get_rotation_index(scaled_model)
    assert aor_index == 2
    assert perp_index == 0
    assert long_index == 1

    # Test X rotation
    scaled_model.segments[child_name].rotations = Rotations.X
    aor_index, perp_index, long_index = sara.get_rotation_index(scaled_model)
    assert aor_index == 0
    assert perp_index == 1
    assert long_index == 2

    # Test Y rotation (should raise NotImplementedError)
    scaled_model.segments[child_name].rotations = Rotations.Y
    with pytest.raises(
        NotImplementedError,
        match=r"This axis combination has not been tested yet. Please make sure that the cross product make sense \(correct order and correct sign\).",
    ):
        sara.get_rotation_index(scaled_model)

    # Test multiple rotations (should raise RuntimeError)
    scaled_model.segments[child_name].rotations = Rotations.XYZ
    with pytest.raises(
        RuntimeError,
        match="The Sara algorithm is meant to be used with a one DoF joint, you have defined rotations Rotations.XYZ for segment tibia_r.",
    ):
        sara.get_rotation_index(scaled_model)


# Test Joint Center Tool
def test_add():

    # Set up
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    scaled_model = BiomechanicalModelReal().from_biomod(
        filepath=leg_model_filepath,
    )

    # Test adding a Score task
    jct = JointCenterTool(scaled_model)

    # Test adding an invalid task
    with pytest.raises(RuntimeError, match="The joint center must be a Score or Sara object."):
        jct.add("not a Score or Sara object")
