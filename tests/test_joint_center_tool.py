import os

from biobuddy.utils.named_list import NamedList
from biobuddy import (
    BiomechanicalModelReal,
    JointCenterTool,
    Score,
    Sara,
    C3dData,
    MarkerWeight,
    Rotations,
    Axis,
    DictData,
    SegmentCoordinateSystemUtils,
)
from biobuddy.model_modifiers.joint_center_tool import (
    RigidSegmentIdentification,
    JointCoordinateModifier,
    get_svd,
)
from biobuddy.utils.linear_algebra import RotoTransMatrix, RotoTransMatrixTimeSeries
import numpy as np
import numpy.testing as npt
import pytest

from test_utils import remove_temporary_biomods, MockEmptyC3dData


def _unit(vector: np.ndarray | list[float]) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def _basis_from_first_axis(first_axis: np.ndarray) -> np.ndarray:
    first_axis = _unit(first_axis)
    candidate = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(first_axis, candidate))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    second_axis = _unit(np.cross(candidate, first_axis))
    third_axis = np.cross(first_axis, second_axis)
    return np.column_stack((first_axis, second_axis, third_axis))


def _rotation_about_x(angle: float) -> np.ndarray:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_angle, -sin_angle],
            [0.0, sin_angle, cos_angle],
        ]
    )


def _global_direction(rt: RotoTransMatrixTimeSeries, local_axis: np.ndarray) -> np.ndarray:
    global_directions = np.zeros((3, len(rt)))
    local_axis = np.asarray(local_axis, dtype=float).reshape(3)
    for frame_index in range(len(rt)):
        global_direction = rt[frame_index].rotation_matrix.rotation_matrix @ local_axis
        global_directions[:, frame_index] = global_direction / np.linalg.norm(global_direction)
    return global_directions


def _perfect_hinge_rt_and_markers(
    nb_frames: int = 60,
) -> tuple[
    RotoTransMatrixTimeSeries,
    RotoTransMatrixTimeSeries,
    DictData,
    DictData,
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build a perfect two-segment hinge with parent 6-DoF motion and child 1-DoF relative rotation.

    The first frame is the static calibration pose. Marker clouds have zero centroid so
    ``rigidify`` recovers the expected technical frames without an extra centroid offset.
    """
    parent_axis_local = _unit([0.25, 0.86, -0.44])
    child_axis_local = _unit([0.0, 0.0, 1.0])
    parent_axis_basis = _basis_from_first_axis(parent_axis_local)
    child_axis_basis = _basis_from_first_axis(child_axis_local)
    parent_hinge_local = np.array([0.08, -0.03, 0.02])
    child_hinge_local = np.array([-0.04, 0.06, 0.03])
    parent_marker_local = np.array(
        [
            [0.18, 0.02, -0.05],
            [-0.12, 0.11, 0.03],
            [0.04, -0.16, 0.10],
            [-0.10, 0.03, -0.08],
        ],
        dtype=float,
    )
    child_marker_local = np.array(
        [
            [0.16, -0.04, 0.06],
            [-0.08, 0.12, -0.07],
            [0.02, -0.13, -0.10],
            [-0.10, 0.05, 0.11],
        ],
        dtype=float,
    )
    parent_marker_local -= parent_marker_local.mean(axis=0)
    child_marker_local -= child_marker_local.mean(axis=0)
    parent_names = tuple(f"P{i + 1}" for i in range(4))
    child_names = tuple(f"C{i + 1}" for i in range(4))
    markers = {
        **{name: np.ones((4, nb_frames)) for name in parent_names},
        **{name: np.ones((4, nb_frames)) for name in child_names},
    }
    hinge_origin_global = np.ones((4, nb_frames))
    rt_parent = RotoTransMatrixTimeSeries(nb_frames)
    rt_child = RotoTransMatrixTimeSeries(nb_frames)
    for frame_index in range(nb_frames):
        parent_angles = np.array(
            [
                0.15 * np.sin(frame_index / 14.0),
                0.10 * np.sin(frame_index / 19.0),
                0.08 * np.sin(frame_index / 11.0),
            ]
        )
        parent_translation = np.array(
            [
                0.10 * np.sin(frame_index / 17.0),
                -0.04 * np.sin(frame_index / 23.0),
                0.06 * np.sin(frame_index / 29.0),
            ]
        )
        if frame_index == 0:
            parent_angles = np.zeros(3)
            parent_translation = np.zeros(3)
        parent_rt = RotoTransMatrix.from_euler_angles_and_translation("xyz", parent_angles, parent_translation)
        parent_rotation = parent_rt.rotation_matrix.rotation_matrix
        parent_translation = parent_rt.translation.reshape(3)
        hinge_angle = -1.2 + 2.4 * frame_index / (nb_frames - 1)
        child_rotation = parent_rotation @ parent_axis_basis @ _rotation_about_x(hinge_angle) @ child_axis_basis.T
        child_translation = (
            parent_rotation @ parent_hinge_local + parent_translation - child_rotation @ child_hinge_local
        )
        rt_parent[frame_index] = parent_rt
        rt_child[frame_index] = RotoTransMatrix.from_rotation_matrix_and_translation(child_rotation, child_translation)
        hinge_origin_global[:3, frame_index] = parent_rotation @ parent_hinge_local + parent_translation
        for marker_index, marker_position in enumerate(parent_marker_local):
            markers[parent_names[marker_index]][:3, frame_index] = (
                parent_rotation @ marker_position + parent_translation
            )
        for marker_index, marker_position in enumerate(child_marker_local):
            markers[child_names[marker_index]][:3, frame_index] = child_rotation @ marker_position + child_translation
    functional_data = DictData(markers)
    static_data = DictData({marker_name: values[:, 0:1] for marker_name, values in markers.items()})
    return (
        rt_parent,
        rt_child,
        functional_data,
        static_data,
        parent_names,
        child_names,
        parent_axis_local,
        child_axis_local,
        parent_hinge_local,
        child_hinge_local,
        hinge_origin_global,
    )


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
    import pyorerun  # type: ignore

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
    animate = False  # Debugging purpose only

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    score_biomod_filepath = parent_path + "/examples/models/leg_without_ghost_parents_score.bioMod"

    hip_functional_trial_path = parent_path + "/examples/data/functional_trials/right_hip.c3d"
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    hip_c3d = C3dData(
        hip_functional_trial_path, first_frame=1, last_frame=499
    )  # Marker inversion happening after the 500th frame in the example data!
    knee_c3d = C3dData(knee_functional_trial_path, first_frame=300, last_frame=821)

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

    joint_center_tool = JointCenterTool(scaled_model, animate_reconstruction=animate)
    # Hip Right
    joint_center_tool.add(
        Score(
            functional_trial=hip_c3d,
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
            functional_trial=knee_c3d,
            parent_name="femur_r",
            child_name="tibia_r",
            parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
            child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
            joint_center_markers=["RLFE", "RMFE"],
            distal_markers=["RLM", "RSPH"],
            is_longitudinal_axis_from_jcs_to_distal_markers=False,
            expected_rotation_axis_orientation=Axis("right_knee_sara", "RLFE", "RMFE"),
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
                    [0.941067, 0.334883, 0.047408, -0.07073673],
                    [-0.335537, 0.906752, 0.255373, -0.02090609],
                    [0.042533, -0.25623, 0.96568, 0.09795744],
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
                    [0.941067, 0.334883, 0.047408, -0.07167729],
                    [-0.335537, 0.906752, 0.255373, -0.02279122],
                    [0.042533, -0.25623, 0.96568, 0.09659233],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            decimal=5,
        )

    assert score_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    if initialize_whole_trial_reconstruction:
        # The translation is the result from SCoRE (and should not change)
        npt.assert_almost_equal(
            score_model.segments["tibia_r"].segment_coordinate_system.scs.translation,
            # Both rotation and translation parts were modified
            np.array([0.0212648, -0.40906054, -0.03103454]),
            decimal=5,
        )
        # The rotation is the result from SARA (and is less stable numerically)
        npt.assert_almost_equal(
            score_model.segments["tibia_r"].segment_coordinate_system.scs.rotation_matrix.rotation_matrix,
            # Both rotation and translation parts were modified
            np.array(
                [
                    [-0.99777445, 0.06657082, 0.00389257],
                    [0.06658717, 0.99150906, 0.11169577],
                    [0.00357613, 0.11170635, -0.9937342],
                ]
            ),
            decimal=5,
        )
    else:
        # The translation is the result from SCoRE (and should not change)
        npt.assert_almost_equal(
            score_model.segments["tibia_r"].segment_coordinate_system.scs.translation,
            np.array([0.02157546, -0.407386, -0.02919023]),
            decimal=5,
        )
        # The rotation is the result from SARA (and is less stable numerically)
        npt.assert_almost_equal(
            score_model.segments["tibia_r"].segment_coordinate_system.scs.rotation_matrix.rotation_matrix,
            np.array(
                [
                    [-0.99777, 0.06546, 0.01267],
                    [0.06644, 0.9922, 0.10551],
                    [-0.00566, 0.10612, -0.99434],
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
        npt.assert_almost_equal(new_marker_tracking_error, 0.8292538655934063, decimal=2)
    else:
        npt.assert_almost_equal(new_marker_tracking_error, 0.8338653905600818, decimal=2)
    npt.assert_array_less(new_marker_tracking_error, original_marker_tracking_error)

    # Animate the output
    if animate:
        from pyorerun import PyoMarkers

        pyomarkers = PyoMarkers(
            data=hip_c3d.get_position(list(marker_weights.keys())),
            channels=list(marker_weights.keys()),
            show_labels=False,
        )
        visualize_modified_model_output(
            leg_model_filepath,
            score_biomod_filepath,
            original_optimal_q,
            new_optimal_q,
            pyomarkers,
        )

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

    # Animate the results
    if animate:
        from pyorerun import PyoMarkers

        pyomarkers = PyoMarkers(
            data=knee_c3d.get_position(marker_names),
            channels=marker_names,
            show_labels=False,
        )
        visualize_modified_model_output(
            leg_model_filepath,
            score_biomod_filepath,
            original_optimal_q,
            new_optimal_q,
            pyomarkers,
        )

    markers_index = scaled_model.markers_indices(marker_names)

    original_markers_reconstructed = scaled_model.markers_in_global(original_optimal_q)[:3, markers_index, :]
    original_marker_position_diff = knee_c3d.get_position(marker_names)[:3, :, :] - original_markers_reconstructed
    original_marker_tracking_error = np.sum(original_marker_position_diff**2)

    new_markers_reconstructed = score_model.markers_in_global(new_optimal_q)[:3, markers_index, :]
    new_marker_position_diff = knee_c3d.get_position(marker_names)[:3, :, :] - new_markers_reconstructed
    new_marker_tracking_error = np.sum(new_marker_position_diff**2)

    npt.assert_almost_equal(original_marker_tracking_error, 4.705350581055244, decimal=2)
    if initialize_whole_trial_reconstruction:
        npt.assert_almost_equal(new_marker_tracking_error, 2.956894901165191, decimal=2)
    else:
        npt.assert_almost_equal(new_marker_tracking_error, 2.995276361344552, decimal=2)
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

    animate = False  # Debugging purpose only

    np.random.seed(42)

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    leg_model_filepath = parent_path + "/examples/models/leg_with_ghost_parents.bioMod"
    score_biomod_filepath = parent_path + "/examples/models/leg_with_ghost_parents_score.bioMod"

    hip_functional_trial_path = parent_path + "/examples/data/functional_trials/right_hip.c3d"
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    hip_c3d = C3dData(
        hip_functional_trial_path, first_frame=250, last_frame=349
    )  # Marker inversion happening after the 500th frame in the example data!
    knee_c3d = C3dData(knee_functional_trial_path, first_frame=300, last_frame=399)

    # Read the .bioMod file
    scaled_model = BiomechanicalModelReal().from_biomod(filepath=leg_model_filepath)
    marker_weights = NamedList[MarkerWeight]()
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
            functional_trial=hip_c3d,
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
            functional_trial=knee_c3d,
            parent_name="femur_r",
            child_name="tibia_r",
            parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
            child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
            joint_center_markers=["RLFE", "RMFE"],
            distal_markers=["RLM", "RSPH"],
            expected_rotation_axis_orientation=Axis("right_knee_sara", "RLFE", "RMFE"),
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
    # The translation is the result from SCoRE (and should not change)
    npt.assert_almost_equal(
        score_model.segments["femur_r_parent_offset"].segment_coordinate_system.scs.translation,
        np.array([-0.0361767, -0.03531768, -0.01128449]),
        decimal=3,
    )
    # The rotation should not change
    npt.assert_almost_equal(
        score_model.segments["femur_r_parent_offset"].segment_coordinate_system.scs.rotation_matrix.rotation_matrix,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        decimal=5,
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
    # The translation is the result from SCoRE (and should not change)
    npt.assert_almost_equal(
        score_model.segments["tibia_r_parent_offset"].segment_coordinate_system.scs.translation,
        np.array([0.00075506, -0.37070545, -0.00658972]),
        decimal=3,
    )
    # The rotation is the result from SARA (and is less stable numerically)
    npt.assert_almost_equal(
        score_model.segments["tibia_r_parent_offset"].segment_coordinate_system.scs.rotation_matrix.rotation_matrix,
        np.array(
            [
                [0.99736617, -0.01657078, -0.07061256],
                [0.00918453, 0.99456928, -0.10367055],
                [0.07194699, 0.10274896, 0.99210195],
            ]
        ),
        decimal=3,
    )

    assert score_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        score_model.segments["tibia_r"].segment_coordinate_system.scs.rt_matrix,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    # Test that the original model did not change
    assert scaled_model.segments["femur_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["femur_r_parent_offset"].segment_coordinate_system.scs.rt_matrix,
        np.array(
            [
                [1.0, 0.0, 0.0, -0.067759],
                [0.0, 1.0, 0.0, -0.06335],
                [0.0, 0.0, 1.0, 0.080026],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert scaled_model.segments["femur_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["femur_r"].segment_coordinate_system.scs.rt_matrix,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert scaled_model.segments["tibia_r_parent_offset"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["tibia_r_parent_offset"].segment_coordinate_system.scs.rt_matrix,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.387741],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert scaled_model.segments["tibia_r"].segment_coordinate_system.is_in_local
    npt.assert_almost_equal(
        scaled_model.segments["tibia_r"].segment_coordinate_system.scs.rt_matrix,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
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
    npt.assert_almost_equal(original_marker_tracking_error, 0.28506843278583055, decimal=2)
    npt.assert_almost_equal(new_marker_tracking_error, 1.541524705667391, decimal=2)

    # Animate the output
    if animate:
        from pyorerun import PyoMarkers

        pyomarkers = PyoMarkers(
            data=hip_c3d.get_position(list(marker_weights.keys())),
            channels=list(marker_weights.keys()),
            show_labels=False,
        )
        visualize_modified_model_output(
            leg_model_filepath,
            score_biomod_filepath,
            original_optimal_q,
            new_optimal_q,
            pyomarkers,
        )

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

    # Animate the results
    if animate:
        from pyorerun import PyoMarkers

        pyomarkers = PyoMarkers(
            data=knee_c3d.get_position(marker_names),
            channels=marker_names,
            show_labels=False,
        )
        visualize_modified_model_output(
            leg_model_filepath,
            score_biomod_filepath,
            original_optimal_q,
            new_optimal_q,
            pyomarkers,
        )

    markers_index = scaled_model.markers_indices(marker_names)

    original_markers_reconstructed = scaled_model.markers_in_global(original_optimal_q)[:3, markers_index, :]
    original_marker_position_diff = knee_c3d.get_position(marker_names)[:3, :, :] - original_markers_reconstructed
    original_marker_tracking_error = np.sum(original_marker_position_diff**2)

    new_markers_reconstructed = score_model.markers_in_global(new_optimal_q)[:3, markers_index, :]
    new_marker_position_diff = knee_c3d.get_position(marker_names)[:3, :, :] - new_markers_reconstructed
    new_marker_tracking_error = np.sum(new_marker_position_diff**2)

    # The error is worse because it is a unit test (for the tests to run quickly)
    npt.assert_almost_equal(original_marker_tracking_error, 0.8846482105592899, decimal=2)
    npt.assert_almost_equal(new_marker_tracking_error, 0.9470458799111221, decimal=2)

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
    c3d_data = C3dData(knee_functional_trial_path, first_frame=300, last_frame=399)

    # Create a test instance
    parent_name = "femur_r"
    child_name = "tibia_r"
    parent_marker_names = ["RGT", "RTHI1", "RTHI2", "RTHI3"]
    child_marker_names = ["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"]
    rsi = Score(
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
        Score(
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
        Score(
            c3d_data,
            parent_name,
            "child_translation",
            parent_marker_names,
            child_marker_names,
        )

    # Test with valid marker movement
    rsi._check_marker_functional_trial_file()  # Should not raise an error

    # Test with no markers
    with pytest.raises(
        RuntimeError,
        match=r"The functional trial file does not contain any frame. Please check the trial again.",
    ):
        rsi_no_markers = Score(
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
        rsi = Score(
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
    c3d_data = C3dData(knee_functional_trial_path, first_frame=300, last_frame=399)

    # Create a test instance
    parent_name = "femur_r"
    child_name = "tibia_r"
    parent_marker_names = ["RGT", "RTHI1", "RTHI2", "RTHI3"]
    child_marker_names = ["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"]
    rsi = Score(
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
    with pytest.raises(
        RuntimeError,
        match=r"The marker_names \['marker1'\] do not match the number of markers 2.",
    ):
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
    knee_c3d = C3dData(knee_functional_trial_path, first_frame=300, last_frame=399)

    child_name = "tibia_r"
    parent_name = "femur_r"
    scaled_model = BiomechanicalModelReal().from_biomod(
        filepath=leg_model_filepath,
    )
    sara = Sara(
        functional_trial=knee_c3d,
        parent_name=parent_name,
        child_name=child_name,
        parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
        child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
        joint_center_markers=["RLFE", "RMFE"],
        distal_markers=["RLM", "RSPH"],
        is_longitudinal_axis_from_jcs_to_distal_markers=False,
        expected_rotation_axis_orientation=Axis("right_knee_sara", "RMFE", "RLFE"),
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


def test_original_rotation_axis_axis():

    np.random.seed(42)

    # Set up
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    knee_c3d = C3dData(knee_functional_trial_path, first_frame=300, last_frame=399)

    child_name = "tibia_r"
    parent_name = "femur_r"
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

    # Test the longitudinal axis calculation
    sara = Sara(
        functional_trial=knee_c3d,
        parent_name=parent_name,
        child_name=child_name,
        parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
        child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
        joint_center_markers=["RLFE", "RMFE"],
        distal_markers=["RLM", "RSPH"],
        is_longitudinal_axis_from_jcs_to_distal_markers=False,
        expected_rotation_axis_orientation=Axis("right_knee_sara", "RMFE", "RLFE"),
        initialize_whole_trial_reconstruction=False,
        animate_rt=False,
    )

    original_axis_global, original_axis_local = sara._original_rotation_axis(scaled_model)
    npt.assert_almost_equal(
        original_axis_global.reshape(
            3,
        ),
        np.array([0.01857409, -0.09155144, 0.01915594]),
        decimal=6,
    )
    npt.assert_almost_equal(
        original_axis_local.reshape(
            3,
        ),
        np.array([0.01996359, 0.00347809, 0.09318246]),
        decimal=6,
    )

    joint_center_tool = JointCenterTool(scaled_model, animate_reconstruction=False)
    joint_center_tool.add(sara)
    score_model = joint_center_tool.replace_joint_centers(marker_weights)
    rt_tibia = score_model.segments["tibia_r"].segment_coordinate_system.scs
    npt.assert_almost_equal(
        rt_tibia.rotation_matrix.rotation_matrix,
        np.array(
            [
                [0.99778346, 0.0657014, -0.01055861],
                [-0.06648268, 0.99106219, -0.11565378],
                [0.00286562, 0.1160994, 0.99323347],
            ]
        ),
        decimal=4,
    )
    npt.assert_almost_equal(rt_tibia.translation, np.array([0.00498373, -0.37616619, -0.0030206]), decimal=4)

    # Test the other direction
    sara = Sara(
        functional_trial=knee_c3d,
        parent_name=parent_name,
        child_name=child_name,
        parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
        child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
        joint_center_markers=["RLFE", "RMFE"],
        distal_markers=["RLM", "RSPH"],
        is_longitudinal_axis_from_jcs_to_distal_markers=False,
        expected_rotation_axis_orientation=Axis("right_knee_sara", "RLFE", "RMFE"),
        initialize_whole_trial_reconstruction=False,
        animate_rt=False,
    )
    original_axis_global, original_axis_local = sara._original_rotation_axis(scaled_model)
    npt.assert_almost_equal(
        original_axis_global.reshape(
            3,
        ),
        np.array([-0.01857409, 0.09155144, -0.01915594]),
        decimal=6,
    )
    npt.assert_almost_equal(
        original_axis_local.reshape(
            3,
        ),
        np.array([-0.01996359, -0.00347809, -0.09318246]),
        decimal=6,
    )

    joint_center_tool = JointCenterTool(scaled_model, animate_reconstruction=False)
    joint_center_tool.add(sara)
    score_model = joint_center_tool.replace_joint_centers(marker_weights)
    rt_tibia_reverse = score_model.segments["tibia_r"].segment_coordinate_system.scs
    npt.assert_almost_equal(
        rt_tibia.rotation_matrix.rotation_matrix,
        np.array(
            [
                [0.99778346, 0.0657014, -0.01055861],
                [-0.06648268, 0.99106219, -0.11565378],
                [0.00286562, 0.1160994, 0.99323347],
            ]
        ),
        decimal=4,
    )
    npt.assert_almost_equal(rt_tibia.translation, np.array([0.00498373, -0.37616619, -0.0030206]), decimal=4)
    # Make sure the axis are in opposite direction
    npt.assert_almost_equal(
        rt_tibia_reverse.rotation_matrix.rotation_matrix @ np.array([0, 0, 1]),
        -(rt_tibia.rotation_matrix.rotation_matrix @ np.array([0, 0, 1])),
        decimal=6,
    )


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


def test_get_svd():
    """Test the get_svd function"""
    np.random.seed(42)

    # Create test RT matrices rotating about all 3 axes
    nb_frames = 10
    rt_parent = RotoTransMatrixTimeSeries(nb_frames)
    rt_child = RotoTransMatrixTimeSeries(nb_frames)

    for i in range(nb_frames):
        rt_parent[i] = RotoTransMatrix.from_euler_angles_and_translation(
            "xyz", np.array([0.31 * i + 0.05, 0.17 * i + 0.11, 0.23 * i + 0.02]), np.array([0, 0, 0])
        )
        rt_child[i] = RotoTransMatrix.from_euler_angles_and_translation(
            "xyz", np.array([0.19 * i + 0.13, 0.29 * i + 0.07, 0.11 * i + 0.17]), np.array([0.1, 0.2, 0.3])
        )

    # Test get_svd
    U, S, V, b = get_svd(rt_parent, rt_child)

    # Check values
    assert U.shape == (30, 6)
    assert S.shape == (6,)
    assert V.shape == (6, 6)
    assert b.shape == (30,)

    # U is too sensitive to be tested :(
    npt.assert_almost_equal(
        S,
        np.array([4.42686475, 4.15458227, 4.11020407, 1.76244788, 1.65512723, 0.6347192]),
        decimal=3,
    )

    # Fix the sign of each column of V, otherwise this would flip depending on the LAPACK backend even though the
    # underlying subspace is identical.
    V_sign = V.copy()
    for i_col in range(V_sign.shape[1]):
        i_max = np.argmax(np.abs(V_sign[:, i_col]))
        if V_sign[i_max, i_col] < 0:
            V_sign[:, i_col] *= -1
    npt.assert_almost_equal(
        V_sign[0, :],
        np.array([0.44883262, 0.32753639, -0.43734333, 0.43734333, -0.32753639, -0.44883262]),
        decimal=3,
    )

    npt.assert_almost_equal(b[:6], np.array([-0.1, -0.2, -0.3, -0.1, -0.2, -0.3]), decimal=3)


def test_score_perform_algorithm():
    """Test Score.perform_algorithm"""
    np.random.seed(42)

    # Create test RT matrices with a known center of rotation
    nb_frames = 20
    rt_parent = RotoTransMatrixTimeSeries(nb_frames)
    rt_child = RotoTransMatrixTimeSeries(nb_frames)

    # Known CoR in parent frame
    cor_parent_expected = np.array([0.1, 0.2, 0.3])

    for i in range(nb_frames):
        # Parent stays fixed
        rt_parent[i] = RotoTransMatrix.from_euler_angles_and_translation(
            "xyz", np.array([0, 0, 0]), np.array([0, 0, 0])
        )
        # Child rotates around the CoR
        angle = i * 0.1
        rt_child[i] = RotoTransMatrix.from_euler_angles_and_translation(
            "xyz", np.array([angle, angle, 0]), cor_parent_expected
        )

    # Run Score algorithm
    cor_global, cor_parent, cor_child, rt_parent_out, rt_child_out = Score.perform_algorithm(
        rt_parent, rt_child, recursive_outlier_removal=False
    )

    # Check that CoR is close to expected
    npt.assert_almost_equal(cor_parent, cor_parent_expected, decimal=6)

    # Check that output RT matrices have same length
    assert len(rt_parent_out) == len(rt_child_out)


def test_sara_perform_algorithm():
    """Test Sara.perform_algorithm"""
    np.random.seed(42)

    # Create test RT matrices with a known axis of rotation
    nb_frames = 20
    rt_parent = RotoTransMatrixTimeSeries(nb_frames)
    rt_child = RotoTransMatrixTimeSeries(nb_frames)

    # Known AoR and CoR
    aor_expected = np.array([1, 0, 0])
    cor_expected = np.array([0.1, 0.2, 0.3])

    for i in range(nb_frames):
        # Parent stays fixed
        rt_parent[i] = RotoTransMatrix.from_euler_angles_and_translation(
            "xyz", np.array([0, 0, 0]), np.array([0, 0, 0])
        )
        # Child rotates around X-axis through CoR
        angle = i * 0.1
        rt_child[i] = RotoTransMatrix.from_euler_angles_and_translation("xyz", np.array([angle, 0, 0]), cor_expected)

    # Run Sara algorithm
    (
        aor_global,
        aor_parent,
        aor_child,
        cor_global,
        cor_parent,
        cor_child,
        rt_parent_out,
        rt_child_out,
    ) = Sara.perform_algorithm(rt_parent, rt_child, recursive_outlier_removal=False)

    # Check that AoR is close to expected (X-axis)
    aor_global_normalized = aor_global / np.linalg.norm(aor_global)
    npt.assert_almost_equal(aor_global_normalized, aor_expected, decimal=1)

    # Check that CoR is close to expected on the unambiguous axes
    npt.assert_almost_equal(cor_parent[1:], cor_expected[1:], decimal=6)

    # Check that output RT matrices have same length
    assert len(rt_parent_out) == len(rt_child_out)


def test_sara_perfect_hinge_assigns_local_axes_to_the_correct_segments():
    """
    SARA should return the parent AoR in the parent local frame and the child AoR
    in the child local frame when the input RTs are exact model segment frames.
    """
    (
        rt_parent,
        rt_child,
        _functional_data,
        _static_data,
        _parent_names,
        _child_names,
        parent_axis_local,
        child_axis_local,
        parent_hinge_local,
        child_hinge_local,
        hinge_origin_global,
    ) = _perfect_hinge_rt_and_markers()

    (
        _aor_global,
        aor_parent_local,
        aor_child_local,
        _cor_global,
        cor_parent_local,
        cor_child_local,
        _rt_parent_out,
        _rt_child_out,
    ) = Sara.perform_algorithm(
        rt_parent,
        rt_child,
        origin_positions_global=hinge_origin_global,
        recursive_outlier_removal=False,
    )

    assert abs(float(np.dot(aor_parent_local, parent_axis_local))) > 1.0 - 1e-10
    assert abs(float(np.dot(aor_child_local, child_axis_local))) > 1.0 - 1e-10
    parent_global_directions = _global_direction(rt_parent, aor_parent_local)
    child_global_directions = _global_direction(rt_child, aor_child_local)
    global_alignment = np.sum(parent_global_directions * child_global_directions, axis=0)
    npt.assert_array_less(1.0 - 1e-10, np.abs(global_alignment))
    npt.assert_almost_equal(cor_parent_local.reshape(3), parent_hinge_local, decimal=10)
    npt.assert_almost_equal(cor_child_local.reshape(3), child_hinge_local, decimal=10)


def test_sara_perfect_marker_hinge_uses_static_technical_frames_after_rigidify():
    """
    With marker-based rigidification, SARA local axes are expressed in the
    static technical marker frames. This guards against swapping parent and child
    columns while documenting why the child local axis can look different from a
    nominal model-local axis.
    """
    (
        _rt_parent,
        _rt_child,
        functional_data,
        static_data,
        parent_names,
        child_names,
        parent_axis_local,
        _child_axis_local,
        _parent_hinge_local,
        _child_hinge_local,
        _hinge_origin_global,
    ) = _perfect_hinge_rt_and_markers()

    parent_functional_data = functional_data.get_partial_dict_data(parent_names)
    child_functional_data = functional_data.get_partial_dict_data(child_names)
    parent_static_data = static_data.get_partial_dict_data(parent_names)
    child_static_data = static_data.get_partial_dict_data(child_names)
    rt_parent = SegmentCoordinateSystemUtils.rigidify(
        functional_data=parent_functional_data,
        static_data=parent_static_data,
    )
    rt_child = SegmentCoordinateSystemUtils.rigidify(
        functional_data=child_functional_data,
        static_data=child_static_data,
    )

    (
        _aor_global,
        aor_parent_local,
        aor_child_local,
        _cor_global,
        _cor_parent_local,
        _cor_child_local,
        _rt_parent_out,
        _rt_child_out,
    ) = Sara.perform_algorithm(
        rt_parent,
        rt_child,
        recursive_outlier_removal=False,
    )

    assert abs(float(np.dot(aor_parent_local, parent_axis_local))) > 1.0 - 1e-10
    assert abs(float(np.dot(aor_child_local, parent_axis_local))) > 1.0 - 1e-10
    parent_global_directions = _global_direction(rt_parent, aor_parent_local)
    child_global_directions = _global_direction(rt_child, aor_child_local)
    global_alignment = np.sum(parent_global_directions * child_global_directions, axis=0)
    npt.assert_array_less(1.0 - 1e-10, np.abs(global_alignment))


def test_sara_perform_algorithm_with_origin_positions():
    """Test Sara.perform_algorithm with origin_positions_global"""

    np.random.seed(42)

    # Create test RT matrices with a known axis of rotation
    nb_frames = 20
    rt_parent = RotoTransMatrixTimeSeries(nb_frames)
    rt_child = RotoTransMatrixTimeSeries(nb_frames)

    # Known AoR and CoR
    aor_expected = np.array([1, 0, 0])
    cor_expected = np.array([0.1, 0.2, 0.3])

    for i in range(nb_frames):
        # Parent stays fixed
        rt_parent[i] = RotoTransMatrix.from_euler_angles_and_translation(
            "xyz", np.array([0, 0, 0]), np.array([1, 0, 0])
        )
        # Child rotates around X-axis through CoR
        angle = i * 0.1
        rt_child[i] = RotoTransMatrix.from_euler_angles_and_translation("xyz", np.array([angle, 0, 0]), cor_expected)

    # Run Sara algorithm
    (
        aor_global,
        aor_parent,
        aor_child,
        cor_global,
        cor_parent,
        cor_child,
        rt_parent_out,
        rt_child_out,
    ) = Sara.perform_algorithm(
        rt_parent,
        rt_child,
        origin_positions_global=np.repeat(cor_expected[:, np.newaxis], nb_frames, axis=1),
        recursive_outlier_removal=False,
    )

    # Check that AoR is close to expected (X-axis)
    aor_global_normalized = aor_global / np.linalg.norm(aor_global)
    npt.assert_almost_equal(aor_global_normalized, aor_expected, decimal=1)

    # Check that CoR is close to expected on all axes
    npt.assert_almost_equal(cor_global[1:], cor_expected[1:], decimal=1)
    npt.assert_almost_equal(cor_global, np.array([0.25365385, 0.17884615, 0.26826923]), decimal=6)


def test_joint_coordinate_modifier():
    """Test JointCoordinateModifier class"""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    original_model = BiomechanicalModelReal().from_biomod(filepath=leg_model_filepath)

    # Create modifier
    modifier = JointCoordinateModifier(original_model)

    # Test that new_model is a copy
    assert modifier.new_model is not modifier.original_model
    assert modifier.new_model.segments["femur_r"].name == original_model.segments["femur_r"].name

    # Test set_new_model
    new_model = BiomechanicalModelReal().from_biomod(filepath=leg_model_filepath)
    modifier.set_new_model(new_model)
    assert modifier.new_model is new_model


def test_check_marker_labeling():
    """Test check_marker_labeling method"""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    c3d_data = C3dData(knee_functional_trial_path, first_frame=300, last_frame=399)

    parent_name = "femur_r"
    child_name = "tibia_r"
    parent_marker_names = ["RGT", "RTHI1", "RTHI2", "RTHI3"]
    child_marker_names = ["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"]

    score = Score(
        c3d_data,
        parent_name,
        child_name,
        parent_marker_names,
        child_marker_names,
    )

    # Set up marker data
    score.parent_markers_global = c3d_data.get_position(parent_marker_names)
    score.child_markers_global = c3d_data.get_position(child_marker_names)

    # Should not raise error with good data
    score.check_marker_labeling()

    # Test with bad data (large jump)
    bad_data = c3d_data.get_position(parent_marker_names).copy()
    bad_data[:, 0, 10] += 0.1  # Add large jump
    score.parent_markers_global = bad_data

    with pytest.raises(RuntimeError, match="The parent markers .* seem to be mislabeled"):
        score.check_marker_labeling()


def test_check_marker_positions():
    """Test check_marker_positions method"""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    c3d_data = C3dData(knee_functional_trial_path, first_frame=300, last_frame=399)

    parent_name = "femur_r"
    child_name = "tibia_r"
    parent_marker_names = ["RGT", "RTHI1", "RTHI2", "RTHI3"]
    child_marker_names = ["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"]

    score = Score(
        c3d_data,
        parent_name,
        child_name,
        parent_marker_names,
        child_marker_names,
    )

    # Set up marker data
    score.parent_static_markers_in_global = c3d_data.get_position(parent_marker_names)[:, :, 0:1]
    score.child_static_markers_in_global = c3d_data.get_position(child_marker_names)[:, :, 0:1]
    score.parent_markers_global = c3d_data.get_position(parent_marker_names)
    score.child_markers_global = c3d_data.get_position(child_marker_names)

    # Should not raise error with consistent data
    score.check_marker_positions()

    # Test with inconsistent data (marker moved between trials)
    bad_static = score.parent_static_markers_in_global.copy()
    bad_static[:, 0, 0] += 0.1  # Move marker significantly
    score.parent_static_markers_in_global = bad_static

    with pytest.raises(
        RuntimeError,
        match="There is a difference in marker placement of more than 1cm between the static trial and the functional trial",
    ):
        score.check_marker_positions()


def test_extract_scs_from_axis():
    """Test Sara._extract_scs_from_axis"""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    knee_c3d = C3dData(knee_functional_trial_path, first_frame=300, last_frame=399)

    scaled_model = BiomechanicalModelReal().from_biomod(filepath=leg_model_filepath)

    sara = Sara(
        functional_trial=knee_c3d,
        parent_name="femur_r",
        child_name="tibia_r",
        parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
        child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
        joint_center_markers=["RLFE", "RMFE"],
        distal_markers=["RLM", "RSPH"],
        is_longitudinal_axis_from_jcs_to_distal_markers=False,
        expected_rotation_axis_orientation=Axis("right_knee_sara", "RLFE", "RMFE"),
    )

    # Create test axis data
    aor_local = np.array([0, 0, 1], dtype=np.float64)  # Z-axis
    joint_center = np.array([0.1, 0.2, 0.3, 1], dtype=np.float64).reshape(4, 1)
    longitudinal_axis = np.array([0, 1, 0, 1], dtype=np.float64).reshape(4, 1)  # Y-axis

    # Extract SCS
    scs = sara._extract_scs_from_axis(scaled_model, aor_local, joint_center, longitudinal_axis)

    # Check that result is a valid RT matrix
    assert isinstance(scs, RotoTransMatrix)
    npt.assert_almost_equal(scs.translation, joint_center[:3, 0])

    # Check that rotation matrix is orthonormal
    rot = scs.rotation_matrix.rotation_matrix
    npt.assert_almost_equal(rot @ rot.T, np.eye(3), decimal=10)
    npt.assert_almost_equal(np.linalg.det(rot), 1.0)

    # Check that Z-axis is the AoR
    npt.assert_almost_equal(rot[:, 2], aor_local)


def test_score_with_nan_frames():
    """Test Score algorithm with NaN frames"""
    np.random.seed(42)

    # Create test RT matrices with some NaN frames
    nb_frames = 20
    rt_parent = RotoTransMatrixTimeSeries(nb_frames)
    rt_child = RotoTransMatrixTimeSeries(nb_frames)

    cor_expected = np.array([0.1, 0.2, 0.3])

    for i in range(nb_frames):
        if i == 5 or i == 10:  # Add NaN frames
            rt_parent[i] = RotoTransMatrix.from_rt_matrix(np.ones((4, 4)) * np.nan)
            rt_child[i] = RotoTransMatrix.from_rt_matrix(np.ones((4, 4)) * np.nan)
        else:
            rt_parent[i] = RotoTransMatrix.from_euler_angles_and_translation(
                "xyz", np.array([0, 0, 0]), np.array([0, 0, 0])
            )
            angle = i * 0.1
            rt_child[i] = RotoTransMatrix.from_euler_angles_and_translation(
                "xyz", np.array([angle, 0, 0]), cor_expected
            )

    # Run Score algorithm
    cor_global, cor_parent, cor_child, rt_parent_out, rt_child_out = Score.perform_algorithm(
        rt_parent, rt_child, recursive_outlier_removal=False
    )

    # Check that algorithm still works
    assert not np.any(np.isnan(cor_parent))
    assert len(rt_parent_out) == len(rt_child_out)


def test_sara_with_longitudinal_axis_direction():
    """Test Sara with different longitudinal axis directions"""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    knee_c3d = C3dData(knee_functional_trial_path, first_frame=300, last_frame=399)

    scaled_model = BiomechanicalModelReal().from_biomod(filepath=leg_model_filepath)

    # Test with is_longitudinal_axis_from_jcs_to_distal_markers=True
    sara_forward = Sara(
        functional_trial=knee_c3d,
        parent_name="femur_r",
        child_name="tibia_r",
        parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
        child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
        joint_center_markers=["RLFE", "RMFE"],
        distal_markers=["RLM", "RSPH"],
        is_longitudinal_axis_from_jcs_to_distal_markers=True,
        expected_rotation_axis_orientation=Axis("right_knee_sara", "RLFE", "RMFE"),
    )

    joint_center_forward, long_axis_forward = sara_forward._longitudinal_axis(scaled_model)

    # Test with is_longitudinal_axis_from_jcs_to_distal_markers=False
    sara_backward = Sara(
        functional_trial=knee_c3d,
        parent_name="femur_r",
        child_name="tibia_r",
        parent_marker_names=["RGT", "RTHI1", "RTHI2", "RTHI3"],
        child_marker_names=["RATT", "RLM", "RSPH", "RLEG1", "RLEG2", "RLEG3"],
        joint_center_markers=["RLFE", "RMFE"],
        distal_markers=["RLM", "RSPH"],
        is_longitudinal_axis_from_jcs_to_distal_markers=False,
        expected_rotation_axis_orientation=Axis("right_knee_sara", "RLFE", "RMFE"),
    )

    joint_center_backward, long_axis_backward = sara_backward._longitudinal_axis(scaled_model)

    # Joint centers should be the same
    npt.assert_almost_equal(joint_center_forward, joint_center_backward)

    # Longitudinal axes should be opposite
    npt.assert_almost_equal(long_axis_forward[:3, 0], -long_axis_backward[:3, 0])
