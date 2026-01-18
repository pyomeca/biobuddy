import os
import numpy as np

from biobuddy import (
    BiomechanicalModelReal,
    MuscleType,
    MuscleStateType,
    C3dData,
    MarkerWeight,
    JointCenterTool,
    Score,
    Sara,
    ScaleTool,
)

from biobuddy.utils.named_list import NamedList

RUN_GUI = False  # Put this to true if you want to see the GUI while debugging
# Otherwise, leave it to False for automated testing (which does nothing...)


def test_model_animate():
    """Test the animate method."""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    leg_model = BiomechanicalModelReal().from_biomod(filepath=leg_filepath)

    # Test that animate runs without error
    if RUN_GUI:
        leg_model.animate()


def test_inverse_kinematics_animate():
    """Test the inverse_kinematics method."""

    np.random.seed(42)

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    hip_functional_trial_path = parent_path + "/examples/data/functional_trials/right_hip.c3d"
    hip_c3d = C3dData(hip_functional_trial_path, first_frame=1, last_frame=49)

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

    if RUN_GUI:
        original_optimal_q, _ = scaled_model.inverse_kinematics(
            marker_positions=hip_c3d.get_position(list(marker_weights.keys()))[:3, :, :],
            marker_names=list(marker_weights.keys()),
            marker_weights=marker_weights,
            method="lm",
            animate_reconstruction=True,
        )


def test_animate_joint_center_tool():
    """
    Test the JointCenterTool animate method.
    Should open 3 windows:
        - Trial reconstruction
        - Hip joint center reconstruction
        - Knee joint center reconstruction
    """
    np.random.seed(42)

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    hip_functional_trial_path = parent_path + "/examples/data/functional_trials/right_hip.c3d"
    knee_functional_trial_path = parent_path + "/examples/data/functional_trials/right_knee.c3d"
    hip_c3d = C3dData(hip_functional_trial_path, first_frame=1, last_frame=49)
    knee_c3d = C3dData(knee_functional_trial_path, first_frame=30, last_frame=82)

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

    joint_center_tool = JointCenterTool(scaled_model, animate_reconstruction=True)
    # Hip Right
    joint_center_tool.add(
        Score(
            functional_trial=hip_c3d,
            parent_name="pelvis",
            child_name="femur_r",
            parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
            child_marker_names=["RLFE", "RMFE", "RTHI1", "RTHI2", "RTHI3"],
            initialize_whole_trial_reconstruction=True,
            animate_rt=True,
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
            initialize_whole_trial_reconstruction=True,
            animate_rt=True,
        )
    )

    if RUN_GUI:
        score_model = joint_center_tool.replace_joint_centers(marker_weights)


def test_scaling_animation():
    np.random.seed(42)

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_relative_path = "Geometry_cleaned"
    osim_filepath = parent_path + "/examples/models/wholebody.osim"
    xml_filepath = parent_path + "/examples/models/wholebody.xml"
    static_filepath = parent_path + "/examples/data/static_rotated.c3d"

    c3d_data = C3dData(
        c3d_path=static_filepath,
        first_frame=0,
        last_frame=136,
    )

    # --- Scale in BioBuddy --- #
    original_model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=cleaned_relative_path,
    )
    original_model.fix_via_points()
    scale_tool = ScaleTool(original_model=original_model).from_xml(filepath=xml_filepath)

    if RUN_GUI:
        scaled_model = scale_tool.scale(
            static_trial=c3d_data,
            mass=69.2,
            q_regularization_weight=0.1,
            make_static_pose_the_models_zero=True,
            visualize_optimal_static_pose=True,
        )


def test_that_temporary_folder_is_accessible():

    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_model_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/biobuddy/temporary_models"
    temporary_model_path = current_path + "/temporary.bioMod"

    scaled_model = BiomechanicalModelReal().from_biomod(
        filepath=leg_model_filepath,
    )
    scaled_model.to_biomod(temporary_model_path)
