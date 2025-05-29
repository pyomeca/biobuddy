"""
This example shows how to use the SCoRE and SARA algorithms to identify the joint center of rotation and axis of rotation, respectively.
The joints are then modified accordingly in the output model.
Please note that this feature requires to acquire functional trials.
TODO: make videos of the functional trials, since it does not seem to exist online yet !
"""

import logging
from pathlib import Path
import numpy as np
import ezc3d
from pyomeca import Markers

from biobuddy import (
    BiomechanicalModelReal,
    MuscleType,
    MuscleStateType,
    MeshParser,
    MeshFormat,
    ScaleTool,
    C3dData,
    RotoTransMatrix,
    MarkerReal,
    JointCenterTool,
    Sara,
    Score,
)


def main(visualization):

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ],
    )

    # Paths
    current_path_file = Path(__file__).parent
    osim_filepath = f"{current_path_file}/models/wholebody.osim"
    geometry_path = f"{current_path_file}/../../external/opensim-models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/models/Geometry_cleaned"
    xml_filepath = f"{current_path_file}/models/wholebody_modified.xml"
    scaled_biomod_filepath = f"{current_path_file}/models/wholebody_scaled_ECH.bioMod"
    score_biomod_filepath = f"{current_path_file}/models/wholebody_score_ECH.bioMod"
    static_filepath = f"{current_path_file}/data/anat_pose_ECH.c3d"
    score_directory = f"{current_path_file}/data/functional_trials"

    static_c3d = ezc3d.c3d(static_filepath, extract_forceplat_data=True)
    summed_force = static_c3d["data"]["platform"][0]["force"] + static_c3d["data"]["platform"][0]["force"]
    mass = np.median(np.linalg.norm(summed_force[:, 2000:9000], axis=0)) / 9.81
    rt_method = "optimization"

    # # Convert the vtp files
    # mesh = MeshParser(geometry_folder=geometry_path)
    # mesh.process_meshes(fail_on_error=False)
    # mesh.write(geometry_cleaned_path, format=MeshFormat.VTP)

    # Read the .osim file
    original_osim_model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir="Geometry_cleaned",
    )
    # original_osim_model.segments["ground"].segment_coordinate_system.scs = np.array(
    #     [
    #         [1.000000, 0.000000, 0.000000, 0.000000],
    #         [0.000000, 0.000000, -1.00000, 0.000000],
    #         [0.000000, 1.000000, 0.000000, 0.000000],
    #         [0.000000, 0.000000, 0.000000, 1.000000],
    #     ]  # Reset the ground to the upward Z axis + standing in the same orientation as the subject
    # )

    # Scale the model
    scale_tool = ScaleTool(original_model=original_osim_model).from_xml(filepath=xml_filepath)
    scaled_model = scale_tool.scale(
        filepath=static_filepath,
        first_frame=500,
        last_frame=600,
        mass=mass,
        q_regularization_weight=0.01,
        make_static_pose_the_models_zero=True,
        visualize_optimal_static_pose=False,
    )
    marker_weights = scale_tool.marker_weights

    # Add to the model the new technical markers that will be used to identify the joint centers
    technical_marker_to_add = {
        "femur_r": ["RTHI1", "RTHI2", "RTHI3"],
        "femur_l": ["LTHI1", "LTHI2", "LTHI3"],
        "tibia_r": ["RLEG1", "RLEG2", "RLEG3"],
        "tibia_l": ["LLEG1", "LLEG2", "LLEG3"],
        "humerus_r": ["RAMR1", "RARM2", "RARM3"],
        "radius_r": ["RFARM1", "RFARM2", "RFARM3"],
        "humerus_l": ["LARM1", "LARM2", "LARM3"],
        "radius_l": ["LFARM1", "LFARM2", "LFARM3"],
    }

    jcs_in_global = scaled_model.forward_kinematics()
    c3d_data = C3dData(static_filepath, first_frame=500, last_frame=600)
    for segment_name in technical_marker_to_add.keys():
        for marker in technical_marker_to_add[segment_name]:
            position_in_global = c3d_data.mean_marker_position(marker)
            rt = RotoTransMatrix()
            rt.from_rt_matrix(jcs_in_global[segment_name])
            position_in_local = rt.inverse @ position_in_global
            scaled_model.segments[segment_name].add_marker(
                MarkerReal(
                    name=marker,
                    parent_name=segment_name,
                    position=position_in_local,
                    is_anatomical=False,
                    is_technical=True,
                )
            )
            marker_weights[marker] = 5.0
    scaled_model.to_biomod(scaled_biomod_filepath)

    # ---------- ECH ---------- #
    # Move the model's joint centers
    joint_center_tool = JointCenterTool(scaled_model, animate_reconstruction=True)
    # Hip Right
    joint_center_tool.add(
        Score(
            filepath=f"{score_directory}/right_hip.c3d",
            parent_name="pelvis",
            child_name="femur_r",
            parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
            child_marker_names=["RLFE", "RMFE"] + technical_marker_to_add["femur_r"],
            first_frame=1,
            last_frame=500,  # Marker inversion happening after this frame in the example data!
            initialize_whole_trial_reconstruction=False,  # True,
            animate_rt=False,
        )
    )
    joint_center_tool.add(
        Sara(
            filepath=f"{score_directory}/right_knee.c3d",
            parent_name="femur_r",
            child_name="tibia_r",
            parent_marker_names=["RGT"] + technical_marker_to_add["femur_r"],
            child_marker_names=["RATT", "RLM", "RSPH"] + technical_marker_to_add["tibia_r"],
            joint_center_markers=["RLFE", "RMFE"],
            distal_markers=["RLM", "RSPH"],
            is_longitudinal_axis_from_jcs_to_distal_markers=False,
            first_frame=300,
            last_frame=922 - 100,
            initialize_whole_trial_reconstruction=False,
            animate_rt=False,
        )
    )
    # ... add all other joints that you want to modify based on the functional trials

    score_model = joint_center_tool.replace_joint_centers(marker_weights)
    score_model.to_biomod(score_biomod_filepath)

    if visualization:
        # Compare the result visually
        t = np.linspace(0, 1, 10)
        viz = pyorerun.PhaseRerun(t)
        q = np.zeros((42, 10))

        # Add the experimental markers from the static trial
        static_exp_markers = C3dData(static_filepath, first_frame=500, last_frame=510)
        static_marker_positions = static_exp_markers.get_position(scaled_model.marker_names)
        pyomarkers = Markers(data=static_marker_positions, channels=scaled_model.marker_names)

        # SCoRE model
        viz_scaled_model = pyorerun.BiorbdModel(score_biomod_filepath)
        viz_scaled_model.options.transparent_mesh = False
        viz_scaled_model.options.show_gravity = True
        viz_scaled_model.options.show_marker_labels = False
        viz_scaled_model.options.show_center_of_mass_labels = False
        viz.add_animated_model(viz_scaled_model, q, tracked_markers=pyomarkers, show_tracked_marker_labels=False)

        # Animate
        viz.rerun_by_frame("Model output")

    return marker_weights


if __name__ == "__main__":
    try:
        import pyorerun

        visualization = True
    except:
        visualization = False

    main(visualization)
