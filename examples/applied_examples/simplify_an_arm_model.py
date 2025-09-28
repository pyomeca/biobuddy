"""
This example shows how to create a very simple model (2 DoFs + 6 muscles planar arm model) from a complex model
(26 dofs + 50 muscles 3D model).
First, many segments, muscles, and dofs are removed from the model.
Second, muscle wrapping objects are replaced with a via points that best matches the lever arm.
Third, the model is flatten into a planar model by projecting all the segments, markers, muscles, and joints onto the
XY plane.
Finally, the model is exported as a new .bioMod file that can be used in complex computations like optimal control.
"""

"""
This file was used to create the arm model used in the reaching learning example.
"""

# Define some components to remove
segments_to_remove = [
                        'thorax_parent_offset',
                        'thorax_translation',
                        'thorax_rotation_transform',
                        'thorax',
                        'clavicle_parent_offset',
                        'clavicle_translation',
                        'thorax_offset_sternoclavicular_r2',
                        'thorax_offset_sternoclavicular_r3',
                        'clavicle_rotation_2',
                        'clavicle_reset_axis',
                        'clavicle',
                        'clavphant_parent_offset',
                        'clavphant_translation',
                        'clavicle_offset_unrotscap_r3',
                        'clavicle_offset_unrotscap_r2',
                        'clavphant_rotation_2',
                        'clavphant_reset_axis',
                        'clavphant',
                        'proximal_row_parent_offset',
                         'proximal_row_translation',
                         'radius_offset_deviation',
                         'proximal_row_rotation_1',
                         'radius_offset_flexion',
                         'proximal_row_reset_axis',
                         'proximal_row_geom_2',
                         'proximal_row_geom_3',
                         'proximal_row_geom_4',
                         'proximal_row',
                         'hand_parent_offset',
                         'hand_translation',
                         'proximal_row_offset_wrist_hand_r1',
                         'proximal_row_offset_wrist_hand_r3',
                         'hand_rotation_2',
                         'hand_reset_axis',
                         'hand_geom_2',
                         'hand_geom_3',
                         'hand_geom_4',
                         'hand_geom_5',
                         'hand_geom_6',
                         'hand_geom_7',
                         'hand_geom_8',
                         'hand_geom_9',
                         'hand_geom_10',
                         'hand_geom_11',
                         'hand_geom_12',
                         'hand_geom_13',
                         'hand_geom_14',
                         'hand_geom_15',
                         'hand_geom_16',
                         'hand_geom_17',
                         'hand_geom_18',
                         'hand_geom_19',
                         'hand_geom_20',
                         'hand_geom_21',
                         'hand_geom_22',
                         'hand_geom_23',
                         'hand']
muscles_to_remove = [
        'DELT1',
        'PECM1',
        'DELT2',
        'SUPSP',
        'INFSP',
        'SUBSC',
        'TMIN',
        'TMAJ',
        'DELT3',
        'CORB',
        'PECM2',
        'PECM3',
        'LAT1',
        'LAT2',
        'LAT3',
        'ANC',
        'SUP',
        'PQ',
        'BRD',
        'PT',
        'ECRL',
        'ECRB',
        'ECU',
        'FCR',
        'FCU',
        'PL',
        'FDSL',
        'FDSR',
        'EDCL',
        'EDCR',
        'EDCM',
        'EDCI',
        'EDM',
        'FDSM',
        'FDSI',
        'FDPL',
        'FDPR',
        'FDPM',
        'FDPI',
        'EIP',
        'EPL',
        'EPB',
        'FPL',
        'APL',
    ]
segment_to_remove_rotation = [
    "clavphant_offset_acromioclavicular_r2",
    "clavphant_offset_acromioclavicular_r3",
    "clavphant_offset_acromioclavicular_r1",
    "scapula_reset_axis",
]
dofs_to_remove = [
         'ground_offset_t_x',
         'ground_offset_t_y',
         'ground_offset_t_z',
         'ground_offset_r_x',
         'ground_offset_r_y',
         'ground_offset_r_z',
         'thorax_offset_sternoclavicular_r2',
         'thorax_offset_sternoclavicular_r3',
         'clavicle_offset_unrotscap_r3',
         'clavicle_offset_unrotscap_r2',
         'clavphant_offset_acromioclavicular_r2',
         'clavphant_offset_acromioclavicular_r3',
         'clavphant_offset_acromioclavicular_r1',
         'scapula_offset_unrothum_r1',
         'scapula_offset_unrothum_r3',
         'scapula_offset_unrothum_r2',
         'scapphant_offset_elv_angle',
         'humphant_offset_shoulder_elv',
         'humphant_offset_shoulder1_r2',
         'humphant1_offset_shoulder_rot',
         'ulna_offset_pro_sup',
         'radius_offset_deviation',
         'radius_offset_flexion'
    ]
rts_to_remove = [
         'scapula_offset_unrothum_r1',
         'scapula_offset_unrothum_r3',
         'scapula_offset_unrothum_r2',
         'scapphant_reset_axis',
         'scapphant_parent_offset',
         'scapphant',
         'scapphant_offset_elv_angle',
         'humphant_rotation_1',
         'humphant_rotation_2',
         'humphant_reset_axis',
         'humphant',
         'humphant_offset_shoulder_elv',
         'humphant_offset_shoulder1_r2',
         'humphant1_rotation_2',
         'humphant1_reset_axis',
         'humphant1',
         'humphant1_offset_shoulder_rot',
         'humerus_offset_elbow_flexion',
         'humerus_rotation_1',
         'humerus_rotation_2',
         'humerus_reset_axis',
         'ulna',
         'ulna_rotation_1',
         'ulna_rotation_2',
         'ulna_reset_axis',
         'ulna_offset_pro_sup',
         'radius_rotation_1',
         'radius_rotation_2',
         'radius_reset_axis',
    ]

import logging
from pathlib import Path
import numpy as np

from biobuddy import (
    MuscleType,
    MuscleStateType,
    MeshParser,
    MeshFormat,
    BiomechanicalModelReal,
    Rotations,
    Translations,
    RotoTransMatrix,
    FlatteningTool,
)

_logger = logging.getLogger(__name__)


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
    )
    visualization_flag = True

    # Paths
    current_path_file = Path(__file__).parent
    biomod_filepath = f"{current_path_file}/../models/simple_arm_model.bioMod"
    osim_filepath = f"{current_path_file}/../models/MOBL_ARMS_41.osim"
    geometry_path = f"{current_path_file}/../models/Geometry_cleaned"

    # # Convert the vtp files
    # mesh = MeshParser(geometry_folder="Geometry")
    # mesh.process_meshes(fail_on_error=False)
    # mesh.write(geometry_path, format=MeshFormat.VTP)

    # Read the original .osim file
    model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=geometry_path,
    )
    # Fix the via points before translating to biomod as there are some conditional and moving via points
    model.fix_via_points(q=np.zeros((model.nb_q,)))

    # Removing unused segments
    for segment in segments_to_remove:
        if segment in model.segment_names:
            model.remove_segment(segment)
    model.segments["scapula_parent_offset"].parent_name = "ground"

    # Removing unused muscles
    model.remove_muscles(muscles_to_remove)
    model.update_muscle_groups()  # to remove muscle groups left empty

    # Remove all markers
    for segment in model.segments:
        segment.markers = []

    # Remove rotations between the scapula and the humerus frames
    for segment_name in segment_to_remove_rotation:
        model.segments[segment_name].segment_coordinate_system.scs = RotoTransMatrix()

    # Place the arm in T-pose
    evelation_idx = model.dof_names.index('humphant_offset_shoulder_elv')
    q_static = np.zeros((model.nb_q,))
    q_static[evelation_idx] = np.pi / 2
    model.modify_model_static_pose(q_static)

    # Removing unused degrees of freedom
    model.remove_dofs(dofs_to_remove)

    # Remove the rotation RT since we removed the dofs associated with them
    for segment in model.segments.copy():
        if segment.name in rts_to_remove:
            model.segments[segment.name].segment_coordinate_system.scs = RotoTransMatrix()

    # Displace the elbow flexion dof to the humerus segment
    model.segments["humerus_offset_elbow_flexion"].rotations = Rotations.NONE
    model.segments["humerus_offset_elbow_flexion"].dof_names = []
    model.segments["humerus_offset_elbow_flexion"].q_ranges = None
    model.segments["humerus"].rotations = Rotations.Z
    model.segments["humerus"].dof_names = ["shoulder_rotZ"]
    model.segments["ulna"].rotations = Rotations.Z
    model.segments["ulna"].dof_names = ["elbow_rotZ"]

    # Remove the segments left empty
    model.segments["scapula"].mesh_file = None
    model.update_segments()

    # Place the zero at the shoulder center
    global_jcs = model.forward_kinematics()
    shoulder_position = global_jcs["humerus"][0].translation
    model.segments["root"].segment_coordinate_system.scs.translation -= shoulder_position

    # Flatten the model (3D -> 2D)
    symmetry_tool = FlatteningTool(model, axis=Translations.Z)
    model = symmetry_tool.flatten()

    # NOTE: Please note that the meshes do not seem to be aligned with the segments in the visualization,
    # but it is a visual artifact, the inertia properties lie on the axis.
    model.segments["humerus"].mesh_file.mesh_rotation = np.array([0.15, 0, 0])
    model.segments["ulna"].mesh_file.mesh_rotation = np.array([0.25, 0, 0])
    model.segments["radius"].mesh_file.mesh_rotation = np.array([0.25, 0, 0])

    # And convert it to a .bioMod file
    model.to_biomod(biomod_filepath, with_mesh=visualization_flag)

    if visualization_flag:
        import pyorerun
        animation = pyorerun.LiveModelAnimation(biomod_filepath, with_q_charts=True)
        animation.rerun()


if __name__ == "__main__":
    main()

