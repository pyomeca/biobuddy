import os
import pytest
import opensim as osim
import shutil
from deepdiff import DeepDiff
from copy import deepcopy
import io
from contextlib import redirect_stdout

import ezc3d
import biorbd
import numpy as np
import numpy.testing as npt

from test_utils import remove_temporary_biomods, create_simple_model, compare_models
from biobuddy.utils.aliases import Point, point_to_array
from biobuddy import (
    BiomechanicalModelReal,
    MuscleType,
    MuscleStateType,
    ScaleTool,
    C3dData,
    SegmentScaling,
    SegmentWiseScaling,
    AxisWiseScaling,
    BodyWiseScaling,
    Translations,
    MarkerWeight,
    ContactReal,
    InertialMeasurementUnitReal,
    RotoTransMatrix,
)

from biobuddy.utils.named_list import NamedList


class MockC3dData:
    def __init__(self):
        self.marker_names = ["parent_marker", "parent_marker2", "child_marker", "child_marker2"]
        # Create marker positions for 10 frames
        self.all_marker_positions = np.zeros((3, 4, 10))
        # parent_marker positions
        self.all_marker_positions[:, 0, :] = np.array(
            [
                [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],  # x
                [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22],  # y
                [0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33],  # z
            ]
        )
        # parent_marker1 positions are all zeros
        # child_marker positions
        self.all_marker_positions[:, 2, :] = np.array(
            [
                [0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44],  # x
                [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55],  # y
                [0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66],  # z
            ]
        )
        # child_marker1 positions are all zeros


def convert_c3d_to_trc(c3d_filepath):
    """
    This function reads the c3d static file and converts it into a trc file that will be used to scale the model in OpenSim.
    The trc file is saved at the same place as the original c3d file.
    """
    trc_filepath = c3d_filepath.replace(".c3d", ".trc")

    c3d = ezc3d.c3d(c3d_filepath)
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]

    frame_rate = c3d["header"]["points"]["frame_rate"]
    marker_data = c3d["data"]["points"][:3, :, :] / 1000  # Convert in meters

    with open(trc_filepath, "w") as f:
        trc_file_name = os.path.basename(trc_filepath)
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_file_name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(
            "{:.2f}\t{:.2f}\t{}\t{}\tm\t{:.2f}\t{}\t{}\n".format(
                frame_rate,
                frame_rate,
                c3d["header"]["points"]["last_frame"],
                len(labels),
                frame_rate,
                c3d["header"]["points"]["first_frame"],
                c3d["header"]["points"]["last_frame"],
            )
        )
        f.write("Frame#\tTime\t" + "\t".join(labels) + "\n")
        f.write("\t\t" + "\t".join([f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(labels))]) + "\n")
        for frame in range(marker_data.shape[2]):
            time = frame / frame_rate
            frame_data = [f"{frame + 1}\t{time:.5f}"]
            for marker_idx in range(len(labels)):
                pos = marker_data[:, marker_idx, frame]
                frame_data.extend([f"{pos[0]:.5f}", f"{pos[1]:.5f}", f"{pos[2]:.5f}"])
            f.write("\t".join(frame_data) + "\n")


def visualize_model_scaling_output(scaled_model, osim_model_scaled, q, marker_names, marker_positions):
    """
    Only for debugging purposes.
    """
    biobuddy_path = "../examples/models/scaled_biobuddy.bioMod"
    osim_path = "../examples/models/scaled_osim.bioMod"
    scaled_model.to_biomod(biobuddy_path, with_mesh=True)
    osim_model_scaled.to_biomod(osim_path, with_mesh=True)

    import pyorerun

    # Compare the result visually
    t = np.linspace(0, 1, marker_positions.shape[2])
    viz = pyorerun.PhaseRerun(t)
    pyomarkers = pyorerun.PyoMarkers(data=marker_positions, marker_names=marker_names, show_labels=False)

    # Model scaled in BioBuddy
    viz_biomod_model = pyorerun.BiorbdModel(biobuddy_path)
    viz_biomod_model.options.transparent_mesh = False
    viz_biomod_model.options.show_gravity = True
    viz_biomod_model.options.show_marker_labels = False
    viz_biomod_model.options.show_center_of_mass_labels = False
    viz.add_animated_model(viz_biomod_model, q, tracked_markers=pyomarkers)

    # Model scaled in OpenSim
    viz_scaled_model = pyorerun.BiorbdModel(osim_path)
    viz_scaled_model.options.transparent_mesh = False
    viz_scaled_model.options.show_gravity = True
    viz_scaled_model.options.show_marker_labels = False
    viz_scaled_model.options.show_center_of_mass_labels = False
    viz.add_animated_model(viz_scaled_model, q)

    # Animate
    viz.rerun("Scaling comparison")

    os.remove(biobuddy_path)
    os.remove(osim_path)


def test_scaling_wholebody():

    np.random.seed(42)

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_relative_path = "Geometry_cleaned"
    osim_filepath = parent_path + "/examples/models/wholebody.osim"
    xml_filepath = parent_path + "/examples/models/wholebody.xml"
    scaled_biomod_filepath = parent_path + "/examples/models/wholebody_scaled.bioMod"
    converted_scaled_osim_filepath = parent_path + "/examples/models/wholebody_converted_scaled.bioMod"
    trc_file_path = parent_path + "/examples/data/static.trc"
    # Markers are rotated since OpenSim has Y-up and biorbd has Z-up
    static_filepath = parent_path + "/examples/data/static_rotated.c3d"
    c3d_data = C3dData(c3d_path=static_filepath)

    # --- Convert the vtp mesh files --- #
    # geometry_path = parent_path + "/external/opensim-models/Geometry"
    # cleaned_geometry_path = parent_path + "/models/Geometry_cleaned"
    # mesh_parser = MeshParser(geometry_path)
    # mesh_parser.process_meshes(fail_on_error=False)
    # mesh_parser.write(cleaned_geometry_path, MeshFormat.VTP)

    # --- Scale in opensim ---#
    # convert_c3d_to_trc(static_filepath)  # To translate c3d to trc
    shutil.copyfile(trc_file_path, parent_path + "/examples/models/static.trc")
    shutil.copyfile(xml_filepath, "wholebody.xml")
    shutil.copyfile(osim_filepath, "wholebody.osim")
    opensim_tool = osim.ScaleTool(xml_filepath)
    opensim_tool.run()

    # --- Read the model scaled in OpenSim and translate to bioMod --- #
    osim_model_scaled = BiomechanicalModelReal().from_osim(
        filepath=parent_path + "/examples/models/scaled.osim",
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=cleaned_relative_path,
    )
    osim_model_scaled.fix_via_points()
    osim_model_scaled.to_biomod(converted_scaled_osim_filepath, with_mesh=False)
    scaled_osim_model = biorbd.Model(converted_scaled_osim_filepath)

    # --- Scale in BioBuddy --- #
    original_model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=cleaned_relative_path,
    )

    # Test errors from moving muscle origin or insertion
    with pytest.raises(
        NotImplementedError,
        match=r"The muscle vas_med_r has a moving insertion. Scaling models with moving via points is not implemented yet. Please run model.fix_via_points\(\) before scaling the model.",
    ):
        scale_tool = ScaleTool(original_model=original_model).from_xml(filepath=xml_filepath)
        scaled_model = scale_tool.scale(
            static_c3d=c3d_data,
            mass=69.2,
            q_regularization_weight=0.1,
            make_static_pose_the_models_zero=False,
            visualize_optimal_static_pose=False,
        )

    original_model.fix_via_points()
    scale_tool = ScaleTool(original_model=original_model).from_xml(filepath=xml_filepath)
    scaled_model = scale_tool.scale(
        static_c3d=c3d_data,
        mass=69.2,
        q_regularization_weight=0.1,
        make_static_pose_the_models_zero=False,
        visualize_optimal_static_pose=False,
    )
    scaled_model.to_biomod(scaled_biomod_filepath, with_mesh=False)
    scaled_biorbd_model = biorbd.Model(scaled_biomod_filepath)

    # --- Test the scaling factors --- #
    marker_names = c3d_data.marker_names
    marker_positions = c3d_data.all_marker_positions[:3, :, :]
    q_zeros = np.zeros((42, marker_positions.shape[2]))
    q_random = np.random.rand(42) * 2 * np.pi

    # # For debugging
    # visualize_model_scaling_output(scaled_model, osim_model_scaled, q_zeros, marker_names, marker_positions)

    # TODO: Find out why there is a discrepancy between the OpenSim and BioBuddy scaling factors to the third decimal.
    # TODO: add the scaling factors in the osim parser and verify against its values
    scaling_factors = {
        # "pelvis": 1.02735211,  # There is a problem with the beginning of the kinematic chain
        "femur_r": 1.02529981,
        "tibia_r": 0.96815558,
        "talus_r": 0.99378823,
        "calcn_r": 1.05661329,
        "toes_r": 1.03316389,
        "torso": 1.0074835807964964,
        # "head_and_neck": 0.96361244,  # There is a problem with the end of the kinematic chain
        "humerus_r": 1.04842859,
        "ulna_r": 0.96014501,
        "radius_r": 0.93660073,
        "lunate_r": 0.95894967,
        "hand_r": 1.05154983,
        # "fingers_r": 1.90933033,  # There is a problem with the end of the kinematic chain
    }
    # This only verifies that the values did not change, not that they are good
    for segment_name, scale_factor in scaling_factors.items():
        biobuddy_scaling_factors = scale_tool.scaling_segments[segment_name].compute_scaling_factors(
            original_model, marker_positions, marker_names
        )
        npt.assert_almost_equal(biobuddy_scaling_factors.to_vector()[0, 0], scale_factor, decimal=5)
        npt.assert_almost_equal(biobuddy_scaling_factors.mass, scale_factor, decimal=5)

    # --- Test masses --- #
    # Total mass
    npt.assert_almost_equal(scaled_osim_model.mass(), 69.2, decimal=5)
    npt.assert_almost_equal(scaled_biorbd_model.mass(), 69.2, decimal=5)

    # TODO: Find out why there is a discrepancy between the OpenSim and BioBuddy scaled masses.
    # Pelvis:
    # Theoretical mass without renormalization -> 0.883668 * 11.776999999999999 = 10.406958035999999
    # Biobuddy -> 9.4381337873063
    # OpenSim -> 6.891020778193859 (seems like we are closer than Opensim !?)
    # Segment mass
    for i_segment, segment_name in enumerate(scaled_model.segments.keys()):
        if scaled_model.segments[segment_name].inertia_parameters is None:
            mass_biobuddy = 0
        else:
            mass_biobuddy = scaled_model.segments[segment_name].inertia_parameters.mass
        mass_to_biorbd = scaled_biorbd_model.segment(i_segment).characteristics().mass()
        # mass_osim = scaled_osim_model.segment(i_segment).characteristics().mass()
        npt.assert_almost_equal(mass_to_biorbd, mass_biobuddy)
        # npt.assert_almost_equal(mass_osim, mass_biobuddy)
        # npt.assert_almost_equal(mass_to_biorbd, mass_osim)
        if segment_name in scaling_factors.keys():
            original_mass = original_model.segments[segment_name].inertia_parameters.mass
            # We have to let a huge buffer here because of the renormalization
            if scaling_factors[segment_name] < 1:
                npt.assert_array_less(mass_biobuddy * 0.9, original_mass)
            else:
                npt.assert_array_less(original_mass, mass_biobuddy * 1.1)

    # CoM
    for i_segment, segment_name in enumerate(scaled_model.segments.keys()):
        print(segment_name)
        if "finger" in segment_name:
            continue
        if scaled_model.segments[segment_name].inertia_parameters is not None:
            # Zero
            com_biobuddy_0 = (
                scaled_model.segments[segment_name]
                .inertia_parameters.center_of_mass[:3]
                .reshape(
                    3,
                )
            ) + scaled_model.segment_coordinate_system_in_global(segment_name).translation
            com_to_biorbd_0 = scaled_biorbd_model.CoMbySegment(q_zeros[:, 0], i_segment).to_array()
            com_osim_0 = scaled_osim_model.CoMbySegment(q_zeros[:, 0], i_segment).to_array()
            npt.assert_almost_equal(com_to_biorbd_0, com_biobuddy_0, decimal=2)
            npt.assert_almost_equal(com_osim_0, com_biobuddy_0, decimal=2)
            npt.assert_almost_equal(com_to_biorbd_0, com_osim_0, decimal=2)
            # Random
            com_biobuddy_rand = scaled_biorbd_model.CoMbySegment(q_random, i_segment).to_array()
            com_osim_rand = scaled_osim_model.CoMbySegment(q_random, i_segment).to_array()
            npt.assert_almost_equal(com_osim_rand, com_biobuddy_rand, decimal=2)

    # Inertia
    for i_segment, segment_name in enumerate(scaled_model.segments.keys()):
        print(segment_name)
        if "finger" in segment_name:
            continue
        if scaled_model.segments[segment_name].inertia_parameters is not None:
            inertia_biobuddy = scaled_model.segments[segment_name].inertia_parameters.inertia[:3, :3]
            mass_to_biorbd = scaled_biorbd_model.segment(i_segment).characteristics().inertia().to_array()
            inertia_osim = scaled_osim_model.segment(i_segment).characteristics().inertia().to_array()
            # Large tolerance since the difference in scaling factor affects largely this value
            npt.assert_almost_equal(mass_to_biorbd, inertia_biobuddy, decimal=5)
            npt.assert_almost_equal(inertia_osim, inertia_biobuddy, decimal=1)
            npt.assert_almost_equal(mass_to_biorbd, inertia_osim, decimal=1)

    # Marker positions
    for i_marker in range(scaled_biorbd_model.nbMarkers()):
        biobuddy_scaled_marker = scaled_biorbd_model.markers(q_zeros[:, 0])[i_marker].to_array()
        osim_scaled_marker = scaled_osim_model.markers(q_zeros[:, 0])[i_marker].to_array()
        # TODO: The tolerance is large since the markers are already replaced based on the static trial.
        npt.assert_almost_equal(osim_scaled_marker, biobuddy_scaled_marker, decimal=1)

    # Via point positions
    for muscle_group in original_model.muscle_groups:
        if muscle_group.name in ["femur_r_to_tibia_r", "femur_l_to_tibia_l"]:
            # These muscle groups have a moving insertion (vas_med_r and vas_med_l), which is not supported yet
            continue

        for muscle in muscle_group.muscles:
            for via_point in muscle.via_points:
                biobuddy_scaled_via_point = (
                    scaled_model.muscle_groups[muscle_group.name]
                    .muscles[muscle.name]
                    .via_points[via_point.name]
                    .position[:3]
                )
                osim_scaled_via_point = (
                    osim_model_scaled.muscle_groups[muscle_group.name]
                    .muscles[muscle.name]
                    .via_points[via_point.name]
                    .position[:3]
                )
                npt.assert_almost_equal(biobuddy_scaled_via_point, osim_scaled_via_point, decimal=5)

            # Muscle properties
            if (
                muscle.name
                in [
                    "semiten_r",
                    "vas_med_r",
                    "vas_lat_r",
                    "med_gas_r",
                    "lat_gas_r",
                    "semiten_l",
                    "vas_med_l",
                    "vas_lat_l",
                    "med_gas_l",
                    "lat_gas_l",
                ]
                or "stern_mast" in muscle.name
            ):
                # Skipping muscles with ConditionalPathPoints and MovingPathPoints
                # Skipping the head since there is a difference in scaling
                # TODO: This could be tested if MultiplierFunction was implemented
                continue
            print(muscle.name)
            biobuddy_optimal_length = scaled_model.muscle_groups[muscle_group.name].muscles[muscle.name].optimal_length
            osim_optimal_length = osim_model_scaled.muscle_groups[muscle_group.name].muscles[muscle.name].optimal_length
            npt.assert_almost_equal(biobuddy_optimal_length, osim_optimal_length, decimal=5)
            biobuddy_tendon_slack_length = (
                scaled_model.muscle_groups[muscle_group.name].muscles[muscle.name].tendon_slack_length
            )
            osim_tendon_slack_length = (
                osim_model_scaled.muscle_groups[muscle_group.name].muscles[muscle.name].tendon_slack_length
            )
            npt.assert_almost_equal(biobuddy_tendon_slack_length, osim_tendon_slack_length, decimal=5)

    # Make sure the experimental markers are at the same position as the model's ones in static pose
    scale_tool = ScaleTool(original_model=original_model).from_xml(filepath=xml_filepath)
    scaled_model = scale_tool.scale(
        static_c3d=c3d_data,
        mass=69.2,
        q_regularization_weight=0.1,
        make_static_pose_the_models_zero=True,
    )
    scaled_model.to_biomod(scaled_biomod_filepath, with_mesh=False)

    # Seems like we loose some precision with the RT transformations
    jcs_in_global = scaled_model.forward_kinematics(q_zeros[:, 0])

    marker_parent = []
    marker_positions = np.zeros((4, scaled_model.nb_markers, c3d_data.nb_frames))
    for i_marker, marker_name in enumerate(scaled_model.marker_names):
        for segment in scaled_model.segments:
            if marker_name in segment.marker_names:
                marker_parent += [segment.name]
                break
        marker_index = c3d_data.marker_names.index(marker_name)
        marker_positions[:, i_marker, :] = c3d_data.all_marker_positions[:, marker_index, :]

    exp_markers = scale_tool.mean_experimental_markers
    biobuddy_markers = scaled_model.markers_in_global(q_zeros[:, 0])
    for i_marker in range(scaled_model.nb_markers):
        theoretical_position_from_exp = np.nanmean(marker_positions[:, i_marker, :], axis=1)
        marker_index = scaled_model.segments[marker_parent[i_marker]].marker_names.index(
            scaled_model.marker_names[i_marker]
        )
        theoretical_position_from_local = (
            jcs_in_global[marker_parent[i_marker]][0]
            @ scaled_model.segments[marker_parent[i_marker]].markers[marker_index].position
        )
        if marker_parent[i_marker] in ["hand_r", "hand_l", "fingers_r", "fingers_l"]:
            # TODO: fix -> There is still a problem with the fingers in BioBuddy
            decimal = 2
        else:
            decimal = 5
        npt.assert_almost_equal(exp_markers[:, i_marker], biobuddy_markers[:, i_marker, 0], decimal=decimal)
        npt.assert_almost_equal(
            exp_markers[:, i_marker],
            theoretical_position_from_exp.reshape(
                4,
            ),
            decimal=decimal,
        )
        npt.assert_almost_equal(
            exp_markers[:, i_marker],
            theoretical_position_from_local.reshape(
                4,
            ),
            decimal=decimal,
        )

    os.remove(scaled_biomod_filepath)
    os.remove(converted_scaled_osim_filepath)
    os.remove(parent_path + "/examples/models/static.trc")
    os.remove("wholebody.xml")
    os.remove("wholebody.osim")
    os.remove(parent_path + "/examples/models/scaled.osim")
    remove_temporary_biomods()


def test_translation_of_scaling_configuration():

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    biomod_config_filepath = parent_path + f"/examples/models/arm26_allbiceps_1dof.bioMod"
    biomod_config_filepath_new = parent_path + f"/examples/models/arm26_allbiceps_1dof_new.bioMod"
    osim_config_filepath = parent_path + f"/examples/models/arm26_allbiceps_1dof.xml"

    # --- Reading a .bioMod scaling configuration and translating it into a .xml configuration --- #
    # Read an .bioMod file
    original_model = BiomechanicalModelReal().from_biomod(filepath=biomod_config_filepath)

    scaling_configuration = ScaleTool(original_model).from_biomod(
        filepath=biomod_config_filepath,
    )

    # And convert it to a .xml file
    scaling_configuration.to_xml(osim_config_filepath)

    # Read the .xml file back
    new_xml_scaling_configuration = ScaleTool(original_model).from_xml(filepath=osim_config_filepath)

    # Rewrite it into a .bioMod to compare with the original one
    new_xml_scaling_configuration.to_biomod(biomod_config_filepath_new, append=False)

    # Reread the .biomod configuration we just printed
    new_biomod_scaling_configuration = ScaleTool(original_model).from_biomod(filepath=biomod_config_filepath_new)

    diff = DeepDiff(scaling_configuration, new_biomod_scaling_configuration, ignore_order=True)
    compare_models(scaling_configuration.scaled_model, new_biomod_scaling_configuration.scaled_model)

    # If the two objects are the same, there are no fields in diff
    assert diff == {}

    os.remove(osim_config_filepath)
    os.remove(biomod_config_filepath_new)


def test_creation_of_scaling_configuration():

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    biomod_config_filepath = parent_path + f"/examples/models/arm26_allbiceps_1dof.bioMod"

    # --- Creating a scaling configuration --- #
    # Read a biomod file to construct a BiomechanicalModelReal object
    original_model = BiomechanicalModelReal().from_biomod(filepath=biomod_config_filepath)

    # Create the scaling configuration
    scaling_configuration = ScaleTool(original_model)

    # Add a scaling segment for the pelvis
    scaling_configuration.add_scaling_segment(
        SegmentScaling(
            name="pelvis",
            scaling_type=SegmentWiseScaling(
                axis=Translations.XYZ,
                marker_pairs=[
                    ["RASIS", "LASIS"],
                    ["RPSIS", "LPSIS"],
                ],
            ),
        )
    )
    assert scaling_configuration.scaling_segments.keys() == ["pelvis"]
    assert isinstance(scaling_configuration.scaling_segments["pelvis"].scaling_type, SegmentWiseScaling)
    assert scaling_configuration.scaling_segments["pelvis"].scaling_type.axis == Translations.XYZ
    assert scaling_configuration.scaling_segments["pelvis"].scaling_type.marker_pairs == [
        ["RASIS", "LASIS"],
        ["RPSIS", "LPSIS"],
    ]

    # Add marker weights for the pelvis segment
    scaling_configuration.add_marker_weight(MarkerWeight(name="RASIS", weight=1.0))
    scaling_configuration.add_marker_weight(MarkerWeight(name="LASIS", weight=1.0))
    scaling_configuration.add_marker_weight(MarkerWeight(name="RPSIS", weight=0.5))
    scaling_configuration.add_marker_weight(MarkerWeight(name="LPSIS", weight=0.5))
    assert scaling_configuration.marker_weights.keys() == ["RASIS", "LASIS", "RPSIS", "LPSIS"]
    assert isinstance(scaling_configuration.marker_weights["RASIS"], MarkerWeight)
    assert scaling_configuration.marker_weights["RASIS"].weight == 1.0
    assert scaling_configuration.marker_weights["LPSIS"].weight == 0.5

    # Test that printing the marker weights works
    scaling_configuration.print_marker_weights()

    # Check that the scaling configuration can be destroyed
    scaling_configuration.remove_scaling_segment("pelvis")
    for marker_names in scaling_configuration.marker_weights.keys():
        scaling_configuration.remove_marker_weight(marker_names)

    # Check that it is actually empty
    assert scaling_configuration.scaling_segments == []
    assert scaling_configuration.marker_weights == []


def test_init():
    """Test initialization of ScaleTool"""
    scale_tool = ScaleTool(original_model=create_simple_model())

    # Test the default values
    assert scale_tool.personalize_mass_distribution is True
    assert scale_tool.max_marker_movement == 0.1
    assert isinstance(scale_tool.scaled_model, BiomechanicalModelReal)
    assert scale_tool.mean_experimental_markers is None
    assert scale_tool.header == ""
    assert scale_tool.original_mass is None
    assert isinstance(scale_tool.scaling_segments, NamedList)
    assert isinstance(scale_tool.marker_weights, NamedList)
    assert scale_tool.warnings == ""


def test_add_and_remove_marker_weight():
    """Test adding marker weights"""
    scale_tool = ScaleTool(original_model=create_simple_model())

    # There is no marker weights at the beginning
    assert scale_tool.marker_weights == []

    # Add a marker weight
    marker_weight1 = MarkerWeight(name="parent_marker", weight=1.5)
    marker_weight2 = MarkerWeight(name="child_marker", weight=2.0)
    scale_tool.add_marker_weight(marker_weight1)
    scale_tool.add_marker_weight(marker_weight2)

    assert len(scale_tool.marker_weights) == 2
    assert scale_tool.marker_weights["parent_marker"].weight == 1.5
    assert scale_tool.marker_weights["child_marker"].weight == 2.0

    # Test that printing the marker weights works
    buffer = io.StringIO()
    # Redirect stdout into the buffer
    with redirect_stdout(buffer):
        scale_tool.print_marker_weights()
    assert buffer.getvalue() == "parent_marker : 1.50\nchild_marker : 2.00\n"

    # Remove the marker weight
    scale_tool.remove_marker_weight("parent_marker")

    assert len(scale_tool.marker_weights) == 1
    assert "child_marker" in scale_tool.marker_weights.keys()
    assert "parent_marker" not in scale_tool.marker_weights.keys()


def test_add_and_remove_scaling_segment():
    """Test adding scaling segments"""
    scale_tool = ScaleTool(original_model=create_simple_model())

    # There are no scaling segments at the beginning
    assert scale_tool.scaling_segments == []

    # Add segment scaling
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="parent",
            scaling_type=SegmentWiseScaling(axis=Translations.XYZ, marker_pairs=[["parent_marker", "parent_marker2"]]),
        )
    )
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="child",
            scaling_type=SegmentWiseScaling(axis=Translations.XYZ, marker_pairs=[["child_marker", "child_marker2"]]),
        )
    )
    assert len(scale_tool.scaling_segments) == 2
    assert scale_tool.scaling_segments.keys() == ["parent", "child"]

    # Remove the scaling segment
    scale_tool.remove_scaling_segment("parent")
    assert len(scale_tool.scaling_segments) == 1
    assert "child" in scale_tool.scaling_segments.keys()
    assert "parent" not in scale_tool.scaling_segments.keys()


def test_check_that_makers_do_not_move():
    """Test checking that markers don't move too much"""
    mock_c3d_data = MockC3dData()
    scale_tool = ScaleTool(original_model=create_simple_model(), max_marker_movement=0.1)

    # Add marker weights
    marker_weight1 = MarkerWeight(name="parent_marker", weight=1.0)
    marker_weight2 = MarkerWeight(name="child_marker", weight=1.0)
    scale_tool.add_marker_weight(marker_weight1)
    scale_tool.add_marker_weight(marker_weight2)

    # This should not raise an error since markers don't move in our mock data
    scale_tool.check_that_makers_do_not_move(mock_c3d_data.all_marker_positions, mock_c3d_data.marker_names)

    # Now make a marker move too much
    moving_data = deepcopy(mock_c3d_data)
    moving_data.all_marker_positions[0, 0, -1] += 0.2  # Move x-coordinate of parent_marker in last frame

    # This should raise an error
    with pytest.raises(
        RuntimeError,
        match="The marker parent_marker moves of approximately 0.2 m during the static trial, which is above the maximal limit of 0.1 m.",
    ):
        scale_tool.check_that_makers_do_not_move(moving_data.all_marker_positions, moving_data.marker_names)


def test_define_mean_experimental_markers():
    """Test defining mean experimental markers"""
    mock_c3d_data = MockC3dData()
    scale_tool = ScaleTool(original_model=create_simple_model())

    scale_tool.define_mean_experimental_markers(mock_c3d_data.all_marker_positions, mock_c3d_data.marker_names)

    assert scale_tool.mean_experimental_markers is not None
    assert scale_tool.mean_experimental_markers.shape == (4, 4)  # 4D coordinates for 4 markers

    # Check mean values
    assert np.isclose(scale_tool.mean_experimental_markers[0, 0], 0.11)  # parent_marker x
    assert np.isclose(scale_tool.mean_experimental_markers[1, 0], 0.22)  # parent_marker y
    assert np.isclose(scale_tool.mean_experimental_markers[2, 0], 0.33)  # parent_marker z

    assert np.isclose(scale_tool.mean_experimental_markers[0, 2], 0.44)  # child_marker x
    assert np.isclose(scale_tool.mean_experimental_markers[1, 2], 0.55)  # child_marker y
    assert np.isclose(scale_tool.mean_experimental_markers[2, 2], 0.66)  # child_marker z


def test_scaling_factors_and_masses_segmentwise():
    """Test getting scaling factors and masses for segment-wise scaling"""
    mock_c3d_data = MockC3dData()
    simple_model = create_simple_model()
    scale_tool = ScaleTool(original_model=simple_model)

    # Add scaling segments
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="parent",
            scaling_type=SegmentWiseScaling(axis=Translations.XYZ, marker_pairs=[["parent_marker", "parent_marker2"]]),
        )
    )
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="child",
            scaling_type=SegmentWiseScaling(axis=Translations.XYZ, marker_pairs=[["child_marker", "child_marker2"]]),
        )
    )

    # Compute the scaling factors
    for segment in scale_tool.scaling_segments:
        segment.compute_scaling_factors(simple_model, mock_c3d_data.all_marker_positions, mock_c3d_data.marker_names)

    # Test the values form get scaling factors and masses
    scaling_factors, segment_masses = scale_tool.get_scaling_factors_and_masses(
        mock_c3d_data.all_marker_positions,
        mock_c3d_data.marker_names,
        mass=20,
        original_mass=15.0,  # 10 + 5
    )
    npt.assert_almost_equal(segment_masses["parent"], 11.759414918809005)
    npt.assert_almost_equal(segment_masses["child"], 8.240585081190993)
    npt.assert_almost_equal(
        scaling_factors["parent"]
        .to_vector()
        .reshape(
            4,
        ),
        np.array([1.84065206, 1.84065206, 1.84065206, 1.0]),
    )
    npt.assert_almost_equal(
        scaling_factors["child"]
        .to_vector()
        .reshape(
            4,
        ),
        np.array([2.57972867, 2.57972867, 2.57972867, 1.0]),
    )


def test_scaling_factors_and_masses_axiswise():
    """
    Test getting scaling factors and masses for axis-wise scaling.
    Since the X and Y axis are defined the same way as in the segment-wise scaling, the values should be the same.
    However, the Z axis is not defined, so the scaling factors should be different.
    """
    mock_c3d_data = MockC3dData()
    simple_model = create_simple_model()
    scale_tool = ScaleTool(original_model=simple_model)

    # Add scaling segments
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="parent",
            scaling_type=AxisWiseScaling(
                marker_pairs={
                    Translations.X: [["parent_marker", "parent_marker2"]],
                    Translations.Y: [["parent_marker", "parent_marker2"]],
                }
            ),
        )
    )
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="child",
            scaling_type=AxisWiseScaling(
                marker_pairs={
                    Translations.X: [["child_marker", "child_marker2"]],
                    Translations.Y: [["child_marker", "child_marker2"]],
                }
            ),
        )
    )

    # Compute the scaling factors
    for segment in scale_tool.scaling_segments:
        segment.compute_scaling_factors(simple_model, mock_c3d_data.all_marker_positions, mock_c3d_data.marker_names)

    # Test the values form get scaling factors and masses
    scaling_factors, segment_masses = scale_tool.get_scaling_factors_and_masses(
        mock_c3d_data.all_marker_positions,
        mock_c3d_data.marker_names,
        mass=20,
        original_mass=15.0,  # 10 + 5
    )
    npt.assert_almost_equal(segment_masses["parent"], 12.298699379214955)
    npt.assert_almost_equal(segment_masses["child"], 7.701300620785044)
    npt.assert_almost_equal(
        scaling_factors["parent"]
        .to_vector()
        .reshape(
            4,
        ),
        np.array([1.84065206, 1.84065206, 1.0, 1.0]),
    )
    npt.assert_almost_equal(
        scaling_factors["child"]
        .to_vector()
        .reshape(
            4,
        ),
        np.array([2.57972867, 2.57972867, 1.0, 1.0]),
    )


def test_scaling_factors_and_masses_bodywise():
    """Test getting scaling factors and masses for body-wise scaling"""
    mock_c3d_data = MockC3dData()
    simple_model = create_simple_model()
    scale_tool = ScaleTool(original_model=simple_model)

    # Add scaling segments
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="parent",
            scaling_type=BodyWiseScaling(subject_height=1.01),
        )
    )
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="child",
            scaling_type=BodyWiseScaling(subject_height=1.01),
        )
    )

    # Test that it raises if the original_model has not height
    with pytest.raises(
        RuntimeError,
        match="The original model height must be set to use BodyWiseScaling. you can set it using `original_model.height = height`.",
    ):
        scale_tool.scaling_segments["parent"].compute_scaling_factors(
            simple_model, mock_c3d_data.all_marker_positions, mock_c3d_data.marker_names
        )

    # Set the height of the original model
    simple_model.height = 1.0

    # Compute the scaling factors
    for segment in scale_tool.scaling_segments:
        segment.compute_scaling_factors(simple_model, mock_c3d_data.all_marker_positions, mock_c3d_data.marker_names)

    # Test the values form get scaling factors and masses
    scaling_factors, segment_masses = scale_tool.get_scaling_factors_and_masses(
        mock_c3d_data.all_marker_positions,
        mock_c3d_data.marker_names,
        mass=20,
        original_mass=15.0,  # 10 + 5
    )
    npt.assert_almost_equal(segment_masses["parent"], 13.333333333333336)
    npt.assert_almost_equal(segment_masses["child"], 6.666666666666668)
    npt.assert_almost_equal(
        scaling_factors["parent"]
        .to_vector()
        .reshape(
            4,
        ),
        np.array([1.01, 1.01, 1.01, 1.0]),
    )
    npt.assert_almost_equal(
        scaling_factors["child"]
        .to_vector()
        .reshape(
            4,
        ),
        np.array([1.01, 1.01, 1.01, 1.0]),
    )


def test_scale_rt():
    """Test scaling of rotation-translation matrix"""
    rt_matrix = RotoTransMatrix()
    rt_matrix.from_rt_matrix(
        np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0]])
    )

    scale_factor = np.array([2.0, 3.0, 4.0, 1.0])

    result = ScaleTool.scale_rt(rt_matrix, scale_factor)

    assert result.rt_matrix[0, 3] == 2.0  # 1.0 * 2.0
    assert result.rt_matrix[1, 3] == 6.0  # 2.0 * 3.0
    assert result.rt_matrix[2, 3] == 12.0  # 3.0 * 4.0

    # Rotation part should remain unchanged
    assert np.array_equal(result.rt_matrix[:3, :3], rt_matrix.rt_matrix[:3, :3])


def test_scale_imu():
    """Test scaling of imu"""
    simple_model = create_simple_model()
    imu_matrix = RotoTransMatrix()
    imu_matrix.from_rt_matrix(
        np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0]])
    )
    original_imu = InertialMeasurementUnitReal(
        name="original_contact",
        parent_name="parent",
        scs=imu_matrix,
    )
    scale_factor = point_to_array(np.array([2.0, 3.0, 4.0, 1.0]))

    result = ScaleTool(simple_model).scale_imu(original_imu, scale_factor)

    assert result.name == original_imu.name
    assert result.parent_name == original_imu.parent_name
    assert result.is_technical == original_imu.is_technical
    assert result.is_anatomical == original_imu.is_anatomical

    assert result.scs.rt_matrix[0, 3] == 2.0  # 1.0 * 2.0
    assert result.scs.rt_matrix[1, 3] == 6.0  # 2.0 * 3.0
    assert result.scs.rt_matrix[2, 3] == 12.0  # 3.0 * 4.0

    # Rotation part should remain unchanged
    assert np.array_equal(result.scs.rt_matrix[:3, :3], original_imu.scs.rt_matrix[:3, :3])


def test_scale_marker():
    """Test scaling of markers"""
    simple_model = create_simple_model()
    original_marker = simple_model.segments["parent"].markers[0]
    scale_factor = np.array([2.0, 3.0, 4.0, 1.0])

    result = ScaleTool(simple_model).scale_marker(original_marker, scale_factor)

    assert result.name == original_marker.name
    assert result.parent_name == original_marker.parent_name
    assert result.is_technical == original_marker.is_technical
    assert result.is_anatomical == original_marker.is_anatomical

    # Check scaled position
    expected_0 = original_marker.position[0] * scale_factor
    expected_1 = original_marker.position[1] * scale_factor
    expected_2 = original_marker.position[2] * scale_factor
    npt.assert_almost_equal(expected_0, np.array([0.2, 0.3, 0.4, 0.1]))
    npt.assert_almost_equal(result.position[0], expected_0)
    npt.assert_almost_equal(expected_1, np.array([0.4, 0.6, 0.8, 0.2]))
    npt.assert_almost_equal(result.position[1], expected_1)
    npt.assert_almost_equal(expected_2, np.array([0.6, 0.9, 1.2, 0.3]))
    npt.assert_almost_equal(result.position[2], expected_2)


def test_scale_contact():
    """Test scaling of contacts"""
    simple_model = create_simple_model()
    original_contact = ContactReal(
        name="original_contact",
        parent_name="parent",
        position=np.array([0.1, 0.2, 0.3, 1.0]),
        axis=Translations.XYZ,
    )
    scale_factor = point_to_array(np.array([2.0, 3.0, 4.0, 1.0]))

    result = ScaleTool(simple_model).scale_contact(original_contact, scale_factor)

    assert result.name == original_contact.name
    assert result.parent_name == original_contact.parent_name
    assert result.axis == original_contact.axis

    # Check scaled position
    expected = original_contact.position * scale_factor
    npt.assert_almost_equal(
        expected.reshape(
            4,
        ),
        np.array([0.2, 0.6, 1.2, 1.0]),
    )
    npt.assert_almost_equal(result.position, expected)


def test_scale_via_point():
    """Test scaling of via points"""
    simple_model = create_simple_model()
    original_via_point = simple_model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points[0]
    scale_factor = point_to_array(np.array([2.0, 3.0, 4.0, 1.0]))

    result = ScaleTool(simple_model).scale_via_point(original_via_point, scale_factor)

    assert result.name == original_via_point.name
    assert result.parent_name == original_via_point.parent_name
    assert result.muscle_name == original_via_point.muscle_name
    assert result.muscle_group == original_via_point.muscle_group

    # Check scaled position
    expected = original_via_point.position * scale_factor
    npt.assert_almost_equal(
        expected.reshape(
            4,
        ),
        np.array([0.4, 0.9, 1.6, 1.0]),
    )
    npt.assert_almost_equal(result.position, expected)


def test_scale_muscle():
    """Test scaling of muscles"""
    simple_model = create_simple_model()
    original_muscle = simple_model.muscle_groups["parent_to_child"].muscles[0]
    origin_scale_factor = point_to_array(np.array([2.0, 3.0, 4.0, 1.0]))
    insertion_scale_factor = point_to_array(np.array([2.0, 3.0, 4.0, 1.0]))

    result = ScaleTool(simple_model).scale_muscle(original_muscle, origin_scale_factor, insertion_scale_factor)

    assert result.name == original_muscle.name
    assert result.muscle_type == original_muscle.muscle_type
    assert result.state_type == original_muscle.state_type
    assert result.muscle_group == original_muscle.muscle_group
    assert result.optimal_length is None  # Will be set later
    assert result.maximal_force == original_muscle.maximal_force
    assert result.tendon_slack_length is None
    assert result.pennation_angle == original_muscle.pennation_angle
    assert result.maximal_excitation == original_muscle.maximal_excitation

    # Check scaled position
    expected_origin = original_muscle.origin_position.position * origin_scale_factor
    npt.assert_almost_equal(
        expected_origin.reshape(
            4,
        ),
        np.array([0.0, 0.3, 0.0, 1.0]),
    )
    npt.assert_almost_equal(result.origin_position.position, expected_origin)

    expected_insertion = original_muscle.insertion_position.position * insertion_scale_factor
    npt.assert_almost_equal(
        expected_insertion.reshape(
            4,
        ),
        np.array([1.0, 1.2, 1.2, 1.0]),
    )
    npt.assert_almost_equal(result.insertion_position.position, expected_insertion)
