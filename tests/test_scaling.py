"""
TODO: test scaling configuration
"""

import os
import pytest
import opensim as osim
import shutil

import ezc3d
import biorbd
import numpy as np
import numpy.testing as npt

from test_utils import remove_temporary_biomods
from biobuddy import BiomechanicalModelReal, MuscleType, MuscleStateType, ScaleTool, C3dData, ReferenceFrame


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
    osim_model_scaled.to_biomod(converted_scaled_osim_filepath, with_mesh=False)
    scaled_osim_model = biorbd.Model(converted_scaled_osim_filepath)

    # --- Scale in BioBuddy --- #
    original_model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=cleaned_relative_path,
    )

    scale_tool = ScaleTool(original_model=original_model).from_xml(filepath=xml_filepath)
    scaled_model = scale_tool.scale(
        static_c3d=c3d_data,
        mass=69.2,
        q_regularization_weight=1,
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
            ) + scaled_model.segment_coordinate_system_in_global(segment_name)[:3, 3, 0]
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
    for via_point_name in original_model.via_points.keys():
        biobuddy_scaled_via_point = scaled_model.via_points[via_point_name].position[:3]
        osim_scaled_via_point = osim_model_scaled.via_points[via_point_name].position[:3]
        npt.assert_almost_equal(biobuddy_scaled_via_point, osim_scaled_via_point, decimal=5)

    # Muscle properties
    for muscle in original_model.muscles.keys():
        if (
            muscle
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
            or "stern_mast" in muscle
        ):
            # Skipping muscles with ConditionalPathPoints and MovingPathPoints
            # Skipping the head since there is a difference in scaling
            continue
        print(muscle)
        biobuddy_optimal_length = scaled_model.muscles[muscle].optimal_length
        osim_optimal_length = osim_model_scaled.muscles[muscle].optimal_length
        npt.assert_almost_equal(biobuddy_optimal_length, osim_optimal_length, decimal=5)
        biobuddy_tendon_slack_length = scaled_model.muscles[muscle].tendon_slack_length
        osim_tendon_slack_length = osim_model_scaled.muscles[muscle].tendon_slack_length
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
    scaled_biorbd_model = biorbd.Model(scaled_biomod_filepath)

    exp_markers = scale_tool.mean_experimental_markers[:, :]
    for i_marker in range(exp_markers.shape[1]):
        biobuddy_scaled_marker = scaled_biorbd_model.markers(q_zeros[:, 0])[i_marker].to_array()
        npt.assert_almost_equal(exp_markers[:, i_marker], biobuddy_scaled_marker, decimal=5)

    os.remove(scaled_biomod_filepath)
    os.remove(converted_scaled_osim_filepath)
    os.remove(parent_path + "/examples/models/static.trc")
    os.remove("wholebody.xml")
    os.remove("wholebody.osim")
    os.remove(parent_path + "/examples/models/scaled.osim")
    remove_temporary_biomods()
