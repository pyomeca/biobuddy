"""
TODO: Add the biomod sclaing configuration + test it
"""

import os
import pytest
import opensim as osim
import shutil

import ezc3d
import biorbd
import numpy as np
import numpy.testing as npt

from biobuddy import BiomechanicalModelReal, MuscleType, MuscleStateType, ScaleTool, C3dData


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


def visualize_model_scaling_output(scaled_biomod_filepath: str, converted_scaled_osim_filepath: str, q):
    """
    Only for debugging purposes.
    """
    import pyorerun

    # Compare the result visually
    t = np.linspace(0, 1, 10)
    viz = pyorerun.PhaseRerun(t)

    # Model scaled in BioBuddy
    viz_biomod_model = pyorerun.BiorbdModel(scaled_biomod_filepath)
    viz_biomod_model.options.transparent_mesh = False
    viz_biomod_model.options.show_gravity = True
    viz.add_animated_model(viz_biomod_model, q)

    # Model scaled in OpenSim
    viz_scaled_model = pyorerun.BiorbdModel(converted_scaled_osim_filepath)
    viz_scaled_model.options.transparent_mesh = False
    viz_scaled_model.options.show_gravity = True
    viz.add_animated_model(viz_scaled_model, q)

    # Animate
    viz.rerun_by_frame("Scaling comparison")


def test_scaling_wholebody():

    np.random.seed(42)

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_relative_path = "Geometry_cleaned"
    osim_filepath = parent_path + "/examples/models/wholebody.osim"
    xml_filepath = parent_path + "/examples/models/wholebody.xml"
    scaled_biomod_filepath = parent_path + "/examples/models/wholebody_scaled.bioMod"
    converted_scaled_osim_filepath = parent_path + "/examples/models/wholebody_converted_scaled.bioMod"
    scaled_osim_filepath = parent_path + "/examples/models/wholebody_scaled.osim"
    static_filepath = parent_path + "/examples/data/static.c3d"
    trc_file_path = parent_path + "/examples/data/static.trc"

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
    osim_model_scaled = BiomechanicalModelReal.from_osim(
        filepath=parent_path + "/examples/models/scaled.osim",
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=cleaned_relative_path,
    )
    osim_model_scaled.to_biomod(converted_scaled_osim_filepath)
    scaled_osim_model = biorbd.Model(converted_scaled_osim_filepath)


    # --- Scale in BioBuddy --- #
    original_model = BiomechanicalModelReal.from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=cleaned_relative_path,
    )

    scale_tool = ScaleTool(original_model=original_model).from_xml(filepath=xml_filepath)
    scaled_model = scale_tool.scale(
        filepath=static_filepath,
        first_frame=0,
        last_frame=531,
        mass=69.2,
        q_regularization_weight=0.1,
        make_static_pose_the_models_zero=False,
    )
    scaled_model.to_biomod(scaled_biomod_filepath)
    scaled_biorbd_model = biorbd.Model(scaled_biomod_filepath)

    q_zeros = np.zeros((42, 10))
    q_random = np.random.rand(42) * 2 * np.pi

    # --- Test the scaling factors --- #
    c3d_data = C3dData(c3d_path=static_filepath, first_frame=100, last_frame=200)
    marker_names = c3d_data.marker_names
    marker_positions = c3d_data.all_marker_positions[:3, :, :]

    # Scaling factors from scaling_factors.osim  (TODO: add the scaling factors in the osim parser)
    scaling_factors = {"pelvis": 0.883668,
                       "femur_r": 1.1075,
                       "tibia_r": 1.00352,
                       "talus_r": 0.961683,
                       "calcn_r": 1.05904,
                       "toes_r": 0.999246,
                       "torso": 1.04094,
                       "head_and_neck": 1.02539,
                       "humerus_r": 1.00517,
                       "ulna_r": 1.12622,
                       "radius_r": 1.04826,
                       "lunate_r": 1.12829,
                       "hand_r": 1.18954,
                       "fingers_r": 1.26327,
                       }
    for segment_name, scale_factor in scaling_factors.items():
        biobuddy_scaling_factors = scale_tool.scaling_segments[segment_name].compute_scaling_factors(
            original_model,
            marker_positions,
            marker_names)
        npt.assert_almost_equal(biobuddy_scaling_factors.mass, scale_factor, decimal=2)


    # --- Test masses --- #
    # Total mass
    npt.assert_almost_equal(scaled_osim_model.mass(), 69.2, decimal=5)
    npt.assert_almost_equal(scaled_biorbd_model.mass(), 69.2, decimal=5)

    # Segment mass
    for i_segment, segment_name in enumerate(scaled_model.segments.keys()):
        if scaled_model.segments[segment_name].inertia_parameters is None:
            mass_biobuddy = 0
        else:
            mass_biobuddy = scaled_model.segments[segment_name].inertia_parameters.mass
        mass_to_biorbd = scaled_biorbd_model.segment(i_segment).characteristics().mass()
        mass_osim = scaled_osim_model.segment(i_segment).characteristics().mass()
        npt.assert_almost_equal(mass_to_biorbd, mass_biobuddy)
        npt.assert_almost_equal(mass_osim, mass_biobuddy)
        npt.assert_almost_equal(mass_to_biorbd, mass_osim)
        if mass_biobuddy > 1e-3:
            original_mass = original_model.segments[segment_name].inertia_parameters.mass
            npt.assert_array_less(original_mass, mass_biobuddy)

    # CoM
    for i_segment, segment_name in enumerate(scaled_model.segments.keys()):
        print(segment_name)
        # Zero
        if scaled_model.segments[segment_name].inertia_parameters is None:
            com_biobuddy = np.zeros((3,))
        else:
            com_biobuddy = (
                scaled_model.segments[segment_name]
                .inertia_parameters.center_of_mass[:3]
                .reshape(
                    3,
                )
            )
        com_to_biorbd = scaled_biorbd_model.CoMbySegment(q_zeros[:, 0], i_segment).to_array()
        com_osim = scaled_osim_model.CoMbySegment(q_zeros[:, 0], i_segment).to_array()
        npt.assert_almost_equal(com_to_biorbd, com_biobuddy)
        npt.assert_almost_equal(com_osim, com_biobuddy)
        npt.assert_almost_equal(com_to_biorbd, com_osim)
        # Random
        com_biobuddy = scaled_biorbd_model.CoMbySegment(q_random, i_segment).to_array()
        com_osim = scaled_osim_model.CoMbySegment(q_random, i_segment).to_array()
        npt.assert_almost_equal(com_osim, com_biobuddy)

    # Make sure the model markers coincide with the experimental markers
    exp_markers = scale_tool.mean_experimental_markers[:, :]
    for i_marker in range(exp_markers.shape[1]):
        biobuddy_scaled_marker = scaled_biorbd_model.markers(q_zeros)[i_marker].to_array()
        osim_scaled_marker = scaled_osim_model.markers(q_zeros)[i_marker].to_array()
        assert np.all(np.abs(exp_markers[:, i_marker, 0] - biobuddy_scaled_marker) < 1e-5)
        assert np.all(np.abs(exp_markers[:, i_marker, 0] - osim_scaled_marker) < 1e-5)
        assert np.all(np.abs(osim_scaled_marker - biobuddy_scaled_marker) < 1e-5)

    # Make sure the muscle properties are the same
