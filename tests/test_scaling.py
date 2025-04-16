"""
TODO: remove biorbd dependency !!!
TODO: Test the muscles and inertial values.
TODO: Add the biomod sclaing configuration + test it
"""

import os
import pytest

import ezc3d
import biorbd
import numpy as np

from biobuddy import MeshParser, MeshFormat, BiomechanicalModelReal, MuscleType, MuscleStateType, ScaleTool


def convert_c3d_to_trc(c3d_file_path):
    """
    This function reads the c3d static file and converts it into a trc file that will be used to scale the model in OpenSim.
    The trc file is saved at the same place as the original c3d file.
    """
    trc_file_path = c3d_file_path.replace(".c3d", ".trc")

    c3d = ezc3d.c3d(c3d_file_path)
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]

    frame_rate = c3d["header"]["points"]["frame_rate"]
    marker_data = c3d["data"]["points"][:3, :, :] / 1000  # Convert in meters

    with open(trc_file_path, "w") as f:
        trc_file_name = os.path.basename(trc_file_path)
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_file_name}\n")
        f.write(
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        )
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



def test_scaling_wholebody():

    # For ortho_norm_basis
    np.random.seed(42)

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geometry_path = parent_path + "/external/opensim-models/Geometry"
    cleaned_geometry_path = parent_path + "/models/geometry_cleaned"
    cleaned_relative_path = "geometry_cleaned"

    osim_file_path = parent_path + "/examples/models/wholebody.osim"
    xml_filepath = parent_path + "/examples/models/wholebody.xml"
    scaled_biomod_file_path = parent_path + "/examples/models/wholebody_scaled.bioMod"
    converted_scaled_osim_file_path = parent_path + "/examples/models/wholebody_converted_scaled.bioMod"
    scaled_osim_file_path = parent_path + "/examples/models/wholebody_scaled.osim"
    static_file_path = parent_path + "/examples/data/static.c3d"

    # Convert the vtp mesh files
    mesh_parser = MeshParser(geometry_path)
    mesh_parser.process_meshes(fail_on_error=False)
    mesh_parser.write(cleaned_geometry_path, MeshFormat.VTP)

    # Read the .osim file
    model = BiomechanicalModelReal.from_osim(
        filepath=osim_file_path,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=cleaned_relative_path,
    )

    # Scale the model in BioBuddy
    scale_tool = ScaleTool(original_model=model).from_xml(filepath=xml_filepath)
    scaled_model = scale_tool.scale(file_path=static_file_path, frame_range=range(100, 200), mass=80
    )
    scaled_model.to_biomod(scaled_biomod_file_path)
    scaled_biorbd_model = biorbd.Model(scaled_biomod_file_path)

    # Scale in Opensim's GUI
    # convert_c3d_to_trc(static_file_path)

    # Import the model scaled in OpeSim's GUI
    osim_model = BiomechanicalModelReal.from_osim(
        filepath=scaled_osim_file_path,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=cleaned_relative_path,
    )
    osim_model.to_biomod(converted_scaled_osim_file_path)
    scaled_osim_model = biorbd.Model(converted_scaled_osim_file_path)

    # Make sure the model markers coincide with the experimental markers
    q_zeros = np.zeros((scaled_biorbd_model.nbQ(), 1))
    exp_markers = np.repeat(scale_tool.mean_experimental_markers[:, :, np.newaxis], 10, axis=2)
    biobuddy_scaled_markers = scaled_biorbd_model.markers(q_zeros).to_array()
    osim_scaled_markers = scaled_osim_model.markers(q_zeros).to_array()
    assert np.all(np.abs(exp_markers - biobuddy_scaled_markers) < 1e-5)
    assert np.all(np.abs(exp_markers - osim_scaled_markers) < 1e-5)
    assert np.all(np.abs(osim_scaled_markers - biobuddy_scaled_markers) < 1e-5)

    # Make sure the muscle properties are the same

    # Make sure the inertial properties are the same


