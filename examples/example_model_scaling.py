"""
This example shows how to scale a model based on a generic model and a static trial.
"""

import logging
from pathlib import Path
import biorbd

from pyomeca import Markers
from biobuddy import BiomechanicalModelReal, MuscleType, MuscleStateType, MeshParser, MeshFormat, ScaleTool


def main():

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
    osim_file_path = f"{current_path_file}/models/wholebody.osim"
    biomod_file_path = f"{current_path_file}/models/wholebody.bioMod"
    geometry_path = f"{current_path_file}/../external/opensim-models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/models/Geometry_cleaned"
    xml_filepath = f"{current_path_file}/models/wholebody.xml"
    scaled_biomod_file_path = f"{current_path_file}/models/wholebody_scaled.bioMod"
    static_file_path = f"{current_path_file}/data/static.c3d"

    # Convert the vtp files
    mesh = MeshParser(geometry_folder=geometry_path)
    mesh.process_meshes(fail_on_error=False)
    mesh.write(geometry_cleaned_path, format=MeshFormat.VTP)

    # Read an .osim file
    model = BiomechanicalModelReal.from_osim(
        filepath=osim_file_path,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir="Geometry_cleaned",
    )

    # Translate into biomod
    model.to_biomod(biomod_file_path)

    # Setup the scaling configuration (which markers to use)
    scale_tool = ScaleTool.from_xml(filepath=xml_filepath)

    # Scale the model
    scaled_model = scale_tool.scale(
        original_model=model, static_trial=static_file_path, frame_range=range(100, 200), mass=80
    )

    # Write the scaled model to a .bioMod file
    scaled_model.to_biomod(scaled_biomod_file_path)

    # Test that the model created is valid
    biorbd.Model(scaled_biomod_file_path)

    # Compare the result visually
    try:
        import pyorerun
    except ImportError:
        raise ImportError("You must install pyorerun to visualize the model")
    import numpy as np

    # Visualization
    t = np.linspace(0, 1, 10)
    viz = pyorerun.PhaseRerun(t)
    q = np.zeros((42, 10))

    # Biorbd model translated from .osim
    viz_biomod_model = pyorerun.BiorbdModel(biomod_file_path)
    viz_biomod_model.options.transparent_mesh = False
    viz_biomod_model.options.show_gravity = True
    viz.add_animated_model(viz_biomod_model, q)

    # Add the experimental markers from the static trial
    fake_exp_markers = np.repeat(scale_tool.mean_experimental_markers[:, :, np.newaxis], 10, axis=2)
    pyomarkers = Markers(data=fake_exp_markers, channels=scaled_model.marker_names)

    # Model output
    viz_scaled_model = pyorerun.BiorbdModel(scaled_biomod_file_path)
    viz_scaled_model.options.transparent_mesh = False
    viz_scaled_model.options.show_gravity = True
    viz.add_animated_model(viz_scaled_model, q, tracked_markers=pyomarkers)

    # TODO: Add the osim models
    #  but DO NOT SCALE IN OPENSIM Python-API as it is broken (aka, the main reason why we are implementing this)

    # Animate
    viz.rerun_by_frame("Model output")


if __name__ == "__main__":
    main()
