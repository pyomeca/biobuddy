"""
This example shows how to read and write models.
"""

import logging
from pathlib import Path
import biorbd

from biobuddy import (
    MuscleType,
    MuscleStateType,
    MeshParser,
    MeshFormat,
    OsimModelParser,
    BiomechanicalModelReal,
)


if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(

        level=logging.DEBUG
        ,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ]
    )

    # Paths
    current_path_file = Path(__file__).parent
    biomod_file_path = f"{current_path_file}/models/wholebody.bioMod"
    osim_file_path = f"{current_path_file}/models/wholebody.osim"
    geometry_path = f"{current_path_file}/../external/opensim-models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/models/Geometry_cleaned"

    # Convert the vtp files
    mesh = MeshParser(geometry_folder=geometry_path)
    mesh.process_meshes()
    mesh.write(geometry_cleaned_path, format=MeshFormat.VTP)

    # Read an .osim file
    model = BiomechanicalModelReal.from_osim(
        filepath = osim_file_path,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir="Geometry_cleaned"
    )

    # And convert it to a .bioMod file
    model.to_biomod(biomod_file_path, with_mesh=False)

    # Test that the model created is valid
    biorbd.Model(biomod_file_path)

    # Compare the result visually
    import pyorerun
    import numpy as np

    # Visualization
    t = np.linspace(0, 1, 10)
    viz = pyorerun.PhaseRerun(t)

    # Model output
    model = pyorerun.BiorbdModel(biomod_file_path)
    model.options.transparent_mesh = False
    model.options.show_gravity = True
    q = np.zeros((model.nb_q, 10))
    viz.add_animated_model(model, q)

    # Model reference
    reference_model = pyorerun.BiorbdModel(biomod_file_path.replace(".bioMod", "_reference.bioMod"))
    reference_model.options.transparent_mesh = False
    reference_model.options.show_gravity = True
    q_ref = np.zeros((reference_model.nb_q, 10))
    q_ref[0, :] = 0.5
    viz.add_animated_model(reference_model, q_ref)

    # Animate
    viz.rerun_by_frame("Model output")

