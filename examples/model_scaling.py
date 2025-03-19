"""
This example shows how to scale a model based on a generic model and a static trial.
"""

from pathlib import Path
import biorbd

from biobuddy import (
    BiomechanicalModel,
    MuscleType,
    MuscleStateType,
    MeshParser,
    MeshFormat,
    OsimModelParser
)


if __name__ == "__main__":

    # Paths
    current_path_file = Path(__file__).parent
    geometry_path = f"{current_path_file}/../external/opensim-models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/models/Geometry_cleaned"
    osim_file_path = f"{current_path_file}/models/wholebody.osim"
    scaled_biomod_file_path = f"{current_path_file}/models/wholebody_scaled.bioMod"

    # Convert the vtp files
    mesh = MeshParser(geometry_folder=geometry_path)
    mesh.process_meshes()
    mesh.write(geometry_cleaned_path, format=MeshFormat.VTP)

    # Read an .osim file
    model = OsimModelParser(osim_file_path,
                           muscle_type=MuscleType.HILL_DE_GROOTE,
                           muscle_state_type=MuscleStateType.DEGROOTE,
                           mesh_dir="Geometry_cleaned").biomechanical_model_real

    # Setup the scaling configuration (which markers to use)
    # TODO

    # Scale the model
    # TODO

    # Write the scaled model to a .bioMod file
    model.to_biomod(scaled_biomod_file_path)

    # Test that the model created is valid
    biorbd.Model(scaled_biomod_file_path)

    # Compare the result visually
    import pyorerun
    import numpy as np

    # Visualization
    t = np.linspace(0, 1, 10)
    viz = pyorerun.PhaseRerun(t)

    # Model output
    model = pyorerun.BiorbdModel(scaled_biomod_file_path)
    model.options.transparent_mesh = False
    model.options.show_gravity = True
    q = np.zeros((model.nb_q, 10))
    viz.add_animated_model(model, q)

    # TODO: Add the osim model
    #  but DO NOT SCALE IN OPENSIM as it is broken (aka, the main reason why we are implementing this)

    # Animate
    viz.rerun_by_frame("Model output")

