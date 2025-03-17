"""
This example shows how to read and write models.
"""

from pathlib import Path
import biorbd

from biobuddy import BiomechanicalModel, MuscleType, MuscleStateType, VtpParser


if __name__ == "__main__":

    # Paths
    current_path_file = Path(__file__).parent
    biomod_file_path = f"{current_path_file}/models/wholebody.bioMod"
    osim_file_path = f"{current_path_file}/models/wholebody.osim"
    geometry_path = f"{current_path_file}/models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/models/Geometry_cleaned"

    # Convert vtp files
    VtpParser(geometry_path, geometry_cleaned_path)

    # Read an .osim file
    model = BiomechanicalModel().from_osim(osim_file_path,
                                           muscle_type=MuscleType.HILL_DE_GROOTE,
                                           muscle_state_type=MuscleStateType.DEGROOTE,
                                           mesh_dir="Geometry_cleaned")

    # And convert it to a .bioMod file
    model.to_biomod(biomod_file_path)

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

