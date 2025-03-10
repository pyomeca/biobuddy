"""
This example shows how to read and write models.
"""

import os
from pathlib import Path

from biobuddy import BiomechanicalModel, MuscleType, MuscleStateType


if __name__ == "__main__":

    # Paths
    current_path_file = Path(__file__).parent
    biomod_file_path = f"{current_path_file}/models/wholebody.bioMod"
    osim_file_path = f"{current_path_file}/models/wholebody.osim"

    # Read an .osim file
    model = BiomechanicalModel().from_osim(osim_file_path,
                                           muscle_type=MuscleType.HILL_DE_GROOTE,
                                           muscle_state_type=MuscleStateType.DEGROOTE)

    # And convert it to a .bioMod file
    model.to_biomod(biomod_file_path)

    # Compare the result visually
    import pyorerun
    import numpy as np

    # Model output
    model = pyorerun.BiorbdModel(biomod_file_path)
    model.options.transparent_mesh = False
    model.options.show_gravity = True

    # Model reference
    reference_model = pyorerun.BiorbdModel(biomod_file_path.replace(".biomod", "reference.biomod"))
    reference_model.options.transparent_mesh = False
    reference_model.options.show_gravity = True

    # Visualization
    t = np.linspace(0, 1, 10)
    q = np.zeros((model.nb_q, 10))
    q_ref = np.zeros((reference_model.nb_q, 10))
    q_ref[0, :] = 0.5

    viz = pyorerun.PhaseRerun(t)
    viz.add_animated_model(model, q)
    viz.add_animated_model(reference_model, np.zeros((model.nb_q, 10)))
    viz.rerun_by_frame("Model output")
