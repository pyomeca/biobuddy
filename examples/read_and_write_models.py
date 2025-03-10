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
