"""
This example shows how to read and write models.
"""

import os
from pathlib import Path

import numpy as np
import biorbd
from biobuddy import (
    BiomechanicalModel,
    MeshFile,
    Segment,
    Contact,
    MuscleGroup,
    Muscle,
    MuscleType,
    MuscleStateType,
    Translations,
    Rotations,
    RangeOfMotion,
    Ranges,
    ViaPoint,
)

if __name__ == "__main__":

    # Paths
    current_path_file = Path(__file__).parent
    biomod_file_path = f"{current_path_file}/models/wholebody.bioMod"
    osim_file_path = f"{current_path_file}/models/wholebody.osim"

    # Read an .osim file
    model = BiomechanicalModel().from_osim(osim_file_path)

    # And convert it to a .bioMod file
    model.to_biomod(biomod_file_path)
