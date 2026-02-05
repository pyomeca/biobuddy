"""
Example script to create a graph visualization from a BioMod file.

This script demonstrates how to:
1. Load a biomechanical model from a .bioMod file
2. Generate a graph representation of the model structure
3. Convert the graph to PNG format for visualization

The output includes visualization of segments, degrees of freedom, via points, and markers.
"""

from biobuddy import BiomechanicalModelReal
import os


# Get the model
biomod_path = "examples/models/arm26_allbiceps_1dof.bioMod"
base_name = "arm26_allbiceps_1dof"
model = BiomechanicalModelReal().from_biomod(biomod_path)
path = os.path.join("examples/data", base_name)

model.write_graphviz(
    path, ghost_segments=True, dof_segments=True, via_points=True, markers=True
)

model.convert_dot_to_png(path)
