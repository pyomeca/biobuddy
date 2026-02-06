"""
Example script to create a graph visualization from a BioMod file.

This script demonstrates how to:
1. Load a biomechanical model from a .bioMod file
2. Generate a graph representation of the model structure
3. Convert the graph to PNG format for visualization

The output includes visualization of segments, degrees of freedom, via points, and markers.
"""

from biobuddy import BiomechanicalModelReal
from pathlib import Path


def create_graph_from_biomod_file():
    # Get the model
    current_path_file = Path(__file__).parent
    base_name = "arm26_allbiceps_1dof"
    biomod_path = f"{current_path_file}/models/{base_name}.bioMod"
    output_path = f"{current_path_file}/data/{base_name}"

    # Load the model from the .bioMod file
    model = BiomechanicalModelReal().from_biomod(biomod_path)

    # Write the graph visualization to a dot and PNG file
    model.write_graphviz(
        output_path,
        include_ghost_segments=True,
        include_dof_segments=True,
        include_via_points=True,
        include_markers=True,
    )

if __name__ == "__main__":
    create_graph_from_biomod_file()