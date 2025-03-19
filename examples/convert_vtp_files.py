"""
This example shows how to convert vtp files to triangles.
"""

from pathlib import Path

from biobuddy import MeshParser, MeshFormat


if __name__ == "__main__":

    # Paths
    current_path_file = Path(__file__).parent
    geometry_path = f"{current_path_file}/../external/opensim-models/Geometry"
    geometry_cleaned_path = f"{current_path_file}/models/Geometry_cleaned"

    # Convert the files
    mesh = MeshParser(geometry_folder=geometry_path)
    mesh.process_meshes()
    mesh.write(geometry_cleaned_path, format=MeshFormat.VTP)
