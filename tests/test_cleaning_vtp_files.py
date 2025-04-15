import os
from pathlib import Path
import shutil

from biobuddy import MeshParser, MeshFormat
import pytest


def test_process_vtp_files():

    # Paths
    current_path_file = Path(__file__).parent
    geometry_path = f"{current_path_file}/../external/opensim-models/Geometry"
    target_path = f"./geometry_processed"

    # Make sure there is not left over files from previous tests
    if os.path.exists(target_path):
        # Remove even if not empty
        shutil.rmtree(target_path)
    os.mkdir(target_path)

    # Then, convert vtp files
    mesh_parser = MeshParser(geometry_path)
    with pytest.raises(RuntimeError, match="The meshes have not been processed yet. Please run process_meshes first."):
        mesh_parser.write(target_path, MeshFormat.VTP)
    mesh_parser.process_meshes(fail_on_error=False)
    mesh_parser.write(target_path, MeshFormat.VTP)

    assert len(mesh_parser.meshes) == 317
    assert len(os.listdir(target_path)) == 313  # There are four .vtp files containing lines instead of polygons

    # Clean the files
    shutil.rmtree(target_path)
