from pathlib import Path

from biobuddy import MeshParser, MeshFormat
import pytest


def test_process_vtp_files(tmp_path: Path):

    # Paths
    current_path_file = Path(__file__).parent
    geometry_path = f"{current_path_file}/../external/opensim-models/Geometry"
    target_path = tmp_path / "geometry_processed"
    target_path.mkdir()

    # Then, convert vtp files
    mesh_parser = MeshParser(geometry_path)
    with pytest.raises(RuntimeError, match="The meshes have not been processed yet. Please run process_meshes first."):
        mesh_parser.write(str(target_path), MeshFormat.VTP)
    mesh_parser.process_meshes(fail_on_error=False)
    mesh_parser.write(str(target_path), MeshFormat.VTP)

    assert len(mesh_parser.meshes) == 317
    assert len(list(target_path.iterdir())) == 313  # There are four .vtp files containing lines instead of polygons
