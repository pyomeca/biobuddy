from pathlib import Path

from biobuddy import BiomechanicalModelReal
from biobuddy.model_parser.fbx import FbxModelParser


def test_fbx_parser_extracts_the_fullbody_skeleton():
    """
    Parse the real FBX example and extract the expected skeleton roots.
    """
    parent_path = Path(__file__).resolve().parent.parent
    filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"

    parser = FbxModelParser(filepath=str(filepath))

    assert parser.version == 7400
    assert len(parser.root_ids) == 1
    root_node = parser.skeleton_nodes[parser.root_ids[0]]
    assert root_node.name == "Hips"
    assert root_node.node_type == "Root"
    assert "HeadEE" not in [node.name for node in parser.skeleton_nodes.values()]


def test_fbx_and_bvh_share_the_same_kinematic_topology():
    """
    Build models from FBX and BVH and compare their segment hierarchy.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    bvh_filepath = parent_path / "examples" / "models" / "fullbody_model.bvh"

    model_from_fbx = BiomechanicalModelReal().from_fbx(filepath=str(fbx_filepath))
    model_from_bvh = BiomechanicalModelReal().from_bvh(filepath=str(bvh_filepath))

    fbx_topology = [(segment.name, segment.parent_name) for segment in model_from_fbx.segments]
    bvh_topology = [(segment.name, segment.parent_name) for segment in model_from_bvh.segments]

    assert fbx_topology == bvh_topology


def test_fbx_model_can_be_exported_to_biomod(tmp_path: Path):
    """
    Export a converted FBX hierarchy to a biorbd-compatible ``.bioMod`` file.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    biomod_filepath = tmp_path / "fullbody_model_from_fbx.bioMod"

    model = BiomechanicalModelReal().from_fbx(filepath=str(fbx_filepath))
    model.to_biomod(filepath=str(biomod_filepath), with_mesh=False)

    content = biomod_filepath.read_text()
    assert "segment\tHips" in content
    assert "\ttranslations\txyz" in content
    assert "\trotations\txyz" in content
