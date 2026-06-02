from pathlib import Path
import shutil
import struct

import numpy as np
import pytest

from biobuddy import BiomechanicalModelReal
from biobuddy.model_parser.fbx.fbx_model_parser import FbxModelParser, _FbxNodeRecord
from biobuddy.model_writer.fbx import FbxModelWriter

from test_utils import compare_models


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

    model_from_biomod = BiomechanicalModelReal().from_biomod(filepath=str(biomod_filepath))
    compare_models(model, model_from_biomod, decimal=5)


def test_translation_fbx_to_biomod_to_fbx(tmp_path: Path):
    """
    Convert FBX to bioMod, then export the bioMod model back to FBX.

    The final FBX export uses Blender. The test is skipped when Blender is not
    installed because Blender is intentionally an optional writer dependency.
    """
    blender_executable = shutil.which("blender")
    if blender_executable is None:
        pytest.skip("Blender is required to run the FBX writer round-trip test.")

    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    biomod_filepath = tmp_path / "fullbody_model_from_fbx.bioMod"
    fbx_translated_filepath = tmp_path / "fullbody_model_translated.fbx"

    model_from_fbx = BiomechanicalModelReal().from_fbx(filepath=str(fbx_filepath))
    model_from_fbx.to_biomod(filepath=str(biomod_filepath), with_mesh=False)

    model_from_biomod = BiomechanicalModelReal().from_biomod(filepath=str(biomod_filepath))
    model_from_biomod.to_fbx(
        filepath=str(fbx_translated_filepath),
        with_mesh=False,
        blender_executable=blender_executable,
    )

    model_from_fbx_2 = BiomechanicalModelReal().from_fbx(filepath=str(fbx_translated_filepath))
    compare_models(model_from_fbx, model_from_fbx_2, decimal=4)


def test_fbx_writer_requires_blender_when_the_executable_is_missing(tmp_path: Path):
    """
    Fail with a clear message when the optional Blender backend is unavailable.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    model = BiomechanicalModelReal().from_fbx(filepath=str(fbx_filepath))

    with pytest.raises(RuntimeError, match="FBX export requires Blender"):
        model.to_fbx(
            filepath=str(tmp_path / "missing_blender.fbx"),
            with_mesh=False,
            blender_executable="definitely-not-a-real-blender-executable",
        )


def test_fbx_writer_serializes_segment_transforms():
    """
    Prepare a Blender payload with both local and global segment transforms.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    model = BiomechanicalModelReal().from_fbx(filepath=str(fbx_filepath))
    writer = FbxModelWriter(filepath="unused.fbx", with_mesh=False)

    payload = writer._payload(model)
    hips_payload = next(segment for segment in payload["segments"] if segment["name"] == "Hips")

    assert payload["armature_name"] == "unused"
    assert len(payload["segments"]) == model.nb_segments
    assert hips_payload["parent_name"] == "root"
    assert np.array(hips_payload["local_translation"]).shape == (3,)
    assert np.array(hips_payload["global_rotation"]).shape == (3, 3)


def test_fbx_writer_rejects_mesh_export(tmp_path: Path):
    """
    Keep the first FBX writer scoped to skeleton export until mesh support lands.
    """
    parent_path = Path(__file__).resolve().parent.parent
    fbx_filepath = parent_path / "examples" / "models" / "fullbody_model.fbx"
    model = BiomechanicalModelReal().from_fbx(filepath=str(fbx_filepath))

    with pytest.raises(NotImplementedError, match="FBX mesh export is not implemented yet"):
        model.to_fbx(
            filepath=str(tmp_path / "with_mesh.fbx"),
            with_mesh=True,
            blender_executable="definitely-not-a-real-blender-executable",
        )


def test_fbx_parser_rejects_non_binary_files(tmp_path: Path):
    """
    Reject ASCII or unrelated files before attempting to parse records.
    """
    filepath = tmp_path / "not_binary.fbx"
    filepath.write_text("FBXHeaderExtension: {}", encoding="utf-8")

    with pytest.raises(ValueError, match="Only binary FBX files are supported"):
        FbxModelParser(filepath=str(filepath))


def test_fbx_parser_decodes_scalar_properties():
    """
    Cover the scalar FBX property readers used by the binary parser.
    """
    value, cursor = FbxModelParser._parse_property(b"S" + struct.pack("<I", 4) + b"Hips", 0)
    assert value == "Hips"
    assert cursor == 9

    value, cursor = FbxModelParser._parse_property(b"I" + struct.pack("<i", 42), 0)
    assert value == 42
    assert cursor == 5

    value, cursor = FbxModelParser._parse_property(b"D" + struct.pack("<d", 1.25), 0)
    assert value == pytest.approx(1.25)
    assert cursor == 9


def test_fbx_parser_rejects_unknown_property_types():
    """
    Reject property types that are not implemented by the minimal parser.
    """
    with pytest.raises(NotImplementedError, match="Unsupported FBX property type"):
        FbxModelParser._parse_property(b"Z", 0)


def test_fbx_parser_property_helpers():
    """
    Exercise the small helpers that normalize FBX names and property vectors.
    """
    model_record = _FbxNodeRecord(
        name="Model",
        children=[
            _FbxNodeRecord(
                name="Properties70",
                children=[
                    _FbxNodeRecord(
                        name="P",
                        properties=["Lcl Translation", "Lcl Translation", "", "A", 1.0, 2.0, 3.0],
                    )
                ],
            )
        ],
    )

    properties = FbxModelParser._properties70_dict(model_record)

    assert FbxModelParser._clean_name("Model:RightArm\x00\x01") == "RightArm"
    assert np.allclose(FbxModelParser._vector3(properties, "Lcl Translation"), [1.0, 2.0, 3.0])
    assert np.allclose(FbxModelParser._vector3(properties, "Missing"), [0.0, 0.0, 0.0])
