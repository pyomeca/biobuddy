import os
from pathlib import Path

import pytest

from biobuddy import BiomechanicalModelReal
from biobuddy.model_parser.bvh import BvhModelParser

from test_utils import compare_models


def test_translation_bvh_to_biomod():
    """Test comprehensive BVH to BioMod translation."""

    parent_path = Path(__file__).resolve().parent.parent
    bvh_filepath = parent_path / "examples" / "models" / "fullbody_model.bvh"
    biomod_reference_filepath = parent_path / "examples" / "models" / "fullbody_model_from_bvh.bioMod"
    biomod_translated_filepath = bvh_filepath.with_name("fullbody_model_translated.bioMod")
    bvh_translated_filepath = bvh_filepath.with_name("fullbody_model_translated.bvh")

    if biomod_translated_filepath.exists():
        os.remove(biomod_translated_filepath)
    if bvh_translated_filepath.exists():
        os.remove(bvh_translated_filepath)

    # Convert BVH to biomod and check that the converted model matches the reference.
    model_from_bvh = BiomechanicalModelReal().from_bvh(filepath=str(bvh_filepath))
    model_from_biomod = BiomechanicalModelReal().from_biomod(filepath=str(biomod_reference_filepath))
    compare_models(model_from_bvh, model_from_biomod, decimal=5)

    # Test that the model created can be exported into .bioMod.
    model_from_bvh.to_biomod(filepath=str(biomod_translated_filepath), with_mesh=False)
    model_from_biomod_2 = BiomechanicalModelReal().from_biomod(filepath=str(biomod_translated_filepath))
    compare_models(model_from_bvh, model_from_biomod_2, decimal=5)

    # Test that the .bioMod can be reconverted into .bvh.
    model_from_biomod.to_bvh(filepath=str(bvh_translated_filepath), with_mesh=False)
    model_from_bvh_2 = BiomechanicalModelReal().from_bvh(filepath=str(bvh_translated_filepath))
    compare_models(model_from_bvh, model_from_bvh_2, decimal=5)

    if biomod_translated_filepath.exists():
        os.remove(biomod_translated_filepath)
    if bvh_translated_filepath.exists():
        os.remove(bvh_translated_filepath)


def test_bvh_parser_reads_hierarchy_and_motion():
    """Parse the real BVH example and expose both the hierarchy and motion data."""

    parent_path = Path(__file__).resolve().parent.parent
    filepath = parent_path / "examples" / "models" / "fullbody_model.bvh"

    parser = BvhModelParser(filepath=str(filepath))

    assert parser.root is not None
    assert parser.root.name == "Hips"
    assert parser.frame_count == 1977
    assert parser.frame_time == pytest.approx(0.038462)
    assert parser.motion_data is not None
    assert parser.motion_data.shape == (1977, parser._count_channels(parser.root))


def test_bvh_parser_rejects_motion_rows_with_wrong_channel_count(tmp_path: Path):
    """Reject BVH motion data when the sample width does not match the hierarchy channels."""

    filepath = tmp_path / "bad_motion.bvh"
    filepath.write_text("""HIERARCHY
ROOT root
{
    OFFSET 0 0 0
    CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
    JOINT Knee
    {
        OFFSET 0 -1 0
        CHANNELS 3 Xrotation Yrotation Zrotation
        End Site
        {
            OFFSET 0 -1 0
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.0333333
0 0 0 10 20 30 1 2 3
1 2 3 40 50 60 4 5
""")

    with pytest.raises(ValueError, match="Each BVH motion row must contain 9 channel values."):
        BvhModelParser(filepath=str(filepath))
