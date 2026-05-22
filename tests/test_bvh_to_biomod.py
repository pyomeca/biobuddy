import os
from pathlib import Path

import numpy as np
import pytest

from biobuddy import (
    BiomechanicalModelReal,
    MarkerReal,
    SegmentCoordinateSystemReal,
    SegmentReal,
)
from biobuddy.components.generic.rigidbody.range_of_motion import RangeOfMotion, Ranges
from biobuddy.model_parser.bvh import BvhModelParser
from biobuddy.utils.enums import Translations

from test_utils import compare_models


def test_translation_bvh_to_biomod():
    """Test comprehensive BVH to BioMod translation."""

    parent_path = Path(__file__).resolve().parent.parent
    bvh_filepath = parent_path / "examples" / "models" / "fullbody_model.bvh"
    biomod_reference_filepath = (
        parent_path / "examples" / "models" / "fullbody_model_from_bvh.bioMod"
    )
    biomod_translated_filepath = bvh_filepath.with_name(
        "fullbody_model_translated.bioMod"
    )
    bvh_translated_filepath = bvh_filepath.with_name("fullbody_model_translated.bvh")

    if biomod_translated_filepath.exists():
        os.remove(biomod_translated_filepath)
    if bvh_translated_filepath.exists():
        os.remove(bvh_translated_filepath)

    # Convert BVH to biomod and check that the converted model matches the reference.
    model_from_bvh = BiomechanicalModelReal().from_bvh(filepath=str(bvh_filepath))
    model_from_biomod = BiomechanicalModelReal().from_biomod(
        filepath=str(biomod_reference_filepath)
    )
    compare_models(model_from_bvh, model_from_biomod, decimal=5)

    # Test that the model created can be exported into .bioMod.
    model_from_bvh.to_biomod(filepath=str(biomod_translated_filepath), with_mesh=False)
    model_from_biomod_2 = BiomechanicalModelReal().from_biomod(
        filepath=str(biomod_translated_filepath)
    )
    compare_models(model_from_bvh, model_from_biomod_2, decimal=5)

    # Test that the .bioMod can be reconverted into .bvh.
    model_from_biomod.to_bvh(filepath=str(bvh_translated_filepath), with_mesh=False)
    model_from_bvh_2 = BiomechanicalModelReal().from_bvh(
        filepath=str(bvh_translated_filepath)
    )
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


def test_bvh_root_offset_is_preserved_in_model_and_biomod(tmp_path: Path):
    """Preserve the BVH root offset on the exported root joint segment."""

    parent_path = Path(__file__).resolve().parent.parent
    filepath = parent_path / "examples" / "models" / "fullbody_model.bvh"
    biomod_filepath = tmp_path / "fullbody_model_root_offset.bioMod"

    model = BiomechanicalModelReal().from_bvh(filepath=str(filepath))
    hips_segment = model.segments["Hips"]

    np.testing.assert_allclose(
        hips_segment.segment_coordinate_system.scs.translation.reshape(-1),
        np.array([0.0, 960.822, 40.4592]),
    )

    model.to_biomod(filepath=str(biomod_filepath), with_mesh=False)
    content = biomod_filepath.read_text()

    assert "segment\tHips" in content
    assert "\tparent\troot" in content
    assert "960.822000" in content
    assert "40.459200" in content


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

    with pytest.raises(
        ValueError, match="Each BVH motion row must contain 9 channel values."
    ):
        BvhModelParser(filepath=str(filepath))


def test_bvh_parser_rejects_files_without_hierarchy(tmp_path: Path):
    """Reject files that do not start with a BVH hierarchy block."""

    filepath = tmp_path / "bad_header.bvh"
    filepath.write_text("MOTION\nFrames: 0\nFrame Time: 0.0333333\n")

    with pytest.raises(
        ValueError, match="A BVH file must start with a HIERARCHY block."
    ):
        BvhModelParser(filepath=str(filepath))


def test_bvh_parser_rejects_invalid_joint_channel_declaration(tmp_path: Path):
    """Reject joints whose declared channel count does not match the provided channels."""

    filepath = tmp_path / "bad_channels.bvh"
    filepath.write_text("""HIERARCHY
ROOT root
{
    OFFSET 0 0 0
    CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation
}
""")

    with pytest.raises(
        ValueError, match="Joint root declares 6 channels but provides 5."
    ):
        BvhModelParser(filepath=str(filepath))


def test_bvh_parser_accepts_hierarchy_without_motion_block(tmp_path: Path):
    """Allow loading a pure BVH hierarchy even when no motion samples are provided."""

    filepath = tmp_path / "hierarchy_only.bvh"
    filepath.write_text("""HIERARCHY
ROOT root
{
    OFFSET 0 0 0
    CHANNELS 0
}
""")

    parser = BvhModelParser(filepath=str(filepath))

    assert parser.root is not None
    assert parser.root.name == "root"
    assert parser.frame_count is None
    assert parser.frame_time is None
    assert parser.motion_data is None


def test_bvh_parser_rejects_invalid_end_site_block(tmp_path: Path):
    """Reject malformed BVH end sites."""

    filepath = tmp_path / "bad_end_site.bvh"
    filepath.write_text("""HIERARCHY
ROOT root
{
    OFFSET 0 0 0
    CHANNELS 0
    End Site
    {
        CHANNELS 0
    }
}
""")

    with pytest.raises(ValueError, match="Expected an OFFSET line inside End Site."):
        BvhModelParser(filepath=str(filepath))


def test_bvh_writer_exports_a_minimal_root_hierarchy(tmp_path: Path):
    """Export a minimal root-only model to BVH."""

    filepath = tmp_path / "minimal.bvh"
    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="root"))

    model.to_bvh(filepath=str(filepath), with_mesh=False)

    content = filepath.read_text()
    assert "ROOT root" in content
    assert "CHANNELS 0" in content
    assert "Frames: 1" in content


@pytest.mark.parametrize(
    "model_factory, expected_message",
    [
        (
            lambda: BiomechanicalModelReal(gravity=np.array([0.0, -9.81, 0.0])),
            "BVH export does not support gravity metadata.",
        ),
        (
            lambda: _model_with_marker(),
            "BVH export does not support segment markers. Segment root cannot be exported.",
        ),
        (
            lambda: _model_with_segment_rotation(),
            "BVH export currently only supports identity local segment rotations. Segment root is rotated.",
        ),
        (
            lambda: _model_with_multiple_roots(),
            "BVH export requires exactly one root segment attached to base.",
        ),
    ],
)
def test_bvh_writer_rejects_unsupported_model_features(
    tmp_path: Path, model_factory, expected_message: str
):
    """Reject model structures that cannot be represented faithfully in BVH."""

    filepath = tmp_path / "unsupported.bvh"
    model = model_factory()

    with pytest.raises((RuntimeError, NotImplementedError), match=expected_message):
        model.to_bvh(filepath=str(filepath), with_mesh=False)


def test_bvh_writer_rejects_segment_ranges(tmp_path: Path):
    """Reject segment range metadata during BVH export."""

    filepath = tmp_path / "ranges.bvh"
    model = BiomechanicalModelReal()
    model.add_segment(
        SegmentReal(
            name="root",
            translations=Translations.X,
            q_ranges=RangeOfMotion(
                range_type=Ranges.Q, min_bound=[-1.0], max_bound=[1.0]
            ),
        )
    )

    with pytest.raises(
        NotImplementedError,
        match="BVH export does not support segment q ranges. Segment root cannot be exported.",
    ):
        model.to_bvh(filepath=str(filepath), with_mesh=False)


def _model_with_marker() -> BiomechanicalModelReal:
    model = BiomechanicalModelReal()
    segment = SegmentReal(name="root")
    segment.add_marker(MarkerReal(name="marker", position=np.zeros(3)))
    model.add_segment(segment)
    return model


def _model_with_segment_rotation() -> BiomechanicalModelReal:
    model = BiomechanicalModelReal()
    model.add_segment(
        SegmentReal(
            name="root",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                angles=np.array([0.1, 0.0, 0.0]),
                angle_sequence="xyz",
                translation=np.zeros(3),
                is_scs_local=True,
            ),
        )
    )
    return model


def _model_with_multiple_roots() -> BiomechanicalModelReal:
    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="root"))
    model.add_segment(SegmentReal(name="other_root", parent_name="base"))
    return model
