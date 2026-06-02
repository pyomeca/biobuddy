import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import BiomechanicalModelReal
from biobuddy.model_parser.bvh import BvhModelParser
from biobuddy.utils.enums import Rotations, Translations

BVH_CONTENT = """HIERARCHY
ROOT Hips
{
    OFFSET 0 0 0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
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
1 2 3 40 50 60 4 5 6
"""


def test_bvh_parser_reads_hierarchy_and_motion(tmp_path):
    """
    Parse a minimal BVH file and expose both the hierarchy and motion data.
    """
    filepath = tmp_path / "minimal.bvh"
    filepath.write_text(BVH_CONTENT)

    parser = BvhModelParser(filepath=str(filepath))

    assert parser.root.name == "Hips"
    assert parser.frame_count == 2
    assert parser.frame_time == pytest.approx(0.0333333)
    npt.assert_array_equal(
        parser.motion_data,
        np.array(
            [
                [0, 0, 0, 10, 20, 30, 1, 2, 3],
                [1, 2, 3, 40, 50, 60, 4, 5, 6],
            ]
        ),
    )


def test_bvh_parser_converts_channels_to_biobuddy_segments(tmp_path):
    """
    Convert BVH channels and offsets into biobuddy segment definitions.
    """
    filepath = tmp_path / "minimal.bvh"
    filepath.write_text(BVH_CONTENT)

    model = BiomechanicalModelReal().from_bvh(filepath=str(filepath))

    assert model.segment_names == ["root", "Hips", "Knee"]
    assert model.segments["Hips"].parent_name == "root"
    assert model.segments["Hips"].translations == Translations.XYZ
    assert model.segments["Hips"].rotations == Rotations.ZXY
    assert model.segments["Knee"].translations == Translations.NONE
    assert model.segments["Knee"].rotations == Rotations.XYZ
    npt.assert_array_equal(
        model.segments["Knee"].segment_coordinate_system.scs.translation,
        np.array([0, -1, 0]),
    )


def test_bvh_model_can_be_exported_to_biomod(tmp_path):
    """
    Export a converted BVH hierarchy to a biorbd-compatible ``.bioMod`` file.
    """
    bvh_filepath = tmp_path / "minimal.bvh"
    biomod_filepath = tmp_path / "minimal.bioMod"
    bvh_filepath.write_text(BVH_CONTENT)

    model = BiomechanicalModelReal().from_bvh(filepath=str(bvh_filepath))
    model.to_biomod(filepath=str(biomod_filepath), with_mesh=False)

    content = biomod_filepath.read_text()
    assert "segment\tHips" in content
    assert "\ttranslations\txyz" in content
    assert "\trotations\tzxy" in content


def test_bvh_parser_rejects_motion_rows_with_wrong_channel_count(tmp_path):
    """
    Reject BVH motion data when the sample width does not match the hierarchy channels.
    """
    filepath = tmp_path / "bad_motion.bvh"
    filepath.write_text(BVH_CONTENT.replace("4 5 6", "4 5"))

    with pytest.raises(ValueError, match="Each BVH motion row must contain 9 channel values."):
        BvhModelParser(filepath=str(filepath))
