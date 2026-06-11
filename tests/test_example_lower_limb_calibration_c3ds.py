from pathlib import Path

import ezc3d
import pytest

from biobuddy.gui.c3d_model_creation import C3dModelPreset, create_model_from_c3d_folder, find_static_c3d_file
from biobuddy.gui.lower_limb_template import LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES
from biobuddy.utils.marker_data import C3dData

EXAMPLE_FOLDER = Path(__file__).parent.parent / "examples" / "data" / "lower_limb_calibration"
EXPECTED_FUNCTIONAL_C3D_FILES = tuple(
    pattern.replace("*", "Test_") for pattern in LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES.values()
)
EXPECTED_C3D_FILES = (
    "Test_func_anat.c3d",
    "anatomical_posture.c3d",
    "functional_trunk.c3d",
    *EXPECTED_FUNCTIONAL_C3D_FILES,
)


@pytest.mark.parametrize("filename", EXPECTED_C3D_FILES)
def test_lower_limb_calibration_example_c3ds_are_lightweight_and_readable(filename):
    filepath = EXAMPLE_FOLDER / filename

    c3d = ezc3d.c3d(str(filepath))
    marker_data = C3dData(str(filepath))

    assert filepath.exists()
    assert c3d["data"]["points"].shape[2] > 0
    assert c3d["data"]["analogs"].shape == (1, 0, 0)
    assert c3d["parameters"]["ANALOG"]["USED"]["value"] == [0]
    assert c3d["parameters"]["POINT"]["RATE"]["value"] == [12.0]
    assert all(":" not in marker_name for marker_name in marker_data.marker_names)
    assert marker_data.nb_frames == c3d["data"]["points"].shape[2]
    assert len(marker_data.marker_names) == c3d["data"]["points"].shape[1]


def test_lower_limb_calibration_example_generates_functional_model():
    result = create_model_from_c3d_folder(EXAMPLE_FOLDER, preset=C3dModelPreset.LOWER_LIMBS)

    assert find_static_c3d_file(EXAMPLE_FOLDER).name == "Test_func_anat.c3d"
    assert result.output_filename == "lower_body_functional.bioMod"
    assert result.marker_reports["static"].missing_markers == ()
    assert set(result.functional_data) == set(LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES)
    assert "LShank" in result.model.segment_names
