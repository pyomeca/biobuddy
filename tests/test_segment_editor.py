import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import (
    BiomechanicalModelReal,
    SegmentEditorData,
    apply_segment_editor_data,
    get_segment_editor_data,
)
from biobuddy.gui.segment_editor import validate_parent_name
from biobuddy.gui.segment_editor import load_model
from biobuddy.components.generic.rigidbody.range_of_motion import RangeOfMotion, Ranges
from biobuddy.components.real.rigidbody.inertia_parameters_real import (
    InertiaParametersReal,
)
from biobuddy.components.real.rigidbody.segment_real import SegmentReal
from biobuddy.utils.enums import Rotations, Translations


def test_get_segment_editor_data_extracts_editable_values():
    """
    Convert a populated segment into form-friendly values.
    """
    segment = SegmentReal(
        name="Thigh",
        parent_name="Pelvis",
        translations=Translations.X,
        rotations=Rotations.YZ,
        q_ranges=RangeOfMotion(Ranges.Q, [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]),
        inertia_parameters=InertiaParametersReal(
            mass=8.0,
            center_of_mass=np.array([0.1, 0.2, 0.3]),
            inertia=np.diag([1.0, 2.0, 3.0]),
        ),
    )

    data = get_segment_editor_data(segment)

    assert data.parent_name == "Pelvis"
    assert data.translations == "x"
    assert data.rotations == "yz"
    assert data.q_min == [-1.0, -2.0, -3.0]
    assert data.q_max == [1.0, 2.0, 3.0]
    assert data.mass == 8.0
    assert data.center_of_mass == [0.1, 0.2, 0.3]
    assert data.inertia_diagonal == [1.0, 2.0, 3.0]


def test_apply_segment_editor_data_updates_segment():
    """
    Apply edited fields back to a segment.
    """
    segment = SegmentReal(name="Shank")
    data = SegmentEditorData(
        parent_name="Thigh",
        translations="",
        rotations="xz",
        q_min=[-1.0, -2.0],
        q_max=[1.0, 2.0],
        mass=4.5,
        center_of_mass=[0.0, -0.2, 0.0],
        inertia_diagonal=[0.1, 0.2, 0.3],
    )

    apply_segment_editor_data(segment, data)

    assert segment.parent_name == "Thigh"
    assert segment.translations == Translations.NONE
    assert segment.rotations == Rotations.XZ
    assert segment.dof_names == ["Shank_rotX", "Shank_rotZ"]
    assert segment.q_ranges.min_bound == [-1.0, -2.0]
    assert segment.q_ranges.max_bound == [1.0, 2.0]
    assert segment.inertia_parameters.mass == 4.5
    npt.assert_array_equal(
        segment.inertia_parameters.center_of_mass[:3, 0], np.array([0.0, -0.2, 0.0])
    )
    npt.assert_array_equal(
        np.diag(segment.inertia_parameters.inertia)[:3], np.array([0.1, 0.2, 0.3])
    )


def test_apply_segment_editor_data_rejects_incompatible_ranges():
    """
    Reject edited ranges whose length no longer matches the segment DoFs.
    """
    segment = SegmentReal(name="Foot", rotations=Rotations.XY)
    data = SegmentEditorData(
        parent_name="Shank",
        translations="",
        rotations="xy",
        q_min=[-1.0],
        q_max=[1.0],
        mass=None,
        center_of_mass=[0.0, 0.0, 0.0],
        inertia_diagonal=[0.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="Expected either 0 or 2 range values"):
        apply_segment_editor_data(segment, data)


def test_validate_parent_name_rejects_unknown_and_self_parents():
    """
    Reject parent choices that would make the edited hierarchy invalid.
    """
    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="Pelvis"))
    model.add_segment(SegmentReal(name="Thigh", parent_name="Pelvis"))

    with pytest.raises(ValueError, match="Unknown parent segment"):
        validate_parent_name(model=model, segment_name="Thigh", parent_name="Missing")

    with pytest.raises(ValueError, match="cannot be its own parent"):
        validate_parent_name(model=model, segment_name="Thigh", parent_name="Thigh")


def test_load_model_supports_bvh(tmp_path):
    """
    Load a minimal BVH file through the GUI-facing loader.
    """
    filepath = tmp_path / "minimal.bvh"
    filepath.write_text(
        "\n".join(
            [
                "HIERARCHY",
                "ROOT Hips",
                "{",
                "    OFFSET 0 0 0",
                "    CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation",
                "}",
                "MOTION",
                "Frames: 1",
                "Frame Time: 0.01",
                "0 0 0 0 0 0",
            ]
        )
    )

    model = load_model(str(filepath))

    assert isinstance(model, BiomechanicalModelReal)
    assert "Hips" in model.segment_names
