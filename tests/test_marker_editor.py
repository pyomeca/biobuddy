import numpy as np
import numpy.testing as npt

from biobuddy import (
    MarkerEditorData,
    add_marker,
    apply_marker_editor_data,
    get_marker_editor_data,
    remove_marker,
)
from biobuddy.components.real.rigidbody.marker_real import MarkerReal
from biobuddy.components.real.rigidbody.segment_real import SegmentReal


def test_marker_editor_round_trip():
    """
    Expose and apply editable marker fields.
    """
    marker = MarkerReal(
        name="LASI",
        parent_name="Pelvis",
        position=np.array([0.1, 0.2, 0.3]),
        is_technical=False,
        is_anatomical=True,
    )

    data = get_marker_editor_data(marker)
    assert data == MarkerEditorData("LASI", [0.1, 0.2, 0.3], False, True)

    apply_marker_editor_data(
        marker, MarkerEditorData("LASI2", [1.0, 2.0, 3.0], True, False)
    )
    assert marker.name == "LASI2"
    npt.assert_array_equal(marker.mean_position[:3], np.array([1.0, 2.0, 3.0]))
    assert marker.is_technical is True
    assert marker.is_anatomical is False


def test_add_and_remove_marker():
    """
    Add and remove markers from a segment through the editor helpers.
    """
    segment = SegmentReal(name="Pelvis")
    add_marker(segment, MarkerEditorData("RASI", [0.0, 1.0, 2.0], True, False))

    assert list(segment.markers.keys()) == ["RASI"]
    remove_marker(segment, "RASI")
    assert list(segment.markers.keys()) == []
