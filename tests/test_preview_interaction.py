from dataclasses import dataclass

from biobuddy.gui.model_editor import _nearest_projected_segment


@dataclass
class _Point:
    x_value: float
    y_value: float

    def x(self):
        return self.x_value

    def y(self):
        return self.y_value


def test_nearest_projected_segment_returns_close_match():
    """
    Resolve the segment nearest to a clicked preview location.
    """
    projected = {"Pelvis": _Point(10, 10), "Thigh": _Point(50, 50)}
    clicked = _Point(12, 12)

    assert _nearest_projected_segment(projected, clicked) == "Pelvis"


def test_nearest_projected_segment_ignores_far_clicks():
    """
    Ignore clicks that do not land near any projected segment.
    """
    projected = {"Pelvis": _Point(10, 10)}
    clicked = _Point(100, 100)

    assert _nearest_projected_segment(projected, clicked) is None


def test_nearest_projected_segment_can_be_reused_for_markers():
    """
    The same helper can resolve marker hits before segment hits.
    """
    projected_markers = {"LASI": _Point(5, 5)}
    clicked = _Point(6, 6)

    assert _nearest_projected_segment(projected_markers, clicked) == "LASI"
