from biobuddy import BiomechanicalModelReal, validate_model_for_editor
from biobuddy.components.real.rigidbody.segment_real import SegmentReal


def test_validate_model_for_editor_reports_success():
    """
    Return a success message for a valid model.
    """
    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="Pelvis"))

    report = validate_model_for_editor(model)

    assert report.is_valid is True
    assert report.messages == ["Model validation passed."]
    assert report.category == "ok"


def test_validate_model_for_editor_reports_errors():
    """
    Return validation failures as user-facing messages.
    """
    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="Pelvis"))
    model.segments["Pelvis"].parent_name = "Missing"

    report = validate_model_for_editor(model)

    assert report.is_valid is False
    assert report.messages
    assert report.category == "kinematic_chain"
