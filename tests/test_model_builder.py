import numpy as np
import pytest

from biobuddy import DictData, Rotations, Translations
from biobuddy.gui.lower_limb_template import lower_limb_template
from biobuddy.gui.model_builder import (
    AxisSpec,
    MarkerEndpointSpec,
    build_generic_model,
    build_real_model,
    compute_dynamic_segment_frames,
    compute_frame_quality,
)
from biobuddy.gui.marker_editor import attach_marker_to_segment
from biobuddy.components.generic.rigidbody.axis import Axis


def test_lower_limb_template_classifies_markers_on_multiple_segments():
    template = lower_limb_template()

    marker_segments = template.marker_segments()

    assert marker_segments["LASI"] == ("Pelvis", "LThigh", "RThigh")
    assert marker_segments["LKNE"] == ("LThigh", "LShank")
    assert marker_segments["LANK"] == ("LShank", "LFoot")


def test_endpoint_and_axis_specs_average_marker_groups():
    data = _synthetic_lower_limb_data()
    endpoint = MarkerEndpointSpec(("LPSI", "RPSI", "LASI", "RASI"))
    axis = AxisSpec.from_markers(Axis.Name.X, ("LPSI", "LASI"), ("RPSI", "RASI"))

    np.testing.assert_allclose(endpoint.evaluate(data)[:, 0], np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(axis.vector(data)[:, 0], np.array([0.2, 0.0, 0.0]))


def test_lower_limb_template_builds_expected_generic_model():
    template = lower_limb_template()
    model = build_generic_model(template)

    assert model.segment_names == [
        "root",
        "Pelvis",
        "Trunk",
        "LThigh",
        "LShank",
        "LFoot",
        "RThigh",
        "RShank",
        "RFoot",
    ]
    assert model.segments["Pelvis"].translations == Translations.XYZ
    assert model.segments["Pelvis"].rotations == Rotations.XYZ
    assert model.segments["LThigh"].rotations == Rotations.XZY
    assert model.segments["LShank"].rotations == Rotations.X
    assert model.segments["LFoot"].rotations == Rotations.XZ
    assert "LKNE" in model.segments["LThigh"].marker_names
    assert "LKNE" in model.segments["LShank"].marker_names


def test_quality_metrics_are_computed_before_orthonormalization():
    template = lower_limb_template()
    quality = compute_frame_quality(template, _synthetic_lower_limb_data())

    assert set(quality) == {"Pelvis", "Trunk", "LThigh", "LShank", "LFoot", "RThigh", "RShank", "RFoot"}
    assert quality["Pelvis"].mean_angle_degrees == pytest.approx(90.0)
    assert quality["LFoot"].first_axis_norm.shape == (3,)
    assert np.all(np.isfinite(quality["LFoot"].angle_degrees))


def test_dynamic_segment_frames_follow_marker_motion_and_marker_means():
    template = lower_limb_template()
    data = _synthetic_lower_limb_data()

    frames = compute_dynamic_segment_frames(template, data)

    assert frames["Pelvis"].shape == (4, 4, 3)
    np.testing.assert_allclose(frames["Pelvis"][:3, 3, 0], np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(frames["Pelvis"][:3, 3, 2], np.array([0.02, 0.0, 1.0]))
    np.testing.assert_allclose(frames["Pelvis"][:3, 0, 0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(frames["Pelvis"][:3, 1, 0], np.array([0.0, 1.0, 0.0]))


def test_lower_limb_template_generates_real_model_from_static_markers():
    template = lower_limb_template()

    model = build_real_model(template=template, static_data=_synthetic_lower_limb_data())

    assert "root" not in model.segment_names
    assert model.segments["Pelvis"].parent_name == "base"
    assert model.segments["LThigh"].parent_name == "Pelvis"
    assert model.segments["LShank"].parent_name == "LThigh"
    assert model.segments["LFoot"].parent_name == "LShank"
    assert model.segments["Pelvis"].nb_q == 6
    assert model.segments["LShank"].nb_q == 1
    assert "LKNE" in model.segments["LThigh"].marker_names
    assert "LKNE" in model.segments["LShank"].marker_names


def test_generated_lower_limb_model_can_be_saved_to_biomod(tmp_path):
    model = build_real_model(template=lower_limb_template(), static_data=_synthetic_lower_limb_data())
    filepath = tmp_path / "lower_body.bioMod"

    model.to_biomod(filepath=str(filepath), with_mesh=False)

    content = filepath.read_text()
    assert "segment\tPelvis" in content
    assert "segment\tLThigh" in content
    assert "marker\tLKNE" in content


def test_marker_can_be_attached_to_another_segment_without_moving_globally():
    model = build_real_model(template=lower_limb_template(), static_data=_synthetic_lower_limb_data())

    source_global = (
        model.segment_coordinate_system_in_global("LFoot") @ model.segments["LFoot"].markers["LTOE"].position
    )
    attached = attach_marker_to_segment(
        model=model,
        source_segment_name="LFoot",
        marker_name="LTOE",
        target_segment_name="LShank",
    )
    target_global = model.segment_coordinate_system_in_global("LShank") @ attached.position

    assert "LTOE" in model.segments["LShank"].marker_names
    np.testing.assert_allclose(target_global, source_global)


def _synthetic_lower_limb_data() -> DictData:
    points = {
        "LPSI": [-0.1, -0.1, 1.0],
        "RPSI": [0.1, -0.1, 1.0],
        "LASI": [-0.1, 0.1, 1.0],
        "RASI": [0.1, 0.1, 1.0],
        "C7": [0.0, -0.1, 1.5],
        "C2": [0.0, -0.1, 1.65],
        "T6": [0.0, -0.1, 1.35],
        "T10": [0.0, -0.1, 1.2],
        "S1": [0.0, -0.08, 1.05],
        "S3": [0.0, -0.1, 1.0],
        "CLAV": [0.0, 0.15, 1.5],
        "STRN": [0.0, 0.1, 1.4],
        "LTHI": [-0.2, 0.02, 0.8],
        "LTHIB": [-0.1, 0.03, 0.78],
        "LTHID": [-0.15, -0.03, 0.7],
        "LKNE": [-0.15, 0.05, 0.55],
        "LKNEM": [-0.05, 0.05, 0.55],
        "LTIB": [-0.18, 0.02, 0.35],
        "LTIBF": [-0.08, 0.02, 0.35],
        "LTIBD": [-0.13, -0.03, 0.25],
        "LANK": [-0.15, 0.0, 0.1],
        "LANKM": [-0.05, 0.0, 0.1],
        "LHEE": [-0.1, -0.05, 0.05],
        "LNAV": [-0.1, 0.15, 0.08],
        "LTOE": [-0.1, 0.35, 0.05],
        "LTOE5": [-0.18, 0.3, 0.05],
        "RTHI": [0.2, 0.02, 0.8],
        "RTHIB": [0.1, 0.03, 0.78],
        "RTHID": [0.15, -0.03, 0.7],
        "RKNE": [0.15, 0.05, 0.55],
        "RKNEM": [0.05, 0.05, 0.55],
        "RTIB": [0.18, 0.02, 0.35],
        "RTIBF": [0.08, 0.02, 0.35],
        "RTIBD": [0.13, -0.03, 0.25],
        "RANK": [0.15, 0.0, 0.1],
        "RANKM": [0.05, 0.0, 0.1],
        "RHEE": [0.1, -0.05, 0.05],
        "RNAV": [0.1, 0.15, 0.08],
        "RTOE": [0.1, 0.35, 0.05],
        "RTOE5": [0.18, 0.3, 0.05],
    }
    marker_dict = {}
    translation = np.array(
        [
            [0.0, 0.01, 0.02],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    for name, position in points.items():
        marker = np.ones((4, 3))
        marker[:3, :] = np.asarray(position, dtype=float)[:, np.newaxis] + translation
        marker_dict[name] = marker
    return DictData(marker_dict)
