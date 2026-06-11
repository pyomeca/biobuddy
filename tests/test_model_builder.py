import numpy as np
import pytest

from biobuddy import DictData, Rotations, Translations
from biobuddy.gui.lower_limb_template import LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES, lower_limb_template
from biobuddy.gui.c3d_model_creation import (
    C3dModelPreset,
    c3d_model_preset_virtual_features,
    create_lower_limb_model_variants_from_marker_data,
    create_model_from_marker_data,
    find_static_c3d_file,
    supported_c3d_model_presets,
    template_for_c3d_model_preset,
)
from biobuddy.gui.virtual_points import marker_pair_virtual_axis, pointing_virtual_point
from biobuddy.gui.model_builder import (
    AxisSpec,
    FunctionalAxisSpec,
    MarkerEndpointSpec,
    build_generic_model,
    build_real_model,
    compute_dynamic_segment_frames,
    compute_frame_quality,
    marker_availability,
    required_functional_markers,
    required_markers,
    required_static_markers,
    template_marker_availability,
)
from biobuddy.gui.marker_editor import attach_marker_to_segment
from biobuddy.components.generic.rigidbody.axis import Axis


def test_lower_limb_template_classifies_only_shared_joint_markers_on_multiple_segments():
    template = lower_limb_template()

    marker_segments = template.marker_segments()

    assert marker_segments["LASI"] == ("Pelvis",)
    assert marker_segments["RASI"] == ("Pelvis",)
    assert marker_segments["LKNE"] == ("LThigh", "LShank")
    assert marker_segments["LANK"] == ("LShank", "LFoot")


def test_endpoint_and_axis_specs_average_marker_groups():
    data = _synthetic_lower_limb_data()
    endpoint = MarkerEndpointSpec(("LPSI", "RPSI", "LASI", "RASI"))
    axis = AxisSpec.from_markers(Axis.Name.X, ("LPSI", "LASI"), ("RPSI", "RASI"))

    np.testing.assert_allclose(endpoint.evaluate(data)[:, 0], np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(axis.vector(data)[:, 0], np.array([0.2, 0.0, 0.0]))


def test_template_reports_required_markers_from_frames_and_functional_trials():
    template = lower_limb_template()

    static_markers = required_static_markers(template)
    functional_markers = required_functional_markers(template)
    all_required_markers = required_markers(template)

    assert static_markers == tuple(sorted(set(static_markers)))
    assert "LASI" in static_markers
    assert "LTOE5" in static_markers
    assert "left_knee_sara" in functional_markers
    assert functional_markers["left_knee_sara"] == tuple(
        sorted(("LTHI", "LTHIB", "LTHID", "LTIB", "LTIBF", "LTIBD", "LKNE", "LKNEM"))
    )
    assert all_required_markers["static"] == static_markers


def test_lower_limb_template_uses_forced_functional_c3d_names():
    template = lower_limb_template()

    file_patterns = {trial.name: trial.file_pattern for trial in template.functional_trials}

    assert file_patterns == LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES
    assert all("*" not in pattern for pattern in file_patterns.values())


def test_marker_availability_reports_presence_and_valid_frame_count():
    data = _synthetic_lower_limb_data()
    data.marker_dict["LASI"][0, 1] = np.nan

    report = marker_availability(data, ("LASI", "LPSI", "MISSING"))

    assert report.total_frame_count == 3
    assert report.complete_frame_count == 0
    assert report.missing_markers == ("MISSING",)
    assert report.markers["LASI"].valid_frame_count == 2
    assert report.markers["LPSI"].valid_frame_count == 3
    assert report.markers["MISSING"].valid_frame_count == 0


def test_template_marker_availability_reports_static_and_functional_trials():
    template = lower_limb_template()
    data = _synthetic_lower_limb_data()

    reports = template_marker_availability(template, data, {"left_knee_sara": data, "ignored": data})

    assert set(reports) == {"static", "left_knee_sara"}
    assert reports["static"].complete_frame_count == 3
    assert reports["left_knee_sara"].complete_frame_count == 3


def test_c3d_model_creation_from_marker_data_returns_model_and_reports():
    result = create_model_from_marker_data(
        template=lower_limb_template(),
        static_data=_synthetic_lower_limb_data(),
        preset=C3dModelPreset.LOWER_LIMBS,
    )

    assert result.preset == C3dModelPreset.LOWER_LIMBS
    assert result.output_filename == "lower_body_functional.bioMod"
    assert "Pelvis" in result.model.segment_names
    assert result.marker_reports["static"].complete_frame_count == 3
    assert result.frame_quality["Pelvis"].mean_angle_degrees == pytest.approx(90.0)


def test_from_scratch_c3d_preset_is_template_free():
    assert C3dModelPreset.FROM_SCRATCH in supported_c3d_model_presets()
    with pytest.raises(NotImplementedError, match="Template-free"):
        template_for_c3d_model_preset(C3dModelPreset.FROM_SCRATCH)


def test_c3d_model_creation_applies_virtual_features_before_generation():
    original_data = _synthetic_lower_limb_data()
    marker_dict = dict(original_data.marker_dict)
    marker_dict["RawLASI"] = marker_dict.pop("LASI")
    data = DictData(marker_dict)

    result = create_model_from_marker_data(
        template=lower_limb_template(),
        static_data=data,
        preset=C3dModelPreset.LOWER_LIMBS,
        static_virtual_points=(pointing_virtual_point("LASI", "RawLASI"),),
        static_virtual_axes=(marker_pair_virtual_axis("PelvisLeftRight", ("LPSI", "RawLASI"), ("RPSI", "RASI")),),
    )

    assert "LASI" in result.static_data.marker_names
    assert "PelvisLeftRight_start" in result.static_data.marker_names
    assert "PelvisLeftRight_end" in result.static_data.marker_names
    assert result.marker_reports["static"].markers["LASI"].valid_frame_count == 3
    assert "Pelvis" in result.model.segment_names


def test_c3d_model_creation_presets_are_explicit_about_supported_generation():
    assert supported_c3d_model_presets() == (
        C3dModelPreset.FROM_SCRATCH,
        C3dModelPreset.FULL_BODY,
        C3dModelPreset.LOWER_LIMBS,
        C3dModelPreset.LOWER_LIMBS_ANATOMICAL,
        C3dModelPreset.UPPER_LIMB,
    )
    assert template_for_c3d_model_preset(C3dModelPreset.LOWER_LIMBS).root_segment_name == "Pelvis"
    assert required_functional_markers(template_for_c3d_model_preset(C3dModelPreset.LOWER_LIMBS)) != {}
    assert required_functional_markers(template_for_c3d_model_preset(C3dModelPreset.LOWER_LIMBS_ANATOMICAL)) == {}
    assert template_for_c3d_model_preset(C3dModelPreset.UPPER_LIMB).name == "Upper-limb from calibration C3D"
    with pytest.raises(NotImplementedError, match="Full-body C3D model creation"):
        template_for_c3d_model_preset(C3dModelPreset.FULL_BODY)
    with pytest.raises(NotImplementedError, match="Template-free"):
        template_for_c3d_model_preset(C3dModelPreset.FROM_SCRATCH)


def test_c3d_model_presets_report_virtual_features_to_reconstruct():
    lower_limb_features = c3d_model_preset_virtual_features(C3dModelPreset.LOWER_LIMBS)
    lower_limb_anatomical_features = c3d_model_preset_virtual_features(C3dModelPreset.LOWER_LIMBS_ANATOMICAL)
    upper_limb_features = c3d_model_preset_virtual_features(C3dModelPreset.UPPER_LIMB)
    full_body_features = c3d_model_preset_virtual_features(C3dModelPreset.FULL_BODY)

    assert any(
        feature.name == "CoR_LThigh_in_Pelvis"
        and feature.role == "score"
        and "parent markers=LPSI,RPSI,LASI,RASI" in feature.description
        and "child markers=LTHI,LTHIB,LTHID" in feature.description
        for feature in lower_limb_features
    )
    assert any(feature.name == "CoR_LFoot_in_LShank" and feature.role == "score" for feature in lower_limb_features)
    assert any(feature.name == "Axis_LKnee_SARA" and feature.role == "sara_axis" for feature in lower_limb_features)
    assert any(feature.name == "Axis_RKnee_SARA" and feature.feature_type == "axis" for feature in lower_limb_features)
    assert len(lower_limb_features) == 6
    assert lower_limb_anatomical_features == ()
    assert c3d_model_preset_virtual_features(C3dModelPreset.FROM_SCRATCH) == ()
    assert any(feature.name == "Thorax_virtual_7" for feature in upper_limb_features)
    assert any(feature.feature_type == "axis" and feature.name == "Clavicule_u_axis" for feature in upper_limb_features)
    assert any(feature.name == "CoR_Thorax_in_Thorax" and feature.role == "score" for feature in full_body_features)


def test_find_static_c3d_file_uses_expected_patterns(tmp_path):
    static_file = tmp_path / "subject_static.c3d"
    static_file.touch()

    assert find_static_c3d_file(tmp_path) == static_file

    (tmp_path / "another_static.c3d").touch()
    with pytest.raises(RuntimeError, match="Expected one static trial"):
        find_static_c3d_file(tmp_path)


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


def test_lower_limb_template_can_disable_score_and_sara():
    template = lower_limb_template(use_functional=False)

    assert required_functional_markers(template) == {}
    assert template.name.endswith("(anatomical markers only)")


def test_lower_limb_functional_template_uses_sara_for_knee_axis():
    template = lower_limb_template(use_functional=True)
    left_shank = next(segment for segment in template.segments if segment.name == "LShank")
    right_shank = next(segment for segment in template.segments if segment.name == "RShank")

    left_axis = left_shank.frame.second_axis
    right_axis = right_shank.frame.second_axis

    assert isinstance(left_axis, FunctionalAxisSpec)
    assert left_axis.trial_name == "left_knee_sara"
    assert left_axis.parent_marker_names == ("LTIBD", "LTIB", "LTIBF")
    assert left_axis.child_marker_names == ("LTHIB", "LTHID", "LTHI")
    assert left_axis.expected_axis.start.marker_names == ("LKNE",)
    assert left_axis.expected_axis.end.marker_names == ("LKNEM",)
    assert left_axis.origin_marker_names == ("LKNE", "LKNEM")
    assert isinstance(right_axis, FunctionalAxisSpec)
    assert right_axis.trial_name == "right_knee_sara"


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
    assert "LASI" in model.segments["Pelvis"].marker_names
    assert "LASI" not in model.segments["LThigh"].marker_names
    assert "RASI" not in model.segments["RThigh"].marker_names
    assert model.segments["Pelvis"].inertia_parameters is None
    assert model.segments["Trunk"].inertia_parameters.mass > 0
    assert model.segments["LThigh"].inertia_parameters.mass > 0
    assert model.segments["RFoot"].inertia_parameters.mass > 0


def test_lower_limb_model_creation_returns_score_and_no_score_variants():
    variants = create_lower_limb_model_variants_from_marker_data(static_data=_synthetic_lower_limb_data())

    assert variants.score.output_filename == "lower_body_score.bioMod"
    assert variants.no_score.output_filename == "lower_body_no_score.bioMod"
    assert variants.score.template.name.endswith("(SCoRE/SARA)")
    assert variants.no_score.template.name.endswith("(anatomical markers only)")
    assert "left_knee_sara" in required_functional_markers(variants.score.template)
    assert required_functional_markers(variants.no_score.template) == {}


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
