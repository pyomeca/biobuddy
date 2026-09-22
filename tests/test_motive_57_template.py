from biobuddy import Rotations, Translations
from biobuddy.components.generic.rigidbody.axis import Axis
from biobuddy.gui.model_builder import (
    FunctionalAxisProjectionPointSpec,
    FunctionalAxisSpec,
    FunctionalCenterSpec,
    FunctionalTrialSpec,
    build_generic_model,
    required_static_markers,
)
from biobuddy.gui.motive_57_template import (
    MOTIVE_57_FUNCTIONAL_C3D_FILENAMES,
    MOTIVE_57_MARKER_NAMES,
    motive_57_functional_trials,
    motive_57_marker_attachments,
    motive_57_template,
)


def test_motive_57_template_contains_expected_chain_and_markers():
    template = motive_57_template(use_functional=True)
    model = build_generic_model(template)
    attachments = motive_57_marker_attachments()
    marker_segments = template.marker_segments()

    assert template.root_segment_name == "Pelvis"
    assert template.name == "BioBuddy Motive (57) from calibration C3D (SCoRE/SARA)"
    assert [segment.name for segment in template.segments] == [
        "Pelvis",
        "Thorax",
        "Head",
        "RThigh",
        "RShank",
        "RFoot",
        "LThigh",
        "LShank",
        "LFoot",
        "RUpperArm",
        "RForearm",
        "RHand",
        "LUpperArm",
        "LForearm",
        "LHand",
    ]
    assert "RDP1" not in marker_segments
    assert "LDP1" not in marker_segments
    assert "RHGT" not in marker_segments
    assert "LHGT" not in marker_segments
    assert "RCAJ" not in marker_segments
    assert "LCAJ" not in marker_segments
    assert marker_segments["RFCC"] == ("RFoot",)
    assert marker_segments["RFM5"] == ("RFoot",)
    assert marker_segments["RFM1"] == ("RFoot",)
    assert marker_segments["RUA"] == ("RUpperArm",)
    assert len(attachments) == 55
    assert marker_segments["LFLE"] == ("LThigh",)
    assert marker_segments["LFME"] == ("LThigh",)
    assert marker_segments["RFLE"] == ("RThigh",)
    assert marker_segments["RFME"] == ("RThigh",)
    assert marker_segments["RUSP"] == ("RForearm", "RHand")
    assert marker_segments["RRSP"] == ("RForearm", "RHand")
    assert marker_segments["LUSP"] == ("LForearm", "LHand")
    assert marker_segments["LRSP"] == ("LForearm", "LHand")
    assert model.segments["Pelvis"].translations == Translations.XYZ
    assert model.segments["Pelvis"].rotations == Rotations.ZXY
    assert model.segments["RShank"].rotations == Rotations.Z
    assert model.segments["RFoot"].rotations == Rotations.ZX
    for side in ("L", "R"):
        thigh = model.segments[f"{side}Thigh"]
        shank = model.segments[f"{side}Shank"]
        assert all(
            thigh.markers[marker_name].is_technical
            for marker_name in (f"{side}FTC", f"{side}TH", f"{side}FLE", f"{side}FME")
        )
        assert f"{side}FLE" not in shank.markers
        assert f"{side}FME" not in shank.markers
        assert all(
            shank.markers[marker_name].is_technical
            for marker_name in (
                f"{side}FAX",
                f"{side}SK",
                f"{side}TTC",
                f"{side}FAL",
                f"{side}TAM",
            )
        )
        hand = model.segments[f"{side}Hand"]
        assert all(hand.markers[marker_name].is_technical for marker_name in (f"{side}HM2", f"{side}USP", f"{side}RSP"))


def test_motive_57_template_uses_correct_anatomical_frames():
    segments = {segment.name: segment for segment in motive_57_template(use_functional=True).segments}

    pelvis = segments["Pelvis"].frame
    assert pelvis.first_axis.name == Axis.Name.Z
    assert pelvis.first_axis.start.marker_names == ("LIAS", "LIPS")
    assert pelvis.first_axis.end.marker_names == ("RIAS", "RIPS")
    assert pelvis.second_axis.name == Axis.Name.X
    assert pelvis.second_axis.start.marker_names == ("LIPS", "RIPS")
    assert pelvis.second_axis.end.marker_names == ("LIAS", "RIAS")
    assert pelvis.axis_to_keep == Axis.Name.Z

    thorax = segments["Thorax"].frame
    assert thorax.first_axis.name == Axis.Name.Y
    assert thorax.first_axis.start.marker_names == ("SXS", "TV7")
    assert thorax.first_axis.end.marker_names == ("SJN", "TV2")

    head = segments["Head"].frame
    assert head.first_axis.name == Axis.Name.Z
    assert head.second_axis.name == Axis.Name.X

    right_foot = segments["RFoot"].frame
    assert right_foot.first_axis.name == Axis.Name.X
    assert right_foot.first_axis.start.marker_names == ("RFCC", "RFCC")
    assert right_foot.first_axis.end.marker_names == ("RFM5", "RFM1")
    assert right_foot.second_axis.name == Axis.Name.Z
    assert right_foot.second_axis.start.marker_names == ("RFM1",)
    assert right_foot.second_axis.end.marker_names == ("RFM5",)

    left_foot = segments["LFoot"].frame
    assert left_foot.first_axis.name == Axis.Name.X
    assert left_foot.first_axis.start.marker_names == ("LFCC", "LFCC")
    assert left_foot.first_axis.end.marker_names == ("LFM5", "LFM1")
    assert left_foot.second_axis.name == Axis.Name.Z
    assert left_foot.second_axis.start.marker_names == ("LFM5",)
    assert left_foot.second_axis.end.marker_names == ("LFM1",)

    right_thigh = segments["RThigh"].frame
    left_thigh = segments["LThigh"].frame
    assert right_thigh.first_axis.name == Axis.Name.Y
    assert left_thigh.first_axis.name == Axis.Name.Y
    assert right_thigh.second_axis.fallback.start.marker_names == ("RFME",)
    assert right_thigh.second_axis.fallback.end.marker_names == ("RFLE",)
    assert left_thigh.second_axis.fallback.start.marker_names == ("LFLE",)
    assert left_thigh.second_axis.fallback.end.marker_names == ("LFME",)

    right_shank = segments["RShank"].frame
    left_shank = segments["LShank"].frame
    assert right_shank.second_axis.fallback.name == Axis.Name.Z
    assert right_shank.second_axis.fallback.start.marker_names == ("RFME",)
    assert right_shank.second_axis.fallback.end.marker_names == ("RFLE",)
    assert right_shank.axis_to_keep == Axis.Name.Z
    assert left_shank.second_axis.fallback.name == Axis.Name.Z
    assert left_shank.second_axis.fallback.start.marker_names == ("LFLE",)
    assert left_shank.second_axis.fallback.end.marker_names == ("LFME",)
    assert left_shank.axis_to_keep == Axis.Name.Z

    upper_arm = segments["RUpperArm"].frame
    assert upper_arm.origin.marker_names == ("RGJC",)
    assert upper_arm.first_axis.start.marker_names == ("RHLE", "RHME")
    assert upper_arm.first_axis.end.marker_names == ("RGJC",)
    assert upper_arm.second_axis.start.marker_names == ("RHME",)
    assert upper_arm.second_axis.end.marker_names == ("RHLE",)

    left_upper_arm = segments["LUpperArm"].frame
    assert left_upper_arm.origin.marker_names == ("LGJC",)
    assert left_upper_arm.first_axis.start.marker_names == ("LHLE", "LHME")
    assert left_upper_arm.first_axis.end.marker_names == ("LGJC",)
    assert left_upper_arm.second_axis.start.marker_names == ("LHLE",)
    assert left_upper_arm.second_axis.end.marker_names == ("LHME",)

    right_forearm = segments["RForearm"].frame
    assert right_forearm.second_axis.start.marker_names == ("RHME", "RUSP")
    assert right_forearm.second_axis.end.marker_names == ("RHLE", "RRSP")
    left_forearm = segments["LForearm"].frame
    assert left_forearm.second_axis.start.marker_names == ("LHLE", "LRSP")
    assert left_forearm.second_axis.end.marker_names == ("LHME", "LUSP")

    right_hand = segments["RHand"].frame
    assert right_hand.second_axis.start.marker_names == ("RUSP",)
    assert right_hand.second_axis.end.marker_names == ("RRSP",)
    left_hand = segments["LHand"].frame
    assert left_hand.second_axis.start.marker_names == ("LRSP",)
    assert left_hand.second_axis.end.marker_names == ("LUSP",)


def test_motive_57_template_declares_functional_trials_and_virtual_requirements():
    template = motive_57_template(use_functional=True)
    segments = {segment.name: segment for segment in template.segments}
    trials = {trial.name: trial for trial in motive_57_functional_trials()}
    required_markers = set(required_static_markers(template))

    assert set(MOTIVE_57_MARKER_NAMES) <= required_markers
    assert {"LGJC", "RGJC"} <= required_markers
    assert {trial.name for trial in template.functional_trials} == set(MOTIVE_57_FUNCTIONAL_C3D_FILENAMES)
    assert trials["right_hip_score"].file_pattern == "*Func_RHip.c3d"
    assert trials["right_hip_score"].alternate_file_patterns == ("*RHip.c3d",)
    assert isinstance(trials["right_hip_score"], FunctionalTrialSpec)
    assert trials["right_hip_score"].file_patterns == (
        "*Func_RHip.c3d",
        "*RHip.c3d",
    )
    assert trials["left_ankle_score"].required_markers == (
        "LFAX",
        "LSK",
        "LTTC",
        "LFAL",
        "LTAM",
        "LFCC",
        "LFM5",
        "LFM2",
        "LFM1",
    )
    assert isinstance(segments["RThigh"].frame.origin, FunctionalCenterSpec)
    assert segments["RThigh"].frame.origin.trial_name == "right_hip_score"
    assert isinstance(segments["RShank"].frame.origin, FunctionalAxisProjectionPointSpec)
    assert segments["RShank"].frame.origin.trial_name == "right_knee_sara"
    assert segments["RShank"].frame.origin.method.value == "sara_direction"
    assert isinstance(segments["RShank"].frame.second_axis, FunctionalAxisSpec)
    assert segments["RShank"].frame.second_axis.trial_name == "right_knee_sara"
    assert segments["RShank"].frame.second_axis.method.value == "sara_direction"
    assert segments["RShank"].frame.second_axis.expected_axis.start.marker_names == ("RFME",)
    assert segments["RShank"].frame.second_axis.expected_axis.end.marker_names == ("RFLE",)
