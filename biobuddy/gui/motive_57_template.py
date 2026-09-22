from __future__ import annotations

from ..components.generic.rigidbody.axis import Axis
from ..utils.enums import Rotations, Translations
from .model_builder import (
    AxisSpec,
    FunctionalAxisProjectionPointSpec,
    FunctionalAxisSpec,
    FunctionalCenterSpec,
    FunctionalMethod,
    FunctionalTrialSpec,
    LocalFrameSpec,
    MarkerAttachmentSpec,
    MarkerEndpointSpec,
    ModelTemplate,
    SegmentSpec,
)

MOTIVE_57_MARKER_NAMES = (
    "LIAS",
    "RIAS",
    "LIPS",
    "RIPS",
    "LFTC",
    "RFTC",
    "SJN",
    "TV2",
    "SXS",
    "TV7",
    "CV7",
    "LAH",
    "RAH",
    "LPH",
    "RPH",
    "LHGT",
    "LCAJ",
    "LHLE",
    "LHME",
    "LUA",
    "LHM2",
    "LUSP",
    "LRSP",
    "RHGT",
    "RCAJ",
    "RHLE",
    "RHME",
    "RUA",
    "RHM2",
    "RUSP",
    "RRSP",
    "LFLE",
    "LFME",
    "LTH",
    "LFAL",
    "LTAM",
    "LSK",
    "LTTC",
    "LFAX",
    "LFM5",
    "LFM2",
    "LFM1",
    "LFCC",
    "LDP1",
    "RFLE",
    "RFME",
    "RTH",
    "RFAL",
    "RTAM",
    "RSK",
    "RTTC",
    "RFAX",
    "RFM5",
    "RFM2",
    "RFM1",
    "RFCC",
    "RDP1",
)

MOTIVE_57_FUNCTIONAL_C3D_FILENAMES = {
    "left_hip_score": "*Func_LHip.c3d",
    "left_knee_sara": "*Func_LKnee.c3d",
    "left_ankle_score": "*Func_LAnkle.c3d",
    "right_hip_score": "*Func_RHip.c3d",
    "right_knee_sara": "*Func_RKnee.c3d",
    "right_ankle_score": "*Func_RAnkle.c3d",
}
MOTIVE_57_P6_FUNCTIONAL_C3D_FILENAMES = {
    "left_hip_score": "*LHip.c3d",
    "left_knee_sara": "*LKnee.c3d",
    "left_ankle_score": "*LAnkle.c3d",
    "right_hip_score": "*RHip.c3d",
    "right_knee_sara": "*RKnee.c3d",
    "right_ankle_score": "*RAnkle.c3d",
}
MOTIVE_57_KNEE_SARA_MAX_STATIC_AXIS_DEVIATION_DEGREES = 30.0


def motive_57_template(use_functional: bool = True) -> ModelTemplate:
    """
    Return the BioBuddy Motive (57) template for calibration C3D model creation.
    """
    markers = motive_57_marker_attachments()
    return ModelTemplate(
        name=(
            "BioBuddy Motive (57) from calibration C3D (SCoRE/SARA)"
            if use_functional
            else "BioBuddy Motive (57) from calibration C3D"
        ),
        segments=(
            _pelvis_segment(),
            _thorax_segment(),
            _head_segment(),
            _thigh_segment("R", use_functional),
            _shank_segment("R", use_functional),
            _foot_segment("R", use_functional),
            _thigh_segment("L", use_functional),
            _shank_segment("L", use_functional),
            _foot_segment("L", use_functional),
            _upper_arm_segment("R"),
            _forearm_segment("R"),
            _hand_segment("R"),
            _upper_arm_segment("L"),
            _forearm_segment("L"),
            _hand_segment("L"),
        ),
        marker_attachments=markers,
        required_static_markers=tuple(sorted(set(MOTIVE_57_MARKER_NAMES) | {"LGJC", "RGJC"})),
        functional_trials=motive_57_functional_trials() if use_functional else (),
        root_segment_name="Pelvis",
    )


def motive_57_functional_trials() -> tuple[FunctionalTrialSpec, ...]:
    """
    Return the functional C3D trials expected by the Motive (57) template.
    """
    trials = []
    for side, label in (("L", "left"), ("R", "right")):
        trials.extend(
            (
                FunctionalTrialSpec(
                    name=f"{label}_hip_score",
                    file_pattern=MOTIVE_57_FUNCTIONAL_C3D_FILENAMES[f"{label}_hip_score"],
                    alternate_file_patterns=(MOTIVE_57_P6_FUNCTIONAL_C3D_FILENAMES[f"{label}_hip_score"],),
                    required_markers=(
                        "LIAS",
                        "RIAS",
                        "LIPS",
                        "RIPS",
                        f"{side}FTC",
                        f"{side}TH",
                        f"{side}FLE",
                        f"{side}FME",
                    ),
                    method=FunctionalMethod.SCORE,
                ),
                FunctionalTrialSpec(
                    name=f"{label}_knee_sara",
                    file_pattern=MOTIVE_57_FUNCTIONAL_C3D_FILENAMES[f"{label}_knee_sara"],
                    alternate_file_patterns=(MOTIVE_57_P6_FUNCTIONAL_C3D_FILENAMES[f"{label}_knee_sara"],),
                    required_markers=(
                        f"{side}FTC",
                        f"{side}TH",
                        f"{side}FLE",
                        f"{side}FME",
                        f"{side}FAX",
                        f"{side}SK",
                        f"{side}TTC",
                        f"{side}FAL",
                        f"{side}TAM",
                    ),
                    method=FunctionalMethod.SARA_DIRECTION,
                ),
                FunctionalTrialSpec(
                    name=f"{label}_ankle_score",
                    file_pattern=MOTIVE_57_FUNCTIONAL_C3D_FILENAMES[f"{label}_ankle_score"],
                    alternate_file_patterns=(MOTIVE_57_P6_FUNCTIONAL_C3D_FILENAMES[f"{label}_ankle_score"],),
                    required_markers=(
                        f"{side}FAX",
                        f"{side}SK",
                        f"{side}TTC",
                        f"{side}FAL",
                        f"{side}TAM",
                        f"{side}FCC",
                        f"{side}FM5",
                        f"{side}FM2",
                        f"{side}FM1",
                    ),
                    method=FunctionalMethod.SCORE,
                ),
            )
        )
    return tuple(trials)


def motive_57_marker_attachments() -> tuple[MarkerAttachmentSpec, ...]:
    """
    Return reconstruction marker attachments for the Motive (57) template.
    """
    attachments = [
        _marker("LIAS", "Pelvis", anatomical=True),
        _marker("RIAS", "Pelvis", anatomical=True),
        _marker("LIPS", "Pelvis", anatomical=True),
        _marker("RIPS", "Pelvis", anatomical=True),
        _marker("SJN", "Thorax", anatomical=True),
        _marker("SXS", "Thorax", anatomical=True),
        _marker("TV2", "Thorax", anatomical=True),
        _marker("TV7", "Thorax", anatomical=True),
        _marker("CV7", "Thorax", anatomical=True),
        _marker("LAH", "Head", anatomical=True),
        _marker("RAH", "Head", anatomical=True),
        _marker("LPH", "Head", anatomical=True),
        _marker("RPH", "Head", anatomical=True),
    ]
    for side in ("R", "L"):
        attachments.extend(
            (
                _marker(f"{side}FTC", f"{side}Thigh"),
                _marker(f"{side}TH", f"{side}Thigh"),
                _marker(
                    f"{side}FLE",
                    f"{side}Thigh",
                    anatomical=True,
                ),
                _marker(
                    f"{side}FME",
                    f"{side}Thigh",
                    anatomical=True,
                ),
                _marker(f"{side}FAX", f"{side}Shank"),
                _marker(f"{side}SK", f"{side}Shank"),
                _marker(f"{side}TTC", f"{side}Shank"),
                _marker(f"{side}FAL", f"{side}Shank", anatomical=True),
                _marker(f"{side}TAM", f"{side}Shank", anatomical=True),
                _marker(f"{side}FCC", f"{side}Foot", anatomical=True),
                _marker(f"{side}FM5", f"{side}Foot", anatomical=True),
                _marker(f"{side}FM2", f"{side}Foot", anatomical=True),
                _marker(f"{side}FM1", f"{side}Foot", anatomical=True),
                _marker(f"{side}UA", f"{side}UpperArm"),
                _marker(f"{side}HLE", f"{side}UpperArm", technical=False, anatomical=True),
                _marker(f"{side}HME", f"{side}UpperArm", technical=False, anatomical=True),
                _marker(f"{side}USP", f"{side}Forearm", technical=False, anatomical=True),
                _marker(f"{side}RSP", f"{side}Forearm", technical=False, anatomical=True),
                _marker(f"{side}HM2", f"{side}Hand", anatomical=True),
                _marker(f"{side}USP", f"{side}Hand"),
                _marker(f"{side}RSP", f"{side}Hand"),
            )
        )
    return tuple(attachments)


def _pelvis_segment() -> SegmentSpec:
    return SegmentSpec(
        name="Pelvis",
        parent_name="root",
        translations=Translations.XYZ,
        rotations=Rotations.ZXY,
        frame=LocalFrameSpec(
            origin=_p("LIAS", "RIAS", "LIPS", "RIPS"),
            first_axis=AxisSpec.from_markers(Axis.Name.Z, ("LIAS", "LIPS"), ("RIAS", "RIPS")),
            second_axis=AxisSpec.from_markers(Axis.Name.X, ("LIPS", "RIPS"), ("LIAS", "RIAS")),
            axis_to_keep=Axis.Name.Z,
        ),
        mesh_points=(_p("LIPS"), _p("RIPS"), _p("RIAS"), _p("LIAS"), _p("LIPS")),
    )


def _thorax_segment() -> SegmentSpec:
    return SegmentSpec(
        name="Thorax",
        parent_name="Pelvis",
        rotations=Rotations.ZXY,
        frame=LocalFrameSpec(
            origin=_p("SJN", "SXS", "TV2", "TV7"),
            first_axis=AxisSpec.from_markers(Axis.Name.Y, ("SXS", "TV7"), ("SJN", "TV2")),
            second_axis=AxisSpec.from_markers(Axis.Name.X, ("TV7", "TV2"), ("SXS", "SJN")),
            axis_to_keep=Axis.Name.Y,
        ),
        mesh_points=(_p("SXS"), _p("SJN"), _p("TV2"), _p("TV7"), _p("SXS")),
    )


def _head_segment() -> SegmentSpec:
    return SegmentSpec(
        name="Head",
        parent_name="Thorax",
        rotations=Rotations.ZXY,
        frame=LocalFrameSpec(
            origin=_p("LAH", "RAH", "LPH", "RPH"),
            first_axis=AxisSpec.from_markers(Axis.Name.Z, ("LAH", "LPH"), ("RAH", "RPH")),
            second_axis=AxisSpec.from_markers(Axis.Name.X, ("LPH", "RPH"), ("LAH", "RAH")),
            axis_to_keep=Axis.Name.Z,
        ),
        mesh_points=(_p("LPH"), _p("RPH"), _p("RAH"), _p("LAH"), _p("LPH")),
    )


def _thigh_segment(side: str, use_functional: bool) -> SegmentSpec:
    hjc = _hip_center_spec(side, use_functional)
    kjc = _knee_projection_spec(side, use_functional)
    kjc_fallback = kjc.fallback if isinstance(kjc, FunctionalAxisProjectionPointSpec) else kjc
    return SegmentSpec(
        name=f"{side}Thigh",
        parent_name="Pelvis",
        rotations=Rotations.ZXY,
        frame=LocalFrameSpec(
            origin=hjc,
            first_axis=AxisSpec(Axis.Name.Y, kjc, hjc),
            second_axis=_knee_axis_spec(side, use_functional),
            axis_to_keep=Axis.Name.Y,
        ),
        mesh_points=(
            hjc.fallback if isinstance(hjc, FunctionalCenterSpec) else hjc,
            _p(f"{side}FTC"),
            _p(f"{side}TH"),
            _p(f"{side}FLE"),
            kjc_fallback,
            _p(f"{side}FME"),
        ),
    )


def _shank_segment(side: str, use_functional: bool) -> SegmentSpec:
    kjc = _knee_projection_spec(side, use_functional)
    ajc = _ankle_center_spec(side, use_functional)
    return SegmentSpec(
        name=f"{side}Shank",
        parent_name=f"{side}Thigh",
        rotations=Rotations.Z,
        frame=LocalFrameSpec(
            origin=kjc,
            first_axis=AxisSpec(Axis.Name.Y, ajc, kjc),
            second_axis=_knee_axis_spec(side, use_functional),
            axis_to_keep=Axis.Name.Z,
        ),
        mesh_points=(
            _p(f"{side}FAX"),
            _p(f"{side}SK"),
            _p(f"{side}TTC"),
            _p(f"{side}FAL"),
            ajc.fallback if isinstance(ajc, FunctionalCenterSpec) else ajc,
            _p(f"{side}TAM"),
        ),
    )


def _foot_segment(side: str, use_functional: bool) -> SegmentSpec:
    ajc = _ankle_center_spec(side, use_functional)
    z_start_marker = f"{side}FM1" if side == "R" else f"{side}FM5"
    z_end_marker = f"{side}FM5" if side == "R" else f"{side}FM1"
    return SegmentSpec(
        name=f"{side}Foot",
        parent_name=f"{side}Shank",
        rotations=Rotations.ZX,
        frame=LocalFrameSpec(
            origin=ajc,
            first_axis=AxisSpec.from_markers(
                Axis.Name.X,
                (f"{side}FCC", f"{side}FCC"),
                (f"{side}FM5", f"{side}FM1"),
            ),
            second_axis=AxisSpec.from_markers(Axis.Name.Z, z_start_marker, z_end_marker),
            axis_to_keep=Axis.Name.X,
        ),
        mesh_points=(
            _p(f"{side}FCC"),
            _p(f"{side}FM5"),
            _p(f"{side}FM2"),
            _p(f"{side}FM1"),
            _p(f"{side}FCC"),
        ),
    )


def _upper_arm_segment(side: str) -> SegmentSpec:
    gjc = _p(f"{side}GJC")
    ejc = _p(f"{side}HLE", f"{side}HME")
    return SegmentSpec(
        name=f"{side}UpperArm",
        parent_name="Thorax",
        rotations=Rotations.ZXY,
        frame=LocalFrameSpec(
            origin=gjc,
            first_axis=AxisSpec(Axis.Name.Y, ejc, gjc),
            second_axis=_humerus_axis(side),
            axis_to_keep=Axis.Name.Y,
        ),
        mesh_points=(
            gjc,
            _p(f"{side}UA"),
            _p(f"{side}HLE"),
            ejc,
            _p(f"{side}HME"),
            gjc,
        ),
    )


def _forearm_segment(side: str) -> SegmentSpec:
    ejc = _p(f"{side}HLE", f"{side}HME")
    wjc = _p(f"{side}USP", f"{side}RSP")
    return SegmentSpec(
        name=f"{side}Forearm",
        parent_name=f"{side}UpperArm",
        rotations=Rotations.ZY,
        frame=LocalFrameSpec(
            origin=ejc,
            first_axis=AxisSpec(Axis.Name.Y, wjc, ejc),
            second_axis=_forearm_axis(side),
            axis_to_keep=Axis.Name.Y,
        ),
        mesh_points=(ejc, _p(f"{side}USP"), wjc, _p(f"{side}RSP"), ejc),
    )


def _hand_segment(side: str) -> SegmentSpec:
    wjc = _p(f"{side}USP", f"{side}RSP")
    return SegmentSpec(
        name=f"{side}Hand",
        parent_name=f"{side}Forearm",
        rotations=Rotations.ZX,
        frame=LocalFrameSpec(
            origin=wjc,
            first_axis=AxisSpec.from_markers(Axis.Name.Y, f"{side}HM2", (f"{side}USP", f"{side}RSP")),
            second_axis=_hand_axis(side),
            axis_to_keep=Axis.Name.Y,
        ),
        mesh_points=(wjc, _p(f"{side}HM2")),
    )


def _hip_center_spec(side: str, use_functional: bool):
    fallback = _p(f"{side}IAS", f"{side}IPS", f"{side}FTC")
    if not use_functional:
        return fallback
    return FunctionalCenterSpec(
        method=FunctionalMethod.SCORE,
        trial_name="left_hip_score" if side == "L" else "right_hip_score",
        parent_marker_names=("LIAS", "RIAS", "LIPS", "RIPS"),
        child_marker_names=(f"{side}FTC", f"{side}TH", f"{side}FLE", f"{side}FME"),
        fallback=fallback,
    )


def _ankle_center_spec(side: str, use_functional: bool):
    fallback = _p(f"{side}FAL", f"{side}TAM")
    if not use_functional:
        return fallback
    return FunctionalCenterSpec(
        method=FunctionalMethod.SCORE,
        trial_name="left_ankle_score" if side == "L" else "right_ankle_score",
        parent_marker_names=(
            f"{side}FAX",
            f"{side}SK",
            f"{side}TTC",
            f"{side}FAL",
            f"{side}TAM",
        ),
        child_marker_names=(f"{side}FCC", f"{side}FM5", f"{side}FM2", f"{side}FM1"),
        fallback=fallback,
    )


def _knee_axis_spec(side: str, use_functional: bool):
    fallback = _humerus_or_knee_axis(side, "FME", "FLE")
    if not use_functional:
        return fallback
    return FunctionalAxisSpec(
        method=FunctionalMethod.SARA_DIRECTION,
        trial_name="left_knee_sara" if side == "L" else "right_knee_sara",
        fallback=fallback,
        parent_marker_names=(f"{side}FTC", f"{side}TH", f"{side}FLE", f"{side}FME"),
        child_marker_names=(
            f"{side}FAX",
            f"{side}SK",
            f"{side}TTC",
            f"{side}FAL",
            f"{side}TAM",
        ),
        expected_axis=fallback,
        origin_marker_names=(f"{side}FLE", f"{side}FME"),
        max_static_axis_deviation_degrees=MOTIVE_57_KNEE_SARA_MAX_STATIC_AXIS_DEVIATION_DEGREES,
    )


def _knee_projection_spec(side: str, use_functional: bool):
    fallback = _p(f"{side}FLE", f"{side}FME")
    if not use_functional:
        return fallback
    sara_axis = _knee_axis_spec(side, use_functional=True)
    return FunctionalAxisProjectionPointSpec(
        method=FunctionalMethod.SARA_DIRECTION,
        trial_name=sara_axis.trial_name,
        parent_marker_names=sara_axis.parent_marker_names,
        child_marker_names=sara_axis.child_marker_names,
        expected_axis=sara_axis.expected_axis,
        origin_marker_names=sara_axis.origin_marker_names,
        point_marker_names=sara_axis.origin_marker_names,
        fallback=fallback,
        max_static_axis_deviation_degrees=sara_axis.max_static_axis_deviation_degrees,
    )


def _humerus_axis(side: str) -> AxisSpec:
    return _humerus_or_knee_axis(side, "HME", "HLE")


def _humerus_or_knee_axis(side: str, medial_suffix: str, lateral_suffix: str) -> AxisSpec:
    if side == "R":
        return AxisSpec.from_markers(Axis.Name.Z, f"{side}{medial_suffix}", f"{side}{lateral_suffix}")
    return AxisSpec.from_markers(Axis.Name.Z, f"{side}{lateral_suffix}", f"{side}{medial_suffix}")


def _forearm_axis(side: str) -> AxisSpec:
    if side == "R":
        return AxisSpec.from_markers(Axis.Name.Z, ("RHME", "RUSP"), ("RHLE", "RRSP"))
    return AxisSpec.from_markers(Axis.Name.Z, ("LHLE", "LRSP"), ("LHME", "LUSP"))


def _hand_axis(side: str) -> AxisSpec:
    if side == "R":
        return AxisSpec.from_markers(Axis.Name.Z, "RUSP", "RRSP")
    return AxisSpec.from_markers(Axis.Name.Z, "LRSP", "LUSP")


def _marker(name: str, *segment_names: str, technical: bool = True, anatomical: bool = False) -> MarkerAttachmentSpec:
    return MarkerAttachmentSpec(
        name=name,
        segment_names=tuple(segment_names),
        is_technical=technical,
        is_anatomical=anatomical,
    )


def _p(*marker_names: str) -> MarkerEndpointSpec:
    return MarkerEndpointSpec(tuple(marker_names))
