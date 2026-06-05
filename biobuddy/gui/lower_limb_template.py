from .model_builder import (
    AxisSpec,
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
from ..components.generic.rigidbody.axis import Axis
from ..utils.enums import Rotations, Translations


def lower_limb_template() -> ModelTemplate:
    """
    Return the lower-body model template inspired by the walking reconstruction example.
    """
    markers = _markers()
    return ModelTemplate(
        name="Lower body from calibration C3D",
        segments=(
            SegmentSpec(
                name="Pelvis",
                parent_name="root",
                translations=Translations.XYZ,
                rotations=Rotations.XYZ,
                frame=LocalFrameSpec(
                    origin=_p("LPSI", "RPSI", "LASI", "RASI"),
                    first_axis=AxisSpec.from_markers(
                        Axis.Name.X,
                        ("LPSI", "LASI"),
                        ("RPSI", "RASI"),
                    ),
                    second_axis=AxisSpec.from_markers(
                        Axis.Name.Y,
                        ("LPSI", "RPSI"),
                        ("LASI", "RASI"),
                    ),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh_points=(_p("LPSI"), _p("RPSI"), _p("RASI"), _p("LASI"), _p("LPSI")),
            ),
            SegmentSpec(
                name="Trunk",
                parent_name="Pelvis",
                translations=Translations.XYZ,
                rotations=Rotations.XYZ,
                frame=LocalFrameSpec(
                    origin=_p("CLAV"),
                    first_axis=AxisSpec.from_markers(
                        Axis.Name.Y,
                        ("T10", "C7"),
                        ("STRN", "CLAV"),
                    ),
                    second_axis=AxisSpec.from_markers(
                        Axis.Name.Z,
                        ("T10", "STRN"),
                        ("C7", "CLAV"),
                    ),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh_points=(
                    _p("S3"),
                    _p("S1"),
                    _p("T10"),
                    _p("T6"),
                    _p("C7"),
                    _p("C2"),
                    _p("C7"),
                    _p("CLAV"),
                    _p("STRN"),
                    _p("T10"),
                ),
            ),
            _thigh_segment(side="L", hip_fallback="LASI", knee_axis_start=("LKNE", "LKNEM")),
            _shank_segment(side="L", knee_axis_start=("LKNE", "LKNEM"), ankle_axis_start=("LANK", "LANKM")),
            _foot_segment(side="L", ankle_origin=("LANK", "LANKM"), ankle_axis=("LANK", "LANKM")),
            _thigh_segment(side="R", hip_fallback="RASI", knee_axis_start=("RKNE", "RKNEM")),
            _shank_segment(side="R", knee_axis_start=("RKNEM", "RKNE"), ankle_axis_start=("RANK", "RANKM")),
            _foot_segment(side="R", ankle_origin=("RANK", "RANKM"), ankle_axis=("RANKM", "RANK")),
        ),
        marker_attachments=markers,
        required_static_markers=tuple(sorted({attachment.name for attachment in markers})),
        functional_trials=(
            FunctionalTrialSpec(
                name="left_hip_score",
                file_pattern="*func_lhip.c3d",
                required_markers=("LPSI", "RPSI", "LASI", "RASI", "LTHI", "LTHIB", "LTHID"),
                method=FunctionalMethod.SCORE,
            ),
            FunctionalTrialSpec(
                name="left_knee_sara",
                file_pattern="*func_lknee.c3d",
                required_markers=("LTHI", "LTHIB", "LTHID", "LTIB", "LTIBF", "LTIBD", "LKNE", "LKNEM"),
                method=FunctionalMethod.SARA,
            ),
            FunctionalTrialSpec(
                name="left_ankle_score",
                file_pattern="*func_lankle.c3d",
                required_markers=("LTIB", "LTIBF", "LTIBD", "LHEE", "LNAV", "LTOE", "LTOE5"),
                method=FunctionalMethod.SCORE,
            ),
            FunctionalTrialSpec(
                name="right_hip_score",
                file_pattern="*func_rhip.c3d",
                required_markers=("LPSI", "RPSI", "LASI", "RASI", "RTHI", "RTHIB", "RTHID"),
                method=FunctionalMethod.SCORE,
            ),
            FunctionalTrialSpec(
                name="right_knee_sara",
                file_pattern="*func_rknee.c3d",
                required_markers=("RTHI", "RTHIB", "RTHID", "RTIB", "RTIBF", "RTIBD", "RKNE", "RKNEM"),
                method=FunctionalMethod.SARA,
            ),
            FunctionalTrialSpec(
                name="right_ankle_score",
                file_pattern="*func_rankle.c3d",
                required_markers=("RTIB", "RTIBF", "RTIBD", "RHEE", "RNAV", "RTOE", "RTOE5"),
                method=FunctionalMethod.SCORE,
            ),
        ),
        root_segment_name="Pelvis",
    )


def _thigh_segment(side: str, hip_fallback: str, knee_axis_start: tuple[str, str]) -> SegmentSpec:
    thigh = f"{side}Thigh"
    return SegmentSpec(
        name=thigh,
        parent_name="Pelvis",
        rotations=Rotations.XZY,
        frame=LocalFrameSpec(
            origin=FunctionalCenterSpec(
                method=FunctionalMethod.SCORE,
                trial_name="left_hip_score" if side == "L" else "right_hip_score",
                parent_marker_names=("LPSI", "RPSI", "LASI", "RASI"),
                child_marker_names=(f"{side}THI", f"{side}THIB", f"{side}THID"),
                fallback=_p(hip_fallback),
            ),
            first_axis=AxisSpec.from_markers(Axis.Name.Z, knee_axis_start, hip_fallback),
            second_axis=AxisSpec.from_markers(Axis.Name.X, "LASI", "RASI"),
            axis_to_keep=Axis.Name.Z,
        ),
        mesh_points=(
            _p(hip_fallback),
            _p(f"{side}THI"),
            _p(f"{side}THIB"),
            _p(hip_fallback),
            _p(f"{side}THIB"),
            _p(f"{side}THID"),
            _p(f"{side}THI"),
            _p(f"{side}THID"),
            _p(*knee_axis_start),
            _p(knee_axis_start[0]),
            _p(knee_axis_start[1]),
        ),
    )


def _shank_segment(side: str, knee_axis_start: tuple[str, str], ankle_axis_start: tuple[str, str]) -> SegmentSpec:
    sara_trial = "left_knee_sara" if side == "L" else "right_knee_sara"
    fallback_axis = AxisSpec.from_markers(Axis.Name.X, knee_axis_start[0], knee_axis_start[1])
    return SegmentSpec(
        name=f"{side}Shank",
        parent_name=f"{side}Thigh",
        rotations=Rotations.X,
        frame=LocalFrameSpec(
            origin=_p(*knee_axis_start),
            first_axis=AxisSpec.from_markers(Axis.Name.Z, ankle_axis_start, knee_axis_start),
            second_axis=FunctionalAxisSpec(
                method=FunctionalMethod.SARA,
                trial_name=sara_trial,
                fallback=fallback_axis,
                parent_marker_names=(f"{side}TIBD", f"{side}TIB", f"{side}TIBF"),
                child_marker_names=(f"{side}THIB", f"{side}THID", f"{side}THI"),
                expected_axis=fallback_axis,
                origin_marker_names=knee_axis_start,
            ),
            axis_to_keep=Axis.Name.X,
        ),
        mesh_points=(
            _p(f"{side}TIBD"),
            _p(f"{side}TIB"),
            _p(f"{side}TIBF"),
            _p(f"{side}TIBD"),
            _p(*ankle_axis_start),
            _p(ankle_axis_start[0]),
            _p(ankle_axis_start[1]),
        ),
    )


def _foot_segment(side: str, ankle_origin: tuple[str, str], ankle_axis: tuple[str, str]) -> SegmentSpec:
    return SegmentSpec(
        name=f"{side}Foot",
        parent_name=f"{side}Shank",
        rotations=Rotations.XZ,
        frame=LocalFrameSpec(
            origin=FunctionalCenterSpec(
                method=FunctionalMethod.SCORE,
                trial_name="left_ankle_score" if side == "L" else "right_ankle_score",
                parent_marker_names=(f"{side}TIB", f"{side}TIBF", f"{side}TIBD"),
                child_marker_names=(f"{side}HEE", f"{side}NAV", f"{side}TOE", f"{side}TOE5"),
                fallback=_p(*ankle_origin),
            ),
            first_axis=AxisSpec.from_markers(Axis.Name.Z, f"{side}TOE", f"{side}HEE"),
            second_axis=AxisSpec.from_markers(Axis.Name.X, ankle_axis[0], ankle_axis[1]),
            axis_to_keep=Axis.Name.Z,
        ),
        mesh_points=(
            _p(f"{side}HEE"),
            _p(f"{side}NAV"),
            _p(f"{side}TOE"),
            _p(f"{side}HEE"),
            _p(f"{side}TOE"),
            _p(f"{side}TOE5"),
            _p(f"{side}HEE"),
            _p(f"{side}TOE5"),
            _p(f"{side}NAV"),
        ),
    )


def _markers() -> tuple[MarkerAttachmentSpec, ...]:
    return (
        _marker("LPSI", "Pelvis", anatomical=True),
        _marker("RPSI", "Pelvis", anatomical=True),
        _marker("LASI", "Pelvis", "LThigh", "RThigh", anatomical=True),
        _marker("RASI", "Pelvis", "LThigh", "RThigh", anatomical=True),
        _marker("C7", "Trunk", anatomical=True),
        _marker("C2", "Trunk", anatomical=True),
        _marker("T6", "Trunk", anatomical=True),
        _marker("T10", "Trunk", anatomical=True),
        _marker("S1", "Trunk", anatomical=True),
        _marker("S3", "Trunk", anatomical=True),
        _marker("CLAV", "Trunk", anatomical=True),
        _marker("STRN", "Trunk", anatomical=True),
        _marker("LTHI", "LThigh"),
        _marker("LTHIB", "LThigh"),
        _marker("LTHID", "LThigh"),
        _marker("LKNE", "LThigh", "LShank", technical=False, anatomical=True),
        _marker("LKNEM", "LThigh", "LShank", technical=False, anatomical=True),
        _marker("LTIB", "LShank"),
        _marker("LTIBF", "LShank"),
        _marker("LTIBD", "LShank"),
        _marker("LANK", "LShank", "LFoot", technical=False, anatomical=True),
        _marker("LANKM", "LShank", "LFoot", technical=False, anatomical=True),
        _marker("LHEE", "LFoot", anatomical=True),
        _marker("LNAV", "LFoot", anatomical=True),
        _marker("LTOE", "LFoot", anatomical=True),
        _marker("LTOE5", "LFoot", anatomical=True),
        _marker("RTHI", "RThigh"),
        _marker("RTHIB", "RThigh"),
        _marker("RTHID", "RThigh"),
        _marker("RKNE", "RThigh", "RShank", technical=False, anatomical=True),
        _marker("RKNEM", "RThigh", "RShank", technical=False, anatomical=True),
        _marker("RTIB", "RShank"),
        _marker("RTIBF", "RShank"),
        _marker("RTIBD", "RShank"),
        _marker("RANK", "RShank", "RFoot", technical=False, anatomical=True),
        _marker("RANKM", "RShank", "RFoot", technical=False, anatomical=True),
        _marker("RHEE", "RFoot"),
        _marker("RNAV", "RFoot"),
        _marker("RTOE", "RFoot"),
        _marker("RTOE5", "RFoot"),
    )


def _marker(name: str, *segment_names: str, technical: bool = True, anatomical: bool = False) -> MarkerAttachmentSpec:
    return MarkerAttachmentSpec(
        name=name,
        segment_names=tuple(segment_names),
        is_technical=technical,
        is_anatomical=anatomical,
    )


def _p(*marker_names: str) -> MarkerEndpointSpec:
    return MarkerEndpointSpec(marker_names=tuple(marker_names))
