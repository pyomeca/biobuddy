import numpy as np

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
from ..characteristics import DeLevaTable, SegmentName, Sex
from ..components.generic.rigidbody.axis import Axis
from ..utils.enums import Rotations, Translations

LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES = {
    "left_hip_score": "*func_lhip.c3d",
    "left_knee_sara": "*func_lknee.c3d",
    "left_ankle_score": "*func_lankle.c3d",
    "right_hip_score": "*func_rhip.c3d",
    "right_knee_sara": "*func_rknee.c3d",
    "right_ankle_score": "*func_rankle.c3d",
}


def lower_limb_template(use_functional: bool = True, include_de_leva: bool = True) -> ModelTemplate:
    """
    Return the lower-body model template inspired by the walking reconstruction example.
    """
    markers = _markers()
    functional_trials = _functional_trials() if use_functional else ()
    return ModelTemplate(
        name=(
            "Lower body from calibration C3D (SCoRE/SARA)"
            if use_functional
            else "Lower body from calibration C3D (anatomical markers only)"
        ),
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
                inertia_name=SegmentName.TRUNK,
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
            _thigh_segment(
                side="L", hip_fallback="LASI", knee_axis_start=("LKNE", "LKNEM"), use_functional=use_functional
            ),
            _shank_segment(
                side="L",
                knee_axis_start=("LKNE", "LKNEM"),
                ankle_axis_start=("LANK", "LANKM"),
                use_functional=use_functional,
            ),
            _foot_segment(
                side="L",
                ankle_origin=("LANK", "LANKM"),
                ankle_axis=("LANK", "LANKM"),
                use_functional=use_functional,
            ),
            _thigh_segment(
                side="R", hip_fallback="RASI", knee_axis_start=("RKNE", "RKNEM"), use_functional=use_functional
            ),
            _shank_segment(
                side="R",
                knee_axis_start=("RKNEM", "RKNE"),
                ankle_axis_start=("RANK", "RANKM"),
                use_functional=use_functional,
            ),
            _foot_segment(
                side="R",
                ankle_origin=("RANK", "RANKM"),
                ankle_axis=("RANKM", "RANK"),
                use_functional=use_functional,
            ),
        ),
        marker_attachments=markers,
        required_static_markers=tuple(sorted({attachment.name for attachment in markers})),
        functional_trials=functional_trials,
        root_segment_name="Pelvis",
        inertia_parameters_factory=lower_limb_de_leva_inertia_parameters if include_de_leva else None,
    )


def _functional_trials() -> tuple[FunctionalTrialSpec, ...]:
    return (
        FunctionalTrialSpec(
            name="left_hip_score",
            file_pattern=LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES["left_hip_score"],
            required_markers=("LPSI", "RPSI", "LASI", "RASI", "LTHI", "LTHIB", "LTHID"),
            method=FunctionalMethod.SCORE,
        ),
        FunctionalTrialSpec(
            name="left_knee_sara",
            file_pattern=LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES["left_knee_sara"],
            required_markers=("LTHI", "LTHIB", "LTHID", "LTIB", "LTIBF", "LTIBD", "LKNE", "LKNEM"),
            method=FunctionalMethod.SARA,
        ),
        FunctionalTrialSpec(
            name="left_ankle_score",
            file_pattern=LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES["left_ankle_score"],
            required_markers=("LTIB", "LTIBF", "LTIBD", "LHEE", "LNAV", "LTOE", "LTOE5"),
            method=FunctionalMethod.SCORE,
        ),
        FunctionalTrialSpec(
            name="right_hip_score",
            file_pattern=LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES["right_hip_score"],
            required_markers=("LPSI", "RPSI", "LASI", "RASI", "RTHI", "RTHIB", "RTHID"),
            method=FunctionalMethod.SCORE,
        ),
        FunctionalTrialSpec(
            name="right_knee_sara",
            file_pattern=LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES["right_knee_sara"],
            required_markers=("RTHI", "RTHIB", "RTHID", "RTIB", "RTIBF", "RTIBD", "RKNE", "RKNEM"),
            method=FunctionalMethod.SARA,
        ),
        FunctionalTrialSpec(
            name="right_ankle_score",
            file_pattern=LOWER_LIMB_FUNCTIONAL_C3D_FILENAMES["right_ankle_score"],
            required_markers=("RTIB", "RTIBF", "RTIBD", "RHEE", "RNAV", "RTOE", "RTOE5"),
            method=FunctionalMethod.SCORE,
        ),
    )


def _thigh_segment(side: str, hip_fallback: str, knee_axis_start: tuple[str, str], use_functional: bool) -> SegmentSpec:
    thigh = f"{side}Thigh"
    origin = (
        FunctionalCenterSpec(
            method=FunctionalMethod.SCORE,
            trial_name="left_hip_score" if side == "L" else "right_hip_score",
            parent_marker_names=("LPSI", "RPSI", "LASI", "RASI"),
            child_marker_names=(f"{side}THI", f"{side}THIB", f"{side}THID"),
            fallback=_p(hip_fallback),
        )
        if use_functional
        else _p(hip_fallback)
    )
    return SegmentSpec(
        name=thigh,
        parent_name="Pelvis",
        rotations=Rotations.XZY,
        inertia_name=SegmentName.THIGH,
        frame=LocalFrameSpec(
            origin=origin,
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


def _shank_segment(
    side: str,
    knee_axis_start: tuple[str, str],
    ankle_axis_start: tuple[str, str],
    use_functional: bool,
) -> SegmentSpec:
    sara_trial = "left_knee_sara" if side == "L" else "right_knee_sara"
    fallback_axis = AxisSpec.from_markers(Axis.Name.X, knee_axis_start[0], knee_axis_start[1])
    second_axis = (
        FunctionalAxisSpec(
            method=FunctionalMethod.SARA,
            trial_name=sara_trial,
            fallback=fallback_axis,
            parent_marker_names=(f"{side}TIBD", f"{side}TIB", f"{side}TIBF"),
            child_marker_names=(f"{side}THIB", f"{side}THID", f"{side}THI"),
            expected_axis=fallback_axis,
            origin_marker_names=knee_axis_start,
        )
        if use_functional
        else fallback_axis
    )
    return SegmentSpec(
        name=f"{side}Shank",
        parent_name=f"{side}Thigh",
        rotations=Rotations.X,
        inertia_name=SegmentName.SHANK,
        frame=LocalFrameSpec(
            origin=_p(*knee_axis_start),
            first_axis=AxisSpec.from_markers(Axis.Name.Z, ankle_axis_start, knee_axis_start),
            second_axis=second_axis,
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


def _foot_segment(
    side: str,
    ankle_origin: tuple[str, str],
    ankle_axis: tuple[str, str],
    use_functional: bool,
) -> SegmentSpec:
    origin = (
        FunctionalCenterSpec(
            method=FunctionalMethod.SCORE,
            trial_name="left_ankle_score" if side == "L" else "right_ankle_score",
            parent_marker_names=(f"{side}TIB", f"{side}TIBF", f"{side}TIBD"),
            child_marker_names=(f"{side}HEE", f"{side}NAV", f"{side}TOE", f"{side}TOE5"),
            fallback=_p(*ankle_origin),
        )
        if use_functional
        else _p(*ankle_origin)
    )
    return SegmentSpec(
        name=f"{side}Foot",
        parent_name=f"{side}Shank",
        rotations=Rotations.XZ,
        inertia_name=SegmentName.FOOT,
        frame=LocalFrameSpec(
            origin=origin,
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


def lower_limb_de_leva_inertia_parameters(data) -> dict[str, object]:
    """
    Build De Leva inertial parameters from marker-origin distances.

    The calibration C3Ds used by this project can have their Z axis pointing downward, so the measurements below are
    Euclidean distances between anatomical marker groups rather than signed vertical coordinates.
    """
    pelvis = _mean_position(data, ("LPSI", "RPSI", "LASI", "RASI"))
    trunk_top = _mean_position(data, ("CLAV", "C7"))
    left_knee = _mean_position(data, ("LKNE", "LKNEM"))
    right_knee = _mean_position(data, ("RKNE", "RKNEM"))
    left_ankle = _mean_position(data, ("LANK", "LANKM"))
    right_ankle = _mean_position(data, ("RANK", "RANKM"))

    thigh_length = np.nanmean(
        (
            _distance(_mean_position(data, ("LASI",)), left_knee),
            _distance(_mean_position(data, ("RASI",)), right_knee),
        )
    )
    shank_length = np.nanmean((_distance(left_knee, left_ankle), _distance(right_knee, right_ankle)))
    trunk_length = _distance(pelvis, trunk_top)
    foot_length = np.nanmean(
        (
            _distance(_mean_position(data, ("LHEE",)), _mean_position(data, ("LTOE",))),
            _distance(_mean_position(data, ("RHEE",)), _mean_position(data, ("RTOE",))),
        )
    )
    hip_width = _distance(_mean_position(data, ("LASI",)), _mean_position(data, ("RASI",)))

    ankle_height = 0.0
    knee_height = float(shank_length)
    hip_height = float(shank_length + thigh_length)
    shoulder_height = float(hip_height + trunk_length)
    total_height = shoulder_height
    shoulder_span = max(float(hip_width), 1e-6)

    de_leva = DeLevaTable(total_mass=100, sex=Sex.MALE)
    de_leva.from_measurements(
        total_height=total_height,
        ankle_height=ankle_height,
        knee_height=knee_height,
        hip_height=hip_height,
        shoulder_height=shoulder_height,
        finger_span=shoulder_span,
        wrist_span=shoulder_span,
        elbow_span=shoulder_span,
        shoulder_span=shoulder_span,
        hip_width=float(hip_width),
        foot_length=float(foot_length),
    )
    return {
        "Trunk": de_leva[SegmentName.TRUNK],
        "LThigh": de_leva[SegmentName.THIGH],
        "LShank": de_leva[SegmentName.SHANK],
        "LFoot": de_leva[SegmentName.FOOT],
        "RThigh": de_leva[SegmentName.THIGH],
        "RShank": de_leva[SegmentName.SHANK],
        "RFoot": de_leva[SegmentName.FOOT],
    }


def _mean_position(data, marker_names: tuple[str, ...]) -> np.ndarray:
    positions = data.get_position(list(marker_names))[:3, :, :]
    return np.nanmean(positions, axis=(1, 2))


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first[:3] - second[:3]))


def _markers() -> tuple[MarkerAttachmentSpec, ...]:
    return (
        _marker("LPSI", "Pelvis", anatomical=True),
        _marker("RPSI", "Pelvis", anatomical=True),
        _marker("LASI", "Pelvis", anatomical=True),
        _marker("RASI", "Pelvis", anatomical=True),
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
