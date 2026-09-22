from __future__ import annotations

from ..components.generic.rigidbody.axis import Axis
from ..utils.enums import Rotations, Translations
from .model_builder import (
    AxisSpec,
    FunctionalAxisProjectionPointSpec,
    FunctionalAxisSpec,
    FunctionalCenterSpec,
    FunctionalMethod,
    LocalFrameSpec,
    MarkerEndpointSpec,
    ModelTemplate,
    SegmentSpec,
)
from .motive_57_template import (
    MOTIVE_57_KNEE_SARA_MAX_STATIC_AXIS_DEVIATION_DEGREES,
    MOTIVE_57_MARKER_NAMES,
    motive_57_functional_trials,
    motive_57_marker_attachments,
)


def motive_57_isb_template(use_functional: bool = True) -> ModelTemplate:
    """
    Return the ISB-oriented Motive (57) template.

    This ISB profile separates joint-coordinate systems from anatomical
    segment orientation. SCoRE/SARA may locate a lower-limb joint center, but
    the physical segment frames remain reconstructed from static landmarks.
    The upper limb follows the same joint/anatomical-frame separation.
    """
    return ModelTemplate(
        name=(
            "BioBuddy Motive (57) ISB from calibration C3D (SCoRE/SARA)"
            if use_functional
            else "BioBuddy Motive (57) ISB from calibration C3D"
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
        marker_attachments=motive_57_marker_attachments(),
        required_static_markers=tuple(sorted(set(MOTIVE_57_MARKER_NAMES) | {"LGJC", "RGJC"})),
        functional_trials=motive_57_functional_trials() if use_functional else (),
        root_segment_name="Pelvis",
    )


def _pelvis_segment() -> SegmentSpec:
    # ISB: origin at mid-ASIS, +Z left-to-right, +X anterior in the pelvic plane.
    return SegmentSpec(
        name="Pelvis",
        parent_name="root",
        translations=Translations.XYZ,
        rotations=Rotations.ZXY,
        frame=LocalFrameSpec(
            origin=_p("LIAS", "RIAS"),
            first_axis=AxisSpec.from_markers(Axis.Name.Z, "LIAS", "RIAS"),
            second_axis=AxisSpec.from_markers(Axis.Name.X, ("LIPS", "RIPS"), ("LIAS", "RIAS")),
            axis_to_keep=Axis.Name.Z,
        ),
        mesh_points=(_p("LIPS"), _p("RIPS"), _p("RIAS"), _p("LIAS"), _p("LIPS")),
    )


def _thorax_segment() -> SegmentSpec:
    # ISB thorax: IJ/SJN origin and the lower-to-upper midline based on PX/T8 and IJ/C7.
    return SegmentSpec(
        name="Thorax",
        parent_name="Pelvis",
        rotations=Rotations.ZXY,
        frame=_thorax_frame(_p("SJN")),
        mesh_points=(_p("SXS"), _p("SJN"), _p("CV7"), _p("TV7"), _p("SXS")),
    )


def _head_segment() -> SegmentSpec:
    # Part I/II does not prescribe a head frame; retain the laboratory X-forward convention.
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
    hip_center = _hip_center_spec(side, use_functional)
    knee_center = _knee_center_spec(side, use_functional)
    knee_fallback = knee_center.fallback if isinstance(knee_center, FunctionalAxisProjectionPointSpec) else knee_center
    return SegmentSpec(
        name=f"{side}Thigh",
        parent_name="Pelvis",
        rotations=Rotations.ZXY,
        joint_segment_name=f"{side}HipJoint",
        joint_frame=LocalFrameSpec(
            origin=hip_center,
            first_axis=AxisSpec.from_markers(Axis.Name.Z, "LIAS", "RIAS"),
            second_axis=AxisSpec.from_markers(Axis.Name.X, ("LIPS", "RIPS"), ("LIAS", "RIAS")),
            axis_to_keep=Axis.Name.Z,
        ),
        frame=LocalFrameSpec(
            origin=hip_center,
            first_axis=AxisSpec(Axis.Name.Y, knee_center, hip_center),
            second_axis=_femoral_axis(side),
            axis_to_keep=Axis.Name.Y,
        ),
        mesh_points=(
            (hip_center.fallback if isinstance(hip_center, FunctionalCenterSpec) else hip_center),
            _p(f"{side}FTC"),
            _p(f"{side}TH"),
            _p(f"{side}FLE"),
            knee_fallback,
            _p(f"{side}FME"),
        ),
    )


def _shank_segment(side: str, use_functional: bool) -> SegmentSpec:
    hip_center = _hip_center_spec(side, use_functional)
    knee_center = _knee_center_spec(side, use_functional)
    intermalleolar_center = _intermalleolar_center_spec(side)
    return SegmentSpec(
        name=f"{side}Shank",
        parent_name=f"{side}Thigh",
        rotations=Rotations.ZXY,
        joint_segment_name=f"{side}KneeJoint",
        joint_frame=LocalFrameSpec(
            origin=knee_center,
            first_axis=AxisSpec(Axis.Name.Y, knee_center, hip_center),
            second_axis=(_sara_axis_spec(side) if use_functional else _femoral_axis(side)),
            axis_to_keep=Axis.Name.Z,
        ),
        frame=LocalFrameSpec(
            # ISB tibia/fibula frame: origin at the intermalleolar midpoint.
            origin=intermalleolar_center,
            first_axis=AxisSpec(Axis.Name.Y, intermalleolar_center, knee_center),
            second_axis=_malleolar_axis(side),
            axis_to_keep=Axis.Name.Y,
        ),
        mesh_points=(
            _p(f"{side}FAX"),
            _p(f"{side}SK"),
            _p(f"{side}TTC"),
            _p(f"{side}FAL"),
            intermalleolar_center,
            _p(f"{side}TAM"),
        ),
    )


def _foot_segment(side: str, use_functional: bool) -> SegmentSpec:
    ankle_center = _ankle_center_spec(side, use_functional)
    knee_center = _knee_center_spec(side, use_functional)
    intermalleolar_center = _intermalleolar_center_spec(side)
    return SegmentSpec(
        name=f"{side}Foot",
        parent_name=f"{side}Shank",
        rotations=Rotations.ZXY,
        joint_segment_name=f"{side}AnkleJoint",
        joint_frame=LocalFrameSpec(
            origin=ankle_center,
            first_axis=AxisSpec(Axis.Name.Y, intermalleolar_center, knee_center),
            second_axis=_malleolar_axis(side),
            axis_to_keep=Axis.Name.Y,
        ),
        frame=LocalFrameSpec(
            origin=intermalleolar_center,
            # ISB calcaneus neutral frame: +Y follows the tibia and +X points anteriorly.
            first_axis=AxisSpec(Axis.Name.Y, intermalleolar_center, knee_center),
            second_axis=AxisSpec.from_markers(
                Axis.Name.X,
                (f"{side}FCC", f"{side}FCC"),
                (f"{side}FM5", f"{side}FM1"),
            ),
            axis_to_keep=Axis.Name.Y,
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
    shoulder_center = _p(f"{side}GJC")
    elbow_center = _p(f"{side}HLE", f"{side}HME")
    return SegmentSpec(
        name=f"{side}UpperArm",
        parent_name="Thorax",
        # No scapular triad is available, so this is the ISB thoracohumeral
        # reporting sequence rather than a true scapulohumeral GH sequence.
        rotations=Rotations.YXY,
        joint_segment_name=f"{side}ShoulderJoint",
        joint_frame=_thorax_frame(shoulder_center),
        # Humerus option 1: GH origin, proximal +Y and epicondylar +Z.
        frame=_humerus_frame(side, shoulder_center),
        mesh_points=(
            shoulder_center,
            _p(f"{side}UA"),
            _p(f"{side}HLE"),
            elbow_center,
            _p(f"{side}HME"),
            shoulder_center,
        ),
    )


def _forearm_segment(side: str) -> SegmentSpec:
    elbow_center = _p(f"{side}HLE", f"{side}HME")
    wrist_center = _p(f"{side}USP", f"{side}RSP")
    ulnar_styloid = _p(f"{side}USP")
    return SegmentSpec(
        name=f"{side}Forearm",
        parent_name=f"{side}UpperArm",
        # ISB elbow JCS: humeral Z, floating X, forearm Y.
        rotations=Rotations.ZXY,
        joint_segment_name=f"{side}ElbowJoint",
        joint_frame=_humerus_frame(side, elbow_center),
        # ISB forearm: US origin, proximal +Y, rightward +Z.
        frame=_forearm_frame(side, ulnar_styloid),
        mesh_points=(
            elbow_center,
            _p(f"{side}USP"),
            wrist_center,
            _p(f"{side}RSP"),
            elbow_center,
        ),
    )


def _hand_segment(side: str) -> SegmentSpec:
    wrist_center = _p(f"{side}USP", f"{side}RSP")
    return SegmentSpec(
        name=f"{side}Hand",
        parent_name=f"{side}Forearm",
        # ISB global wrist JCS: forearm Z, floating X, metacarpal Y.
        rotations=Rotations.ZXY,
        joint_segment_name=f"{side}WristJoint",
        joint_frame=_forearm_frame(side, wrist_center),
        frame=LocalFrameSpec(
            # HM2 is the available proxy for the recommended third metacarpal.
            origin=wrist_center,
            first_axis=AxisSpec.from_markers(Axis.Name.Y, f"{side}HM2", (f"{side}USP", f"{side}RSP")),
            second_axis=_styloid_axis(side),
            axis_to_keep=Axis.Name.Y,
        ),
        mesh_points=(wrist_center, _p(f"{side}HM2")),
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
    fallback = _intermalleolar_center_spec(side)
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


def _intermalleolar_center_spec(side: str) -> MarkerEndpointSpec:
    return _p(f"{side}FAL", f"{side}TAM")


def _knee_center_spec(side: str, use_functional: bool):
    fallback = _p(f"{side}FLE", f"{side}FME")
    if not use_functional:
        return fallback
    sara_axis = _sara_axis_spec(side)
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


def _sara_axis_spec(side: str) -> FunctionalAxisSpec:
    expected_axis = _femoral_axis(side)
    return FunctionalAxisSpec(
        method=FunctionalMethod.SARA_DIRECTION,
        trial_name="left_knee_sara" if side == "L" else "right_knee_sara",
        fallback=expected_axis,
        parent_marker_names=(f"{side}FTC", f"{side}TH", f"{side}FLE", f"{side}FME"),
        child_marker_names=(
            f"{side}FAX",
            f"{side}SK",
            f"{side}TTC",
            f"{side}FAL",
            f"{side}TAM",
        ),
        expected_axis=expected_axis,
        origin_marker_names=(f"{side}FLE", f"{side}FME"),
        max_static_axis_deviation_degrees=MOTIVE_57_KNEE_SARA_MAX_STATIC_AXIS_DEVIATION_DEGREES,
    )


def _femoral_axis(side: str) -> AxisSpec:
    return _femoral_or_humeral_axis(side, "FME", "FLE")


def _malleolar_axis(side: str) -> AxisSpec:
    return _femoral_or_humeral_axis(side, "TAM", "FAL")


def _femoral_or_humeral_axis(side: str, medial_suffix: str, lateral_suffix: str) -> AxisSpec:
    if side == "R":
        return AxisSpec.from_markers(Axis.Name.Z, f"{side}{medial_suffix}", f"{side}{lateral_suffix}")
    return AxisSpec.from_markers(Axis.Name.Z, f"{side}{lateral_suffix}", f"{side}{medial_suffix}")


def _thorax_frame(origin: MarkerEndpointSpec) -> LocalFrameSpec:
    return LocalFrameSpec(
        origin=origin,
        first_axis=AxisSpec.from_markers(Axis.Name.Y, ("SXS", "TV7"), ("SJN", "CV7")),
        second_axis=AxisSpec.from_markers(Axis.Name.X, ("TV7", "CV7"), ("SXS", "SJN")),
        axis_to_keep=Axis.Name.Y,
    )


def _humerus_frame(side: str, origin: MarkerEndpointSpec) -> LocalFrameSpec:
    shoulder_center = _p(f"{side}GJC")
    elbow_center = _p(f"{side}HLE", f"{side}HME")
    return LocalFrameSpec(
        origin=origin,
        first_axis=AxisSpec(Axis.Name.Y, elbow_center, shoulder_center),
        second_axis=_femoral_or_humeral_axis(side, "HME", "HLE"),
        axis_to_keep=Axis.Name.Y,
    )


def _forearm_frame(side: str, origin: MarkerEndpointSpec) -> LocalFrameSpec:
    elbow_center = _p(f"{side}HLE", f"{side}HME")
    return LocalFrameSpec(
        origin=origin,
        first_axis=AxisSpec(Axis.Name.Y, _p(f"{side}USP"), elbow_center),
        second_axis=_styloid_axis(side),
        axis_to_keep=Axis.Name.Y,
    )


def _styloid_axis(side: str) -> AxisSpec:
    if side == "R":
        return AxisSpec.from_markers(Axis.Name.Z, "RUSP", "RRSP")
    return AxisSpec.from_markers(Axis.Name.Z, "LRSP", "LUSP")


def _p(*marker_names: str) -> MarkerEndpointSpec:
    return MarkerEndpointSpec(tuple(marker_names))
