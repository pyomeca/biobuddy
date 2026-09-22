from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..utils.marker_data import (
    C3dData,
    MarkerData,
    marker_data_with_stripped_prefixes,
)
from .full_body_model202_template import (
    model202_segment_specs,
    full_body_model202_template,
)
from .lower_limb_template import lower_limb_template
from .lower_limb_template import LOWER_LIMB_KNEE_SARA_MAX_STATIC_AXIS_DEVIATION_DEGREES
from .motive_57_template import (
    MOTIVE_57_FUNCTIONAL_C3D_FILENAMES,
    MOTIVE_57_KNEE_SARA_MAX_STATIC_AXIS_DEVIATION_DEGREES,
    motive_57_functional_trials,
    motive_57_template,
)
from .motive_57_isb_template import motive_57_isb_template
from .upper_limb_template import (
    upper_limb_template,
    upper_limb_virtual_feature_requirements,
)
from .model_builder import (
    FrameQuality,
    MarkerAvailabilityReport,
    ModelTemplate,
    build_real_model,
    compute_frame_quality,
    load_functional_c3d_trials,
    template_marker_availability,
)
from .virtual_points import (
    VirtualAxisDefinition,
    VirtualPointDefinition,
    marker_data_with_virtual_features,
    predictive_rab2002_shoulder_cor,
)

ProgressCallback = Callable[[str], None]

DEFAULT_STATIC_C3D_PATTERNS = (
    "Test_anato.c3d",
    "Test_main.c3d",
    "main_markers.c3d",
    "*static*.c3d",
    "*Static*.c3d",
    "*func_anat.c3d",
    "anatomical_posture.c3d",
)


class C3dModelPreset(Enum):
    """
    Model templates that can be created from calibration C3D files.
    """

    FULL_BODY = "full_body"
    MOTIVE_57 = "motive_57"
    MOTIVE_57_ISB = "motive_57_isb"
    LOWER_LIMBS = "lower_limbs"
    LOWER_LIMBS_ANATOMICAL = "lower_limbs_anatomical"
    UPPER_LIMB = "upper_limb"
    FROM_SCRATCH = "from_scratch"


@dataclass(frozen=True)
class C3dModelCreationResult:
    """
    Result of a C3D-driven model creation workflow.
    """

    model: BiomechanicalModelReal
    template: ModelTemplate
    preset: C3dModelPreset
    marker_reports: dict[str, MarkerAvailabilityReport]
    frame_quality: dict[str, FrameQuality]
    output_filename: str
    static_data: MarkerData
    functional_data: dict[str, MarkerData]


@dataclass(frozen=True)
class C3dModelCreationVariantResults:
    """
    Pair of lower-limb models generated with and without functional SCoRE/SARA calibration.
    """

    score: C3dModelCreationResult
    no_score: C3dModelCreationResult


@dataclass(frozen=True)
class C3dPresetVirtualFeature:
    """
    Point or axis that must be reconstructed before a preset can be generated.
    """

    name: str
    feature_type: str
    segment_name: str
    role: str
    description: str


def supported_c3d_model_presets() -> tuple[C3dModelPreset, ...]:
    """
    Return presets shown in the C3D creation workflow.
    """
    return (
        C3dModelPreset.FROM_SCRATCH,
        C3dModelPreset.FULL_BODY,
        C3dModelPreset.MOTIVE_57,
        C3dModelPreset.MOTIVE_57_ISB,
        C3dModelPreset.LOWER_LIMBS,
        C3dModelPreset.LOWER_LIMBS_ANATOMICAL,
        C3dModelPreset.UPPER_LIMB,
    )


def c3d_model_preset_from_cli_value(value: str | C3dModelPreset) -> C3dModelPreset:
    """
    Resolve a command-line preset value to a C3D model preset.
    """
    if isinstance(value, C3dModelPreset):
        return value
    raw_value = value.strip().lower()
    for preset in C3dModelPreset:
        if raw_value == preset.value.lower():
            return preset
    normalized_value = value.strip().lower().replace("_", "-")
    aliases = {
        "full-body": C3dModelPreset.FULL_BODY,
        "model202": C3dModelPreset.FULL_BODY,
        "motive-57": C3dModelPreset.MOTIVE_57,
        "biobuddy-motive-57": C3dModelPreset.MOTIVE_57,
        "biomech-motive-57": C3dModelPreset.MOTIVE_57,
        "biomech-motive": C3dModelPreset.MOTIVE_57,
        "motive-57-isb": C3dModelPreset.MOTIVE_57_ISB,
        "biobuddy-motive-57-isb": C3dModelPreset.MOTIVE_57_ISB,
        "biomech-motive-57-isb": C3dModelPreset.MOTIVE_57_ISB,
        "lower-limbs": C3dModelPreset.LOWER_LIMBS_ANATOMICAL,
        "lower-limbs-anatomical": C3dModelPreset.LOWER_LIMBS_ANATOMICAL,
        "lower-limbs-functional": C3dModelPreset.LOWER_LIMBS,
        "upper-limb": C3dModelPreset.UPPER_LIMB,
        "from-scratch": C3dModelPreset.FROM_SCRATCH,
    }
    if normalized_value in aliases:
        return aliases[normalized_value]
    available_values = ", ".join(sorted(set(aliases) | {preset.value for preset in C3dModelPreset}))
    raise ValueError(f"Unsupported C3D model preset '{value}'. Available aliases: {available_values}.")


def default_static_virtual_points_for_c3d_model_preset(
    preset: C3dModelPreset,
) -> tuple[VirtualPointDefinition, ...]:
    """
    Return static virtual points required by a preset before model generation.
    """
    if preset in {C3dModelPreset.MOTIVE_57, C3dModelPreset.MOTIVE_57_ISB}:
        return (
            predictive_rab2002_shoulder_cor("LGJC", "LCAJ", "LHME", "LHLE"),
            predictive_rab2002_shoulder_cor("RGJC", "RCAJ", "RHME", "RHLE"),
        )
    return ()


def c3d_model_preset_virtual_features(
    preset: C3dModelPreset,
) -> tuple[C3dPresetVirtualFeature, ...]:
    """
    Return virtual features that still need explicit reconstruction for a preset.
    """
    if preset == C3dModelPreset.LOWER_LIMBS:
        return _lower_limb_score_virtual_features()
    if preset == C3dModelPreset.LOWER_LIMBS_ANATOMICAL:
        return ()
    if preset in {C3dModelPreset.MOTIVE_57, C3dModelPreset.MOTIVE_57_ISB}:
        return _motive_57_virtual_features()
    if preset == C3dModelPreset.FROM_SCRATCH:
        return ()
    if preset == C3dModelPreset.UPPER_LIMB:
        return tuple(
            C3dPresetVirtualFeature(
                name=requirement.name,
                feature_type=requirement.feature_type,
                segment_name=requirement.segment_name,
                role=requirement.role,
                description=(
                    f"Matlab indices {', '.join(str(index) for index in requirement.matlab_indices)} "
                    f"for {requirement.role}"
                ),
            )
            for requirement in upper_limb_virtual_feature_requirements()
        )
    if preset == C3dModelPreset.FULL_BODY:
        return _full_body_score_sara_virtual_features()
    raise ValueError(f"Unsupported C3D model preset: {preset}.")


def _lower_limb_score_virtual_features() -> tuple[C3dPresetVirtualFeature, ...]:
    """
    Return lower-limb functional centers and axes used by the lower-body template.
    """
    score_specs = (
        (
            "CoR_Trunk_wrt_Pelvis",
            "Trunk",
            "trunk_score",
            ("LPSI", "RPSI", "LASI", "RASI"),
            ("T10", "T6", "C7", "C2", "CLAV", "STRN"),
        ),
        (
            "CoR_LThigh_wrt_Pelvis",
            "LThigh",
            "left_hip_score",
            ("LPSI", "RPSI", "LASI", "RASI"),
            ("LTHI", "LTHIB", "LTHID"),
        ),
        (
            "CoR_LFoot_wrt_LShank",
            "LFoot",
            "left_ankle_score",
            ("LTIB", "LTIBF", "LTIBD"),
            ("LHEE", "LNAV", "LTOE", "LTOE5"),
        ),
        (
            "CoR_RThigh_wrt_Pelvis",
            "RThigh",
            "right_hip_score",
            ("LPSI", "RPSI", "LASI", "RASI"),
            ("RTHI", "RTHIB", "RTHID"),
        ),
        (
            "CoR_RFoot_wrt_RShank",
            "RFoot",
            "right_ankle_score",
            ("RTIB", "RTIBF", "RTIBD"),
            ("RHEE", "RNAV", "RTOE", "RTOE5"),
        ),
    )
    score_features = tuple(
        C3dPresetVirtualFeature(
            name=name,
            feature_type="point",
            segment_name=segment_name,
            role="score",
            description=(
                f"trial={trial_name}; parent markers={','.join(parent_markers)}; "
                f"child markers={','.join(child_markers)}"
            ),
        )
        for name, segment_name, trial_name, parent_markers, child_markers in score_specs
    )
    sara_specs = (
        (
            "Axis_LKnee_SARA",
            "LShank",
            "left_knee_sara",
            ("LTIBD", "LTIB", "LTIBF"),
            ("LTHIB", "LTHID", "LTHI"),
            ("LKNE", "LKNEM"),
        ),
        (
            "Axis_RKnee_SARA",
            "RShank",
            "right_knee_sara",
            ("RTHID", "RTHI", "RTHIB"),
            ("RTIB", "RTIBF", "RTIBD"),
            ("RKNEM", "RKNE"),
        ),
    )
    sara_features = tuple(
        C3dPresetVirtualFeature(
            name=name,
            feature_type="axis",
            segment_name=segment_name,
            role="sara_axis",
            description=(
                f"trial={trial_name}; parent markers={','.join(parent_markers)}; "
                f"child markers={','.join(child_markers)}; expected axis={','.join(expected_axis)}; "
                f"origin markers={','.join(expected_axis)}; "
                f"max static deviation={LOWER_LIMB_KNEE_SARA_MAX_STATIC_AXIS_DEVIATION_DEGREES:g}"
            ),
        )
        for name, segment_name, trial_name, parent_markers, child_markers, expected_axis in sara_specs
    )
    projected_knee_features = (
        C3dPresetVirtualFeature(
            name="Proj_LKnee_on_Axis_LKnee_SARA",
            feature_type="point",
            segment_name="LShank",
            role="axis_projection",
            description="point=LKNE,LKNEM; axis=Axis_LKnee_SARA; trial=left_knee_sara",
        ),
        C3dPresetVirtualFeature(
            name="Proj_RKnee_on_Axis_RKnee_SARA",
            feature_type="point",
            segment_name="RShank",
            role="axis_projection",
            description="point=RKNE,RKNEM; axis=Axis_RKnee_SARA; trial=right_knee_sara",
        ),
    )
    return score_features + projected_knee_features + sara_features


def _motive_57_virtual_features() -> tuple[C3dPresetVirtualFeature, ...]:
    """
    Return Motive (57) functional centers, axes, and Rab GH centers.
    """
    features = [
        C3dPresetVirtualFeature(
            name="LGJC",
            feature_type="point",
            segment_name="LUpperArm",
            role="rab2002_shoulder",
            description="method=Rab2002; point=LCAJ; mid=LHME,LHLE; fraction=0.17; exclude LCAJ and LHGT from humerus reconstruction",
        ),
        C3dPresetVirtualFeature(
            name="RGJC",
            feature_type="point",
            segment_name="RUpperArm",
            role="rab2002_shoulder",
            description="method=Rab2002; point=RCAJ; mid=RHME,RHLE; fraction=0.17; exclude RCAJ and RHGT from humerus reconstruction",
        ),
    ]
    for side, label in (("L", "left"), ("R", "right")):
        features.extend(
            (
                C3dPresetVirtualFeature(
                    name=f"CoR_{side}Thigh_wrt_Pelvis",
                    feature_type="point",
                    segment_name=f"{side}Thigh",
                    role="score",
                    description=(
                        f"trial={label}_hip_score; parent markers=LIAS,RIAS,LIPS,RIPS; "
                        f"child markers={side}FTC,{side}TH,{side}FLE,{side}FME"
                    ),
                ),
                C3dPresetVirtualFeature(
                    name=f"Axis_{side}Knee_SARA",
                    feature_type="axis",
                    segment_name=f"{side}Shank",
                    role="sara_axis",
                    description=(
                        f"trial={label}_knee_sara; parent markers={side}FTC,{side}TH,{side}FLE,{side}FME; "
                        f"child markers={side}FAX,{side}SK,{side}TTC,{side}FAL,{side}TAM; "
                        f"expected axis={_motive_57_knee_axis_markers(side)}; origin markers={side}FLE,{side}FME; "
                        f"max static deviation={MOTIVE_57_KNEE_SARA_MAX_STATIC_AXIS_DEVIATION_DEGREES:g}"
                    ),
                ),
                C3dPresetVirtualFeature(
                    name=f"Proj_{side}Knee_on_Axis_{side}Knee_SARA",
                    feature_type="point",
                    segment_name=f"{side}Shank",
                    role="axis_projection",
                    description=f"point={side}FLE,{side}FME; axis=Axis_{side}Knee_SARA; trial={label}_knee_sara",
                ),
                C3dPresetVirtualFeature(
                    name=f"CoR_{side}Foot_wrt_{side}Shank",
                    feature_type="point",
                    segment_name=f"{side}Foot",
                    role="score",
                    description=(
                        f"trial={label}_ankle_score; parent markers={side}FAX,{side}SK,{side}TTC,{side}FAL,{side}TAM; "
                        f"child markers={side}FCC,{side}FM5,{side}FM2,{side}FM1"
                    ),
                ),
            )
        )
    return tuple(features)


def _motive_57_knee_axis_markers(side: str) -> str:
    if side == "R":
        return "RFME,RFLE"
    return "LFLE,LFME"


def _full_body_score_sara_virtual_features() -> tuple[C3dPresetVirtualFeature, ...]:
    """
    Return full-body Model202 functional centers and axes reconstructed from generic C3D trials.
    """
    features: list[C3dPresetVirtualFeature] = []
    segments_by_name = {segment.name: segment for segment in model202_segment_specs()}
    for segment in model202_segment_specs():
        if segment.parent_name in {"", "base", "root"}:
            continue
        parent = segments_by_name[segment.parent_name]
        trial_name = _full_body_trial_name(segment)
        if segment.joint == "aor":
            expected_axis = _full_body_aor_expected_axis(segment)
            axis_name = _full_body_axis_name(segment)
            features.append(
                C3dPresetVirtualFeature(
                    name=axis_name,
                    feature_type="axis",
                    segment_name=segment.name,
                    role="sara_axis",
                    description=(
                        f"trial={trial_name}; parent markers={','.join(parent.marker_names)}; "
                        f"child markers={','.join(segment.marker_names)}; expected axis={','.join(expected_axis)}; "
                        f"origin markers={','.join(expected_axis)}"
                    ),
                )
            )
            features.append(
                C3dPresetVirtualFeature(
                    name=_full_body_joint_center_name(segment),
                    feature_type="point",
                    segment_name=segment.name,
                    role="axis_projection",
                    description=f"point={','.join(expected_axis)}; axis={axis_name}; trial={trial_name}",
                )
            )
        else:
            features.append(
                C3dPresetVirtualFeature(
                    name=_full_body_joint_center_name(segment),
                    feature_type="point",
                    segment_name=segment.name,
                    role="score",
                    description=(
                        f"trial={trial_name}; parent markers={','.join(parent.marker_names)}; "
                        f"child markers={','.join(segment.marker_names)}"
                    ),
                )
            )
    return tuple(features)


def _full_body_trial_name(segment) -> str:
    method_suffix = "sara" if segment.joint == "aor" else "score"
    return f"{segment.name.lower()}_{segment.parent_name.lower()}_{method_suffix}"


def _full_body_joint_center_name(segment) -> str:
    return f"CoR_{segment.name}_wrt_{segment.parent_name}"


def _full_body_axis_name(segment) -> str:
    return f"Axis_{segment.name}_SARA"


def _full_body_aor_expected_axis(segment) -> tuple[str, str]:
    if segment.name == "JambeD":
        return "CONDEXTD", "CONDINTD"
    if segment.name == "JambeG":
        return "CONDINTG", "CONEXTG"
    parent = next(candidate for candidate in model202_segment_specs() if candidate.name == segment.parent_name)
    return (
        parent.marker_names[segment.functional_axis_indices[0] - 1],
        parent.marker_names[segment.functional_axis_indices[1] - 1],
    )


def template_for_c3d_model_preset(preset: C3dModelPreset) -> ModelTemplate:
    """
    Return the model template associated with a C3D creation preset.
    """
    if preset == C3dModelPreset.LOWER_LIMBS:
        return lower_limb_template(use_functional=True)
    if preset == C3dModelPreset.LOWER_LIMBS_ANATOMICAL:
        return lower_limb_template(use_functional=False)
    if preset == C3dModelPreset.FROM_SCRATCH:
        raise NotImplementedError(
            "Template-free C3D model creation is an interactive drafting workflow. Add segments, markers, axes, "
            "DoFs, and virtual markers in the GUI, then export a reusable template before generating a BioMod model."
        )
    if preset == C3dModelPreset.FULL_BODY:
        return full_body_model202_template(use_functional=True)
    if preset == C3dModelPreset.MOTIVE_57:
        return motive_57_template(use_functional=True)
    if preset == C3dModelPreset.MOTIVE_57_ISB:
        return motive_57_isb_template(use_functional=True)
    if preset == C3dModelPreset.UPPER_LIMB:
        return upper_limb_template()
    raise ValueError(f"Unsupported C3D model preset: {preset}.")


def create_model_from_c3d_folder(
    calibration_folder: Path,
    preset: C3dModelPreset = C3dModelPreset.LOWER_LIMBS,
    static_patterns: tuple[str, ...] = DEFAULT_STATIC_C3D_PATTERNS,
    static_virtual_points: tuple[VirtualPointDefinition, ...] = (),
    static_virtual_axes: tuple[VirtualAxisDefinition, ...] = (),
    functional_virtual_points: dict[str, tuple[VirtualPointDefinition, ...]] | None = None,
    functional_virtual_axes: dict[str, tuple[VirtualAxisDefinition, ...]] | None = None,
    marker_name_prefixes_to_strip: tuple[str, ...] = (),
    progress_callback: ProgressCallback | None = None,
) -> C3dModelCreationResult:
    """
    Create a model from a calibration folder containing C3D files.
    """
    _notify_progress(progress_callback, "Preparing C3D model template...")
    template = template_for_c3d_model_preset(preset)
    _notify_progress(progress_callback, "Searching for the static C3D trial...")
    static_file = find_static_c3d_file(calibration_folder, static_patterns)
    _notify_progress(progress_callback, f"Loading static C3D: {static_file.name}")
    static_data = C3dData(str(static_file))
    static_data = marker_data_with_stripped_prefixes(static_data, marker_name_prefixes_to_strip)
    _notify_progress(progress_callback, "Loading functional C3D trials...")
    functional_data = load_functional_c3d_trials(
        template=template,
        calibration_folder=calibration_folder,
        marker_name_prefixes_to_strip=marker_name_prefixes_to_strip,
        progress_callback=progress_callback,
    )
    return create_model_from_marker_data(
        template=template,
        static_data=static_data,
        functional_data=functional_data,
        preset=preset,
        output_filename=_default_output_filename(preset),
        static_virtual_points=static_virtual_points,
        static_virtual_axes=static_virtual_axes,
        functional_virtual_points=functional_virtual_points,
        functional_virtual_axes=functional_virtual_axes,
        progress_callback=progress_callback,
    )


def create_lower_limb_model_variants_from_c3d_folder(
    calibration_folder: Path,
    static_patterns: tuple[str, ...] = DEFAULT_STATIC_C3D_PATTERNS,
) -> C3dModelCreationVariantResults:
    """
    Create lower-limb models with functional SCoRE/SARA enabled and disabled from a C3D folder.
    """
    static_data = C3dData(str(find_static_c3d_file(calibration_folder, static_patterns)))
    score_template = lower_limb_template(use_functional=True)
    no_score_template = lower_limb_template(use_functional=False)
    functional_data = load_functional_c3d_trials(template=score_template, calibration_folder=calibration_folder)
    return create_lower_limb_model_variants_from_marker_data(
        static_data=static_data,
        functional_data=functional_data,
        score_template=score_template,
        no_score_template=no_score_template,
    )


def create_lower_limb_model_variants_from_marker_data(
    static_data: MarkerData,
    functional_data: dict[str, MarkerData] | None = None,
    score_template: ModelTemplate | None = None,
    no_score_template: ModelTemplate | None = None,
) -> C3dModelCreationVariantResults:
    """
    Create lower-limb model variants matching ``use_score=True`` and ``use_score=False`` workflows.
    """
    score_template = lower_limb_template(use_functional=True) if score_template is None else score_template
    no_score_template = lower_limb_template(use_functional=False) if no_score_template is None else no_score_template
    return C3dModelCreationVariantResults(
        score=create_model_from_marker_data(
            template=score_template,
            static_data=static_data,
            functional_data=functional_data,
            preset=C3dModelPreset.LOWER_LIMBS,
            output_filename="lower_body_score.bioMod",
        ),
        no_score=create_model_from_marker_data(
            template=no_score_template,
            static_data=static_data,
            functional_data={},
            preset=C3dModelPreset.LOWER_LIMBS_ANATOMICAL,
            output_filename="lower_body_no_score.bioMod",
        ),
    )


def create_model_from_marker_data(
    template: ModelTemplate,
    static_data: MarkerData,
    functional_data: dict[str, MarkerData] | None = None,
    preset: C3dModelPreset = C3dModelPreset.LOWER_LIMBS,
    output_filename: str | None = None,
    static_virtual_points: tuple[VirtualPointDefinition, ...] = (),
    static_virtual_axes: tuple[VirtualAxisDefinition, ...] = (),
    functional_virtual_points: dict[str, tuple[VirtualPointDefinition, ...]] | None = None,
    functional_virtual_axes: dict[str, tuple[VirtualAxisDefinition, ...]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> C3dModelCreationResult:
    """
    Create a model from already-loaded marker data.
    """
    functional_data = {} if functional_data is None else functional_data
    _notify_progress(progress_callback, "Applying virtual markers to static data...")
    static_data = marker_data_with_virtual_features(
        static_data,
        point_definitions=static_virtual_points,
        axis_definitions=static_virtual_axes,
    )
    _notify_progress(progress_callback, "Applying virtual markers to functional data...")
    functional_data = _functional_data_with_virtual_features(
        functional_data=functional_data,
        point_definitions_by_trial=functional_virtual_points,
        axis_definitions_by_trial=functional_virtual_axes,
    )
    _notify_progress(progress_callback, "Checking marker availability...")
    marker_reports = template_marker_availability(template, static_data, functional_data)
    _notify_progress(progress_callback, "Building biomechanical model...")
    model = build_real_model(template=template, static_data=static_data, functional_data=functional_data)
    _notify_progress(progress_callback, "Computing frame quality metrics...")
    frame_quality = compute_frame_quality(template, static_data)
    _notify_progress(progress_callback, "Finalizing generated model...")
    return C3dModelCreationResult(
        model=model,
        template=template,
        preset=preset,
        marker_reports=marker_reports,
        frame_quality=frame_quality,
        output_filename=output_filename or _default_output_filename(preset),
        static_data=static_data,
        functional_data=functional_data,
    )


def _notify_progress(progress_callback: ProgressCallback | None, message: str) -> None:
    """
    Notify an optional GUI progress reporter about the current C3D creation step.
    """
    if progress_callback is not None:
        progress_callback(message)


def find_static_c3d_file(
    calibration_folder: Path,
    static_patterns: tuple[str, ...] = DEFAULT_STATIC_C3D_PATTERNS,
) -> Path:
    """
    Find the static/anatomical C3D used to instantiate marker-defined frames.
    """
    for static_pattern in static_patterns:
        matches = list(calibration_folder.glob(static_pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"Expected one static trial matching '{static_pattern}', found {len(matches)}.")
    patterns = ", ".join(static_patterns)
    raise RuntimeError(f"No static trial found. Expected one file matching one of: {patterns}.")


def _default_output_filename(preset: C3dModelPreset) -> str:
    if preset == C3dModelPreset.LOWER_LIMBS:
        return "lower_body_functional.bioMod"
    if preset == C3dModelPreset.LOWER_LIMBS_ANATOMICAL:
        return "lower_body.bioMod"
    if preset == C3dModelPreset.FULL_BODY:
        return "full_body.bioMod"
    if preset == C3dModelPreset.MOTIVE_57:
        return "motive_57.bioMod"
    if preset == C3dModelPreset.MOTIVE_57_ISB:
        return "motive_57_isb.bioMod"
    if preset == C3dModelPreset.UPPER_LIMB:
        return "upper_limb.bioMod"
    if preset == C3dModelPreset.FROM_SCRATCH:
        return "from_scratch.bioMod"
    raise ValueError(f"Unsupported C3D model preset: {preset}.")


def _functional_data_with_virtual_features(
    functional_data: dict[str, MarkerData],
    point_definitions_by_trial: dict[str, tuple[VirtualPointDefinition, ...]] | None,
    axis_definitions_by_trial: dict[str, tuple[VirtualAxisDefinition, ...]] | None,
) -> dict[str, MarkerData]:
    """
    Apply trial-specific virtual features to functional C3D marker data.
    """
    point_definitions_by_trial = {} if point_definitions_by_trial is None else point_definitions_by_trial
    axis_definitions_by_trial = {} if axis_definitions_by_trial is None else axis_definitions_by_trial
    augmented_data = {}
    for trial_name, data in functional_data.items():
        augmented_data[trial_name] = marker_data_with_virtual_features(
            data,
            point_definitions=point_definitions_by_trial.get(trial_name, ()),
            axis_definitions=axis_definitions_by_trial.get(trial_name, ()),
        )
    return augmented_data
