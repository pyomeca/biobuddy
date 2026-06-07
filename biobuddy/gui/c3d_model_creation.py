from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..utils.marker_data import C3dData, MarkerData
from .full_body_bela_template import bela_unresolved_marker_references, bela_virtual_marker_reference_map
from .lower_limb_template import lower_limb_template
from .upper_limb_template import upper_limb_template, upper_limb_virtual_feature_requirements
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
)


class C3dModelPreset(Enum):
    """
    Model templates that can be created from calibration C3D files.
    """

    FULL_BODY = "full_body"
    LOWER_LIMBS = "lower_limbs"
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
        C3dModelPreset.LOWER_LIMBS,
        C3dModelPreset.UPPER_LIMB,
    )


def c3d_model_preset_virtual_features(preset: C3dModelPreset) -> tuple[C3dPresetVirtualFeature, ...]:
    """
    Return virtual features that still need explicit reconstruction for a preset.
    """
    if preset == C3dModelPreset.LOWER_LIMBS:
        return _lower_limb_score_virtual_features()
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
        features = []
        virtual_reference_map = bela_virtual_marker_reference_map()
        for segment_name, indices in bela_unresolved_marker_references().items():
            for index in indices:
                default_name = f"{segment_name}_virtual_{index}"
                name, method, description = virtual_reference_map.get(
                    (segment_name, index),
                    (default_name, "legacy_matlab_reference", f"BeLa Matlab local index {index}"),
                )
                features.append(
                    C3dPresetVirtualFeature(
                        name=name,
                        feature_type="axis" if method == "sara" and "direction" in name else "point",
                        segment_name=segment_name,
                        role=method,
                        description=description,
                    )
                )
        return tuple(features)
    raise ValueError(f"Unsupported C3D model preset: {preset}.")


def _lower_limb_score_virtual_features() -> tuple[C3dPresetVirtualFeature, ...]:
    """
    Return lower-limb SCoRE centers used by the lower-body template.
    """
    score_specs = (
        (
            "CoR_LThigh_in_Pelvis",
            "LThigh",
            "left_hip_score",
            ("LPSI", "RPSI", "LASI", "RASI"),
            ("LTHI", "LTHIB", "LTHID"),
        ),
        (
            "CoR_LFoot_in_LShank",
            "LFoot",
            "left_ankle_score",
            ("LTIB", "LTIBF", "LTIBD"),
            ("LHEE", "LNAV", "LTOE", "LTOE5"),
        ),
        (
            "CoR_RThigh_in_Pelvis",
            "RThigh",
            "right_hip_score",
            ("LPSI", "RPSI", "LASI", "RASI"),
            ("RTHI", "RTHIB", "RTHID"),
        ),
        (
            "CoR_RFoot_in_RShank",
            "RFoot",
            "right_ankle_score",
            ("RTIB", "RTIBF", "RTIBD"),
            ("RHEE", "RNAV", "RTOE", "RTOE5"),
        ),
    )
    return tuple(
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


def template_for_c3d_model_preset(preset: C3dModelPreset) -> ModelTemplate:
    """
    Return the model template associated with a C3D creation preset.
    """
    if preset == C3dModelPreset.LOWER_LIMBS:
        return lower_limb_template()
    if preset == C3dModelPreset.FROM_SCRATCH:
        raise NotImplementedError(
            "Template-free C3D model creation is an interactive drafting workflow. Add segments, markers, axes, "
            "DoFs, and virtual markers in the GUI, then export a reusable template before generating a BioMod model."
        )
    if preset == C3dModelPreset.FULL_BODY:
        raise NotImplementedError(
            "Full-body C3D model creation still needs CoR/SARA virtual-point reconstruction before it can generate "
            "a BioMod model."
        )
    if preset == C3dModelPreset.UPPER_LIMB:
        return upper_limb_template()
    raise ValueError(f"Unsupported C3D model preset: {preset}.")


def create_model_from_c3d_folder(
    calibration_folder: Path,
    preset: C3dModelPreset = C3dModelPreset.LOWER_LIMBS,
    static_patterns: tuple[str, ...] = ("*static*.c3d", "*func_anat.c3d"),
    static_virtual_points: tuple[VirtualPointDefinition, ...] = (),
    static_virtual_axes: tuple[VirtualAxisDefinition, ...] = (),
    functional_virtual_points: dict[str, tuple[VirtualPointDefinition, ...]] | None = None,
    functional_virtual_axes: dict[str, tuple[VirtualAxisDefinition, ...]] | None = None,
) -> C3dModelCreationResult:
    """
    Create a model from a calibration folder containing C3D files.
    """
    template = template_for_c3d_model_preset(preset)
    static_data = C3dData(str(find_static_c3d_file(calibration_folder, static_patterns)))
    functional_data = load_functional_c3d_trials(template=template, calibration_folder=calibration_folder)
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
) -> C3dModelCreationResult:
    """
    Create a model from already-loaded marker data.
    """
    functional_data = {} if functional_data is None else functional_data
    static_data = marker_data_with_virtual_features(
        static_data,
        point_definitions=static_virtual_points,
        axis_definitions=static_virtual_axes,
    )
    functional_data = _functional_data_with_virtual_features(
        functional_data=functional_data,
        point_definitions_by_trial=functional_virtual_points,
        axis_definitions_by_trial=functional_virtual_axes,
    )
    marker_reports = template_marker_availability(template, static_data, functional_data)
    model = build_real_model(template=template, static_data=static_data, functional_data=functional_data)
    frame_quality = compute_frame_quality(template, static_data)
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


def find_static_c3d_file(
    calibration_folder: Path,
    static_patterns: tuple[str, ...] = ("*static*.c3d", "*func_anat.c3d"),
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
        return "lower_body.bioMod"
    if preset == C3dModelPreset.FULL_BODY:
        return "full_body.bioMod"
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
