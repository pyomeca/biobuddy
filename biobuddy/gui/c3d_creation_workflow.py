from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..utils.marker_data import C3dData, MarkerData
from .c3d_model_creation import C3dModelPreset, c3d_model_preset_virtual_features
from .full_body_bela_template import bela_segment_specs
from .lower_limb_template import lower_limb_template
from .model_builder import ModelTemplate, required_functional_markers, required_static_markers
from .upper_limb_template import upper_limb_template


@dataclass(frozen=True)
class C3dWorkflowStep:
    """
    One step in the C3D-driven model creation workflow.
    """

    number: int
    name: str
    description: str


@dataclass(frozen=True)
class C3dFileRole:
    """
    Generic file name expected by a C3D model creation preset.
    """

    role: str
    generic_name: str
    description: str
    required: bool = False


@dataclass(frozen=True)
class C3dSegmentMarkerGroup:
    """
    Markers associated with one segment in a preset.
    """

    segment_name: str
    marker_names: tuple[str, ...]


@dataclass(frozen=True)
class C3dCreationWorkflow:
    """
    Complete GUI-facing description of a preset workflow.
    """

    preset: C3dModelPreset
    steps: tuple[C3dWorkflowStep, ...]
    file_roles: tuple[C3dFileRole, ...]
    segment_marker_groups: tuple[C3dSegmentMarkerGroup, ...]


def c3d_creation_workflow(preset: C3dModelPreset) -> C3dCreationWorkflow:
    """
    Return the GUI-facing workflow for one model preset.
    """
    return C3dCreationWorkflow(
        preset=preset,
        steps=c3d_creation_workflow_steps(),
        file_roles=c3d_file_roles_for_preset(preset),
        segment_marker_groups=c3d_segment_marker_groups_for_preset(preset),
    )


def c3d_creation_workflow_steps() -> tuple[C3dWorkflowStep, ...]:
    """
    Return the shared workflow steps for creating a model from C3D data.
    """
    return (
        C3dWorkflowStep(1, "New from C3D", "Start a C3D-driven model creation session."),
        C3dWorkflowStep(2, "Choose a C3D file", "Load the anatomical/static trial used to inspect markers."),
        C3dWorkflowStep(3, "List markers", "Inspect all marker names available in the selected C3D file."),
        C3dWorkflowStep(4, "Create segments", "Create or load the segment chain for the selected preset."),
        C3dWorkflowStep(5, "Assign markers", "Assign raw markers to one or more segments."),
        C3dWorkflowStep(6, "Create virtual markers", "Add points from equations, pointing trials, or functional C3Ds."),
        C3dWorkflowStep(7, "Create axes", "Define anatomical, technical, SCORE, or SARA axes."),
        C3dWorkflowStep(8, "Child translations", "Choose whether child segments get translations."),
        C3dWorkflowStep(9, "Initial rotation", "Set initial rotations from a matrix or anatomical posture C3D."),
        C3dWorkflowStep(10, "DoF", "Choose rotations/translations and ranges of motion."),
        C3dWorkflowStep(11, "Generate template", "Save a reusable template for other participants."),
        C3dWorkflowStep(12, "Rename C3Ds", "Rename calibration files with generic preset-specific names."),
    )


def marker_names_from_c3d_file(filepath: str | Path) -> tuple[str, ...]:
    """
    Load marker names from one C3D file.
    """
    return tuple(C3dData(str(filepath)).marker_names)


def c3d_file_roles_for_preset(preset: C3dModelPreset) -> tuple[C3dFileRole, ...]:
    """
    Return recommended generic C3D names for a preset.
    """
    if preset == C3dModelPreset.LOWER_LIMBS:
        return (
            C3dFileRole("static", "static_anatomical.c3d", "Static anatomical trial.", required=True),
            C3dFileRole("left_hip_score", "functional_left_hip_score.c3d", "Left hip SCORE trial."),
            C3dFileRole("left_knee_sara", "functional_left_knee_sara.c3d", "Left knee SARA trial."),
            C3dFileRole("left_ankle_score", "functional_left_ankle_score.c3d", "Left ankle SCORE trial."),
            C3dFileRole("right_hip_score", "functional_right_hip_score.c3d", "Right hip SCORE trial."),
            C3dFileRole("right_knee_sara", "functional_right_knee_sara.c3d", "Right knee SARA trial."),
            C3dFileRole("right_ankle_score", "functional_right_ankle_score.c3d", "Right ankle SCORE trial."),
        )
    if preset == C3dModelPreset.UPPER_LIMB:
        return (
            C3dFileRole("static", "static_anatomical.c3d", "Static upper-limb anatomical trial.", required=True),
            C3dFileRole("pointing", "pointing_virtual_markers.c3d", "Pointing trial for virtual anatomical points."),
            C3dFileRole("shoulder_score", "functional_shoulder_score.c3d", "Shoulder center functional trial."),
            C3dFileRole("elbow_sara", "functional_elbow_sara.c3d", "Elbow flexion axis SARA trial."),
            C3dFileRole("forearm_sara", "functional_forearm_sara.c3d", "Forearm pronation/supination SARA trial."),
            C3dFileRole("wrist_score", "functional_wrist_score.c3d", "Wrist center functional trial."),
        )
    if preset == C3dModelPreset.FULL_BODY:
        return (
            C3dFileRole("static", "static_anatomical.c3d", "Static full-body anatomical trial.", required=True),
            C3dFileRole("pointing", "pointing_virtual_markers.c3d", "Pointing trial for virtual joint centers."),
            C3dFileRole("lower_limb_score_sara", "functional_lower_limb_score_sara.c3d", "Lower-limb CoR and axes."),
            C3dFileRole("upper_limb_score_sara", "functional_upper_limb_score_sara.c3d", "Upper-limb CoR and axes."),
            C3dFileRole("spine_head", "functional_spine_head.c3d", "Thorax, head, and trunk virtual points."),
        )
    raise ValueError(f"Unsupported C3D model preset: {preset}.")


def c3d_segment_marker_groups_for_preset(preset: C3dModelPreset) -> tuple[C3dSegmentMarkerGroup, ...]:
    """
    Return segment marker groups for raw markers in a preset.
    """
    if preset == C3dModelPreset.LOWER_LIMBS:
        return _groups_from_model_template(lower_limb_template())
    if preset == C3dModelPreset.UPPER_LIMB:
        return _groups_from_model_template(upper_limb_template())
    if preset == C3dModelPreset.FULL_BODY:
        return tuple(
            C3dSegmentMarkerGroup(segment.name, tuple(segment.marker_names)) for segment in bela_segment_specs()
        )
    raise ValueError(f"Unsupported C3D model preset: {preset}.")


def c3d_workflow_summary(preset: C3dModelPreset, data: MarkerData | None = None) -> str:
    """
    Return a compact text summary for the selected preset and optional C3D data.
    """
    workflow = c3d_creation_workflow(preset)
    lines = [f"Preset: {preset.value}", f"Segments: {len(workflow.segment_marker_groups)}"]
    if data is not None:
        expected_markers = _expected_marker_names_for_preset(preset)
        present_markers = set(data.marker_names)
        missing_markers = sorted(expected_markers - present_markers)
        extra_markers = sorted(present_markers - expected_markers)
        lines.extend(
            (
                f"C3D markers: {len(data.marker_names)}",
                f"Expected markers: {len(expected_markers)}",
                f"Missing expected markers: {len(missing_markers)}",
                f"Extra C3D markers: {len(extra_markers)}",
            )
        )
    virtual_features = c3d_model_preset_virtual_features(preset)
    lines.append(f"Virtual features to define: {len(virtual_features)}")
    lines.append(f"Generic C3D names: {', '.join(role.generic_name for role in workflow.file_roles)}")
    return "\n".join(lines)


def c3d_template_payload(preset: C3dModelPreset) -> dict:
    """
    Return a serializable template payload for reusing the workflow on another participant.
    """
    workflow = c3d_creation_workflow(preset)
    return {
        "preset": preset.value,
        "steps": [asdict(step) for step in workflow.steps],
        "c3d_file_roles": [asdict(role) for role in workflow.file_roles],
        "segment_marker_groups": [asdict(group) for group in workflow.segment_marker_groups],
        "virtual_features": [asdict(feature) for feature in c3d_model_preset_virtual_features(preset)],
    }


def _groups_from_model_template(template: ModelTemplate) -> tuple[C3dSegmentMarkerGroup, ...]:
    marker_segments = {}
    for attachment in template.marker_attachments:
        for segment_name in attachment.segment_names:
            marker_segments.setdefault(segment_name, []).append(attachment.name)
    return tuple(
        C3dSegmentMarkerGroup(segment.name, tuple(marker_segments.get(segment.name, ())))
        for segment in template.segments
    )


def _expected_marker_names_for_preset(preset: C3dModelPreset) -> set[str]:
    if preset == C3dModelPreset.LOWER_LIMBS:
        template = lower_limb_template()
        return set(required_static_markers(template)) | set().union(*required_functional_markers(template).values())
    if preset == C3dModelPreset.UPPER_LIMB:
        return set(required_static_markers(upper_limb_template()))
    if preset == C3dModelPreset.FULL_BODY:
        return {marker_name for segment in bela_segment_specs() for marker_name in segment.marker_names}
    raise ValueError(f"Unsupported C3D model preset: {preset}.")
