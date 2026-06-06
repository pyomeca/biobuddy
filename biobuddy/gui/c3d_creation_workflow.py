from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..utils.marker_data import C3dData, MarkerData
from .c3d_model_creation import C3dModelPreset, c3d_model_preset_virtual_features
from .full_body_bela_template import bela_segment_specs, rotations_from_matlab_dof, translations_from_matlab_dof
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


@dataclass(frozen=True)
class C3dVirtualMarkerDraft:
    """
    Editable virtual marker definition in a C3D workflow draft.
    """

    name: str
    method: str
    segment_name: str
    source: str = ""
    equation: str = ""


@dataclass(frozen=True)
class C3dAxisDraft:
    """
    Editable axis definition in a C3D workflow draft.
    """

    name: str
    segment_name: str
    axis: str
    start_markers: tuple[str, ...]
    end_markers: tuple[str, ...]
    method: str = "markers"


@dataclass(frozen=True)
class C3dSegmentSettingsDraft:
    """
    Editable kinematic settings for one segment.
    """

    segment_name: str
    translations: str = ""
    rotations: str = ""
    q_min: tuple[float, ...] = ()
    q_max: tuple[float, ...] = ()
    child_translation: bool = False
    initial_rotation_method: str = "identity"
    initial_rotation_source: str = ""
    initial_rotation_matrix: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class C3dFileAssignmentDraft:
    """
    Editable mapping from a generic C3D role to a participant-specific file.
    """

    role: str
    generic_name: str
    source_path: str = ""


@dataclass(frozen=True)
class C3dWorkflowDraft:
    """
    Editable state for an interactive C3D model creation session.
    """

    preset: C3dModelPreset
    segment_marker_groups: tuple[C3dSegmentMarkerGroup, ...]
    virtual_markers: tuple[C3dVirtualMarkerDraft, ...]
    axes: tuple[C3dAxisDraft, ...]
    segment_settings: tuple[C3dSegmentSettingsDraft, ...] = ()
    file_assignments: tuple[C3dFileAssignmentDraft, ...] = ()


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


def c3d_workflow_draft(preset: C3dModelPreset) -> C3dWorkflowDraft:
    """
    Return an editable draft initialized from a preset workflow.
    """
    virtual_features = c3d_model_preset_virtual_features(preset)
    return C3dWorkflowDraft(
        preset=preset,
        segment_marker_groups=c3d_segment_marker_groups_for_preset(preset),
        virtual_markers=tuple(
            C3dVirtualMarkerDraft(
                name=feature.name,
                method=_virtual_feature_default_method(feature.feature_type, feature.role),
                segment_name=feature.segment_name,
                source=feature.description,
            )
            for feature in virtual_features
            if feature.feature_type == "point"
        ),
        axes=tuple(
            C3dAxisDraft(
                name=feature.name,
                segment_name=feature.segment_name,
                axis="",
                start_markers=(f"{feature.name}_start",),
                end_markers=(f"{feature.name}_end",),
                method=_virtual_feature_default_method(feature.feature_type, feature.role),
            )
            for feature in virtual_features
            if feature.feature_type == "axis"
        ),
        segment_settings=_initial_segment_settings(preset),
        file_assignments=tuple(
            C3dFileAssignmentDraft(role=role.role, generic_name=role.generic_name)
            for role in c3d_file_roles_for_preset(preset)
        ),
    )


def add_segment_to_draft(draft: C3dWorkflowDraft, segment_name: str) -> C3dWorkflowDraft:
    """
    Add a segment to an editable draft.
    """
    segment_name = segment_name.strip()
    if segment_name == "":
        raise ValueError("Segment name cannot be empty.")
    if any(group.segment_name == segment_name for group in draft.segment_marker_groups):
        raise ValueError(f"Segment '{segment_name}' already exists.")
    return C3dWorkflowDraft(
        preset=draft.preset,
        segment_marker_groups=draft.segment_marker_groups + (C3dSegmentMarkerGroup(segment_name, ()),),
        virtual_markers=draft.virtual_markers,
        axes=draft.axes,
        segment_settings=draft.segment_settings + (C3dSegmentSettingsDraft(segment_name),),
        file_assignments=draft.file_assignments,
    )


def remove_segment_from_draft(draft: C3dWorkflowDraft, segment_name: str) -> C3dWorkflowDraft:
    """
    Remove a segment and its draft virtual definitions.
    """
    return C3dWorkflowDraft(
        preset=draft.preset,
        segment_marker_groups=tuple(
            group for group in draft.segment_marker_groups if group.segment_name != segment_name
        ),
        virtual_markers=tuple(marker for marker in draft.virtual_markers if marker.segment_name != segment_name),
        axes=tuple(axis for axis in draft.axes if axis.segment_name != segment_name),
        segment_settings=tuple(setting for setting in draft.segment_settings if setting.segment_name != segment_name),
        file_assignments=draft.file_assignments,
    )


def assign_marker_to_segment(draft: C3dWorkflowDraft, segment_name: str, marker_name: str) -> C3dWorkflowDraft:
    """
    Assign a marker to a segment. Markers may belong to several segments.
    """
    marker_name = marker_name.strip()
    if marker_name == "":
        raise ValueError("Marker name cannot be empty.")
    groups = []
    found_segment = False
    for group in draft.segment_marker_groups:
        if group.segment_name != segment_name:
            groups.append(group)
            continue
        found_segment = True
        marker_names = group.marker_names if marker_name in group.marker_names else group.marker_names + (marker_name,)
        groups.append(C3dSegmentMarkerGroup(group.segment_name, marker_names))
    if not found_segment:
        raise ValueError(f"Segment '{segment_name}' does not exist.")
    return _replace_draft_groups(draft, tuple(groups))


def unassign_marker_from_segment(draft: C3dWorkflowDraft, segment_name: str, marker_name: str) -> C3dWorkflowDraft:
    """
    Remove one marker assignment from one segment.
    """
    groups = []
    for group in draft.segment_marker_groups:
        if group.segment_name == segment_name:
            groups.append(
                C3dSegmentMarkerGroup(
                    group.segment_name,
                    tuple(name for name in group.marker_names if name != marker_name),
                )
            )
        else:
            groups.append(group)
    return _replace_draft_groups(draft, tuple(groups))


def add_virtual_marker_to_draft(
    draft: C3dWorkflowDraft,
    name: str,
    method: str,
    segment_name: str,
    source: str = "",
    equation: str = "",
) -> C3dWorkflowDraft:
    """
    Add or replace a virtual marker definition.
    """
    marker = C3dVirtualMarkerDraft(
        name=_require_name(name, "Virtual marker"),
        method=_require_name(method, "Virtual marker method"),
        segment_name=_require_name(segment_name, "Virtual marker segment"),
        source=source.strip(),
        equation=equation.strip(),
    )
    return C3dWorkflowDraft(
        preset=draft.preset,
        segment_marker_groups=draft.segment_marker_groups,
        virtual_markers=tuple(existing for existing in draft.virtual_markers if existing.name != marker.name)
        + (marker,),
        axes=draft.axes,
        segment_settings=draft.segment_settings,
        file_assignments=draft.file_assignments,
    )


def remove_virtual_marker_from_draft(draft: C3dWorkflowDraft, name: str) -> C3dWorkflowDraft:
    """
    Remove a virtual marker definition by name.
    """
    return C3dWorkflowDraft(
        preset=draft.preset,
        segment_marker_groups=draft.segment_marker_groups,
        virtual_markers=tuple(marker for marker in draft.virtual_markers if marker.name != name),
        axes=draft.axes,
        segment_settings=draft.segment_settings,
        file_assignments=draft.file_assignments,
    )


def add_axis_to_draft(
    draft: C3dWorkflowDraft,
    name: str,
    segment_name: str,
    axis: str,
    start_markers: tuple[str, ...],
    end_markers: tuple[str, ...],
    method: str = "markers",
) -> C3dWorkflowDraft:
    """
    Add or replace an axis definition.
    """
    axis_definition = C3dAxisDraft(
        name=_require_name(name, "Axis"),
        segment_name=_require_name(segment_name, "Axis segment"),
        axis=_require_name(axis, "Axis name"),
        start_markers=tuple(name.strip() for name in start_markers if name.strip() != ""),
        end_markers=tuple(name.strip() for name in end_markers if name.strip() != ""),
        method=_require_name(method, "Axis method"),
    )
    if len(axis_definition.start_markers) == 0 or len(axis_definition.end_markers) == 0:
        raise ValueError("An axis needs at least one start marker and one end marker.")
    return C3dWorkflowDraft(
        preset=draft.preset,
        segment_marker_groups=draft.segment_marker_groups,
        virtual_markers=draft.virtual_markers,
        axes=tuple(existing for existing in draft.axes if existing.name != axis_definition.name) + (axis_definition,),
        segment_settings=draft.segment_settings,
        file_assignments=draft.file_assignments,
    )


def remove_axis_from_draft(draft: C3dWorkflowDraft, name: str) -> C3dWorkflowDraft:
    """
    Remove an axis definition by name.
    """
    return C3dWorkflowDraft(
        preset=draft.preset,
        segment_marker_groups=draft.segment_marker_groups,
        virtual_markers=draft.virtual_markers,
        axes=tuple(axis for axis in draft.axes if axis.name != name),
        segment_settings=draft.segment_settings,
        file_assignments=draft.file_assignments,
    )


def update_segment_settings_in_draft(
    draft: C3dWorkflowDraft,
    segment_name: str,
    translations: str,
    rotations: str,
    q_min: tuple[float, ...] = (),
    q_max: tuple[float, ...] = (),
    child_translation: bool = False,
    initial_rotation_method: str = "identity",
    initial_rotation_source: str = "",
    initial_rotation_matrix: tuple[tuple[float, float, float], ...] | None = None,
) -> C3dWorkflowDraft:
    """
    Add or replace kinematic settings for one segment.
    """
    matrix = (
        C3dSegmentSettingsDraft.initial_rotation_matrix if initial_rotation_matrix is None else initial_rotation_matrix
    )
    setting = C3dSegmentSettingsDraft(
        segment_name=_require_name(segment_name, "Segment"),
        translations=translations.strip(),
        rotations=rotations.strip(),
        q_min=tuple(float(value) for value in q_min),
        q_max=tuple(float(value) for value in q_max),
        child_translation=child_translation,
        initial_rotation_method=_require_name(initial_rotation_method, "Initial rotation method"),
        initial_rotation_source=initial_rotation_source.strip(),
        initial_rotation_matrix=matrix,
    )
    return C3dWorkflowDraft(
        preset=draft.preset,
        segment_marker_groups=draft.segment_marker_groups,
        virtual_markers=draft.virtual_markers,
        axes=draft.axes,
        segment_settings=tuple(existing for existing in draft.segment_settings if existing.segment_name != segment_name)
        + (setting,),
        file_assignments=draft.file_assignments,
    )


def assign_c3d_file_role_to_draft(draft: C3dWorkflowDraft, role: str, source_path: str) -> C3dWorkflowDraft:
    """
    Assign a participant C3D file to a generic role.
    """
    role = _require_name(role, "C3D role")
    source_path = _require_name(source_path, "C3D source path")
    assignments = []
    found_role = False
    for assignment in draft.file_assignments:
        if assignment.role == role:
            found_role = True
            assignments.append(
                C3dFileAssignmentDraft(
                    role=assignment.role,
                    generic_name=assignment.generic_name,
                    source_path=source_path,
                )
            )
        else:
            assignments.append(assignment)
    if not found_role:
        raise ValueError(f"C3D role '{role}' does not exist.")
    return _replace_draft_file_assignments(draft, tuple(assignments))


def clear_c3d_file_role_from_draft(draft: C3dWorkflowDraft, role: str) -> C3dWorkflowDraft:
    """
    Clear a participant C3D file assignment.
    """
    assignments = tuple(
        C3dFileAssignmentDraft(assignment.role, assignment.generic_name) if assignment.role == role else assignment
        for assignment in draft.file_assignments
    )
    return _replace_draft_file_assignments(draft, assignments)


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


def c3d_template_payload_from_draft(draft: C3dWorkflowDraft) -> dict:
    """
    Return a serializable template payload from the current editable draft.
    """
    workflow = c3d_creation_workflow(draft.preset)
    return {
        "preset": draft.preset.value,
        "steps": [asdict(step) for step in workflow.steps],
        "c3d_file_roles": [asdict(role) for role in workflow.file_roles],
        "segment_marker_groups": [asdict(group) for group in draft.segment_marker_groups],
        "virtual_markers": [asdict(marker) for marker in draft.virtual_markers],
        "axes": [asdict(axis) for axis in draft.axes],
        "segment_settings": [asdict(setting) for setting in draft.segment_settings],
        "c3d_file_assignments": [asdict(assignment) for assignment in draft.file_assignments],
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


def _replace_draft_groups(
    draft: C3dWorkflowDraft,
    segment_marker_groups: tuple[C3dSegmentMarkerGroup, ...],
) -> C3dWorkflowDraft:
    return C3dWorkflowDraft(
        preset=draft.preset,
        segment_marker_groups=segment_marker_groups,
        virtual_markers=draft.virtual_markers,
        axes=draft.axes,
        segment_settings=draft.segment_settings,
        file_assignments=draft.file_assignments,
    )


def _replace_draft_file_assignments(
    draft: C3dWorkflowDraft,
    file_assignments: tuple[C3dFileAssignmentDraft, ...],
) -> C3dWorkflowDraft:
    return C3dWorkflowDraft(
        preset=draft.preset,
        segment_marker_groups=draft.segment_marker_groups,
        virtual_markers=draft.virtual_markers,
        axes=draft.axes,
        segment_settings=draft.segment_settings,
        file_assignments=file_assignments,
    )


def _require_name(value: str, label: str) -> str:
    value = value.strip()
    if value == "":
        raise ValueError(f"{label} cannot be empty.")
    return value


def _virtual_feature_default_method(feature_type: str, role: str) -> str:
    if feature_type == "axis":
        return "sara" if "axis" in role else "markers"
    if "legacy" in role:
        return "pointing_or_regression"
    if "axis" in role:
        return "functional_or_pointing"
    return "pointing"


def _initial_segment_settings(preset: C3dModelPreset) -> tuple[C3dSegmentSettingsDraft, ...]:
    if preset == C3dModelPreset.LOWER_LIMBS:
        return tuple(_settings_from_model_template(lower_limb_template()))
    if preset == C3dModelPreset.UPPER_LIMB:
        return tuple(_settings_from_model_template(upper_limb_template()))
    if preset == C3dModelPreset.FULL_BODY:
        return tuple(
            C3dSegmentSettingsDraft(
                segment_name=segment.name,
                translations=translations_from_matlab_dof(segment) or "",
                rotations=rotations_from_matlab_dof(segment) or "",
                child_translation=translations_from_matlab_dof(segment) is not None,
            )
            for segment in bela_segment_specs()
        )
    raise ValueError(f"Unsupported C3D model preset: {preset}.")


def _settings_from_model_template(template: ModelTemplate) -> tuple[C3dSegmentSettingsDraft, ...]:
    settings = []
    for segment in template.segments:
        settings.append(
            C3dSegmentSettingsDraft(
                segment_name=segment.name,
                translations="" if segment.translations.value is None else segment.translations.value,
                rotations="" if segment.rotations.value is None else segment.rotations.value,
                child_translation=segment.translations.value is not None,
            )
        )
    return tuple(settings)
