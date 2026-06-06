import numpy as np

from biobuddy import DictData
from biobuddy.gui.c3d_creation_workflow import (
    add_axis_to_draft,
    add_segment_to_draft,
    add_virtual_marker_to_draft,
    assign_c3d_file_role_to_draft,
    assign_markers_to_segment,
    assign_marker_to_segment,
    c3d_creation_workflow,
    c3d_creation_workflow_steps,
    c3d_file_roles_for_preset,
    c3d_segment_marker_groups_for_preset,
    c3d_template_payload,
    c3d_template_payload_from_draft,
    c3d_virtual_marker_method_examples,
    c3d_workflow_draft,
    c3d_workflow_progress,
    c3d_workflow_summary,
    clear_c3d_file_role_from_draft,
    remove_axis_from_draft,
    remove_segment_from_draft,
    remove_virtual_marker_from_draft,
    unassign_markers_from_segment,
    unassign_marker_from_segment,
    update_segment_settings_in_draft,
    validate_c3d_workflow_draft,
)
from biobuddy.gui.c3d_model_creation import C3dModelPreset


def test_c3d_creation_workflow_steps_match_expected_pipeline():
    steps = c3d_creation_workflow_steps()

    assert [step.name for step in steps] == [
        "New from C3D",
        "Choose main C3D",
        "Create technical segments",
        "Create virtual markers",
        "Create anatomical segments",
        "Segment coordinate systems",
        "Initial rotations",
        "DoF",
        "Parent-child transforms",
        "Validate",
        "Generate template",
        "Generate model",
    ]


def test_c3d_file_roles_use_generic_names_for_three_presets():
    lower_limb_names = {role.generic_name for role in c3d_file_roles_for_preset(C3dModelPreset.LOWER_LIMBS)}
    upper_limb_names = {role.generic_name for role in c3d_file_roles_for_preset(C3dModelPreset.UPPER_LIMB)}
    full_body_names = {role.generic_name for role in c3d_file_roles_for_preset(C3dModelPreset.FULL_BODY)}

    assert "main_markers.c3d" in lower_limb_names
    assert "functional_left_knee_sara.c3d" in lower_limb_names
    assert "pointing_virtual_markers.c3d" in upper_limb_names
    assert "functional_upper_limb_score_sara.c3d" in full_body_names


def test_c3d_segment_marker_groups_cover_three_models():
    lower_limb_groups = c3d_segment_marker_groups_for_preset(C3dModelPreset.LOWER_LIMBS)
    upper_limb_groups = c3d_segment_marker_groups_for_preset(C3dModelPreset.UPPER_LIMB)
    full_body_groups = c3d_segment_marker_groups_for_preset(C3dModelPreset.FULL_BODY)

    assert any(group.segment_name == "Pelvis" and "LASI" in group.marker_names for group in lower_limb_groups)
    assert any(group.segment_name == "Arm" and "EPICl" in group.marker_names for group in upper_limb_groups)
    assert any(group.segment_name == "Thorax" and "MANU" in group.marker_names for group in full_body_groups)
    assert any(group.segment_name == "Trunk" and group.parent_name == "Pelvis" for group in lower_limb_groups)
    assert all(group.segment_type == "anatomical" for group in full_body_groups)


def test_c3d_workflow_summary_reports_marker_counts_and_virtual_features():
    marker_dict = {}
    for marker_name in ("LASI", "RASI", "LPSI"):
        marker = np.ones((4, 2))
        marker_dict[marker_name] = marker
    data = DictData(marker_dict)

    summary = c3d_workflow_summary(C3dModelPreset.LOWER_LIMBS, data)

    assert "Preset: lower_limbs" in summary
    assert "C3D markers: 3" in summary
    assert "Missing expected markers:" in summary


def test_c3d_creation_workflow_collects_roles_and_segments():
    workflow = c3d_creation_workflow(C3dModelPreset.UPPER_LIMB)

    assert workflow.preset == C3dModelPreset.UPPER_LIMB
    assert len(workflow.steps) == 12
    assert any(role.role == "pointing" for role in workflow.file_roles)
    assert any(group.segment_name == "Scapula" for group in workflow.segment_marker_groups)


def test_c3d_template_payload_is_serializable_and_contains_virtual_features():
    payload = c3d_template_payload(C3dModelPreset.UPPER_LIMB)

    assert payload["preset"] == "upper_limb"
    assert len(payload["steps"]) == 12
    assert any(group["segment_name"] == "Arm" for group in payload["segment_marker_groups"])
    assert any(feature["name"] == "Thorax_virtual_7" for feature in payload["virtual_features"])


def test_c3d_workflow_draft_edits_segment_marker_assignments():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)

    draft = add_segment_to_draft(draft, "CustomSegment", parent_name="Pelvis", segment_type="technical")
    draft = assign_marker_to_segment(draft, "CustomSegment", "CUSTOM1")
    draft = assign_marker_to_segment(draft, "CustomSegment", "CUSTOM1")

    custom_group = next(group for group in draft.segment_marker_groups if group.segment_name == "CustomSegment")
    assert custom_group.marker_names == ("CUSTOM1",)
    assert custom_group.parent_name == "Pelvis"
    assert custom_group.segment_type == "technical"

    draft = unassign_marker_from_segment(draft, "CustomSegment", "CUSTOM1")
    custom_group = next(group for group in draft.segment_marker_groups if group.segment_name == "CustomSegment")
    assert custom_group.marker_names == ()

    draft = remove_segment_from_draft(draft, "CustomSegment")
    assert all(group.segment_name != "CustomSegment" for group in draft.segment_marker_groups)


def test_c3d_workflow_draft_edits_multiple_segment_marker_assignments():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    draft = add_segment_to_draft(draft, "CustomSegment")

    draft = assign_markers_to_segment(draft, "CustomSegment", ("CUSTOM1", "CUSTOM2", "CUSTOM1"))
    custom_group = next(group for group in draft.segment_marker_groups if group.segment_name == "CustomSegment")

    assert custom_group.marker_names == ("CUSTOM1", "CUSTOM2")

    draft = unassign_markers_from_segment(draft, "CustomSegment", ("CUSTOM1", "MISSING_MARKER"))
    custom_group = next(group for group in draft.segment_marker_groups if group.segment_name == "CustomSegment")

    assert custom_group.marker_names == ("CUSTOM2",)


def test_c3d_workflow_draft_edits_virtual_markers_and_axes():
    draft = c3d_workflow_draft(C3dModelPreset.UPPER_LIMB)

    draft = add_virtual_marker_to_draft(
        draft,
        name="ShoulderCoR",
        method="regression",
        segment_name="Arm",
        source="static_anatomical.c3d",
        equation="example_predictive_shoulder_cor",
    )
    draft = add_axis_to_draft(
        draft,
        name="ElbowFlexionAxis",
        segment_name="LowerArm1",
        axis="x",
        start_markers=("EPICm",),
        end_markers=("EPICl",),
        method="sara",
    )

    payload = c3d_template_payload_from_draft(draft)

    assert any(marker["name"] == "ShoulderCoR" for marker in payload["virtual_markers"])
    assert any(axis["name"] == "ElbowFlexionAxis" for axis in payload["axes"])

    draft = remove_virtual_marker_from_draft(draft, "ShoulderCoR")
    draft = remove_axis_from_draft(draft, "ElbowFlexionAxis")

    assert all(marker.name != "ShoulderCoR" for marker in draft.virtual_markers)
    assert all(axis.name != "ElbowFlexionAxis" for axis in draft.axes)


def test_c3d_workflow_draft_replaces_virtual_marker_and_axis_with_same_name():
    draft = c3d_workflow_draft(C3dModelPreset.UPPER_LIMB)
    draft = add_virtual_marker_to_draft(
        draft,
        name="ShoulderCoR",
        method="pointing",
        segment_name="Arm",
        source="pointing_virtual_markers.c3d",
    )
    draft = add_virtual_marker_to_draft(
        draft,
        name="ShoulderCoR",
        method="regression",
        segment_name="Thorax",
        source="static_anatomical.c3d",
        equation="example_predictive_shoulder_cor(D)",
    )
    draft = add_axis_to_draft(
        draft,
        name="ElbowFlexionAxis",
        segment_name="LowerArm1",
        axis="x",
        start_markers=("EPICm",),
        end_markers=("EPICl",),
    )
    draft = add_axis_to_draft(
        draft,
        name="ElbowFlexionAxis",
        segment_name="LowerArm1",
        axis="z",
        start_markers=("ELB_START",),
        end_markers=("ELB_END",),
        method="sara",
    )

    shoulder_markers = [marker for marker in draft.virtual_markers if marker.name == "ShoulderCoR"]
    elbow_axes = [axis for axis in draft.axes if axis.name == "ElbowFlexionAxis"]

    assert len(shoulder_markers) == 1
    assert shoulder_markers[0].method == "regression"
    assert shoulder_markers[0].segment_name == "Thorax"
    assert len(elbow_axes) == 1
    assert elbow_axes[0].axis == "z"
    assert elbow_axes[0].method == "sara"


def test_c3d_workflow_draft_edits_segment_settings_and_file_assignments():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)

    draft = update_segment_settings_in_draft(
        draft,
        segment_name="LShank",
        translations="z",
        rotations="xz",
        q_min=(-1.0, -0.5),
        q_max=(1.0, 0.5),
        child_translation=True,
        initial_rotation_method="anatomical_c3d",
        initial_rotation_source="static_anatomical.c3d",
    )
    draft = assign_c3d_file_role_to_draft(draft, "main", "/tmp/main_markers.c3d")

    payload = c3d_template_payload_from_draft(draft)

    lshank_settings = next(setting for setting in payload["segment_settings"] if setting["segment_name"] == "LShank")
    assert lshank_settings["translations"] == "z"
    assert lshank_settings["rotations"] == "xz"
    assert lshank_settings["child_translation"] is True
    assert lshank_settings["initial_rotation_method"] == "anatomical_c3d"
    assert any(
        assignment["role"] == "main" and assignment["source_path"] == "/tmp/main_markers.c3d"
        for assignment in payload["c3d_file_assignments"]
    )

    draft = clear_c3d_file_role_from_draft(draft, "main")
    assert next(assignment for assignment in draft.file_assignments if assignment.role == "main").source_path == ""


def test_c3d_workflow_draft_validation_reports_empty_custom_segment():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    draft = add_segment_to_draft(draft, "CustomSegment")

    issues = validate_c3d_workflow_draft(draft)

    assert any(
        issue.severity == "warning" and issue.category == "segments" and "CustomSegment" in issue.message
        for issue in issues
    )


def test_c3d_workflow_draft_validation_reports_technical_segment_requirements():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    draft = add_segment_to_draft(draft, "TechnicalFemur", parent_name="UnknownParent", segment_type="technical")
    draft = assign_marker_to_segment(draft, "TechnicalFemur", "THI1")

    issues = validate_c3d_workflow_draft(draft)

    assert any(issue.severity == "warning" and "at least 3 markers" in issue.message for issue in issues)
    assert any(issue.severity == "error" and "UnknownParent" in issue.message for issue in issues)


def test_c3d_workflow_draft_validation_reports_missing_c3d_markers():
    marker_dict = {"LASI": np.ones((4, 2)), "RASI": np.ones((4, 2))}
    data = DictData(marker_dict)
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    draft = add_segment_to_draft(draft, "CustomSegment")
    draft = assign_marker_to_segment(draft, "CustomSegment", "NOT_IN_C3D")

    issues = validate_c3d_workflow_draft(draft, data)

    assert any(issue.severity == "error" and "NOT_IN_C3D" in issue.message for issue in issues)


def test_c3d_workflow_draft_validation_reports_axis_marker_typo():
    draft = c3d_workflow_draft(C3dModelPreset.UPPER_LIMB)
    draft = add_axis_to_draft(
        draft,
        name="CustomAxis",
        segment_name="Arm",
        axis="x",
        start_markers=("KNOWN_START",),
        end_markers=("MISSPELLED_MARKER",),
    )
    draft = assign_marker_to_segment(draft, "Arm", "KNOWN_START")

    issues = validate_c3d_workflow_draft(draft)

    assert any(issue.category == "axes" and "MISSPELLED_MARKER" in issue.message for issue in issues)


def test_c3d_workflow_draft_validation_reports_dof_limit_length_mismatch():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    draft = update_segment_settings_in_draft(
        draft,
        segment_name="LShank",
        translations="z",
        rotations="xz",
        q_min=(-1.0, -0.5),
        q_max=(-1.0, -0.5, 0.5),
    )

    issues = validate_c3d_workflow_draft(draft)

    assert any(issue.severity == "error" and "q_min values for 3 DoF" in issue.message for issue in issues)


def test_c3d_template_payload_from_draft_includes_validation_issues():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)

    payload = c3d_template_payload_from_draft(draft)

    assert any(issue["category"] == "c3d files" for issue in payload["validation_issues"])


def test_c3d_workflow_progress_reports_pending_c3d_selection():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)

    progress = c3d_workflow_progress(draft)

    assert progress[0].status == "done"
    assert progress[1].name == "Choose main C3D"
    assert progress[1].status == "pending"


def test_c3d_workflow_progress_reports_loaded_markers_and_assigned_required_roles():
    marker_dict = {"LASI": np.ones((4, 2)), "RASI": np.ones((4, 2)), "LPSI": np.ones((4, 2))}
    data = DictData(marker_dict)
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    draft = assign_c3d_file_role_to_draft(draft, "main", "/tmp/main_markers.c3d")

    progress = c3d_workflow_progress(draft, data)
    progress_by_name = {step.name: step for step in progress}

    assert progress_by_name["Choose main C3D"].status == "done"
    assert progress_by_name["Generate model"].status == "done"


def test_c3d_virtual_marker_method_examples_document_regression_and_sara():
    examples = c3d_virtual_marker_method_examples()

    assert any(
        example.method == "regression" and "example_predictive_hip_cor" in example.equation_example
        for example in examples
    )
    assert any(
        example.method == "sara" and "functional_left_knee_sara.c3d" in example.source_example for example in examples
    )


def test_c3d_template_payload_from_draft_includes_progress_and_method_examples():
    draft = c3d_workflow_draft(C3dModelPreset.UPPER_LIMB)

    payload = c3d_template_payload_from_draft(draft)

    assert any(step["name"] == "Segment coordinate systems" for step in payload["progress"])
    assert any(example["method"] == "score" for example in payload["virtual_marker_method_examples"])
