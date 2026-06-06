import numpy as np

from biobuddy import DictData
from biobuddy.gui.c3d_creation_workflow import (
    add_axis_to_draft,
    add_segment_to_draft,
    add_virtual_marker_to_draft,
    assign_c3d_file_role_to_draft,
    assign_marker_to_segment,
    c3d_creation_workflow,
    c3d_creation_workflow_steps,
    c3d_file_roles_for_preset,
    c3d_segment_marker_groups_for_preset,
    c3d_template_payload,
    c3d_template_payload_from_draft,
    c3d_workflow_draft,
    c3d_workflow_summary,
    clear_c3d_file_role_from_draft,
    remove_axis_from_draft,
    remove_segment_from_draft,
    remove_virtual_marker_from_draft,
    unassign_marker_from_segment,
    update_segment_settings_in_draft,
)
from biobuddy.gui.c3d_model_creation import C3dModelPreset


def test_c3d_creation_workflow_steps_match_expected_pipeline():
    steps = c3d_creation_workflow_steps()

    assert [step.name for step in steps] == [
        "New from C3D",
        "Choose a C3D file",
        "List markers",
        "Create segments",
        "Assign markers",
        "Create virtual markers",
        "Create axes",
        "Child translations",
        "Initial rotation",
        "DoF",
        "Generate template",
        "Rename C3Ds",
    ]


def test_c3d_file_roles_use_generic_names_for_three_presets():
    lower_limb_names = {role.generic_name for role in c3d_file_roles_for_preset(C3dModelPreset.LOWER_LIMBS)}
    upper_limb_names = {role.generic_name for role in c3d_file_roles_for_preset(C3dModelPreset.UPPER_LIMB)}
    full_body_names = {role.generic_name for role in c3d_file_roles_for_preset(C3dModelPreset.FULL_BODY)}

    assert "static_anatomical.c3d" in lower_limb_names
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

    draft = add_segment_to_draft(draft, "CustomSegment")
    draft = assign_marker_to_segment(draft, "CustomSegment", "CUSTOM1")
    draft = assign_marker_to_segment(draft, "CustomSegment", "CUSTOM1")

    custom_group = next(group for group in draft.segment_marker_groups if group.segment_name == "CustomSegment")
    assert custom_group.marker_names == ("CUSTOM1",)

    draft = unassign_marker_from_segment(draft, "CustomSegment", "CUSTOM1")
    custom_group = next(group for group in draft.segment_marker_groups if group.segment_name == "CustomSegment")
    assert custom_group.marker_names == ()

    draft = remove_segment_from_draft(draft, "CustomSegment")
    assert all(group.segment_name != "CustomSegment" for group in draft.segment_marker_groups)


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
    draft = assign_c3d_file_role_to_draft(draft, "static", "/tmp/subject01_static.c3d")

    payload = c3d_template_payload_from_draft(draft)

    lshank_settings = next(setting for setting in payload["segment_settings"] if setting["segment_name"] == "LShank")
    assert lshank_settings["translations"] == "z"
    assert lshank_settings["rotations"] == "xz"
    assert lshank_settings["child_translation"] is True
    assert lshank_settings["initial_rotation_method"] == "anatomical_c3d"
    assert any(
        assignment["role"] == "static" and assignment["source_path"] == "/tmp/subject01_static.c3d"
        for assignment in payload["c3d_file_assignments"]
    )

    draft = clear_c3d_file_role_from_draft(draft, "static")
    assert next(assignment for assignment in draft.file_assignments if assignment.role == "static").source_path == ""
