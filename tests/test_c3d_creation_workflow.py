import numpy as np

from biobuddy import DictData
from biobuddy.gui.c3d_creation_workflow import (
    c3d_creation_workflow,
    c3d_creation_workflow_steps,
    c3d_file_roles_for_preset,
    c3d_segment_marker_groups_for_preset,
    c3d_template_payload,
    c3d_workflow_summary,
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
