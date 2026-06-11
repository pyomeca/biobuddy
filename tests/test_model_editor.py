import numpy as np

from biobuddy import (
    BiomechanicalModelReal,
)
from biobuddy.gui.c3d_creation_workflow import c3d_workflow_draft, remove_axis_from_draft
from biobuddy.gui.c3d_model_creation import C3dModelPreset
from biobuddy.gui.model_editor import (
    _c3d_file_names_from_folder,
    _c3d_generation_log,
    _export_model_to_path,
    _joint_name_from_segments,
    _marker_frame_position,
    _matching_c3d_file_for_expected_name,
    _marker_name_mapping_for_c3d,
    _predictive_virtual_marker_method_from_label,
    _python_code_from_c3d_draft,
    _remap_c3d_workflow_draft_markers,
    _axis_projection_axis_from_payload,
    _axis_projection_point_markers_from_payload,
    _strip_participant_prefix_from_c3d_data,
    _strip_participant_prefix_from_marker_names,
    _score_segments_from_payload,
    _split_marker_names,
    _strip_score_segment_payload,
    _source_with_c3d_assignment,
    _c3d_source_name_from_virtual_feature_source,
    _trial_name_from_virtual_feature_source,
    _virtual_axis_name_from_feature_list_text,
)
from biobuddy.gui.segment_editor import load_model


def test_load_model_supports_bvh(tmp_path):
    """
    Load a minimal BVH file through the GUI-facing loader.
    """
    filepath = tmp_path / "minimal.bvh"
    filepath.write_text(
        "\n".join(
            [
                "HIERARCHY",
                "ROOT Hips",
                "{",
                "    OFFSET 0 0 0",
                "    CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation",
                "}",
                "MOTION",
                "Frames: 1",
                "Frame Time: 0.01",
                "0 0 0 0 0 0",
            ]
        )
    )

    model = load_model(str(filepath))

    assert isinstance(model, BiomechanicalModelReal)
    assert "Hips" in model.segment_names


def test_virtual_marker_editor_payload_helpers_preserve_score_settings():
    """
    Parse the compact GUI payload used for SCoRE/SARA proximal and distal segment settings.
    """
    payload = "proximal=PelvisTech; distal=ThighTech; helper=condyles=ME,LE"

    assert _score_segments_from_payload(payload) == ("PelvisTech", "ThighTech")
    assert _strip_score_segment_payload(payload) == "condyles=ME,LE"
    assert _split_marker_names("LASI, RASI; LPSI") == ("LASI", "RASI", "LPSI")


def test_virtual_marker_editor_suggests_joint_names_from_segment_pairs():
    """
    Infer understandable virtual marker names from proximal/distal technical segment names.
    """
    assert _joint_name_from_segments("Pelvis", "LThigh") == "Left_Hip"
    assert _joint_name_from_segments("LThigh", "LShank") == "Left_Knee"


def test_virtual_marker_axis_list_text_can_remove_axis_from_draft():
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    axis_list_text = "[axis] Axis_LKnee_SARA | LShank | sara | trial=left_knee_sara"

    axis_name = _virtual_axis_name_from_feature_list_text(axis_list_text)
    updated_draft = remove_axis_from_draft(draft, axis_name)

    assert axis_name == "Axis_LKnee_SARA"
    assert _virtual_axis_name_from_feature_list_text("CoR_LThigh_in_Pelvis | LThigh | score") is None
    assert any(axis.name == "Axis_LKnee_SARA" for axis in draft.axes)
    assert all(axis.name != "Axis_LKnee_SARA" for axis in updated_draft.axes)


def test_axis_projection_payload_parses_point_and_axis_sources():
    point_markers = _axis_projection_point_markers_from_payload("point=LKNE,LKNEM")
    axis_reference, axis_start, axis_end = _axis_projection_axis_from_payload("axis=Axis_LKnee_SARA")
    marker_axis_reference, marker_axis_start, marker_axis_end = _axis_projection_axis_from_payload(
        "axis_start=LKNE,LKNEM; axis_end=LANK,LANKM"
    )

    assert point_markers == ("LKNE", "LKNEM")
    assert axis_reference == "Axis_LKnee_SARA"
    assert axis_start == ()
    assert axis_end == ()
    assert marker_axis_reference == ""
    assert marker_axis_start == ("LKNE", "LKNEM")
    assert marker_axis_end == ("LANK", "LANKM")


def test_c3d_file_names_from_folder_lists_available_c3d_files(tmp_path):
    """
    The virtual marker GUI should offer C3D files from the selected pipeline folder.
    """
    (tmp_path / "left_hip_functional.c3d").write_text("")
    (tmp_path / "notes.txt").write_text("")
    (tmp_path / "right_hip_functional.c3d").write_text("")

    assert _c3d_file_names_from_folder(str(tmp_path)) == (
        "left_hip_functional.c3d",
        "right_hip_functional.c3d",
    )


def test_c3d_file_matching_accepts_template_names_and_functional_patterns(tmp_path):
    (tmp_path / "Test_func_anat.c3d").write_text("")
    (tmp_path / "Test_func_lknee.c3d").write_text("")

    assert _matching_c3d_file_for_expected_name(str(tmp_path), "Test_func_anat.c3d").name == "Test_func_anat.c3d"
    assert _matching_c3d_file_for_expected_name(str(tmp_path), "*func_lknee.c3d").name == "Test_func_lknee.c3d"
    assert _matching_c3d_file_for_expected_name(str(tmp_path), "missing.c3d") is None


def test_virtual_feature_source_keeps_trial_metadata_with_c3d_assignment():
    source = "trial=left_knee_sara; parent markers=LTIBD,LTIB,LTIBF"
    assigned = _source_with_c3d_assignment(source, "/tmp/Test_func_lknee.c3d")

    assert _trial_name_from_virtual_feature_source(assigned) == "left_knee_sara"
    assert _c3d_source_name_from_virtual_feature_source(assigned) == "Test_func_lknee.c3d"
    assert "parent markers=LTIBD,LTIB,LTIBF" in assigned


def test_marker_name_mapping_matches_normalized_c3d_names():
    """
    Template marker names can be automatically matched to participant-specific C3D naming.
    """
    mapping = _marker_name_mapping_for_c3d(("LASI", "RASI", "LTHIB"), ("L_ASI", "rasi", "L-THIB", "extra"))

    assert mapping == {"LASI": "L_ASI", "RASI": "rasi", "LTHIB": "L-THIB"}


def test_marker_name_mapping_matches_participant_prefixed_c3d_names():
    """
    C3D participant namespaces should not prevent matching markers to a template.
    """
    mapping = _marker_name_mapping_for_c3d(("S3", "T6", "C2"), ("P01_MH:S3", "P01_MH:T6", "P01_MH:C2"))

    assert mapping == {"S3": "P01_MH:S3", "T6": "P01_MH:T6", "C2": "P01_MH:C2"}


def test_strip_participant_prefix_from_marker_names():
    """
    Users can remove C3D participant prefixes such as P01_MH: from marker names.
    """
    assert _strip_participant_prefix_from_marker_names(("P01_MH:S3", "P01_MH:T6", "LASI")) == ("S3", "T6", "LASI")


def test_strip_participant_prefix_from_c3d_data_changes_marker_names_in_place():
    """
    The GUI strips marker names on loaded C3D data while preserving marker order.
    """

    class FakeC3dData:
        marker_names = ["P01_MH:S3", "P01_MH:T6"]

    data = FakeC3dData()
    _strip_participant_prefix_from_c3d_data(data)

    assert data.marker_names == ["S3", "T6"]


def test_remap_c3d_workflow_draft_markers_updates_segment_groups():
    """
    Loading a C3D should update template marker references before the user edits segment assignments.
    """
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    mapping = {"LASI": "L_ASI", "RASI": "R_ASI"}
    updated = _remap_c3d_workflow_draft_markers(draft, mapping)
    pelvis = next(group for group in updated.segment_marker_groups if group.segment_name == "Pelvis")

    assert pelvis.marker_names[:2] == ("LPSI", "RPSI")
    assert "L_ASI" in pelvis.marker_names
    assert "R_ASI" in pelvis.marker_names


def test_marker_frame_position_reads_selected_frame():
    """
    The technical segment preview should display marker coordinates at the slider frame.
    """

    class FakeC3dData:
        marker_names = ["LASI"]
        nb_frames = 2

        def get_position(self, marker_names):
            values = np.ones((4, 1, 2))
            values[:3, 0, 0] = (1.0, 2.0, 3.0)
            values[:3, 0, 1] = (4.0, 5.0, 6.0)
            return values

    assert _marker_frame_position(FakeC3dData(), "LASI", 1) == (4.0, 5.0, 6.0)


def test_c3d_generation_log_reports_virtual_marker_local_offset_context():
    """
    Keep a trace that SCoRE/SARA markers are global markers with local offsets reserved for model construction.
    """
    draft = c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS)
    lines = _c3d_generation_log(draft, None, "/tmp/c3d", ("LASI", "RASI"))

    assert any("Preset: lower_limbs" in line for line in lines)
    assert any("Virtual markers:" in line for line in lines)
    assert any("global marker added to marker pool" in line for line in lines)


def test_predictive_virtual_marker_method_label_maps_to_internal_key():
    """
    Keep readable predictive labels in the GUI while storing explicit method names in the draft.
    """
    assert _predictive_virtual_marker_method_from_label("Hara 2016 hip") == "hara2016_hip"
    assert _predictive_virtual_marker_method_from_label("harrington2007_hip") == "harrington2007_hip"


def test_python_code_from_c3d_draft_serializes_preset_value():
    """
    The generated script must contain editable JSON-like data, not Python enum reprs.
    """
    code = _python_code_from_c3d_draft(c3d_workflow_draft(C3dModelPreset.LOWER_LIMBS), "/tmp/c3d")

    assert "C3dModelPreset" not in code
    assert '"preset": "lower_limbs"' in code
    assert "C3D_FOLDER = '/tmp/c3d'" in code


def test_export_model_to_path_dispatches_by_extension(tmp_path):
    """
    The model editor export button should use every writer supported by BioBuddy.
    """
    calls = []

    class FakeModel:
        def to_biomod(self, filepath):
            calls.append(("biomod", filepath))

        def to_osim(self, filepath):
            calls.append(("osim", filepath))

        def to_urdf(self, filepath):
            calls.append(("urdf", filepath))

        def to_bvh(self, filepath):
            calls.append(("bvh", filepath))

    model = FakeModel()
    for extension in (".bioMod", ".osim", ".urdf", ".bvh"):
        _export_model_to_path(model, str(tmp_path / f"model{extension}"))

    assert [call[0] for call in calls] == ["biomod", "osim", "urdf", "bvh"]
