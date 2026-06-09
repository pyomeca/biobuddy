from biobuddy import (
    BiomechanicalModelReal,
)
from biobuddy.gui.c3d_creation_workflow import c3d_workflow_draft
from biobuddy.gui.c3d_model_creation import C3dModelPreset
from biobuddy.gui.model_editor import (
    _c3d_file_names_from_folder,
    _c3d_generation_log,
    _export_model_to_path,
    _joint_name_from_segments,
    _predictive_virtual_marker_method_from_label,
    _python_code_from_c3d_draft,
    _score_segments_from_payload,
    _split_marker_names,
    _strip_score_segment_payload,
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
