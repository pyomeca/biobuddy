from biobuddy import (
    BiomechanicalModelReal,
)
from biobuddy.gui.model_editor import _score_segments_from_payload, _split_marker_names, _strip_score_segment_payload
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
