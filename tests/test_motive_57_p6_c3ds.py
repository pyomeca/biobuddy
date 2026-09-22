from pathlib import Path

import ezc3d

from biobuddy.gui.build_c3d_model import DEFAULT_P6_MOTIVE_C3D_FOLDER, build_model_from_c3d_cli
from examples.prepare_motive_57_p6_c3ds import motive_57_p6_preparation_jobs

EXPECTED_FRAME_COUNTS = {
    "Example_Static.c3d": 155,
    "Example_LHip.c3d": 601,
    "Example_LKnee.c3d": 376,
    "Example_LAnkle.c3d": 471,
    "Example_RHip.c3d": 530,
    "Example_RKnee.c3d": 357,
    "Example_RAnkle.c3d": 390,
}
IDENTIFYING_TEXT_TOKENS = ("Skeleton_001", "P6", "2026-06-30", "captury_models", "mickaelbegon")


def test_motive_57_p6_example_c3ds_are_lightweight_and_anonymized():
    job_names = {job.output_name for job in motive_57_p6_preparation_jobs()}

    assert set(EXPECTED_FRAME_COUNTS) == job_names
    assert {path.name for path in DEFAULT_P6_MOTIVE_C3D_FOLDER.glob("*.c3d")} == job_names

    for output_name, expected_frame_count in EXPECTED_FRAME_COUNTS.items():
        c3d = ezc3d.c3d(str(DEFAULT_P6_MOTIVE_C3D_FOLDER / output_name))
        labels = c3d["parameters"]["POINT"]["LABELS"]["value"]

        assert c3d["data"]["points"].shape == (4, 57, expected_frame_count)
        assert c3d["data"]["analogs"].shape == (1, 0, 0)
        assert c3d["parameters"]["POINT"]["RATE"]["value"] == [24.0]
        assert c3d["parameters"]["ANALOG"]["USED"]["value"] == [0]
        assert all(":" not in label for label in labels)
        assert all(not any(token in label for token in IDENTIFYING_TEXT_TOKENS) for label in labels)
        assert _identifying_parameter_values(c3d["parameters"]) == []


def test_motive_57_p6_example_can_build_a_model(tmp_path):
    output_path = build_model_from_c3d_cli(
        DEFAULT_P6_MOTIVE_C3D_FOLDER,
        preset="motive_57",
        output=tmp_path / "motive_57_p6.bioMod",
        with_mesh=False,
        quiet=True,
    )

    assert output_path == tmp_path / "motive_57_p6.bioMod"
    assert output_path.exists()
    text = output_path.read_text()
    assert "segment\tPelvis" in text
    assert "segment\tLThigh" in text
    assert "segment\tRShank" in text


def _identifying_parameter_values(parameters: dict) -> list[str]:
    hits = []
    for group in parameters.values():
        if not isinstance(group, dict):
            continue
        for parameter in group.values():
            if not isinstance(parameter, dict) or "value" not in parameter:
                continue
            values = parameter["value"]
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str) and any(token in value for token in IDENTIFYING_TEXT_TOKENS):
                    hits.append(value)
    return hits
