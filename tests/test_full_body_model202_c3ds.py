from pathlib import Path

import ezc3d
import numpy as np

from examples.prepare_full_body_model202_c3ds import full_body_model202_preparation_jobs

EXAMPLE_FOLDER = Path(__file__).parent.parent / "examples" / "data" / "full_body_model202"


def test_full_body_model202_example_c3ds_are_filtered_and_lightweight():
    for job in full_body_model202_preparation_jobs():
        filepath = EXAMPLE_FOLDER / job.output_name
        c3d = ezc3d.c3d(str(filepath))
        labels = tuple(c3d["parameters"]["POINT"]["LABELS"]["value"])
        required_indices = tuple(labels.index(marker_name) for marker_name in job.required_markers)
        required_points = c3d["data"]["points"][:, required_indices, :]

        assert filepath.exists()
        assert c3d["data"]["points"].shape[2] > 0
        assert c3d["data"]["analogs"].shape == (1, 0, 0)
        assert c3d["parameters"]["ANALOG"]["USED"]["value"] == [0]
        assert c3d["parameters"]["POINT"]["RATE"]["value"] == [20.0]
        assert all(":" not in label for label in labels)
        assert np.isfinite(required_points[:3]).all()
        assert (np.linalg.norm(required_points[:3], axis=0) > 0).all()
        assert (required_points[3] >= 0).all()


def test_full_body_model202_example_frame_counts_match_valid_frame_subsampling():
    frame_counts = {
        job.output_name: ezc3d.c3d(str(EXAMPLE_FOLDER / job.output_name))["data"]["points"].shape[2]
        for job in full_body_model202_preparation_jobs()
    }

    assert frame_counts["Test_anato.c3d"] == 51
    assert frame_counts["Test_main.c3d"] == 68
    assert frame_counts["Test_func_thorax_pelvis.c3d"] == 730
    assert frame_counts["Test_func_left_shank_left_thigh.c3d"] == 42
