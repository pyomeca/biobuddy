"""
Prepare lightweight full-body Model202 C3D files for the GUI examples.

The source folder contains the original calibration C3D files. This script writes
small generic files in ``examples/data/full_body_model202`` by:

1. removing participant prefixes from marker names,
2. keeping only frames where all markers required by the corresponding model
   feature are valid,
3. keeping one valid frame out of ten,
4. removing analog channels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import ezc3d
import numpy as np

from biobuddy.gui.full_body_model202_template import (
    MODEL202_SEGMENTS,
    full_body_model202_functional_trials,
    full_body_model202_template,
)
from biobuddy.gui.model_builder import required_static_markers


@dataclass(frozen=True)
class C3dPreparationJob:
    source_name: str
    output_name: str
    required_markers: tuple[str, ...]


def full_body_model202_preparation_jobs() -> tuple[C3dPreparationJob, ...]:
    """
    Return every source/output pair needed by the full-body Model202 example.
    """
    template = full_body_model202_template(use_functional=True)
    jobs = [
        C3dPreparationJob("Anato.c3d", "Test_anato.c3d", required_static_markers(template)),
        C3dPreparationJob("BeLa_1.c3d", "Test_main.c3d", required_static_markers(template)),
    ]
    trial_by_segment_name = {trial.name: trial for trial in full_body_model202_functional_trials()}
    for segment_index, segment in enumerate(MODEL202_SEGMENTS, start=1):
        if segment.parent_name in {"", "base", "root"}:
            continue
        trial_name = (
            f"{segment.name.lower()}_{segment.parent_name.lower()}_{'sara' if segment.joint == 'aor' else 'score'}"
        )
        trial = trial_by_segment_name[trial_name]
        jobs.append(
            C3dPreparationJob(
                source_name=f"BeLa_{segment_index}.c3d",
                output_name=trial.file_pattern,
                required_markers=trial.required_markers,
            )
        )
    return tuple(jobs)


def prepare_full_body_model202_c3ds(
    source_folder: Path,
    output_folder: Path,
    subsample_step: int = 10,
) -> dict[str, tuple[int, int, int]]:
    """
    Prepare all C3D files and return frame-count metadata per output file.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    reports = {}
    for job in full_body_model202_preparation_jobs():
        source_path = source_folder / job.source_name
        output_path = output_folder / job.output_name
        reports[job.output_name] = prepare_c3d_file(
            source_path=source_path,
            output_path=output_path,
            required_markers=job.required_markers,
            subsample_step=subsample_step,
        )
    return reports


def prepare_c3d_file(
    source_path: Path,
    output_path: Path,
    required_markers: tuple[str, ...],
    subsample_step: int = 10,
) -> tuple[int, int, int]:
    """
    Write one lightweight C3D and return ``(input_frames, valid_frames, output_frames)``.
    """
    if subsample_step <= 0:
        raise ValueError("subsample_step must be a positive integer.")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    c3d = ezc3d.c3d(str(source_path))
    labels = tuple(_strip_participant_prefix(label) for label in c3d["parameters"]["POINT"]["LABELS"]["value"])
    required_indices = _required_marker_indices(labels, required_markers)
    input_frame_count = c3d["data"]["points"].shape[2]
    valid_frame_indices = _valid_frame_indices(c3d["data"]["points"], required_indices)
    if len(valid_frame_indices) == 0:
        raise RuntimeError(f"No valid frame found in {source_path.name} for required markers {required_markers}.")
    selected_frame_indices = valid_frame_indices[::subsample_step]
    _write_lightweight_c3d(c3d, labels, selected_frame_indices, subsample_step, output_path)
    return input_frame_count, len(valid_frame_indices), len(selected_frame_indices)


def _required_marker_indices(labels: tuple[str, ...], required_markers: tuple[str, ...]) -> tuple[int, ...]:
    missing_markers = tuple(marker for marker in required_markers if marker not in labels)
    if missing_markers:
        raise RuntimeError(f"Missing required markers in source C3D: {', '.join(missing_markers)}")
    return tuple(labels.index(marker) for marker in required_markers)


def _valid_frame_indices(points: np.ndarray, required_indices: tuple[int, ...]) -> np.ndarray:
    required_points = points[:, required_indices, :]
    finite_xyz = np.isfinite(required_points[:3]).all(axis=(0, 1))
    non_zero_xyz = (np.linalg.norm(required_points[:3], axis=0) > 0).all(axis=0)
    residual_is_valid = (required_points[3] >= 0).all(axis=0)
    return np.flatnonzero(finite_xyz & non_zero_xyz & residual_is_valid)


def _write_lightweight_c3d(
    c3d,
    labels: tuple[str, ...],
    selected_frame_indices: np.ndarray,
    subsample_step: int,
    output_path: Path,
) -> None:
    c3d["parameters"]["POINT"]["LABELS"]["value"] = list(labels)
    c3d["data"]["points"] = c3d["data"]["points"][:, :, selected_frame_indices].copy()
    c3d["data"]["analogs"] = np.zeros((1, 0, 0))
    c3d["parameters"]["POINT"]["FRAMES"]["value"] = [int(c3d["data"]["points"].shape[2])]
    c3d["parameters"]["POINT"]["RATE"]["value"] = [
        float(c3d["parameters"]["POINT"]["RATE"]["value"][0]) / subsample_step
    ]
    c3d["parameters"]["ANALOG"]["USED"]["value"] = [0]
    if "RATE" in c3d["parameters"]["ANALOG"]:
        c3d["parameters"]["ANALOG"]["RATE"]["value"] = [0.0]
    if "meta_points" in c3d["data"]:
        del c3d["data"]["meta_points"]
    temporary_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temporary_output_path.exists():
        temporary_output_path.unlink()
    c3d.write(str(temporary_output_path))
    temporary_output_path.replace(output_path)


def _strip_participant_prefix(label: str) -> str:
    return label.split(":", maxsplit=1)[-1]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-folder",
        type=Path,
        default=Path("/Users/mickaelbegon/Downloads/MODEL202"),
        help="Folder containing the original Model202 C3D files.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path(__file__).parent / "data" / "full_body_model202",
        help="Folder where lightweight example C3D files are written.",
    )
    parser.add_argument("--subsample-step", type=int, default=10, help="Keep one valid frame out of this many.")
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    reports = prepare_full_body_model202_c3ds(
        source_folder=args.source_folder,
        output_folder=args.output_folder,
        subsample_step=args.subsample_step,
    )
    for output_name, (input_frames, valid_frames, output_frames) in reports.items():
        print(f"{output_name}: {input_frames} source frames -> {valid_frames} valid -> {output_frames} saved")


if __name__ == "__main__":
    main()
