from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.linear_algebra import RotoTransMatrixTimeSeries


@dataclass(frozen=True)
class FunctionalFrameSelectionOptions:
    """
    Options used to remove redundant functional calibration frames.
    """

    enabled: bool = False
    max_frames: int = 200
    min_rotation_degrees: float = 2.0
    min_translation: float = 0.005
    rotation_weight: float = 1.0
    translation_weight: float = 0.3
    manual_frame_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class FunctionalFrameSelectionReport:
    """
    Summary of frame validity and diversity for one functional calibration trial.
    """

    total_frames: int
    valid_rt_frames: int
    selected_frames: int
    selected_indices: tuple[int, ...]
    rotation_range_degrees: float
    translation_range: float
    options: FunctionalFrameSelectionOptions


def prepare_functional_rt_pair(
    rt_parent: RotoTransMatrixTimeSeries,
    rt_child: RotoTransMatrixTimeSeries,
    options: FunctionalFrameSelectionOptions | None = None,
) -> tuple[RotoTransMatrixTimeSeries, RotoTransMatrixTimeSeries, FunctionalFrameSelectionReport]:
    """
    Optionally keep selected valid frames from a parent/child functional rototranslation pair.
    """
    options = FunctionalFrameSelectionOptions() if options is None else options
    report = functional_frame_selection_report(rt_parent, rt_child, options)
    if not options.enabled and len(options.manual_frame_indices) == 0:
        return rt_parent, rt_child, report
    if len(report.selected_indices) == 0:
        return rt_parent, rt_child, report
    parent_subset, child_subset = subset_rt_pair(rt_parent, rt_child, report.selected_indices)
    return parent_subset, child_subset, report


def functional_frame_selection_report(
    rt_parent: RotoTransMatrixTimeSeries,
    rt_child: RotoTransMatrixTimeSeries,
    options: FunctionalFrameSelectionOptions | None = None,
) -> FunctionalFrameSelectionReport:
    """
    Return frame validity and relative-motion diversity metrics.
    """
    options = FunctionalFrameSelectionOptions() if options is None else options
    total_frames = min(len(rt_parent), len(rt_child))
    valid_indices = _valid_rt_indices(rt_parent, rt_child)
    rotation_range, translation_range = _relative_motion_ranges(rt_parent, rt_child, valid_indices)
    if len(options.manual_frame_indices) != 0:
        manual_indices = {int(index) for index in options.manual_frame_indices}
        selected_indices = tuple(index for index in valid_indices if index in manual_indices)
    elif options.enabled:
        selected_indices = _select_diverse_indices(rt_parent, rt_child, valid_indices, options)
    else:
        selected_indices = tuple(int(index) for index in valid_indices)
    return FunctionalFrameSelectionReport(
        total_frames=total_frames,
        valid_rt_frames=len(valid_indices),
        selected_frames=len(selected_indices),
        selected_indices=selected_indices,
        rotation_range_degrees=rotation_range,
        translation_range=translation_range,
        options=options,
    )


def subset_rt_pair(
    rt_parent: RotoTransMatrixTimeSeries,
    rt_child: RotoTransMatrixTimeSeries,
    indices: tuple[int, ...] | list[int] | np.ndarray,
) -> tuple[RotoTransMatrixTimeSeries, RotoTransMatrixTimeSeries]:
    """
    Return parent and child RT time series restricted to the provided original frame indices.
    """
    indices = tuple(int(index) for index in indices)
    parent_array = rt_parent.to_numpy()[:, :, indices]
    child_array = rt_child.to_numpy()[:, :, indices]
    return (
        RotoTransMatrixTimeSeries.from_closest_rt_matrix(parent_array),
        RotoTransMatrixTimeSeries.from_closest_rt_matrix(child_array),
    )


def subset_points_by_frame(points: np.ndarray, indices: tuple[int, ...] | list[int] | np.ndarray) -> np.ndarray:
    """
    Return 3D/4D point trajectories restricted to selected original frame indices.
    """
    if points is None:
        return points
    points = np.asarray(points)
    if points.ndim != 2:
        return points
    if points.shape[1] == 1:
        return points
    return points[:, tuple(int(index) for index in indices)]


def format_functional_frame_report(report: FunctionalFrameSelectionReport, unit_scale: float = 1000.0) -> str:
    """
    Return a compact readable summary for logs and UI labels.
    """
    if len(report.options.manual_frame_indices) != 0:
        selected = f"{report.selected_frames} selected manual frames"
    elif report.options.enabled:
        selected = f"{report.selected_frames} selected diverse frames"
    else:
        selected = f"{report.selected_frames} candidate frames"
    translation = report.translation_range * unit_scale
    return (
        f"{report.total_frames} total frames; {report.valid_rt_frames} valid rigid frames; {selected}; "
        f"relative rotation range={report.rotation_range_degrees:.1f} deg; "
        f"relative translation range={translation:.1f} mm"
    )


def _valid_rt_indices(rt_parent: RotoTransMatrixTimeSeries, rt_child: RotoTransMatrixTimeSeries) -> tuple[int, ...]:
    parent = rt_parent.to_numpy()
    child = rt_child.to_numpy()
    frame_count = min(parent.shape[2], child.shape[2])
    valid = np.isfinite(parent[:, :, :frame_count]).all(axis=(0, 1)) & np.isfinite(child[:, :, :frame_count]).all(
        axis=(0, 1)
    )
    return tuple(int(index) for index in np.flatnonzero(valid))


def _select_diverse_indices(
    rt_parent: RotoTransMatrixTimeSeries,
    rt_child: RotoTransMatrixTimeSeries,
    valid_indices: tuple[int, ...],
    options: FunctionalFrameSelectionOptions,
) -> tuple[int, ...]:
    if len(valid_indices) <= 1:
        return valid_indices
    valid_indices_array = np.asarray(valid_indices, dtype=int)
    relative_transforms = _relative_transforms(rt_parent, rt_child, valid_indices)
    selected_positions = [0]
    is_selected = np.zeros((len(valid_indices),), dtype=bool)
    is_selected[0] = True
    translation_scale = _translation_scale(rt_parent, rt_child, valid_indices)
    max_frames = max(1, min(options.max_frames, len(valid_indices)))
    min_rotation = np.deg2rad(max(options.min_rotation_degrees, 0.0))
    min_translation = max(options.min_translation, 0.0)

    while np.any(~is_selected) and len(selected_positions) < max_frames:
        candidate_positions = np.flatnonzero(~is_selected)
        candidate_transforms = relative_transforms[candidate_positions]
        nearest_rotation = np.full((len(candidate_positions),), np.inf)
        nearest_translation = np.full((len(candidate_positions),), np.inf)
        for selected_position in selected_positions:
            selected_transform = relative_transforms[selected_position]
            rotations = _rotation_distance_to_matrix(candidate_transforms[:, :3, :3], selected_transform[:3, :3])
            translations = np.linalg.norm(candidate_transforms[:, :3, 3] - selected_transform[:3, 3], axis=1)
            nearest_rotation = np.minimum(nearest_rotation, rotations)
            nearest_translation = np.minimum(nearest_translation, translations)
        scores = options.rotation_weight * nearest_rotation + options.translation_weight * (
            nearest_translation / translation_scale
        )
        best_local_index = int(np.nanargmax(scores))
        best_position = int(candidate_positions[best_local_index])
        best_rotation = float(nearest_rotation[best_local_index])
        best_translation = float(nearest_translation[best_local_index])
        if not np.isfinite(scores[best_local_index]):
            break
        if best_rotation < min_rotation and best_translation < min_translation:
            break
        selected_positions.append(best_position)
        is_selected[best_position] = True

    return tuple(sorted(int(index) for index in valid_indices_array[selected_positions]))


def _relative_motion_ranges(
    rt_parent: RotoTransMatrixTimeSeries,
    rt_child: RotoTransMatrixTimeSeries,
    indices: tuple[int, ...],
) -> tuple[float, float]:
    if len(indices) <= 1:
        return 0.0, 0.0
    transforms = _relative_transforms(rt_parent, rt_child, indices)
    reference = transforms[0]
    rotations = _rotation_distance_to_matrix(transforms[1:, :3, :3], reference[:3, :3])
    translations = np.linalg.norm(transforms[1:, :3, 3] - reference[:3, 3], axis=1)
    return float(np.rad2deg(np.nanmax(rotations))), float(np.nanmax(translations))


def _translation_scale(
    rt_parent: RotoTransMatrixTimeSeries,
    rt_child: RotoTransMatrixTimeSeries,
    indices: tuple[int, ...],
) -> float:
    translations = [np.linalg.norm(_relative_transform(rt_parent, rt_child, index)[:3, 3]) for index in indices]
    scale = float(np.nanmedian(translations))
    return scale if np.isfinite(scale) and scale > 1e-9 else 1.0


def _relative_transform(
    rt_parent: RotoTransMatrixTimeSeries,
    rt_child: RotoTransMatrixTimeSeries,
    index: int,
) -> np.ndarray:
    return np.linalg.inv(rt_parent[index].rt_matrix) @ rt_child[index].rt_matrix


def _relative_transforms(
    rt_parent: RotoTransMatrixTimeSeries,
    rt_child: RotoTransMatrixTimeSeries,
    indices: tuple[int, ...],
) -> np.ndarray:
    parent = np.moveaxis(rt_parent.to_numpy()[:, :, indices], 2, 0)
    child = np.moveaxis(rt_child.to_numpy()[:, :, indices], 2, 0)
    return np.linalg.inv(parent) @ child


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    value = (np.trace(relative) - 1.0) / 2.0
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def _rotation_distance_to_matrix(rotations: np.ndarray, reference: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(reference, 0, 1) @ rotations
    traces = np.trace(relative, axis1=1, axis2=2)
    values = (traces - 1.0) / 2.0
    return np.arccos(np.clip(values, -1.0, 1.0))
