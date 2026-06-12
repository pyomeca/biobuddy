from dataclasses import dataclass

import numpy as np

from ..components.real.rigidbody.marker_real import MarkerReal
from ..components.real.rigidbody.segment_real import SegmentReal
from ..components.real.biomechanical_model_real import BiomechanicalModelReal


@dataclass
class MarkerEditorData:
    """
    Editable values exposed for a marker.
    """

    name: str
    position: list[float]
    is_technical: bool
    is_anatomical: bool


def get_marker_editor_data(marker: MarkerReal) -> MarkerEditorData:
    """
    Convert a marker into form-friendly values.
    """
    return MarkerEditorData(
        name=marker.name,
        position=marker.mean_position[:3].tolist(),
        is_technical=marker.is_technical,
        is_anatomical=marker.is_anatomical,
    )


def apply_marker_editor_data(marker: MarkerReal, data: MarkerEditorData) -> None:
    """
    Apply edited values to an existing marker.
    """
    marker.name = data.name
    marker.position = np.array(data.position)
    marker.is_technical = data.is_technical
    marker.is_anatomical = data.is_anatomical


def add_marker(segment: SegmentReal, data: MarkerEditorData) -> MarkerReal:
    """
    Create and attach a marker to a segment.
    """
    if data.name in segment.markers.keys():
        raise ValueError(f"Marker '{data.name}' already exists on segment '{segment.name}'.")
    marker = MarkerReal(
        name=data.name,
        parent_name=segment.name,
        position=np.array(data.position),
        is_technical=data.is_technical,
        is_anatomical=data.is_anatomical,
    )
    segment.add_marker(marker)
    return marker


def remove_marker(segment: SegmentReal, marker_name: str) -> None:
    """
    Remove a marker from a segment.
    """
    segment.remove_marker(marker_name)


def attach_marker_to_segment(
    model: BiomechanicalModelReal,
    source_segment_name: str,
    marker_name: str,
    target_segment_name: str,
) -> MarkerReal:
    """
    Attach an existing marker to another segment while preserving its global position.
    """
    if marker_name not in model.segments[source_segment_name].marker_names:
        raise ValueError(f"Marker '{marker_name}' does not exist on segment '{source_segment_name}'.")
    if marker_name in model.segments[target_segment_name].marker_names:
        raise ValueError(f"Marker '{marker_name}' already exists on segment '{target_segment_name}'.")

    source_marker = model.segments[source_segment_name].markers[marker_name]
    source_scs = model.segment_coordinate_system_in_global(source_segment_name)
    target_scs = model.segment_coordinate_system_in_global(target_segment_name)
    global_position = source_scs @ source_marker.position
    target_position = target_scs.inverse @ global_position
    marker = MarkerReal(
        name=source_marker.name,
        parent_name=target_segment_name,
        position=target_position,
        is_technical=source_marker.is_technical,
        is_anatomical=source_marker.is_anatomical,
    )
    model.segments[target_segment_name].add_marker(marker)
    return marker
