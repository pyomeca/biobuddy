"""
Small helpers for C3D model-creation source lists.

The GUI shows C3D markers, virtual markers, and virtual axes in the same source lists. Keeping the label formatting
and parsing here avoids scattering string conventions through the model editor widgets.
"""


def is_virtual_feature_axis(axis) -> bool:
    """
    Return whether an axis should appear as a virtual feature.
    """
    return axis.method == "sara" and axis.name.startswith("Axis_")


def anatomical_axis_source_labels(workflow_draft, marker_pool: tuple[str, ...]) -> tuple[str, ...]:
    """
    Return labels shown in anatomical and virtual-feature source lists.

    Anatomical frames often need a joint center or functional axis owned by a neighbouring segment, so the list uses
    all virtual features from the draft instead of filtering on the selected segment.
    """
    source_labels = list(dict.fromkeys(marker_pool))
    source_labels.extend(
        f"{marker.name} | virtual marker | {marker.segment_name}" for marker in workflow_draft.virtual_markers
    )
    source_labels.extend(
        f"[axis] {axis.name} | virtual axis | {axis.segment_name}"
        for axis in workflow_draft.axes
        if is_virtual_feature_axis(axis)
    )
    return tuple(dict.fromkeys(source_labels))


def axis_source_name_from_list_text(text: str) -> str:
    """
    Return the marker or axis name stored from a source-list label.
    """
    source_name = text.split("|", maxsplit=1)[0].strip()
    if source_name.startswith("[axis]"):
        source_name = source_name.removeprefix("[axis]").strip()
    return source_name
