from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from ..components.generic.rigidbody.axis import Axis
from ..utils.marker_data import DictData, MarkerData
from .model_builder import AxisSpec, MarkerEndpointSpec


class VirtualPointMethod(Enum):
    """
    Families of methods used to create points before model generation.
    """

    POINTING = "pointing"
    MARKER_MEAN = "marker_mean"
    GLOBAL_LINEAR_REGRESSION = "global_linear_regression"
    LOCAL_FRAME_REGRESSION = "local_frame_regression"
    SCORE = "score"
    SARA_AXIS = "sara_axis"


class VirtualAxisMethod(Enum):
    """
    Families of methods used to create axes before model generation.
    """

    POINT_PAIR = "point_pair"
    MARKER_PAIR = "marker_pair"
    LOCAL_FRAME_AXIS = "local_frame_axis"
    GLOBAL_VECTOR = "global_vector"
    SARA = "sara"


@dataclass(frozen=True)
class VirtualPointDefinition:
    """
    Definition of one point computed before model generation.

    The evaluator returns point positions with shape ``3 x nb_frames`` or
    ``4 x nb_frames``. The public ``evaluate`` method normalizes the result to
    homogeneous ``4 x nb_frames`` coordinates.
    """

    name: str
    method: VirtualPointMethod
    required_markers: tuple[str, ...]
    evaluator: Callable[[MarkerData], np.ndarray]
    description: str = ""

    def evaluate(self, data: MarkerData) -> np.ndarray:
        """
        Evaluate this virtual point on marker data.
        """
        missing_markers = sorted(set(self.required_markers) - set(data.marker_names))
        if missing_markers:
            raise ValueError(f"Virtual point '{self.name}' is missing markers: {', '.join(missing_markers)}")

        positions = np.asarray(self.evaluator(data), dtype=float)
        if positions.ndim == 1:
            positions = positions[:, np.newaxis]
        if positions.shape[0] == 3:
            homogeneous_positions = np.ones((4, positions.shape[1]))
            homogeneous_positions[:3, :] = positions
            return homogeneous_positions
        if positions.shape[0] == 4:
            return positions
        raise ValueError(
            f"Virtual point '{self.name}' should evaluate to shape 3 x frames or 4 x frames, "
            f"but got {positions.shape}."
        )


@dataclass(frozen=True)
class VirtualAxisDefinition:
    """
    Definition of one axis computed before model generation.

    The evaluator returns a pair ``(start, end)``. Each point may be ``3 x frames``
    or homogeneous ``4 x frames``; the public ``evaluate`` method normalizes both
    to homogeneous ``4 x frames`` coordinates.
    """

    name: str
    method: VirtualAxisMethod
    required_markers: tuple[str, ...]
    evaluator: Callable[[MarkerData], tuple[np.ndarray, np.ndarray]]
    description: str = ""

    def evaluate(self, data: MarkerData) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate this virtual axis on marker data.
        """
        missing_markers = sorted(set(self.required_markers) - set(data.marker_names))
        if missing_markers:
            raise ValueError(f"Virtual axis '{self.name}' is missing markers: {', '.join(missing_markers)}")
        start, end = self.evaluator(data)
        return _homogeneous_point(start, self.name), _homogeneous_point(end, self.name)

    def vector(self, data: MarkerData) -> np.ndarray:
        """
        Evaluate the axis vector over time.
        """
        start, end = self.evaluate(data)
        return end[:3, :] - start[:3, :]


def compute_virtual_points(
    data: MarkerData,
    definitions: tuple[VirtualPointDefinition, ...] | list[VirtualPointDefinition],
) -> dict[str, np.ndarray]:
    """
    Compute several virtual points from the same marker data.
    """
    return {definition.name: definition.evaluate(data) for definition in definitions}


def compute_virtual_axes(
    data: MarkerData,
    definitions: tuple[VirtualAxisDefinition, ...] | list[VirtualAxisDefinition],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute several virtual axes from the same marker data.
    """
    return {definition.name: definition.evaluate(data) for definition in definitions}


def marker_data_with_virtual_points(
    data: MarkerData,
    definitions: tuple[VirtualPointDefinition, ...] | list[VirtualPointDefinition],
) -> DictData:
    """
    Return a DictData copy containing original markers and computed virtual points.
    """
    return marker_data_with_virtual_features(data, point_definitions=definitions)


def marker_data_with_virtual_axes(
    data: MarkerData,
    definitions: tuple[VirtualAxisDefinition, ...] | list[VirtualAxisDefinition],
    suffixes: tuple[str, str] = ("_start", "_end"),
) -> DictData:
    """
    Return a DictData copy containing original markers and axis endpoints.
    """
    return marker_data_with_virtual_features(data, axis_definitions=definitions, axis_suffixes=suffixes)


def marker_data_with_virtual_features(
    data: MarkerData,
    point_definitions: tuple[VirtualPointDefinition, ...] | list[VirtualPointDefinition] = (),
    axis_definitions: tuple[VirtualAxisDefinition, ...] | list[VirtualAxisDefinition] = (),
    axis_suffixes: tuple[str, str] = ("_start", "_end"),
) -> DictData:
    """
    Return a DictData copy containing original markers, virtual points, and virtual axis endpoints.

    Virtual axes are stored as two regular marker-like points named
    ``<axis_name>_start`` and ``<axis_name>_end`` by default. Templates can then
    use these names exactly like C3D markers when defining a local frame.
    """
    marker_dict = {name: data.get_position([name])[:, 0, :] for name in data.marker_names}
    marker_dict.update(compute_virtual_points(data, point_definitions))
    for definition in axis_definitions:
        start, end = definition.evaluate(data)
        marker_dict[f"{definition.name}{axis_suffixes[0]}"] = start
        marker_dict[f"{definition.name}{axis_suffixes[1]}"] = end
    return DictData(marker_dict=marker_dict, first_frame=0, last_frame=data.nb_frames - 1)


def pointing_virtual_point(name: str, marker_name: str) -> VirtualPointDefinition:
    """
    Create a virtual point from a marker obtained by pointing/palpation.
    """
    return VirtualPointDefinition(
        name=name,
        method=VirtualPointMethod.POINTING,
        required_markers=(marker_name,),
        evaluator=lambda data: data.get_position([marker_name])[:3, 0, :],
        description=f"Pointing marker '{marker_name}' copied as '{name}'.",
    )


def point_pair_virtual_axis(
    name: str,
    start_point: VirtualPointDefinition,
    end_point: VirtualPointDefinition,
    description: str = "",
) -> VirtualAxisDefinition:
    """
    Create an axis from two virtual points.
    """
    required_markers = tuple(sorted(set(start_point.required_markers) | set(end_point.required_markers)))

    def evaluator(data: MarkerData) -> tuple[np.ndarray, np.ndarray]:
        return start_point.evaluate(data), end_point.evaluate(data)

    return VirtualAxisDefinition(
        name=name,
        method=VirtualAxisMethod.POINT_PAIR,
        required_markers=required_markers,
        evaluator=evaluator,
        description=description,
    )


def marker_pair_virtual_axis(
    name: str,
    start_markers: tuple[str, ...],
    end_markers: tuple[str, ...],
    description: str = "",
) -> VirtualAxisDefinition:
    """
    Create an axis from two marker groups.
    """
    start_point = marker_mean_virtual_point(f"{name}_start", start_markers)
    end_point = marker_mean_virtual_point(f"{name}_end", end_markers)
    return point_pair_virtual_axis(name=name, start_point=start_point, end_point=end_point, description=description)


def global_vector_virtual_axis(
    name: str,
    origin: VirtualPointDefinition,
    vector: tuple[float, float, float] | Callable[[MarkerData], np.ndarray],
    length: float = 1.0,
    extra_required_markers: tuple[str, ...] = (),
    description: str = "",
) -> VirtualAxisDefinition:
    """
    Create an axis from a virtual origin and a global vector.
    """
    required_markers = tuple(sorted(set(origin.required_markers) | set(extra_required_markers)))

    def evaluator(data: MarkerData) -> tuple[np.ndarray, np.ndarray]:
        start = origin.evaluate(data)
        raw_vector = vector(data) if callable(vector) else np.asarray(vector, dtype=float)[:, np.newaxis]
        axis_vector = _broadcast_vector(raw_vector, data.nb_frames)
        axis_vector = length * _normalize(axis_vector)
        end = np.ones_like(start)
        end[:3, :] = start[:3, :] + axis_vector
        return start, end

    return VirtualAxisDefinition(
        name=name,
        method=VirtualAxisMethod.GLOBAL_VECTOR,
        required_markers=required_markers,
        evaluator=evaluator,
        description=description,
    )


def local_frame_virtual_axis(
    name: str,
    origin: MarkerEndpointSpec,
    first_axis: AxisSpec,
    second_axis: AxisSpec,
    axis_to_keep: Axis.Name,
    local_direction: tuple[float, float, float] | Callable[[MarkerData], np.ndarray],
    length: float = 1.0,
    extra_required_markers: tuple[str, ...] = (),
    description: str = "",
) -> VirtualAxisDefinition:
    """
    Create an axis whose direction is expressed in a dynamic marker-defined frame.
    """
    required_markers = tuple(
        sorted(
            set(origin.marker_names)
            | set(first_axis.start.marker_names)
            | set(first_axis.end.marker_names)
            | set(second_axis.start.marker_names)
            | set(second_axis.end.marker_names)
            | set(extra_required_markers)
        )
    )

    def evaluator(data: MarkerData) -> tuple[np.ndarray, np.ndarray]:
        frame = _dynamic_local_frame(data, origin, first_axis, second_axis, axis_to_keep)
        direction = (
            local_direction(data)
            if callable(local_direction)
            else np.asarray(local_direction, dtype=float)[:, np.newaxis]
        )
        direction = _broadcast_vector(direction, data.nb_frames)
        direction = length * _normalize(direction)
        rotation = frame[:, :, :3]
        translation = frame[:, :, 3]
        start = np.ones((4, data.nb_frames))
        end = np.ones((4, data.nb_frames))
        start[:3, :] = translation.T
        end[:3, :] = (np.einsum("fij,jf->fi", rotation, direction) + translation).T
        return start, end

    return VirtualAxisDefinition(
        name=name,
        method=VirtualAxisMethod.LOCAL_FRAME_AXIS,
        required_markers=required_markers,
        evaluator=evaluator,
        description=description,
    )


def sara_virtual_axis_placeholder(
    name: str,
    parent_marker_names: tuple[str, ...],
    child_marker_names: tuple[str, ...],
    condyle_axis: VirtualAxisDefinition | None = None,
    description: str = "",
) -> VirtualAxisDefinition:
    """
    Reserve a SARA axis definition until the functional trial evaluator is wired.

    This object intentionally raises at evaluation time. It lets templates declare
    which markers and optional condyle direction will be needed before generation.
    """
    required_markers = set(parent_marker_names) | set(child_marker_names)
    if condyle_axis is not None:
        required_markers.update(condyle_axis.required_markers)

    def evaluator(data: MarkerData) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("SARA virtual axis evaluation requires functional trial data.")

    return VirtualAxisDefinition(
        name=name,
        method=VirtualAxisMethod.SARA,
        required_markers=tuple(sorted(required_markers)),
        evaluator=evaluator,
        description=description,
    )


def marker_mean_virtual_point(name: str, marker_names: tuple[str, ...]) -> VirtualPointDefinition:
    """
    Create a virtual point from the mean of several markers.
    """
    endpoint = MarkerEndpointSpec(marker_names)
    return VirtualPointDefinition(
        name=name,
        method=VirtualPointMethod.MARKER_MEAN,
        required_markers=marker_names,
        evaluator=endpoint.evaluate,
        description=f"Mean point of {', '.join(marker_names)}.",
    )


def global_linear_regression_virtual_point(
    name: str,
    marker_weights: dict[str, float],
    intercept: tuple[float, float, float] = (0.0, 0.0, 0.0),
    description: str = "",
) -> VirtualPointDefinition:
    """
    Create a point from a global linear combination of marker positions.

    This is useful for simple equations such as weighted landmarks. Predictive
    joint-center equations that use anatomical axes should usually use
    ``local_frame_regression_virtual_point`` instead.
    """
    required_markers = tuple(marker_weights)

    def evaluator(data: MarkerData) -> np.ndarray:
        positions = np.asarray(intercept, dtype=float)[:, np.newaxis] * np.ones((1, data.nb_frames))
        for marker_name, weight in marker_weights.items():
            positions += weight * data.get_position([marker_name])[:3, 0, :]
        return positions

    return VirtualPointDefinition(
        name=name,
        method=VirtualPointMethod.GLOBAL_LINEAR_REGRESSION,
        required_markers=required_markers,
        evaluator=evaluator,
        description=description,
    )


def local_frame_regression_virtual_point(
    name: str,
    origin: MarkerEndpointSpec,
    first_axis: AxisSpec,
    second_axis: AxisSpec,
    axis_to_keep: Axis.Name,
    local_offset: tuple[float, float, float] | Callable[[MarkerData], np.ndarray],
    extra_required_markers: tuple[str, ...] = (),
    description: str = "",
) -> VirtualPointDefinition:
    """
    Create a predictive point from offsets expressed in a marker-defined local frame.

    ``local_offset`` may be a constant ``(x, y, z)`` or a callable returning a
    ``3 x nb_frames`` array. The callable form supports regressions whose offsets
    depend on measurements such as pelvis width or shoulder width.
    """
    required_markers = tuple(
        sorted(
            set(origin.marker_names)
            | set(first_axis.start.marker_names)
            | set(first_axis.end.marker_names)
            | set(second_axis.start.marker_names)
            | set(second_axis.end.marker_names)
            | set(extra_required_markers)
        )
    )

    def evaluator(data: MarkerData) -> np.ndarray:
        frame = _dynamic_local_frame(data, origin, first_axis, second_axis, axis_to_keep)
        offset = local_offset(data) if callable(local_offset) else np.asarray(local_offset, dtype=float)[:, np.newaxis]
        if offset.ndim == 1:
            offset = offset[:, np.newaxis]
        if offset.shape[1] == 1 and data.nb_frames != 1:
            offset = np.repeat(offset, data.nb_frames, axis=1)
        rotation = frame[:, :, :3]
        translation = frame[:, :, 3]
        return (np.einsum("fij,jf->fi", rotation, offset) + translation).T

    return VirtualPointDefinition(
        name=name,
        method=VirtualPointMethod.LOCAL_FRAME_REGRESSION,
        required_markers=required_markers,
        evaluator=evaluator,
        description=description,
    )


def example_predictive_hip_cor(side: str) -> VirtualPointDefinition:
    """
    Return an example pelvis-regression hip center definition.

    The coefficients are illustrative placeholders expressed in the pelvis frame.
    They show where a lab-specific Bell/Harrington-like equation can be plugged in.
    """
    side = side.upper()
    if side not in {"D", "G"}:
        raise ValueError("Expected side 'D' or 'G'.")
    side_sign = 1.0 if side == "D" else -1.0
    origin = MarkerEndpointSpec(("EIASD", "EIASG", "EIPSD", "EIPSG"))
    first_axis = AxisSpec.from_markers(Axis.Name.X, "EIASG", "EIASD")
    second_axis = AxisSpec.from_markers(Axis.Name.Y, ("EIPSD", "EIPSG"), ("EIASD", "EIASG"))

    def offset(data: MarkerData) -> np.ndarray:
        pelvis_width = _distance(data, "EIASD", "EIASG")
        pelvis_depth = _distance_between_centers(data, ("EIASD", "EIASG"), ("EIPSD", "EIPSG"))
        return np.vstack((side_sign * 0.24 * pelvis_width, -0.30 * pelvis_depth, -0.33 * pelvis_width))

    return local_frame_regression_virtual_point(
        name=f"HipCoR{side}",
        origin=origin,
        first_axis=first_axis,
        second_axis=second_axis,
        axis_to_keep=Axis.Name.X,
        local_offset=offset,
        description="Example pelvis-width/depth predictive hip CoR. Replace coefficients with the chosen equation.",
    )


def example_predictive_shoulder_cor(side: str) -> VirtualPointDefinition:
    """
    Return an example shoulder-regression center definition.

    The coefficients are illustrative placeholders expressed in a scapula/acromion
    frame. They are meant as a wiring example, not as a prescribed clinical model.
    """
    side = side.upper()
    if side not in {"D", "G"}:
        raise ValueError("Expected side 'D' or 'G'.")
    side_sign = 1.0 if side == "D" else -1.0
    origin = MarkerEndpointSpec((f"ACRANT{side}", f"ACRPOST{side}"))
    first_axis = AxisSpec.from_markers(Axis.Name.X, f"CLAV1{side}", f"ACRANT{side}")
    second_axis = AxisSpec.from_markers(Axis.Name.Y, f"ACRPOST{side}", f"ACRANT{side}")

    def offset(data: MarkerData) -> np.ndarray:
        acromion_depth = _distance(data, f"ACRANT{side}", f"ACRPOST{side}")
        clavicle_to_acromion = _distance(data, f"CLAV1{side}", f"ACRANT{side}")
        return np.vstack((side_sign * 0.15 * clavicle_to_acromion, -0.50 * acromion_depth, -0.20 * acromion_depth))

    return local_frame_regression_virtual_point(
        name=f"ShoulderCoR{side}",
        origin=origin,
        first_axis=first_axis,
        second_axis=second_axis,
        axis_to_keep=Axis.Name.X,
        local_offset=offset,
        description="Example acromion/scapula predictive shoulder CoR. Replace coefficients with the chosen equation.",
    )


def _dynamic_local_frame(
    data: MarkerData,
    origin: MarkerEndpointSpec,
    first_axis: AxisSpec,
    second_axis: AxisSpec,
    axis_to_keep: Axis.Name,
) -> np.ndarray:
    first_name = first_axis.name
    second_name = second_axis.name
    third_name = _third_axis_name(first_name, second_name)
    first_vector = first_axis.vector(data)
    second_vector = second_axis.vector(data)
    third_vector = np.cross(first_vector, second_vector, axis=0)
    if axis_to_keep == first_name:
        second_vector = np.cross(third_vector, first_vector, axis=0)
    elif axis_to_keep == second_name:
        first_vector = np.cross(second_vector, third_vector, axis=0)
    else:
        raise ValueError("axis_to_keep must be one of the two defined axes.")

    frame = np.zeros((3, 4, data.nb_frames))
    frame[:, first_name, :] = _normalize(first_vector)
    frame[:, second_name, :] = _normalize(second_vector)
    frame[:, third_name, :] = _normalize(third_vector)
    frame[:, 3, :] = origin.evaluate(data)
    return np.moveaxis(frame, 2, 0)


def _third_axis_name(first_name: Axis.Name, second_name: Axis.Name) -> Axis.Name:
    missing_names = {Axis.Name.X, Axis.Name.Y, Axis.Name.Z} - {first_name, second_name}
    if len(missing_names) != 1:
        raise ValueError("A local frame must define two different axes.")
    return missing_names.pop()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=0)
    return np.divide(vector, norm, out=np.full_like(vector, np.nan, dtype=float), where=norm[np.newaxis, :] != 0)


def _homogeneous_point(point: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    if point.ndim == 1:
        point = point[:, np.newaxis]
    if point.shape[0] == 3:
        homogeneous_point = np.ones((4, point.shape[1]))
        homogeneous_point[:3, :] = point
        return homogeneous_point
    if point.shape[0] == 4:
        return point
    raise ValueError(f"Virtual axis '{name}' endpoint should have shape 3 x frames or 4 x frames, got {point.shape}.")


def _broadcast_vector(vector: np.ndarray, nb_frames: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.ndim == 1:
        vector = vector[:, np.newaxis]
    if vector.shape[0] != 3:
        raise ValueError(f"Expected a 3D vector, got shape {vector.shape}.")
    if vector.shape[1] == 1 and nb_frames != 1:
        return np.repeat(vector, nb_frames, axis=1)
    if vector.shape[1] != nb_frames:
        raise ValueError(f"Expected vector with 1 or {nb_frames} frames, got {vector.shape[1]}.")
    return vector


def _distance(data: MarkerData, start_marker: str, end_marker: str) -> np.ndarray:
    vector = data.get_position([end_marker])[:3, 0, :] - data.get_position([start_marker])[:3, 0, :]
    return np.linalg.norm(vector, axis=0)


def _distance_between_centers(
    data: MarkerData,
    start_markers: tuple[str, ...],
    end_markers: tuple[str, ...],
) -> np.ndarray:
    vector = (
        data.markers_center_position(list(end_markers))[:3, :]
        - data.markers_center_position(list(start_markers))[:3, :]
    )
    return np.linalg.norm(vector, axis=0)
