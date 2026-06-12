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
    AXIS_PROJECTION = "axis_projection"
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


def axis_projection_virtual_point(
    name: str,
    point: VirtualPointDefinition,
    axis: VirtualAxisDefinition,
    description: str = "",
) -> VirtualPointDefinition:
    """
    Project a point onto an axis.

    The projected point is computed independently at each frame. ``point`` may be a raw marker copy, a marker mean, or
    any other virtual point. ``axis`` may come from marker groups or from a functional axis definition once that axis is
    evaluable in the current marker data.
    """
    required_markers = tuple(sorted(set(point.required_markers) | set(axis.required_markers)))

    def evaluator(data: MarkerData) -> np.ndarray:
        point_position = point.evaluate(data)[:3, :]
        axis_start, axis_end = axis.evaluate(data)
        axis_start = axis_start[:3, :]
        axis_vector = axis_end[:3, :] - axis_start
        axis_unit = _normalize(axis_vector)
        distance_along_axis = np.sum((point_position - axis_start) * axis_unit, axis=0, keepdims=True)
        return axis_start + axis_unit * distance_along_axis

    return VirtualPointDefinition(
        name=name,
        method=VirtualPointMethod.AXIS_PROJECTION,
        required_markers=required_markers,
        evaluator=evaluator,
        description=description,
    )


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


def hara2016_hip_center_local(leg_length_mm: float, side: str) -> np.ndarray:
    """
    Return Hara et al. 2016 hip joint center coordinates in a pelvis frame.

    The pelvis frame convention is x anterior, y left, z superior, in millimeters.
    """
    side = _normalized_side(side)
    leg_length_mm = float(leg_length_mm)
    if leg_length_mm <= 0:
        raise ValueError("leg_length_mm must be positive.")
    x_anterior = 11.0 - 0.063 * leg_length_mm
    y_lateral = 8.0 + 0.086 * leg_length_mm
    z_superior = -9.0 - 0.078 * leg_length_mm
    y_left = -y_lateral if side == "right" else y_lateral
    return np.array([x_anterior, y_left, z_superior], dtype=float)


def harrington2007_hip_center_local_hara_axes(
    pelvic_width_mm: np.ndarray | float,
    pelvic_depth_mm: np.ndarray | float,
    leg_length_mm: float,
    side: str,
) -> np.ndarray:
    """
    Return a Harrington et al. 2007 hip center variant in Hara-like pelvis axes.

    The axes are x anterior, y left, z superior, in millimeters. Axis/sign conventions
    should be validated against the lab's reference implementation before clinical use.
    """
    side = _normalized_side(side)
    pelvic_width_mm = np.asarray(pelvic_width_mm, dtype=float)
    pelvic_depth_mm = np.asarray(pelvic_depth_mm, dtype=float)
    leg_length_mm = float(leg_length_mm)
    if np.any(pelvic_width_mm <= 0) or np.any(pelvic_depth_mm <= 0) or leg_length_mm <= 0:
        raise ValueError("pelvic_width_mm, pelvic_depth_mm and leg_length_mm must be positive.")
    x_anterior = -9.9 - 0.24 * pelvic_depth_mm
    y_lateral = 7.9 + 0.16 * pelvic_width_mm + 0.28 * pelvic_depth_mm
    z_superior = -7.1 - 0.16 * pelvic_width_mm - 0.04 * leg_length_mm
    y_left = -y_lateral if side == "right" else y_lateral
    return np.vstack((x_anterior, y_left, z_superior))


def predictive_hara2016_hip_cor(
    name: str,
    side: str,
    leg_length_mm: float,
    right_asis: str = "RASI",
    left_asis: str = "LASI",
    right_psis: str = "RPSI",
    left_psis: str = "LPSI",
) -> VirtualPointDefinition:
    """
    Create a Hara 2016 predictive hip center from ASIS/PSIS pelvis landmarks.
    """
    required_markers = (right_asis, left_asis, right_psis, left_psis)
    local_offset = hara2016_hip_center_local(leg_length_mm, side)

    def evaluator(data: MarkerData) -> np.ndarray:
        origin, rotation, _width, _depth = _pelvis_frame_from_asis_psis(
            data,
            right_asis=right_asis,
            left_asis=left_asis,
            right_psis=right_psis,
            left_psis=left_psis,
        )
        return _local_offset_to_global(origin, rotation, local_offset[:, np.newaxis])

    return VirtualPointDefinition(
        name=name,
        method=VirtualPointMethod.LOCAL_FRAME_REGRESSION,
        required_markers=required_markers,
        evaluator=evaluator,
        description="Hara et al. 2016 predictive hip center in a pelvis ASIS/PSIS frame.",
    )


def predictive_harrington2007_hip_cor(
    name: str,
    side: str,
    leg_length_mm: float,
    right_asis: str = "RASI",
    left_asis: str = "LASI",
    right_psis: str = "RPSI",
    left_psis: str = "LPSI",
) -> VirtualPointDefinition:
    """
    Create a Harrington 2007 predictive hip center from ASIS/PSIS pelvis landmarks.
    """
    required_markers = (right_asis, left_asis, right_psis, left_psis)

    def evaluator(data: MarkerData) -> np.ndarray:
        origin, rotation, pelvic_width, pelvic_depth = _pelvis_frame_from_asis_psis(
            data,
            right_asis=right_asis,
            left_asis=left_asis,
            right_psis=right_psis,
            left_psis=left_psis,
        )
        local_offset = harrington2007_hip_center_local_hara_axes(
            pelvic_width_mm=pelvic_width,
            pelvic_depth_mm=pelvic_depth,
            leg_length_mm=leg_length_mm,
            side=side,
        )
        return _local_offset_to_global(origin, rotation, local_offset)

    return VirtualPointDefinition(
        name=name,
        method=VirtualPointMethod.LOCAL_FRAME_REGRESSION,
        required_markers=required_markers,
        evaluator=evaluator,
        description="Harrington et al. 2007 predictive hip center in a pelvis ASIS/PSIS frame.",
    )


def predictive_sobral2025_shoulder_cor(
    name: str,
    side: str,
    age_years: float,
    sex: str | int | float | bool,
    height_m: float,
    weight_kg: float,
    angulus_acromialis: str,
    acromioclavicular: str,
    angulus_inferior: str,
    trigonum_spinae: str,
    allow_left_with_warning: bool = False,
) -> VirtualPointDefinition:
    """
    Create a Sobral 2025 predictive glenohumeral center from scapula landmarks.
    """
    normalized_side = _normalized_side(side)
    if normalized_side == "left" and not allow_left_with_warning:
        raise ValueError("Sobral 2025 left-side use requires allow_left_with_warning=True after local validation.")
    required_markers = (angulus_acromialis, acromioclavicular, angulus_inferior, trigonum_spinae)

    def evaluator(data: MarkerData) -> np.ndarray:
        aa = data.get_position([angulus_acromialis])[:3, 0, :]
        ac = data.get_position([acromioclavicular])[:3, 0, :]
        ai = data.get_position([angulus_inferior])[:3, 0, :]
        ts = data.get_position([trigonum_spinae])[:3, 0, :]
        origin, rotation = _scapula_frame_from_landmarks(aa=aa, ts=ts, ai=ai)
        ac_local = _global_points_to_local(origin, rotation, ac)
        ai_local = _global_points_to_local(origin, rotation, ai)
        features = {
            "AC_x": ac_local[0, :],
            "AC_y": ac_local[1, :],
            "AC_z": ac_local[2, :],
            "AI_z": ai_local[2, :],
            "L_AA_TS": _column_distance(aa, ts),
            "L_AA_AI": _column_distance(aa, ai),
            "L_AA_AC": _column_distance(aa, ac),
            "L_TS_AI": _column_distance(ts, ai),
            "L_TS_AC": _column_distance(ts, ac),
            "L_AI_AC": _column_distance(ai, ac),
            "A": float(age_years),
            "S": _sex_code(sex),
            "H": float(height_m),
            "W": float(weight_kg),
        }
        local = _sobral2025_glenohumeral_center_local_from_features(features)
        return _local_offset_to_global(origin, rotation, local)

    return VirtualPointDefinition(
        name=name,
        method=VirtualPointMethod.LOCAL_FRAME_REGRESSION,
        required_markers=required_markers,
        evaluator=evaluator,
        description="Sobral et al. 2025 predictive glenohumeral center in a scapula landmark frame.",
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


def _pelvis_frame_from_asis_psis(
    data: MarkerData,
    right_asis: str,
    left_asis: str,
    right_psis: str,
    left_psis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_asis = data.get_position([right_asis])[:3, 0, :]
    l_asis = data.get_position([left_asis])[:3, 0, :]
    r_psis = data.get_position([right_psis])[:3, 0, :]
    l_psis = data.get_position([left_psis])[:3, 0, :]
    origin = 0.5 * (r_asis + l_asis)
    y_left = _normalize(l_asis - r_asis)
    mid_psis = 0.5 * (r_psis + l_psis)
    posterior = mid_psis - origin
    posterior_projected = posterior - np.sum(posterior * y_left, axis=0)[np.newaxis, :] * y_left
    x_anterior = _normalize(-posterior_projected)
    z_superior = _normalize(np.cross(x_anterior, y_left, axis=0))
    y_left = _normalize(np.cross(z_superior, x_anterior, axis=0))
    rotation = _rotation_from_axis_columns(x_anterior, y_left, z_superior)
    pelvic_width = _column_distance(r_asis, l_asis)
    pelvic_depth = np.abs(np.sum((mid_psis - origin) * x_anterior, axis=0))
    return origin, rotation, pelvic_width, pelvic_depth


def _scapula_frame_from_landmarks(aa: np.ndarray, ts: np.ndarray, ai: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_lateral = _normalize(aa - ts)
    x_normal = _normalize(np.cross(ai - aa, z_lateral, axis=0))
    y_axis = _normalize(np.cross(z_lateral, x_normal, axis=0))
    return aa, _rotation_from_axis_columns(x_normal, y_axis, z_lateral)


def _rotation_from_axis_columns(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    return np.stack((x_axis.T, y_axis.T, z_axis.T), axis=2)


def _local_offset_to_global(origin: np.ndarray, rotation: np.ndarray, local_offset: np.ndarray) -> np.ndarray:
    local_offset = _broadcast_vector(local_offset, origin.shape[1])
    return (np.einsum("fij,jf->fi", rotation, local_offset) + origin.T).T


def _global_points_to_local(origin: np.ndarray, rotation: np.ndarray, global_points: np.ndarray) -> np.ndarray:
    return np.einsum("fji,jf->fi", rotation, global_points - origin).T


def _column_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.norm(first - second, axis=0)


def _normalized_side(side: str) -> str:
    side = side.lower()
    if side in {"right", "r", "d"}:
        return "right"
    if side in {"left", "l", "g"}:
        return "left"
    raise ValueError("side must be right/left, R/L, or D/G.")


def _sex_code(sex: str | int | float | bool) -> float:
    if isinstance(sex, str):
        sex = sex.lower()
        if sex in {"female", "f", "woman", "0"}:
            return 0.0
        if sex in {"male", "m", "man", "1"}:
            return 1.0
        raise ValueError("sex must be female/male or 0/1.")
    return float(sex)


def _sobral2025_glenohumeral_center_local_from_features(features: dict[str, np.ndarray | float]) -> np.ndarray:
    f = {key: np.asarray(value, dtype=float) for key, value in features.items()}
    x = (
        25.5316
        + 0.6334 * f["AC_x"]
        + 0.7842 * f["AC_y"]
        - 0.0832 * f["AI_z"]
        - 0.2673 * f["L_AA_TS"]
        + 0.0365 * f["L_AA_AI"]
        - 0.5353 * f["L_AA_AC"]
        + 0.0843 * f["L_TS_AI"]
        + 0.2350 * f["L_TS_AC"]
        - 0.1246 * f["L_AI_AC"]
        - 0.0237 * f["A"]
        + 2.1296 * f["S"]
        - 1.1900 * f["H"]
        + 0.0221 * f["W"]
    )
    y = (
        -6.7070
        - 0.2514 * f["AC_x"]
        + 0.7558 * f["AC_y"]
        + 0.0264 * f["AI_z"]
        - 0.0620 * f["L_AA_AI"]
        - 1.7641 * f["S"]
        - 5.7094 * f["H"]
    )
    z = (
        -22.5233
        + 0.5954 * f["AC_x"]
        + 0.0600 * f["AC_y"]
        + 0.1085 * f["AI_z"]
        + 0.3983 * f["AC_z"]
        - 0.3880 * f["L_AA_AC"]
        - 0.0197 * f["L_TS_AI"]
        - 0.0934 * f["L_TS_AC"]
        + 0.0558 * f["A"]
        + 0.2930 * f["S"]
        + 26.4715 * f["H"]
        - 0.0361 * f["W"]
    )
    return np.vstack((x, y, z))


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
