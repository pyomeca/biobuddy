from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

from ..components.generic.biomechanical_model import BiomechanicalModel
from ..components.generic.rigidbody.axis import Axis
from ..components.generic.rigidbody.marker import Marker
from ..components.generic.rigidbody.mesh import Mesh
from ..components.generic.rigidbody.segment import Segment
from ..components.generic.rigidbody.segment_coordinate_system import (
    SegmentCoordinateSystem,
    SegmentCoordinateSystemUtils,
)
from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..utils.enums import Rotations, Translations
from ..utils.marker_data import C3dData, MarkerData, marker_data_with_stripped_prefixes

ProgressCallback = Callable[[str], None]


class FunctionalMethod(Enum):
    """
    Available functional calibration methods for generated templates.
    """

    SCORE = "score"
    SARA = "sara"
    SARA_DIRECTION = "sara_direction"


def is_sara_direction_method(method: FunctionalMethod | str) -> bool:
    """
    Return whether a method should be evaluated as a SARA direction plus origin point.

    ``sara`` is kept as a legacy alias for older saved drafts/templates; new templates
    should use ``sara_direction`` to make the endpoint semantics explicit.
    """
    method_value = method.value if isinstance(method, FunctionalMethod) else str(method)
    return method_value in {
        FunctionalMethod.SARA.value,
        FunctionalMethod.SARA_DIRECTION.value,
    }


@dataclass(frozen=True)
class MarkerEndpointSpec:
    """
    A point defined as the mean of one or more experimental markers.
    """

    marker_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.marker_names) == 0:
            raise ValueError("A marker endpoint must contain at least one marker.")

    @classmethod
    def from_value(cls, value: str | tuple[str, ...] | list[str] | "MarkerEndpointSpec") -> "MarkerEndpointSpec":
        """
        Normalize a marker name or marker list into an endpoint specification.
        """
        if isinstance(value, MarkerEndpointSpec):
            return value
        if isinstance(value, str):
            return cls((value,))
        return cls(tuple(value))

    def to_callable(
        self, functional_data: dict[str, MarkerData] | None = None
    ) -> Callable[[MarkerData, BiomechanicalModelReal], np.ndarray]:
        """
        Return a BioBuddy-compatible callable that evaluates this point.
        """
        return SegmentCoordinateSystemUtils.mean_markers(list(self.marker_names))

    def evaluate(self, data: MarkerData) -> np.ndarray:
        """
        Evaluate the endpoint over all frames.
        """
        return data.markers_center_position(list(self.marker_names))[:3, :]


@dataclass(frozen=True)
class FunctionalCenterSpec:
    """
    Optional functional joint-center definition with an anatomical fallback.
    """

    method: FunctionalMethod
    trial_name: str
    parent_marker_names: tuple[str, ...]
    child_marker_names: tuple[str, ...]
    fallback: MarkerEndpointSpec

    def to_callable(
        self, functional_data: dict[str, MarkerData] | None
    ) -> Callable[[MarkerData, BiomechanicalModelReal], np.ndarray]:
        """
        Return the functional center callable when data is available, otherwise use the fallback point.
        """
        if self.method != FunctionalMethod.SCORE or functional_data is None or self.trial_name not in functional_data:
            return self.fallback.to_callable()
        return SegmentCoordinateSystemUtils.score(
            functional_data=functional_data[self.trial_name],
            parent_marker_names=list(self.parent_marker_names),
            child_marker_names=list(self.child_marker_names),
            visualize=False,
        )


@dataclass(frozen=True)
class FunctionalAxisProjectionPointSpec:
    """
    Point obtained by projecting a marker group onto a functional SARA axis.
    """

    method: FunctionalMethod
    trial_name: str
    parent_marker_names: tuple[str, ...]
    child_marker_names: tuple[str, ...]
    expected_axis: "AxisSpec"
    origin_marker_names: tuple[str, ...]
    point_marker_names: tuple[str, ...]
    fallback: MarkerEndpointSpec
    max_static_axis_deviation_degrees: float | None = None

    @property
    def marker_names(self) -> tuple[str, ...]:
        marker_names = list(self.fallback.marker_names)
        marker_names.extend(self.parent_marker_names)
        marker_names.extend(self.child_marker_names)
        marker_names.extend(self.expected_axis.start.marker_names)
        marker_names.extend(self.expected_axis.end.marker_names)
        marker_names.extend(self.origin_marker_names)
        marker_names.extend(self.point_marker_names)
        return tuple(dict.fromkeys(marker_names))

    def to_callable(
        self, functional_data: dict[str, MarkerData] | None
    ) -> Callable[[MarkerData, BiomechanicalModelReal], np.ndarray]:
        """
        Return the projection callable when SARA data is available, otherwise use the fallback point.
        """
        if (
            not is_sara_direction_method(self.method)
            or functional_data is None
            or self.trial_name not in functional_data
        ):
            return self.fallback.to_callable()

        functional_trial = functional_data[self.trial_name]

        def projected_point(markers: MarkerData, model: BiomechanicalModelReal) -> np.ndarray:
            sara_axis = _sara_axis_with_static_fallback(
                name=self.expected_axis.name,
                functional_data=functional_trial,
                parent_marker_names=list(self.parent_marker_names),
                child_marker_names=list(self.child_marker_names),
                expected_axis=self.expected_axis,
                fallback_axis=self.expected_axis,
                origin_positions_global=lambda sara_markers, sara_model: sara_markers.markers_center_position(
                    list(self.origin_marker_names)
                )[:3, :],
                max_static_axis_deviation_degrees=self.max_static_axis_deviation_degrees,
                visualize=False,
            )
            axis_start = sara_axis.start.function(markers, model)[:3].reshape(3, 1)
            axis_end = sara_axis.end.function(markers, model)[:3].reshape(3, 1)
            point = markers.markers_center_position(list(self.point_marker_names))[:3, :]
            axis_vector = axis_end - axis_start
            axis_norm = np.linalg.norm(axis_vector, axis=0, keepdims=True)
            axis_unit = np.divide(
                axis_vector,
                axis_norm,
                out=np.zeros_like(axis_vector),
                where=axis_norm > 1e-12,
            )
            distance = np.sum((point - axis_start) * axis_unit, axis=0, keepdims=True)
            projected = np.ones((4, markers.nb_frames))
            projected[:3, :] = axis_start + axis_unit * distance
            return np.nanmean(projected, axis=1)

        return projected_point

    def evaluate(self, data: MarkerData) -> np.ndarray:
        """
        Evaluate the anatomical fallback over all frames for quality metrics.
        """
        return self.fallback.evaluate(data)


@dataclass(frozen=True)
class AxisSpec:
    """
    A local-frame axis defined from two marker endpoints.
    """

    name: Axis.Name
    start: MarkerEndpointSpec | FunctionalCenterSpec | FunctionalAxisProjectionPointSpec
    end: MarkerEndpointSpec | FunctionalCenterSpec | FunctionalAxisProjectionPointSpec

    @classmethod
    def from_markers(
        cls,
        name: Axis.Name,
        start: str | tuple[str, ...] | list[str] | MarkerEndpointSpec,
        end: str | tuple[str, ...] | list[str] | MarkerEndpointSpec,
    ) -> "AxisSpec":
        """
        Build an axis from marker names. Multiple markers are averaged at each endpoint.
        """
        return cls(
            name=name,
            start=MarkerEndpointSpec.from_value(start),
            end=MarkerEndpointSpec.from_value(end),
        )

    def to_axis(self, functional_data: dict[str, MarkerData] | None = None) -> Axis:
        """
        Return the BioBuddy generic axis object.
        """
        return Axis(
            name=self.name,
            start=_axis_endpoint_to_axis_value(self.start, functional_data),
            end=_axis_endpoint_to_axis_value(self.end, functional_data),
        )

    def vector(self, data: MarkerData) -> np.ndarray:
        """
        Evaluate the raw, non-normalized axis vector over all frames.
        """
        return _evaluate_axis_endpoint(self.end, data) - _evaluate_axis_endpoint(self.start, data)


def _axis_endpoint_to_axis_value(endpoint, functional_data: dict[str, MarkerData] | None):
    """
    Return a marker-name shortcut when possible, otherwise a callable endpoint.
    """
    if isinstance(endpoint, MarkerEndpointSpec) and len(endpoint.marker_names) == 1:
        return endpoint.marker_names[0]
    return endpoint.to_callable(functional_data)


def _evaluate_axis_endpoint(endpoint, data: MarkerData) -> np.ndarray:
    """
    Evaluate an endpoint for quality metrics, using functional fallbacks when needed.
    """
    if isinstance(endpoint, FunctionalCenterSpec):
        return endpoint.fallback.evaluate(data)
    return endpoint.evaluate(data)


@dataclass(frozen=True)
class FunctionalAxisSpec:
    """
    Optional functional axis definition with an anatomical fallback axis.
    """

    method: FunctionalMethod
    trial_name: str
    fallback: AxisSpec
    parent_marker_names: tuple[str, ...] = ()
    child_marker_names: tuple[str, ...] = ()
    expected_axis: AxisSpec | None = None
    origin_marker_names: tuple[str, ...] = ()
    max_static_axis_deviation_degrees: float | None = None

    def to_axis(self, functional_data: dict[str, MarkerData] | None) -> Axis:
        """
        Return the functional axis when possible, otherwise return the anatomical fallback.
        """
        if (
            not is_sara_direction_method(self.method)
            or functional_data is None
            or self.trial_name not in functional_data
        ):
            return self.fallback.to_axis(functional_data=None)
        if self.expected_axis is None or len(self.origin_marker_names) == 0:
            return self.fallback.to_axis(functional_data=None)

        return _sara_axis_with_static_fallback(
            name=self.fallback.name,
            functional_data=functional_data[self.trial_name],
            parent_marker_names=list(self.parent_marker_names),
            child_marker_names=list(self.child_marker_names),
            expected_axis=self.expected_axis,
            fallback_axis=self.fallback,
            origin_positions_global=lambda markers, model: markers.markers_center_position(
                list(self.origin_marker_names)
            )[:3, :],
            max_static_axis_deviation_degrees=self.max_static_axis_deviation_degrees,
            visualize=False,
        )


def _sara_axis_with_static_fallback(
    *,
    name: Axis.Name,
    functional_data: MarkerData,
    parent_marker_names: list[str],
    child_marker_names: list[str],
    expected_axis: AxisSpec,
    fallback_axis: AxisSpec,
    origin_positions_global: Callable | None,
    max_static_axis_deviation_degrees: float | None,
    visualize: bool,
) -> Axis:
    """
    Return a SARA axis that falls back to its anatomical axis when static quality is too poor.
    """
    sara_axis = SegmentCoordinateSystemUtils.sara(
        name=name,
        functional_data=functional_data,
        parent_marker_names=parent_marker_names,
        child_marker_names=child_marker_names,
        expected_rotation_axis_orientation=expected_axis.to_axis(functional_data=None),
        origin_positions_global=origin_positions_global,
        visualize=visualize,
    )
    if max_static_axis_deviation_degrees is None:
        return sara_axis

    fallback = fallback_axis.to_axis(functional_data=None)
    selected_points_cache = {}

    def selected_points(markers: MarkerData, model: BiomechanicalModelReal) -> tuple[np.ndarray, np.ndarray]:
        cache_key = id(markers)
        if cache_key not in selected_points_cache:
            sara_start = np.asarray(sara_axis.start.function(markers, model), dtype=float)
            sara_end = np.asarray(sara_axis.end.function(markers, model), dtype=float)
            fallback_start = np.asarray(fallback.start.function(markers, model), dtype=float)
            fallback_end = np.asarray(fallback.end.function(markers, model), dtype=float)
            deviation = _axis_deviation_degrees(sara_start, sara_end, fallback_start, fallback_end)
            use_fallback = bool(np.isfinite(deviation) and deviation > float(max_static_axis_deviation_degrees))
            selected_points_cache[cache_key] = (
                fallback_start if use_fallback else sara_start,
                fallback_end if use_fallback else sara_end,
            )
        return selected_points_cache[cache_key]

    return Axis(
        name=name,
        start=lambda markers, model: selected_points(markers, model)[0],
        end=lambda markers, model: selected_points(markers, model)[1],
    )


def _axis_deviation_degrees(start_a: np.ndarray, end_a: np.ndarray, start_b: np.ndarray, end_b: np.ndarray) -> float:
    """
    Return the oriented angle between two static axis directions.
    """
    direction_a = _mean_axis_direction(start_a, end_a)
    direction_b = _mean_axis_direction(start_b, end_b)
    norm_a = np.linalg.norm(direction_a)
    norm_b = np.linalg.norm(direction_b)
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return float("nan")
    cosine = float(np.clip(np.dot(direction_a / norm_a, direction_b / norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _mean_axis_direction(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Collapse an axis endpoint pair to one finite 3D direction.
    """
    start_points = _as_point_series(start)
    end_points = _as_point_series(end)
    frame_count = min(start_points.shape[1], end_points.shape[1])
    if frame_count == 0:
        return np.full((3,), np.nan)
    return np.nanmean(end_points[:, :frame_count] - start_points[:, :frame_count], axis=1)


def _as_point_series(value: np.ndarray) -> np.ndarray:
    """
    Normalize a point or point time series to shape 3 x frames.
    """
    points = np.asarray(value, dtype=float)
    if points.ndim == 1:
        return points[:3].reshape(3, 1)
    if points.ndim == 2:
        return points[:3, :]
    return np.asarray(points[:3], dtype=float).reshape(3, -1)


@dataclass(frozen=True)
class LocalFrameSpec:
    """
    Segment coordinate-system definition.
    """

    origin: MarkerEndpointSpec | FunctionalCenterSpec | FunctionalAxisProjectionPointSpec
    first_axis: AxisSpec
    second_axis: AxisSpec | FunctionalAxisSpec
    axis_to_keep: Axis.Name

    def to_scs(self, functional_data: dict[str, MarkerData] | None = None) -> SegmentCoordinateSystem:
        """
        Return the BioBuddy generic segment coordinate system.
        """
        origin = (
            self.origin.to_callable(functional_data)
            if isinstance(self.origin, (FunctionalCenterSpec, FunctionalAxisProjectionPointSpec))
            else self.origin.to_callable()
        )
        first_axis = self.first_axis.to_axis(functional_data=functional_data)
        second_axis = (
            self.second_axis.to_axis(functional_data)
            if isinstance(self.second_axis, FunctionalAxisSpec)
            else self.second_axis.to_axis(functional_data=functional_data)
        )
        return SegmentCoordinateSystem(
            origin=origin,
            first_axis=first_axis,
            second_axis=second_axis,
            axis_to_keep=self.axis_to_keep,
        )

    def quality_axes(self) -> tuple[AxisSpec, AxisSpec]:
        """
        Return the raw anatomical axes used for quality metrics.
        """
        second_axis = (
            self.second_axis.fallback if isinstance(self.second_axis, FunctionalAxisSpec) else self.second_axis
        )
        return self.first_axis, second_axis

    def origin_endpoint(self) -> MarkerEndpointSpec:
        """
        Return a marker-defined origin for dynamic visualization.
        """
        if isinstance(self.origin, (FunctionalCenterSpec, FunctionalAxisProjectionPointSpec)):
            return self.origin.fallback
        return self.origin


@dataclass(frozen=True)
class MarkerAttachmentSpec:
    """
    Marker classification entry.

    A marker may be attached to more than one segment. This is useful for anatomical
    landmarks that also define adjacent segment frames.
    """

    name: str
    segment_names: tuple[str, ...]
    is_technical: bool = True
    is_anatomical: bool = False


@dataclass(frozen=True)
class SegmentSpec:
    """
    Declarative segment definition for a generated model.
    """

    name: str
    parent_name: str
    translations: Translations = Translations.NONE
    rotations: Rotations = Rotations.NONE
    frame: LocalFrameSpec | None = None
    joint_frame: LocalFrameSpec | None = None
    joint_segment_name: str | None = None
    mesh_points: tuple[MarkerEndpointSpec, ...] = ()
    inertia_name: object | None = None

    @property
    def has_separate_joint_frame(self) -> bool:
        """Return whether this anatomical segment is driven through a separate joint segment."""
        return self.joint_frame is not None

    @property
    def resolved_joint_segment_name(self) -> str:
        """Return the explicit or conventional name of the separate joint segment."""
        return self.joint_segment_name or f"{self.name}Joint"


@dataclass(frozen=True)
class FunctionalTrialSpec:
    """
    A C3D trial needed for optional functional calibration.
    """

    name: str
    file_pattern: str
    required_markers: tuple[str, ...]
    method: FunctionalMethod
    alternate_file_patterns: tuple[str, ...] = ()

    @property
    def file_patterns(self) -> tuple[str, ...]:
        """
        Return all accepted filename patterns, ordered from canonical to fallback.
        """
        return (self.file_pattern,) + tuple(self.alternate_file_patterns)


@dataclass(frozen=True)
class MarkerAvailability:
    """
    Availability summary for one required marker.
    """

    name: str
    is_present: bool
    valid_frame_count: int
    total_frame_count: int

    @property
    def missing_frame_count(self) -> int:
        return self.total_frame_count - self.valid_frame_count


@dataclass(frozen=True)
class MarkerAvailabilityReport:
    """
    Availability summary for a set of required markers.
    """

    required_markers: tuple[str, ...]
    markers: dict[str, MarkerAvailability]
    complete_frame_count: int
    total_frame_count: int

    @property
    def missing_markers(self) -> tuple[str, ...]:
        return tuple(name for name, marker in self.markers.items() if not marker.is_present)


@dataclass(frozen=True)
class FrameQuality:
    """
    Raw-frame quality metrics computed before orthonormalization.
    """

    segment_name: str
    angle_degrees: np.ndarray
    first_axis_norm: np.ndarray
    second_axis_norm: np.ndarray

    @property
    def mean_angle_degrees(self) -> float:
        return float(np.nanmean(self.angle_degrees))


@dataclass(frozen=True)
class ModelTemplate:
    """
    Full declarative model template.
    """

    name: str
    segments: tuple[SegmentSpec, ...]
    marker_attachments: tuple[MarkerAttachmentSpec, ...]
    required_static_markers: tuple[str, ...]
    functional_trials: tuple[FunctionalTrialSpec, ...] = ()
    root_segment_name: str | None = None
    inertia_parameters_factory: Callable[[MarkerData], dict[str, object]] | None = None

    def marker_segments(self) -> dict[str, tuple[str, ...]]:
        """
        Return the marker-to-segment classification map.
        """
        marker_segments = {}
        for attachment in self.marker_attachments:
            marker_segments[attachment.name] = tuple(
                dict.fromkeys(marker_segments.get(attachment.name, ()) + attachment.segment_names)
            )
        return marker_segments


def required_static_markers(template: ModelTemplate) -> tuple[str, ...]:
    """
    Derive all static/anatomical markers needed to instantiate a template.
    """
    marker_names = set(template.required_static_markers)
    for attachment in template.marker_attachments:
        marker_names.add(attachment.name)
    for segment in template.segments:
        if segment.frame is not None:
            marker_names.update(_marker_names_from_frame(segment.frame))
        if segment.joint_frame is not None:
            marker_names.update(_marker_names_from_frame(segment.joint_frame))
        for mesh_point in segment.mesh_points:
            marker_names.update(mesh_point.marker_names)
    return tuple(sorted(marker_names))


def required_functional_markers(template: ModelTemplate) -> dict[str, tuple[str, ...]]:
    """
    Return required markers for each optional functional calibration trial.
    """
    return {trial.name: tuple(sorted(trial.required_markers)) for trial in template.functional_trials}


def required_markers(template: ModelTemplate) -> dict[str, tuple[str, ...]]:
    """
    Return all required markers grouped by trial role.
    """
    markers = {"static": required_static_markers(template)}
    markers.update(required_functional_markers(template))
    return markers


def marker_availability(
    data: MarkerData, required_marker_names: tuple[str, ...] | list[str]
) -> MarkerAvailabilityReport:
    """
    Report which markers are present and how many frames are valid for each marker.
    """
    required_marker_names = tuple(sorted(set(required_marker_names)))
    marker_reports = {}
    valid_by_marker = []
    for marker_name in required_marker_names:
        is_present = marker_name in data.marker_names
        valid_frames = np.zeros(data.nb_frames, dtype=bool)
        if is_present:
            positions = data.get_position([marker_name])[:3, 0, :]
            valid_frames = np.all(np.isfinite(positions), axis=0)
        marker_reports[marker_name] = MarkerAvailability(
            name=marker_name,
            is_present=is_present,
            valid_frame_count=int(np.sum(valid_frames)),
            total_frame_count=data.nb_frames,
        )
        valid_by_marker.append(valid_frames)

    if len(valid_by_marker) == 0:
        complete_frame_count = data.nb_frames
    else:
        complete_frame_count = int(np.sum(np.logical_and.reduce(valid_by_marker)))

    return MarkerAvailabilityReport(
        required_markers=required_marker_names,
        markers=marker_reports,
        complete_frame_count=complete_frame_count,
        total_frame_count=data.nb_frames,
    )


def template_marker_availability(
    template: ModelTemplate,
    static_data: MarkerData,
    functional_data: dict[str, MarkerData] | None = None,
) -> dict[str, MarkerAvailabilityReport]:
    """
    Report marker availability for the static trial and any provided functional trials.
    """
    reports = {"static": marker_availability(static_data, required_static_markers(template))}
    if functional_data is None:
        return reports
    functional_requirements = required_functional_markers(template)
    for trial_name, data in functional_data.items():
        if trial_name in functional_requirements:
            reports[trial_name] = marker_availability(data, functional_requirements[trial_name])
    return reports


def _marker_names_from_frame(frame: LocalFrameSpec) -> set[str]:
    """
    Collect the marker names needed by the marker-defined parts of a frame.
    """
    marker_names = _marker_names_from_origin(frame.origin)
    marker_names.update(_marker_names_from_axis(frame.first_axis))
    marker_names.update(_marker_names_from_axis(frame.second_axis))
    return marker_names


def _marker_names_from_origin(
    origin: MarkerEndpointSpec | FunctionalCenterSpec | FunctionalAxisProjectionPointSpec,
) -> set[str]:
    if isinstance(origin, FunctionalCenterSpec):
        marker_names = set(origin.fallback.marker_names)
        marker_names.update(origin.parent_marker_names)
        marker_names.update(origin.child_marker_names)
        return marker_names
    return set(origin.marker_names)


def _marker_names_from_axis(axis: AxisSpec | FunctionalAxisSpec) -> set[str]:
    if isinstance(axis, FunctionalAxisSpec):
        marker_names = _marker_names_from_axis(axis.fallback)
        marker_names.update(axis.parent_marker_names)
        marker_names.update(axis.child_marker_names)
        marker_names.update(axis.origin_marker_names)
        if axis.expected_axis is not None:
            marker_names.update(_marker_names_from_axis(axis.expected_axis))
        return marker_names
    marker_names = _marker_names_from_origin(axis.start)
    marker_names.update(_marker_names_from_origin(axis.end))
    return marker_names


def build_generic_model(
    template: ModelTemplate,
    functional_data: dict[str, MarkerData] | None = None,
    inertia_parameters_by_segment: dict[str, object] | None = None,
) -> BiomechanicalModel:
    """
    Build a generic BioBuddy model from a model template.
    """
    model = BiomechanicalModel()
    inertia_parameters_by_segment = {} if inertia_parameters_by_segment is None else inertia_parameters_by_segment
    for segment_spec in template.segments:
        parent_name = segment_spec.parent_name
        translations = segment_spec.translations
        rotations = segment_spec.rotations
        if segment_spec.has_separate_joint_frame:
            joint_name = segment_spec.resolved_joint_segment_name
            model.add_segment(
                Segment(
                    name=joint_name,
                    parent_name=segment_spec.parent_name,
                    translations=translations,
                    rotations=rotations,
                    segment_coordinate_system=segment_spec.joint_frame.to_scs(functional_data=functional_data),
                )
            )
            parent_name = joint_name
            translations = Translations.NONE
            rotations = Rotations.NONE
        segment = Segment(
            name=segment_spec.name,
            parent_name=parent_name,
            translations=translations,
            rotations=rotations,
            segment_coordinate_system=(
                None if segment_spec.frame is None else segment_spec.frame.to_scs(functional_data=functional_data)
            ),
            mesh=(
                None
                if len(segment_spec.mesh_points) == 0
                else Mesh(
                    tuple(point.to_callable() for point in segment_spec.mesh_points),
                    is_local=False,
                )
            ),
            inertia_parameters=inertia_parameters_by_segment.get(segment_spec.name),
        )
        model.add_segment(segment)

    for attachment in template.marker_attachments:
        for segment_name in attachment.segment_names:
            model.segments[segment_name].add_marker(
                Marker(
                    name=attachment.name,
                    function=attachment.name,
                    is_technical=attachment.is_technical,
                    is_anatomical=attachment.is_anatomical,
                )
            )
    return model


def build_real_model(
    template: ModelTemplate,
    static_data: MarkerData,
    functional_data: dict[str, MarkerData] | None = None,
) -> BiomechanicalModelReal:
    """
    Generate a real model from a template and marker calibration data.
    """
    missing_markers = sorted(set(required_static_markers(template)) - set(static_data.marker_names))
    if missing_markers:
        raise ValueError(f"Missing required static markers: {', '.join(missing_markers)}")

    inertia_parameters_by_segment = (
        {} if template.inertia_parameters_factory is None else template.inertia_parameters_factory(static_data)
    )
    model = build_generic_model(
        template=template,
        functional_data=functional_data,
        inertia_parameters_by_segment=inertia_parameters_by_segment,
    ).to_real(static_data)
    if template.root_segment_name is not None and "root" in model.segment_names:
        model.segments[template.root_segment_name].parent_name = model.segments["root"].parent_name
        model.segments[template.root_segment_name].segment_coordinate_system = model.segments[
            "root"
        ].segment_coordinate_system
        model.segments._remove("root")
    return model


def load_functional_c3d_trials(
    template: ModelTemplate,
    calibration_folder: Path,
    marker_name_prefixes_to_strip: tuple[str, ...] = (),
    progress_callback: ProgressCallback | None = None,
) -> dict[str, MarkerData]:
    """
    Load optional functional trials requested by a template.
    """
    functional_data = {}
    for trial_spec in template.functional_trials:
        matches = []
        matched_pattern = None
        for file_pattern in trial_spec.file_patterns:
            matches = list(calibration_folder.glob(file_pattern))
            if len(matches) != 0:
                matched_pattern = file_pattern
                break
        if len(matches) == 0:
            continue
        if len(matches) > 1:
            raise RuntimeError(
                f"Expected one '{trial_spec.name}' trial matching '{matched_pattern}', found {len(matches)}."
            )
        if progress_callback is not None:
            progress_callback(f"Loading functional C3D {trial_spec.name}: {matches[0].name}")
        data = C3dData(str(matches[0]))
        data = marker_data_with_stripped_prefixes(data, marker_name_prefixes_to_strip)
        if progress_callback is not None:
            progress_callback(f"Checking functional markers: {trial_spec.name}")
        missing_markers = sorted(set(trial_spec.required_markers) - set(data.marker_names))
        if missing_markers:
            raise ValueError(f"Trial '{trial_spec.name}' is missing markers: {', '.join(missing_markers)}")
        functional_data[trial_spec.name] = data
    return functional_data


def build_real_model_from_c3d_folder(
    template: ModelTemplate,
    calibration_folder: Path,
    static_patterns: tuple[str, ...] = (
        "Test_anato.c3d",
        "Test_main.c3d",
        "*static*.c3d",
        "*func_anat.c3d",
    ),
) -> BiomechanicalModelReal:
    """
    Generate a real model from a calibration folder.
    """
    static_matches = []
    for static_pattern in static_patterns:
        static_matches = list(calibration_folder.glob(static_pattern))
        if static_matches:
            break
    if len(static_matches) != 1:
        patterns = ", ".join(static_patterns)
        raise RuntimeError(
            f"Expected exactly one static trial matching one of {patterns}, found {len(static_matches)}."
        )
    static_data = C3dData(str(static_matches[0]))
    functional_data = load_functional_c3d_trials(template=template, calibration_folder=calibration_folder)
    return build_real_model(template=template, static_data=static_data, functional_data=functional_data)


def compute_frame_quality(template: ModelTemplate, data: MarkerData) -> dict[str, FrameQuality]:
    """
    Compute quality indicators for each marker-defined segment frame.

    The reported angle is measured between the two raw vectors that define the
    construction plane, before any cross product or orthonormalization.
    """
    quality = {}
    for segment in template.segments:
        frames = ((segment.name, segment.frame),)
        if segment.joint_frame is not None:
            frames += ((segment.resolved_joint_segment_name, segment.joint_frame),)
        for frame_name, frame in frames:
            if frame is None:
                continue
            first_axis, second_axis = frame.quality_axes()
            first_vector = first_axis.vector(data)
            second_vector = second_axis.vector(data)
            first_norm = np.linalg.norm(first_vector, axis=0)
            second_norm = np.linalg.norm(second_vector, axis=0)
            dot = np.sum(first_vector * second_vector, axis=0)
            denominator = first_norm * second_norm
            cosine = np.divide(
                dot,
                denominator,
                out=np.full_like(dot, np.nan, dtype=float),
                where=denominator != 0,
            )
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            quality[frame_name] = FrameQuality(
                segment_name=frame_name,
                angle_degrees=angle,
                first_axis_norm=first_norm,
                second_axis_norm=second_norm,
            )
    return quality


def compute_dynamic_segment_frames(template: ModelTemplate, data: MarkerData) -> dict[str, np.ndarray]:
    """
    Compute dynamic global segment frames from marker-defined template frames.

    Returns
    -------
    dict[str, np.ndarray]
        One ``4 x 4 x n_frames`` matrix per segment.
    """
    frames = {}
    for segment in template.segments:
        frame_specs = ((segment.name, segment.frame),)
        if segment.joint_frame is not None:
            frame_specs += ((segment.resolved_joint_segment_name, segment.joint_frame),)
        for frame_name, frame_spec in frame_specs:
            if frame_spec is None:
                continue
            first_axis, second_axis = frame_spec.quality_axes()
            first_axis, second_axis, third_name = _ordered_axes_and_third_name(first_axis, second_axis)
            first_vector = first_axis.vector(data)
            second_vector = second_axis.vector(data)
            axis_to_keep = frame_spec.axis_to_keep

            first_name = first_axis.name
            second_name = second_axis.name
            if first_name == second_name:
                raise ValueError(f"Frame '{frame_name}' defines two axes with the same name.")

            third_vector = np.cross(first_vector, second_vector, axis=0)
            if axis_to_keep == first_name:
                second_vector = np.cross(third_vector, first_vector, axis=0)
            elif axis_to_keep == second_name:
                first_vector = np.cross(second_vector, third_vector, axis=0)
            else:
                raise ValueError(f"Frame '{frame_name}' axis_to_keep must be one of the two defined axes.")

            rt = np.zeros((4, 4, data.nb_frames))
            rt[:3, first_name, :] = _normalize(first_vector)
            rt[:3, second_name, :] = _normalize(second_vector)
            rt[:3, third_name, :] = _normalize(third_vector)
            rt[:3, 3, :] = frame_spec.origin_endpoint().evaluate(data)
            rt[3, 3, :] = 1.0
            frames[frame_name] = rt
    return frames


def _third_axis_name(first_name: Axis.Name, second_name: Axis.Name) -> Axis.Name:
    axis_names = {Axis.Name.X, Axis.Name.Y, Axis.Name.Z}
    missing = axis_names - {first_name, second_name}
    if len(missing) != 1:
        raise ValueError("A local frame must define two different axes.")
    return missing.pop()


def _ordered_axes_and_third_name(first_axis: AxisSpec, second_axis: AxisSpec) -> tuple[AxisSpec, AxisSpec, Axis.Name]:
    """
    Match the axis ordering used by ``SegmentCoordinateSystem.get_axes``.
    """
    if first_axis.name == second_axis.name:
        raise ValueError("The two axes cannot be the same axis")

    if first_axis.name == Axis.Name.X:
        third_name = Axis.Name.Y if second_axis.name == Axis.Name.Z else Axis.Name.Z
        if second_axis.name == Axis.Name.Z:
            first_axis, second_axis = second_axis, first_axis
    elif first_axis.name == Axis.Name.Y:
        third_name = Axis.Name.Z if second_axis.name == Axis.Name.X else Axis.Name.X
        if second_axis.name == Axis.Name.X:
            first_axis, second_axis = second_axis, first_axis
    elif first_axis.name == Axis.Name.Z:
        third_name = Axis.Name.X if second_axis.name == Axis.Name.Y else Axis.Name.Y
        if second_axis.name == Axis.Name.Y:
            first_axis, second_axis = second_axis, first_axis
    else:
        raise ValueError("first_axis should be an X, Y or Z axis")
    return first_axis, second_axis, third_name


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=0)
    return np.divide(
        vector,
        norm,
        out=np.full_like(vector, np.nan, dtype=float),
        where=norm[np.newaxis, :] != 0,
    )
