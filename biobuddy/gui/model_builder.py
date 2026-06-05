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
from ..utils.marker_data import C3dData, MarkerData


class FunctionalMethod(Enum):
    """
    Available functional calibration methods for generated templates.
    """

    SCORE = "score"
    SARA = "sara"


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

    def to_callable(self) -> Callable[[MarkerData, BiomechanicalModelReal], np.ndarray]:
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
class AxisSpec:
    """
    A local-frame axis defined from two marker endpoints.
    """

    name: Axis.Name
    start: MarkerEndpointSpec
    end: MarkerEndpointSpec

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

    def to_axis(self) -> Axis:
        """
        Return the BioBuddy generic axis object.
        """
        return Axis(
            name=self.name,
            start=self.start.to_callable(),
            end=self.end.to_callable(),
        )

    def vector(self, data: MarkerData) -> np.ndarray:
        """
        Evaluate the raw, non-normalized axis vector over all frames.
        """
        return self.end.evaluate(data) - self.start.evaluate(data)


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

    def to_axis(self, functional_data: dict[str, MarkerData] | None) -> Axis:
        """
        Return the functional axis when possible, otherwise return the anatomical fallback.
        """
        if self.method != FunctionalMethod.SARA or functional_data is None or self.trial_name not in functional_data:
            return self.fallback.to_axis()
        if self.expected_axis is None or len(self.origin_marker_names) == 0:
            return self.fallback.to_axis()

        return SegmentCoordinateSystemUtils.sara(
            name=self.fallback.name,
            functional_data=functional_data[self.trial_name],
            parent_marker_names=list(self.parent_marker_names),
            child_marker_names=list(self.child_marker_names),
            expected_rotation_axis_orientation=self.expected_axis.to_axis(),
            origin_positions_global=lambda markers, model: markers.markers_center_position(
                list(self.origin_marker_names)
            )[:3, :],
            visualize=False,
        )


@dataclass(frozen=True)
class LocalFrameSpec:
    """
    Segment coordinate-system definition.
    """

    origin: MarkerEndpointSpec | FunctionalCenterSpec
    first_axis: AxisSpec
    second_axis: AxisSpec | FunctionalAxisSpec
    axis_to_keep: Axis.Name

    def to_scs(self, functional_data: dict[str, MarkerData] | None = None) -> SegmentCoordinateSystem:
        """
        Return the BioBuddy generic segment coordinate system.
        """
        origin = (
            self.origin.to_callable(functional_data)
            if isinstance(self.origin, FunctionalCenterSpec)
            else self.origin.to_callable()
        )
        second_axis = (
            self.second_axis.to_axis(functional_data)
            if isinstance(self.second_axis, FunctionalAxisSpec)
            else self.second_axis.to_axis()
        )
        return SegmentCoordinateSystem(
            origin=origin,
            first_axis=self.first_axis.to_axis(),
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
        if isinstance(self.origin, FunctionalCenterSpec):
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
    mesh_points: tuple[MarkerEndpointSpec, ...] = ()


@dataclass(frozen=True)
class FunctionalTrialSpec:
    """
    A C3D trial needed for optional functional calibration.
    """

    name: str
    file_pattern: str
    required_markers: tuple[str, ...]
    method: FunctionalMethod


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

    def marker_segments(self) -> dict[str, tuple[str, ...]]:
        """
        Return the marker-to-segment classification map.
        """
        return {attachment.name: attachment.segment_names for attachment in self.marker_attachments}


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


def _marker_names_from_origin(origin: MarkerEndpointSpec | FunctionalCenterSpec) -> set[str]:
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
    marker_names = set(axis.start.marker_names)
    marker_names.update(axis.end.marker_names)
    return marker_names


def build_generic_model(
    template: ModelTemplate, functional_data: dict[str, MarkerData] | None = None
) -> BiomechanicalModel:
    """
    Build a generic BioBuddy model from a model template.
    """
    model = BiomechanicalModel()
    for segment_spec in template.segments:
        segment = Segment(
            name=segment_spec.name,
            parent_name=segment_spec.parent_name,
            translations=segment_spec.translations,
            rotations=segment_spec.rotations,
            segment_coordinate_system=(
                None if segment_spec.frame is None else segment_spec.frame.to_scs(functional_data=functional_data)
            ),
            mesh=(
                None
                if len(segment_spec.mesh_points) == 0
                else Mesh(tuple(point.to_callable() for point in segment_spec.mesh_points), is_local=False)
            ),
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

    model = build_generic_model(template=template, functional_data=functional_data).to_real(static_data)
    if template.root_segment_name is not None and "root" in model.segment_names:
        model.segments[template.root_segment_name].parent_name = model.segments["root"].parent_name
        model.segments[template.root_segment_name].segment_coordinate_system = model.segments[
            "root"
        ].segment_coordinate_system
        model.segments._remove("root")
    return model


def load_functional_c3d_trials(template: ModelTemplate, calibration_folder: Path) -> dict[str, MarkerData]:
    """
    Load optional functional trials requested by a template.
    """
    functional_data = {}
    for trial_spec in template.functional_trials:
        matches = list(calibration_folder.glob(trial_spec.file_pattern))
        if len(matches) == 0:
            continue
        if len(matches) > 1:
            raise RuntimeError(f"Expected one '{trial_spec.name}' trial, found {len(matches)}.")
        data = C3dData(str(matches[0]))
        missing_markers = sorted(set(trial_spec.required_markers) - set(data.marker_names))
        if missing_markers:
            raise ValueError(f"Trial '{trial_spec.name}' is missing markers: {', '.join(missing_markers)}")
        functional_data[trial_spec.name] = data
    return functional_data


def build_real_model_from_c3d_folder(
    template: ModelTemplate,
    calibration_folder: Path,
    static_patterns: tuple[str, ...] = ("*static*.c3d", "*func_anat.c3d"),
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
        if segment.frame is None:
            continue
        first_axis, second_axis = segment.frame.quality_axes()
        first_vector = first_axis.vector(data)
        second_vector = second_axis.vector(data)
        first_norm = np.linalg.norm(first_vector, axis=0)
        second_norm = np.linalg.norm(second_vector, axis=0)
        dot = np.sum(first_vector * second_vector, axis=0)
        denominator = first_norm * second_norm
        cosine = np.divide(dot, denominator, out=np.full_like(dot, np.nan, dtype=float), where=denominator != 0)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        quality[segment.name] = FrameQuality(
            segment_name=segment.name,
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
        if segment.frame is None:
            continue
        first_axis, second_axis = segment.frame.quality_axes()
        first_axis, second_axis, third_name = _ordered_axes_and_third_name(first_axis, second_axis)
        first_vector = first_axis.vector(data)
        second_vector = second_axis.vector(data)
        axis_to_keep = segment.frame.axis_to_keep

        first_name = first_axis.name
        second_name = second_axis.name
        if first_name == second_name:
            raise ValueError(f"Segment '{segment.name}' defines two axes with the same name.")

        third_vector = np.cross(first_vector, second_vector, axis=0)
        if axis_to_keep == first_name:
            second_vector = np.cross(third_vector, first_vector, axis=0)
        elif axis_to_keep == second_name:
            first_vector = np.cross(second_vector, third_vector, axis=0)
        else:
            raise ValueError(f"Segment '{segment.name}' axis_to_keep must be one of the two defined axes.")

        rt = np.zeros((4, 4, data.nb_frames))
        rt[:3, first_name, :] = _normalize(first_vector)
        rt[:3, second_name, :] = _normalize(second_vector)
        rt[:3, third_name, :] = _normalize(third_vector)
        rt[:3, 3, :] = segment.frame.origin_endpoint().evaluate(data)
        rt[3, 3, :] = 1.0
        frames[segment.name] = rt
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
    return np.divide(vector, norm, out=np.full_like(vector, np.nan, dtype=float), where=norm[np.newaxis, :] != 0)
