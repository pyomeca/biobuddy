from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..components.generic.rigidbody.axis import Axis
from ..utils.enums import Rotations, Translations
from .model_builder import (
    AxisSpec,
    LocalFrameSpec,
    MarkerAttachmentSpec,
    MarkerEndpointSpec,
    ModelTemplate,
    SegmentSpec,
)


@dataclass(frozen=True)
class UpperLimbSegmentSpec:
    """
    Raw segment definition extracted from the IRSST upper-limb Matlab model 2.
    """

    name: str
    parent_name: str
    marker_names: tuple[str, ...]
    u_indices: tuple[int, ...]
    v_indices: tuple[int, ...]
    origin_indices: tuple[int, ...]
    keep_axis_index: int
    axis_label: str
    joint: str
    joint_determination_type: str
    dof: tuple[int, ...]
    rotations: str | None
    translations: str | None = None
    rotation_sequence: str = "xyz"
    mass: float | None = None
    center_of_mass: tuple[float, float, float] | None = None
    inertia_diagonal: tuple[float, float, float] | None = None
    functional_axis_indices: tuple[int, ...] = ()
    anatomical_joint_marker_index: int | None = None

    @property
    def unresolved_marker_indices(self) -> tuple[int, ...]:
        """
        Return local indices that cannot be mapped to raw markers on this segment.
        """
        raw_indices = self.u_indices + self.v_indices + self.origin_indices + self.functional_axis_indices
        missing_indices = {abs(index) for index in raw_indices if abs(index) > len(self.marker_names)}
        return tuple(sorted(missing_indices))


@dataclass(frozen=True)
class UpperLimbVirtualFeatureRequirement:
    """
    Virtual point or axis needed before the upper-limb template can be personalized.
    """

    name: str
    feature_type: str
    segment_name: str
    matlab_indices: tuple[int, ...]
    role: str


def upper_limb_segment_specs() -> tuple[UpperLimbSegmentSpec, ...]:
    """
    Return the IRSST upper-limb model-2 chain extracted from Matlab files.
    """
    return UPPER_LIMB_SEGMENTS


def upper_limb_template() -> ModelTemplate:
    """
    Return a declarative upper-limb template using explicit virtual feature placeholders.

    The historical Matlab model references several points and axes that are not
    raw C3D markers. Those are named in the template as regular marker-like
    placeholders so the C3D creation workflow can fill them with pointing,
    regression, SCORE, or SARA definitions before model generation.
    """
    return ModelTemplate(
        name="Upper-limb from calibration C3D",
        segments=tuple(_segment_to_template(segment) for segment in UPPER_LIMB_SEGMENTS),
        marker_attachments=upper_limb_marker_attachments(),
        required_static_markers=tuple(sorted(upper_limb_marker_names())),
        root_segment_name="Pelvis",
    )


def upper_limb_marker_names() -> tuple[str, ...]:
    """
    Return all raw upper-limb marker names in Matlab order.
    """
    marker_names = []
    for segment in UPPER_LIMB_SEGMENTS:
        marker_names.extend(segment.marker_names)
    return tuple(marker_names)


def upper_limb_marker_attachments() -> tuple[MarkerAttachmentSpec, ...]:
    """
    Return one marker attachment per raw marker and owning segment.
    """
    attachments = []
    for segment in UPPER_LIMB_SEGMENTS:
        for marker_name in segment.marker_names:
            attachments.append(
                MarkerAttachmentSpec(
                    name=marker_name,
                    segment_names=(segment.name,),
                    is_technical=True,
                    is_anatomical=True,
                )
            )
    return tuple(attachments)


def upper_limb_virtual_feature_requirements() -> tuple[UpperLimbVirtualFeatureRequirement, ...]:
    """
    Return virtual points and axes needed by the current upper-limb template.
    """
    requirements = []
    for segment in UPPER_LIMB_SEGMENTS:
        for role, indices in (
            ("u_axis", segment.u_indices),
            ("v_axis", segment.v_indices),
            ("origin", segment.origin_indices),
        ):
            requirements.extend(_virtual_point_requirements_from_indices(segment, indices, role))
        requirements.extend(_virtual_axis_requirements_from_indices(segment, segment.u_indices, "u_axis"))
        requirements.extend(_virtual_axis_requirements_from_indices(segment, segment.v_indices, "v_axis"))
    return tuple(_unique_requirements(requirements))


def upper_limb_unresolved_marker_references() -> dict[str, tuple[int, ...]]:
    """
    Return local Matlab indices that refer to virtual/anatomical points.
    """
    return {
        segment.name: segment.unresolved_marker_indices
        for segment in UPPER_LIMB_SEGMENTS
        if len(segment.unresolved_marker_indices) != 0
    }


def upper_limb_virtual_point_name(segment_name: str, matlab_index: int) -> str:
    """
    Return the placeholder name for a non-raw Matlab point index.
    """
    return f"{segment_name}_virtual_{matlab_index}"


def upper_limb_virtual_axis_endpoint_names(segment_name: str, axis_role: str) -> tuple[str, str]:
    """
    Return placeholder endpoint names for a virtual axis.
    """
    return f"{segment_name}_{axis_role}_start", f"{segment_name}_{axis_role}_end"


def upper_limb_inertia_by_segment() -> dict[str, dict[str, np.ndarray | float | None]]:
    """
    Return inertial parameters available in the reference Model.s2mMod.
    """
    return {
        segment.name: {
            "mass": segment.mass,
            "center_of_mass": (
                None if segment.center_of_mass is None else np.array(segment.center_of_mass, dtype=float)
            ),
            "inertia": (
                None if segment.inertia_diagonal is None else np.diag(np.array(segment.inertia_diagonal, dtype=float))
            ),
        }
        for segment in UPPER_LIMB_SEGMENTS
    }


def _segment_to_template(segment: UpperLimbSegmentSpec) -> SegmentSpec:
    return SegmentSpec(
        name=segment.name,
        parent_name="root" if segment.parent_name == "base" else segment.parent_name,
        translations=_translations_from_string(segment.translations),
        rotations=_rotations_from_string(segment.rotations),
        frame=LocalFrameSpec(
            origin=_endpoint_from_indices(segment, segment.origin_indices, role="origin"),
            first_axis=_axis_from_indices(segment, segment.u_indices, role="u_axis", axis_name=_u_axis_name(segment)),
            second_axis=_axis_from_indices(segment, segment.v_indices, role="v_axis", axis_name=_v_axis_name(segment)),
            axis_to_keep=_axis_name_from_label(segment.axis_label),
        ),
        mesh_points=tuple(MarkerEndpointSpec((marker_name,)) for marker_name in segment.marker_names),
    )


def _endpoint_from_indices(
    segment: UpperLimbSegmentSpec,
    signed_indices: tuple[int, ...],
    role: str,
) -> MarkerEndpointSpec:
    if len(signed_indices) == 0:
        return MarkerEndpointSpec((upper_limb_virtual_point_name(segment.name, 0),))
    return MarkerEndpointSpec(tuple(_marker_or_virtual_point_name(segment, abs(index)) for index in signed_indices))


def _axis_from_indices(
    segment: UpperLimbSegmentSpec,
    signed_indices: tuple[int, ...],
    role: str,
    axis_name: Axis.Name,
) -> AxisSpec:
    start_names, end_names = _signed_index_groups(segment, signed_indices)
    if len(start_names) == 0 or len(end_names) == 0:
        start_name, end_name = upper_limb_virtual_axis_endpoint_names(segment.name, role)
        return AxisSpec.from_markers(axis_name, start_name, end_name)
    return AxisSpec.from_markers(axis_name, start_names, end_names)


def _signed_index_groups(
    segment: UpperLimbSegmentSpec,
    signed_indices: tuple[int, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    start_names = []
    end_names = []
    for signed_index in signed_indices:
        point_name = _marker_or_virtual_point_name(segment, abs(signed_index))
        if signed_index < 0:
            start_names.append(point_name)
        else:
            end_names.append(point_name)
    return tuple(start_names), tuple(end_names)


def _marker_or_virtual_point_name(segment: UpperLimbSegmentSpec, matlab_index: int) -> str:
    if matlab_index <= len(segment.marker_names):
        return segment.marker_names[matlab_index - 1]
    return upper_limb_virtual_point_name(segment.name, matlab_index)


def _virtual_point_requirements_from_indices(
    segment: UpperLimbSegmentSpec,
    signed_indices: tuple[int, ...],
    role: str,
) -> tuple[UpperLimbVirtualFeatureRequirement, ...]:
    requirements = []
    for signed_index in signed_indices:
        matlab_index = abs(signed_index)
        if matlab_index > len(segment.marker_names):
            requirements.append(
                UpperLimbVirtualFeatureRequirement(
                    name=upper_limb_virtual_point_name(segment.name, matlab_index),
                    feature_type="point",
                    segment_name=segment.name,
                    matlab_indices=(matlab_index,),
                    role=role,
                )
            )
    return tuple(requirements)


def _virtual_axis_requirements_from_indices(
    segment: UpperLimbSegmentSpec,
    signed_indices: tuple[int, ...],
    role: str,
) -> tuple[UpperLimbVirtualFeatureRequirement, ...]:
    start_names, end_names = _signed_index_groups(segment, signed_indices)
    if len(start_names) != 0 and len(end_names) != 0:
        return ()
    return (
        UpperLimbVirtualFeatureRequirement(
            name=f"{segment.name}_{role}",
            feature_type="axis",
            segment_name=segment.name,
            matlab_indices=tuple(abs(index) for index in signed_indices),
            role=role,
        ),
    )


def _unique_requirements(
    requirements: list[UpperLimbVirtualFeatureRequirement],
) -> tuple[UpperLimbVirtualFeatureRequirement, ...]:
    unique = {}
    for requirement in requirements:
        unique[(requirement.feature_type, requirement.name, requirement.role)] = requirement
    return tuple(unique.values())


def _u_axis_name(segment: UpperLimbSegmentSpec) -> Axis.Name:
    kept_axis_name = _axis_name_from_label(segment.axis_label)
    return kept_axis_name if segment.keep_axis_index == 1 else _complementary_axis_name(kept_axis_name)


def _v_axis_name(segment: UpperLimbSegmentSpec) -> Axis.Name:
    kept_axis_name = _axis_name_from_label(segment.axis_label)
    return kept_axis_name if segment.keep_axis_index == 2 else _complementary_axis_name(kept_axis_name)


def _axis_name_from_label(axis_label: str) -> Axis.Name:
    axis_names = {"x": Axis.Name.X, "y": Axis.Name.Y, "z": Axis.Name.Z}
    return axis_names[axis_label.lower()]


def _complementary_axis_name(axis_name: Axis.Name) -> Axis.Name:
    if axis_name == Axis.Name.X:
        return Axis.Name.Y
    if axis_name == Axis.Name.Y:
        return Axis.Name.Z
    if axis_name == Axis.Name.Z:
        return Axis.Name.X
    raise ValueError(f"Unsupported axis name: {axis_name}.")


def _rotations_from_string(rotations: str | None) -> Rotations:
    if rotations is None:
        return Rotations.NONE
    return Rotations(rotations)


def _translations_from_string(translations: str | None) -> Translations:
    if translations is None:
        return Translations.NONE
    return Translations(translations)


UPPER_LIMB_SEGMENTS = (
    UpperLimbSegmentSpec(
        name="Pelvis",
        parent_name="base",
        marker_names=("ASISl", "ASISr", "PSISl", "PSISr"),
        u_indices=(-1, 2),
        v_indices=(1, 2, -3, -4),
        origin_indices=(1, 3, 2, 4),
        keep_axis_index=2,
        axis_label="x",
        joint="cor",
        joint_determination_type="functional",
        dof=(1, 2, 3, 4, 5, 6),
        rotations="xyz",
        translations="xyz",
    ),
    UpperLimbSegmentSpec(
        name="Thorax",
        parent_name="Pelvis",
        marker_names=("STER", "STERl", "STERr", "T1", "T10", "XIPH"),
        u_indices=(-5, 6),
        v_indices=(1, 4, -5, -6),
        origin_indices=(7,),
        keep_axis_index=1,
        axis_label="y",
        joint="cor",
        joint_determination_type="functional",
        dof=(7, 8, 9, 10, 11, 12),
        rotations="xyz",
        translations="xyz",
        mass=48.71,
        center_of_mass=(-0.000676, 0.016992, 0.123972),
        inertia_diagonal=(1.0, 1.0, 1.0),
    ),
    UpperLimbSegmentSpec(
        name="Clavicule",
        parent_name="Thorax",
        marker_names=("CLAVm", "CLAVl", "CLAV_ant", "CLAV_post", "CLAV_SC"),
        u_indices=(),
        v_indices=(-6, 7),
        origin_indices=(6,),
        keep_axis_index=2,
        axis_label="z",
        joint="cor",
        joint_determination_type="anatomical",
        dof=(-13, -14, 15, 0, 0, 0),
        rotations="zyx",
        rotation_sequence="zyx",
        anatomical_joint_marker_index=5,
    ),
    UpperLimbSegmentSpec(
        name="Scapula",
        parent_name="Clavicule",
        marker_names=("ACRO_tip", "SCAP_AA", "SCAPl", "SCAPm", "SCAP_CP", "SCAP_RS", "SCAP_SA", "SCAP_IA", "CLAV_AC"),
        u_indices=(-8, 2),
        v_indices=(-6, 2),
        origin_indices=(10,),
        keep_axis_index=2,
        axis_label="z",
        joint="cor",
        joint_determination_type="anatomical",
        dof=(-16, -17, 18, 0, 0, 0),
        rotations="zyx",
        rotation_sequence="zyx",
        anatomical_joint_marker_index=9,
    ),
    UpperLimbSegmentSpec(
        name="Arm",
        parent_name="Scapula",
        marker_names=("DELT", "ARMl", "ARMm", "ARMp_up", "ARMp_do", "EPICl", "EPICm"),
        u_indices=(8, -9),
        v_indices=(10,),
        origin_indices=(8,),
        keep_axis_index=1,
        axis_label="z",
        joint="cor",
        joint_determination_type="anatomical",
        dof=(-19, -20, -21, 22, 23, 24),
        rotations="zyz",
        translations="xyz",
        rotation_sequence="zyzz",
        mass=1.30,
        center_of_mass=(0.021139, 0.022642, -0.071303),
        inertia_diagonal=(0.001835, 0.001605, 0.000520),
        functional_axis_indices=(7, 6),
        anatomical_joint_marker_index=9,
    ),
    UpperLimbSegmentSpec(
        name="LowerArm1",
        parent_name="Arm",
        marker_names=("LARMm", "LARMl", "LARM_elb", "LARM_ant"),
        u_indices=(-1, -2, 5, 5),
        v_indices=(6,),
        origin_indices=(5,),
        keep_axis_index=2,
        axis_label="z",
        joint="aor",
        joint_determination_type="functional",
        dof=(0, 0, 25, 0, 0, 0),
        rotations="x",
        functional_axis_indices=(1, 2),
    ),
    UpperLimbSegmentSpec(
        name="LowerArm2",
        parent_name="LowerArm1",
        marker_names=("STYLr", "STYLr_up", "STYLu", "WRIST"),
        u_indices=(6,),
        v_indices=(1, -3),
        origin_indices=(5,),
        keep_axis_index=1,
        axis_label="z",
        joint="aor",
        joint_determination_type="functional",
        dof=(0, 0, -26, 0, 0, 0),
        rotations="z",
        mass=0.70,
        center_of_mass=(-0.018328, 0.010440, -0.002528),
        inertia_diagonal=(0.003520, 0.003413, 0.000457),
        functional_axis_indices=(1, 2),
    ),
    UpperLimbSegmentSpec(
        name="Hand",
        parent_name="LowerArm2",
        marker_names=("INDEX", "LASTC", "MEDH", "LATH"),
        u_indices=(4, 3, -1, -2),
        v_indices=(1, -2),
        origin_indices=(5,),
        keep_axis_index=1,
        axis_label="z",
        joint="cor",
        joint_determination_type="functional",
        dof=(-27, -28, 0, 0, 0, 0),
        rotations="xy",
        mass=0.29,
        center_of_mass=(-0.001984, -0.016656, -0.047286),
        inertia_diagonal=(0.000363, 0.000265, 0.000144),
    ),
)
