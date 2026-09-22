from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..components.generic.rigidbody.axis import Axis
from ..components.generic.rigidbody.inertia_parameters import InertiaParameters
from ..utils.enums import Rotations, Translations
from .model_builder import (
    AxisSpec,
    FunctionalAxisProjectionPointSpec,
    FunctionalAxisSpec,
    FunctionalCenterSpec,
    FunctionalMethod,
    FunctionalTrialSpec,
    LocalFrameSpec,
    MarkerAttachmentSpec,
    MarkerEndpointSpec,
    ModelTemplate,
    SegmentSpec,
)


@dataclass(frozen=True)
class Model202SegmentSpec:
    """
    Raw segment definition extracted from the BeLa Matlab configuration files.

    The Matlab files use signed marker indices for the local frame axes. Positive
    indices define one endpoint and negative indices define the other endpoint.
    Some indices are greater than the number of raw markers of the segment; those
    entries are kept as-is because they likely refer to virtual joint centers.
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
    dof: tuple[int, ...]
    rotation_sequence: str
    mass: float
    center_of_mass: tuple[float, float, float]
    inertia_diagonal: tuple[float, float, float]
    joint_determination_type: str | None = None
    joint_special_treatment: str | None = None
    functional_axis_indices: tuple[int, ...] = ()

    @property
    def unresolved_marker_indices(self) -> tuple[int, ...]:
        """
        Return local marker indices that cannot be mapped to this segment markers.
        """
        raw_indices = self.u_indices + self.v_indices + self.origin_indices + self.functional_axis_indices
        missing_indices = {abs(index) for index in raw_indices if abs(index) > len(self.marker_names)}
        return tuple(sorted(missing_indices))


@dataclass(frozen=True)
class S2mMarkerSpec:
    """
    Marker entry read from a historical ``.s2mMod`` file.
    """

    name: str
    parent_name: str
    position: tuple[float, float, float]
    is_technical: bool


@dataclass(frozen=True)
class S2mSegmentSpec:
    """
    Segment entry read from a historical ``.s2mMod`` file.
    """

    name: str
    parent_name: str
    rt: np.ndarray
    translations: str | None
    rotations: str | None
    mass: float | None
    center_of_mass: tuple[float, float, float] | None
    inertia: np.ndarray | None
    markers: tuple[S2mMarkerSpec, ...]


def model202_segment_specs() -> tuple[Model202SegmentSpec, ...]:
    """
    Return the full-body BeLa chain extracted from the Matlab configuration.
    """
    return MODEL202_SEGMENTS


def full_body_model202_template(use_functional: bool = True) -> ModelTemplate:
    """
    Return the full-body Model202/BeLa template from the Matlab configuration.

    The Matlab model appends functional points to each segment after its raw C3D
    markers: first the proximal joint center, then optional functional axis data
    for AOR joints, then distal child joint centers. The Python template resolves
    these local indices to SCoRE centers or SARA axes so the GUI can reconstruct
    the same kinematic-chain ingredients from the functional C3D trials.
    """
    return ModelTemplate(
        name="Full body Model202 from calibration C3D",
        segments=tuple(_segment_to_template(segment, use_functional=use_functional) for segment in MODEL202_SEGMENTS),
        marker_attachments=model202_marker_attachments(),
        required_static_markers=tuple(sorted(model202_marker_names())),
        functional_trials=(full_body_model202_functional_trials() if use_functional else ()),
        root_segment_name="Pelvis",
        inertia_parameters_factory=lambda data: _generic_inertia_parameters_by_segment(model202_inertia_by_segment()),
    )


def full_body_model202_functional_trials() -> tuple[FunctionalTrialSpec, ...]:
    """
    Return the functional C3D trials expected by the full-body Model202 template.
    """
    trials = []
    for segment in MODEL202_SEGMENTS:
        if segment.parent_name in {"", "base", "root"}:
            continue
        parent = _segment_by_name(segment.parent_name)
        method = FunctionalMethod.SARA_DIRECTION if segment.joint == "aor" else FunctionalMethod.SCORE
        required_markers = list(parent.marker_names)
        required_markers.extend(segment.marker_names)
        if segment.joint == "aor":
            required_markers.extend(_aor_expected_axis_markers(segment))
        trials.append(
            FunctionalTrialSpec(
                name=_functional_trial_name(segment),
                file_pattern=_functional_c3d_filename(segment),
                required_markers=tuple(dict.fromkeys(required_markers)),
                method=method,
            )
        )
    return tuple(trials)


def full_body_model202_functional_c3d_filenames() -> dict[str, str]:
    """
    Return the generic functional C3D names used by the Model202 example data.
    """
    return {
        "main": "Test_anato.c3d",
        "anatomical": "Test_anato.c3d",
        **{
            _functional_trial_name(segment): _functional_c3d_filename(segment)
            for segment in MODEL202_SEGMENTS
            if segment.parent_name not in {"", "base", "root"}
        },
    }


def rotations_from_matlab_dof(segment: Model202SegmentSpec) -> str | None:
    """
    Convert Matlab rotational DoFs to a BioMod rotation sequence.

    The signs in ``conf.S(s).dof`` are used by the Matlab pipeline to compare
    left and right sides. They do not change which axes exist in the model.
    """
    axes = "xyz"
    rotations = "".join(axis for axis, dof in zip(axes, segment.dof[:3]) if dof != 0)
    return rotations or None


def translations_from_matlab_dof(segment: Model202SegmentSpec) -> str | None:
    """
    Convert Matlab translational DoFs to a BioMod translation sequence.
    """
    if segment.name != "Pelvis":
        return None
    axes = "xyz"
    translations = "".join(axis for axis, dof in zip(axes, segment.dof[:3]) if dof != 0)
    return translations or None


def model202_marker_names() -> tuple[str, ...]:
    """
    Return all raw BeLa marker names in the Matlab order.
    """
    names = []
    for segment in MODEL202_SEGMENTS:
        names.extend(segment.marker_names)
    return tuple(names)


def model202_marker_attachments() -> tuple[MarkerAttachmentSpec, ...]:
    """
    Return one marker attachment per raw marker and owning segment.
    """
    attachments = []
    for segment in MODEL202_SEGMENTS:
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


def model202_unresolved_marker_references() -> dict[str, tuple[int, ...]]:
    """
    Report local Matlab indices that are not raw marker indices for each segment.
    """
    return {
        segment.name: segment.unresolved_marker_indices
        for segment in MODEL202_SEGMENTS
        if len(segment.unresolved_marker_indices) != 0
    }


def model202_virtual_marker_reference_map() -> dict[tuple[str, int], tuple[str, str, str]]:
    """
    Infer BeLa virtual marker names from the Matlab append-to-parent/child SCORE logic.

    Returns
    -------
    dict[tuple[str, int], tuple[str, str, str]]
        Mapping ``(segment_name, local_index)`` to ``(name, method, description)``.
    """
    reference_map: dict[tuple[str, int], tuple[str, str, str]] = {}
    segment_by_name = {segment.name: segment for segment in MODEL202_SEGMENTS}
    children_by_parent: dict[str, list[Model202SegmentSpec]] = {segment.name: [] for segment in MODEL202_SEGMENTS}
    for segment in MODEL202_SEGMENTS:
        if segment.parent_name in children_by_parent:
            children_by_parent[segment.parent_name].append(segment)
    for segment in MODEL202_SEGMENTS:
        if segment.parent_name in {"", "base", "root"} or segment.parent_name not in segment_by_name:
            continue
        parent_name = segment.parent_name
        child_index = len(segment.marker_names) + 1
        reference_map[(segment.name, child_index)] = _virtual_feature_reference(segment, parent_name, "child")
        if segment.joint_determination_type == "functional" and len(segment.functional_axis_indices) != 0:
            reference_map[(segment.name, child_index + 1)] = (
                _aor_axis_name(segment),
                "sara",
                f"SARA axis direction for {segment.name}, oriented with anatomical landmarks.",
            )
    for parent_name, children in children_by_parent.items():
        parent = segment_by_name[parent_name]
        first_child_index = len(parent.marker_names) + 1
        if parent.parent_name not in {"", "base", "root"}:
            first_child_index += 1
        if parent.joint == "aor":
            first_child_index += 1
        for child_index, child in enumerate(children, start=first_child_index):
            reference_map[(parent_name, child_index)] = _virtual_feature_reference(child, parent_name, "parent")
    return reference_map


def _virtual_feature_reference(segment: Model202SegmentSpec, parent_name: str, frame_role: str) -> tuple[str, str, str]:
    if segment.joint == "aor":
        return (
            _joint_center_name(segment),
            "axis_projection",
            f"Joint center from SARA knee axis projection, referenced from {frame_role} segment.",
        )
    return (
        _joint_center_name(segment),
        "score",
        f"CoR from functional SCORE trial, referenced from {frame_role} segment.",
    )


def signed_marker_groups(
    segment: Model202SegmentSpec,
    signed_indices: tuple[int, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Convert signed Matlab indices into two marker groups when all indices are raw markers.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        ``(negative_group, positive_group)``. Each group can be averaged to define
        one endpoint of an axis vector.
    """
    negative_group = []
    positive_group = []
    for signed_index in signed_indices:
        marker_index = abs(signed_index)
        if marker_index > len(segment.marker_names):
            raise ValueError(
                f"Segment '{segment.name}' references marker index {marker_index}, "
                f"but only {len(segment.marker_names)} raw markers are defined."
            )
        marker_name = segment.marker_names[marker_index - 1]
        if signed_index < 0:
            negative_group.append(marker_name)
        else:
            positive_group.append(marker_name)
    return tuple(negative_group), tuple(positive_group)


def _segment_to_template(segment: Model202SegmentSpec, use_functional: bool) -> SegmentSpec:
    return SegmentSpec(
        name=segment.name,
        parent_name="root" if segment.parent_name == "base" else segment.parent_name,
        translations=_translations_from_string(translations_from_matlab_dof(segment)),
        rotations=_rotations_from_string(rotations_from_matlab_dof(segment)),
        frame=LocalFrameSpec(
            origin=_endpoint_from_indices(
                segment,
                segment.origin_indices,
                role="origin",
                use_functional=use_functional,
            ),
            first_axis=_axis_from_indices(
                segment,
                segment.u_indices,
                role="u_axis",
                axis_name=_u_axis_name(segment),
                use_functional=use_functional,
            ),
            second_axis=_second_axis_from_indices(segment, use_functional=use_functional),
            axis_to_keep=_axis_name_from_label(segment.axis_label),
        ),
        mesh_points=tuple(MarkerEndpointSpec((marker_name,)) for marker_name in segment.marker_names),
    )


def _second_axis_from_indices(segment: Model202SegmentSpec, use_functional: bool) -> AxisSpec | FunctionalAxisSpec:
    axis_name = _v_axis_name(segment)
    if use_functional and segment.joint == "aor":
        return _aor_axis_spec(segment, axis_name=axis_name)
    return _axis_from_indices(
        segment,
        segment.v_indices,
        role="v_axis",
        axis_name=axis_name,
        use_functional=use_functional,
    )


def _endpoint_from_indices(
    segment: Model202SegmentSpec,
    signed_indices: tuple[int, ...],
    role: str,
    use_functional: bool,
) -> MarkerEndpointSpec | FunctionalCenterSpec | FunctionalAxisProjectionPointSpec:
    if len(signed_indices) == 0:
        return MarkerEndpointSpec((f"{segment.name}_{role}_virtual_0",))
    endpoint_names = [_marker_or_virtual_point_name(segment, abs(index), use_functional) for index in signed_indices]
    first_endpoint = endpoint_names[0]
    if all(endpoint == first_endpoint for endpoint in endpoint_names) and not isinstance(first_endpoint, str):
        return first_endpoint
    if all(isinstance(endpoint, str) for endpoint in endpoint_names):
        return MarkerEndpointSpec(tuple(endpoint_names))
    return MarkerEndpointSpec(tuple(_endpoint_display_name(endpoint) for endpoint in endpoint_names))


def _axis_from_indices(
    segment: Model202SegmentSpec,
    signed_indices: tuple[int, ...],
    role: str,
    axis_name: Axis.Name,
    use_functional: bool,
) -> AxisSpec:
    start_indices = tuple(abs(index) for index in signed_indices if index < 0)
    end_indices = tuple(abs(index) for index in signed_indices if index > 0)
    if len(start_indices) == 0 and len(end_indices) == 0:
        start_name, end_name = (
            f"{segment.name}_{role}_start",
            f"{segment.name}_{role}_end",
        )
        return AxisSpec.from_markers(axis_name, start_name, end_name)
    if len(start_indices) == 0:
        start_names = _endpoint_from_indices(
            segment,
            segment.origin_indices,
            role="axis_start",
            use_functional=use_functional,
        )
        end_names = _endpoint_from_indices(segment, end_indices, role="axis_end", use_functional=use_functional)
    elif len(end_indices) == 0:
        start_names = _endpoint_from_indices(segment, start_indices, role="axis_start", use_functional=use_functional)
        end_names = _endpoint_from_indices(
            segment,
            segment.origin_indices,
            role="axis_end",
            use_functional=use_functional,
        )
    else:
        start_names, end_names = _signed_index_groups(segment, signed_indices, use_functional=use_functional)
    return AxisSpec(name=axis_name, start=start_names, end=end_names)


def _signed_index_groups(
    segment: Model202SegmentSpec,
    signed_indices: tuple[int, ...],
    use_functional: bool,
) -> tuple[
    MarkerEndpointSpec | FunctionalCenterSpec | FunctionalAxisProjectionPointSpec,
    MarkerEndpointSpec | FunctionalCenterSpec | FunctionalAxisProjectionPointSpec,
]:
    start_indices = tuple(abs(index) for index in signed_indices if index < 0)
    end_indices = tuple(abs(index) for index in signed_indices if index > 0)
    return (
        _endpoint_from_indices(segment, start_indices, role="axis_start", use_functional=use_functional),
        _endpoint_from_indices(segment, end_indices, role="axis_end", use_functional=use_functional),
    )


def _marker_or_virtual_point_name(
    segment: Model202SegmentSpec,
    matlab_index: int,
    use_functional: bool,
) -> str | FunctionalCenterSpec | FunctionalAxisProjectionPointSpec:
    if matlab_index <= len(segment.marker_names):
        return segment.marker_names[matlab_index - 1]
    if not use_functional:
        return _virtual_marker_name_from_index(segment, matlab_index)
    return _virtual_endpoint_from_index(segment, matlab_index)


def _virtual_endpoint_from_index(
    segment: Model202SegmentSpec,
    matlab_index: int,
) -> FunctionalCenterSpec | FunctionalAxisProjectionPointSpec:
    if _is_aor_axis_index(segment, matlab_index):
        return _aor_projection_spec(segment)
    child = _child_from_parent_local_index(segment, matlab_index)
    if child is not None:
        return _aor_projection_spec(child) if child.joint == "aor" else _joint_center_spec(child)
    if segment.parent_name not in {"", "base", "root"} and matlab_index == len(segment.marker_names) + 1:
        return _aor_projection_spec(segment) if segment.joint == "aor" else _joint_center_spec(segment)
    return MarkerEndpointSpec((_virtual_marker_name_from_index(segment, matlab_index),))


def _joint_center_spec(segment: Model202SegmentSpec) -> FunctionalCenterSpec:
    parent = _segment_by_name(segment.parent_name)
    return FunctionalCenterSpec(
        method=FunctionalMethod.SCORE,
        trial_name=_functional_trial_name(segment),
        parent_marker_names=parent.marker_names,
        child_marker_names=segment.marker_names,
        fallback=MarkerEndpointSpec(segment.marker_names),
    )


def _aor_projection_spec(
    segment: Model202SegmentSpec,
) -> FunctionalAxisProjectionPointSpec:
    parent = _segment_by_name(segment.parent_name)
    expected_axis = _aor_expected_axis_spec(segment)
    expected_markers = _aor_expected_axis_markers(segment)
    return FunctionalAxisProjectionPointSpec(
        method=FunctionalMethod.SARA_DIRECTION,
        trial_name=_functional_trial_name(segment),
        parent_marker_names=parent.marker_names,
        child_marker_names=segment.marker_names,
        expected_axis=expected_axis,
        origin_marker_names=expected_markers,
        point_marker_names=expected_markers,
        fallback=MarkerEndpointSpec(expected_markers),
    )


def _aor_axis_spec(segment: Model202SegmentSpec, axis_name: Axis.Name) -> FunctionalAxisSpec:
    parent = _segment_by_name(segment.parent_name)
    expected_axis = _aor_expected_axis_spec(segment)
    return FunctionalAxisSpec(
        method=FunctionalMethod.SARA_DIRECTION,
        trial_name=_functional_trial_name(segment),
        fallback=AxisSpec.from_markers(axis_name, expected_axis.start.marker_names, expected_axis.end.marker_names),
        parent_marker_names=parent.marker_names,
        child_marker_names=segment.marker_names,
        expected_axis=expected_axis,
        origin_marker_names=_aor_expected_axis_markers(segment),
    )


def _aor_expected_axis_spec(segment: Model202SegmentSpec) -> AxisSpec:
    start_marker, end_marker = _aor_expected_axis_markers(segment)
    return AxisSpec.from_markers(_axis_name_from_label(segment.axis_label), start_marker, end_marker)


def _aor_expected_axis_markers(segment: Model202SegmentSpec) -> tuple[str, str]:
    if segment.name == "JambeD":
        return "CONDEXTD", "CONDINTD"
    if segment.name == "JambeG":
        return "CONDINTG", "CONEXTG"
    if len(segment.functional_axis_indices) >= 2:
        parent = _segment_by_name(segment.parent_name)
        return (
            parent.marker_names[segment.functional_axis_indices[0] - 1],
            parent.marker_names[segment.functional_axis_indices[1] - 1],
        )
    raise ValueError(f"Segment '{segment.name}' does not define a functional AOR orientation.")


def _child_from_parent_local_index(parent: Model202SegmentSpec, matlab_index: int) -> Model202SegmentSpec | None:
    child_index = len(parent.marker_names) + 1
    if parent.parent_name not in {"", "base", "root"}:
        child_index += 1
    if parent.joint == "aor":
        child_index += 1
    for child in _children_by_parent_name(parent.name):
        if matlab_index == child_index:
            return child
        child_index += 1
    return None


def _is_aor_axis_index(segment: Model202SegmentSpec, matlab_index: int) -> bool:
    if segment.joint != "aor":
        return False
    return segment.parent_name not in {"", "base", "root"} and matlab_index == len(segment.marker_names) + 2


def _virtual_marker_name_from_index(segment: Model202SegmentSpec, matlab_index: int) -> str:
    name, _, _ = model202_virtual_marker_reference_map().get(
        (segment.name, matlab_index),
        (f"{segment.name}_virtual_{matlab_index}", "", ""),
    )
    return name


def _endpoint_display_name(
    endpoint: str | FunctionalCenterSpec | FunctionalAxisProjectionPointSpec,
) -> str:
    if isinstance(endpoint, str):
        return endpoint
    if isinstance(endpoint, FunctionalCenterSpec):
        return _joint_center_name(_segment_by_trial_name(endpoint.trial_name))
    if isinstance(endpoint, FunctionalAxisProjectionPointSpec):
        return _joint_center_name(_segment_by_trial_name(endpoint.trial_name))
    return str(endpoint)


def _joint_center_name(segment: Model202SegmentSpec) -> str:
    return f"CoR_{segment.name}_wrt_{segment.parent_name}"


def _aor_axis_name(segment: Model202SegmentSpec) -> str:
    return f"Axis_{segment.name}_SARA"


def _functional_trial_name(segment: Model202SegmentSpec) -> str:
    method_suffix = "sara" if segment.joint == "aor" else "score"
    return f"{segment.name.lower()}_{segment.parent_name.lower()}_{method_suffix}"


def _functional_c3d_filename(segment: Model202SegmentSpec) -> str:
    return f"Test_func_{_english_segment_label(segment.name)}_{_english_segment_label(segment.parent_name)}.c3d"


def _english_segment_label(segment_name: str) -> str:
    labels = {
        "Tete": "head",
        "EpauleD": "right_shoulder",
        "BrasD": "right_arm",
        "ABrasD": "right_forearm",
        "MainD": "right_hand",
        "EpauleG": "left_shoulder",
        "BrasG": "left_arm",
        "ABrasG": "left_forearm",
        "MainG": "left_hand",
        "CuisseD": "right_thigh",
        "JambeD": "right_shank",
        "PiedD": "right_foot",
        "CuisseG": "left_thigh",
        "JambeG": "left_shank",
        "PiedG": "left_foot",
    }
    return labels.get(segment_name, segment_name.lower())


def _u_axis_name(segment: Model202SegmentSpec) -> Axis.Name:
    kept_axis_name = _axis_name_from_label(segment.axis_label)
    return kept_axis_name if segment.keep_axis_index == 1 else _complementary_axis_name(kept_axis_name)


def _v_axis_name(segment: Model202SegmentSpec) -> Axis.Name:
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
    return Rotations.NONE if rotations is None else Rotations(rotations)


def _translations_from_string(translations: str | None) -> Translations:
    return Translations.NONE if translations is None else Translations(translations)


def _segment_by_name(segment_name: str) -> Model202SegmentSpec:
    return next(segment for segment in MODEL202_SEGMENTS if segment.name == segment_name)


def _segment_by_trial_name(trial_name: str) -> Model202SegmentSpec:
    return next(segment for segment in MODEL202_SEGMENTS if _functional_trial_name(segment) == trial_name)


def _children_by_parent_name(parent_name: str) -> tuple[Model202SegmentSpec, ...]:
    return tuple(segment for segment in MODEL202_SEGMENTS if segment.parent_name == parent_name)


def model202_inertia_by_segment() -> dict[str, dict[str, np.ndarray | float]]:
    """
    Return BeLa inertial parameters in a convenient dictionary.
    """
    return _inertia_by_segment(MODEL202_INERTIAL_PARAMETERS)


def _generic_inertia_parameters_by_segment(
    inertia_by_segment: dict[str, dict[str, np.ndarray | float]],
) -> dict[str, InertiaParameters]:
    """
    Wrap historical numeric inertial parameters into generic constants.
    """
    generic_parameters = {}
    for segment_name, parameters in inertia_by_segment.items():
        mass = float(parameters["mass"])
        center_of_mass = np.asarray(parameters["center_of_mass"], dtype=float)
        inertia = np.asarray(parameters["inertia"], dtype=float)
        generic_parameters[segment_name] = InertiaParameters(
            mass=lambda data, model, value=mass: value,
            center_of_mass=lambda data, model, value=center_of_mass: value,
            inertia=lambda data, model, value=inertia: value,
            is_local=True,
        )
    return generic_parameters


def guse_inertia_by_segment() -> dict[str, dict[str, np.ndarray | float]]:
    """
    Return GuSe inertial parameters in a convenient dictionary.
    """
    return _inertia_by_segment(GUSE_INERTIAL_PARAMETERS)


def subject_inertia_by_segment(
    subject_name: str,
) -> dict[str, dict[str, np.ndarray | float]]:
    """
    Return inertial parameters for a supported subject.
    """
    subject_key = subject_name.lower()
    if subject_key == "bela":
        return model202_inertia_by_segment()
    if subject_key == "guse":
        return guse_inertia_by_segment()
    raise ValueError(f"Unsupported subject '{subject_name}'. Expected 'BeLa' or 'GuSe'.")


def parse_s2m_model(filepath: str | Path) -> tuple[S2mSegmentSpec, ...]:
    """
    Parse the subset of an old ``.s2mMod`` or BioMod-like file needed for model comparison.
    """
    lines = Path(filepath).read_text(errors="replace").splitlines()
    segments = []
    current_segment: dict | None = None
    current_marker: dict | None = None
    i_line = 0
    while i_line < len(lines):
        line = lines[i_line].strip()
        tokens = line.split()
        if len(tokens) == 0:
            i_line += 1
            continue

        keyword = tokens[0]
        if keyword == "segment":
            current_segment = {
                "name": tokens[1],
                "parent_name": "base",
                "rt": np.eye(4),
                "translations": None,
                "rotations": None,
                "mass": None,
                "center_of_mass": None,
                "inertia": None,
                "markers": [],
            }
            segments.append(current_segment)
        elif current_segment is not None and current_marker is None and keyword == "parent":
            current_segment["parent_name"] = tokens[1]
        elif current_segment is not None and keyword == "RT":
            current_segment["rt"] = np.array(
                [[float(value) for value in lines[i_line + row].strip().split()] for row in range(1, 5)],
                dtype=float,
            )
            i_line += 4
        elif current_segment is not None and keyword == "translations":
            current_segment["translations"] = tokens[1]
        elif current_segment is not None and keyword == "rotations":
            current_segment["rotations"] = tokens[1]
        elif current_segment is not None and keyword == "mass":
            current_segment["mass"] = float(tokens[1])
        elif current_segment is not None and keyword == "inertia":
            current_segment["inertia"] = np.array(
                [[float(value) for value in lines[i_line + row].strip().split()] for row in range(1, 4)],
                dtype=float,
            )
            i_line += 3
        elif current_segment is not None and keyword == "com":
            current_segment["center_of_mass"] = tuple(float(value) for value in tokens[1:4])
        elif current_segment is not None and keyword == "marker":
            current_marker = {
                "name": tokens[1],
                "parent_name": current_segment["name"],
                "position": (0.0, 0.0, 0.0),
                "is_technical": False,
            }
        elif current_segment is not None and current_marker is not None and keyword == "parent":
            current_marker["parent_name"] = tokens[1]
        elif current_segment is not None and current_marker is not None and keyword == "position":
            current_marker["position"] = tuple(float(value) for value in tokens[1:4])
        elif current_segment is not None and current_marker is not None and keyword == "technical":
            current_marker["is_technical"] = bool(int(tokens[1]))
        elif current_segment is not None and current_marker is not None and keyword == "endmarker":
            current_segment["markers"].append(S2mMarkerSpec(**current_marker))
            current_marker = None

        i_line += 1

    return tuple(
        S2mSegmentSpec(
            name=segment["name"],
            parent_name=segment["parent_name"],
            rt=segment["rt"],
            translations=segment["translations"],
            rotations=segment["rotations"],
            mass=segment["mass"],
            center_of_mass=segment["center_of_mass"],
            inertia=segment["inertia"],
            markers=tuple(segment["markers"]),
        )
        for segment in segments
    )


def _inertia_by_segment(
    inertial_parameters: dict[str, tuple[float, tuple[float, float, float], tuple[float, float, float]]],
) -> dict[str, dict[str, np.ndarray | float]]:
    return {
        segment_name: {
            "mass": mass,
            "center_of_mass": np.array(center_of_mass, dtype=float),
            "inertia": np.diag(np.array(inertia_diagonal, dtype=float)),
        }
        for segment_name, (
            mass,
            center_of_mass,
            inertia_diagonal,
        ) in inertial_parameters.items()
    }


MODEL202_INERTIAL_PARAMETERS = {
    "Pelvis": (11.5688, (0.0, 0.0, 0.1147), (0.0801, 0.1117, 0.0975)),
    "Thorax": (20.8032, (0.0, 0.0, 0.1130523729), (0.6281, 0.7118, 0.2277)),
    "Tete": (5.8472, (0.0, 0.0, 0.128), (0.1142, 0.1142, 0.0187)),
    "EpauleD": (1.6452, (0.1123, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "BrasD": (2.5570, (0.0, 0.0, -0.1425), (0.0203, 0.0203, 0.0036)),
    "ABrasD": (1.1968, (0.0, 0.0, -0.1216), (0.0074, 0.0074, 0.0048)),
    "MainD": (
        0.5401,
        (0.0201989512, -0.0490185172, -0.027307392),
        (0.0027, 0.0029, 0.0003),
    ),
    "EpauleG": (1.6452, (-0.1123, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "BrasG": (2.5570, (0.0, 0.0, -0.1425), (0.0203, 0.0203, 0.0036)),
    "ABrasG": (1.1968, (0.0, 0.0, -0.1216), (0.0074, 0.0074, 0.0048)),
    "MainG": (
        0.5401,
        (-0.0264342737, -0.0469823183, -0.0252076569),
        (0.0027, 0.0029, 0.0003),
    ),
    "CuisseD": (8.5549, (0.0, 0.0, -0.1764), (0.1211, 0.1211, 0.0321)),
    "JambeD": (4.2391, (0.0, 0.0, -0.1989), (0.0835, 0.0835, 0.0064)),
    "PiedD": (1.1323, (0.0, 0.0, -0.0476), (0.0068, 0.0066, 0.0012)),
    "CuisseG": (8.5549, (0.0, 0.0, -0.1764), (0.1211, 0.1211, 0.0321)),
    "JambeG": (4.2391, (0.0, 0.0, -0.1989), (0.0835, 0.0835, 0.0064)),
    "PiedG": (1.1323, (0.0, 0.0, -0.0476), (0.0068, 0.0066, 0.0012)),
}


GUSE_INERTIAL_PARAMETERS = {
    "Pelvis": (9.5842, (0.0, 0.0, 0.0918), (0.0477, 0.0848, 0.0778)),
    "Thorax": (17.5526, (0.0, 0.0, 0.1901), (0.4477, 0.5136, 0.1613)),
    "Tete": (4.5437, (0.0, 0.0, 0.0817), (0.0708, 0.0708, 0.0140)),
    "EpauleD": (0.4813, (0.0858, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "BrasD": (1.8084, (0.0, 0.0, -0.1112), (0.0090, 0.0090, 0.0021)),
    "ABrasD": (1.0604, (0.0, 0.0, -0.1135), (0.0065, 0.0065, 0.0007)),
    "MainD": (0.4421, (0.0, -0.0672, 0.0), (0.0012, 0.0013, 0.0002)),
    "EpauleG": (0.4813, (-0.0858, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "BrasG": (1.8084, (0.0, 0.0, -0.1112), (0.0090, 0.0090, 0.0021)),
    "ABrasG": (1.0604, (0.0, 0.0, -0.1135), (0.0065, 0.0065, 0.0007)),
    "MainG": (0.4421, (0.0, -0.0672, 0.0), (0.0012, 0.0013, 0.0002)),
    "CuisseD": (9.3755, (0.0, 0.0, -0.1732), (0.1299, 0.1699, 0.0357)),
    "JambeD": (2.7063, (0.0, 0.0, -0.1380), (0.0391, 0.0391, 0.0039)),
    "PiedD": (0.9358, (0.0, 0.0, -0.0605), (0.0054, 0.0053, 0.0008)),
    "CuisseG": (9.3755, (0.0, 0.0, -0.1732), (0.1299, 0.1699, 0.0357)),
    "JambeG": (2.7063, (0.0, 0.0, -0.1380), (0.0391, 0.0391, 0.0039)),
    "PiedG": (0.9358, (0.0, 0.0, -0.0605), (0.0054, 0.0053, 0.0008)),
}


MODEL202_SEGMENTS = (
    Model202SegmentSpec(
        name="Pelvis",
        parent_name="base",
        marker_names=("EIASD", "CID", "EIPSD", "EIPSG", "CIG", "EIASG"),
        u_indices=(1, -6),
        v_indices=(1, 6, -3, -4),
        origin_indices=(1, 3, 4, 6),
        keep_axis_index=2,
        axis_label="x",
        joint="cor",
        dof=(1, 2, 3, 4, 5, 6),
        rotation_sequence="xyz",
        mass=11.5688,
        center_of_mass=(0.0, 0.0, 0.1147),
        inertia_diagonal=(0.0801, 0.1117, 0.0975),
    ),
    Model202SegmentSpec(
        name="Thorax",
        parent_name="Pelvis",
        marker_names=("MANU", "MIDSTERNUM", "XIPHOIDE", "C7", "D3", "D10"),
        u_indices=(3, -6),
        v_indices=(4, 1, -7, -7),
        origin_indices=(7,),
        keep_axis_index=2,
        axis_label="y",
        joint="cor",
        dof=(-7, 8, 9, -7, -7, -7),
        rotation_sequence="xyz",
        mass=20.8032,
        center_of_mass=(0.0, 0.0, 0.3350),
        inertia_diagonal=(0.6281, 0.7118, 0.2277),
    ),
    Model202SegmentSpec(
        name="Tete",
        parent_name="Thorax",
        marker_names=("ZYGD", "TEMPD", "GLABELLE", "TEMPG", "ZYGG"),
        u_indices=(1, 5, -6),
        v_indices=(1, -5),
        origin_indices=(6,),
        keep_axis_index=2,
        axis_label="z",
        joint="cor",
        dof=(-10, 11, 12, -8, -8, -8),
        rotation_sequence="xyz",
        mass=5.8472,
        center_of_mass=(0.0, 0.0, 0.128),
        inertia_diagonal=(0.1142, 0.1142, 0.0187),
    ),
    Model202SegmentSpec(
        name="EpauleD",
        parent_name="Thorax",
        marker_names=("CLAV1D", "CLAV2D", "ACRANTD", "ACRPOSTD", "SCAPD"),
        u_indices=(7, -6),
        v_indices=(3, -5),
        origin_indices=(7,),
        keep_axis_index=1,
        axis_label="x",
        joint="cor",
        dof=(0, -13, -14, -9, -9, -9),
        rotation_sequence="xyz",
        mass=1.6452,
        center_of_mass=(-0.1123, 0.0, 0.0),
        inertia_diagonal=(0.0, 0.0, 0.0),
    ),
    Model202SegmentSpec(
        name="BrasD",
        parent_name="EpauleD",
        marker_names=("DELTD", "BICEPSD", "TRICEPSD", "EPICOND", "EPITROD"),
        u_indices=(6, -7),
        v_indices=(4, -5),
        origin_indices=(6,),
        keep_axis_index=1,
        axis_label="z",
        joint="cor",
        dof=(15, -16, -17, -8, -8, -8),
        rotation_sequence="xyz",
        mass=2.5570,
        center_of_mass=(0.0, 0.0, -0.1425),
        inertia_diagonal=(0.0203, 0.0203, 0.0036),
    ),
    Model202SegmentSpec(
        name="ABrasD",
        parent_name="BrasD",
        marker_names=(
            "OLE1D",
            "OLE2D",
            "BRACHD",
            "BRACHANTD",
            "ABRAPOSTD",
            "ABRASANTD",
            "ULNAD",
            "RADIUSD",
        ),
        u_indices=(9, -10),
        v_indices=(8, -7),
        origin_indices=(9,),
        keep_axis_index=1,
        axis_label="z",
        joint="cor",
        dof=(18, 0, -19, -7, -7, -7),
        rotation_sequence="xyz",
        mass=1.1968,
        center_of_mass=(0.0, 0.0, -0.1216),
        inertia_diagonal=(0.0074, 0.0074, 0.0048),
    ),
    Model202SegmentSpec(
        name="MainD",
        parent_name="ABrasD",
        marker_names=("METAC5D", "METAC2D", "MIDMETAC3D"),
        u_indices=(3,),
        v_indices=(2, -1),
        origin_indices=(4,),
        keep_axis_index=2,
        axis_label="z",
        joint="cor",
        dof=(20, -21, 0, -10, -10, -10),
        rotation_sequence="xyz",
        mass=0.5401,
        center_of_mass=(0.0, -0.0839, 0.0),
        inertia_diagonal=(0.0027, 0.0029, 0.0003),
    ),
    Model202SegmentSpec(
        name="EpauleG",
        parent_name="Thorax",
        marker_names=("CLAV1G", "CLAV2G", "CLAV3G", "ACRANTG", "ACRPOSTG", "SCAPG"),
        u_indices=(7, -8),
        v_indices=(4, -6),
        origin_indices=(7,),
        keep_axis_index=1,
        axis_label="x",
        joint="cor",
        dof=(0, 22, 23, -10, -10, -10),
        rotation_sequence="xyz",
        mass=1.6452,
        center_of_mass=(-0.1123, 0.0, 0.0),
        inertia_diagonal=(0.0, 0.0, 0.0),
    ),
    Model202SegmentSpec(
        name="BrasG",
        parent_name="EpauleG",
        marker_names=("DELTG", "BICEPSG", "TRICEPSG", "EPICONG", "EPITROG"),
        u_indices=(6, -7),
        v_indices=(5, -4),
        origin_indices=(6,),
        keep_axis_index=1,
        axis_label="z",
        joint="cor",
        dof=(24, 25, 26, -8, -8, -8),
        rotation_sequence="xyz",
        mass=2.5570,
        center_of_mass=(0.0, 0.0, -0.1425),
        inertia_diagonal=(0.0203, 0.0203, 0.0036),
    ),
    Model202SegmentSpec(
        name="ABrasG",
        parent_name="BrasG",
        marker_names=(
            "OLE1G",
            "OLE2G",
            "BRACHG",
            "BRACHANTG",
            "ABRAPOSTG",
            "ABRANTG",
            "ULNAG",
            "RADIUSG",
        ),
        u_indices=(9, -10),
        v_indices=(-8, 7),
        origin_indices=(9,),
        keep_axis_index=1,
        axis_label="z",
        joint="cor",
        dof=(27, 0, 28, -7, -7, -7),
        rotation_sequence="xyz",
        mass=1.1968,
        center_of_mass=(0.0, 0.0, -0.1216),
        inertia_diagonal=(0.0074, 0.0074, 0.0048),
    ),
    Model202SegmentSpec(
        name="MainG",
        parent_name="ABrasG",
        marker_names=("METAC5G", "METAC2G", "MIDMETAC3G"),
        u_indices=(3,),
        v_indices=(1, -2),
        origin_indices=(4,),
        keep_axis_index=2,
        axis_label="z",
        joint="cor",
        dof=(29, 30, 0, -10, -10, -10),
        rotation_sequence="xyz",
        mass=0.5401,
        center_of_mass=(0.0, -0.0839, 0.0),
        inertia_diagonal=(0.0027, 0.0029, 0.0003),
    ),
    Model202SegmentSpec(
        name="CuisseD",
        parent_name="Pelvis",
        marker_names=("ISCHIO1D", "TFLD", "ISCHIO2D", "CONDEXTD", "CONDINTD"),
        u_indices=(6, -7),
        v_indices=(4, -5),
        origin_indices=(6,),
        keep_axis_index=1,
        axis_label="z",
        joint="cor",
        dof=(31, -32, -33, -8, -8, -8),
        rotation_sequence="xyz",
        mass=8.5549,
        center_of_mass=(0.0, 0.0, -0.1764),
        inertia_diagonal=(0.1211, 0.1211, 0.0321),
    ),
    Model202SegmentSpec(
        name="JambeD",
        parent_name="CuisseD",
        marker_names=("CRETED", "JAMBLATD", "TUBD", "ACHILED", "MALEXTD", "MALINTD"),
        u_indices=(7, -9),
        v_indices=(-8,),
        origin_indices=(7,),
        keep_axis_index=2,
        axis_label="z",
        joint="aor",
        dof=(-34, 0, 0, -7, -7, -7),
        rotation_sequence="xyz",
        mass=4.2391,
        center_of_mass=(0.0, 0.0, -0.1989),
        inertia_diagonal=(0.0835, 0.0835, 0.0064),
        joint_determination_type="functional",
        joint_special_treatment="knee",
        functional_axis_indices=(5, 4),
    ),
    Model202SegmentSpec(
        name="PiedD",
        parent_name="JambeD",
        marker_names=(
            "CALCD",
            "MIDMETA4D",
            "MIDMETA1D",
            "SCAPHOIDED",
            "METAT5D",
            "METAT1D",
        ),
        u_indices=(7, 7, -5, -6),
        v_indices=(5, -6),
        origin_indices=(7,),
        keep_axis_index=1,
        axis_label="z",
        joint="cor",
        dof=(35, 0, -36, -8, -8, -8),
        rotation_sequence="xyz",
        mass=1.1323,
        center_of_mass=(0.0, 0.0, -0.0476),
        inertia_diagonal=(0.0068, 0.0066, 0.0012),
    ),
    Model202SegmentSpec(
        name="CuisseG",
        parent_name="Pelvis",
        marker_names=("ISCHIO1G", "TFLG", "ISCHIO2G", "CONEXTG", "CONDINTG"),
        u_indices=(6, -7),
        v_indices=(-4, 5),
        origin_indices=(6,),
        keep_axis_index=1,
        axis_label="z",
        joint="cor",
        dof=(37, 38, 39, -9, -9, -9),
        rotation_sequence="xyz",
        mass=8.5549,
        center_of_mass=(0.0, 0.0, -0.1764),
        inertia_diagonal=(0.1211, 0.1211, 0.0321),
    ),
    Model202SegmentSpec(
        name="JambeG",
        parent_name="CuisseG",
        marker_names=("CRETEG", "JAMBLATG", "TUBG", "ACHILLEG", "MALEXTG", "MALINTG"),
        u_indices=(7, -9),
        v_indices=(8,),
        origin_indices=(7,),
        keep_axis_index=2,
        axis_label="z",
        joint="aor",
        dof=(-40, 0, 0, -7, -7, -7),
        rotation_sequence="xyz",
        mass=4.2391,
        center_of_mass=(0.0, 0.0, -0.1989),
        inertia_diagonal=(0.0835, 0.0835, 0.0064),
        joint_determination_type="functional",
        joint_special_treatment="knee",
        functional_axis_indices=(5, 4),
    ),
    Model202SegmentSpec(
        name="PiedG",
        parent_name="JambeG",
        marker_names=(
            "CALCG",
            "MIDMETA4G",
            "MIDMETA1G",
            "SCAPHOIDEG",
            "METAT5G",
            "METAT1G",
        ),
        u_indices=(7, 7, -5, -6),
        v_indices=(-5, 6),
        origin_indices=(7,),
        keep_axis_index=1,
        axis_label="z",
        joint="cor",
        dof=(41, 0, 42, -8, -8, -8),
        rotation_sequence="xyz",
        mass=1.1323,
        center_of_mass=(0.0, 0.0, -0.0476),
        inertia_diagonal=(0.0068, 0.0066, 0.0012),
    ),
)
