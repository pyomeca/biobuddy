from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model_builder import MarkerAttachmentSpec


@dataclass(frozen=True)
class BelaSegmentSpec:
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


def bela_segment_specs() -> tuple[BelaSegmentSpec, ...]:
    """
    Return the full-body BeLa chain extracted from the Matlab configuration.
    """
    return BELA_SEGMENTS


def rotations_from_matlab_dof(segment: BelaSegmentSpec) -> str | None:
    """
    Convert Matlab rotational DoFs to a BioMod rotation sequence.

    The signs in ``conf.S(s).dof`` are used by the Matlab pipeline to compare
    left and right sides. They do not change which axes exist in the model.
    """
    axes = "xyz"
    rotations = "".join(axis for axis, dof in zip(axes, segment.dof[:3]) if dof != 0)
    return rotations or None


def translations_from_matlab_dof(segment: BelaSegmentSpec) -> str | None:
    """
    Convert Matlab translational DoFs to a BioMod translation sequence.
    """
    if segment.name != "Pelvis":
        return None
    axes = "xyz"
    translations = "".join(axis for axis, dof in zip(axes, segment.dof[:3]) if dof != 0)
    return translations or None


def bela_marker_names() -> tuple[str, ...]:
    """
    Return all raw BeLa marker names in the Matlab order.
    """
    names = []
    for segment in BELA_SEGMENTS:
        names.extend(segment.marker_names)
    return tuple(names)


def bela_marker_attachments() -> tuple[MarkerAttachmentSpec, ...]:
    """
    Return one marker attachment per raw marker and owning segment.
    """
    attachments = []
    for segment in BELA_SEGMENTS:
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


def bela_unresolved_marker_references() -> dict[str, tuple[int, ...]]:
    """
    Report local Matlab indices that are not raw marker indices for each segment.
    """
    return {
        segment.name: segment.unresolved_marker_indices
        for segment in BELA_SEGMENTS
        if len(segment.unresolved_marker_indices) != 0
    }


def signed_marker_groups(
    segment: BelaSegmentSpec,
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


def bela_inertia_by_segment() -> dict[str, dict[str, np.ndarray | float]]:
    """
    Return BeLa inertial parameters in a convenient dictionary.
    """
    return _inertia_by_segment(BELA_INERTIAL_PARAMETERS)


def guse_inertia_by_segment() -> dict[str, dict[str, np.ndarray | float]]:
    """
    Return GuSe inertial parameters in a convenient dictionary.
    """
    return _inertia_by_segment(GUSE_INERTIAL_PARAMETERS)


def subject_inertia_by_segment(subject_name: str) -> dict[str, dict[str, np.ndarray | float]]:
    """
    Return inertial parameters for a supported subject.
    """
    subject_key = subject_name.lower()
    if subject_key == "bela":
        return bela_inertia_by_segment()
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
        for segment_name, (mass, center_of_mass, inertia_diagonal) in inertial_parameters.items()
    }


BELA_INERTIAL_PARAMETERS = {
    "Pelvis": (11.5688, (0.0, 0.0, 0.1147), (0.0801, 0.1117, 0.0975)),
    "Thorax": (20.8032, (0.0, 0.0, 0.1130523729), (0.6281, 0.7118, 0.2277)),
    "Tete": (5.8472, (0.0, 0.0, 0.128), (0.1142, 0.1142, 0.0187)),
    "EpauleD": (1.6452, (0.1123, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "BrasD": (2.5570, (0.0, 0.0, -0.1425), (0.0203, 0.0203, 0.0036)),
    "ABrasD": (1.1968, (0.0, 0.0, -0.1216), (0.0074, 0.0074, 0.0048)),
    "MainD": (0.5401, (0.0201989512, -0.0490185172, -0.027307392), (0.0027, 0.0029, 0.0003)),
    "EpauleG": (1.6452, (-0.1123, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "BrasG": (2.5570, (0.0, 0.0, -0.1425), (0.0203, 0.0203, 0.0036)),
    "ABrasG": (1.1968, (0.0, 0.0, -0.1216), (0.0074, 0.0074, 0.0048)),
    "MainG": (0.5401, (-0.0264342737, -0.0469823183, -0.0252076569), (0.0027, 0.0029, 0.0003)),
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


BELA_SEGMENTS = (
    BelaSegmentSpec(
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
        name="ABrasD",
        parent_name="BrasD",
        marker_names=("OLE1D", "OLE2D", "BRACHD", "BRACHANTD", "ABRAPOSTD", "ABRASANTD", "ULNAD", "RADIUSD"),
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
        name="ABrasG",
        parent_name="BrasG",
        marker_names=("OLE1G", "OLE2G", "BRACHG", "BRACHANTG", "ABRAPOSTG", "ABRANTG", "ULNAG", "RADIUSG"),
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
        name="PiedD",
        parent_name="JambeD",
        marker_names=("CALCD", "MIDMETA4D", "MIDMETA1D", "SCAPHOIDED", "METAT5D", "METAT1D"),
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
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
    BelaSegmentSpec(
        name="PiedG",
        parent_name="JambeG",
        marker_names=("CALCG", "MIDMETA4G", "MIDMETA1G", "SCAPHOIDEG", "METAT5G", "METAT1G"),
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
