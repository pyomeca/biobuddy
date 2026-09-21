from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ..characteristics import DeLevaTable, Sex, YeadonDensitySet, YeadonSegmentName, YeadonTable
from ..components.generic.rigidbody.range_of_motion import RangeOfMotion, Ranges
from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ..components.real.rigidbody.segment_real import SegmentReal
from ..utils.enums import Rotations, Translations

DE_LEVA_MODEL_NAME = "de Leva"
YEADON_MODEL_NAME = "Yeadon"
_DE_LEVA_SIMPLE_SEGMENT_NAMES = (
    "TRUNK",
    "HEAD",
    "R_THIGH",
    "R_SHANK",
    "R_FOOT",
    "L_THIGH",
    "L_SHANK",
    "L_FOOT",
    "R_UPPER_ARM",
    "R_LOWER_ARM",
    "R_HAND",
    "L_UPPER_ARM",
    "L_LOWER_ARM",
    "L_HAND",
)


@dataclass
class SegmentEditorData:
    """
    Editable fields exposed by the first version of the model editor.

    Parameters
    ----------
    parent_name
        The parent segment name.
    translations
        The translation sequence, for example ``"xyz"`` or ``""``.
    rotations
        The rotation sequence, for example ``"xyz"`` or ``""``.
    q_min
        The minimum generalized coordinates for the segment.
    q_max
        The maximum generalized coordinates for the segment.
    mass
        The segment mass.
    center_of_mass
        The segment center of mass in local coordinates.
    inertia_matrix
        The segment inertia matrix terms.
    """

    # TODO: add name, segment_coordinate_system, dof_names, qdot_ranges, mesh, mesh_file

    parent_name: str
    translations: str
    rotations: str
    q_min: list[float]
    q_max: list[float]
    mass: float | None
    center_of_mass: list[float]
    inertia_matrix: list[list[float]]

    @property
    def inertia_diagonal(self) -> list[float]:
        """
        Return the principal moments for callers that only need diagonal terms.
        """
        return np.diag(np.asarray(self.inertia_matrix, dtype=float)).tolist()


def load_model(filepath: str) -> BiomechanicalModelReal:
    """
    Load a supported model format for edition.

    Parameters
    ----------
    filepath
        The model filepath.

    Returns
    -------
    BiomechanicalModelReal
        The parsed biomechanical model.
    """
    extension = Path(filepath).suffix.lower()
    if extension == ".biomod":
        return BiomechanicalModelReal().from_biomod(filepath=filepath)
    if extension == ".osim":
        return BiomechanicalModelReal().from_osim(filepath=filepath)
    if extension == ".urdf":
        return BiomechanicalModelReal().from_urdf(filepath=filepath)
    if extension == ".bvh":
        return BiomechanicalModelReal().from_bvh(filepath=filepath)
    raise ValueError(f"Unsupported model format '{extension}'. Expected .bioMod, .osim, .urdf, or .bvh.")


def get_segment_editor_data(segment: SegmentReal) -> SegmentEditorData:
    """
    Convert a segment into form-friendly scalar values.

    Parameters
    ----------
    segment
        The segment to expose in the GUI.
    """
    q_min = [] if segment.q_ranges is None else list(segment.q_ranges.min_bound)
    q_max = [] if segment.q_ranges is None else list(segment.q_ranges.max_bound)

    if segment.inertia_parameters is None:
        mass = None
        center_of_mass = [0.0, 0.0, 0.0]
        inertia_matrix = np.zeros((3, 3)).tolist()
    else:
        mass = segment.inertia_parameters.mass
        center_of_mass = np.nanmean(segment.inertia_parameters.center_of_mass, axis=1)[:3].tolist()
        inertia_matrix = np.asarray(segment.inertia_parameters.inertia, dtype=float)[:3, :3].tolist()

    return SegmentEditorData(
        parent_name=segment.parent_name,
        translations=("" if segment.translations == Translations.NONE else segment.translations.value),
        rotations=("" if segment.rotations == Rotations.NONE else segment.rotations.value),
        q_min=q_min,
        q_max=q_max,
        mass=mass,
        center_of_mass=center_of_mass,
        inertia_matrix=inertia_matrix,
    )


def apply_segment_editor_data(segment: SegmentReal, data: SegmentEditorData) -> None:
    """
    Apply form values to an existing segment.

    Parameters
    ----------
    segment
        The segment to mutate.
    data
        The edited values.
    """
    translations = Translations.NONE if data.translations == "" else Translations(data.translations)
    rotations = Rotations.NONE if data.rotations == "" else Rotations(data.rotations)

    segment.parent_name = data.parent_name
    segment.translations = translations
    segment.rotations = rotations
    segment.dof_names = None

    if len(data.q_min) != len(data.q_max):
        raise ValueError("The minimum and maximum range vectors must have the same length.")
    if len(data.q_min) not in {0, segment.nb_q}:
        raise ValueError(f"Expected either 0 or {segment.nb_q} range values, got {len(data.q_min)}.")
    segment.q_ranges = None if len(data.q_min) == 0 else RangeOfMotion(Ranges.Q, data.q_min, data.q_max)

    inertia_matrix = _coerce_inertia_matrix(data.inertia_matrix)
    has_inertia = data.mass is not None or any(data.center_of_mass) or np.any(inertia_matrix)
    if not has_inertia:
        segment.inertia_parameters = None
        return

    segment.inertia_parameters = InertiaParametersReal(
        mass=data.mass,
        center_of_mass=np.array(data.center_of_mass),
        inertia=inertia_matrix,
    )


def available_inertial_models() -> tuple[str, ...]:
    """
    Return the inertial tables that can be configured from the GUI.
    """
    return (DE_LEVA_MODEL_NAME, YEADON_MODEL_NAME)


def inertial_model_segment_names(model_name: str) -> tuple[str, ...]:
    """
    Return source segment names available for a GUI-selected inertial model.
    """
    if model_name == DE_LEVA_MODEL_NAME:
        return _DE_LEVA_SIMPLE_SEGMENT_NAMES
    if model_name == YEADON_MODEL_NAME:
        return tuple(segment.value for segment in YeadonSegmentName)
    raise ValueError(f"Unknown inertial model '{model_name}'.")


def build_inertial_parameters_from_model(
    model_name: str,
    segment_name: str,
    model_parameters: Mapping[str, object],
) -> InertiaParametersReal:
    """
    Build real inertial parameters for one source segment from a supported inertial table.
    """
    if model_name == DE_LEVA_MODEL_NAME:
        table = DeLevaTable(
            total_mass=float(model_parameters["total_mass"]),
            sex=Sex(str(model_parameters["sex"])),
        )
        table.from_height(total_height=float(model_parameters["total_height"]))
        model = table.to_simple_model()
        return model.segments[segment_name].inertia_parameters

    if model_name == YEADON_MODEL_NAME:
        total_mass = _optional_float(model_parameters.get("total_mass"))
        table = YeadonTable(
            model_parameters["measurements"],
            symmetric=_bool_parameter(model_parameters.get("symmetric", True)),
            density_set=str(model_parameters.get("density_set", YeadonDensitySet.DEMPSTER.value)),
            total_mass=total_mass,
        )
        return table[segment_name]

    raise ValueError(f"Unknown inertial model '{model_name}'.")


def _coerce_inertia_matrix(value: list[list[float]]) -> np.ndarray:
    inertia = np.asarray(value, dtype=float)
    if inertia.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 inertia matrix, got shape {inertia.shape}.")
    return inertia


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return float(value)


def _bool_parameter(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def validate_parent_name(model: BiomechanicalModelReal, segment_name: str, parent_name: str) -> None:
    """
    Validate that a parent assignment stays inside the model hierarchy.

    Parameters
    ----------
    model
        The edited biomechanical model.
    segment_name
        The segment whose parent is being edited.
    parent_name
        The requested parent segment.
    """
    if parent_name == segment_name:
        raise ValueError("A segment cannot be its own parent.")
    if parent_name != "base" and parent_name not in model.segment_names:
        raise ValueError(f"Unknown parent segment '{parent_name}'.")
