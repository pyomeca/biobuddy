from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..components.generic.rigidbody.range_of_motion import RangeOfMotion, Ranges
from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ..components.real.rigidbody.segment_real import SegmentReal
from ..utils.enums import Rotations, Translations


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
    inertia_diagonal
        The segment diagonal inertia terms.
    """

    # TODO: add name, segment_coordinate_system, dof_names, qdot_ranges, mesh, mesh_file

    parent_name: str
    translations: str
    rotations: str
    q_min: list[float]
    q_max: list[float]
    mass: float | None
    center_of_mass: list[float]
    inertia_diagonal: list[float]


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
        inertia_diagonal = [0.0, 0.0, 0.0]
    else:
        mass = segment.inertia_parameters.mass
        center_of_mass = np.nanmean(segment.inertia_parameters.center_of_mass, axis=1)[:3].tolist()
        inertia_diagonal = np.diag(segment.inertia_parameters.inertia)[:3].tolist()

    return SegmentEditorData(
        parent_name=segment.parent_name,
        translations=("" if segment.translations == Translations.NONE else segment.translations.value),
        rotations=("" if segment.rotations == Rotations.NONE else segment.rotations.value),
        q_min=q_min,
        q_max=q_max,
        mass=mass,
        center_of_mass=center_of_mass,
        inertia_diagonal=inertia_diagonal,
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

    has_inertia = data.mass is not None or any(data.center_of_mass) or any(data.inertia_diagonal)
    if not has_inertia:
        segment.inertia_parameters = None
        return

    segment.inertia_parameters = InertiaParametersReal(
        mass=data.mass,
        center_of_mass=np.array(data.center_of_mass),
        inertia=np.diag(data.inertia_diagonal),
    )


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
