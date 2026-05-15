from dataclasses import dataclass

import numpy as np

from ..components.real.force.muscle_real import MuscleReal
from ..components.real.force.via_point_real import ViaPointReal


@dataclass
class MuscleEditorData:
    """
    Editable values exposed for an existing muscle.
    """

    optimal_length: float | None
    maximal_force: float | None
    tendon_slack_length: float | None
    pennation_angle: float | None
    maximal_velocity: float | None
    maximal_excitation: float | None


@dataclass
class ViaPointEditorData:
    """
    Editable values exposed for a fixed via point.
    """

    name: str
    parent_name: str
    position: list[float]


def get_muscle_editor_data(muscle: MuscleReal) -> MuscleEditorData:
    """
    Convert a muscle into form-friendly scalar values.
    """
    return MuscleEditorData(
        optimal_length=muscle.optimal_length,
        maximal_force=muscle.maximal_force,
        tendon_slack_length=muscle.tendon_slack_length,
        pennation_angle=muscle.pennation_angle,
        maximal_velocity=muscle.maximal_velocity,
        maximal_excitation=muscle.maximal_excitation,
    )


def apply_muscle_editor_data(muscle: MuscleReal, data: MuscleEditorData) -> None:
    """
    Apply edited scalar muscle parameters.
    """
    muscle.optimal_length = data.optimal_length
    muscle.maximal_force = data.maximal_force
    muscle.tendon_slack_length = data.tendon_slack_length
    muscle.pennation_angle = data.pennation_angle
    muscle.maximal_velocity = data.maximal_velocity
    muscle.maximal_excitation = data.maximal_excitation


def get_via_point_editor_data(via_point: ViaPointReal) -> ViaPointEditorData:
    """
    Convert a via point into form-friendly values.
    """
    return ViaPointEditorData(
        name=via_point.name,
        parent_name=via_point.parent_name,
        position=via_point.position[:3, 0].tolist(),
    )


def apply_via_point_editor_data(via_point: ViaPointReal, data: ViaPointEditorData) -> None:
    """
    Apply edited fixed via-point values.
    """
    if via_point.condition is not None or via_point.movement is not None:
        raise ValueError("Only fixed via points can be edited in the desktop editor MVP.")
    via_point.name = data.name
    via_point.parent_name = data.parent_name
    via_point.position = np.array(data.position)


def add_via_point(muscle: MuscleReal, data: ViaPointEditorData) -> ViaPointReal:
    """
    Create and attach a fixed via point to a muscle.
    """
    if data.name in muscle.via_points.keys():
        raise ValueError(f"Via point '{data.name}' already exists on muscle '{muscle.name}'.")
    via_point = ViaPointReal(
        name=data.name,
        parent_name=data.parent_name,
        muscle_name=muscle.name,
        muscle_group=muscle.muscle_group,
        position=np.array(data.position),
    )
    muscle.add_via_point(via_point)
    return via_point


def remove_via_point(muscle: MuscleReal, via_point_name: str) -> None:
    """
    Remove a via point from a muscle.
    """
    muscle.remove_via_point(via_point_name)


def get_origin_editor_data(muscle: MuscleReal) -> ViaPointEditorData:
    """
    Return editable origin values for a muscle.
    """
    return get_via_point_editor_data(muscle.origin_position)


def get_insertion_editor_data(muscle: MuscleReal) -> ViaPointEditorData:
    """
    Return editable insertion values for a muscle.
    """
    return get_via_point_editor_data(muscle.insertion_position)


def apply_origin_editor_data(muscle: MuscleReal, data: ViaPointEditorData) -> None:
    """
    Apply edited origin values to a muscle.
    """
    apply_via_point_editor_data(muscle.origin_position, data)


def apply_insertion_editor_data(muscle: MuscleReal, data: ViaPointEditorData) -> None:
    """
    Apply edited insertion values to a muscle.
    """
    apply_via_point_editor_data(muscle.insertion_position, data)
