import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import (
    MuscleEditorData,
    ViaPointEditorData,
    add_via_point,
    apply_insertion_editor_data,
    apply_muscle_editor_data,
    apply_origin_editor_data,
    apply_via_point_editor_data,
    get_insertion_editor_data,
    get_muscle_editor_data,
    get_origin_editor_data,
    get_via_point_editor_data,
    remove_via_point,
)
from biobuddy.components.muscle_utils import MuscleStateType, MuscleType
from biobuddy.components.real.force.muscle_real import MuscleReal
from biobuddy.components.real.force.via_point_real import ViaPointReal


def _build_muscle() -> MuscleReal:
    return MuscleReal(
        name="Biceps",
        muscle_type=MuscleType.HILL_DE_GROOTE,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="Arm",
        origin_position=ViaPointReal("origin", "Scapula", position=np.array([0.0, 0.0, 0.0])),
        insertion_position=ViaPointReal("insertion", "Radius", position=np.array([1.0, 0.0, 0.0])),
        optimal_length=0.1,
        maximal_force=100.0,
        tendon_slack_length=0.2,
        pennation_angle=0.0,
        maximal_velocity=10.0,
        maximal_excitation=1.0,
    )


def test_muscle_editor_round_trip():
    """
    Expose and apply scalar muscle parameters.
    """
    muscle = _build_muscle()
    data = get_muscle_editor_data(muscle)
    assert data.maximal_force == 100.0

    apply_muscle_editor_data(
        muscle,
        MuscleEditorData(
            optimal_length=0.2,
            maximal_force=150.0,
            tendon_slack_length=0.3,
            pennation_angle=0.1,
            maximal_velocity=12.0,
            maximal_excitation=0.9,
        ),
    )

    assert muscle.optimal_length == 0.2
    assert muscle.maximal_force == 150.0
    assert muscle.tendon_slack_length == 0.3


def test_via_point_editor_add_apply_remove():
    """
    Add, edit, and remove fixed via points from a muscle.
    """
    muscle = _build_muscle()
    via_point = add_via_point(muscle, ViaPointEditorData("vp1", "Humerus", [0.1, 0.2, 0.3]))

    assert get_via_point_editor_data(via_point) == ViaPointEditorData("vp1", "Humerus", [0.1, 0.2, 0.3])
    apply_via_point_editor_data(via_point, ViaPointEditorData("vp2", "Ulna", [1.0, 2.0, 3.0]))
    assert via_point.name == "vp2"
    npt.assert_array_equal(via_point.position[:3, 0], np.array([1.0, 2.0, 3.0]))

    muscle.via_points._remove("vp2")
    assert muscle.via_points.keys() == []


def test_via_point_editor_rejects_moving_or_conditional_points():
    """
    Keep the MVP limited to fixed via points.
    """
    muscle = _build_muscle()
    via_point = add_via_point(muscle, ViaPointEditorData("vp1", "Humerus", [0.1, 0.2, 0.3]))
    via_point.movement = object()

    with pytest.raises(ValueError, match="Only fixed via points"):
        apply_via_point_editor_data(via_point, ViaPointEditorData("vp1", "Humerus", [1.0, 2.0, 3.0]))


def test_origin_and_insertion_editor_round_trip():
    """
    Edit muscle origin and insertion values through dedicated helpers.
    """
    muscle = _build_muscle()

    assert get_origin_editor_data(muscle).parent_name == "Scapula"
    assert get_insertion_editor_data(muscle).parent_name == "Radius"

    apply_origin_editor_data(muscle, ViaPointEditorData("origin2", "Humerus", [0.1, 0.2, 0.3]))
    apply_insertion_editor_data(muscle, ViaPointEditorData("insertion2", "Ulna", [0.4, 0.5, 0.6]))

    assert muscle.origin_position.name == "origin2"
    assert muscle.origin_position.parent_name == "Humerus"
    assert muscle.insertion_position.name == "insertion2"
    assert muscle.insertion_position.parent_name == "Ulna"
