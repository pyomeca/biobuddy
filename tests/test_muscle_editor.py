import numpy as np
import numpy.testing as npt
import pytest
from dataclasses import fields

from biobuddy import (
    MuscleEditorData,
    ViaPointEditorData,
    add_via_point,
    add_muscle,
    add_muscle_group,
    apply_insertion_editor_data,
    apply_muscle_editor_data,
    apply_origin_editor_data,
    apply_via_point_editor_data,
    get_insertion_editor_data,
    get_muscle_editor_data,
    get_origin_editor_data,
    get_via_point_editor_data,
    remove_via_point,
    remove_muscle,
    remove_muscle_group,
)
from biobuddy.components.muscle_utils import MuscleStateType, MuscleType
from biobuddy.components.real.force.muscle_real import MuscleReal
from biobuddy.components.real.force.via_point_real import ViaPointReal
from biobuddy.components.real.biomechanical_model_real import BiomechanicalModelReal
from biobuddy.components.real.rigidbody.segment_real import SegmentReal


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


def _build_via_point() -> ViaPointReal:
    return ViaPointReal(
        name="via_point",
        parent_name="Humerus",
        muscle_name="Biceps",
        muscle_group="Arm",
        position=np.array([0.1, 0.2, 0.3]),
        condition=None,
        movement=None,
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


def test_add_and_remove_muscle_group_and_muscle():
    """
    Create and remove muscle containers from the editor helpers.
    """
    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="Scapula"))
    model.add_segment(SegmentReal(name="Radius", parent_name="Scapula"))

    muscle_group = add_muscle_group(model, "Arm", "Scapula", "Radius")
    muscle = add_muscle(muscle_group, "Biceps")

    assert model.muscle_group_names == ["Arm"]
    assert muscle_group.muscle_names == ["Biceps"]
    assert muscle.origin_position.parent_name == "Scapula"
    assert muscle.insertion_position.parent_name == "Radius"

    remove_muscle(muscle_group, "Biceps")
    assert muscle_group.muscle_names == []
    remove_muscle_group(model, "Arm")
    assert model.muscle_group_names == []


def test_muscle_data_class():
    """
    Test that the MuscleEditorData class covers all the necessary fields of MuscleReal for future improvements.
    """
    editor_fields = {f"_{f.name}" for f in fields(MuscleEditorData)}
    muscle = _build_muscle()
    real_fields = set(vars(muscle).keys()) - {
        "via_points",
        "_muscle_group",
        "_origin_position",
        "_insertion_position",
        "_name",  # TODO: to be added
        "_muscle_type",  # TODO: to be added
        "_state_type",  # TODO: to be added
    }

    assert real_fields == editor_fields, (
        f"Fields mismatch between MuscleReal and MuscleEditorData.\n"
        f"  In MuscleReal but not MuscleEditorData: {real_fields - editor_fields}\n"
        f"  In MuscleEditorData but not MuscleReal: {editor_fields - real_fields}"
    )


def test_via_point_data_class():
    """
    Test that the ViaPointEditorData class covers all the necessary fields of ViaPointReal for future improvements.
    """
    editor_fields = {f"_{f.name}" for f in fields(ViaPointEditorData)}
    via_point = _build_via_point()
    real_fields = set(vars(via_point).keys()) - {
        "_muscle_name",
        "_muscle_group",
        "_condition",
        "_movement",
    }

    assert real_fields == editor_fields, (
        f"Fields mismatch between ViaPointReal and ViaPointEditorData.\n"
        f"  In ViaPointReal but not ViaPointEditorData: {real_fields - editor_fields}\n"
        f"  In ViaPointEditorData but not ViaPointReal: {editor_fields - real_fields}"
    )
