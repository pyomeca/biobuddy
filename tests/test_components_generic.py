import numpy as np
import numpy.testing as npt
import pytest
from lxml import etree

from biobuddy import (
    Muscle,
    ViaPoint,
    MuscleGroup,
    MuscleType,
    MuscleStateType,
    RangeOfMotion,
    Ranges,
    Marker,
    Axis,
    SegmentCoordinateSystem,
    Mesh,
    MeshFile,
    InertiaParameters,
    Contact,
    Segment,
    Translations,
    Rotations,
    RotoTransMatrix,
    SegmentCoordinateSystemUtils,
    DictData,
    BiomechanicalModelReal,
    RotoTransMatrixTimeSeries,
)
from biobuddy.components.generic.rigidbody.segment_coordinate_system import _visualize_score
from biobuddy.utils.named_list import NamedList
from test_utils import MockC3dData, get_xml_str


MOCK_RT = RotoTransMatrix.from_euler_angles_and_translation("xyz", np.array([0.1, 0.9, 0.5]), np.array([0.5, 0.5, 0.5]))


# ------- Via Point ------- #
def test_init_via_points():

    # Test initialization with default values
    via_point = ViaPoint(name="test_via_point")
    assert via_point.name == "test_via_point"
    assert via_point.parent_name is None
    via_point.parent_name = "parent1"
    assert via_point.parent_name == "parent1"
    assert via_point.muscle_name is None
    via_point.muscle_name = "muscle1"
    assert via_point.muscle_name == "muscle1"
    assert via_point.muscle_group is None
    via_point.muscle_group = "group1"
    assert via_point.muscle_group == "group1"

    # Test with string position function
    via_point = ViaPoint(name="test_via_point", position_function="marker1")
    # Call the position function with a mock marker dictionary
    mock_markers_data = DictData({"marker1": np.array([1, 2, 3, 1]).reshape(4, 1)})
    result = via_point.position_function(mock_markers_data, None)
    np.testing.assert_array_equal(
        result.reshape(
            4,
        ),
        np.array([1, 2, 3, 1]),
    )

    # Test with callable position function
    custom_func = lambda m, bio: np.array([4, 5, 6, 1])
    via_point = ViaPoint(name="test_via_point", position_function=custom_func)
    result = via_point.position_function(None, None)
    np.testing.assert_array_equal(result, np.array([4, 5, 6, 1]))


def test_to_via_point_local():
    # Mock the ViaPointReal class
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Crete a via point
    via_point = ViaPoint(
        name="test_via_point",
        parent_name="parent1",
        muscle_name="muscle1",
        muscle_group="group1",
        is_local=True,
    )
    # Not possible to evaluate the via point without a position function
    with pytest.raises(
        RuntimeError, match="You must provide a position function to evaluate the ViaPoint into a ViaPointReal."
    ):
        via_point_real = via_point.to_via_point(mock_data, mock_model)

    # Set the function
    via_point.position_function = lambda m, bio: np.mean(m.get_position(["HV"]), axis=2)
    expected_position = np.array([0.5758053, 0.60425486, 1.67896849, 1.0])
    npt.assert_almost_equal(
        np.mean(mock_data.get_position(["HV"]), axis=2).reshape(
            4,
        ),
        expected_position,
    )
    npt.assert_almost_equal(
        np.mean(mock_data.values["HV"], axis=1).reshape(
            4,
        ),
        expected_position,
    )

    # Call to_via_point
    via_point_real = via_point.to_via_point(mock_data, mock_model)
    npt.assert_almost_equal(
        via_point.position_function(mock_data, mock_model).reshape(
            4,
        ),
        expected_position,
    )
    npt.assert_almost_equal(
        np.mean(via_point_real.position, axis=1).reshape(
            4,
        ),
        expected_position,
    )

    # Set the marker name
    via_point.position_function = "HV"

    # Call to_via_point
    via_point_real = via_point.to_via_point(mock_data, mock_model)
    npt.assert_almost_equal(
        np.mean(via_point_real.position, axis=1).reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )


def test_to_via_point_global():
    # Mock the ViaPointReal class
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Crete a via point
    via_point = ViaPoint(
        name="test_via_point",
        parent_name="parent1",
        muscle_name="muscle1",
        muscle_group="group1",
        is_local=False,
    )
    # Not possible to evaluate the via point without a position function
    with pytest.raises(
        RuntimeError, match="You must provide a position function to evaluate the ViaPoint into a ViaPointReal."
    ):
        via_point_real = via_point.to_via_point(mock_data, mock_model, MOCK_RT)

    # Set the function
    via_point.position_function = lambda m, bio: np.mean(m.get_position(["HV"]), axis=2)
    expected_position = np.array([-0.65174504, 0.60837317, 0.78210787, 1.0])
    npt.assert_almost_equal(
        np.linalg.inv(MOCK_RT.rt_matrix)
        @ np.mean(mock_data.get_position(["HV"]), axis=2).reshape(
            4,
        ),
        expected_position,
    )

    # Call to_via_point
    via_point_real = via_point.to_via_point(mock_data, mock_model, MOCK_RT)
    # Make sure the position_function is in global coordinates
    npt.assert_almost_equal(
        np.mean(via_point.position_function(mock_data, mock_model), axis=1).reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )
    # And the via point real position is in local
    npt.assert_almost_equal(
        np.mean(via_point_real.position, axis=1).reshape(
            4,
        ),
        expected_position,
    )

    # Set the marker name
    via_point.position_function = "HV"

    # Call to_via_point
    via_point_real = via_point.to_via_point(mock_data, mock_model, MOCK_RT)
    npt.assert_almost_equal(
        np.mean(via_point_real.position, axis=1).reshape(
            4,
        ),
        expected_position,
    )


# ------- Muscle ------- #
def test_init_muscle():
    # Create mock functions for muscle parameters
    mock_optimal_length = lambda params, bio: 0.1
    mock_maximal_force = lambda params, bio: 100.0
    mock_tendon_slack = lambda params, bio: 0.2
    mock_pennation_angle = lambda params, bio: 0.1
    mock_maximal_velocity = lambda params, bio: 10.0

    # Create mock via points
    origin = ViaPoint(name="origin", parent_name="segment1")
    insertion = ViaPoint(name="insertion", parent_name="segment2")

    # Test initialization with default maximal_excitation
    muscle = Muscle(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="test_group",
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    assert muscle.name == "test_muscle"
    assert muscle.muscle_type == MuscleType.HILL
    assert muscle.state_type == MuscleStateType.DEGROOTE
    assert muscle.muscle_group == "test_group"
    assert muscle.origin_position == origin
    assert muscle.insertion_position == insertion
    assert muscle.maximal_excitation == 1.0
    assert isinstance(muscle.via_points, NamedList)
    assert len(muscle.via_points) == 0

    # Test with custom maximal_excitation
    muscle = Muscle(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="test_group",
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
        maximal_excitation=0.8,
    )
    assert muscle.maximal_excitation == 0.8


def test_muscle_via_points():
    # Create mock functions for muscle parameters
    mock_optimal_length = lambda params, bio: 0.1
    mock_maximal_force = lambda params, bio: 100.0
    mock_tendon_slack = lambda params, bio: 0.2
    mock_pennation_angle = lambda params, bio: 0.1
    mock_maximal_velocity = lambda params, bio: 10.0

    # Create mock via points
    origin = ViaPoint(name="origin", parent_name="segment1")
    insertion = ViaPoint(name="insertion", parent_name="segment2")

    # Create a muscle
    muscle = Muscle(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="test_group",
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    # Test adding a via point with no muscle name
    via_point1 = ViaPoint(name="via1")
    muscle.add_via_point(via_point1)

    # Check that the via point was added and muscle_name was set
    assert len(muscle.via_points) == 1
    assert via_point1.muscle_name == "test_muscle"

    # Test adding a via point with matching muscle name
    via_point2 = ViaPoint(name="via2", muscle_name="test_muscle")
    muscle.add_via_point(via_point2)
    assert len(muscle.via_points) == 2

    # Test adding a via point with non-matching muscle name
    via_point3 = ViaPoint(name="via3", muscle_name="other_muscle")
    with pytest.raises(
        ValueError,
        match="The via points's muscle should be the same as the 'key'. Alternatively, via_point.muscle_name can be left undefined",
    ):
        muscle.add_via_point(via_point3)

    # Test removing a via point
    muscle.remove_via_point("via1")
    assert len(muscle.via_points) == 1
    assert muscle.via_points[0].name == "via2"


def test_muscle_origin_insertion():
    # Create mock functions for muscle parameters
    mock_optimal_length = lambda params, bio: 0.1
    mock_maximal_force = lambda params, bio: 100.0
    mock_tendon_slack = lambda params, bio: 0.2
    mock_pennation_angle = lambda params, bio: 0.1
    mock_maximal_velocity = lambda params, bio: 10.0

    # Create a muscle with no origin/insertion initially
    muscle = Muscle(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="test_group",
        origin_position=None,
        insertion_position=None,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    # Test setting origin with no muscle name
    origin = ViaPoint(name="origin", parent_name="segment1")
    muscle.origin_position = origin
    assert muscle.origin_position == origin
    assert origin.muscle_name == "test_muscle"
    assert origin.muscle_group == "test_group"

    # Test setting insertion with no muscle name
    insertion = ViaPoint(name="insertion", parent_name="segment2")
    muscle.insertion_position = insertion
    assert muscle.insertion_position == insertion
    assert insertion.muscle_name == "test_muscle"
    assert insertion.muscle_group == "test_group"

    # Test setting origin with non-matching muscle name
    origin_bad = ViaPoint(name="origin_bad", parent_name="segment1", muscle_name="other_muscle")
    with pytest.raises(
        ValueError, match="The origin's muscle other_muscle should be the same as the muscle's name test_muscle"
    ):
        muscle.origin_position = origin_bad

    # Test setting insertion with non-matching muscle group
    insertion_bad = ViaPoint(name="insertion_bad", parent_name="segment2", muscle_group="other_group")
    with pytest.raises(
        ValueError,
        match="The insertion's muscle group other_group should be the same as the muscle's muscle group test_group",
    ):
        muscle.insertion_position = insertion_bad


def test_muscle_to_muscle_local():
    # Create mock functions for muscle parameters
    mock_optimal_length = lambda params, bio: 0.1
    mock_maximal_force = lambda params, bio: 100.0
    mock_tendon_slack = lambda params, bio: 0.2
    mock_pennation_angle = lambda params, bio: 0.1
    mock_maximal_velocity = lambda params, bio: 10.0

    # Create mock via points
    origin = ViaPoint(name="origin", parent_name="segment1", position_function="HV", is_local=True)
    insertion = ViaPoint(name="insertion", parent_name="segment2", position_function="HV", is_local=True)

    # Create a muscle
    muscle = Muscle(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="test_group",
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Call to_muscle
    muscle_real = muscle.to_muscle(mock_data, mock_model, MOCK_RT)

    # Basic verification that the conversion happened
    assert muscle_real.name == "test_muscle"
    assert muscle_real.muscle_group == "test_group"
    assert muscle_real.maximal_excitation == 1.0

    # Test the muscle parameters evaluation
    npt.assert_almost_equal(muscle_real.optimal_length, 0.1)
    npt.assert_almost_equal(muscle_real.maximal_force, 100.0)
    npt.assert_almost_equal(muscle_real.tendon_slack_length, 0.2)
    npt.assert_almost_equal(muscle_real.pennation_angle, 0.1)

    # Test the origin and insertion positions
    npt.assert_almost_equal(
        np.mean(muscle_real.origin_position.position, axis=1).reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )
    npt.assert_almost_equal(
        np.mean(muscle_real.insertion_position.position, axis=1).reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )


def test_muscle_to_muscle_global():
    # Create mock functions for muscle parameters
    mock_optimal_length = lambda params, bio: 0.1
    mock_maximal_force = lambda params, bio: 100.0
    mock_tendon_slack = lambda params, bio: 0.2
    mock_pennation_angle = lambda params, bio: 0.1
    mock_maximal_velocity = lambda params, bio: 10.0

    # Create mock via points
    origin = ViaPoint(name="origin", parent_name="segment1", position_function="HV", is_local=False)
    insertion = ViaPoint(name="insertion", parent_name="segment2", position_function="HV", is_local=False)

    # Create a muscle
    muscle = Muscle(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="test_group",
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Call to_muscle
    muscle_real = muscle.to_muscle(mock_data, mock_model, MOCK_RT)

    # Basic verification that the conversion happened
    assert muscle_real.name == "test_muscle"
    assert muscle_real.muscle_group == "test_group"
    assert muscle_real.maximal_excitation == 1.0

    # Test the muscle parameters evaluation
    npt.assert_almost_equal(muscle_real.optimal_length, 0.1)
    npt.assert_almost_equal(muscle_real.maximal_force, 100.0)
    npt.assert_almost_equal(muscle_real.tendon_slack_length, 0.2)
    npt.assert_almost_equal(muscle_real.pennation_angle, 0.1)

    # Test the origin and insertion positions
    npt.assert_almost_equal(
        np.mean(muscle_real.origin_position.position, axis=1).reshape(
            4,
        ),
        np.array([-0.65174504, 0.60837317, 0.78210787, 1.0]),
    )
    npt.assert_almost_equal(
        np.mean(muscle_real.insertion_position.position, axis=1).reshape(
            4,
        ),
        np.array([-0.65174504, 0.60837317, 0.78210787, 1.0]),
    )


def test_muscle_functions():
    # Create mock functions for muscle parameters with known return values
    mock_optimal_length = lambda params, bio: 0.15
    mock_maximal_force = lambda params, bio: 150.0
    mock_tendon_slack = lambda params, bio: 0.25
    mock_pennation_angle = lambda params, bio: 0.12
    mock_maximal_velocity = lambda params, bio: 12.0

    # Create mock via points
    origin = ViaPoint(name="origin", parent_name="segment1", position_function="HV")
    insertion = ViaPoint(name="insertion", parent_name="segment2", position_function="HV")

    # Create a muscle
    muscle = Muscle(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="test_group",
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Call to_muscle
    muscle_real = muscle.to_muscle(mock_data, mock_model, MOCK_RT)

    # Test the muscle parameters evaluation with known values
    npt.assert_almost_equal(muscle_real.optimal_length, 0.15)
    npt.assert_almost_equal(muscle_real.maximal_force, 150.0)
    npt.assert_almost_equal(muscle_real.tendon_slack_length, 0.25)
    npt.assert_almost_equal(muscle_real.pennation_angle, 0.12)


# ------- Muscle Group ------- #
def test_init_muscle_group():
    # Test initialization
    muscle_group = MuscleGroup(name="test_group", origin_parent_name="segment1", insertion_parent_name="segment2")

    assert muscle_group.name == "test_group"
    assert muscle_group.origin_parent_name == "segment1"
    assert muscle_group.insertion_parent_name == "segment2"
    assert isinstance(muscle_group.muscles, NamedList)
    assert len(muscle_group.muscles) == 0

    # Test validation - same origin and insertion
    with pytest.raises(ValueError, match="The origin and insertion parent names cannot be the same."):
        MuscleGroup(name="test_group", origin_parent_name="segment1", insertion_parent_name="segment1")


def test_muscle_group_add_remove_muscle():
    muscle_group = MuscleGroup(name="test_group", origin_parent_name="segment1", insertion_parent_name="segment2")

    # Create mock functions for muscle parameters
    mock_optimal_length = lambda params, bio: 0.1
    mock_maximal_force = lambda params, bio: 100.0
    mock_tendon_slack = lambda params, bio: 0.2
    mock_pennation_angle = lambda params, bio: 0.1
    mock_maximal_velocity = lambda params, bio: 10.0

    # Create mock via points
    origin = ViaPoint(name="origin", parent_name="segment1")
    insertion = ViaPoint(name="insertion", parent_name="segment2")

    # Create a muscle with no muscle_group
    muscle1 = Muscle(
        name="muscle1",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group=None,
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    # Add muscle to group
    muscle_group.add_muscle(muscle1)

    # Verify muscle was added and muscle_group was set
    assert len(muscle_group.muscles) == 1
    assert muscle1.muscle_group == "test_group"

    # Create a muscle with matching muscle_group
    origin = ViaPoint(name="origin", parent_name="segment1")
    insertion = ViaPoint(name="insertion", parent_name="segment2")
    muscle2 = Muscle(
        name="muscle2",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="test_group",
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    # Add muscle to group
    muscle_group.add_muscle(muscle2)
    assert len(muscle_group.muscles) == 2

    # Create a muscle with non-matching muscle_group
    origin = ViaPoint(name="origin", parent_name="segment1")
    insertion = ViaPoint(name="insertion", parent_name="segment2")
    muscle3 = Muscle(
        name="muscle3",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="other_group",
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    # Adding should raise ValueError
    with pytest.raises(
        ValueError,
        match="The muscle's muscle_group should be the same as the 'key'. Alternatively, muscle.muscle_group can be left undefined",
    ):
        muscle_group.add_muscle(muscle3)

    # Remove one muscle
    muscle_group.remove_muscle("muscle1")

    # Verify it was removed
    assert len(muscle_group.muscles) == 1
    assert muscle_group.muscles[0].name == "muscle2"


def test_muscle_group_properties():
    muscle_group = MuscleGroup(name="test_group", origin_parent_name="segment1", insertion_parent_name="segment2")

    # Create mock functions for muscle parameters
    mock_optimal_length = lambda params, bio: 0.1
    mock_maximal_force = lambda params, bio: 100.0
    mock_tendon_slack = lambda params, bio: 0.2
    mock_pennation_angle = lambda params, bio: 0.1
    mock_maximal_velocity = lambda params, bio: 10.0

    # Create mock via points
    origin = ViaPoint(name="origin", parent_name="segment1")
    insertion = ViaPoint(name="insertion", parent_name="segment2")

    # Test with empty muscle group
    assert muscle_group.nb_muscles == 0
    assert muscle_group.muscle_names == []

    # Add muscles to group
    muscle1 = Muscle(
        name="muscle1",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group=None,
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    origin = ViaPoint(name="origin", parent_name="segment1")
    insertion = ViaPoint(name="insertion", parent_name="segment2")
    muscle2 = Muscle(
        name="muscle2",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group=None,
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    muscle_group.add_muscle(muscle1)
    muscle_group.add_muscle(muscle2)

    # Test properties
    assert muscle_group.nb_muscles == 2
    assert muscle_group.muscle_names == ["muscle1", "muscle2"]


def test_muscle_group_with_via_points():
    muscle_group = MuscleGroup(name="test_group", origin_parent_name="segment1", insertion_parent_name="segment2")

    # Create mock functions for muscle parameters
    mock_optimal_length = lambda params, bio: 0.1
    mock_maximal_force = lambda params, bio: 100.0
    mock_tendon_slack = lambda params, bio: 0.2
    mock_pennation_angle = lambda params, bio: 0.1
    mock_maximal_velocity = lambda params, bio: 10.0

    # Create via points with position functions
    origin = ViaPoint(name="origin", parent_name="segment1", position_function="HV")
    insertion = ViaPoint(name="insertion", parent_name="segment2", position_function="HV")
    via1 = ViaPoint(name="via1", parent_name="segment1", position_function="HV")
    via2 = ViaPoint(name="via2", parent_name="segment2", position_function="HV")

    # Create a muscle with via points
    muscle = Muscle(
        name="muscle1",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group=None,
        origin_position=origin,
        insertion_position=insertion,
        optimal_length_function=mock_optimal_length,
        maximal_force_function=mock_maximal_force,
        tendon_slack_length_function=mock_tendon_slack,
        pennation_angle_function=mock_pennation_angle,
        maximal_velocity_function=mock_maximal_velocity,
    )

    # Add via points to the muscle
    muscle.add_via_point(via1)
    muscle.add_via_point(via2)

    # Add muscle to group
    muscle_group.add_muscle(muscle)

    # Verify muscle was added with via points
    assert len(muscle_group.muscles) == 1
    assert len(muscle_group.muscles[0].via_points) == 2

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Convert to real muscle
    muscle_real = muscle.to_muscle(mock_data, mock_model, MOCK_RT)

    # Test the via points positions
    npt.assert_almost_equal(
        np.mean(muscle_real.via_points[0].position, axis=1).reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )
    npt.assert_almost_equal(
        np.mean(muscle_real.via_points[1].position, axis=1).reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )


# ------- Range of Motion ------- #
def test_init_range_of_motion():
    # Test initialization with Q range type
    min_bound = [0.0, -1.0, -2.0]
    max_bound = [1.0, 2.0, 3.0]
    q_range = RangeOfMotion(Ranges.Q, min_bound, max_bound)

    assert q_range.range_type == Ranges.Q
    npt.assert_array_equal(q_range.min_bound, min_bound)
    npt.assert_array_equal(q_range.max_bound, max_bound)

    # Test initialization with Qdot range type
    min_bound_qdot = [-10.0, -20.0]
    max_bound_qdot = [10.0, 20.0]
    qdot_range = RangeOfMotion(Ranges.Qdot, min_bound_qdot, max_bound_qdot)

    assert qdot_range.range_type == Ranges.Qdot
    npt.assert_array_equal(qdot_range.min_bound, min_bound_qdot)
    npt.assert_array_equal(qdot_range.max_bound, max_bound_qdot)

    # Test that the min bound must be smaller
    with pytest.raises(
        ValueError, match="The min_bound must be smaller than the max_bound for each degree of freedom, got 1.0 > 0.0."
    ):
        q_range = RangeOfMotion(Ranges.Q, max_bound, min_bound)

    # Test that min and max must be of the same length
    with pytest.raises(ValueError, match="The min_bound and max_bound must have the same length, got 2 and 3."):
        q_range = RangeOfMotion(Ranges.Q, [0.0, -1.0], [1.0, 2.0, 3.0])

    # Test that range_type must be valid
    with pytest.raises(TypeError, match=r"range_type must be an instance of Ranges Enum, got \<class 'str'\>"):
        q_range = RangeOfMotion("invalid_type", min_bound, max_bound)


def test_range_of_motion_to_biomod():
    # Test Q range to_biomod
    min_bound = [0.0, -1.0, -2.0]
    max_bound = [1.0, 2.0, 3.0]
    q_range = RangeOfMotion(Ranges.Q, min_bound, max_bound)

    expected_q_output = "\trangesQ \n\t\t0.000000\t1.000000\n\t\t-1.000000\t2.000000\n\t\t-2.000000\t3.000000\n\n"
    assert q_range.to_biomod() == expected_q_output

    # Test Qdot range to_biomod
    min_bound_qdot = [-10.0, -20.0]
    max_bound_qdot = [10.0, 20.0]
    qdot_range = RangeOfMotion(Ranges.Qdot, min_bound_qdot, max_bound_qdot)

    expected_qdot_output = "\trangesQdot \n\t\t-10.000000\t10.000000\n\t\t-20.000000\t20.000000\n\n"
    assert qdot_range.to_biomod() == expected_qdot_output


def test_range_of_motion_to_urdf():
    # Test Q range to_urdf
    min_bound = [0.0, -1.0, -2.0]
    max_bound = [1.0, 2.0, 3.0]
    q_range = RangeOfMotion(Ranges.Q, min_bound, max_bound)

    # Generate xml
    fake_urdf_model = etree.Element("robot", name="fake_model")
    fake_limit = etree.SubElement(fake_urdf_model, "limit", name="fake_limit")
    q_range.to_urdf(fake_limit)
    urdf_content = get_xml_str(fake_urdf_model)
    expected_str = '<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n<robot name="fake_model">\n  <limit name="fake_limit" lower="0.0" upper="1.0"/>\n</robot>\n'
    assert urdf_content == expected_str

    with pytest.raises(
        NotImplementedError,
        match="URDF only supports Ranges.Q limits.",
    ):
        qdot_range = RangeOfMotion(Ranges.Qdot, min_bound, max_bound)
        qdot_range.to_urdf(fake_limit)


def test_range_of_motion_to_osim():
    # Test Q range to_osim
    min_bound = [-1.0]
    max_bound = [2.0]
    q_range = RangeOfMotion(Ranges.Q, min_bound, max_bound)

    assert q_range.to_osim() == (q_range.min_bound, q_range.max_bound)

    with pytest.raises(
        NotImplementedError,
        match="OpenSim only supports Ranges.Q limits.",
    ):
        qdot_range = RangeOfMotion(Ranges.Qdot, min_bound, max_bound)
        qdot_range.to_osim()


# ------- Marker ------- #
def test_init_marker():
    # Test initialization with default values
    marker = Marker(name="test_marker")
    assert marker.name == "test_marker"
    assert marker.parent_name is None
    assert marker.is_technical is True
    assert marker.is_anatomical is False

    # Test initialization with custom values
    marker = Marker(
        name="test_marker",
        function=lambda m, bio: np.array([1, 2, 3, 1]),
        parent_name="segment1",
        is_technical=False,
        is_anatomical=True,
    )
    assert marker.name == "test_marker"
    assert marker.parent_name == "segment1"
    assert marker.is_technical is False
    assert marker.is_anatomical is True

    # Test with string function
    marker = Marker(name="test_marker", function="HV")
    # Call the function with a mock marker dictionary
    mock_markers_data = DictData({"HV": np.array([1, 2, 3, 1]).reshape(4, 1)})
    result = marker.function(mock_markers_data, None)
    npt.assert_array_equal(
        result.reshape(
            4,
        ),
        np.array([1, 2, 3, 1]),
    )


def test_marker_to_marker_local():
    # Create a marker with a position function
    marker = Marker(name="SUP", parent_name="segment1", is_local=True)

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Marker without a position function is by default its name in the c3d
    marker_real = marker.to_marker(mock_data, mock_model, MOCK_RT)
    npt.assert_almost_equal(
        marker_real.position.reshape(
            4,
        ),
        np.array([0.5515919, 0.60041439, 1.37607094, 1.0]),
    )

    # Set the function
    marker.function = lambda m, bio: np.mean(m.get_position(["HV"]), axis=2)

    # Call to_via_marker
    marker_real = marker.to_marker(mock_data, mock_model, MOCK_RT)
    npt.assert_almost_equal(
        marker.function(mock_data, mock_model).reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )

    # Test the marker position
    npt.assert_almost_equal(
        marker_real.position.reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )


def test_marker_to_marker_global():
    # Create a marker with a position function
    marker = Marker(name="SUP", parent_name="segment1", is_local=False)

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Marker without a position function is by default its name in the c3d
    marker_real = marker.to_marker(mock_data, mock_model, MOCK_RT)
    npt.assert_almost_equal(
        marker_real.position.reshape(
            4,
        ),
        np.array([-0.47436502, 0.4726582, 0.57603569, 1.0]),
    )

    # Set the function
    marker.function = lambda m, bio: np.mean(m.get_position(["HV"]), axis=2)

    # Call to_via_point
    marker_real = marker.to_marker(mock_data, mock_model, MOCK_RT)
    npt.assert_almost_equal(
        marker.function(mock_data, mock_model).reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )

    # Test the marker position
    npt.assert_almost_equal(
        marker_real.position.reshape(
            4,
        ),
        np.array([-0.65174504, 0.60837317, 0.78210787, 1.0]),
    )


# ------- Axis ------- #
def test_init_axis():
    # Test initialization
    axis = Axis(name=Axis.Name.X, start="marker1", end="marker2")

    assert axis.name == Axis.Name.X
    assert isinstance(axis.start, Marker)
    assert axis.start.name == "marker1"
    assert callable(axis.start.function)
    assert isinstance(axis.end, Marker)
    assert axis.end.name == "marker2"
    assert callable(axis.end.function)


def test_axis_to_axis_global():
    # Create an axis with marker functions
    axis = Axis(name=Axis.Name.X, start="HV", end="HV")  # Using same marker for simplicity

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Convert to real axis
    axis_real = axis.to_axis(mock_data, mock_model, MOCK_RT)

    # Test the axis properties
    assert axis_real.name == Axis.Name.X

    # Test start and end positions
    npt.assert_almost_equal(
        axis_real.start_point.position.reshape(
            4,
        ),
        np.array([-0.65174504, 0.60837317, 0.78210787, 1.0]),
    )
    npt.assert_almost_equal(
        axis_real.end_point.position.reshape(
            4,
        ),
        np.array([-0.65174504, 0.60837317, 0.78210787, 1.0]),
    )


def test_axis_to_axis_local():
    # Create an axis with marker functions
    axis = Axis(
        name=Axis.Name.X, start=Marker("HV", is_local=True), end=Marker("HV", is_local=True)
    )  # Using same marker for simplicity

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Convert to real axis
    axis_real = axis.to_axis(mock_data, mock_model, MOCK_RT)

    # Test the axis properties
    assert axis_real.name == Axis.Name.X

    # Test start and end positions
    npt.assert_almost_equal(
        axis_real.start_point.position.reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )
    npt.assert_almost_equal(
        axis_real.end_point.position.reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )


# ------- Segment Coordinate System ------- #
def test_init_segment_coordinate_system():
    # Create axes for the coordinate system
    first_axis = Axis(name=Axis.Name.X, start="marker1", end="marker2")
    second_axis = Axis(name=Axis.Name.Y, start="marker1", end="marker3")

    # Test initialization
    scs = SegmentCoordinateSystem(
        origin="marker1", first_axis=first_axis, second_axis=second_axis, axis_to_keep=Axis.Name.X
    )

    assert isinstance(scs.origin, Marker)
    assert scs.first_axis == first_axis
    assert scs.second_axis == second_axis
    assert scs.axis_to_keep == Axis.Name.X


def test_segment_coordinate_system_to_scs_global():
    # Create axes for the coordinate system
    first_axis = Axis(name=Axis.Name.Z, start="HV", end="SUP")
    second_axis = Axis(name=Axis.Name.Y, start="LA", end="RA")

    # Create a segment coordinate system
    scs = SegmentCoordinateSystem(origin="HV", first_axis=first_axis, second_axis=second_axis, axis_to_keep=Axis.Name.Z)

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    result = scs.to_scs(mock_data, mock_model, MOCK_RT)

    npt.assert_almost_equal(
        result.scs.rt_matrix,
        np.array(
            [
                [0.6345035, -0.50665328, 0.58370178, -0.65174504],
                [-0.27024383, -0.85294839, -0.44659525, 0.60837317],
                [0.72413644, 0.12562444, -0.67811866, 0.78210787],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_segment_coordinate_system_to_scs_local():
    # Create axes for the coordinate system
    first_axis = Axis(name=Axis.Name.Z, start=Marker("HV", is_local=True), end=Marker("SUP", is_local=True))
    second_axis = Axis(name=Axis.Name.Y, start=Marker("LA", is_local=True), end=Marker("RA", is_local=True))

    # Create a segment coordinate system
    scs = SegmentCoordinateSystem(
        origin=Marker("HV", is_local=True), first_axis=first_axis, second_axis=second_axis, axis_to_keep=Axis.Name.Z
    )

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    result = scs.to_scs(mock_data, mock_model, MOCK_RT)

    npt.assert_almost_equal(
        result.scs.rt_matrix,
        np.array(
            [
                [0.99390305, 0.07621052, -0.07967867, 0.5758053],
                [0.07544024, -0.99707024, -0.01263779, 0.60425486],
                [-0.08040836, 0.00654976, -0.99674049, 1.67896849],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


# ------- SegmentCoordinateSystemUtils ------- #
def test_rigidify():
    np.random.seed(0)
    nb_frames = 100
    markers = {
        "marker1": np.array([0.05, 0.12, -0.19, 1]),
        "marker2": np.array([0.04, 0.01, 0.16, 1]),
        "marker3": np.array([0.01, -0.02, 0.04, 1]),
        "marker4": np.array([-0.05, 0.10, -0.09, 1]),
        "marker5": np.array([0.15, 0.15, 0.03, 1]),
        "marker6": np.array([-0.001, 0.07, 0.10, 1]),
    }

    # Define a set of markers
    mock_static_data = DictData(markers)
    expected_euler = np.array([1.0, 0.0, 0.0])
    fake_functional_trial = {name: np.ones((4, nb_frames)) for name in markers.keys()}
    for name in markers.keys():
        rt_this_marker = RotoTransMatrix().from_euler_angles_and_translation(
            "xyz",
            expected_euler,
            np.zeros(
                4,
            ),
        )
        for i_frame in range(nb_frames):
            fake_functional_trial[name][:, i_frame] = np.reshape(
                rt_this_marker
                @ (
                    markers[name].reshape(
                        4,
                    )
                    + np.random.random((4,)) * 0.1
                    - 0.05
                ),
                (4,),
            )
    mock_functional_data = DictData(fake_functional_trial)

    # Rigidify the markers with a static trial
    rigidified_rt = SegmentCoordinateSystemUtils.rigidify(
        functional_data=mock_functional_data,
        static_data=mock_static_data,
    )

    # Check the translation
    centroid_functional_marker_position = np.mean(
        mock_functional_data.markers_center_position(mock_functional_data.marker_names), axis=1
    )
    centroid_static_marker_position = mock_static_data.markers_center_position(mock_static_data.marker_names).reshape(
        4,
    )
    # There are differences due to the random noise added, but they should be small
    assert np.all(
        np.abs(
            np.reshape(rt_this_marker.inverse @ centroid_functional_marker_position, (4,))
            - centroid_static_marker_position
        )
        < 0.01
    )
    assert np.all(
        np.abs(rigidified_rt.mean_homogenous_matrix().translation - centroid_functional_marker_position[:3]) < 0.0001
    )
    assert np.all(
        np.abs(
            rigidified_rt.mean_homogenous_matrix().translation
            - np.reshape(rt_this_marker @ centroid_static_marker_position, (4,))[:3]
        )
        < 0.01
    )

    # Check the rotation (the sign of the Euler angles is inverted)
    assert np.all(np.abs(rigidified_rt.mean_homogenous_matrix().euler_angles("xyz") - -expected_euler) < 0.02)


def test_mean_markers():
    # Create mock marker data
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Test mean_markers with a single marker
    mean_func = SegmentCoordinateSystemUtils.mean_markers(["HV"])
    result = mean_func(mock_data, mock_model)

    # Should return the mean position of HV marker
    expected = np.mean(mock_data.get_position(["HV"]), axis=2)[:, 0]
    npt.assert_almost_equal(result, expected)

    # Test mean_markers with multiple markers
    mean_func = SegmentCoordinateSystemUtils.mean_markers(["HV", "STR", "SUP"])
    result = mean_func(mock_data, mock_model)

    # Should return the mean position of all three markers
    expected = np.nanmean(mock_data.markers_center_position(["HV", "STR", "SUP"]), axis=1)
    npt.assert_almost_equal(result, expected)


def test_score():
    # Create mock marker data for parent and child segments
    np.random.seed(42)
    nb_frames = 50

    # Create parent markers (relatively stationary)
    parent_markers = {
        "parent1": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.1], [0.2], [0.3], [1.0]]),
        "parent2": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.15], [0.25], [0.35], [1.0]]),
        "parent3": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.2], [0.3], [0.4], [1.0]]),
    }

    # Create child markers (moving relative to rt)
    rt_matrix_time_series = RotoTransMatrixTimeSeries(nb_frames)
    for i_frame in range(nb_frames):
        rt_matrix_time_series[i_frame] = RotoTransMatrix().from_euler_angles_and_translation(
            "xyz", np.array([i_frame * 0.01, 0, i_frame * 0.02]), np.array([0.2, 0.2, 0.2])
        )
    child_markers = {
        "child1": rt_matrix_time_series
        @ (np.random.randn(4, nb_frames) * 0.01 + np.array([[0.3], [0.4], [0.5], [1.0]])),
        "child2": rt_matrix_time_series
        @ (np.random.randn(4, nb_frames) * 0.01 + np.array([[0.35], [0.45], [0.55], [1.0]])),
        "child3": rt_matrix_time_series
        @ (np.random.randn(4, nb_frames) * 0.01 + np.array([[0.4], [0.5], [0.6], [1.0]])),
    }

    all_markers = {**parent_markers, **child_markers}
    functional_data = DictData(all_markers)

    # Create static data (first frame)
    static_markers = {name: data[:, 0:1] for name, data in all_markers.items()}
    static_data = DictData(static_markers)

    # Create score function
    score_func = SegmentCoordinateSystemUtils.score(
        functional_data=functional_data,
        parent_marker_names=["parent1", "parent2", "parent3"],
        child_marker_names=["child1", "child2", "child3"],
        visualize=False,
    )

    # Call the score function
    mock_model = BiomechanicalModelReal()
    result_cor = score_func(static_data, mock_model)
    # CoR close to [0.35, 0.45, 0.55] + [0.2, 0.2, 0.2]
    npt.assert_almost_equal(result_cor, np.array([0.56213852, 0.62106317, 0.73310333, 1.0]))

    # Test that calling twice returns the same result (caching)
    result_cor2 = score_func(static_data, mock_model)
    npt.assert_array_equal(result_cor, result_cor2)


def test_sara():
    # Create mock marker data for parent and child segments
    np.random.seed(42)
    nb_frames = 50

    # Create parent markers (relatively stationary)
    parent_markers = {
        "parent1": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.1], [0.2], [0.3], [1.0]]),
        "parent2": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.15], [0.25], [0.35], [1.0]]),
        "parent3": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.2], [0.3], [0.4], [1.0]]),
    }

    # Create child markers (moving relative to rt)
    rt_matrix_time_series = RotoTransMatrixTimeSeries(nb_frames)
    for i_frame in range(nb_frames):
        # Only rotate on the X-axis (with a little something on the Z-axis)
        rt_matrix_time_series[i_frame] = RotoTransMatrix().from_euler_angles_and_translation(
            "xyz", np.array([i_frame * 0.05, 0, i_frame * 0.0001]), np.array([0.2, 0.2, 0.2])
        )
    child_markers = {
        "child1": rt_matrix_time_series
        @ (np.random.randn(4, nb_frames) * 0.01 + np.array([[0.3], [0.4], [0.5], [1.0]])),
        "child2": rt_matrix_time_series
        @ (np.random.randn(4, nb_frames) * 0.01 + np.array([[0.35], [0.45], [0.55], [1.0]])),
        "child3": rt_matrix_time_series
        @ (np.random.randn(4, nb_frames) * 0.01 + np.array([[0.4], [0.5], [0.6], [1.0]])),
    }

    all_markers = {**parent_markers, **child_markers}
    functional_data = DictData(all_markers)

    # Create static data (first frame)
    static_markers = {name: data[:, 0:1] for name, data in all_markers.items()}
    static_data = DictData(static_markers)

    # Create SARA axis
    sara_axis = SegmentCoordinateSystemUtils.sara(
        name=Axis.Name.X,
        functional_data=functional_data,
        parent_marker_names=["parent1", "parent2", "parent3"],
        child_marker_names=["child1", "child2", "child3"],
        visualize=False,
    )

    # Verify it returns an Axis object
    assert isinstance(sara_axis, Axis)
    assert sara_axis.name == Axis.Name.X

    # Evaluate the sara function
    mock_model = BiomechanicalModelReal()
    result_aor = sara_axis.to_axis(static_data, mock_model, scs=RotoTransMatrix())

    # TODO: @pariterre -> the AoR is suspicious ?
    npt.assert_almost_equal(
        result_aor.start_point.position.reshape(
            4,
        ),
        np.array([0.4174604, 0.48005109, 0.55874133, 1.0]),
    )
    npt.assert_almost_equal(
        result_aor.end_point.position.reshape(
            4,
        ),
        np.array([0.81595683, 0.8115111, 0.82951004, 1.0]),
    )
    npt.assert_almost_equal(
        result_aor.axis().reshape(
            4,
        ),
        np.array([0.39849643, 0.33146001, 0.27076872, 0.0]),
    )

    # Test that calling twice returns the same result (caching)
    result_aor2 = sara_axis.to_axis(static_data, mock_model, scs=RotoTransMatrix())
    npt.assert_array_equal(result_aor.start_point.position, result_aor2.start_point.position)
    npt.assert_array_equal(result_aor.end_point.position, result_aor2.end_point.position)
    npt.assert_array_equal(result_aor.axis(), result_aor2.axis())


def test_visualize_score_with_point():
    """Test the structure of the _visualize_score plot with a point (SCoRE)"""
    np.random.seed(42)
    nb_frames = 10

    # Create simple marker data
    parent_markers = {
        "parent1": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.1], [0.2], [0.3], [1.0]]),
        "parent2": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.15], [0.25], [0.35], [1.0]]),
    }

    child_markers = {
        "child1": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.3], [0.4], [0.5], [1.0]]),
        "child2": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.35], [0.45], [0.55], [1.0]]),
    }

    all_markers = {**parent_markers, **child_markers}
    data = DictData(all_markers)

    # Create RT matrices
    rt_parent = RotoTransMatrixTimeSeries(nb_frames)
    rt_child = RotoTransMatrixTimeSeries(nb_frames)
    for i in range(nb_frames):
        rt_parent[i] = RotoTransMatrix.from_euler_angles_and_translation(
            "xyz", np.array([0.1 * i, 0, 0]), np.array([0.1, 0.2, 0.3])
        )
        rt_child[i] = RotoTransMatrix.from_euler_angles_and_translation(
            "xyz", np.array([0.2 * i, 0, 0]), np.array([0.3, 0.4, 0.5])
        )

    # Create a center of rotation point
    cor_global = np.random.randn(4, nb_frames) * 0.01 + np.array([[0.2], [0.3], [0.4], [1.0]])

    # Call the visualization function
    figure = _visualize_score(data, rt_parent, rt_child, cor_global)

    # Check that we have frames
    assert len(figure.frames) == nb_frames

    # Check the first frame structure
    first_frame_data = figure.frames[0].data

    # Should have:
    # - 3 parent RT axes (red, green, blue)
    # - 3 child RT axes (red, green, blue)
    # - 1 marker scatter
    # - 1 CoR point scatter
    assert len(first_frame_data) == 8

    # Check parent axes (first 3 traces)
    assert first_frame_data[0].line.color == "red"
    npt.assert_almost_equal(np.array(first_frame_data[0].x), np.array([0.1, 0.15]))
    npt.assert_almost_equal(np.array(first_frame_data[0].y), np.array([0.2, 0.2]))
    npt.assert_almost_equal(np.array(first_frame_data[0].z), np.array([0.3, 0.3]))
    assert first_frame_data[1].line.color == "green"
    npt.assert_almost_equal(np.array(first_frame_data[1].x), np.array([0.1, 0.1]))
    npt.assert_almost_equal(np.array(first_frame_data[1].y), np.array([0.2, 0.25]))
    npt.assert_almost_equal(np.array(first_frame_data[1].z), np.array([0.3, 0.3]))
    assert first_frame_data[2].line.color == "blue"
    npt.assert_almost_equal(np.array(first_frame_data[2].x), np.array([0.1, 0.1]))
    npt.assert_almost_equal(np.array(first_frame_data[2].y), np.array([0.2, 0.2]))
    npt.assert_almost_equal(np.array(first_frame_data[2].z), np.array([0.3, 0.35]))

    # Check child axes (next 3 traces)
    assert first_frame_data[3].line.color == "red"
    npt.assert_almost_equal(np.array(first_frame_data[3].x), np.array([0.3, 0.35]))
    npt.assert_almost_equal(np.array(first_frame_data[3].y), np.array([0.4, 0.4]))
    npt.assert_almost_equal(np.array(first_frame_data[3].z), np.array([0.5, 0.5]))
    assert first_frame_data[4].line.color == "green"
    npt.assert_almost_equal(np.array(first_frame_data[4].x), np.array([0.3, 0.3]))
    npt.assert_almost_equal(np.array(first_frame_data[4].y), np.array([0.4, 0.45]))
    npt.assert_almost_equal(np.array(first_frame_data[4].z), np.array([0.5, 0.5]))
    assert first_frame_data[5].line.color == "blue"
    npt.assert_almost_equal(np.array(first_frame_data[5].x), np.array([0.3, 0.3]))
    npt.assert_almost_equal(np.array(first_frame_data[5].y), np.array([0.4, 0.4]))
    npt.assert_almost_equal(np.array(first_frame_data[5].z), np.array([0.5, 0.55]))

    # Check markers scatter
    assert first_frame_data[6].mode == "markers"
    assert first_frame_data[6].marker.color == "blue"
    npt.assert_almost_equal(np.array(first_frame_data[6].x), np.array([0.10496714, 0.19536582, 0.31465649, 0.99398293]))
    npt.assert_almost_equal(np.array(first_frame_data[6].y), np.array([0.15738467, 0.25324084, 0.34520826, 1.00361396]))
    npt.assert_almost_equal(np.array(first_frame_data[6].z), np.array([0.29780328, 0.40097078, 0.48584629, 0.98081229]))

    # Check CoR point
    assert first_frame_data[7].mode == "markers"
    assert first_frame_data[7].marker.color == "red"
    assert first_frame_data[7].marker.size == 5
    npt.assert_almost_equal(np.array(first_frame_data[7].x), np.array([0.1902531832977268]))
    npt.assert_almost_equal(np.array(first_frame_data[7].y), np.array([0.2911048557037448]))
    npt.assert_almost_equal(np.array(first_frame_data[7].z), np.array([0.4062566734776501]))

    # Check layout
    assert figure.layout.title.text == "Score Point Visualization"
    assert figure.layout.scene.xaxis.title.text == "X"
    assert figure.layout.scene.yaxis.title.text == "Y"
    assert figure.layout.scene.zaxis.title.text == "Z"
    assert figure.layout.scene.aspectmode == "data"

    # Check sliders
    assert len(figure.layout.sliders) == 1
    assert len(figure.layout.sliders[0].steps) == nb_frames


def test_visualize_score_with_axis():
    """Test the structure of the _visualize_score plot with an axis (SARA)"""
    np.random.seed(42)
    nb_frames = 10

    # Create simple marker data
    parent_markers = {
        "parent1": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.1], [0.2], [0.3], [1.0]]),
        "parent2": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.15], [0.25], [0.35], [1.0]]),
    }

    child_markers = {
        "child1": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.3], [0.4], [0.5], [1.0]]),
        "child2": np.random.randn(4, nb_frames) * 0.01 + np.array([[0.35], [0.45], [0.55], [1.0]]),
    }

    all_markers = {**parent_markers, **child_markers}
    data = DictData(all_markers)

    # Create RT matrices
    rt_parent = RotoTransMatrixTimeSeries(nb_frames)
    rt_child = RotoTransMatrixTimeSeries(nb_frames)
    for i in range(nb_frames):
        rt_parent[i] = RotoTransMatrix.from_euler_angles_and_translation(
            "xyz", np.array([0.1 * i, 0, 0]), np.array([0.1, 0.2, 0.3])
        )
        rt_child[i] = RotoTransMatrix.from_euler_angles_and_translation(
            "xyz", np.array([0.2 * i, 0, 0]), np.array([0.3, 0.4, 0.5])
        )

    # Create axis of rotation (start and end points)
    start_aor = np.random.randn(4, nb_frames) * 0.01 + np.array([[0.2], [0.3], [0.4], [1.0]])
    end_aor = np.random.randn(4, nb_frames) * 0.01 + np.array([[0.3], [0.4], [0.5], [1.0]])
    cor_global = [start_aor, end_aor]

    # Call the visualization function
    figure = _visualize_score(data, rt_parent, rt_child, cor_global)

    # Check that we have frames
    assert len(figure.frames) == nb_frames

    # Check the first frame structure
    first_frame_data = figure.frames[0].data

    # Should have:
    # - 3 parent RT axes (red, green, blue)
    # - 3 child RT axes (red, green, blue)
    # - 1 marker scatter
    # - 1 AoR line
    assert len(first_frame_data) == 8

    # Check parent axes (first 3 traces)
    assert first_frame_data[0].line.color == "red"
    npt.assert_almost_equal(np.array(first_frame_data[0].x), np.array([0.1, 0.15]))
    npt.assert_almost_equal(np.array(first_frame_data[0].y), np.array([0.2, 0.2]))
    npt.assert_almost_equal(np.array(first_frame_data[0].z), np.array([0.3, 0.3]))
    assert first_frame_data[1].line.color == "green"
    npt.assert_almost_equal(np.array(first_frame_data[1].x), np.array([0.1, 0.1]))
    npt.assert_almost_equal(np.array(first_frame_data[1].y), np.array([0.2, 0.25]))
    npt.assert_almost_equal(np.array(first_frame_data[1].z), np.array([0.3, 0.3]))
    assert first_frame_data[2].line.color == "blue"
    npt.assert_almost_equal(np.array(first_frame_data[2].x), np.array([0.1, 0.1]))
    npt.assert_almost_equal(np.array(first_frame_data[2].y), np.array([0.2, 0.2]))
    npt.assert_almost_equal(np.array(first_frame_data[2].z), np.array([0.3, 0.35]))

    # Check child axes (next 3 traces)
    assert first_frame_data[3].line.color == "red"
    npt.assert_almost_equal(np.array(first_frame_data[3].x), np.array([0.3, 0.35]))
    npt.assert_almost_equal(np.array(first_frame_data[3].y), np.array([0.4, 0.4]))
    npt.assert_almost_equal(np.array(first_frame_data[3].z), np.array([0.5, 0.5]))
    assert first_frame_data[4].line.color == "green"
    npt.assert_almost_equal(np.array(first_frame_data[4].x), np.array([0.3, 0.3]))
    npt.assert_almost_equal(np.array(first_frame_data[4].y), np.array([0.4, 0.45]))
    npt.assert_almost_equal(np.array(first_frame_data[4].z), np.array([0.5, 0.5]))
    assert first_frame_data[5].line.color == "blue"
    npt.assert_almost_equal(np.array(first_frame_data[5].x), np.array([0.3, 0.3]))
    npt.assert_almost_equal(np.array(first_frame_data[5].y), np.array([0.4, 0.4]))
    npt.assert_almost_equal(np.array(first_frame_data[5].z), np.array([0.5, 0.55]))

    # Check markers scatter
    assert first_frame_data[6].mode == "markers"
    assert first_frame_data[6].marker.color == "blue"
    npt.assert_almost_equal(np.array(first_frame_data[6].x), np.array([0.10496714, 0.19536582, 0.31465649, 0.99398293]))
    npt.assert_almost_equal(np.array(first_frame_data[6].y), np.array([0.15738467, 0.25324084, 0.34520826, 1.00361396]))
    npt.assert_almost_equal(np.array(first_frame_data[6].z), np.array([0.29780328, 0.40097078, 0.48584629, 0.98081229]))

    # Check AoR line
    assert first_frame_data[7].mode == "lines"
    assert first_frame_data[7].line.color == "red"
    assert first_frame_data[7].line.width == 5
    npt.assert_almost_equal(np.array(first_frame_data[7].x), np.array([0.19025318, 0.30357787]))
    npt.assert_almost_equal(np.array(first_frame_data[7].y), np.array([0.29110486, 0.40570891]))
    npt.assert_almost_equal(np.array(first_frame_data[7].z), np.array([0.40625667, 0.52314659]))

    # Check layout
    assert figure.layout.title.text == "Score Point Visualization"
    assert figure.layout.scene.xaxis.title.text == "X"
    assert figure.layout.scene.yaxis.title.text == "Y"
    assert figure.layout.scene.zaxis.title.text == "Z"
    assert figure.layout.scene.aspectmode == "data"

    # Check sliders
    assert len(figure.layout.sliders) == 1
    assert len(figure.layout.sliders[0].steps) == nb_frames


# ------- Mesh ------- #
def test_init_mesh():
    # Test initialization with string functions
    mesh = Mesh(functions=("marker1", "marker2", "marker3"))
    assert len(mesh.functions) == 3

    # Test initialization with callable functions
    func1 = lambda m, bio: np.array([1, 2, 3, 1])
    func2 = lambda m, bio: np.array([4, 5, 6, 1])
    mesh = Mesh(functions=(func1, func2))
    assert len(mesh.functions) == 2

    # Test mixed initialization
    mesh = Mesh(functions=("marker1", func1))
    assert len(mesh.functions) == 2


def test_mesh_to_mesh_global():
    # Create a mesh with marker functions
    mesh = Mesh(functions=("HV", "STR", "SUP", "HV"))

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Convert to real mesh
    mesh_real = mesh.to_mesh(mock_data, mock_model, MOCK_RT)

    npt.assert_almost_equal(
        mesh_real.positions,
        np.array(
            [
                [-0.65174504, -0.31463354, -0.47436502, -0.65174504],
                [0.60837317, 0.35748207, 0.4726582, 0.60837317],
                [0.78210787, 0.49102046, 0.57603569, 0.78210787],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )


def test_mesh_to_mesh_local():
    # Create a mesh with marker functions
    mesh = Mesh(functions=("HV", "STR", "SUP", "HV"), is_local=True)

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Convert to real mesh
    mesh_real = mesh.to_mesh(mock_data, mock_model, MOCK_RT)

    npt.assert_almost_equal(
        mesh_real.positions,
        np.array(
            [
                [0.5758053, 0.60645725, 0.5515919, 0.5758053],
                [0.60425486, 0.59659578, 0.60041439, 0.60425486],
                [1.67896849, 1.16874875, 1.37607094, 1.67896849],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )


# ------- Mesh File ------- #
def test_init_mesh_file():
    # Test initialization with minimal parameters
    mesh_file = MeshFile(mesh_file_name="test.obj", mesh_file_directory="mesh_file/dir")
    assert mesh_file.mesh_file_name == "test.obj"
    assert mesh_file.mesh_file_directory == "mesh_file/dir"
    assert mesh_file.mesh_color is None
    assert mesh_file.scaling_function is None
    assert mesh_file.rotation_function is None
    assert mesh_file.translation_function is None

    # Test initialization with all parameters
    scaling_func = lambda data, model: np.array([1, 1, 1])
    rotation_func = lambda data, model: np.eye(3)
    translation_func = lambda data, model: np.array([0, 0, 0])

    mesh_file = MeshFile(
        mesh_file_name="test.obj",
        mesh_file_directory="mesh_file/dir",
        mesh_color=np.array([1.0, 0.0, 0.0]),
        scaling_function=scaling_func,
        rotation_function=rotation_func,
        translation_function=translation_func,
    )

    assert mesh_file.mesh_file_name == "test.obj"
    assert mesh_file.mesh_file_directory == "mesh_file/dir"
    npt.assert_array_equal(mesh_file.mesh_color, np.array([1.0, 0.0, 0.0]))
    assert mesh_file.scaling_function == scaling_func
    assert mesh_file.rotation_function == rotation_func
    assert mesh_file.translation_function == translation_func


def test_mesh_file_to_mesh_file_real():

    # Initialization with all parameters
    scaling_func = lambda data, model: np.array([1.1, 1.1, 1.1])
    rotation_func = lambda data, model: np.array([1, 0, 0])
    translation_func = lambda data, model: np.array([0.1, 0.1, 0.1])

    mesh_file = MeshFile(
        mesh_file_name="test.obj",
        mesh_file_directory="mesh_file/dir",
        mesh_color=np.array([1.0, 1.0, 1.0]),
        scaling_function=scaling_func,
        rotation_function=rotation_func,
        translation_function=translation_func,
    )

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Convert to real mesh
    mesh_real = mesh_file.to_mesh_file(mock_data, mock_model)

    npt.assert_almost_equal(
        mesh_real.mesh_scale.reshape(
            4,
        ),
        np.array([1.1, 1.1, 1.1, 1.0]),
    )
    npt.assert_almost_equal(
        mesh_real.mesh_translation.reshape(
            4,
        ),
        np.array([0.1, 0.1, 0.1, 1.0]),
    )
    npt.assert_almost_equal(
        mesh_real.mesh_rotation.reshape(
            4,
        ),
        np.array([1, 0, 0, 1]),
    )


# ------- Inertia Parameters ------- #
def test_init_inertia_parameters():
    # Test initialization with no parameters
    inertia_params = InertiaParameters()
    assert inertia_params.relative_mass is None
    assert inertia_params.center_of_mass is None
    assert inertia_params.inertia is None

    # Test initialization with all parameters
    mass_func = lambda data, model: 10.0
    com_func = lambda data, model: np.array([0.1, 0.2, 0.3])
    inertia_func = lambda data, model: np.array([1.0, 2.0, 3.0])

    inertia_params = InertiaParameters(mass=mass_func, center_of_mass=com_func, inertia=inertia_func)

    assert inertia_params.relative_mass == mass_func
    assert inertia_params.center_of_mass == com_func
    assert inertia_params.inertia == inertia_func


def test_inertia_parameters_to_inertia_global():
    # Create inertia parameters with functions
    mass_func = lambda data, model: 10.0
    com_func = lambda data, model: np.array([0.1, 0.2, 0.3])
    inertia_func = lambda data, model: np.array([1.0, 2.0, 3.0])

    inertia_params = InertiaParameters(mass=mass_func, center_of_mass=com_func, inertia=inertia_func, is_local=False)

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Convert to real inertia parameters
    inertia_real = inertia_params.to_inertia(mock_data, mock_model, MOCK_RT)

    # Test the properties of the real inertia parameters
    npt.assert_almost_equal(inertia_real.mass, 10.0)
    npt.assert_almost_equal(
        inertia_real.center_of_mass.reshape(
            4,
        ),
        np.array([-0.25467601, -0.22376214, -0.41841443, 1.0]),
    )
    npt.assert_almost_equal(
        inertia_real.inertia.reshape(4, 4), np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])
    )


def test_inertia_parameters_to_inertia_local():
    # Create inertia parameters with functions
    mass_func = lambda data, model: 10.0
    com_func = lambda data, model: np.array([0.1, 0.2, 0.3])
    inertia_func = lambda data, model: np.array([1.0, 2.0, 3.0])

    inertia_params = InertiaParameters(mass=mass_func, center_of_mass=com_func, inertia=inertia_func, is_local=True)

    # Mock data and model
    mock_data = MockC3dData()
    mock_model = BiomechanicalModelReal()

    # Convert to real inertia parameters
    inertia_real = inertia_params.to_inertia(mock_data, mock_model, MOCK_RT)

    # Test the properties of the real inertia parameters
    npt.assert_almost_equal(inertia_real.mass, 10.0)
    npt.assert_almost_equal(
        inertia_real.center_of_mass.reshape(
            4,
        ),
        np.array([0.1, 0.2, 0.3, 1]),
    )
    npt.assert_almost_equal(
        inertia_real.inertia.reshape(4, 4), np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])
    )


def test_radii_of_gyration_to_inertia():
    # Test the static method
    mass = 10.0
    coef = (0.1, 0.2, 0.3)
    start = np.array([0, 0, 0, 1])
    end = np.array([1, 0, 0, 1])

    inertia = InertiaParameters.radii_of_gyration_to_inertia(mass, coef, start, end)

    # The length is 1.0, so the radii are [0.1, 0.2, 0.3]
    # The inertia values should be mass * radius^2
    expected = mass * np.array([0.01, 0.04, 0.09])
    npt.assert_almost_equal(inertia, expected)

    # Test with 2D arrays
    start_2d = np.array([[0, 0, 0, 1], [0, 0, 0, 1]]).T
    end_2d = np.array([[1, 0, 0, 1], [1, 0, 0, 1]]).T

    inertia_2d = InertiaParameters.radii_of_gyration_to_inertia(mass, coef, start_2d, end_2d)
    npt.assert_almost_equal(inertia_2d, expected)


# ------- Contact ------- #
def test_init_contact():
    # Test initialization with minimal parameters
    contact = Contact(name="test_contact", parent_name="segment1")
    assert contact.name == "test_contact"
    assert contact.parent_name == "segment1"
    assert contact.axis is None

    contact = Contact(
        name="test_contact",
        function=lambda m, bio: np.array([1, 2, 3, 1]),
        parent_name="segment1",
        axis=Translations.XYZ,
    )

    assert contact.name == "test_contact"
    assert contact.parent_name == "segment1"
    assert contact.axis == Translations.XYZ

    # Test with string function
    contact = Contact(name="test_contact", function="HV", parent_name="segment1")
    # Call the function with a mock marker dictionary
    mock_data = MockC3dData()
    result = contact.function(mock_data, None)
    npt.assert_almost_equal(
        result.reshape(
            4,
        ),
        np.array([0.5758053, 0.60425486, 1.67896849, 1.0]),
    )


def test_contact_to_contact_global():
    # Create a contact with a position function
    contact = Contact(name="test_contact", function="HV", parent_name="segment1")

    # Mock data
    mock_model = BiomechanicalModelReal()
    mock_data = MockC3dData()

    # Convert to real contact
    contact_real = contact.to_contact(mock_data, mock_model, MOCK_RT)

    # Test the contact properties
    assert contact_real.name == "test_contact"
    assert contact_real.parent_name == "segment1"

    # Test the contact position
    npt.assert_almost_equal(
        np.mean(contact_real.position, axis=1).reshape(
            4,
        ),
        np.array([-0.651745, 0.6083732, 0.7821079, 1.0]),
    )


def test_contact_to_contact_local():
    # Create a contact with a position function
    contact = Contact(name="test_contact", function="HV", parent_name="segment1", is_local=True)

    # Mock data
    mock_model = BiomechanicalModelReal()
    mock_data = MockC3dData()

    # Convert to real contact
    contact_real = contact.to_contact(mock_data, mock_model, MOCK_RT)

    # Test the contact properties
    assert contact_real.name == "test_contact"
    assert contact_real.parent_name == "segment1"

    # Test the contact position
    npt.assert_almost_equal(
        np.mean(contact_real.position, axis=1).reshape(
            4,
        ),
        np.array([0.5758053, 0.6042549, 1.6789685, 1.0]),
    )


# ------- Segment ------- #
def test_init_segment():

    # Test initialization with minimal parameters
    segment = Segment(name="test_segment")
    assert segment.name == "test_segment"
    assert segment.parent_name == "base"
    assert segment.translations == Translations.NONE
    assert segment.rotations == Rotations.NONE
    assert len(segment.dof_names) == 0
    assert segment.q_ranges is None
    assert segment.qdot_ranges is None
    assert len(segment.markers) == 0
    assert len(segment.contacts) == 0
    assert segment.segment_coordinate_system is None
    assert segment.inertia_parameters is None
    assert segment.mesh is None
    assert segment.mesh_file is None

    # Test initialization with custom parameters
    translations = Translations.XYZ
    rotations = Rotations.XYZ
    dof_names = ["dof1", "dof2", "dof3", "dof4", "dof5", "dof6"]
    q_ranges = RangeOfMotion(Ranges.Q, [-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1])
    qdot_ranges = RangeOfMotion(Ranges.Qdot, [-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10])

    segment = Segment(
        name="test_segment",
        parent_name="parent_segment",
        translations=translations,
        rotations=rotations,
        dof_names=dof_names,
        q_ranges=q_ranges,
        qdot_ranges=qdot_ranges,
    )

    assert segment.name == "test_segment"
    assert segment.parent_name == "parent_segment"
    assert segment.translations == Translations.XYZ
    assert segment.rotations == Rotations.XYZ
    assert segment.dof_names == dof_names
    assert segment.q_ranges == q_ranges
    assert segment.qdot_ranges == qdot_ranges


def test_segment_dof_names_auto_generation():

    # Test auto-generation of dof_names
    segment = Segment(name="test_segment", translations=Translations.XY, rotations=Rotations.Z)

    expected_dof_names = ["test_segment_transX", "test_segment_transY", "test_segment_rotZ"]
    assert segment.dof_names == expected_dof_names

    # Test mismatch between dof_names length and actual DoFs
    with pytest.raises(RuntimeError, match="The number of DoF names .* does not match the number of DoFs"):
        Segment(
            name="test_segment",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            dof_names=["dof1"],  # Only one name for 6 DoFs
        )


def test_segment_add_remove_marker():
    # Create a segment
    segment = Segment(name="test_segment")

    # Create a marker with no parent
    marker = Marker(name="test_marker")

    # Add marker to segment
    segment.add_marker(marker)

    # Verify marker was added and parent_name was set
    assert len(segment.markers) == 1
    assert marker.parent_name == "test_segment"

    # Create a marker with matching parent_name
    marker2 = Marker(name="test_marker2", parent_name="test_segment")
    segment.add_marker(marker2)
    assert len(segment.markers) == 2

    # Create a marker with non-matching parent_name
    marker3 = Marker(name="test_marker3", parent_name="other_segment")
    with pytest.raises(ValueError, match="The marker name should be the same as the 'key'"):
        segment.add_marker(marker3)

    # Remove a marker
    segment.remove_marker(marker.name)
    assert len(segment.markers) == 1
    assert segment.markers[0].name == "test_marker2"

    # Remove a marker that does not exist
    with pytest.raises(
        AttributeError, match="The item named test_marker cannot be removed because it it not in the list."
    ):
        segment.remove_marker("test_marker")


def test_segment_add_remove_contact():
    # Create a segment
    segment = Segment(name="test_segment")

    # Create a contact with matching parent_name
    contact = Contact(name="test_contact", parent_name="test_segment")

    # Add contact to segment
    segment.add_contact(contact)

    # Verify contact was added
    assert len(segment.contacts) == 1
    assert contact.parent_name == "test_segment"

    # Create a contact with no parent_name
    contact2 = Contact(name="test_contact2", parent_name=None)
    segment.add_contact(contact2)
    assert segment.name == segment.contacts["test_contact2"].parent_name

    # Create a contact with non-matching parent_name
    contact3 = Contact(name="test_contact3", parent_name="other_segment")
    with pytest.raises(ValueError, match="The contact name should be the same as the 'key'"):
        segment.add_contact(contact3)

    # Remove a contact
    segment.remove_contact(contact.name)
    segment.remove_contact("test_contact2")
    assert len(segment.contacts) == 0

    # Remove a contact that does not exist
    with pytest.raises(
        AttributeError, match="The item named test_contact cannot be removed because it it not in the list."
    ):
        segment.remove_contact("test_contact")
