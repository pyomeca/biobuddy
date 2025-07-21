import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import (Muscle, ViaPoint, MuscleGroup, MuscleType, MuscleStateType)
from biobuddy.utils.named_list import NamedList
from test_utils import MockC3dData


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
    markers = {"marker1": np.array([1, 2, 3])}
    result = via_point.position_function(markers, None)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    # Test with callable position function
    custom_func = lambda m, bio: np.array([4, 5, 6])
    via_point = ViaPoint(name="test_via_point", position_function=custom_func)
    result = via_point.position_function(None, None)
    np.testing.assert_array_equal(result, np.array([4, 5, 6]))


def test_to_via_point():
    # Mock the ViaPointReal class and from_data method
    mock_data = MockC3dData()
    mock_model = None

    # Crete a via point
    via_point = ViaPoint(
        name="test_via_point",
        parent_name="parent1",
        muscle_name="muscle1",
        muscle_group="group1",
    )
    # Not possible to evaluate the via point without a position function
    with pytest.raises(RuntimeError, match="You must provide a position function to evaluate the ViaPoint into a ViaPointReal."):
        via_point_real = via_point.to_via_point(mock_data, mock_model)

    # Set the function
    via_point.position_function = lambda m, bio: m["HV"]

    # Call to_via_point
    via_point_real = via_point.to_via_point(mock_data, mock_model)
    npt.assert_almost_equal(np.mean(via_point.position_function(mock_data.values, mock_model), axis=1).reshape(4, ),
                           np.array([0.5758053 , 0.60425486, 1.67896849, 1.        ]))
    npt.assert_almost_equal(np.mean(via_point_real.position, axis=1).reshape(4, ), np.array([0.5758053 , 0.60425486, 1.67896849, 1.        ]))


    # Set the marker name
    via_point.position_function = "HV"

    # Call to_via_point
    via_point_real = via_point.to_via_point(mock_data, mock_model)
    npt.assert_almost_equal(np.mean(via_point_real.position, axis=1).reshape(4, ), np.array([0.5758053 , 0.60425486, 1.67896849, 1.        ]))


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
        maximal_velocity_function=mock_maximal_velocity
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
        maximal_excitation=0.8
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
        maximal_velocity_function=mock_maximal_velocity
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
    with pytest.raises(ValueError, match="The via points's muscle should be the same as the 'key'. Alternatively, via_point.muscle_name can be left undefined"):
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
        maximal_velocity_function=mock_maximal_velocity
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
    with pytest.raises(ValueError, match="The origin's muscle other_muscle should be the same as the muscle's name test_muscle"):
        muscle.origin_position = origin_bad
    
    # Test setting insertion with non-matching muscle group
    insertion_bad = ViaPoint(name="insertion_bad", parent_name="segment2", muscle_group="other_group")
    with pytest.raises(ValueError, match="The insertion's muscle group other_group should be the same as the muscle's muscle group test_group"):
        muscle.insertion_position = insertion_bad


def test_muscle_to_muscle():
    # Create mock functions for muscle parameters
    mock_optimal_length = lambda params, bio: 0.1
    mock_maximal_force = lambda params, bio: 100.0
    mock_tendon_slack = lambda params, bio: 0.2
    mock_pennation_angle = lambda params, bio: 0.1
    mock_maximal_velocity = lambda params, bio: 10.0
    
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
        maximal_velocity_function=mock_maximal_velocity
    )
    
    # Mock data and model
    mock_data = MockC3dData()
    mock_model = None
    
    # Call to_muscle
    muscle_real = muscle.to_muscle(mock_data, mock_model)
    
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
    npt.assert_almost_equal(np.mean(muscle_real.origin_position.position, axis=1).reshape(4, ), np.array([0.5758053 , 0.60425486, 1.67896849, 1.])) 
    npt.assert_almost_equal(np.mean(muscle_real.insertion_position.position, axis=1).reshape(4, ), np.array([0.5758053 , 0.60425486, 1.67896849, 1.])) 


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
        maximal_velocity_function=mock_maximal_velocity
    )
    
    # Mock data and model
    mock_data = MockC3dData()
    mock_model = None
    
    # Call to_muscle
    muscle_real = muscle.to_muscle(mock_data, mock_model)
    
    # Test the muscle parameters evaluation with known values
    npt.assert_almost_equal(muscle_real.optimal_length, 0.15)
    npt.assert_almost_equal(muscle_real.maximal_force, 150.0)
    npt.assert_almost_equal(muscle_real.tendon_slack_length, 0.25)
    npt.assert_almost_equal(muscle_real.pennation_angle, 0.12)


# ------- Muscle Group ------- #
def test_init_muscle_group():
    # Test initialization
    muscle_group = MuscleGroup(
        name="test_group",
        origin_parent_name="segment1",
        insertion_parent_name="segment2"
    )
    
    assert muscle_group.name == "test_group"
    assert muscle_group.origin_parent_name == "segment1"
    assert muscle_group.insertion_parent_name == "segment2"
    assert isinstance(muscle_group.muscles, NamedList)
    assert len(muscle_group.muscles) == 0
    
    # Test validation - same origin and insertion
    with pytest.raises(ValueError, match="The origin and insertion parent names cannot be the same."):
        MuscleGroup(
            name="test_group",
            origin_parent_name="segment1",
            insertion_parent_name="segment1"
        )


def test_muscle_group_add_remove_muscle():
    muscle_group = MuscleGroup(
        name="test_group",
        origin_parent_name="segment1",
        insertion_parent_name="segment2"
    )
    
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
        maximal_velocity_function=mock_maximal_velocity
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
        maximal_velocity_function=mock_maximal_velocity
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
        maximal_velocity_function=mock_maximal_velocity
    )
    
    # Adding should raise ValueError
    with pytest.raises(ValueError, match="The muscle's muscle_group should be the same as the 'key'. Alternatively, muscle.muscle_group can be left undefined"):
        muscle_group.add_muscle(muscle3)
    
    # Remove one muscle
    muscle_group.remove_muscle("muscle1")
    
    # Verify it was removed
    assert len(muscle_group.muscles) == 1
    assert muscle_group.muscles[0].name == "muscle2"


def test_muscle_group_properties():
    muscle_group = MuscleGroup(
        name="test_group",
        origin_parent_name="segment1",
        insertion_parent_name="segment2"
    )
    
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
        maximal_velocity_function=mock_maximal_velocity
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
        maximal_velocity_function=mock_maximal_velocity
    )
    
    muscle_group.add_muscle(muscle1)
    muscle_group.add_muscle(muscle2)
    
    # Test properties
    assert muscle_group.nb_muscles == 2
    assert muscle_group.muscle_names == ["muscle1", "muscle2"]


def test_muscle_group_with_via_points():
    muscle_group = MuscleGroup(
        name="test_group",
        origin_parent_name="segment1",
        insertion_parent_name="segment2"
    )
    
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
        maximal_velocity_function=mock_maximal_velocity
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
    mock_model = None
    
    # Convert to real muscle
    muscle_real = muscle.to_muscle(mock_data, mock_model)
    
    # Test the via points positions
    npt.assert_almost_equal(np.mean(muscle_real.via_points[0].position, axis=1).reshape(4, ), 
                            np.array([0.5758053, 0.60425486, 1.67896849, 1.]))
    npt.assert_almost_equal(np.mean(muscle_real.via_points[1].position, axis=1).reshape(4, ),
                            np.array([0.5758053, 0.60425486, 1.67896849, 1.]))

