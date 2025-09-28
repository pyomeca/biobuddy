import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from biobuddy.components.real.biomechanical_model_real import BiomechanicalModelReal
from biobuddy.components.real.rigidbody.segment_real import SegmentReal
from biobuddy.components.real.rigidbody.marker_real import MarkerReal
from biobuddy.components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from biobuddy.components.real.muscle.muscle_group_real import MuscleGroupReal
from biobuddy.components.real.muscle.muscle_real import MuscleReal
from biobuddy.components.real.muscle.via_point_real import ViaPointReal
from biobuddy.utils.enums import Translations
from biobuddy.utils.linear_algebra import RotoTransMatrix, Point
from biobuddy.model_modifiers.flattening_tool import FlatteningTool, AXIS_TO_INDEX


def create_test_model():
    """Create a simple model for testing the flattening tool"""
    model = BiomechanicalModelReal()
    
    # Create a segment with rotated coordinate system
    segment1 = SegmentReal(name="segment1")
    segment1.segment_coordinate_system = SegmentCoordinateSystemReal(
        scs=RotoTransMatrix().from_rt_matrix(np.eye(4))
    )
    segment1.segment_coordinate_system.scs.translation = np.array([1.0, 2.0, 3.0])
    
    # Add a marker to the segment
    marker1 = MarkerReal(name="marker1", parent_name="segment1")
    marker1.position = np.array([1.0, 2.0, 3.0])
    segment1.add_marker(marker1)
    
    # Add inertia parameters
    segment1.inertia_parameters = type('InertiaParameters', (), {
        'center_of_mass': np.array([1.0, 2.0, 3.0])
    })
    
    # Add a contact point
    contact1 = type('Contact', (), {'position': np.array([1.0, 2.0, 3.0])})
    segment1.contacts = [contact1]
    
    # Add an IMU
    imu1 = type('IMU', (), {'scs': RotoTransMatrix()})
    imu1.scs.translation = np.array([1.0, 2.0, 3.0])
    segment1.imus = [imu1]
    
    model.add_segment(segment1)
    
    # Create a muscle group with muscles and via points
    muscle_group = MuscleGroupReal(
        name="muscle_group1",
        origin_parent_name="segment1",
        insertion_parent_name="segment1"
    )
    
    muscle = MuscleReal(name="muscle1")
    muscle.origin_position = type('Position', (), {'position': np.array([1.0, 2.0, 3.0])})
    muscle.insertion_position = type('Position', (), {'position': np.array([1.0, 2.0, 3.0])})
    
    via_point = ViaPointReal(parent_name="segment1")
    via_point.position = np.array([1.0, 2.0, 3.0])
    muscle.via_points = [via_point]
    
    muscle_group.add_muscle(muscle)
    model.muscle_groups.append(muscle_group)
    
    return model


def test_flattening_tool_initialization():
    """Test the initialization of the FlatteningTool"""
    model = create_test_model()
    flattening_tool = FlatteningTool(model, Translations.Y)
    
    assert flattening_tool.original_model is model
    assert flattening_tool.axis == Translations.Y
    assert flattening_tool.flattened_model is not model  # Should be a deep copy


def test_check_model():
    """Test the _check_model method"""
    model = create_test_model()
    flattening_tool = FlatteningTool(model, Translations.Y)
    
    # This should not raise an error
    flattening_tool._check_model()
    
    # Now let's make a rotated coordinate system
    model.segments[0].segment_coordinate_system.scs.rotation_matrix = np.array([
        [0.866, -0.5, 0],
        [0.5, 0.866, 0],
        [0, 0, 1]
    ])  # 30 degree rotation around Z
    
    flattening_tool = FlatteningTool(model, Translations.Y)
    
    # This should raise an error
    with pytest.raises(ValueError):
        flattening_tool._check_model()


def test_modify_jcs():
    """Test the _modify_jcs method"""
    model = create_test_model()
    flattening_tool = FlatteningTool(model, Translations.Y)
    
    # Before modification
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].segment_coordinate_system.scs.translation,
        np.array([1.0, 2.0, 3.0])
    )
    
    flattening_tool._modify_jcs()
    
    # After modification, Y should be 0
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].segment_coordinate_system.scs.translation,
        np.array([1.0, 0.0, 3.0])
    )
    
    # Original model should be unchanged
    assert_almost_equal(
        flattening_tool.original_model.segments[0].segment_coordinate_system.scs.translation,
        np.array([1.0, 2.0, 3.0])
    )


def test_modify_com():
    """Test the _modify_com method"""
    model = create_test_model()
    flattening_tool = FlatteningTool(model, Translations.Y)
    
    # Before modification
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].inertia_parameters.center_of_mass,
        np.array([1.0, 2.0, 3.0])
    )
    
    flattening_tool._modify_com()
    
    # After modification, Y should be 0
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].inertia_parameters.center_of_mass,
        np.array([1.0, 0.0, 3.0])
    )
    
    # Original model should be unchanged
    assert_almost_equal(
        flattening_tool.original_model.segments[0].inertia_parameters.center_of_mass,
        np.array([1.0, 2.0, 3.0])
    )


def test_modify_markers():
    """Test the _modify_markers method"""
    model = create_test_model()
    flattening_tool = FlatteningTool(model, Translations.Y)
    
    # Before modification
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].markers[0].position,
        np.array([1.0, 2.0, 3.0])
    )
    
    flattening_tool._modify_markers()
    
    # After modification, Y should be 0
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].markers[0].position,
        np.array([1.0, 0.0, 3.0])
    )
    
    # Original model should be unchanged
    assert_almost_equal(
        flattening_tool.original_model.segments[0].markers[0].position,
        np.array([1.0, 2.0, 3.0])
    )


def test_modify_contacts():
    """Test the _modify_contacts method"""
    model = create_test_model()
    flattening_tool = FlatteningTool(model, Translations.Y)
    
    # Before modification
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].contacts[0].position,
        np.array([1.0, 2.0, 3.0])
    )
    
    flattening_tool._modify_contacts()
    
    # After modification, Y should be 0
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].contacts[0].position,
        np.array([1.0, 0.0, 3.0])
    )
    
    # Original model should be unchanged
    assert_almost_equal(
        flattening_tool.original_model.segments[0].contacts[0].position,
        np.array([1.0, 2.0, 3.0])
    )


def test_modify_imus():
    """Test the _modify_imus method"""
    model = create_test_model()
    flattening_tool = FlatteningTool(model, Translations.Y)
    
    # Before modification
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].imus[0].scs.translation,
        np.array([1.0, 2.0, 3.0])
    )
    
    flattening_tool._modify_imus()
    
    # After modification, Y should be 0
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].imus[0].scs.translation,
        np.array([1.0, 0.0, 3.0])
    )
    
    # Original model should be unchanged
    assert_almost_equal(
        flattening_tool.original_model.segments[0].imus[0].scs.translation,
        np.array([1.0, 2.0, 3.0])
    )


def test_modify_muscles():
    """Test the _modify_muscles method"""
    model = create_test_model()
    flattening_tool = FlatteningTool(model, Translations.Y)
    
    # Before modification
    assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].origin_position.position,
        np.array([1.0, 2.0, 3.0])
    )
    assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].via_points[0].position,
        np.array([1.0, 2.0, 3.0])
    )
    assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].insertion_position.position,
        np.array([1.0, 2.0, 3.0])
    )
    
    flattening_tool._modify_muscles()
    
    # After modification, Y should be 0
    assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].origin_position.position,
        np.array([1.0, 0.0, 3.0])
    )
    assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].via_points[0].position,
        np.array([1.0, 0.0, 3.0])
    )
    assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].insertion_position.position,
        np.array([1.0, 0.0, 3.0])
    )
    
    # Original model should be unchanged
    assert_almost_equal(
        flattening_tool.original_model.muscle_groups[0].muscles[0].origin_position.position,
        np.array([1.0, 2.0, 3.0])
    )


def test_flatten():
    """Test the flatten method"""
    model = create_test_model()
    flattening_tool = FlatteningTool(model, Translations.Y)
    
    # Before flattening
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].segment_coordinate_system.scs.translation,
        np.array([1.0, 2.0, 3.0])
    )
    assert_almost_equal(
        flattening_tool.flattened_model.segments[0].markers[0].position,
        np.array([1.0, 2.0, 3.0])
    )
    
    flattened_model = flattening_tool.flatten()
    
    # After flattening, Y should be 0 everywhere
    assert_almost_equal(
        flattened_model.segments[0].segment_coordinate_system.scs.translation,
        np.array([1.0, 0.0, 3.0])
    )
    assert_almost_equal(
        flattened_model.segments[0].markers[0].position,
        np.array([1.0, 0.0, 3.0])
    )
    assert_almost_equal(
        flattened_model.segments[0].inertia_parameters.center_of_mass,
        np.array([1.0, 0.0, 3.0])
    )
    assert_almost_equal(
        flattened_model.segments[0].contacts[0].position,
        np.array([1.0, 0.0, 3.0])
    )
    assert_almost_equal(
        flattened_model.segments[0].imus[0].scs.translation,
        np.array([1.0, 0.0, 3.0])
    )
    assert_almost_equal(
        flattened_model.muscle_groups[0].muscles[0].origin_position.position,
        np.array([1.0, 0.0, 3.0])
    )
    assert_almost_equal(
        flattened_model.muscle_groups[0].muscles[0].via_points[0].position,
        np.array([1.0, 0.0, 3.0])
    )
    assert_almost_equal(
        flattened_model.muscle_groups[0].muscles[0].insertion_position.position,
        np.array([1.0, 0.0, 3.0])
    )
    
    # Original model should be unchanged
    assert_almost_equal(
        flattening_tool.original_model.segments[0].segment_coordinate_system.scs.translation,
        np.array([1.0, 2.0, 3.0])
    )


def test_axis_to_index():
    """Test the AXIS_TO_INDEX mapping"""
    assert AXIS_TO_INDEX[Translations.X] == 0
    assert AXIS_TO_INDEX[Translations.Y] == 1
    assert AXIS_TO_INDEX[Translations.Z] == 2


def test_different_axes():
    """Test flattening with different axes"""
    model = create_test_model()
    
    # Test X axis
    flattening_tool_x = FlatteningTool(model, Translations.X)
    flattened_model_x = flattening_tool_x.flatten()
    assert_almost_equal(
        flattened_model_x.segments[0].segment_coordinate_system.scs.translation,
        np.array([0.0, 2.0, 3.0])
    )
    
    # Test Z axis
    flattening_tool_z = FlatteningTool(model, Translations.Z)
    flattened_model_z = flattening_tool_z.flatten()
    assert_almost_equal(
        flattened_model_z.segments[0].segment_coordinate_system.scs.translation,
        np.array([1.0, 2.0, 0.0])
    )


# TODO: Test the modify_muscle_parameters function which is called in flatten()
# This would require more knowledge about how the function works and what it modifies

# TODO: Test with a more complex model with multiple segments and hierarchical relationships

# TODO: Test with segments that have no markers, contacts, imus, or muscles to ensure
# the code handles these cases gracefully
