"""
Tests for biobuddy.characteristics.de_leva module
"""
import numpy as np
import numpy.testing as npt
import pytest

from biobuddy.characteristics.de_leva import DeLevaTable, Sex, SegmentName, point_on_vector
from biobuddy.components.generic.rigidbody.inertia_parameters import InertiaParameters


def test_point_on_vector():
    """Test point_on_vector function with various inputs."""
    # Test basic functionality
    start = np.array([0, 0, 0])
    end = np.array([10, 0, 0])
    
    # Test at start (coef=0)
    result = point_on_vector(0.0, start, end)
    npt.assert_almost_equal(result, start)
    
    # Test at end (coef=1) 
    result = point_on_vector(1.0, start, end)
    npt.assert_almost_equal(result, end)
    
    # Test at midpoint (coef=0.5)
    result = point_on_vector(0.5, start, end)
    expected = np.array([5, 0, 0])
    npt.assert_almost_equal(result, expected)
    
    # Test with 3D vectors
    start = np.array([1, 2, 3])
    end = np.array([4, 6, 9])
    result = point_on_vector(0.5, start, end)
    expected = np.array([2.5, 4, 6])
    npt.assert_almost_equal(result, expected)
    
    # Test with coefficient > 1 (extrapolation)
    result = point_on_vector(2.0, start, end)
    expected = np.array([7, 10, 15])
    npt.assert_almost_equal(result, expected)
    
    # Test with negative coefficient
    result = point_on_vector(-0.5, start, end)
    expected = np.array([-0.5, 0, 0])
    npt.assert_almost_equal(result, expected)


def test_sex_enum():
    """Test Sex enum values."""
    assert Sex.MALE.value == "male"
    assert Sex.FEMALE.value == "female"
    
    # Test that we can access both values
    assert Sex.MALE is not None
    assert Sex.FEMALE is not None
    
    # Test they are different
    assert Sex.MALE != Sex.FEMALE


def test_segment_name_enum():
    """Test SegmentName enum values."""
    expected_segments = [
        "HEAD", "TRUNK", "UPPER_ARM", "LOWER_ARM", 
        "HAND", "THIGH", "SHANK", "FOOT"
    ]
    
    for segment_name in expected_segments:
        segment = getattr(SegmentName, segment_name)
        assert segment.value == segment_name
    
    # Test that all segments are accessible
    assert len(list(SegmentName)) == len(expected_segments)


def test_de_leva_table_constructor():
    """Test DeLevaTable constructor."""
    total_mass = 70.0
    
    # Test male constructor
    male_table = DeLevaTable(total_mass, Sex.MALE)
    assert male_table.sex == Sex.MALE
    assert hasattr(male_table, 'inertial_table')
    
    # Test female constructor  
    female_table = DeLevaTable(total_mass, Sex.FEMALE)
    assert female_table.sex == Sex.FEMALE
    assert hasattr(female_table, 'inertial_table')
    
    # Test that both tables have the same structure
    male_segments = set(male_table.inertial_table[Sex.MALE].keys())
    female_segments = set(female_table.inertial_table[Sex.FEMALE].keys())
    assert male_segments == female_segments
    
    # Test that all expected segments are present
    expected_segments = set(SegmentName)
    assert male_segments == expected_segments


def test_de_leva_table_getitem():
    """Test DeLevaTable.__getitem__ method."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)
    female_table = DeLevaTable(total_mass, Sex.FEMALE)
    
    # Test that we can access all segments
    for segment in SegmentName:
        male_params = male_table[segment]
        female_params = female_table[segment]
        
        # Both should return InertiaParameters objects
        assert isinstance(male_params, InertiaParameters)
        assert isinstance(female_params, InertiaParameters)
        
        # Both should have the required attributes
        assert hasattr(male_params, 'relative_mass')
        assert hasattr(male_params, 'center_of_mass') 
        assert hasattr(male_params, 'inertia')


def test_de_leva_table_mass_calculations():
    """Test that mass calculations are correct."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)
    female_table = DeLevaTable(total_mass, Sex.FEMALE)
    
    # Test some known mass fractions for males (from De Leva 1996)
    # HEAD: 0.0694, TRUNK: 0.4346, etc.
    expected_male_masses = {
        SegmentName.HEAD: 0.0694 * total_mass,
        SegmentName.TRUNK: 0.4346 * total_mass,
        SegmentName.UPPER_ARM: 2 * 0.0271 * total_mass,  # bilateral
        SegmentName.LOWER_ARM: 2 * 0.0162 * total_mass,  # bilateral
        SegmentName.HAND: 2 * 0.0061 * total_mass,       # bilateral
        SegmentName.THIGH: 2 * 0.1416 * total_mass,      # bilateral
        SegmentName.SHANK: 2 * 0.0433 * total_mass,      # bilateral
        SegmentName.FOOT: 2 * 0.0137 * total_mass,       # bilateral
    }
    
    # Test some known mass fractions for females
    expected_female_masses = {
        SegmentName.HEAD: 0.0669 * total_mass,
        SegmentName.TRUNK: 0.4257 * total_mass,
        SegmentName.UPPER_ARM: 2 * 0.0255 * total_mass,  # bilateral
        SegmentName.LOWER_ARM: 2 * 0.0138 * total_mass,  # bilateral
        SegmentName.HAND: 2 * 0.0056 * total_mass,       # bilateral
        SegmentName.THIGH: 2 * 0.1478 * total_mass,      # bilateral
        SegmentName.SHANK: 2 * 0.0481 * total_mass,      # bilateral
        SegmentName.FOOT: 2 * 0.0129 * total_mass,       # bilateral
    }
    
    # Test mass calculations (we need mock markers for this)
    # TODO: Create proper test with mock marker data to validate mass calculations
    # For now, just test that the mass functions exist and are callable
    for segment in SegmentName:
        male_params = male_table[segment]
        female_params = female_table[segment]
        
        assert callable(male_params.relative_mass)
        assert callable(female_params.relative_mass)


def test_de_leva_table_center_of_mass():
    """Test center of mass calculations."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)
    
    # Test that center of mass functions exist and are callable
    for segment in SegmentName:
        params = male_table[segment]
        assert callable(params.center_of_mass)
    
    # TODO: Create proper test with mock marker data to validate CoM calculations
    # The CoM calculation depends on marker positions which require actual marker data


def test_de_leva_table_inertia():
    """Test inertia calculations."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)
    
    # Test that inertia functions exist and are callable
    for segment in SegmentName:
        params = male_table[segment]
        assert callable(params.inertia)
    
    # TODO: Create proper test with mock marker data to validate inertia calculations
    # The inertia calculation depends on marker positions which require actual marker data


def test_radii_of_gyration_to_inertia():
    """Test the radii_of_gyration_to_inertia static method."""
    mass = 5.0
    coef = (0.3, 0.4, 0.2)  # radii of gyration coefficients
    start = np.array([0, 0, 0])
    end = np.array([1, 0, 0])  # 1 unit length
    
    # Expected: length = 1, so r_squared = coef^2, inertia = mass * r_squared
    expected_inertia = mass * np.array([0.09, 0.16, 0.04])  # mass * coef^2
    
    result = InertiaParameters.radii_of_gyration_to_inertia(mass, coef, start, end)
    npt.assert_almost_equal(result, expected_inertia)
    
    # Test with different length
    end = np.array([2, 0, 0])  # 2 units length
    expected_inertia = mass * np.array([0.36, 0.64, 0.16])  # mass * (coef * 2)^2
    
    result = InertiaParameters.radii_of_gyration_to_inertia(mass, coef, start, end)
    npt.assert_almost_equal(result, expected_inertia)
    
    # Test with 3D vectors
    start = np.array([1, 1, 1])
    end = np.array([4, 5, 1])  # length = sqrt(9 + 16) = 5
    length = 5.0
    expected_inertia = mass * (np.array(coef) * length) ** 2
    
    result = InertiaParameters.radii_of_gyration_to_inertia(mass, coef, start, end)
    npt.assert_almost_equal(result, expected_inertia)


def test_de_leva_table_comprehensive():
    """Comprehensive test to ensure data consistency."""
    total_mass = 70.0
    
    # Test both sexes
    for sex in [Sex.MALE, Sex.FEMALE]:
        table = DeLevaTable(total_mass, sex)
        
        # Test that all segments are accessible
        for segment in SegmentName:
            params = table[segment]
            
            # Test that all required functions are present
            assert params.relative_mass is not None
            assert params.center_of_mass is not None  
            assert params.inertia is not None
            
            # Test that functions are callable
            assert callable(params.relative_mass)
            assert callable(params.center_of_mass)
            assert callable(params.inertia)


def test_de_leva_table_with_mock_data():
    """Test De Leva table with mock marker data."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)
    
    # Create mock markers dictionary
    mock_markers = {
        'TOP_HEAD': np.array([0, 0, 10]),
        'SHOULDER': np.array([0, 0, 8]),
        'PELVIS': np.array([0, 0, 5]),
        'ELBOW': np.array([1, 0, 7]),
        'WRIST': np.array([2, 0, 6]),
        'FINGER': np.array([3, 0, 6]),
        'KNEE': np.array([0, 0, 3]),
        'ANKLE': np.array([0, 0, 1]),
        'TOE': np.array([0, 1, 0]),
    }
    
    # Test HEAD segment calculations
    head_params = male_table[SegmentName.HEAD]
    
    # Test mass calculation - should be 0.0694 * total_mass
    expected_mass = 0.0694 * total_mass
    actual_mass = head_params.relative_mass(mock_markers, None)
    npt.assert_almost_equal(actual_mass, expected_mass)
    
    # Test center of mass calculation - should be along vector from TOP_HEAD to SHOULDER
    expected_com = point_on_vector(0.5002, mock_markers['TOP_HEAD'], mock_markers['SHOULDER'])
    actual_com = head_params.center_of_mass(mock_markers, None)
    npt.assert_almost_equal(actual_com, expected_com)
    
    # Test inertia calculation
    expected_inertia = InertiaParameters.radii_of_gyration_to_inertia(
        mass=expected_mass,
        coef=(0.303, 0.315, 0.261),
        start=mock_markers['TOP_HEAD'],
        end=mock_markers['SHOULDER']
    )
    actual_inertia = head_params.inertia(mock_markers, None)
    npt.assert_almost_equal(actual_inertia, expected_inertia)


def test_bilateral_segments_mass_factor():
    """Test that bilateral segments have the correct mass factor (x2)."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)
    
    # Bilateral segments should have 2x the unilateral mass
    bilateral_segments = [
        SegmentName.UPPER_ARM, SegmentName.LOWER_ARM, SegmentName.HAND,
        SegmentName.THIGH, SegmentName.SHANK, SegmentName.FOOT
    ]
    
    # Create minimal mock markers for testing
    mock_markers = {
        'SHOULDER': np.array([0, 0, 8]),
        'ELBOW': np.array([1, 0, 7]),
        'WRIST': np.array([2, 0, 6]),
        'FINGER': np.array([3, 0, 6]),
        'PELVIS': np.array([0, 0, 5]),
        'KNEE': np.array([0, 0, 3]),
        'ANKLE': np.array([0, 0, 1]),
        'TOE': np.array([0, 1, 0]),
    }
    
    # Test upper arm - should be 2 * 0.0271 * total_mass for males
    upper_arm_params = male_table[SegmentName.UPPER_ARM]
    expected_mass = 2 * 0.0271 * total_mass
    actual_mass = upper_arm_params.relative_mass(mock_markers, None)
    npt.assert_almost_equal(actual_mass, expected_mass)
    
    # Test thigh - should be 2 * 0.1416 * total_mass for males  
    thigh_params = male_table[SegmentName.THIGH]
    expected_mass = 2 * 0.1416 * total_mass
    actual_mass = thigh_params.relative_mass(mock_markers, None)
    npt.assert_almost_equal(actual_mass, expected_mass)


def test_sex_differences():
    """Test that male and female tables have different values."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)
    female_table = DeLevaTable(total_mass, Sex.FEMALE)
    
    # Create mock markers for testing
    mock_markers = {
        'TOP_HEAD': np.array([0, 0, 10]),
        'SHOULDER': np.array([0, 0, 8]),
        'PELVIS': np.array([0, 0, 5]),
        'ELBOW': np.array([1, 0, 7]),
        'WRIST': np.array([2, 0, 6]),
        'FINGER': np.array([3, 0, 6]),
        'KNEE': np.array([0, 0, 3]),
        'ANKLE': np.array([0, 0, 1]),
        'TOE': np.array([0, 1, 0]),
    }
    
    # Test that head mass is different between males and females
    male_head_mass = male_table[SegmentName.HEAD].relative_mass(mock_markers, None)
    female_head_mass = female_table[SegmentName.HEAD].relative_mass(mock_markers, None)
    
    # Should be 0.0694 vs 0.0669 * total_mass
    npt.assert_almost_equal(male_head_mass, 0.0694 * total_mass)
    npt.assert_almost_equal(female_head_mass, 0.0669 * total_mass)
    assert male_head_mass != female_head_mass
    
    # Test that trunk mass is different
    male_trunk_mass = male_table[SegmentName.TRUNK].relative_mass(mock_markers, None) 
    female_trunk_mass = female_table[SegmentName.TRUNK].relative_mass(mock_markers, None)
    
    # Should be 0.4346 vs 0.4257 * total_mass
    npt.assert_almost_equal(male_trunk_mass, 0.4346 * total_mass)
    npt.assert_almost_equal(female_trunk_mass, 0.4257 * total_mass)
    assert male_trunk_mass != female_trunk_mass


def test_de_leva_table_different_masses():
    """Test De Leva table with different total masses."""
    masses = [50.0, 70.0, 100.0, 120.0]
    
    for total_mass in masses:
        male_table = DeLevaTable(total_mass, Sex.MALE)
        female_table = DeLevaTable(total_mass, Sex.FEMALE)
        
        # Test that calculations scale correctly
        mock_markers = {
            'TOP_HEAD': np.array([0, 0, 10]),
            'SHOULDER': np.array([0, 0, 8]),
            'PELVIS': np.array([0, 0, 5]),
            'ELBOW': np.array([1, 0, 7]),
            'WRIST': np.array([2, 0, 6]),
            'FINGER': np.array([3, 0, 6]),
            'KNEE': np.array([0, 0, 3]),
            'ANKLE': np.array([0, 0, 1]),
            'TOE': np.array([0, 1, 0]),
        }
        
        # Test head mass scales correctly
        male_head_mass = male_table[SegmentName.HEAD].relative_mass(mock_markers, None)
        female_head_mass = female_table[SegmentName.HEAD].relative_mass(mock_markers, None)
        
        npt.assert_almost_equal(male_head_mass, 0.0694 * total_mass)
        npt.assert_almost_equal(female_head_mass, 0.0669 * total_mass)


def test_de_leva_table_edge_cases():
    """Test edge cases for De Leva table."""
    # Test with very small mass
    small_mass = 0.1
    table = DeLevaTable(small_mass, Sex.MALE)
    
    mock_markers = {
        'TOP_HEAD': np.array([0, 0, 10]),
        'SHOULDER': np.array([0, 0, 8]),
        'PELVIS': np.array([0, 0, 5]),
        'ELBOW': np.array([1, 0, 7]),
        'WRIST': np.array([2, 0, 6]),
        'FINGER': np.array([3, 0, 6]),
        'KNEE': np.array([0, 0, 3]),
        'ANKLE': np.array([0, 0, 1]),
        'TOE': np.array([0, 1, 0]),
    }
    
    # Should still work with very small masses
    head_mass = table[SegmentName.HEAD].relative_mass(mock_markers, None)
    expected = 0.0694 * small_mass
    npt.assert_almost_equal(head_mass, expected)
    
    # Test with very large mass
    large_mass = 200.0
    table = DeLevaTable(large_mass, Sex.FEMALE)
    head_mass = table[SegmentName.HEAD].relative_mass(mock_markers, None)
    expected = 0.0669 * large_mass
    npt.assert_almost_equal(head_mass, expected)


if __name__ == "__main__":
    pytest.main([__file__])