"""
Tests for biobuddy.characteristics.de_leva module
"""

import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import DeLevaTable, Sex, SegmentName, InertiaParameters, BiomechanicalModel
from biobuddy.characteristics.de_leva import point_on_vector


MOCK_DATA = {
    "TOP_HEAD": np.array([0, 0, 10]),
    "SHOULDER": np.array([0, 0, 8]),
    "PELVIS": np.array([0, 0, 5]),
    "ELBOW": np.array([1, 0, 7]),
    "WRIST": np.array([2, 0, 6]),
    "FINGER": np.array([3, 0, 6]),
    "KNEE": np.array([0, 0, 3]),
    "ANKLE": np.array([0, 0, 1]),
    "TOE": np.array([0, 1, 0]),
}


def get_biomechanical_model():
    """Create a simple model to test the De Leva table with"""

    model = BiomechanicalModel()
    model.add_segment(
        Segment(
            name="TRUNK",
            translations=Translations.YZ,
            rotations=Rotations.X,
            inertia_parameters=de_leva[SegmentName.TRUNK],
        )
    )
    model.segments["TRUNK"].add_marker(Marker("PELVIS"))

    model.add_segment(
        Segment(
            name="HEAD",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystem(
                "BOTTOM_HEAD",
                first_axis=Axis(name=Axis.Name.Z, start="BOTTOM_HEAD", end="HEAD_Z"),
                second_axis=Axis(name=Axis.Name.X, start="BOTTOM_HEAD", end="HEAD_XZ"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(("BOTTOM_HEAD", "TOP_HEAD", "HEAD_Z", "HEAD_XZ", "BOTTOM_HEAD")),
            inertia_parameters=de_leva[SegmentName.HEAD],
        )
    )
    model.segments["HEAD"].add_marker(Marker("BOTTOM_HEAD"))
    model.segments["HEAD"].add_marker(Marker("TOP_HEAD"))
    model.segments["HEAD"].add_marker(Marker("HEAD_Z"))
    model.segments["HEAD"].add_marker(Marker("HEAD_XZ"))

    model.add_segment(
        Segment(
            name="UPPER_ARM",
            parent_name="TRUNK",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="SHOULDER",
                first_axis=Axis(name=Axis.Name.X, start="SHOULDER", end="SHOULDER_X"),
                second_axis=Axis(name=Axis.Name.Y, start="SHOULDER", end="SHOULDER_XY"),
                axis_to_keep=Axis.Name.X,
            ),
            inertia_parameters=de_leva[SegmentName.UPPER_ARM],
        )
    )
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER"))
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER_X"))
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER_XY"))

    model.add_segment(
        Segment(
            name="LOWER_ARM",
            parent_name="UPPER_ARM",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="ELBOW",
                first_axis=Axis(name=Axis.Name.Y, start="ELBOW", end="ELBOW_Y"),
                second_axis=Axis(name=Axis.Name.X, start="ELBOW", end="ELBOW_XY"),
                axis_to_keep=Axis.Name.Y,
            ),
            inertia_parameters=de_leva[SegmentName.LOWER_ARM],
        )
    )
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW"))
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW_Y"))
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW_XY"))

    model.add_segment(
        Segment(
            name="HAND",
            parent_name="LOWER_ARM",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="WRIST",
                first_axis=Axis(name=Axis.Name.Y, start="WRIST", end="HAND_Y"),
                second_axis=Axis(name=Axis.Name.Z, start="WRIST", end="HAND_YZ"),
                axis_to_keep=Axis.Name.Y,
            ),
            inertia_parameters=de_leva[SegmentName.HAND],
        )
    )
    model.segments["HAND"].add_marker(Marker("WRIST"))
    model.segments["HAND"].add_marker(Marker("FINGER"))
    model.segments["HAND"].add_marker(Marker("HAND_Y"))
    model.segments["HAND"].add_marker(Marker("HAND_YZ"))

    model.add_segment(
        Segment(
            name="THIGH",
            parent_name="TRUNK",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="THIGH_ORIGIN",
                first_axis=Axis(name=Axis.Name.X, start="THIGH_ORIGIN", end="THIGH_X"),
                second_axis=Axis(name=Axis.Name.Y, start="THIGH_ORIGIN", end="THIGH_Y"),
                axis_to_keep=Axis.Name.X,
            ),
            inertia_parameters=de_leva[SegmentName.THIGH],
        )
    )
    model.segments["THIGH"].add_marker(Marker("THIGH_ORIGIN"))
    model.segments["THIGH"].add_marker(Marker("THIGH_X"))
    model.segments["THIGH"].add_marker(Marker("THIGH_Y"))

    model.add_segment(
        Segment(
            name="SHANK",
            parent_name="THIGH",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="KNEE",
                first_axis=Axis(name=Axis.Name.Z, start="KNEE", end="KNEE_Z"),
                second_axis=Axis(name=Axis.Name.X, start="KNEE", end="KNEE_XZ"),
                axis_to_keep=Axis.Name.Z,
            ),
            inertia_parameters=de_leva[SegmentName.SHANK],
        )
    )
    model.segments["SHANK"].add_marker(Marker("KNEE"))
    model.segments["SHANK"].add_marker(Marker("KNEE_Z"))
    model.segments["SHANK"].add_marker(Marker("KNEE_XZ"))

    model.add_segment(
        Segment(
            name="FOOT",
            parent_name="SHANK",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="ANKLE",
                first_axis=Axis(name=Axis.Name.Z, start="ANKLE", end="ANKLE_Z"),
                second_axis=Axis(name=Axis.Name.Y, start="ANKLE", end="ANKLE_YZ"),
                axis_to_keep=Axis.Name.Z,
            ),
            inertia_parameters=de_leva[SegmentName.FOOT],
        )
    )
    model.segments["FOOT"].add_marker(Marker("ANKLE"))
    model.segments["FOOT"].add_marker(Marker("TOE"))
    model.segments["FOOT"].add_marker(Marker("ANKLE_Z"))
    model.segments["FOOT"].add_marker(Marker("ANKLE_YZ"))
    return model


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
    expected_segments = ["HEAD", "TRUNK", "UPPER_ARM", "LOWER_ARM", "HAND", "THIGH", "SHANK", "FOOT"]

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
    assert hasattr(male_table, "inertial_table")

    # Test female constructor
    female_table = DeLevaTable(total_mass, Sex.FEMALE)
    assert female_table.sex == Sex.FEMALE
    assert hasattr(female_table, "inertial_table")

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
        assert hasattr(male_params, "relative_mass")
        assert hasattr(male_params, "center_of_mass")
        assert hasattr(male_params, "inertia")


def test_de_leva_table_mass_calculations():
    """Test that mass calculations are correct."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)
    female_table = DeLevaTable(total_mass, Sex.FEMALE)

    # Test the MASS values
    expected_male_masses = {
        SegmentName.HEAD: 0.0694 * total_mass,
        SegmentName.TRUNK: 0.4346 * total_mass,
        SegmentName.UPPER_ARM: 2 * 0.0271 * total_mass,  # bilateral
        SegmentName.LOWER_ARM: 2 * 0.0162 * total_mass,  # bilateral
        SegmentName.HAND: 2 * 0.0061 * total_mass,  # bilateral
        SegmentName.THIGH: 2 * 0.1416 * total_mass,  # bilateral
        SegmentName.SHANK: 2 * 0.0433 * total_mass,  # bilateral
        SegmentName.FOOT: 2 * 0.0137 * total_mass,  # bilateral
    }
    expected_female_masses = {
        SegmentName.HEAD: 0.0669 * total_mass,
        SegmentName.TRUNK: 0.4257 * total_mass,
        SegmentName.UPPER_ARM: 2 * 0.0255 * total_mass,  # bilateral
        SegmentName.LOWER_ARM: 2 * 0.0138 * total_mass,  # bilateral
        SegmentName.HAND: 2 * 0.0056 * total_mass,  # bilateral
        SegmentName.THIGH: 2 * 0.1478 * total_mass,  # bilateral
        SegmentName.SHANK: 2 * 0.0481 * total_mass,  # bilateral
        SegmentName.FOOT: 2 * 0.0129 * total_mass,  # bilateral
    }
    for segment in expected_male_masses.keys():
        # Male
        npt.assert_almost_equal(
            male_table[segment].relative_mass(MOCK_DATA, BiomechanicalModel()), expected_male_masses[segment]
        )
        # Female
        npt.assert_almost_equal(
            female_table[segment].relative_mass(MOCK_DATA, BiomechanicalModel()), expected_female_masses[segment]
        )

    # Test the center of mass values : start + coef (end - start)
    expected_male_com = {
        SegmentName.HEAD: MOCK_DATA["TOP_HEAD"] + 0.5002 * (MOCK_DATA["SHOULDER"] - MOCK_DATA["TOP_HEAD"]),
        SegmentName.TRUNK: MOCK_DATA["SHOULDER"] + 0.5138 * (MOCK_DATA["PELVIS"] - MOCK_DATA["SHOULDER"]),
        SegmentName.UPPER_ARM: MOCK_DATA["SHOULDER"] + 0.5772 * (MOCK_DATA["ELBOW"] - MOCK_DATA["SHOULDER"]),
        SegmentName.LOWER_ARM: MOCK_DATA["ELBOW"] + 0.4574 * (MOCK_DATA["WRIST"] - MOCK_DATA["ELBOW"]),
        SegmentName.HAND: MOCK_DATA["WRIST"] + 0.7900 * (MOCK_DATA["FINGER"] - MOCK_DATA["WRIST"]),
        SegmentName.THIGH: MOCK_DATA["PELVIS"] + 0.4095 * (MOCK_DATA["KNEE"] - MOCK_DATA["PELVIS"]),
        SegmentName.SHANK: MOCK_DATA["KNEE"] + 0.4459 * (MOCK_DATA["ANKLE"] - MOCK_DATA["KNEE"]),
        SegmentName.FOOT: MOCK_DATA["ANKLE"] + 0.4415 * (MOCK_DATA["TOE"] - MOCK_DATA["ANKLE"]),
    }
    expected_female_com = {
        SegmentName.HEAD: MOCK_DATA["TOP_HEAD"] + 0.4841 * (MOCK_DATA["SHOULDER"] - MOCK_DATA["TOP_HEAD"]),
        SegmentName.TRUNK: MOCK_DATA["SHOULDER"] + 0.4964 * (MOCK_DATA["PELVIS"] - MOCK_DATA["SHOULDER"]),
        SegmentName.UPPER_ARM: MOCK_DATA["SHOULDER"] + 0.5754 * (MOCK_DATA["ELBOW"] - MOCK_DATA["SHOULDER"]),
        SegmentName.LOWER_ARM: MOCK_DATA["ELBOW"] + 0.4559 * (MOCK_DATA["WRIST"] - MOCK_DATA["ELBOW"]),
        SegmentName.HAND: MOCK_DATA["WRIST"] + 0.7474 * (MOCK_DATA["FINGER"] - MOCK_DATA["WRIST"]),
        SegmentName.THIGH: MOCK_DATA["PELVIS"] + 0.3612 * (MOCK_DATA["KNEE"] - MOCK_DATA["PELVIS"]),
        SegmentName.SHANK: MOCK_DATA["KNEE"] + 0.4416 * (MOCK_DATA["ANKLE"] - MOCK_DATA["KNEE"]),
        SegmentName.FOOT: MOCK_DATA["ANKLE"] + 0.4014 * (MOCK_DATA["TOE"] - MOCK_DATA["ANKLE"]),
    }
    for segment in expected_male_com.keys():
        # Male
        npt.assert_almost_equal(
            male_table[segment].center_of_mass(MOCK_DATA, BiomechanicalModel()), expected_male_com[segment]
        )
        # Female
        npt.assert_almost_equal(
            female_table[segment].center_of_mass(MOCK_DATA, BiomechanicalModel()), expected_female_com[segment]
        )

    # Test inertia values
    # Male
    npt.assert_almost_equal(
        male_table[SegmentName.HEAD].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([1.78403249, 1.9281402, 1.32372727]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.TRUNK].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([29.45628403, 25.63734953, 7.81994468]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.UPPER_ARM].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([0.6163353, 0.54907527, 0.18942683]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.LOWER_ARM].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([0.34553434, 0.3185406, 0.06641158]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.HAND].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([0.33680394, 0.22474633, 0.13732405]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.THIGH].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([8.58307834, 8.58307834, 1.7604505]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.SHANK].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([1.5767262, 1.50340025, 0.25724703]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.FOOT].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([0.25336396, 0.2302559, 0.05898234]),
    )

    # Female
    npt.assert_almost_equal(
        female_table[SegmentName.HEAD].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([1.37569681, 1.6301523, 1.27604257]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.TRUNK].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([25.27673356, 22.86703742, 5.79533932]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.UPPER_ARM].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([0.55180776, 0.482664, 0.15639456]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.LOWER_ARM].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([0.26321954, 0.25521334, 0.0341423]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.HAND].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([0.22105742, 0.16159494, 0.0879844]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.THIGH].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([11.26977365, 10.96642893, 2.17216339]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.SHANK].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([1.97820678, 1.9202405, 0.23296946]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.FOOT].inertia(MOCK_DATA, BiomechanicalModel()),
        np.array([0.32291641, 0.28116169, 0.05553811]),
    )


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


def test_bilateral_segments_mass_factor():
    """Test that bilateral segments have the correct mass factor (x2)."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)

    # Bilateral segments should have 2x the unilateral mass
    bilateral_segments = [
        SegmentName.UPPER_ARM,
        SegmentName.LOWER_ARM,
        SegmentName.HAND,
        SegmentName.THIGH,
        SegmentName.SHANK,
        SegmentName.FOOT,
    ]

    # Test upper arm - should be 2 * 0.0271 * total_mass for males
    upper_arm_params = male_table[SegmentName.UPPER_ARM]
    expected_mass = 2 * 0.0271 * total_mass
    actual_mass = upper_arm_params.relative_mass(MOCK_DATA, None)
    npt.assert_almost_equal(actual_mass, expected_mass)

    # Test thigh - should be 2 * 0.1416 * total_mass for males
    thigh_params = male_table[SegmentName.THIGH]
    expected_mass = 2 * 0.1416 * total_mass
    actual_mass = thigh_params.relative_mass(MOCK_DATA, None)
    npt.assert_almost_equal(actual_mass, expected_mass)


def test_sex_differences():
    """Test that male and female tables have different values."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)
    female_table = DeLevaTable(total_mass, Sex.FEMALE)

    # Test that head mass is different between males and females
    male_head_mass = male_table[SegmentName.HEAD].relative_mass(MOCK_DATA, None)
    female_head_mass = female_table[SegmentName.HEAD].relative_mass(MOCK_DATA, None)

    # Should be 0.0694 vs 0.0669 * total_mass
    npt.assert_almost_equal(male_head_mass, 0.0694 * total_mass)
    npt.assert_almost_equal(female_head_mass, 0.0669 * total_mass)
    assert male_head_mass != female_head_mass

    # Test that trunk mass is different
    male_trunk_mass = male_table[SegmentName.TRUNK].relative_mass(MOCK_DATA, None)
    female_trunk_mass = female_table[SegmentName.TRUNK].relative_mass(MOCK_DATA, None)

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

        # Test head mass scales correctly
        male_head_mass = male_table[SegmentName.HEAD].relative_mass(MOCK_DATA, None)
        female_head_mass = female_table[SegmentName.HEAD].relative_mass(MOCK_DATA, None)

        npt.assert_almost_equal(male_head_mass, 0.0694 * total_mass)
        npt.assert_almost_equal(female_head_mass, 0.0669 * total_mass)


def test_de_leva_table_edge_cases():
    """Test edge cases for De Leva table."""
    # Test with very small mass
    small_mass = 0.1
    table = DeLevaTable(small_mass, Sex.MALE)

    # Should still work with very small masses
    head_mass = table[SegmentName.HEAD].relative_mass(MOCK_DATA, None)
    expected = 0.0694 * small_mass
    npt.assert_almost_equal(head_mass, expected)

    # Test with very large mass
    large_mass = 200.0
    table = DeLevaTable(large_mass, Sex.FEMALE)
    head_mass = table[SegmentName.HEAD].relative_mass(MOCK_DATA, None)
    expected = 0.0669 * large_mass
    npt.assert_almost_equal(head_mass, expected)


if __name__ == "__main__":
    pytest.main([__file__])
