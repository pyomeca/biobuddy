"""
Tests for biobuddy.characteristics.de_leva module
"""

import numpy as np
import numpy.testing as npt
import pytest
import biorbd

from biobuddy import (
    DeLevaTable,
    Sex,
    SegmentName,
    InertiaParameters,
    BiomechanicalModel,
    Segment,
    Translations,
    Rotations,
    Marker,
    Axis,
    SegmentCoordinateSystem,
    Mesh,
)
from biobuddy.characteristics.de_leva import point_on_vector_in_local, point_on_vector_in_global


class MOCK_DATA:
    def __init__(self):
        self.values = {
            "TOP_HEAD": np.array([0, 0, 10]),
            "BOTTOM_HEAD": np.array([0, 0, 9]),
            "HEAD_Z": np.array([0, 0, 9.5]),
            "HEAD_XZ": np.array([0.5, 0.5, 9.5]),
            "SHOULDER": np.array([0, 0, 8]),
            "SHOULDER_X": np.array([0, 0.5, 8]),
            "SHOULDER_XY": np.array([0.5, 0.5, 8]),
            "PELVIS": np.array([0, 0, 5]),
            "ELBOW": np.array([1, 0, 7]),
            "ELBOW_Y": np.array([1, 0.5, 7]),
            "ELBOW_XY": np.array([1.5, 0.5, 7]),
            "WRIST": np.array([2, 0, 6]),
            "FINGER": np.array([3, 0, 5.5]),
            "HAND_Y": np.array([2, 0.5, 6]),
            "HAND_YZ": np.array([2.5, 0.5, 6]),
            "KNEE": np.array([0, 0, 3]),
            "KNEE_Z": np.array([0, 0, 0.35]),
            "KNEE_XZ": np.array([0.5, 0.5, 3]),
            "ANKLE": np.array([0, 0, 1]),
            "ANKLE_Z": np.array([0, 0.5, 1]),
            "ANKLE_YZ": np.array([0.5, 0.5, 1]),
            "TOE": np.array([0, 0.3, 0]),
            "HEEL": np.array([0, -0.01, 0]),
            "THIGH_ORIGIN": np.array([0, 0, 4]),
            "THIGH_X": np.array([0, 0.5, 4]),
            "THIGH_Y": np.array([0.5, 0.5, 4]),
            "TRUNK": np.array([0, 0, 6]),
            "TRUNK_Y": np.array([0, 0.5, 6]),
            "TRUNK_Z": np.array([0, 0, 6.5]),
            "TRUNK_X": np.array([0.5, 0, 6]),
        }


def get_biomechanical_model(de_leva):
    """Create a simple model to test the De Leva table with"""

    model = BiomechanicalModel()
    model.add_segment(
        Segment(
            name="TRUNK",
            translations=Translations.YZ,
            rotations=Rotations.X,
            inertia_parameters=de_leva[SegmentName.TRUNK],
            mesh=Mesh(("PELVIS", "SHOULDER")),
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
    model.segments["FOOT"].add_marker(Marker("HEEL"))
    model.segments["FOOT"].add_marker(Marker("ANKLE_Z"))
    model.segments["FOOT"].add_marker(Marker("ANKLE_YZ"))
    return model


def test_point_on_vector():
    """Test point_on_vector_in_local and point_on_vector_in_global function with various inputs."""
    # Test basic functionality
    start = np.array([5, 0, 0])
    end = np.array([10, 0, 0])

    # Test at start (coef=0)
    result = point_on_vector_in_local(0.0, start, end)
    npt.assert_almost_equal(result, np.array([0, 0, 0]))
    result = point_on_vector_in_global(0.0, start, end)
    npt.assert_almost_equal(result, start)

    # Test at end (coef=1)
    result = point_on_vector_in_local(1.0, start, end)
    npt.assert_almost_equal(result, np.array([5, 0, 0]))
    result = point_on_vector_in_global(1.0, start, end)
    npt.assert_almost_equal(result, end)

    # Test at midpoint (coef=0.5)
    result = point_on_vector_in_local(0.5, start, end)
    npt.assert_almost_equal(result, np.array([2.5, 0, 0]))
    result = point_on_vector_in_global(0.5, start, end)
    npt.assert_almost_equal(result, np.array([7.5, 0, 0]))

    # Test with 3D vectors
    start = np.array([1, 2, 3])
    end = np.array([4, 6, 9])
    result = point_on_vector_in_local(0.5, start, end)
    npt.assert_almost_equal(result, np.array([1.5, 2.0, 3.0]))
    result = point_on_vector_in_global(0.5, start, end)
    npt.assert_almost_equal(result, np.array([2.5, 4, 6]))

    # Test with coefficient > 1 (extrapolation)
    result = point_on_vector_in_local(2.0, start, end)
    npt.assert_almost_equal(result, np.array([6, 8, 12]))
    result = point_on_vector_in_global(2.0, start, end)
    npt.assert_almost_equal(result, np.array([7, 10, 15]))

    # Test with negative coefficient
    result = point_on_vector_in_local(-0.5, start, end)
    npt.assert_almost_equal(result, np.array([-1.5, -2.0, -3.0]))
    result = point_on_vector_in_global(-0.5, start, end)
    npt.assert_almost_equal(result, np.array([-0.5, 0, 0]))


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


def test_de_leva_table_constructor_from_data():
    """Test DeLevaTable constructor."""
    total_mass = 70.0

    # Test male constructor
    male_table = DeLevaTable(total_mass, Sex.MALE)
    male_table.from_data(MOCK_DATA())
    assert male_table.sex == Sex.MALE
    assert hasattr(male_table, "inertial_table")

    # Test female constructor
    female_table = DeLevaTable(total_mass, Sex.FEMALE)
    female_table.from_data(MOCK_DATA())
    assert female_table.sex == Sex.FEMALE
    assert hasattr(female_table, "inertial_table")

    # Test that both tables have the same structure
    male_segments = set(male_table.inertial_table[Sex.MALE].keys())
    female_segments = set(female_table.inertial_table[Sex.FEMALE].keys())
    segment_names = [m.value for m in male_segments]
    assert male_segments == female_segments

    # Test that all expected segments are present
    expected_segments = set(SegmentName)
    assert male_segments == expected_segments
    assert all(
        segment_name in ["LOWER_ARM", "HAND", "HEAD", "SHANK", "THIGH", "TRUNK", "UPPER_ARM", "FOOT"]
        for segment_name in segment_names
    )

    # Test some values
    npt.assert_almost_equal(male_table.pelvis_position, np.array([0.0, 0.0, 5, 1.0]))
    npt.assert_almost_equal(male_table.top_head_position, np.array([0.0, 0.0, 10, 1.0]))
    npt.assert_almost_equal(male_table.right_shoulder_position, np.array([0.0, 0.0, 8, 1.0]))
    npt.assert_almost_equal(male_table.left_shoulder_position, np.array([0.0, 0.0, 8, 1.0]))
    npt.assert_almost_equal(male_table.right_elbow_position, np.array([0.0, 0.0, 7, 1.0]))
    npt.assert_almost_equal(male_table.left_elbow_position, np.array([0.0, 0.0, 7, 1.0]))
    npt.assert_almost_equal(male_table.right_wrist_position, np.array([0.0, 0.0, 6, 1.0]))
    npt.assert_almost_equal(male_table.left_wrist_position, np.array([0.0, 0.0, 6, 1.0]))
    npt.assert_almost_equal(male_table.right_finger_position, np.array([0.0, 0.0, 5.5, 1.0]))
    npt.assert_almost_equal(male_table.left_finger_position, np.array([0.0, 0.0, 5.5, 1.0]))
    npt.assert_almost_equal(male_table.right_knee_position, np.array([0.0, 0.0, 3, 1.0]))
    npt.assert_almost_equal(male_table.left_knee_position, np.array([0.0, 0.0, 3, 1.0]))
    npt.assert_almost_equal(male_table.right_ankle_position, np.array([0.0, 0.0, 1, 1.0]))
    npt.assert_almost_equal(male_table.left_ankle_position, np.array([0.0, 0.0, 1, 1.0]))
    npt.assert_almost_equal(male_table.right_toe_position, np.array([0.31, 0.0, 1, 1.0]))
    npt.assert_almost_equal(male_table.left_toe_position, np.array([0.31, 0.0, 1, 1.0]))

    # Test intermediate values
    npt.assert_almost_equal(male_table.total_height, 10)
    npt.assert_almost_equal(male_table.pelvis_height, 5)
    npt.assert_almost_equal(male_table.trunk_length, 3)
    npt.assert_almost_equal(male_table.hand_length, 0.5)
    npt.assert_almost_equal(male_table.lower_arm_length, 1)
    npt.assert_almost_equal(male_table.upper_arm_length, 1)
    npt.assert_almost_equal(male_table.shoulder_width, 0)
    npt.assert_almost_equal(male_table.thigh_length, 2)
    npt.assert_almost_equal(male_table.shank_length, 2)
    npt.assert_almost_equal(male_table.hip_width, 0)
    npt.assert_almost_equal(male_table.foot_length, 0.31)

    # Test the simple model
    male_model = male_table.to_simple_model()
    npt.assert_almost_equal(
        male_model.total_com_in_global().reshape(
            4,
        ),
        np.array([3.75010100e-03, 0.00000000e00, 5.51640248e00, 1.00000000e00]),
    )
    male_model.to_biomod("temporary_path.bioMod")
    male_model_biomod = biorbd.Model("temporary_path.bioMod")
    npt.assert_almost_equal(
        male_model_biomod.bodyInertia(np.zeros((male_model.nb_q,))).to_array(),
        np.array([[290.94771259, 0.0, 1.18558758], [0.0, 287.1803482, 0.0], [1.18558758, 0.0, 11.3342956]]),
    )


def test_de_leva_table_constructor_from_measurements():
    """Test DeLevaTable constructor."""
    total_mass = 70.0
    total_height = 1.70
    ankle_height = 0.01
    knee_height = 0.4
    pelvis_height = 0.95
    shoulder_height = 1.5
    finger_span = 1.70
    wrist_span = 1.5
    elbow_span = 1.2
    shoulder_span = 0.0
    foot_length = 0.35
    hip_width = 0.0

    # Test male constructor
    male_table = DeLevaTable(total_mass, Sex.MALE)
    male_table.from_measurements(
        total_height,
        ankle_height,
        knee_height,
        pelvis_height,
        shoulder_height,
        finger_span,
        wrist_span,
        elbow_span,
        shoulder_span,
        hip_width,
        foot_length,
    )
    assert male_table.sex == Sex.MALE
    assert hasattr(male_table, "inertial_table")

    # Test female constructor
    female_table = DeLevaTable(total_mass, Sex.FEMALE)
    female_table.from_measurements(
        total_height,
        ankle_height,
        knee_height,
        pelvis_height,
        shoulder_height,
        finger_span,
        wrist_span,
        elbow_span,
        shoulder_span,
        hip_width,
        foot_length,
    )
    assert female_table.sex == Sex.FEMALE
    assert hasattr(female_table, "inertial_table")

    # Test that both tables have the same structure
    male_segments = set(male_table.inertial_table[Sex.MALE].keys())
    female_segments = set(female_table.inertial_table[Sex.FEMALE].keys())
    segment_names = [m.value for m in male_segments]
    assert male_segments == female_segments

    # Test that all expected segments are present
    expected_segments = set(SegmentName)
    assert male_segments == expected_segments
    assert all(
        segment_name in ["LOWER_ARM", "HAND", "HEAD", "SHANK", "THIGH", "TRUNK", "UPPER_ARM", "FOOT"]
        for segment_name in segment_names
    )

    # Test some values
    npt.assert_almost_equal(female_table.pelvis_position, np.array([0.0, 0.0, 0.95, 1.0]))
    npt.assert_almost_equal(female_table.top_head_position, np.array([0.0, 0.0, 1.7, 1.0]))
    npt.assert_almost_equal(female_table.right_shoulder_position, np.array([0.0, 0.0, 1.5, 1.0]))
    npt.assert_almost_equal(female_table.left_shoulder_position, np.array([0.0, 0.0, 1.5, 1.0]))
    npt.assert_almost_equal(female_table.right_elbow_position, np.array([0.0, 0.0, 0.9, 1.0]))
    npt.assert_almost_equal(female_table.left_elbow_position, np.array([0.0, 0.0, 0.9, 1.0]))
    npt.assert_almost_equal(female_table.right_wrist_position, np.array([0.0, 0.0, 0.75, 1.0]))
    npt.assert_almost_equal(female_table.left_wrist_position, np.array([0.0, 0.0, 0.75, 1.0]))
    npt.assert_almost_equal(female_table.right_finger_position, np.array([0.0, 0.0, 0.65, 1.0]))
    npt.assert_almost_equal(female_table.left_finger_position, np.array([0.0, 0.0, 0.65, 1.0]))
    npt.assert_almost_equal(female_table.right_knee_position, np.array([0.0, 0.0, 0.4, 1.0]))
    npt.assert_almost_equal(female_table.left_knee_position, np.array([0.0, 0.0, 0.4, 1.0]))
    npt.assert_almost_equal(female_table.right_ankle_position, np.array([0.0, 0.0, 0.01, 1.0]))
    npt.assert_almost_equal(female_table.left_ankle_position, np.array([0.0, 0.0, 0.01, 1.0]))
    npt.assert_almost_equal(female_table.right_toe_position, np.array([0.35, 0.0, 0.01, 1.0]))
    npt.assert_almost_equal(female_table.left_toe_position, np.array([0.35, 0.0, 0.01, 1.0]))

    # Test intermediate values
    npt.assert_almost_equal(female_table.total_height, total_height)
    npt.assert_almost_equal(female_table.pelvis_height, pelvis_height)
    npt.assert_almost_equal(female_table.trunk_length, 0.55)
    npt.assert_almost_equal(female_table.hand_length, 0.1)
    npt.assert_almost_equal(female_table.lower_arm_length, 0.15)
    npt.assert_almost_equal(female_table.upper_arm_length, 0.6)
    npt.assert_almost_equal(female_table.shoulder_width, 0)
    npt.assert_almost_equal(female_table.thigh_length, 0.55)
    npt.assert_almost_equal(female_table.shank_length, 0.39)
    npt.assert_almost_equal(female_table.hip_width, 0)
    npt.assert_almost_equal(female_table.foot_length, 0.35)

    # Test the simple model
    female_model = female_table.to_simple_model()
    npt.assert_almost_equal(
        female_model.total_com_in_global().reshape(
            4,
        ),
        np.array([0.00362464, 0.0, 0.96795513, 1.0]),
    )
    female_model.to_biomod("temporary_path.bioMod")
    female_model_biomod = biorbd.Model("temporary_path.bioMod")
    npt.assert_almost_equal(
        female_model_biomod.bodyInertia(np.zeros((female_model.nb_q,))).to_array(),
        np.array([[12.4874833, 0.0, 0.24305711], [0.0, 12.40340046, 0.0], [0.24305711, 0.0, 0.44760615]]),
    )


def test_de_leva_table_getitem():
    """Test DeLevaTable.__getitem__ method."""
    total_mass = 70.0
    male_table = DeLevaTable(total_mass, Sex.MALE)
    male_table.from_data(MOCK_DATA())
    female_table = DeLevaTable(total_mass, Sex.FEMALE)
    female_table.from_data(MOCK_DATA())

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
    mock_values = MOCK_DATA().values
    male_table = DeLevaTable(total_mass, Sex.MALE)
    male_table.from_data(MOCK_DATA())
    female_table = DeLevaTable(total_mass, Sex.FEMALE)
    female_table.from_data(MOCK_DATA())

    # Test the MASS values
    expected_male_masses = {
        SegmentName.HEAD: 0.0694 * total_mass,
        SegmentName.TRUNK: 0.4346 * total_mass,
        SegmentName.UPPER_ARM: 0.0271 * total_mass,  # bilateral
        SegmentName.LOWER_ARM: 0.0162 * total_mass,  # bilateral
        SegmentName.HAND: 0.0061 * total_mass,  # bilateral
        SegmentName.THIGH: 0.1416 * total_mass,  # bilateral
        SegmentName.SHANK: 0.0433 * total_mass,  # bilateral
        SegmentName.FOOT: 0.0137 * total_mass,  # bilateral
    }
    expected_female_masses = {
        SegmentName.HEAD: 0.0669 * total_mass,
        SegmentName.TRUNK: 0.4257 * total_mass,
        SegmentName.UPPER_ARM: 0.0255 * total_mass,  # bilateral
        SegmentName.LOWER_ARM: 0.0138 * total_mass,  # bilateral
        SegmentName.HAND: 0.0056 * total_mass,  # bilateral
        SegmentName.THIGH: 0.1478 * total_mass,  # bilateral
        SegmentName.SHANK: 0.0481 * total_mass,  # bilateral
        SegmentName.FOOT: 0.0129 * total_mass,  # bilateral
    }
    for segment in expected_male_masses.keys():
        # Male
        npt.assert_almost_equal(
            male_table[segment].relative_mass(mock_values, BiomechanicalModel()), expected_male_masses[segment]
        )
        # Female
        npt.assert_almost_equal(
            female_table[segment].relative_mass(mock_values, BiomechanicalModel()), expected_female_masses[segment]
        )

    # Test the center of mass values : coef (end - start)
    expected_male_com = {
        SegmentName.HEAD: (1 - 0.5002) * (mock_values["TOP_HEAD"] - mock_values["SHOULDER"]),
        SegmentName.TRUNK: (1 - 0.5138) * (mock_values["SHOULDER"] - mock_values["PELVIS"]),
        SegmentName.UPPER_ARM: np.array([0, 0, (1 - 0.5772) * (mock_values["ELBOW"][2] - mock_values["SHOULDER"][2])]),
        SegmentName.LOWER_ARM: np.array([0, 0, (1 - 0.4574) * (mock_values["WRIST"][2] - mock_values["ELBOW"][2])]),
        SegmentName.HAND: np.array([0, 0, 0.3624 * (mock_values["FINGER"][2] - mock_values["WRIST"][2])]),
        SegmentName.THIGH: np.array([0, 0, 0.4095 * (mock_values["KNEE"][2] - mock_values["PELVIS"][2])]),
        SegmentName.SHANK: np.array([0, 0, 0.4459 * (mock_values["ANKLE"][2] - mock_values["KNEE"][2])]),
        SegmentName.FOOT: np.array([0.4415 * (mock_values["TOE"][1] - mock_values["HEEL"][1]), 0, 0]),
    }
    expected_female_com = {
        SegmentName.HEAD: (1 - 0.4841) * (mock_values["TOP_HEAD"] - mock_values["SHOULDER"]),
        SegmentName.TRUNK: (1 - 0.4964) * (mock_values["SHOULDER"] - mock_values["PELVIS"]),
        SegmentName.UPPER_ARM: np.array([0, 0, (1 - 0.5754) * (mock_values["ELBOW"][2] - mock_values["SHOULDER"][2])]),
        SegmentName.LOWER_ARM: np.array([0, 0, (1 - 0.4559) * (mock_values["WRIST"][2] - mock_values["ELBOW"][2])]),
        SegmentName.HAND: np.array([0, 0, 0.3427 * (mock_values["FINGER"][2] - mock_values["WRIST"][2])]),
        SegmentName.THIGH: np.array([0, 0, 0.3612 * (mock_values["KNEE"][2] - mock_values["PELVIS"][2])]),
        SegmentName.SHANK: np.array([0, 0, 0.4416 * (mock_values["ANKLE"][2] - mock_values["KNEE"][2])]),
        SegmentName.FOOT: np.array([0.4014 * (mock_values["TOE"][1] - mock_values["HEEL"][1]), 0, 0]),
    }
    for segment in expected_male_com.keys():
        # Male
        npt.assert_almost_equal(
            male_table[segment].center_of_mass(mock_values, BiomechanicalModel())[:3], expected_male_com[segment][:3]
        )
        # Female
        npt.assert_almost_equal(
            female_table[segment].center_of_mass(mock_values, BiomechanicalModel())[:3],
            expected_female_com[segment][:3],
        )

    # Test inertia values
    # Male
    npt.assert_almost_equal(
        male_table[SegmentName.HEAD].inertia(mock_values, BiomechanicalModel()),
        np.array([1.78403249, 1.9281402, 1.32372727]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.TRUNK].inertia(mock_values, BiomechanicalModel()),
        np.array([29.45628403, 25.63734953, 7.81994468]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.UPPER_ARM].inertia(mock_values, BiomechanicalModel()),
        np.array([0.15408382, 0.13726882, 0.04735671]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.LOWER_ARM].inertia(mock_values, BiomechanicalModel()),
        np.array([0.08638358, 0.07963515, 0.01660289]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.HAND].inertia(mock_values, BiomechanicalModel()),
        np.array([0.00885427, 0.00589527, 0.00361413]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.THIGH].inertia(mock_values, BiomechanicalModel()),
        np.array([4.29153917, 4.29153917, 0.88022525]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.SHANK].inertia(mock_values, BiomechanicalModel()),
        np.array([0.7883631, 0.75170012, 0.12862352]),
    )
    npt.assert_almost_equal(
        male_table[SegmentName.FOOT].inertia(mock_values, BiomechanicalModel()),
        np.array([0.00608707, 0.0055319, 0.00141705]),
    )

    # Female
    npt.assert_almost_equal(
        female_table[SegmentName.HEAD].inertia(mock_values, BiomechanicalModel()),
        np.array([1.37569681, 1.6301523, 1.27604257]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.TRUNK].inertia(mock_values, BiomechanicalModel()),
        np.array([25.27673356, 22.86703742, 5.79533932]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.UPPER_ARM].inertia(mock_values, BiomechanicalModel()),
        np.array([0.13795194, 0.120666, 0.03909864]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.LOWER_ARM].inertia(mock_values, BiomechanicalModel()),
        np.array([0.06580489, 0.06380333, 0.00853558]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.HAND].inertia(mock_values, BiomechanicalModel()),
        np.array([0.00583453, 0.00423987, 0.00331789]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.THIGH].inertia(mock_values, BiomechanicalModel()),
        np.array([5.63488682, 5.48321446, 1.0860817]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.SHANK].inertia(mock_values, BiomechanicalModel()),
        np.array([0.98910339, 0.96012025, 0.11648473]),
    )
    npt.assert_almost_equal(
        female_table[SegmentName.FOOT].inertia(mock_values, BiomechanicalModel()),
        np.array([0.00775807, 0.00675491, 0.0013343]),
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
        table.from_data(MOCK_DATA())

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


def test_sex_differences():
    """Test that male and female tables have different values."""
    total_mass = 70.0
    mock_values = MOCK_DATA().values
    male_table = DeLevaTable(total_mass, Sex.MALE)
    male_table.from_data(MOCK_DATA())
    female_table = DeLevaTable(total_mass, Sex.FEMALE)
    female_table.from_data(MOCK_DATA())

    # Test that head mass is different between males and females
    male_head_mass = male_table[SegmentName.HEAD].relative_mass(mock_values, None)
    female_head_mass = female_table[SegmentName.HEAD].relative_mass(mock_values, None)

    # Should be 0.0694 vs 0.0669 * total_mass
    npt.assert_almost_equal(male_head_mass, 0.0694 * total_mass)
    npt.assert_almost_equal(female_head_mass, 0.0669 * total_mass)
    assert male_head_mass != female_head_mass

    # Test that trunk mass is different
    male_trunk_mass = male_table[SegmentName.TRUNK].relative_mass(mock_values, None)
    female_trunk_mass = female_table[SegmentName.TRUNK].relative_mass(mock_values, None)

    # Should be 0.4346 vs 0.4257 * total_mass
    npt.assert_almost_equal(male_trunk_mass, 0.4346 * total_mass)
    npt.assert_almost_equal(female_trunk_mass, 0.4257 * total_mass)
    assert male_trunk_mass != female_trunk_mass


def test_de_leva_table_different_masses():
    """Test De Leva table with different total masses."""
    masses = [50.0, 70.0, 100.0, 120.0]
    mock_values = MOCK_DATA().values

    for total_mass in masses:
        male_table = DeLevaTable(total_mass, Sex.MALE)
        male_table.from_data(MOCK_DATA())
        female_table = DeLevaTable(total_mass, Sex.FEMALE)
        female_table.from_data(MOCK_DATA())

        # Test head mass scales correctly
        male_head_mass = male_table[SegmentName.HEAD].relative_mass(mock_values, None)
        female_head_mass = female_table[SegmentName.HEAD].relative_mass(mock_values, None)

        npt.assert_almost_equal(male_head_mass, 0.0694 * total_mass)
        npt.assert_almost_equal(female_head_mass, 0.0669 * total_mass)


def test_de_leva_table_edge_cases():
    """Test edge cases for De Leva table."""
    # Test with very small mass
    small_mass = 0.1
    table = DeLevaTable(small_mass, Sex.MALE)
    table.from_data(MOCK_DATA())
    mock_values = MOCK_DATA().values

    # Should still work with very small masses
    head_mass = table[SegmentName.HEAD].relative_mass(mock_values, None)
    expected = 0.0694 * small_mass
    npt.assert_almost_equal(head_mass, expected)

    # Test with very large mass
    large_mass = 200.0
    table = DeLevaTable(large_mass, Sex.FEMALE)
    table.from_data(MOCK_DATA())
    head_mass = table[SegmentName.HEAD].relative_mass(mock_values, None)
    expected = 0.0669 * large_mass
    npt.assert_almost_equal(head_mass, expected)


def test_model_evaluation():
    """Test that the model can be evaluated with the De Leva table."""
    total_mass = 70.0
    sex = Sex.FEMALE
    de_leva_table = DeLevaTable(total_mass=total_mass, sex=sex)
    de_leva_table.from_data(MOCK_DATA())

    model = get_biomechanical_model(de_leva_table)

    # Check only the trunk segment
    segment = model.segments[1]
    assert segment.name == SegmentName.TRUNK.value
    assert segment.parent_name == "root"
    assert segment.translations == Translations.YZ
    assert segment.rotations == Rotations.X
    assert segment.q_ranges is None
    assert segment.qdot_ranges is None
    assert segment.segment_coordinate_system is None
    npt.assert_almost_equal(segment.inertia_parameters.relative_mass(MOCK_DATA().values, model), 29.7990)
    npt.assert_almost_equal(
        segment.inertia_parameters.center_of_mass(MOCK_DATA().values, model)[:3], np.array([0.0, 0.0, 1.5108])
    )
    npt.assert_almost_equal(
        segment.inertia_parameters.inertia(MOCK_DATA().values, model)[:3],
        np.array([25.27673356, 22.86703742, 5.79533932]),
    )
    npt.assert_almost_equal(segment.mesh.functions[0](MOCK_DATA().values, model)[:3], np.array([0, 0, 5]))
    npt.assert_almost_equal(segment.mesh.functions[1](MOCK_DATA().values, model)[:3], np.array([0, 0, 8]))
    assert segment.mesh_file is None

    model_real = model.to_real(MOCK_DATA())

    # Check only the trunk segment
    segment = model_real.segments[1]
    assert segment.name == SegmentName.TRUNK.value
    assert segment.parent_name == "root"
    assert segment.translations == Translations.YZ
    assert segment.rotations == Rotations.X
    assert segment.q_ranges is None
    assert segment.qdot_ranges is None
    npt.assert_almost_equal(segment.segment_coordinate_system.scs.rt_matrix, np.eye(4))
    npt.assert_almost_equal(segment.inertia_parameters.mass, 29.7990)
    npt.assert_almost_equal(segment.inertia_parameters.center_of_mass[:3, 0], np.array([0.0, 0.0, 1.5108]))
    npt.assert_almost_equal(
        np.diag(segment.inertia_parameters.inertia)[:3], np.array([25.27673356, 22.86703742, 5.79533932])
    )
    npt.assert_almost_equal(segment.mesh.positions[:3, 0], np.array([0, 0, 5]))
    npt.assert_almost_equal(segment.mesh.positions[:3, 1], np.array([0, 0, 8]))
    assert segment.mesh_file is None
