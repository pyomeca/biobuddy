import os
import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import (
    YeadonDensitySet,
    YeadonSegmentName,
    YeadonTable,
    YeadonMeasures,
    LengthUnits,
)

MALE1_MEASUREMENTS_CM = YeadonMeasures(
    Ls1L=11.9,
    La2L=26.7,
    Lb2L=26.1,
    Ls2L=19.6,
    La3L=36.9,
    Lb3L=34.9,
    Ls3L=34.7,
    La4L=54.1,
    Lb4L=52.3,
    Ls4L=54.0,
    La5L=5.5,
    Lb5L=5.0,
    Ls5L=50.0,
    La6L=8.6,
    Lb6L=7.8,
    Ls6L=12.0,
    La7L=16.0,
    Lb7L=16.1,
    Ls7L=17.1,
    Ls8L=26.3,
    La0p=39.4,
    Lb0p=42.5,
    La1p=34.7,
    Lb1p=33.5,
    Ls0p=93.0,
    La2p=28.2,
    Lb2p=28.2,
    Ls1p=88.0,
    La3p=28.9,
    Lb3p=28.9,
    Ls2p=90.0,
    La4p=17.0,
    Lb4p=17.0,
    Ls3p=101.8,
    La5p=24.5,
    Lb5p=24.5,
    Ls5p=42.1,
    La6p=21.0,
    Lb6p=19.6,
    Ls6p=52.0,
    La7p=11.7,
    Lb7p=11.9,
    Ls7p=57.9,
    La4w=5.6,
    Lb4w=5.7,
    Ls0w=31.1,
    La5w=9.5,
    Lb5w=9.5,
    Ls1w=30.6,
    La6w=7.7,
    Lb6w=7.9,
    Ls2w=31.3,
    La7w=4.8,
    Lb7w=4.6,
    Ls3w=37.1,
    Ls4w=42.5,
    Lj1L=10.1,
    Lk1L=10.1,
    Ls4d=18.2,
    Lj3L=41.0,
    Lk3L=44.0,
    Lj4L=55.4,
    Lk4L=59.7,
    Lj5L=79.7,
    Lk5L=82.9,
    Lj6L=1.0,
    Lk6L=1.0,
    Lj8L=13.5,
    Lk8L=13.2,
    Lj9L=18.0,
    Lk9L=16.4,
    Lj1p=61.5,
    Lk1p=63.5,
    Lj2p=53.8,
    Lk2p=55.6,
    Lj3p=35.5,
    Lk3p=34.5,
    Lj4p=39.0,
    Lk4p=38.3,
    Lj5p=22.9,
    Lk5p=24.7,
    Lj6p=31.3,
    Lk6p=30.9,
    Lj7p=24.6,
    Lk7p=24.3,
    Lj8p=22.6,
    Lk8p=22.9,
    Lj9p=14.1,
    Lk9p=14.6,
    Lj8w=9.2,
    Lk8w=9.1,
    Lj9w=6.2,
    Lk9w=6.5,
    Lj6d=11.8,
    Lk6d=10.7,
    units=LengthUnits.CM,
)


def test_table_initialization():

    # Default parameters
    table = YeadonTable()
    assert table.symmetric == True
    assert table.density_set == "Dempster"
    assert table.total_mass is None

    # Other parameters
    table = YeadonTable(
        symmetric=False,
        density_set=YeadonDensitySet.CLAUSER,
        total_mass=100,
    )
    assert table.symmetric == False
    assert table.density_set == "Clauser"
    assert table.total_mass == 100

    # Other empty attributes
    assert table.human is None
    assert table.inertial_table == {}
    assert table.measures is None
    assert table.pelvis_position is None
    assert table.thorax_position is None
    assert table.chest_head_position is None
    assert table.top_head_position is None
    assert table.right_shoulder_position is None
    assert table.right_elbow_position is None
    assert table.right_wrist_position is None
    assert table.left_shoulder_position is None
    assert table.left_elbow_position is None
    assert table.left_wrist_position is None
    assert table.right_hip_position is None
    assert table.right_knee_position is None
    assert table.right_ankle_position is None
    assert table.left_hip_position is None
    assert table.left_knee_position is None
    assert table.left_ankle_position is None


def test_yeadon_table_from_measurements_vs_from_file():

    # Create table from measurements
    table_from_measurements = YeadonTable(
        symmetric=True,
        density_set=YeadonDensitySet.DEMPSTER,
        total_mass=81.7,
    )
    table_from_measurements.from_measurements(MALE1_MEASUREMENTS_CM)

    # Create table from file
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = parent_path + "/examples/models"
    measures_filepath = f"{root_path}/yeadon_measures.txt"
    table_from_file = YeadonTable(
        symmetric=True,
        density_set=YeadonDensitySet.DEMPSTER,
    )
    table_from_file.from_file(measures_filepath)

    # Total mass
    npt.assert_almost_equal(table_from_measurements.mass, 81.7)
    npt.assert_almost_equal(table_from_file.mass, 81.7)

    # Whole body inertial characteristics
    npt.assert_array_almost_equal(
        table_from_measurements.center_of_mass.reshape(3), np.array([0.0, 0.0, 0.05607438]), decimal=6
    )
    npt.assert_array_almost_equal(
        table_from_file.center_of_mass.reshape(3), np.array([0.0, 0.0, 0.05607438]), decimal=6
    )
    npt.assert_array_almost_equal(
        np.diag(table_from_measurements.inertia), np.array([11.82405469, 12.63679392, 1.18010840]), decimal=5
    )
    npt.assert_array_almost_equal(
        np.diag(table_from_file.inertia), np.array([11.82405469, 12.63679392, 1.18010840]), decimal=5
    )

    # All other inertia characteristics by segment
    expected_segment_inertia = {
        YeadonSegmentName.PELVIS: (
            12.924675966035618,
            np.array([0.0, 0.0, 0.09628887]),
            np.array([0.08072403, 0.12168453, 0.13281317]),
        ),
        YeadonSegmentName.THORAX: (
            10.058056232683622,
            np.array([0.0, 0.0, 0.07808894]),
            np.array([0.04603482, 0.09715065, 0.11883165]),
        ),
        YeadonSegmentName.CHEST_HEAD: (
            16.97462828606515,
            np.array([0.0, 0.0, 0.14598986]),
            np.array([0.24446939, 0.33287978, 0.17705416]),
        ),
        YeadonSegmentName.LEFT_UPPER_ARM: (
            2.7752720379491986,
            np.array([0.0, 0.0, -0.12392401]),
            np.array([0.01789092, 0.01789092, 0.00438295]),
        ),
        YeadonSegmentName.LEFT_FOREARM_HAND: (
            2.078396467185324,
            np.array([0.0, 0.0, -0.16510591]),
            np.array([0.02833551, 0.02851044, 0.00165017]),
        ),
        YeadonSegmentName.RIGHT_UPPER_ARM: (
            2.7752720379491986,
            np.array([0.0, 0.0, -0.12392401]),
            np.array([0.01789092, 0.01789092, 0.00438295]),
        ),
        YeadonSegmentName.RIGHT_FOREARM_HAND: (
            2.078396467185324,
            np.array([0.0, 0.0, -0.16510591]),
            np.array([0.02833551, 0.02851044, 0.00165017]),
        ),
        YeadonSegmentName.LEFT_THIGH: (
            11.197398479113708,
            np.array([0.0, 0.0, -0.18684523]),
            np.array([0.18938242, 0.18938242, 0.04690137]),
        ),
        YeadonSegmentName.LEFT_SHANK_FOOT: (
            4.820252773359574,
            np.array([0.0, 0.0, -0.23620995]),
            np.array([0.1144481, 0.1143982, 0.00676904]),
        ),
        YeadonSegmentName.RIGHT_THIGH: (
            11.197398479113708,
            np.array([0.0, 0.0, -0.18684523]),
            np.array([0.18938242, 0.18938242, 0.04690137]),
        ),
        YeadonSegmentName.RIGHT_SHANK_FOOT: (
            4.820252773359574,
            np.array([0.0, 0.0, -0.23620995]),
            np.array([0.1144481, 0.1143982, 0.00676904]),
        ),
    }
    for segment_name, (expected_mass, expected_com, expected_inertia_diag) in expected_segment_inertia.items():
        for table in (table_from_measurements, table_from_file):
            segment = table[segment_name]
            npt.assert_almost_equal(segment.mass, expected_mass)
            npt.assert_array_almost_equal(segment.center_of_mass[:3, 0], expected_com, decimal=6)
            npt.assert_array_almost_equal(np.diag(segment.inertia)[:3], expected_inertia_diag, decimal=6)

    # Test joint positions
    expected_positions = {
        "pelvis_position": np.array([0.0, 0.0, 0.0, 1.0]),
        "thorax_position": np.array([0.0, 0.0, 0.196, 1.0]),
        "chest_head_position": np.array([0.0, 0.0, 0.347, 1.0]),
        "top_head_position": np.array([0.0, 0.0, 0.763, 1.0]),
        "right_shoulder_position": np.array([-0.2125, 0.0, 0.54, 1.0]),
        "right_elbow_position": np.array([-0.2125, 0.0, 0.276, 1.0]),
        "right_wrist_position": np.array([-0.2125, 0.0, -0.1525, 1.0]),
        "left_shoulder_position": np.array([0.2125, 0.0, 0.54, 1.0]),
        "left_elbow_position": np.array([0.2125, 0.0, 0.276, 1.0]),
        "left_wrist_position": np.array([0.2125, 0.0, -0.1525, 1.0]),
        "right_hip_position": np.array([-0.07775, 0.0, 0.0, 1.0]),
        "right_knee_position": np.array([-0.07775, 0.0, -0.425, 1.0]),
        "right_ankle_position": np.array([-0.07775, 0.0, -0.985, 1.0]),
        "left_hip_position": np.array([0.07775, 0.0, 0.0, 1.0]),
        "left_knee_position": np.array([0.07775, 0.0, -0.425, 1.0]),
        "left_ankle_position": np.array([0.07775, 0.0, -0.985, 1.0]),
    }
    for attribute_name, expected_position in expected_positions.items():
        npt.assert_almost_equal(getattr(table_from_measurements, attribute_name), expected_position, decimal=6)
        npt.assert_almost_equal(getattr(table_from_file, attribute_name), expected_position, decimal=6)


def _parse_yeadon_measurement_file(filepath: str) -> tuple[dict[str, float], float | None]:
    """
    Minimal reader for the YAML-style measurement files used by `yeadon`/`YeadonTable.to_file`.
    Returns the 95 measurements converted to meters, and the total mass (if present) in kilograms.
    """
    raw_values = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, value = line.split(":")
            raw_values[name.strip()] = float(value.strip())

    conversion_factor = raw_values.pop("measurementconversionfactor")
    total_mass = raw_values.pop("totalmass", None)
    measurements_in_meters = {name: value * conversion_factor for name, value in raw_values.items()}
    return measurements_in_meters, total_mass


def test_yeadon_table_to_file_exports_yeadon_measurements():

    # Create table from file (symmetric=False so the raw measurements are preserved and can be compared as-is;
    # symmetric=True would average the left/right measurements together before they get exported)
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = parent_path + "/examples/models"
    measures_filepath = f"{root_path}/yeadon_measures.txt"
    table_from_file = YeadonTable(
        symmetric=False,
        density_set=YeadonDensitySet.DEMPSTER,
    )
    table_from_file.from_file(measures_filepath)

    # Export to file
    new_measures_filepath = f"{root_path}/yeadon_measures_export.txt"
    table_from_file.to_file(new_measures_filepath)

    # Read both files and make sure their content match (the original file and the exported file do not use the
    # same units/order, so we compare the parsed measurements in meters and the total mass in kg instead of the
    # raw text line by line).
    original_measurements, original_total_mass = _parse_yeadon_measurement_file(measures_filepath)
    exported_measurements, exported_total_mass = _parse_yeadon_measurement_file(new_measures_filepath)

    assert set(exported_measurements.keys()) == set(YeadonTable.measurement_names())
    assert set(original_measurements.keys()) == set(YeadonTable.measurement_names())
    for name in YeadonTable.measurement_names():
        npt.assert_almost_equal(exported_measurements[name], original_measurements[name], decimal=6)
    npt.assert_almost_equal(exported_total_mass, original_total_mass, decimal=6)

    # Delete the bad file created
    os.remove(new_measures_filepath)


def test_yeadon_table_creates_simple_model():
    # Create table from measurements
    table = YeadonTable(
        symmetric=True,
        density_set=YeadonDensitySet.DEMPSTER,
        total_mass=81.7,
    )
    table.from_measurements(MALE1_MEASUREMENTS_CM)

    model = table.to_simple_model()

    assert model.segment_names == [
        "root",
        "P",
        "T",
        "C",
        "A1",
        "A2",
        "B1",
        "B2",
        "J1",
        "J2",
        "K1",
        "K2",
    ]
    assert model.segments["P"].parent_name == "root"
    assert model.segments["T"].parent_name == "P"
    assert model.segments["C"].parent_name == "T"
    assert model.segments["A1"].parent_name == "C"
    assert model.segments["A2"].parent_name == "A1"
    assert model.segments["B1"].parent_name == "C"
    assert model.segments["B2"].parent_name == "B1"
    assert model.segments["J1"].parent_name == "P"
    assert model.segments["J2"].parent_name == "J1"
    assert model.segments["K1"].parent_name == "P"
    assert model.segments["K2"].parent_name == "K1"
    npt.assert_almost_equal(model.mass, table.mass)

    # Each segment of the simple model should carry the same inertial characteristics as the table
    for segment_name in YeadonSegmentName:
        model_segment = model.segments[segment_name.value]
        table_segment = table[segment_name]
        npt.assert_almost_equal(model_segment.inertia_parameters.mass, table_segment.mass)
        npt.assert_array_almost_equal(model_segment.inertia_parameters.center_of_mass, table_segment.center_of_mass)
        npt.assert_array_almost_equal(model_segment.inertia_parameters.inertia, table_segment.inertia)

    # The segment coordinate systems should be positioned at the joint centers computed from the measurements
    # (since the segments are stacked without any rotation in this default/neutral configuration, each child's
    # local origin is simply the difference between its own joint center and its parent's)
    npt.assert_array_almost_equal(
        model.segments["P"].segment_coordinate_system.scs.translation, table.pelvis_position[:3]
    )
    npt.assert_array_almost_equal(
        model.segments["T"].segment_coordinate_system.scs.translation,
        table.thorax_position[:3] - table.pelvis_position[:3],
    )
    npt.assert_array_almost_equal(
        model.segments["C"].segment_coordinate_system.scs.translation,
        table.chest_head_position[:3] - table.thorax_position[:3],
    )
    npt.assert_array_almost_equal(
        model.segments["A1"].segment_coordinate_system.scs.translation,
        table.left_shoulder_position[:3] - table.chest_head_position[:3],
    )
    npt.assert_array_almost_equal(
        model.segments["A2"].segment_coordinate_system.scs.translation,
        table.left_elbow_position[:3] - table.left_shoulder_position[:3],
    )
    npt.assert_array_almost_equal(
        model.segments["B1"].segment_coordinate_system.scs.translation,
        table.right_shoulder_position[:3] - table.chest_head_position[:3],
    )
    npt.assert_array_almost_equal(
        model.segments["B2"].segment_coordinate_system.scs.translation,
        table.right_elbow_position[:3] - table.right_shoulder_position[:3],
    )
    npt.assert_array_almost_equal(
        model.segments["J1"].segment_coordinate_system.scs.translation,
        table.left_hip_position[:3] - table.pelvis_position[:3],
    )
    npt.assert_array_almost_equal(
        model.segments["J2"].segment_coordinate_system.scs.translation,
        table.left_knee_position[:3] - table.left_hip_position[:3],
    )
    npt.assert_array_almost_equal(
        model.segments["K1"].segment_coordinate_system.scs.translation,
        table.right_hip_position[:3] - table.pelvis_position[:3],
    )
    npt.assert_array_almost_equal(
        model.segments["K2"].segment_coordinate_system.scs.translation,
        table.right_knee_position[:3] - table.right_hip_position[:3],
    )


def test_yeadon_measures_unit_conversion():
    """The measurements should be stored internally in meters, regardless of the input units."""
    names = YeadonTable.measurement_names()

    measures_m = YeadonMeasures(**{name: 0.5 for name in names}, units=LengthUnits.M)
    measures_cm = YeadonMeasures(**{name: 50.0 for name in names}, units=LengthUnits.CM)
    measures_mm = YeadonMeasures(**{name: 500.0 for name in names}, units=LengthUnits.MM)

    for name in names:
        npt.assert_almost_equal(getattr(measures_m, name), 0.5)
        npt.assert_almost_equal(getattr(measures_cm, name), 0.5)
        npt.assert_almost_equal(getattr(measures_mm, name), 0.5)


def test_yeadon_table_symmetric_vs_asymmetric():
    """With symmetric=True, the left and right limb segments should have identical inertial characteristics
    even though the underlying measurements differ between the left (La*/Lj*) and right (Lb*/Lk*) sides."""

    table_symmetric = YeadonTable(symmetric=True, density_set=YeadonDensitySet.DEMPSTER)
    table_symmetric.from_measurements(MALE1_MEASUREMENTS_CM)

    npt.assert_almost_equal(
        table_symmetric[YeadonSegmentName.LEFT_UPPER_ARM].mass,
        table_symmetric[YeadonSegmentName.RIGHT_UPPER_ARM].mass,
    )
    npt.assert_almost_equal(
        table_symmetric[YeadonSegmentName.LEFT_THIGH].mass,
        table_symmetric[YeadonSegmentName.RIGHT_THIGH].mass,
    )

    table_asymmetric = YeadonTable(symmetric=False, density_set=YeadonDensitySet.DEMPSTER)
    table_asymmetric.from_measurements(MALE1_MEASUREMENTS_CM)

    assert not np.isclose(
        table_asymmetric[YeadonSegmentName.LEFT_UPPER_ARM].mass,
        table_asymmetric[YeadonSegmentName.RIGHT_UPPER_ARM].mass,
    )
    assert not np.isclose(
        table_asymmetric[YeadonSegmentName.LEFT_THIGH].mass,
        table_asymmetric[YeadonSegmentName.RIGHT_THIGH].mass,
    )

    # Using symmetric=True or symmetric=False should not mutate the original measurements object
    npt.assert_almost_equal(MALE1_MEASUREMENTS_CM.La0p, 0.394)
    npt.assert_almost_equal(MALE1_MEASUREMENTS_CM.Lb0p, 0.425)


def test_yeadon_table_getitem_accepts_string_and_enum():
    table = YeadonTable(symmetric=True, density_set=YeadonDensitySet.DEMPSTER, total_mass=81.7)
    table.from_measurements(MALE1_MEASUREMENTS_CM)

    for segment_name in YeadonSegmentName:
        by_enum = table[segment_name]
        by_string = table[segment_name.value]
        npt.assert_almost_equal(by_enum.mass, by_string.mass)
        npt.assert_array_almost_equal(by_enum.center_of_mass, by_string.center_of_mass)
        npt.assert_array_almost_equal(by_enum.inertia, by_string.inertia)
