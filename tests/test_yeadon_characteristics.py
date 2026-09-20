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
    # ... other positions


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

    # All other inertia characteristics by segment
    # Pelvis
    pelvis_from_measurements = table_from_measurements[YeadonSegmentName.PELVIS]
    npt.assert_almost_equal(pelvis_from_measurements.mass, 12.924675966035618)
    npt.assert_array_almost_equal(pelvis_from_measurements.center_of_mass[:3, 0], np.array([0., 0., 0.09628887]), decimal=6)
    npt.assert_array_almost_equal(np.diag(pelvis_from_measurements.inertia)[:3], np.array([0.08072403, 0.12168453, 0.13281317]), decimal=6)
    pelvis_from_file = table_from_file[YeadonSegmentName.PELVIS]
    npt.assert_almost_equal(pelvis_from_file.mass, 12.924675966035618)
    npt.assert_array_almost_equal(pelvis_from_file.center_of_mass[:3, 0], np.array([0., 0., 0.09628887]), decimal=6)
    npt.assert_array_almost_equal(np.diag(pelvis_from_file.inertia)[:3], np.array([0.08072403, 0.12168453, 0.13281317]), decimal=6)

    # Claude: Please add others here

    # Test joint positions
    npt.assert_almost_equal(table_from_measurements.pelvis_position, np.array([0., 0., 0., 1.]), decimal=6)
    npt.assert_almost_equal(table_from_file.pelvis_position, np.array([0., 0., 0., 1.]), decimal=6)
    # CLaude: please add all the others



def test_yeadon_table_to_file_exports_yeadon_measurements():

    # Create table from file
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = parent_path + "/examples/models"
    measures_filepath = f"{root_path}/yeadon_measures.txt"
    table_from_file = YeadonTable(
        symmetric=True,
        density_set=YeadonDensitySet.DEMPSTER,
    )
    table_from_file.from_file(measures_filepath)

    # Export to file
    new_measures_filepath = f"{root_path}/yeadon_measures_export.txt"
    table_from_file.to_file(new_measures_filepath)

    # Claude: Please read both files and make sure they match line by line

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
    assert model.segments["A2"].parent_name == "A1"
    assert model.segments["K2"].parent_name == "K1"
    npt.assert_almost_equal(model.mass, table.mass)

    # Claude: please add any other things to test in the simple model
