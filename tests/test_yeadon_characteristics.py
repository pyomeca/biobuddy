import sys
import types

import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import (
    YEADON_MEASUREMENT_NAMES,
    YEADON_MEASUREMENT_SPECS,
    YeadonDensitySet,
    YeadonSegmentName,
    YeadonTable,
)

MALE1_MEASUREMENTS_CM = {
    "Ls1L": 4.9,
    "Ls2L": 18.8,
    "Ls3L": 33.4,
    "Ls4L": 47.2,
    "Ls5L": 50.2,
    "Ls6L": 14.9,
    "Ls7L": 19.0,
    "Ls8L": 29.2,
    "Ls0p": 86.9,
    "Ls1p": 75.0,
    "Ls2p": 73.9,
    "Ls3p": 80.6,
    "Ls5p": 37.5,
    "Ls6p": 49.8,
    "Ls7p": 57.0,
    "Ls0w": 32.4,
    "Ls1w": 27.4,
    "Ls2w": 25.1,
    "Ls3w": 28.0,
    "Ls4w": 30.3,
    "Ls4d": 15.1,
    "La2L": 26.1,
    "La3L": 33.0,
    "La4L": 50.8,
    "La5L": 3.0,
    "La6L": 8.8,
    "La7L": 17.7,
    "La0p": 28.5,
    "La1p": 24.2,
    "La2p": 23.0,
    "La3p": 23.5,
    "La4p": 15.4,
    "La5p": 21.0,
    "La6p": 21.5,
    "La7p": 10.7,
    "La4w": 5.0,
    "La5w": 7.3,
    "La6w": 9.8,
    "La7w": 4.9,
    "Lb2L": 26.6,
    "Lb3L": 32.4,
    "Lb4L": 51.9,
    "Lb5L": 1.5,
    "Lb6L": 7.2,
    "Lb7L": 16.6,
    "Lb0p": 29.0,
    "Lb1p": 24.2,
    "Lb2p": 22.7,
    "Lb3p": 23.4,
    "Lb4p": 15.2,
    "Lb5p": 20.9,
    "Lb6p": 22.4,
    "Lb7p": 11.0,
    "Lb4w": 5.2,
    "Lb5w": 7.2,
    "Lb6w": 9.6,
    "Lb7w": 5.0,
    "Lj1L": 10.1,
    "Lj3L": 42.6,
    "Lj4L": 56.5,
    "Lj5L": 81.6,
    "Lj6L": 2.5,
    "Lj8L": 14.9,
    "Lj9L": 21.0,
    "Lj1p": 52.0,
    "Lj2p": 48.9,
    "Lj3p": 35.1,
    "Lj4p": 35.2,
    "Lj5p": 25.5,
    "Lj6p": 30.2,
    "Lj7p": 23.3,
    "Lj8p": 22.8,
    "Lj9p": 18.5,
    "Lj8w": 8.7,
    "Lj9w": 8.2,
    "Lj6d": 11.4,
    "Lk1L": 9.7,
    "Lk3L": 42.2,
    "Lk4L": 55.8,
    "Lk5L": 80.6,
    "Lk6L": 2.5,
    "Lk8L": 14.5,
    "Lk9L": 20.8,
    "Lk1p": 55.0,
    "Lk2p": 50.0,
    "Lk3p": 34.8,
    "Lk4p": 34.9,
    "Lk5p": 23.0,
    "Lk6p": 31.0,
    "Lk7p": 23.7,
    "Lk8p": 22.2,
    "Lk9p": 19.5,
    "Lk8w": 8.6,
    "Lk9w": 8.3,
    "Lk6d": 11.1,
}


class FakeSegment:
    def __init__(self, index: int, name: str):
        self.mass = float(index + 1)
        self.rel_center_of_mass = np.array([[0.01 * index], [0.02 * index], [0.03 * index]])
        self.rel_inertia = np.diag([0.1 * (index + 1), 0.2 * (index + 1), 0.3 * (index + 1)])
        self.pos = np.array([[index], [index + 0.25], [index + 0.5]], dtype=float)
        self.end_pos = self.pos + np.array([[0.0], [0.0], [1.0]])
        self.rot_mat = np.eye(3)
        self.label = name


class FakeHuman:
    instances = []

    def __init__(self, measurements, CFG=None, symmetric=True, density_set="Dempster"):
        self.measurements = measurements
        self.CFG = CFG
        self.symmetric = symmetric
        self.density_set = density_set
        self.scale_calls = []
        for index, segment_name in enumerate(YeadonSegmentName):
            setattr(self, segment_name.value, FakeSegment(index, segment_name.value))
        self._update_whole_body_properties()
        self.instances.append(self)

    def scale_human_by_mass(self, total_mass):
        self.scale_calls.append(total_mass)
        ratio = total_mass / self.mass
        for segment_name in YeadonSegmentName:
            segment = getattr(self, segment_name.value)
            segment.mass *= ratio
            segment.rel_inertia *= ratio
        self._update_whole_body_properties()

    def _update_whole_body_properties(self):
        self.segments = [getattr(self, segment_name.value) for segment_name in YeadonSegmentName]
        self.mass = sum(segment.mass for segment in self.segments)
        self.center_of_mass = np.zeros((3, 1))
        self.inertia = np.eye(3)


@pytest.fixture()
def fake_yeadon(monkeypatch):
    module = types.SimpleNamespace(Human=FakeHuman)
    monkeypatch.setitem(sys.modules, "yeadon", module)
    FakeHuman.instances = []
    return module


def test_yeadon_table_adapts_segments_to_biobuddy_inertia(fake_yeadon):
    measurements = {name: 1.0 for name in YEADON_MEASUREMENT_NAMES}
    table = YeadonTable(
        measurements,
        configuration={"somersault": 0.0},
        symmetric=False,
        density_set=YeadonDensitySet.CLAUSER,
        total_mass=110.0,
    )

    assert FakeHuman.instances[0].measurements == measurements
    assert FakeHuman.instances[0].CFG == {"somersault": 0.0}
    assert FakeHuman.instances[0].symmetric is False
    assert FakeHuman.instances[0].density_set == "Clauser"
    assert FakeHuman.instances[0].scale_calls == [110.0]
    npt.assert_almost_equal(table.mass, 110.0)

    pelvis = table[YeadonSegmentName.PELVIS]
    npt.assert_almost_equal(pelvis.mass, 110.0 / 66.0)
    npt.assert_array_equal(pelvis.center_of_mass[:3, 0], np.array([0.0, 0.0, 0.0]))
    npt.assert_array_almost_equal(np.diag(pelvis.inertia)[:3], np.array([0.1, 0.2, 0.3]) * 110.0 / 66.0)

    right_shank_foot = table["K2"]
    npt.assert_almost_equal(right_shank_foot.mass, 11.0 * 110.0 / 66.0)
    npt.assert_array_equal(table.segment_origin("K2"), np.array([10.0, 10.25, 10.5, 1.0]))
    npt.assert_array_equal(table.segment_end("K2"), np.array([10.0, 10.25, 11.5, 1.0]))


def test_yeadon_table_from_measurement_matches_de_leva_style(fake_yeadon):
    measurements = {name: 1.0 for name in YEADON_MEASUREMENT_NAMES}
    table = YeadonTable()

    table.from_measurement(
        measurements,
        configuration={"somersault": 0.0},
        symmetric=False,
        density_set=YeadonDensitySet.CHANDLER,
        total_mass=90.0,
    )

    assert FakeHuman.instances[0].measurements == measurements
    assert FakeHuman.instances[0].CFG == {"somersault": 0.0}
    assert FakeHuman.instances[0].symmetric is False
    assert FakeHuman.instances[0].density_set == "Chandler"
    assert FakeHuman.instances[0].scale_calls == [90.0]
    npt.assert_almost_equal(table.mass, 90.0)


def test_yeadon_table_from_file_forwards_measurement_path(fake_yeadon, tmp_path):
    filepath = tmp_path / "measurements.txt"
    filepath.write_text("measurementconversionfactor: 1\n")
    table = YeadonTable()

    table.from_file(filepath, total_mass=80.0)

    assert FakeHuman.instances[0].measurements == str(filepath)
    assert FakeHuman.instances[0].scale_calls == [80.0]


def test_yeadon_table_to_file_exports_yeadon_measurements(fake_yeadon, tmp_path):
    measurements = {name: index + 0.25 for index, name in enumerate(YEADON_MEASUREMENT_NAMES)}
    table = YeadonTable(measurements, total_mass=75.0)
    filepath = tmp_path / "exported_measurements.txt"

    table.to_file(filepath)

    lines = filepath.read_text().splitlines()
    assert lines[0] == "Ls1L: 0.25"
    assert lines[1] == "Ls2L: 1.25"
    assert "totalmass: 75" in lines
    assert lines[-1] == "measurementconversionfactor: 1"
    assert len([line for line in lines if line.split(":")[0] in YEADON_MEASUREMENT_NAMES]) == len(
        YEADON_MEASUREMENT_NAMES
    )


def test_yeadon_table_creates_simple_model(fake_yeadon):
    table = YeadonTable({name: 1.0 for name in YEADON_MEASUREMENT_NAMES})

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


def test_yeadon_measurement_specs_cover_all_measurements():
    assert len(YEADON_MEASUREMENT_NAMES) == 95
    assert len(YEADON_MEASUREMENT_SPECS) == len(YEADON_MEASUREMENT_NAMES)
    assert {spec.name for spec in YEADON_MEASUREMENT_SPECS} == set(YEADON_MEASUREMENT_NAMES)
    assert {spec.group for spec in YEADON_MEASUREMENT_SPECS} == {
        "Torso",
        "Left arm",
        "Right arm",
        "Left leg",
        "Right leg",
    }


def test_yeadon_table_matches_real_yeadon_sample():
    pytest.importorskip("yeadon")
    measurements = {name: value * 0.01 for name, value in MALE1_MEASUREMENTS_CM.items()}

    table = YeadonTable(measurements)

    npt.assert_almost_equal(table.mass, 58.200488588422544)
    npt.assert_array_almost_equal(table.inertia, np.diag([9.63093850, 9.99497872, 5.45117742e-01]))
    npt.assert_array_almost_equal(table.center_of_mass, np.array([[0.0], [0.0], [1.19967938e-02]]))
    npt.assert_almost_equal(table[YeadonSegmentName.PELVIS].mass, table.human.P.mass)
    npt.assert_array_almost_equal(table[YeadonSegmentName.PELVIS].center_of_mass[:3], table.human.P.rel_center_of_mass)
    npt.assert_array_almost_equal(table[YeadonSegmentName.PELVIS].inertia[:3, :3], table.human.P.rel_inertia)
