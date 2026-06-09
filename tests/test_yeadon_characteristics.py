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


class FakeSegment:
    def __init__(self, index: int, name: str):
        self.mass = float(index + 1)
        self.rel_center_of_mass = np.array(
            [[0.01 * index], [0.02 * index], [0.03 * index]]
        )
        self.rel_inertia = np.diag(
            [0.1 * (index + 1), 0.2 * (index + 1), 0.3 * (index + 1)]
        )
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
        self.segments = [
            getattr(self, segment_name.value) for segment_name in YeadonSegmentName
        ]
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
    npt.assert_array_almost_equal(
        np.diag(pelvis.inertia)[:3], np.array([0.1, 0.2, 0.3]) * 110.0 / 66.0
    )

    right_shank_foot = table["K2"]
    npt.assert_almost_equal(right_shank_foot.mass, 11.0 * 110.0 / 66.0)
    npt.assert_array_equal(
        table.segment_origin("K2"), np.array([10.0, 10.25, 10.5, 1.0])
    )
    npt.assert_array_equal(table.segment_end("K2"), np.array([10.0, 10.25, 11.5, 1.0]))


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
    assert {spec.name for spec in YEADON_MEASUREMENT_SPECS} == set(
        YEADON_MEASUREMENT_NAMES
    )
    assert {spec.group for spec in YEADON_MEASUREMENT_SPECS} == {
        "Torso",
        "Left arm",
        "Right arm",
        "Left leg",
        "Right leg",
    }
