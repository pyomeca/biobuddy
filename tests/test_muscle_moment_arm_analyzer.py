import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from examples.muscle_moment_arm_analyzer import MuscleMomentArmAnalyzer
from biobuddy import BiomechanicalModelReal
import biorbd
from pathlib import Path

# ---------------------------
# Fixtures
# ---------------------------


@pytest.fixture
def fake_model():
    model = MagicMock()
    model.nb_q = 2
    model.nb_muscles = 3
    model.dof_names = ["q1", "q2"]
    model.muscle_names = ["m1", "m2", "m3"]

    # DOF ranges: shape (2, nb_q)
    model.get_dof_ranges.return_value = np.array([[-1.0, -2.0], [1.0, 2.0]])

    return model


@pytest.fixture
def analyzer(fake_model):
    with (
        patch(
            "examples.muscle_moment_arm_analyzer.BiomechanicalModelReal"
        ) as mock_bio_model,
        patch("examples.muscle_moment_arm_analyzer.biorbd.Model") as mock_biorbd,
    ):

        mock_bio_model().from_biomod.return_value = fake_model

        fake_biorbd = MagicMock()
        mock_biorbd.return_value = fake_biorbd

        analyzer = MuscleMomentArmAnalyzer("fake_path.bioMod")
        analyzer.model = fake_model
        analyzer.biorbd_model = fake_biorbd

        return analyzer


current_path_file = Path(__file__).parent.parent
MODEL_EXAMPLE = f"{current_path_file}/examples/models/wholebody_reference.bioMod"


@pytest.fixture
def analyzer_example():
    return MuscleMomentArmAnalyzer(MODEL_EXAMPLE)


def test_initialization(analyzer_example):

    assert isinstance(analyzer_example.model, BiomechanicalModelReal)
    assert isinstance(analyzer_example.biorbd_model, biorbd.Model)

    assert analyzer_example.model.nb_q > 0
    assert analyzer_example.model.nb_muscles > 0

    assert analyzer_example.accurate_ranges_array.shape == (
        analyzer_example.model.nb_q,
        2,
    )
    assert np.array_equal(
        analyzer_example.accurate_ranges_array,
        np.zeros((analyzer_example.model.nb_q, 2)),
    )

    assert analyzer_example.sign_lever_arm == {}


# ---------------------------------------------------------------------

# ---------------------------
# Tests : sign_lever_arm
# ---------------------------


def test_create_sign_lever_arm_user_valid(analyzer):
    signs = np.array([[1, -1, 0], [0, 1, -1]])

    analyzer.create_sign_lever_arm_user(signs)

    assert analyzer.sign_lever_arm["q1"]["m1"] == 1
    assert analyzer.sign_lever_arm["q1"]["m2"] == -1
    assert analyzer.sign_lever_arm["q1"]["m3"] == 0
    assert analyzer.sign_lever_arm["q2"]["m1"] == 0
    assert analyzer.sign_lever_arm["q2"]["m2"] == 1
    assert analyzer.sign_lever_arm["q2"]["m3"] == -1


def test_create_sign_lever_arm_user_invalid_shape(analyzer):
    signs = np.array([[1, 0]])

    with pytest.raises(ValueError):
        analyzer.create_sign_lever_arm_user(signs)


def test_create_sign_lever_arm_user_invalid_values(analyzer):
    signs = np.array([[2, 0, 1], [0, 1, -1]])

    with pytest.raises(ValueError):
        analyzer.create_sign_lever_arm_user(signs)


# ---------------------------
# Tests : joint states
# ---------------------------


def test_compute_joint_states(analyzer):
    states = analyzer.compute_joint_states(5)

    assert states.shape == (2, 5)
    assert np.all(states[0] >= -1.0)
    assert np.all(states[0] <= 1.0)


# ---------------------------
# Tests : moment arm (mocked)
# ---------------------------


def test_compute_moment_arm(analyzer):
    nb_states = 4
    states = np.zeros((2, nb_states))

    fake_jacobian = MagicMock()
    fake_jacobian.to_array.return_value = np.ones((3, 2))

    analyzer.biorbd_model.musclesLengthJacobian.return_value = fake_jacobian

    result = analyzer.compute_moment_arm(states)

    assert result.shape == (2, 3, nb_states)
    assert np.all(result == 1)


# ---------------------------
# Tests : zero finding
# ---------------------------


def test_find_zero_newton(analyzer):
    q_init = np.array([0.5, 0.0])

    def fake_jac(q):
        class Obj:
            def to_array(self_inner):
                # zero at q[0] = 0
                return np.array([[q[0], 0], [q[0], 0], [q[0], 0]])

        return Obj()

    analyzer.biorbd_model.musclesLengthJacobian.side_effect = fake_jac

    q_star = analyzer.find_zero_newton(q_init, joint_id=0, muscle_id=0)

    assert q_star is not None
    assert abs(q_star[0]) < 1e-6


# ---------------------------
# Tests : compute_moment_arm_ranges
# ---------------------------


# 1. Case: all zeros
# ---------------------------
def test_all_zero(analyzer):
    analyzer.compute_joint_states = lambda n: np.zeros((2, 5))
    analyzer.compute_moment_arm = lambda states: np.zeros((2, 3, 5))

    result = analyzer.compute_moment_arm_ranges()

    dof_ranges = analyzer.model.get_dof_ranges()

    for i, q in enumerate(["q1", "q2"]):
        for m in ["m1", "m2", "m3"]:
            assert result[q][m] == [
                {
                    "range": (dof_ranges[0, i], dof_ranges[1, i]),
                    "sign": 0,
                }
            ]


# 2. Case: constant positive sign
# ---------------------------
def test_constant_positive(analyzer):
    analyzer.compute_joint_states = lambda n: np.zeros((2, 5))
    analyzer.compute_moment_arm = lambda states: np.ones((2, 3, 5))

    def fake_jacobian(q):
        class Obj:
            def to_array(self_inner):
                return np.ones((3, 2))

        return Obj()

    analyzer.biorbd_model.musclesLengthJacobian.side_effect = fake_jacobian

    result = analyzer.compute_moment_arm_ranges()

    for q in ["q1", "q2"]:
        for m in ["m1", "m2", "m3"]:
            ranges = result[q][m]
            assert len(ranges) == 1
            assert ranges[0]["sign"] == 1


# 3. Case: sign change
# ---------------------------
def test_sign_change(analyzer):
    states = np.array([[-1, -0.5, 0.5, 1, 1], [0, 0, 0, 0, 0]])
    analyzer.compute_joint_states = lambda n: states

    R = np.zeros((2, 3, 5))
    R[0, 0, :] = [-1, -1, 1, 1, 1]

    analyzer.compute_moment_arm = lambda states: R

    analyzer.find_zero_newton = lambda q, iq, im: np.array([0.0, 0.0])

    def fake_jacobian(q):
        class Obj:
            def to_array(self_inner):
                val = q[0]
                return np.array([[val, 0], [val, 0], [val, 0]])

        return Obj()

    analyzer.biorbd_model.musclesLengthJacobian.side_effect = fake_jacobian

    result = analyzer.compute_moment_arm_ranges()

    ranges = result["q1"]["m1"]

    assert len(ranges) >= 2

    signs = {r["sign"] for r in ranges}
    assert signs == {-1, 1}


# 4. Case: tolerance threshold
# ---------------------------
def test_tolerance(analyzer):
    analyzer.compute_joint_states = lambda n: np.zeros((2, 5))

    small_values = 1e-8
    R = np.full((2, 3, 5), small_values)

    analyzer.compute_moment_arm = lambda states: R

    result = analyzer.compute_moment_arm_ranges(tol=1e-6)

    for q in ["q1", "q2"]:
        for m in ["m1", "m2", "m3"]:
            assert result[q][m][0]["sign"] == 0


# 5. Output structure validation
# ---------------------------
def test_output_structure(analyzer):
    analyzer.compute_joint_states = lambda n: np.zeros((2, 5))
    analyzer.compute_moment_arm = lambda states: np.ones((2, 3, 5))

    def fake_jacobian(q):
        class Obj:
            def to_array(self_inner):
                return np.ones((3, 2))

        return Obj()

    analyzer.biorbd_model.musclesLengthJacobian.side_effect = fake_jacobian

    result = analyzer.compute_moment_arm_ranges()

    assert isinstance(result, dict)

    for q in analyzer.model.dof_names:
        assert q in result
        for m in analyzer.model.muscle_names:
            assert m in result[q]
            assert isinstance(result[q][m], list)
            assert "range" in result[q][m][0]
            assert "sign" in result[q][m][0]


# ---------
# Tests : get
# ---------
def test_get_ranges_from_idx_q(analyzer):
    sign_dict = {
        "q1": {"m1": [{"range": (-1, 0), "sign": -1}]},
        "q2": {"m2": [{"range": (0, 1), "sign": 1}]},
    }

    result = analyzer.get_ranges_from_idx_q(sign_dict, idx_q=0)

    assert result == sign_dict["q1"]


def test_get_ranges_from_idx_q_and_m(analyzer):
    sign_dict = {
        "q1": {
            "m1": [{"range": (-1, 0), "sign": -1}],
            "m2": [{"range": (0, 1), "sign": 1}],
        }
    }

    result = analyzer.get_ranges_from_idx_q_and_m(sign_dict, idx_q=0, idx_m=1)

    assert result == sign_dict["q1"]["m2"]


# ---------------------------
# Tests : merge_ranges_joint
# ---------------------------
def test_merge_ranges_joint(analyzer):
    # mock ranges_by_joint structure
    fake_ranges = {
        "q1": {
            "m1": [{"range": (-1.0, 0.0), "sign": -1}],
            "m2": [{"range": (0.0, 1.0), "sign": 1}],
            "m3": [{"range": (-0.5, 0.5), "sign": 1}],
        }
    }

    # override method dependency
    analyzer.get_ranges_from_idx_q = lambda idx_q: fake_ranges["q1"]

    result = analyzer.merge_ranges_joint(idx_q=0)

    # expected unique sorted bounds
    expected = sorted([-1.0, 0.0, 1.0, -0.5, 0.5])

    assert result == expected


def test_merge_ranges_joint_duplicates(analyzer):
    fake_ranges = {
        "q1": {
            "m1": [{"range": (-1.0, 0.0), "sign": -1}],
            "m2": [{"range": (0.0, 1.0), "sign": 1}],
            "m3": [{"range": (-1.0, 1.0), "sign": 1}],
        }
    }

    analyzer.get_ranges_from_idx_q = lambda idx_q: fake_ranges["q1"]

    result = analyzer.merge_ranges_joint(idx_q=0)

    # duplicates must be removed
    assert result == [-1.0, 0.0, 1.0]


# ---------------------------
# Tests : merge_ranges_joint empty case
# ---------------------------
def test_merge_ranges_joint_empty(analyzer):
    analyzer.get_ranges_from_idx_q = lambda idx_q: {}

    result = analyzer.merge_ranges_joint(idx_q=0)

    assert result == []


# -------------------
# Tests : test accurate range
# -------------------


# Error: no sign_lever_arm
# ---------------------------
def test_no_sign_lever_arm(analyzer):
    analyzer.sign_lever_arm = {}

    with pytest.raises(ValueError):
        analyzer.accurate_ranges_from_true_sign()


# Basic filtering (correct case)
# ---------------------------
def test_accurate_ranges_filtering(analyzer):
    analyzer.sign_lever_arm = {
        "q1": {"m1": 1, "m2": -1, "m3": 0},
        "q2": {"m1": 1, "m2": 1, "m3": -1},
    }

    analyzer._ranges_by_joint = {
        "q1": {
            "m1": [{"range": (-1, 1), "sign": 1}, {"range": (-1, 1), "sign": -1}],
            "m2": [{"range": (-1, 1), "sign": -1}],
            "m3": [{"range": (-1, 1), "sign": 0}],
        },
        "q2": {
            "m1": [{"range": (-1, 1), "sign": 1}],
            "m2": [{"range": (-1, 1), "sign": 1}],
            "m3": [{"range": (-1, 1), "sign": -1}],
        },
    }

    analyzer.accurate_ranges_from_true_sign()

    result = analyzer.accurate_ranges_by_joint

    assert result["q1"]["m1"] == [{"range": (-1, 1), "sign": 1}]
    assert result["q1"]["m2"] == [{"range": (-1, 1), "sign": -1}]
    assert result["q1"]["m3"] == [{"range": (-1, 1), "sign": 0}]


# Missing expected sign
# ---------------------------
def test_missing_expected_sign(analyzer, capsys):
    analyzer.sign_lever_arm = {
        "q1": {"m1": 1, "m2": 1, "m3": 1},
        "q2": {"m1": 1, "m2": 1, "m3": 1},
    }

    analyzer._ranges_by_joint = {
        "q1": {
            "m1": [{"range": (-1, 1), "sign": -1}],  # mismatch
            "m2": [{"range": (-1, 1), "sign": 1}],
            "m3": [{"range": (-1, 1), "sign": 1}],
        },
        "q2": {
            "m1": [{"range": (-1, 1), "sign": 1}],
            "m2": [{"range": (-1, 1), "sign": 1}],
            "m3": [{"range": (-1, 1), "sign": 1}],
        },
    }

    analyzer.accurate_ranges_from_true_sign()

    with pytest.warns(UserWarning):
        analyzer.accurate_ranges_from_true_sign()


# -------------------
# Tests : compare ranges
# -------------------
def test_compare_ranges_no_difference(analyzer, capsys):
    analyzer.sign_lever_arm = {
        "q1": {"m1": 1},
    }

    accurate_ranges = {
        "q1": {
            "m1": [{"range": (-1, 1), "sign": 1}],
        }
    }

    diff = analyzer.compare_ranges_and_user_sign(accurate_ranges)

    captured = capsys.readouterr()

    assert diff is False
    assert "Correct" in captured.out


# compare_ranges: mismatch sign
# ---------------------------
def test_compare_ranges_sign_mismatch(analyzer):
    analyzer.sign_lever_arm = {
        "q1": {"m1": 1},
    }

    accurate_ranges = {
        "q1": {
            "m1": [{"range": (-1, 1), "sign": -1}],
        }
    }

    with pytest.warns(UserWarning, match="Sign mismatch"):
        diff = analyzer.compare_ranges_and_user_sign(accurate_ranges)

    assert diff is True


# compare_ranges: missing muscle
# ---------------------------
def test_compare_ranges_missing_muscle(analyzer):
    analyzer.sign_lever_arm = {
        "q1": {"m1": 1},  # expected_sign != 0 → should trigger warning
    }

    accurate_ranges = {"q1": {}}  # m1 is missing

    with pytest.warns(UserWarning, match="m1 missing in accurate_ranges"):
        diff = analyzer.compare_ranges_and_user_sign(accurate_ranges)

    assert diff is True


# compare_ranges: missing dof
# ---------------------------
def test_compare_ranges_missing_dof(analyzer):
    analyzer.sign_lever_arm = {
        "q1": {"m1": 1},
    }

    accurate_ranges = {}

    with pytest.raises(ValueError, match="not found"):
        analyzer.compare_ranges_and_user_sign(accurate_ranges)


# ---------------------------
# Tests : create_accurate_rom
# ---------------------------
def test_create_accurate_rom_basic(analyzer):
    analyzer.model.get_dof_ranges.return_value = np.array([[-1.0, -2.0], [1.0, 2.0]])

    analyzer.model.dof_names = ["q1", "q2"]
    analyzer.model.muscle_names = ["m1"]

    analyzer.accurate_ranges_by_joint = {
        "q1": {
            "m1": [{"range": (-0.5, 0.8), "sign": 1}],
        },
        "q2": {
            "m1": [{"range": (-1.5, 1.5), "sign": 1}],
        },
    }

    result = analyzer.create_accurate_rom()

    assert result.shape == (2, 2)

    # q1: max lower bound, min upper bound
    assert result[0, 0] == -0.5
    assert result[0, 1] == 0.8

    # q2
    assert result[1, 0] == -1.5
    assert result[1, 1] == 1.5


def test_create_accurate_rom_triggers_computation(analyzer):
    analyzer.accurate_ranges_by_joint = {}

    analyzer.sign_lever_arm = {
        "q1": {"m1": 1},
        "q2": {"m1": 1},
    }

    analyzer.model.dof_names = ["q1", "q2"]
    analyzer.model.muscle_names = ["m1"]

    analyzer.accurate_ranges_from_true_sign = lambda: None

    analyzer.model.get_dof_ranges.return_value = np.array([[-1.0, -2.0], [1.0, 2.0]])

    analyzer.accurate_ranges_by_joint = {
        "q1": {"m1": [{"range": (-0.5, 0.5), "sign": 1}]},
        "q2": {"m1": [{"range": (-0.5, 0.5), "sign": 1}]},
    }

    result = analyzer.create_accurate_rom()

    assert result is not None
    assert result.shape == (2, 2)


# --------------------
# Tests : get_correct_part_mvt
# --------------------


# get_correct_part_mvt shape error
# ---------------------------
def test_get_correct_part_mvt_shape_error(analyzer):
    q = np.zeros((1, 10))  # wrong nb_q

    with pytest.raises(ValueError, match="Incorrect shape"):
        analyzer.get_correct_part_mvt(q)


# get_correct_part_mvt all correct
# ---------------------------
def test_get_correct_part_mvt_all_correct(analyzer):
    analyzer.model.nb_q = 2

    analyzer.accurate_ranges_array = np.array(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    q = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.0, 0.1, 0.2],
        ]
    )

    correct_idx, incorrect_idx, correct_q, incorrect_q = analyzer.get_correct_part_mvt(
        q
    )

    assert len(correct_idx) == 1
    assert len(incorrect_idx) == 0


# get_correct_part_mvt all incorrect
# ---------------------------
def test_get_correct_part_mvt_all_incorrect(analyzer):
    analyzer.model.nb_q = 2

    analyzer.accurate_ranges_array = np.array(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    q = np.array(
        [
            [2.0, 2.0],
            [2.0, 2.0],
        ]
    )

    correct_idx, incorrect_idx, correct_q, incorrect_q = analyzer.get_correct_part_mvt(
        q
    )

    assert len(correct_idx) == 0
    assert len(incorrect_idx) == 1


# get_correct_part_mvt auto-call ROM
# ---------------------------
def test_get_correct_part_mvt_auto_rom(analyzer):
    analyzer.model.nb_q = 2

    analyzer.accurate_ranges_array = np.zeros((2, 2))

    analyzer.create_accurate_rom = lambda: np.array(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    q = np.array(
        [
            [0.0, 0.5],
            [0.0, 0.5],
        ]
    )

    result = analyzer.get_correct_part_mvt(q)

    assert len(result) == 4
