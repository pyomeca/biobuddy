import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

from biobuddy import BiomechanicalModelReal, MuscleValidator, PathPointCondition
from test_utils import create_simple_model


def test_muscle_validator_initialization():
    """Test that MuscleValidator initializes correctly"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5

    validator = MuscleValidator(model, nb_states=nb_states)

    assert validator.model == model
    assert validator.nb_states == nb_states
    assert validator.custom_ranges is None
    assert validator.states.shape == (model.nb_q, nb_states)
    npt.assert_almost_equal(validator.states[0], np.array([-10.0, -5.0, 0.0, 5.0, 10.0]))  # Translations
    npt.assert_almost_equal(
        validator.states[10], np.array([-1.570796, -0.785398, 0.0, 0.785398, 1.570796])
    )  # Rotations
    assert validator.muscle_max_force.shape == (model.nb_muscles, nb_states)
    npt.assert_almost_equal(
        validator.muscle_max_force[0],
        np.array([2047.08373459, 699.990814, 873.3046729, 1028.95791238, 35355.79749411]),
        decimal=6,
    )
    npt.assert_almost_equal(
        validator.muscle_max_force[10],
        np.array([8.06355638e-03, 3.28366501e-02, 6.13244468e02, 3.11771105e01, 1.85302109e01]),
        decimal=6,
    )
    assert validator.muscle_min_force.shape == (model.nb_muscles, nb_states)
    npt.assert_almost_equal(
        validator.muscle_min_force[0],
        np.array([1976.40387739, 165.83431579, 678.39680172, 873.00264632, 35353.03402975]),
        decimal=6,
    )
    npt.assert_almost_equal(validator.muscle_min_force[10], np.array([0, 0, 0, 0, 0]))
    assert validator.muscle_lengths.shape == (model.nb_muscles, nb_states)
    npt.assert_almost_equal(
        validator.muscle_lengths[0],
        np.array([0.16516437, 0.13038862, 0.14978456, 0.15338417, 0.20725017]),
        decimal=6,
    )
    npt.assert_almost_equal(
        validator.muscle_lengths[10],
        np.array([-0.02782064, -0.02137136, 0.03933778, 0.02474164, 0.0203703]),
        decimal=6,
    )
    assert validator.muscle_optimal_lengths.shape == (model.nb_muscles,)
    npt.assert_almost_equal(
        validator.muscle_optimal_lengths,
        np.array(
            [
                0.0976,
                0.1367,
                0.0976,
                0.2324,
                0.1385,
                0.0976,
                0.0976,
                0.1367,
                0.2324,
                0.1385,
                0.0535,
                0.201,
                0.109,
                0.52,
                0.095,
                0.089,
                0.084,
                0.06,
                0.064,
                0.05,
                0.031,
                0.098,
                0.049,
                0.0535,
                0.201,
                0.109,
                0.52,
                0.095,
                0.089,
                0.084,
                0.06,
                0.064,
                0.05,
                0.031,
                0.098,
                0.049,
                0.12,
                0.12,
                0.2238,
                0.2238,
                0.108,
                0.108,
                0.134,
                0.134,
                0.1138,
                0.1138,
                0.1157,
                0.1157,
                0.1726,
                0.098,
                0.1726,
                0.098,
                0.0628,
                0.081,
                0.0628,
                0.081,
            ]
        ),
    )
    assert validator.muscle_moment_arm.shape == (model.nb_q, model.nb_muscles, nb_states)
    npt.assert_almost_equal(
        validator.muscle_moment_arm[7, 11, :],
        np.array([0.02094623, -0.06511116, -0.02315659, -0.0280816, -0.06025606]),
        decimal=6,
    )
    npt.assert_almost_equal(
        validator.muscle_moment_arm[34, 43, :],
        np.array([-0.00555095, -0.0141655, 0.00812161, -0.01066432, -0.0067984]),
        decimal=6,
    )
    assert validator.muscle_max_torque.shape == (model.nb_q, model.nb_muscles, nb_states)
    npt.assert_almost_equal(
        validator.muscle_max_torque[7, 11, :],
        np.array([-8.55859959, 22.39304372, 1.19972076, 11.51196094, 24.47050633]),
        decimal=6,
    )
    npt.assert_almost_equal(
        validator.muscle_max_torque[34, 43, :],
        np.array([-91.2986461, 96.43727881, -17.1287718, -139.83077543, 274.85113004]),
        decimal=6,
    )
    assert validator.muscle_min_torque.shape == (model.nb_q, nb_states)
    npt.assert_almost_equal(
        validator.muscle_min_torque[7, :],
        np.array([-6.93669192e-02, 1.03022872e-13, -2.44079000e00, 2.67311044e-01, 1.85973336e-01]),
        decimal=6,
    )
    npt.assert_almost_equal(
        validator.muscle_min_torque[34, :],
        np.array([-94.68727797, 90.64623914, -15.96985543, -140.96772137, 271.91400534]),
        decimal=6,
    )


def test_muscle_validator_with_custom_ranges():
    """Test MuscleValidator with custom ranges"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5

    custom_ranges = np.array([[-0.5] * model.nb_q, [0.5] * model.nb_q])

    validator = MuscleValidator(model, nb_states=nb_states, custom_ranges=custom_ranges)

    assert validator.custom_ranges is not None
    np.testing.assert_array_equal(validator.custom_ranges, custom_ranges)
    assert validator.states.shape == (model.nb_q, nb_states)
    npt.assert_almost_equal(validator.states[0], np.array([-0.5, -0.25, 0.0, 0.25, 0.5]))  # Translations
    npt.assert_almost_equal(validator.states[10], np.array([-0.5, -0.25, 0.0, 0.25, 0.5]))  # Rotations

    # Check that the states are inside the range
    for joint_idx in range(model.nb_q):
        assert np.all(validator.states[joint_idx, :] >= custom_ranges[0][joint_idx])
        assert np.all(validator.states[joint_idx, :] <= custom_ranges[1][joint_idx])


def test_states_from_model_ranges():
    """Test states_from_model_ranges method"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5

    validator = MuscleValidator(model, nb_states=nb_states)
    states = validator.states_from_model_ranges()

    assert states.shape == (model.nb_q, nb_states)

    ranges = model.get_dof_ranges()
    for joint_idx in range(model.nb_q):
        npt.assert_almost_equal(states[joint_idx, 0], ranges[0][joint_idx])
        npt.assert_almost_equal(states[joint_idx, -1], ranges[1][joint_idx])

    # With a model that does not have ranges defined
    simple_model = create_simple_model()
    for segment in simple_model.segments:
        if segment.q_ranges is not None:
            segment.q_ranges = None
    simple_validator = MuscleValidator(simple_model, nb_states=nb_states)
    states = simple_validator.states_from_model_ranges()
    for joint_idx in range(simple_model.nb_q):
        npt.assert_almost_equal(states[joint_idx, 0], -np.pi)
        npt.assert_almost_equal(states[joint_idx, -1], np.pi)


def test_states_from_custom_ranges():
    """Test states generation with custom ranges"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5

    custom_ranges = np.array([[-1.0] * model.nb_q, [1.0] * model.nb_q])
    validator = MuscleValidator(model, nb_states=nb_states, custom_ranges=custom_ranges)

    states = validator.states

    for joint_idx in range(model.nb_q):
        np.testing.assert_almost_equal(states[joint_idx, 0], -1.0)
        np.testing.assert_almost_equal(states[joint_idx, -1], 1.0)

    # Check that an error is raised if the custom ranges shape is incorrect
    invalid_custom_ranges = np.ones((1, model.nb_q))
    with pytest.raises(
        NotImplementedError,
        match="Either all ranges or no ranges could be provided for now. Expected shape 2 x 42.If you fall on this error please contact the developers.",
    ):
        MuscleValidator(model, nb_states=nb_states, custom_ranges=invalid_custom_ranges)


def test_muscle_forces_stored_correctly():
    """Test that muscle forces are stored correctly in validator"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 15

    validator = MuscleValidator(model, nb_states=nb_states)

    assert validator.muscle_max_force.shape == (model.nb_muscles, nb_states)
    assert validator.muscle_min_force.shape == (model.nb_muscles, nb_states)

    assert np.all(validator.muscle_max_force >= validator.muscle_min_force)


def test_muscle_lengths_stored_correctly():
    """Test that muscle lengths are stored correctly"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 15

    validator = MuscleValidator(model, nb_states=nb_states)

    assert validator.muscle_lengths.shape == (model.nb_muscles, nb_states)


def test_return_optimal_lengths():
    """Test return_optimal_lengths method"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5

    validator = MuscleValidator(model, nb_states=nb_states)

    optimal_lengths = validator.return_optimal_lengths()

    assert optimal_lengths.shape == (model.nb_muscles,)
    assert np.all(optimal_lengths > 0)

    muscle_idx = 0
    for muscle_group in model.muscle_groups:
        for muscle in muscle_group.muscles:
            np.testing.assert_almost_equal(optimal_lengths[muscle_idx], muscle.optimal_length)
            muscle_idx += 1


def test_with_a_non_biomodable_model():
    """Test that MuscleValidator raises error with non-biomodable model"""
    simple_model = create_simple_model()
    simple_model.muscle_groups["parent_to_child"].muscles["muscle1"].origin_position.condition = PathPointCondition(
        dof_name="parent_transX", range_min=-2, range_max=2
    )
    with pytest.raises(
        NotImplementedError,
        match="Only biorbd is supported as a backend for now. If you need other dynamics engines, please contact the developers. The model provided is not biomodable: Muscle origin cannot be conditional..",
    ):
        MuscleValidator(simple_model)


def test_plot_force_length_structure():
    """Test the structure of the force-length plot"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5

    validator = MuscleValidator(model, nb_states=nb_states)
    figure = validator.plot_force_length()

    # Check annotations
    annotations = [a.text for a in figure.layout.annotations]
    assert "muscle_Forces" in annotations
    assert "muscle_Lengths" in annotations

    # Check traces
    for i, muscle_name in enumerate(model.muscle_names):
        base_idx = i * 4
        assert figure.data[base_idx].name == f"{muscle_name}_Max_Force"
        assert figure.data[base_idx + 1].name == f"{muscle_name}_Min_Force"
        assert figure.data[base_idx + 2].name == f"{muscle_name}_Length"
        assert figure.data[base_idx + 3].name == f"{muscle_name}_Optimal_Length"

    # Check some max force traces data
    np.testing.assert_array_almost_equal(
        figure.data[0].y, np.array([2047.08373459, 699.990814, 873.3046729, 1028.95791238, 35355.79749411])
    )
    np.testing.assert_array_almost_equal(
        figure.data[4].y, np.array([1924.07686538, 3783.97697508, 374.19781426, 5317.08515414, 228.65087896])
    )
    np.testing.assert_array_almost_equal(
        figure.data[8].y, np.array([800.2374742, 1052.59894657, 1064.27015197, 667.60001786, 1329.10366036])
    )

    # Check some min force traces data
    np.testing.assert_array_almost_equal(
        figure.data[1].y, np.array([1976.40387739, 165.83431579, 678.39680172, 873.00264632, 35353.03402975])
    )
    np.testing.assert_array_almost_equal(
        figure.data[5].y, np.array([1920.63777166, 3782.41123082, 352.58587874, 5316.02843144, 189.89998032])
    )
    np.testing.assert_array_almost_equal(
        figure.data[9].y, np.array([91.81895011, 11.69461538, 5.44299064, 244.05075529, 1214.47280551])
    )

    # Check some muscle length traces data
    np.testing.assert_array_almost_equal(
        figure.data[2].y, np.array([0.16516437, 0.13038862, 0.14978456, 0.15338417, 0.20725017])
    )
    np.testing.assert_array_almost_equal(
        figure.data[6].y, np.array([0.26043333, 0.27430547, 0.22589219, 0.28127744, 0.2134293])
    )
    np.testing.assert_array_almost_equal(
        figure.data[10].y, np.array([0.12293962, 0.10440445, 0.10116109, 0.13553692, 0.15812581])
    )

    # Check the length of the data traces
    assert len(figure.data) == model.nb_muscles * 4

    # Check the text
    assert figure.layout.title.text == "Muscular Force–Length"
    assert figure.layout.xaxis.title.text == "Range (rad)"
    assert figure.layout.xaxis2.title.text == "Range (rad)"
    assert figure.layout.yaxis.title.text == "Force (N)"
    assert figure.layout.yaxis2.title.text == "Length (m)"

    # Check the menu
    assert figure.layout.updatemenus is not None
    assert len(figure.layout.updatemenus) == 1
    buttons = figure.layout.updatemenus[0].buttons
    assert len(buttons) == model.nb_muscles

    # check the buttons labels
    buttons = figure.layout.updatemenus[0].buttons
    button_labels = [b.label for b in buttons]
    for muscle_name in model.muscle_names:
        assert muscle_name in button_labels


def test_plot_moment_arm_structure():
    """Test the structure of the moment arm plot"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5

    validator = MuscleValidator(model, nb_states=nb_states)
    figure = validator.plot_moment_arm()

    # Check annotations (subplot titles should be muscle names)
    annotations = [a.text for a in figure.layout.annotations]
    for muscle_name in model.muscle_names:
        assert muscle_name in annotations

    # Check traces - one trace per DOF per muscle
    for i, muscle_name in enumerate(model.muscle_names):
        for j, dof_name in enumerate(model.dof_names):
            trace_idx = j * model.nb_muscles + i
            assert figure.data[trace_idx].name == f"{muscle_name}_Moment_Arm"

    # Check some moment arm traces data
    np.testing.assert_array_almost_equal(
        figure.data[2078].y,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        figure.data[1456].y,
        np.array([0.00876453, -0.01154376, 0.00177781, 0.01221064, -0.0060508]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        figure.data[591].y,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        figure.data[521].y,
        np.array([-0.02761741, -0.0309185, -0.01097717, 0.01949086, 0.03205424]),
        decimal=6,
    )

    # Check the length of the data traces (nb_dof * nb_muscles)
    assert len(figure.data) == model.nb_q * model.nb_muscles

    # Check the text
    assert figure.layout.title.text == "Moment arm"

    # Check axis labels
    assert figure.layout.yaxis.title.text == "Moment arm (m)"

    # Check the menu
    assert figure.layout.updatemenus is not None
    assert len(figure.layout.updatemenus) == 1
    buttons = figure.layout.updatemenus[0].buttons
    assert len(buttons) == model.nb_q

    # Check the buttons labels
    button_labels = [b.label for b in buttons]
    for dof_name in model.dof_names:
        assert dof_name in button_labels


def test_plot_torques_structure():
    """Test the structure of the torques plot"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5

    validator = MuscleValidator(model, nb_states=nb_states)
    figure = validator.plot_torques()

    # Check annotations (subplot titles should be muscle names)
    annotations = [a.text for a in figure.layout.annotations]
    for muscle_name in model.muscle_names:
        assert muscle_name in annotations

    # Check traces - two traces (max and min) per DOF per muscle
    for i, muscle_name in enumerate(model.muscle_names):
        for j, dof_name in enumerate(model.dof_names):
            trace_idx_max = j * model.nb_muscles * 2 + i * 2
            trace_idx_min = j * model.nb_muscles * 2 + i * 2 + 1
            assert figure.data[trace_idx_max].name == f"{muscle_name}_Max_Torque"
            assert figure.data[trace_idx_min].name == f"{muscle_name}_Min_Torque"

    # Check some max torque traces data
    np.testing.assert_array_almost_equal(
        figure.data[2372].y,
        np.array([-1.35442174, -5.87193092, 0.0, 46.79111287, 1.17353167]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        figure.data[4260].y,
        np.array([0.22466436, -0.18870296, 0.68962033, 0.05989628, -0.13047899]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        figure.data[1244].y,
        np.array([11.51150145, 6.30667511, 0.96892677, -14.096754, -38.60689506]),
        decimal=6,
    )

    # Check some min torque traces data
    np.testing.assert_array_almost_equal(
        figure.data[2373].y,
        np.array([-1.35442174, -5.87193092, 0.0, 46.79111287, 1.17353167]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        figure.data[4261].y,
        np.array([0.22466436, -0.18870296, 0.68962033, 0.05989628, -0.13047899]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        figure.data[1243].y,
        np.array([11.51150145, 6.30667511, 0.96892677, -14.096754, -38.60689506]),
        decimal=6,
    )

    # Check the length of the data traces (nb_dof * nb_muscles * 2)
    assert len(figure.data) == model.nb_q * model.nb_muscles * 2

    # Check the text
    assert figure.layout.title.text == "Torque"

    # Check axis labels
    assert figure.layout.yaxis.title.text == "Torque (N.m)"

    # Check the menu
    assert figure.layout.updatemenus is not None
    assert len(figure.layout.updatemenus) == 1
    buttons = figure.layout.updatemenus[0].buttons
    assert len(buttons) == model.nb_q

    # Check the buttons labels
    button_labels = [b.label for b in buttons]
    for dof_name in model.dof_names:
        assert dof_name in button_labels


def test_muscle_validator_raises_on_invalid_model():
    """Test that MuscleValidator raises error for non-biomodable models"""
    from biobuddy.components.real.biomechanical_model_real import BiomechanicalModelReal

    no_dof_model = BiomechanicalModelReal()
    with pytest.raises(
        ValueError,
        match="Your model has no degrees of freedom. Please provide a model with at least one degree of freedom.",
    ):
        MuscleValidator(no_dof_model)

    no_muscle_model = create_simple_model()
    muscle_group_names = no_muscle_model.muscle_group_names.copy()
    for muscle_group_name in muscle_group_names:
        no_muscle_model.remove_muscle_group(muscle_group_name)
    with pytest.raises(ValueError, match="Your model has no muscles. Please provide a model with at least one muscle."):
        MuscleValidator(no_muscle_model)


def test_muscle_forces_activation_relationship():
    """Test that max force (activation=1) is greater than or equal to min force (activation=0)"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5

    validator = MuscleValidator(model, nb_states=nb_states)

    for muscle_idx in range(model.nb_muscles):
        for state_idx in range(nb_states):
            assert (
                validator.muscle_max_force[muscle_idx, state_idx] >= validator.muscle_min_force[muscle_idx, state_idx]
            )
