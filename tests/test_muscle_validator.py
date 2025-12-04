import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

from biobuddy import BiomechanicalModelReal, MuscleValidator


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
    npt.assert_almost_equal(validator.states[0], np.array([-10.,  -5.,   0.,   5.,  10.]))  # Translations
    npt.assert_almost_equal(validator.states[10], np.array([-1.570796, -0.785398,  0.      ,  0.785398,  1.570796]))  # Rotations
    assert validator.muscle_max_force.shape == (model.nb_muscles, nb_states)
    npt.assert_almost_equal(validator.muscle_max_force[0], np.array([ 2047.08373459,   699.990814  ,   873.3046729 ,  1028.95791238,
       35355.79749411]))
    npt.assert_almost_equal(validator.muscle_max_force[10], np.array([8.06355638e-03, 3.28366501e-02, 6.13244468e+02, 3.11771105e+01,
       1.85302109e+01]))
    assert validator.muscle_min_force.shape == (model.nb_muscles, nb_states)
    npt.assert_almost_equal(validator.muscle_min_force[0], np.array([ 1976.40387739,   165.83431579,   678.39680172,   873.00264632,
       35353.03402975]))
    npt.assert_almost_equal(validator.muscle_min_force[10], np.array([0, 0, 0, 0, 0]))
    assert validator.muscle_lengths.shape == (model.nb_muscles, nb_states)
    npt.assert_almost_equal(validator.muscle_lengths[0], np.array([0.16516437, 0.13038862, 0.14978456, 0.15338417, 0.20725017]))
    npt.assert_almost_equal(validator.muscle_lengths[10], np.array([-0.02782064, -0.02137136,  0.03933778,  0.02474164,  0.0203703 ]))
    assert validator.muscle_optimal_lengths.shape == (model.nb_muscles, )
    npt.assert_almost_equal(validator.muscle_optimal_lengths, np.array([0.0976, 0.1367, 0.0976, 0.2324, 0.1385, 0.0976, 0.0976, 0.1367,
       0.2324, 0.1385, 0.0535, 0.201 , 0.109 , 0.52  , 0.095 , 0.06  ,
       0.064 , 0.05  , 0.031 , 0.098 , 0.049 , 0.0535, 0.201 , 0.109 ,
       0.52  , 0.095 , 0.06  , 0.064 , 0.05  , 0.031 , 0.098 , 0.049 ,
       0.12  , 0.12  , 0.2238, 0.2238, 0.108 , 0.108 , 0.134 , 0.134 ,
       0.1138, 0.1138, 0.1157, 0.1157, 0.1726, 0.098 , 0.1726, 0.098 ,
       0.0628, 0.081 , 0.0628, 0.081 ]))
    assert validator.muscle_moment_arm.shape == (model.nb_q, model.nb_muscles, nb_states)
    npt.assert_almost_equal(validator.muscle_moment_arm[7, 11, :], np.array([ 0.01702134, -0.06367495, -0.02969015, -0.02648772, -0.04840938]))
    npt.assert_almost_equal(validator.muscle_moment_arm[34, 43, :], np.array([ 0.01169217, -0.00614974,  0.02490801,  0.01319443, -0.00754774]))
    assert validator.muscle_max_torque.shape == (model.nb_q, model.nb_muscles, nb_states)
    npt.assert_almost_equal(validator.muscle_max_torque[7, 11, :], np.array([ -6.87949845,  23.94079864, -15.20118408,   8.76100787,
        16.88174848]))
    npt.assert_almost_equal(validator.muscle_max_torque[34, 43, :], np.array([ -96.37131862,   93.93907444,  -31.5198799 , -147.62853678,
        274.2484967 ]))
    assert validator.muscle_min_torque.shape == (model.nb_q, nb_states)
    npt.assert_almost_equal(validator.muscle_min_torque[7, :], np.array([ 11.51297038,   6.30633666,   0.96889746, -13.75417984,
       -45.35209531]))
    npt.assert_almost_equal(validator.muscle_min_torque[34, :], np.array([ -94.68727368,   90.64623855,  -15.96985247, -140.96772428,
        271.91400559]))


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
    npt.assert_almost_equal(validator.states[0], np.array([-0.5 , -0.25,  0.  ,  0.25,  0.5 ]))  # Translations
    npt.assert_almost_equal(validator.states[10], np.array([-0.5 , -0.25,  0.  ,  0.25,  0.5 ]))  # Rotations

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
    with pytest.raises(NotImplementedError, match="Either all ranges or no ranges could be provided for now. Expected shape 2 x 42.If you fall on this error please contact the developers."):
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
    assert np.all(validator.muscle_lengths > 0)


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
        figure.data[0].y,
        np.array([ 2047.08373459,   699.990814,   873.3046729,  1028.95791238, 35355.79749411])
    )
    np.testing.assert_array_almost_equal(
        figure.data[4].y,
        np.array([1924.07686538, 3783.97697508,  374.19781426, 5317.08515414, 228.65087896])
    )
    np.testing.assert_array_almost_equal(
        figure.data[8].y,
        np.array([ 800.2374742 , 1052.59894657, 1064.27015197,  667.60001786, 1329.10366036])
    )

    # Check some min force traces data
    np.testing.assert_array_almost_equal(
        figure.data[1].y,
        np.array([ 1976.40387739,   165.83431579,   678.39680172,   873.00264632,
       35353.03402975])
    )
    np.testing.assert_array_almost_equal(
        figure.data[5].y,
        np.array([1920.63777166, 3782.41123082,  352.58587874, 5316.02843144,
        189.89998032])
    )
    np.testing.assert_array_almost_equal(
        figure.data[9].y,
        np.array([  91.81895011,   11.69461538,    5.44299064,  244.05075529,
       1214.47280551])
    )

    # Check some muscle length traces data
    np.testing.assert_array_almost_equal(
        figure.data[2].y,
        np.array([0.16516437, 0.13038862, 0.14978456, 0.15338417, 0.20725017])
    )
    np.testing.assert_array_almost_equal(
        figure.data[6].y,
        np.array([0.26043333, 0.27430547, 0.22589219, 0.28127744, 0.2134293 ])
    )
    np.testing.assert_array_almost_equal(
        figure.data[10].y,
        np.array([0.12293962, 0.10440445, 0.10116109, 0.13553692, 0.15812581])
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
    figure = validator.plot_moment_arms()

    # TODO

def test_plot_torques_structure():
    """Test the structure of the torques plot"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5
    
    validator = MuscleValidator(model, nb_states=nb_states)
    figure = validator.plot_torques()

    # TODO


def test_muscle_validator_raises_on_invalid_model():
    """Test that MuscleValidator raises error for non-biomodable models"""
    from biobuddy.components.real.biomechanical_model_real import BiomechanicalModelReal
    
    invalid_model = BiomechanicalModelReal()
    
    with pytest.raises(NotImplementedError):
        MuscleValidator(invalid_model)


def test_muscle_forces_activation_relationship():
    """Test that max force (activation=1) is greater than or equal to min force (activation=0)"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    nb_states = 5
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    for muscle_idx in range(model.nb_muscles):
        for state_idx in range(nb_states):
            assert validator.muscle_max_force[muscle_idx, state_idx] >= validator.muscle_min_force[muscle_idx, state_idx]
