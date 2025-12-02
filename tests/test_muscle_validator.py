import numpy as np
import pytest
from pathlib import Path
import plotly.graph_objects as go

from biobuddy import BiomechanicalModelReal
from biobuddy.validation.validate_muscles import MuscleValidator


def get_test_model():
    """Helper function to load a test model"""
    current_path_file = Path(__file__).parent.parent
    model_path = f"{current_path_file}/examples/models/wholebody_reference.bioMod"
    model = BiomechanicalModelReal().from_biomod(model_path)
    return model


def test_muscle_validator_initialization():
    """Test that MuscleValidator initializes correctly"""
    model = get_test_model()
    nb_states = 50
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    assert validator.model == model
    assert validator.nb_states == nb_states
    assert validator.custom_ranges is None
    assert validator.states.shape[1] == nb_states
    assert validator.muscle_max_force.shape[0] == model.nb_muscles
    assert validator.muscle_min_force.shape[0] == model.nb_muscles
    assert validator.muscle_lengths.shape[0] == model.nb_muscles
    assert validator.muscle_optimal_lengths.shape[0] == model.nb_muscles
    assert validator.muscle_moment_arm.shape[0] == model.nb_q
    assert validator.muscle_moment_arm.shape[1] == model.nb_muscles
    assert validator.muscle_max_torque.shape[0] == model.nb_q
    assert validator.muscle_min_torque.shape[0] == model.nb_q


def test_muscle_validator_with_custom_ranges():
    """Test MuscleValidator with custom ranges"""
    model = get_test_model()
    nb_states = 30
    
    custom_ranges = np.array([[-0.5] * model.nb_q, [0.5] * model.nb_q])
    
    validator = MuscleValidator(model, nb_states=nb_states, custom_ranges=custom_ranges)
    
    assert validator.custom_ranges is not None
    np.testing.assert_array_equal(validator.custom_ranges, custom_ranges)
    assert validator.states.shape == (model.nb_q, nb_states)


def test_states_from_model_ranges():
    """Test states_from_model_ranges method"""
    model = get_test_model()
    nb_states = 50
    
    validator = MuscleValidator(model, nb_states=nb_states)
    states = validator.states_from_model_ranges()
    
    assert states.shape == (model.nb_q, nb_states)
    
    ranges = model.get_dof_ranges()
    if ranges.size == 0:
        ranges = np.array([[-np.pi] * model.nb_q, [np.pi] * model.nb_q])
    
    for joint_idx in range(model.nb_q):
        np.testing.assert_almost_equal(states[joint_idx, 0], ranges[0][joint_idx])
        np.testing.assert_almost_equal(states[joint_idx, -1], ranges[1][joint_idx])


def test_states_from_custom_ranges():
    """Test states generation with custom ranges"""
    model = get_test_model()
    nb_states = 25
    
    custom_ranges = np.array([[-1.0] * model.nb_q, [1.0] * model.nb_q])
    validator = MuscleValidator(model, nb_states=nb_states, custom_ranges=custom_ranges)
    
    states = validator.states
    
    for joint_idx in range(model.nb_q):
        np.testing.assert_almost_equal(states[joint_idx, 0], -1.0)
        np.testing.assert_almost_equal(states[joint_idx, -1], 1.0)


def test_compute_muscle_forces():
    """Test compute_muscle_forces method"""
    model = get_test_model()
    nb_states = 20
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    max_force, min_force = validator.compute_muscle_forces()
    
    assert max_force.shape == (model.nb_muscles, nb_states)
    assert min_force.shape == (model.nb_muscles, nb_states)
    
    assert np.all(max_force >= min_force)
    assert np.all(min_force >= 0)


def test_muscle_forces_stored_correctly():
    """Test that muscle forces are stored correctly in validator"""
    model = get_test_model()
    nb_states = 15
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    assert validator.muscle_max_force.shape == (model.nb_muscles, nb_states)
    assert validator.muscle_min_force.shape == (model.nb_muscles, nb_states)
    
    assert np.all(validator.muscle_max_force >= validator.muscle_min_force)


def test_compute_muscle_lengths():
    """Test compute_muscle_lengths method"""
    model = get_test_model()
    nb_states = 20
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    muscle_lengths = validator.compute_muscle_lengths()
    
    assert muscle_lengths.shape == (model.nb_muscles, nb_states)
    assert np.all(muscle_lengths > 0)


def test_muscle_lengths_stored_correctly():
    """Test that muscle lengths are stored correctly"""
    model = get_test_model()
    nb_states = 15
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    assert validator.muscle_lengths.shape == (model.nb_muscles, nb_states)
    assert np.all(validator.muscle_lengths > 0)


def test_return_optimal_lengths():
    """Test return_optimal_lengths method"""
    model = get_test_model()
    nb_states = 20
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    optimal_lengths = validator.return_optimal_lengths()
    
    assert optimal_lengths.shape == (model.nb_muscles,)
    assert np.all(optimal_lengths > 0)
    
    muscle_idx = 0
    for muscle_group in model.muscle_groups:
        for muscle in muscle_group.muscles:
            np.testing.assert_almost_equal(optimal_lengths[muscle_idx], muscle.optimal_length)
            muscle_idx += 1


def test_compute_moment_arm():
    """Test compute_moment_arm method"""
    model = get_test_model()
    nb_states = 20
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    moment_arm = validator.compute_moment_arm()
    
    assert moment_arm.shape == (model.nb_q, model.nb_muscles, nb_states)


def test_moment_arm_stored_correctly():
    """Test that moment arms are stored correctly"""
    model = get_test_model()
    nb_states = 15
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    assert validator.muscle_moment_arm.shape == (model.nb_q, model.nb_muscles, nb_states)


def test_compute_torques():
    """Test compute_torques method"""
    model = get_test_model()
    nb_states = 20
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    max_torques, min_torques = validator.compute_torques()
    
    assert max_torques.shape == (model.nb_q, model.nb_muscles, nb_states)
    assert min_torques.shape == (model.nb_q, nb_states)


def test_torques_stored_correctly():
    """Test that torques are stored correctly"""
    model = get_test_model()
    nb_states = 15
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    assert validator.muscle_max_torque.shape == (model.nb_q, model.nb_muscles, nb_states)
    assert validator.muscle_min_torque.shape == (model.nb_q, nb_states)


def test_plot_force_length_creates_figure():
    """Test that plot_force_length creates a valid plotly figure"""
    model = get_test_model()
    nb_states = 10
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    validator.plot_force_length()


def test_plot_force_length_structure():
    """Test the structure of the force-length plot"""
    model = get_test_model()
    nb_states = 10
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    import plotly.io as pio
    original_renderer = pio.renderers.default
    pio.renderers.default = None
    
    try:
        from unittest.mock import patch
        with patch('plotly.graph_objects.Figure.show'):
            validator.plot_force_length()
    finally:
        pio.renderers.default = original_renderer


def test_plot_moment_arm_creates_figure():
    """Test that plot_moment_arm creates a valid plotly figure"""
    model = get_test_model()
    nb_states = 10
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    validator.plot_moment_arm()


def test_plot_moment_arm_structure():
    """Test the structure of the moment arm plot"""
    model = get_test_model()
    nb_states = 10
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    import plotly.io as pio
    original_renderer = pio.renderers.default
    pio.renderers.default = None
    
    try:
        from unittest.mock import patch
        with patch('plotly.graph_objects.Figure.show'):
            validator.plot_moment_arm()
    finally:
        pio.renderers.default = original_renderer


def test_plot_torques_creates_figure():
    """Test that plot_torques creates a valid plotly figure"""
    model = get_test_model()
    nb_states = 10
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    validator.plot_torques()


def test_plot_torques_structure():
    """Test the structure of the torques plot"""
    model = get_test_model()
    nb_states = 10
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    import plotly.io as pio
    original_renderer = pio.renderers.default
    pio.renderers.default = None
    
    try:
        from unittest.mock import patch
        with patch('plotly.graph_objects.Figure.show'):
            validator.plot_torques()
    finally:
        pio.renderers.default = original_renderer


def test_muscle_validator_raises_on_invalid_model():
    """Test that MuscleValidator raises error for non-biomodable models"""
    from biobuddy.components.real.biomechanical_model_real import BiomechanicalModelReal
    
    invalid_model = BiomechanicalModelReal()
    
    with pytest.raises(NotImplementedError):
        MuscleValidator(invalid_model)


def test_consistency_between_methods():
    """Test consistency between different computed values"""
    model = get_test_model()
    nb_states = 20
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    assert validator.muscle_max_force.shape[1] == validator.muscle_lengths.shape[1]
    assert validator.muscle_max_force.shape[1] == validator.muscle_moment_arm.shape[2]
    assert validator.muscle_max_force.shape[1] == validator.muscle_max_torque.shape[2]
    
    assert validator.muscle_max_force.shape[0] == validator.muscle_lengths.shape[0]
    assert validator.muscle_max_force.shape[0] == validator.muscle_optimal_lengths.shape[0]


def test_states_are_within_ranges():
    """Test that generated states are within the specified ranges"""
    model = get_test_model()
    nb_states = 30
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    ranges = model.get_dof_ranges()
    if ranges.size == 0:
        ranges = np.array([[-np.pi] * model.nb_q, [np.pi] * model.nb_q])
    
    for joint_idx in range(model.nb_q):
        assert np.all(validator.states[joint_idx, :] >= ranges[0][joint_idx])
        assert np.all(validator.states[joint_idx, :] <= ranges[1][joint_idx])


def test_muscle_forces_activation_relationship():
    """Test that max force (activation=1) is greater than or equal to min force (activation=0)"""
    model = get_test_model()
    nb_states = 20
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    for muscle_idx in range(model.nb_muscles):
        for state_idx in range(nb_states):
            assert validator.muscle_max_force[muscle_idx, state_idx] >= validator.muscle_min_force[muscle_idx, state_idx]


def test_optimal_length_count_matches_muscles():
    """Test that the number of optimal lengths matches the number of muscles"""
    model = get_test_model()
    nb_states = 20
    
    validator = MuscleValidator(model, nb_states=nb_states)
    
    total_muscles = sum(len(muscle_group.muscles) for muscle_group in model.muscle_groups)
    
    assert validator.muscle_optimal_lengths.shape[0] == total_muscles
    assert validator.muscle_optimal_lengths.shape[0] == model.nb_muscles
