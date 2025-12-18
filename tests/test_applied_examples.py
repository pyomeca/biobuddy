"""
Tests for the examples in the folder examples/applied_examples
"""

import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import C3dData


def test_plug_in_gait_example():
    from examples.applied_examples.plugin_gait import PlugInGait

    # Create the generic model
    plugin_gait_model = PlugInGait(
        body_mass=70.0,
        shoulder_offset=0.1,
        elbow_width=0.15,
        wrist_width=0.02,
        hand_thickness=0.01,
        leg_length={"R": 1.0, "L": 1.0},
        ankle_width=0.1,
        include_upper_body=True,
    )
    plugin_gait_model.build_plug_in_gait()

    # Create the real model
    static_trial_path = "../examples/data/static_plugin_gait.c3d"  # @pariterre: add anonymous data here
    static_data = C3dData(static_trial_path)
    biomechanical_model_real = plugin_gait_model.to_real(static_data)

    # Test values ---------------

    # @pariterre: Change the test values
    # Total mass
    npt.assert_almost_equal(biomechanical_model_real.mass, 70.0, decimal=4)

    value_to_change = 0.0  # Jut for now
    assert biomechanical_model_real.nb_q == value_to_change
    assert biomechanical_model_real.nb_markers == value_to_change
    assert biomechanical_model_real.nb_segments == value_to_change
    assert biomechanical_model_real.nb_contacts == value_to_change
    assert biomechanical_model_real.nb_muscles == value_to_change

    nb_q = biomechanical_model_real.nb_q
    q_zeros = np.zeros((nb_q,))

    # Joint coordinate system in global
    jcs = biomechanical_model_real.forward_kinematics(q_zeros)
    npt.assert_almost_equal(jcs["LFemur"].rt_matrix, np.array([[value_to_change]]), decimal=5)
    npt.assert_almost_equal(jcs["RRadius"].rt_matrix, np.array([[value_to_change]]), decimal=5)

    # Markers position in global
    markers = biomechanical_model_real.markers_in_global(q_zeros)
    npt.assert_almost_equal(
        markers[:3, 0],
        np.array([value_to_change]),
        decimal=5,
    )
    npt.assert_almost_equal(
        markers[:3, 10],
        np.array([value_to_change]),
        decimal=5,
    )

    # CoM position in global
    com = biomechanical_model_real.total_com_in_global(q_zeros)
    npt.assert_almost_equal(com[:3], np.array([value_to_change]))

    # Segment CoM position in global
    com_1 = biomechanical_model_real.segment_com_in_global("LHand", q_zeros)
    npt.assert_almost_equal(com_1[:3], np.array([value_to_change]))
    com_2 = biomechanical_model_real.segment_com_in_global("Pelvis", q_zeros)
    npt.assert_almost_equal(com_2[:3], np.array([value_to_change]))

    # Segment inertia
    inertia_1 = biomechanical_model_real.segments["RHumerus"].inertia_parameters.inertia
    npt.assert_almost_equal(inertia_1, np.array([value_to_change]), decimal=5)
    inertia_2 = biomechanical_model_real.segments["Thorax"].inertia_parameters.inertia
    npt.assert_almost_equal(inertia_2, np.array([value_to_change]), decimal=5)
