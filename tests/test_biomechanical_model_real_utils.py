import os
import numpy as np
import numpy.testing as npt

import biorbd

from biobuddy import BiomechanicalModelReal, MuscleType, MuscleStateType
from biobuddy.components.real.biomechanical_model_real_utils import markers_in_global, forward_kinematics


def test_biomechanics_model_real_utils_functions():


    # For ortho_norm_basis
    np.random.seed(42)

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    osim_file_path = parent_path + "/examples/models/wholebody.osim"
    biorbd_file_path = osim_file_path.replace(".osim", ".bioMod")

    # Define models
    model = BiomechanicalModelReal.from_osim(
        filepath=osim_file_path,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
    )
    model.to_biomod(biorbd_file_path)
    model_biorbd = biorbd.Model(biorbd_file_path)

    nb_q = model.nb_q
    assert nb_q == 42
    nb_markers = model.nb_markers
    assert nb_markers == 49
    nb_segments = model.nb_segments
    assert nb_segments == 200

    q_random = np.random.rand(nb_q)

    # Forward kinematics
    jcs_biobuddy = forward_kinematics(model, q_random)
    for i_segment in range(nb_segments):
        jcs_biorbd = model_biorbd.globalJCS(q_random, i_segment).to_array()
        npt.assert_array_almost_equal(
            jcs_biobuddy[model.segments[i_segment].name],
            jcs_biorbd,
            decimal=5,
        )

    # Markers position in global
    markers_biobuddy = markers_in_global(model, q_random)
    for i_marker in range(nb_markers):
        markers_biorbd = model_biorbd.markers(q_random)[i_marker].to_array()
        npt.assert_array_almost_equal(
            markers_biobuddy[:3, i_marker].reshape(3, ),
            markers_biorbd,
            decimal=4
        )
