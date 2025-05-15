import os
import numpy as np
import numpy.testing as npt

import biorbd

from biobuddy import BiomechanicalModelReal, MuscleType, MuscleStateType


def test_biomechanics_model_real_utils_functions():
    """
    The wholebody.osim model is used as it has ghost segments.
    The leg_without_ghost_parents.bioMod is used as it has an RT different from the identity matrix.
    """
    np.random.seed(42)

    # --- wholebody.osim --- #
    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wholebody_filepath = parent_path + "/examples/models/wholebody.osim"
    wholebody_biorbd_filepath = wholebody_filepath.replace(".osim", ".bioMod")

    # Define models
    wholebody_model = BiomechanicalModelReal.from_osim(
        filepath=wholebody_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
    )
    wholebody_model.to_biomod(wholebody_biorbd_filepath)
    wholebody_model_biorbd = biorbd.Model(wholebody_biorbd_filepath)

    nb_q = wholebody_model.nb_q
    assert nb_q == 42
    nb_markers = wholebody_model.nb_markers
    assert nb_markers == 49
    nb_segments = wholebody_model.nb_segments
    assert nb_segments == 196

    q_random = np.random.rand(nb_q)

    # Forward kinematics
    jcs_biobuddy = wholebody_model.forward_kinematics(q_random)
    for i_segment in range(nb_segments):
        jcs_biorbd = wholebody_model_biorbd.globalJCS(q_random, i_segment).to_array()
        npt.assert_array_almost_equal(
            jcs_biobuddy[wholebody_model.segments[i_segment].name][:, :, 0],
            jcs_biorbd,
            decimal=5,
        )

    # Markers position in global
    markers_biobuddy = wholebody_model.markers_in_global(q_random)
    for i_marker in range(nb_markers):
        markers_biorbd = wholebody_model_biorbd.markers(q_random)[i_marker].to_array()
        npt.assert_array_almost_equal(
            markers_biobuddy[:3, i_marker].reshape(
                3,
            ),
            markers_biorbd,
            decimal=4,
        )

    # --- leg_without_ghost_parents.bioMod --- #
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    # Define models
    leg_model = BiomechanicalModelReal.from_biomod(
        filepath=leg_filepath,
    )
    leg_model_biorbd = biorbd.Model(leg_filepath)

    nb_q = leg_model.nb_q
    assert nb_q == 12
    nb_markers = leg_model.nb_markers
    assert nb_markers == 19
    nb_segments = leg_model.nb_segments
    assert nb_segments == 7

    q_random = np.random.rand(nb_q)

    # Forward kinematics
    jcs_biobuddy = leg_model.forward_kinematics(q_random)
    for i_segment in range(nb_segments):
        jcs_biorbd = leg_model_biorbd.globalJCS(q_random, i_segment).to_array()
        npt.assert_array_almost_equal(
            jcs_biobuddy[leg_model.segments[i_segment].name][:, :, 0],
            jcs_biorbd,
            decimal=5,
        )

    # Markers position in global
    markers_biobuddy = leg_model.markers_in_global(q_random)
    for i_marker in range(nb_markers):
        markers_biorbd = leg_model_biorbd.markers(q_random)[i_marker].to_array()
        npt.assert_array_almost_equal(
            markers_biobuddy[:3, i_marker].reshape(
                3,
            ),
            markers_biorbd,
            decimal=4,
        )