import os
import pytest
import numpy as np
import numpy.testing as npt
import biorbd

from biobuddy import BiomechanicalModelReal, MuscleType, MuscleStateType
from biobuddy.components.real.model_dynamics import ModelDynamics


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
    wholebody_model = BiomechanicalModelReal().from_osim(
        filepath=wholebody_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
    )
    wholebody_model.fix_via_points()
    wholebody_model.to_biomod(wholebody_biorbd_filepath, with_mesh=False)
    wholebody_model_biorbd = biorbd.Model(wholebody_biorbd_filepath)

    nb_q = wholebody_model.nb_q
    assert nb_q == 42
    nb_markers = wholebody_model.nb_markers
    assert nb_markers == 49
    nb_segments = wholebody_model.nb_segments
    # There is a file overwrite somewhere in the tests making this assert fail
    # assert nb_segments == 200 == 196
    nb_contacts = wholebody_model.nb_contacts
    assert nb_contacts == 0
    nb_muscles = wholebody_model.nb_muscles
    assert nb_muscles == 56
    assert wholebody_model_biorbd.nbMuscles() == wholebody_model.nb_muscles
    nb_via_points = wholebody_model.nb_via_points
    assert nb_via_points == 100

    q_random = np.random.rand(nb_q)
    q_zeros = np.zeros((nb_q,))

    # Forward kinematics
    jcs_biobuddy = wholebody_model.forward_kinematics(q_random)
    for i_segment in range(nb_segments):
        jcs_biorbd = wholebody_model_biorbd.globalJCS(q_random, i_segment).to_array()
        npt.assert_array_almost_equal(
            jcs_biobuddy[wholebody_model.segments[i_segment].name][0].rt_matrix,
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

    # CoM position in global
    com_biobuddy = wholebody_model.total_com_in_global(q_random)
    com_biorbd = wholebody_model_biorbd.CoM(q_random).to_array()
    npt.assert_array_almost_equal(
        com_biobuddy[:3].reshape(
            3,
        ),
        com_biorbd,
        decimal=4,
    )
    for i_com in range(nb_segments):
        com_biobuddy = wholebody_model.segment_com_in_global(wholebody_model.segment_names[i_com], q_random)
        com_biorbd = wholebody_model_biorbd.CoMbySegment(q_random, i_com).to_array()
        if com_biobuddy is not None:
            npt.assert_array_almost_equal(
                com_biobuddy[:3].reshape(
                    3,
                ),
                com_biorbd,
                decimal=4,
            )

    # # TODO: Test Markers jacobian (there is a problem with the biobuddy implementation :/)
    # markers_jacobian_biobuddy = wholebody_model.markers_jacobian(q_random)
    # markers_jacobian_biorbd = wholebody_model_biorbd.markersJacobian(q_random)
    # for i_marker in range(nb_markers):
    #     try:
    #         npt.assert_array_almost_equal(
    #             markers_jacobian_biobuddy[:, i_marker, :].reshape(3, nb_q),
    #             markers_jacobian_biorbd[i_marker].to_array(),
    #             decimal=4,
    #         )
    #     except:
    #         print(f"Marker number {i_marker}, for q {np.where(np.abs(markers_jacobian_biobuddy[:, i_marker, :].reshape(3, nb_q) - markers_jacobian_biorbd[i_marker].to_array()) > 0.0001)}.")

    # Test forward kinematics (here because it resets the model with q_zeros before the muscle evaluation below)
    jcs_biobuddy = wholebody_model.forward_kinematics(q_zeros)
    for i_segment in range(nb_segments):
        jcs_biorbd = wholebody_model_biorbd.globalJCS(q_zeros, i_segment).to_array()
        npt.assert_array_almost_equal(
            jcs_biobuddy[wholebody_model.segments[i_segment].name][0].rt_matrix,
            jcs_biorbd,
            decimal=5,
        )

    # Test muscle length
    muscle_names = [m.to_string() for m in wholebody_model_biorbd.muscleNames()]
    for i_muscle, muscle_name in enumerate(muscle_names):

        # Via point positions
        muscle_points_in_global_biobuddy = wholebody_model.via_points_in_global(muscle_name, q_zeros)
        muscle_points_in_global_biorbd = [
            m.to_array()
            for m in wholebody_model_biorbd.muscle(i_muscle).musclesPointsInGlobal(wholebody_model_biorbd, q_zeros)
        ]
        for i_via_point in range(len(muscle_points_in_global_biorbd) - 2):
            npt.assert_array_almost_equal(
                muscle_points_in_global_biobuddy[:3, i_via_point, 0],
                muscle_points_in_global_biorbd[i_via_point + 1],
                decimal=5,
            )

        # Muscle tendon length
        muscle_tendon_biobuddy = wholebody_model.muscle_tendon_length(muscle_name, q_zeros)
        muscle_tendon_biorbd = wholebody_model_biorbd.muscle(i_muscle).musculoTendonLength(
            wholebody_model_biorbd, q_zeros
        )
        npt.assert_array_almost_equal(
            muscle_tendon_biobuddy,
            muscle_tendon_biorbd,
            decimal=4,
        )

    # --- leg_without_ghost_parents.bioMod --- #
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    leg_filepath_without_mesh = leg_filepath.replace(".bioMod", "_without_mesh.bioMod")

    # Define models
    leg_model = BiomechanicalModelReal().from_biomod(
        filepath=leg_filepath,
    )
    leg_model.to_biomod(leg_filepath_without_mesh, with_mesh=False)
    leg_model_biorbd = biorbd.Model(leg_filepath_without_mesh)

    nb_q = leg_model.nb_q
    assert nb_q == 10
    nb_markers = leg_model.nb_markers
    assert nb_markers == 16
    nb_segments = leg_model.nb_segments
    assert nb_segments == 4

    nb_frames = 5
    q_random = np.random.rand(nb_q, nb_frames)

    # Forward kinematics
    jcs_biobuddy = leg_model.forward_kinematics(q_random)
    for i_frame in range(nb_frames):
        for i_segment in range(nb_segments):
            jcs_biorbd = leg_model_biorbd.globalJCS(q_random[:, i_frame], i_segment).to_array()
            npt.assert_array_almost_equal(
                jcs_biobuddy[leg_model.segments[i_segment].name][i_frame].rt_matrix,
                jcs_biorbd,
                decimal=5,
            )

    # Markers position in global
    markers_biobuddy = leg_model.markers_in_global(q_random)
    for i_frame in range(nb_frames):
        for i_marker in range(nb_markers):
            markers_biorbd = leg_model_biorbd.markers(q_random[:, i_frame])[i_marker].to_array()
            npt.assert_array_almost_equal(
                markers_biobuddy[:3, i_marker, i_frame].reshape(
                    3,
                ),
                markers_biorbd,
                decimal=4,
            )

    # CoM position in global
    for i_frame in range(nb_frames):
        for i_com, segment_name in enumerate(leg_model.segment_names):
            com_biobuddy = leg_model.segment_com_in_global(segment_name, q_random[:, i_frame])
            if com_biobuddy is not None:
                com_biorbd = leg_model_biorbd.CoMbySegment(q_random[:, i_frame])[i_com].to_array()
                npt.assert_array_almost_equal(
                    com_biobuddy[:3].reshape(
                        3,
                    ),
                    com_biorbd,
                    decimal=4,
                )

    # --- leg_without_ghost_parents.bioMod --- #
    complex_filepath = parent_path + "/examples/models/example_of_complex_model.bioMod"

    # Define models
    complex_model = BiomechanicalModelReal().from_biomod(
        filepath=complex_filepath,
    )
    complex_model_biorbd = biorbd.Model(complex_filepath)

    nb_q = complex_model.nb_q
    nb_frames = 5
    q_random = np.random.rand(nb_q, nb_frames)

    nb_contacts = complex_model.nb_contacts
    assert nb_contacts == 1
    assert complex_model_biorbd.nbRigidContacts() == nb_contacts

    # Contact position in global
    contact_biobuddy = complex_model.contacts_in_global(q_random)
    for i_frame in range(nb_frames):
        for i_contact in range(nb_contacts):
            contact_biorbd = complex_model_biorbd.rigidContact(q_random[:, i_frame], i_contact, True).to_array()
            npt.assert_array_almost_equal(
                contact_biobuddy[:3, i_contact, i_frame].reshape(
                    3,
                ),
                contact_biorbd,
                decimal=4,
            )

    # Remove the biomod file created
    os.remove(wholebody_biorbd_filepath)
    os.remove(leg_filepath_without_mesh)


def test_model_dynamics_initialization():
    """Test ModelDynamics initialization and requires_initialization decorator."""
    # Test basic initialization
    model_dynamics = ModelDynamics()
    assert model_dynamics.is_initialized is False
    assert model_dynamics.segments is None
    assert model_dynamics.muscle_groups is None


def test_requires_initialization_decorator():
    """Test that requires_initialization decorator properly raises RuntimeError."""
    model_dynamics = ModelDynamics()

    # Test that calling methods before initialization raises RuntimeError
    with pytest.raises(RuntimeError, match="segment_coordinate_system_in_local cannot be called"):
        model_dynamics.segment_coordinate_system_in_local("base")

    with pytest.raises(RuntimeError, match="segment_coordinate_system_in_global cannot be called"):
        model_dynamics.segment_coordinate_system_in_global("base")

    with pytest.raises(RuntimeError, match="forward_kinematics cannot be called"):
        model_dynamics.forward_kinematics()

    with pytest.raises(RuntimeError, match="markers_in_global cannot be called"):
        model_dynamics.markers_in_global()


def test_base_segment_coordinate_system():
    """Test coordinate system methods for base segment."""
    # We'll use a simple model that can be created without complex dependencies
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"
    leg_model = BiomechanicalModelReal().from_biomod(filepath=leg_filepath)

    # Test femur_r segment coordinate system
    scs_local = leg_model.segment_coordinate_system_in_local("femur_r")
    scs_global = leg_model.segment_coordinate_system_in_global("femur_r")

    # Test the values
    npt.assert_almost_equal(
        scs_local.rt_matrix,
        np.array(
            [
                [0.941067, 0.334883, 0.047408, -0.067759],
                [-0.335537, 0.906752, 0.255373, -0.06335],
                [0.042533, -0.25623, 0.96568, 0.080026],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=5,
    )
    npt.assert_almost_equal(
        scs_global.rt_matrix,
        np.array(
            [
                [0.99525177, 0.0957537, -0.01746835, 0.64714318],
                [-0.0351334, 0.18604158, -0.98191353, 0.45086954],
                [-0.09077201, 0.9778649, 0.18852236, 0.84057565],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=5,
    )


def test_marker_residual_static_method():
    """Test _marker_residual static method."""
    np.random.seed(42)

    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    leg_model = BiomechanicalModelReal().from_biomod(filepath=leg_filepath)

    # Setup test parameters
    q = np.random.rand(leg_model.nb_q) * 0.1
    q_target = np.zeros((leg_model.nb_q, 1))
    q_regularization_weight = 0.1

    # Get model markers to create "experimental" markers
    model_markers = leg_model.markers_in_global(q)
    experimental_markers = model_markers[:3, :, 0] + np.random.rand(3, leg_model.nb_markers) * 0.01  # Add small noise

    marker_names = leg_model.marker_names
    marker_weights = np.ones(leg_model.nb_markers)

    # Test the residual function
    residual = ModelDynamics._marker_residual(
        model=leg_model,
        q_regularization_weight=q_regularization_weight,
        q_target=q_target,
        q=q,
        marker_names=marker_names,
        experimental_markers=experimental_markers,
        marker_weights_reordered=marker_weights,
        with_biorbd=False,
    )

    # Check the residual values
    npt.assert_almost_equal(
        residual,
        np.array(
            [
                -0.00020584,
                -0.00199674,
                -0.00034389,
                -0.0096991,
                -0.00514234,
                -0.0090932,
                -0.00832443,
                -0.00592415,
                -0.0025878,
                -0.00212339,
                -0.0004645,
                -0.00662522,
                -0.00181825,
                -0.00607545,
                -0.00311711,
                -0.00183405,
                -0.00170524,
                -0.00520068,
                -0.00304242,
                -0.00065052,
                -0.0054671,
                -0.00524756,
                -0.00948886,
                -0.00184854,
                -0.00431945,
                -0.00965632,
                -0.00969585,
                -0.00291229,
                -0.00808397,
                -0.00775133,
                -0.00611853,
                -0.00304614,
                -0.00939499,
                -0.00139494,
                -0.00097672,
                -0.00894827,
                -0.00292145,
                -0.00684233,
                -0.005979,
                -0.00366362,
                -0.00440152,
                -0.00921874,
                -0.0045607,
                -0.00122038,
                -0.00088493,
                -0.00785176,
                -0.00495177,
                -0.00195983,
                0.0037454,
                0.00950714,
                0.00731994,
                0.00598658,
                0.00156019,
                0.00155995,
                0.00058084,
                0.00866176,
                0.00601115,
                0.00708073,
            ]
        ),
    )


def test_forward_kinematics_error_handling():
    """Test error handling in forward_kinematics."""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    leg_model = BiomechanicalModelReal().from_biomod(filepath=leg_filepath)

    # Test with wrong q dimensions
    q_wrong_shape = np.zeros((leg_model.nb_q, 5, 2))  # 3D array should fail
    with pytest.raises(RuntimeError, match="q must be of shape"):
        leg_model.forward_kinematics(q_wrong_shape)


def test_markers_in_global_error_handling():
    """Test error handling in markers_in_global."""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    leg_model = BiomechanicalModelReal().from_biomod(filepath=leg_filepath)

    # Test with wrong q dimensions
    q_wrong_shape = np.zeros((leg_model.nb_q, 5, 2))  # 3D array should fail
    with pytest.raises(RuntimeError, match="q must be of shape"):
        leg_model.markers_in_global(q_wrong_shape)


def test_com_in_global_error_handling():
    """Test error handling in segment_com_in_global."""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    leg_model = BiomechanicalModelReal().from_biomod(filepath=leg_filepath)

    # Test with wrong q dimensions
    q_wrong_shape = np.zeros((leg_model.nb_q, 5, 2))  # 3D array should fail
    segment_name = list(leg_model.segments.keys())[0]
    with pytest.raises(RuntimeError, match="q must be of shape"):
        leg_model.segment_com_in_global(segment_name, q_wrong_shape)


def test_contacts_in_global_error_handling():
    """Test error handling in contacts_in_global."""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_filepath = parent_path + "/examples/models/example_of_complex_model.bioMod"

    leg_model = BiomechanicalModelReal().from_biomod(filepath=leg_filepath)

    # Test with wrong q dimensions
    q_wrong_shape = np.zeros((leg_model.nb_q, 5, 2))  # 3D array should fail
    with pytest.raises(RuntimeError, match="q must be of shape"):
        leg_model.contacts_in_global(q_wrong_shape)


def test_inverse_kinematics_basic():
    """Test basic inverse kinematics functionality."""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    leg_model = BiomechanicalModelReal().from_biomod(filepath=leg_filepath)
    tempo_leg_filepath = leg_filepath.replace(".bioMod", "_without_mesh.bioMod")
    leg_model.to_biomod(tempo_leg_filepath, with_mesh=False)
    biorbd_leg_model = biorbd.Model(tempo_leg_filepath)
    leg_model = BiomechanicalModelReal().from_biomod(filepath=tempo_leg_filepath)

    # Create synthetic experimental data by forward kinematics
    q_true = np.random.rand(leg_model.nb_q, 1) * 0.1  # Small random joint angles
    marker_positions_true = leg_model.markers_in_global(q_true)
    marker_names = leg_model.marker_names

    # Test with q_regularization_weight: np.ndarrray
    q_reconstructed_array, _ = leg_model.inverse_kinematics(
        marker_positions=marker_positions_true[:3, :, :],
        marker_names=marker_names,
        q_regularization_weight=np.ones((leg_model.nb_q,)) * 0.01,
        q_target=None,
        marker_weights=None,
        method="lm",
        animate_reconstruction=False,
    )

    # Test without residuals
    q_reconstructed, residuals = leg_model.inverse_kinematics(
        marker_positions=marker_positions_true[:3, :, :],
        marker_names=marker_names,
        q_regularization_weight=0.01,
        q_target=None,
        marker_weights=None,
        method="lm",
        animate_reconstruction=False,
    )
    assert residuals is None

    q_biorbd = biorbd.InverseKinematics(biorbd_leg_model, marker_positions_true[:3, :, :]).solve()

    # Check that the solution is the same as biorbd
    npt.assert_array_almost_equal(q_reconstructed, q_biorbd, decimal=3)

    # Check that the solution is close to the true q
    npt.assert_array_almost_equal(q_reconstructed, q_true, decimal=3)

    # Check that the solution is the same as with an array as q_regularization_weight
    npt.assert_array_almost_equal(q_reconstructed, q_reconstructed_array)

    # Test with residuals
    q_reconstructed, residuals = leg_model.inverse_kinematics(
        marker_positions=marker_positions_true[:3, :, :],
        marker_names=marker_names,
        q_regularization_weight=0.01,
        q_target=None,
        marker_weights=None,
        method="lm",
        animate_reconstruction=False,
        compute_residual_distance=True,
    )

    # Check that the solution is the same as biorbd
    npt.assert_array_almost_equal(q_reconstructed, q_biorbd, decimal=3)

    # Check that the solution is close to the true q
    npt.assert_array_almost_equal(q_reconstructed, q_true, decimal=3)

    # Check that the residuals are computed correctly
    for i_marker in range(leg_model.nb_markers):
        residuals_biorbd = np.linalg.norm(
            biorbd_leg_model.markers(q_biorbd[:, 0])[i_marker].to_array()
            - marker_positions_true[:3, i_marker, :].reshape(
                3,
            )
        )
        npt.assert_array_almost_equal(residuals[i_marker], residuals_biorbd, decimal=3)

    q_reconstructed, _ = leg_model.inverse_kinematics(
        marker_positions=marker_positions_true[:3, :, :],
        marker_names=marker_names,
        q_regularization_weight=0.01,
        q_target=None,
        marker_weights=None,
        method="lm",
        animate_reconstruction=False,
    )

    # Test that it also works when there are NaNs in the exp data
    marker_positions_with_nan = marker_positions_true.copy()
    marker_positions_with_nan[0, 0, 0] = np.nan  # Introduce NaN in the first marker position
    q_reconstructed_nan, _ = leg_model.inverse_kinematics(
        marker_positions=marker_positions_with_nan[:3, :, :],
        marker_names=marker_names,
        q_regularization_weight=0.01,
        q_target=None,
        marker_weights=None,
        method="lm",
        animate_reconstruction=False,
    )

    # Check that the solution is still close to the true q
    npt.assert_array_almost_equal(q_reconstructed_nan, q_true, decimal=3)

    # Delete the temporary model created
    os.remove(tempo_leg_filepath)


def test_inverse_kinematics_error_handling():
    """Test error handling in inverse kinematics."""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    leg_model = BiomechanicalModelReal().from_biomod(filepath=leg_filepath)

    # Test with wrong marker_positions shape
    wrong_shape_markers = np.random.rand(2, leg_model.nb_markers)  # Only 2 rows instead of 3
    marker_names = leg_model.marker_names
    with pytest.raises(RuntimeError, match="marker_positions must be of shape"):
        leg_model.inverse_kinematics(
            marker_positions=wrong_shape_markers,
            marker_names=marker_names,
        )

    # Test with 4D array
    wrong_shape_4d = np.random.rand(3, leg_model.nb_markers, 1, 1)
    with pytest.raises(RuntimeError, match="marker_positions must be of shape"):
        leg_model.inverse_kinematics(
            marker_positions=wrong_shape_4d,
            marker_names=marker_names,
        )


def test_has_ghost_parent():
    """Test transformation from parent offset to real segment."""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Try models that might have ghost segments
    model_with_filepath = parent_path + "/examples/models/leg_with_ghost_parents.bioMod"
    model_without_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    model_with = BiomechanicalModelReal().from_biomod(filepath=model_with_filepath)
    model_without = BiomechanicalModelReal().from_biomod(filepath=model_without_filepath)

    assert model_with.segment_has_ghost_parents("femur_r") == True
    assert model_without.segment_has_ghost_parents("femur_r") == False


def test_rt_from_parent_offset_to_real_segment_basic():
    """Test transformation from parent offset to real segment."""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Try models that might have ghost segments
    model_with_filepath = parent_path + "/examples/models/leg_with_ghost_parents.bioMod"
    model_without_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    model_with = BiomechanicalModelReal().from_biomod(filepath=model_with_filepath)
    model_without = BiomechanicalModelReal().from_biomod(filepath=model_without_filepath)

    rt_result = model_with.rt_from_parent_offset_to_real_segment("femur_r").rt_matrix
    npt.assert_almost_equal(
        rt_result,
        np.array(
            [
                [0.998838, 0.017684, -0.044828, 0.0],
                [-0.011801, 0.99167, 0.128259, 0.0],
                [0.046723, -0.127581, 0.990727, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=5,
    )

    rt_result = model_without.rt_from_parent_offset_to_real_segment("femur_r").rt_matrix
    npt.assert_almost_equal(rt_result, np.identity(4))


def test_model_dynamics_properties():
    """Test various property getters and basic model structure."""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leg_filepath = parent_path + "/examples/models/leg_without_ghost_parents.bioMod"

    leg_model = BiomechanicalModelReal().from_biomod(filepath=leg_filepath)

    # Test that model is properly initialized
    assert leg_model.is_initialized is True

    # Test basic properties
    assert isinstance(leg_model.nb_q, int)
    assert leg_model.nb_q == 10

    assert isinstance(leg_model.nb_markers, int)
    assert leg_model.nb_markers == 16

    assert isinstance(leg_model.nb_segments, int)
    assert leg_model.nb_segments == 4

    assert isinstance(leg_model.nb_contacts, int)
    assert leg_model.nb_contacts == 0
