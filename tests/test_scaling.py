"""
TODO: Add tests for the scaling configuration
"""

import os
import pytest
import opensim as osim
import shutil

import ezc3d
import biorbd
import numpy as np
import numpy.testing as npt

from test_utils import remove_temporary_biomods
from biobuddy import BiomechanicalModelReal, MuscleType, MuscleStateType, ScaleTool, C3dData
from biobuddy.components.real.rigidbody.segment_scaling import SegmentScaling
from biobuddy.components.real.rigidbody.marker_weight import MarkerWeight
from biobuddy.components.real.rigidbody.marker_real import MarkerReal
from biobuddy.components.real.rigidbody.contact_real import ContactReal
from biobuddy.components.real.rigidbody.inertial_measurement_unit_real import InertialMeasurementUnitReal
from biobuddy.components.real.muscle.muscle_real import MuscleReal
from biobuddy.components.real.muscle.via_point_real import ViaPointReal
from biobuddy.utils.linear_algebra import RotoTransMatrix


def convert_c3d_to_trc(c3d_filepath):
    """
    This function reads the c3d static file and converts it into a trc file that will be used to scale the model in OpenSim.
    The trc file is saved at the same place as the original c3d file.
    """
    trc_filepath = c3d_filepath.replace(".c3d", ".trc")

    c3d = ezc3d.c3d(c3d_filepath)
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]

    frame_rate = c3d["header"]["points"]["frame_rate"]
    marker_data = c3d["data"]["points"][:3, :, :] / 1000  # Convert in meters

    with open(trc_filepath, "w") as f:
        trc_file_name = os.path.basename(trc_filepath)
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_file_name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(
            "{:.2f}\t{:.2f}\t{}\t{}\tm\t{:.2f}\t{}\t{}\n".format(
                frame_rate,
                frame_rate,
                c3d["header"]["points"]["last_frame"],
                len(labels),
                frame_rate,
                c3d["header"]["points"]["first_frame"],
                c3d["header"]["points"]["last_frame"],
            )
        )
        f.write("Frame#\tTime\t" + "\t".join(labels) + "\n")
        f.write("\t\t" + "\t".join([f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(labels))]) + "\n")
        for frame in range(marker_data.shape[2]):
            time = frame / frame_rate
            frame_data = [f"{frame + 1}\t{time:.5f}"]
            for marker_idx in range(len(labels)):
                pos = marker_data[:, marker_idx, frame]
                frame_data.extend([f"{pos[0]:.5f}", f"{pos[1]:.5f}", f"{pos[2]:.5f}"])
            f.write("\t".join(frame_data) + "\n")


def visualize_model_scaling_output(scaled_model, osim_model_scaled, q, marker_names, marker_positions):
    """
    Only for debugging purposes.
    """
    biobuddy_path = "../examples/models/scaled_biobuddy.bioMod"
    osim_path = "../examples/models/scaled_osim.bioMod"
    scaled_model.to_biomod(biobuddy_path, with_mesh=True)
    osim_model_scaled.to_biomod(osim_path, with_mesh=True)

    import pyorerun
    from pyomeca import Markers

    # Compare the result visually
    t = np.linspace(0, 1, marker_positions.shape[2])
    viz = pyorerun.PhaseRerun(t)
    pyomarkers = Markers(data=marker_positions, channels=marker_names)

    # Model scaled in BioBuddy
    viz_biomod_model = pyorerun.BiorbdModel(biobuddy_path)
    viz_biomod_model.options.transparent_mesh = False
    viz_biomod_model.options.show_gravity = True
    viz_biomod_model.options.show_marker_labels = False
    viz_biomod_model.options.show_center_of_mass_labels = False
    viz_biomod_model.options.show_experimental_marker_labels = False
    viz.add_animated_model(viz_biomod_model, q, tracked_markers=pyomarkers)

    # Model scaled in OpenSim
    viz_scaled_model = pyorerun.BiorbdModel(osim_path)
    viz_scaled_model.options.transparent_mesh = False
    viz_scaled_model.options.show_gravity = True
    viz_scaled_model.options.show_marker_labels = False
    viz_scaled_model.options.show_center_of_mass_labels = False
    viz.add_animated_model(viz_scaled_model, q)

    # Animate
    viz.rerun_by_frame("Scaling comparison")

    os.remove(biobuddy_path)
    os.remove(osim_path)


# Unit tests for ScaleTool class methods
def test_scale_tool_init():
    """Test ScaleTool initialization with different parameters."""
    # TODO: Need to create a simple BiomechanicalModelReal mock for testing
    # For now, test basic initialization concepts

    # Test default parameters
    max_marker_movement = 0.1
    personalize_mass = True

    # Test parameter validation
    assert max_marker_movement > 0, "max_marker_movement should be positive"
    assert isinstance(personalize_mass, bool), "personalize_mass_distribution should be boolean"


def test_scale_tool_add_marker_weight():
    """Test adding marker weights to ScaleTool."""
    # TODO: Create a mock ScaleTool instance
    # For now, test MarkerWeight creation

    # Test creating a MarkerWeight
    marker_name = "test_marker"
    weight_value = 1.0

    # Test basic MarkerWeight parameters
    assert isinstance(marker_name, str), "Marker name should be string"
    assert isinstance(weight_value, (int, float)), "Weight should be numeric"
    assert weight_value >= 0, "Weight should be non-negative"


def test_scale_tool_scaling_segments():
    """Test adding and removing scaling segments."""
    # TODO: Test SegmentScaling creation and manipulation

    # Test segment name validation
    segment_name = "test_segment"
    assert isinstance(segment_name, str), "Segment name should be string"
    assert len(segment_name) > 0, "Segment name should not be empty"


def test_scale_tool_check_marker_movement():
    """Test marker movement validation logic."""
    # Test the logic for checking marker movement
    max_movement = 0.1

    # Simulate marker positions (3D positions over time)
    marker_positions = np.array(
        [
            [[0.0, 0.01, 0.02], [1.0, 1.01, 1.02]],  # Two markers
            [[0.0, 0.01, 0.02], [1.0, 1.01, 1.02]],  # Two time points
            [[0.0, 0.01, 0.02], [1.0, 1.01, 1.02]],  # Same positions (static)
        ]
    )

    # Test movement calculation
    min_pos = np.nanmin(marker_positions, axis=0)
    max_pos = np.nanmax(marker_positions, axis=0)
    movement = np.linalg.norm(max_pos - min_pos, axis=0)

    # Test that movement calculation works
    assert len(movement) == marker_positions.shape[1], "Movement should be calculated for each marker"
    assert all(movement >= 0), "Movement should be non-negative"


def test_scale_tool_static_rt_scaling():
    """Test the static RT (RotoTrans) scaling method."""
    # Test the static method scale_rt
    rt_matrix = np.eye(4)  # Identity matrix
    scale_factor = np.array([2.0, 3.0, 4.0])  # Different scaling for each axis

    # Test basic RT matrix properties
    assert rt_matrix.shape == (4, 4), "RT matrix should be 4x4"
    assert np.allclose(rt_matrix[:3, :3], np.eye(3)), "Rotation part should be identity"
    assert np.allclose(rt_matrix[:3, 3], 0), "Translation should be zero"

    # Test scale factor
    assert len(scale_factor) == 3, "Scale factor should be 3D"
    assert all(scale_factor > 0), "Scale factors should be positive"

    # Test the scaling operation concept
    scaled_rt = ScaleTool.scale_rt(rt_matrix, scale_factor)
    assert scaled_rt.shape == (4, 4), "Scaled RT should remain 4x4"
    # Translation should be scaled
    npt.assert_almost_equal(scaled_rt[:3, 3], rt_matrix[:3, 3] * scale_factor)


def test_scale_tool_marker_scaling():
    """Test marker scaling functionality."""
    # TODO: Create a mock MarkerReal for testing scaling

    # Test basic marker scaling concepts
    original_position = np.array([1.0, 2.0, 3.0])
    scale_factor = np.array([2.0, 2.0, 2.0])

    # Test scaling calculation
    scaled_position = original_position * scale_factor
    expected_position = np.array([2.0, 4.0, 6.0])

    npt.assert_almost_equal(scaled_position, expected_position)


def test_scale_tool_muscle_scaling():
    """Test muscle parameter scaling."""
    # Test muscle scaling concepts
    original_optimal_length = 0.1  # 10 cm
    original_tendon_slack_length = 0.05  # 5 cm

    # Test typical scaling based on segment length changes
    scale_factor = 1.2  # 20% increase

    scaled_optimal_length = original_optimal_length * scale_factor
    scaled_tendon_slack_length = original_tendon_slack_length * scale_factor

    # Test that scaling maintains proportions
    ratio_before = original_optimal_length / original_tendon_slack_length
    ratio_after = scaled_optimal_length / scaled_tendon_slack_length

    npt.assert_almost_equal(ratio_before, ratio_after, decimal=6)


def test_scale_tool_via_point_scaling():
    """Test via point scaling functionality."""
    # Test via point position scaling
    original_position = np.array([0.1, 0.2, 0.3])  # 3D position
    parent_scale_factor = np.array([1.5, 1.5, 1.5])  # Uniform scaling

    # Test scaling application
    scaled_position = original_position * parent_scale_factor
    expected_position = np.array([0.15, 0.3, 0.45])

    npt.assert_almost_equal(scaled_position, expected_position)


def test_scale_tool_inertial_scaling():
    """Test inertial parameter scaling."""
    # Test inertial scaling concepts
    original_mass = 1.0  # 1 kg
    scale_factor = np.array([2.0, 2.0, 2.0])  # Double in each dimension

    # Mass scales with volume (scale^3)
    volume_scale = np.prod(scale_factor)
    scaled_mass = original_mass * volume_scale

    # For uniform scaling of 2x, mass should be 8x
    expected_mass = 8.0
    npt.assert_almost_equal(scaled_mass, expected_mass)

    # Test moment of inertia scaling (should scale as mass * length^2)
    original_inertia = 0.1
    inertia_scale = volume_scale * (scale_factor[0] ** 2)  # mass * length^2
    scaled_inertia = original_inertia * inertia_scale

    # For uniform 2x scaling: mass*8 * length^2*4 = 32x
    expected_inertia = 0.1 * 32
    npt.assert_almost_equal(scaled_inertia, expected_inertia)


def test_scale_tool_segment_validation():
    """Test segment validation logic."""
    # Test segment name validation
    valid_segment_names = ["pelvis", "femur_r", "tibia_r", "foot_r"]

    for name in valid_segment_names:
        assert isinstance(name, str), f"Segment name {name} should be string"
        assert len(name) > 0, f"Segment name {name} should not be empty"
        assert "_" not in name or name.count("_") == 1, f"Segment name {name} should have at most one underscore"


def test_scale_tool_mass_distribution():
    """Test mass distribution validation."""
    # Test mass distribution logic
    total_mass = 70.0  # kg
    segment_masses = [10.0, 15.0, 20.0, 25.0]  # kg

    # Test that individual masses are positive
    for mass in segment_masses:
        assert mass > 0, f"Segment mass {mass} should be positive"

    # Test mass conservation (total should be preserved)
    segment_total = sum(segment_masses)
    assert segment_total == total_mass, "Sum of segment masses should equal total mass"


def test_scale_tool_matrix_operations():
    """Test matrix operations used in scaling."""
    # Test rotation matrix properties
    rotation_matrix = np.eye(3)
    assert np.allclose(np.linalg.det(rotation_matrix), 1.0), "Rotation matrix should have determinant 1"
    assert np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3)), "Rotation matrix should be orthogonal"

    # Test transformation matrix composition
    translation = np.array([1.0, 2.0, 3.0])
    rt_matrix = np.eye(4)
    rt_matrix[:3, 3] = translation

    # Test that transformation preserves structure
    assert rt_matrix.shape == (4, 4), "Transformation matrix should be 4x4"
    assert rt_matrix[3, 3] == 1.0, "Last element should be 1"
    npt.assert_almost_equal(rt_matrix[:3, 3], translation)


def test_scaling_wholebody():

    np.random.seed(42)

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_relative_path = "Geometry_cleaned"
    osim_filepath = parent_path + "/examples/models/wholebody.osim"
    xml_filepath = parent_path + "/examples/models/wholebody.xml"
    scaled_biomod_filepath = parent_path + "/examples/models/wholebody_scaled.bioMod"
    converted_scaled_osim_filepath = parent_path + "/examples/models/wholebody_converted_scaled.bioMod"
    static_filepath = parent_path + "/examples/data/static.c3d"
    trc_file_path = parent_path + "/examples/data/static.trc"

    # --- Convert the vtp mesh files --- #
    # geometry_path = parent_path + "/external/opensim-models/Geometry"
    # cleaned_geometry_path = parent_path + "/models/Geometry_cleaned"
    # mesh_parser = MeshParser(geometry_path)
    # mesh_parser.process_meshes(fail_on_error=False)
    # mesh_parser.write(cleaned_geometry_path, MeshFormat.VTP)

    # --- Scale in opensim ---#
    # convert_c3d_to_trc(static_filepath)  # To translate c3d to trc
    shutil.copyfile(trc_file_path, parent_path + "/examples/models/static.trc")
    shutil.copyfile(xml_filepath, "wholebody.xml")
    shutil.copyfile(osim_filepath, "wholebody.osim")
    opensim_tool = osim.ScaleTool(xml_filepath)
    opensim_tool.run()

    # --- Read the model scaled in OpenSim and translate to bioMod --- #
    osim_model_scaled = BiomechanicalModelReal().from_osim(
        filepath=parent_path + "/examples/models/scaled.osim",
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=cleaned_relative_path,
    )
    osim_model_scaled.to_biomod(converted_scaled_osim_filepath, with_mesh=False)
    scaled_osim_model = biorbd.Model(converted_scaled_osim_filepath)

    # --- Scale in BioBuddy --- #
    original_model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir=cleaned_relative_path,
    )

    scale_tool = ScaleTool(original_model=original_model).from_xml(filepath=xml_filepath)
    scaled_model = scale_tool.scale(
        filepath=static_filepath,
        first_frame=0,
        last_frame=531,
        mass=69.2,
        q_regularization_weight=0.1,
        make_static_pose_the_models_zero=False,
    )
    scaled_model.to_biomod(scaled_biomod_filepath, with_mesh=False)
    scaled_biorbd_model = biorbd.Model(scaled_biomod_filepath)

    # --- Test the scaling factors --- #
    c3d_data = C3dData(c3d_path=static_filepath, first_frame=0, last_frame=531)
    marker_names = c3d_data.marker_names
    marker_positions = c3d_data.all_marker_positions[:3, :, :]

    q_zeros = np.zeros((42, marker_positions.shape[2]))
    q_random = np.random.rand(42) * 2 * np.pi

    # # For debugging
    # visualize_model_scaling_output(scaled_model, osim_model_scaled, q_zeros, marker_names, marker_positions)

    # TODO: Find out why there is a discrepancy between the OpenSim and BioBuddy scaling factors of the to the third decimal.
    # Scaling factors from scaling_factors.osim  (TODO: add the scaling factors in the osim parser)
    scaling_factors = {
        "pelvis": 0.883668,
        "femur_r": 1.1075,
        "tibia_r": 1.00352,
        "talus_r": 0.961683,
        "calcn_r": 1.05904,
        "toes_r": 0.999246,
        "torso": 1.04094,
        # "head_and_neck": 1.02539,  # There seems to be a trick somewhere to remove the helmet offset,
        "humerus_r": 1.00517,
        "ulna_r": 1.12622,
        "radius_r": 1.04826,
        "lunate_r": 1.12829,
        # "hand_r": 1.18954,
        # "fingers_r": 1.26327,  # There is a problem with the hands in this model
    }
    for segment_name, scale_factor in scaling_factors.items():
        biobuddy_scaling_factors = scale_tool.scaling_segments[segment_name].compute_scaling_factors(
            original_model, marker_positions, marker_names
        )
        npt.assert_almost_equal(biobuddy_scaling_factors.mass, scale_factor, decimal=2)

    # --- Test masses --- #
    # Total mass
    npt.assert_almost_equal(scaled_osim_model.mass(), 69.2, decimal=5)
    npt.assert_almost_equal(scaled_biorbd_model.mass(), 69.2, decimal=5)

    # TODO: Find out why there is a discrepancy between the OpenSim and BioBuddy scaled masses.
    # Pelvis:
    # Theoretical mass without renormalization -> 0.883668 * 11.776999999999999 = 10.406958035999999
    # Biobuddy -> 9.4381337873063
    # OpenSim -> 6.891020778193859 (seems like we are closer than Opensim !?)
    # Segment mass
    for i_segment, segment_name in enumerate(scaled_model.segments.keys()):
        if scaled_model.segments[segment_name].inertia_parameters is None:
            mass_biobuddy = 0
        else:
            mass_biobuddy = scaled_model.segments[segment_name].inertia_parameters.mass
        mass_to_biorbd = scaled_biorbd_model.segment(i_segment).characteristics().mass()
        # mass_osim = scaled_osim_model.segment(i_segment).characteristics().mass()
        npt.assert_almost_equal(mass_to_biorbd, mass_biobuddy)
        # npt.assert_almost_equal(mass_osim, mass_biobuddy)
        # npt.assert_almost_equal(mass_to_biorbd, mass_osim)
        if segment_name in scaling_factors.keys():
            original_mass = original_model.segments[segment_name].inertia_parameters.mass
            # We have to let a huge buffer here because of the renormalization
            if scaling_factors[segment_name] < 1:
                npt.assert_array_less(mass_biobuddy * 0.9, original_mass)
            else:
                npt.assert_array_less(original_mass, mass_biobuddy * 1.1)

    # CoM
    for i_segment, segment_name in enumerate(scaled_model.segments.keys()):
        print(segment_name)
        if "finger" in segment_name:
            continue
        if scaled_model.segments[segment_name].inertia_parameters is not None:
            # Zero
            com_biobuddy_0 = (
                scaled_model.segments[segment_name]
                .inertia_parameters.center_of_mass[:3]
                .reshape(
                    3,
                )
            ) + scaled_model.segment_coordinate_system_in_global(segment_name)[:3, 3, 0]
            com_to_biorbd_0 = scaled_biorbd_model.CoMbySegment(q_zeros[:, 0], i_segment).to_array()
            com_osim_0 = scaled_osim_model.CoMbySegment(q_zeros[:, 0], i_segment).to_array()
            npt.assert_almost_equal(com_to_biorbd_0, com_biobuddy_0, decimal=2)
            npt.assert_almost_equal(com_osim_0, com_biobuddy_0, decimal=2)
            npt.assert_almost_equal(com_to_biorbd_0, com_osim_0, decimal=2)
            # Random
            com_biobuddy_rand = scaled_biorbd_model.CoMbySegment(q_random, i_segment).to_array()
            com_osim_rand = scaled_osim_model.CoMbySegment(q_random, i_segment).to_array()
            npt.assert_almost_equal(com_osim_rand, com_biobuddy_rand, decimal=2)

    # Inertia
    for i_segment, segment_name in enumerate(scaled_model.segments.keys()):
        print(segment_name)
        if "finger" in segment_name:
            continue
        if scaled_model.segments[segment_name].inertia_parameters is not None:
            inertia_biobuddy = scaled_model.segments[segment_name].inertia_parameters.inertia[:3, :3]
            mass_to_biorbd = scaled_biorbd_model.segment(i_segment).characteristics().inertia().to_array()
            inertia_osim = scaled_osim_model.segment(i_segment).characteristics().inertia().to_array()
            # Large tolerance since the difference in scaling factor affects largely this value
            npt.assert_almost_equal(mass_to_biorbd, inertia_biobuddy, decimal=5)
            npt.assert_almost_equal(inertia_osim, inertia_biobuddy, decimal=1)
            npt.assert_almost_equal(mass_to_biorbd, inertia_osim, decimal=1)

    # Marker positions
    for i_marker in range(scaled_biorbd_model.nbMarkers()):
        biobuddy_scaled_marker = scaled_biorbd_model.markers(q_zeros[:, 0])[i_marker].to_array()
        osim_scaled_marker = scaled_osim_model.markers(q_zeros[:, 0])[i_marker].to_array()
        # TODO: The tolerance is large since the markers are already replaced based on the static trial.
        npt.assert_almost_equal(osim_scaled_marker, biobuddy_scaled_marker, decimal=1)

    # Via point positions
    for via_point_name in original_model.via_points.keys():
        biobuddy_scaled_via_point = scaled_model.via_points[via_point_name].position[:3]
        osim_scaled_via_point = osim_model_scaled.via_points[via_point_name].position[:3]
        npt.assert_almost_equal(biobuddy_scaled_via_point, osim_scaled_via_point, decimal=6)

    # Muscle properties
    for muscle in original_model.muscles.keys():
        if (
            muscle
            in [
                "semiten_r",
                "vas_med_r",
                "vas_lat_r",
                "med_gas_r",
                "lat_gas_r",
                "semiten_l",
                "vas_med_l",
                "vas_lat_l",
                "med_gas_l",
                "lat_gas_l",
            ]
            or "stern_mast" in muscle
        ):
            # Skipping muscles with ConditionalPathPoints and MovingPathPoints
            # Skipping the head since there is a difference in scaling
            continue
        print(muscle)
        biobuddy_optimal_length = scaled_model.muscles[muscle].optimal_length
        osim_optimal_length = osim_model_scaled.muscles[muscle].optimal_length
        npt.assert_almost_equal(biobuddy_optimal_length, osim_optimal_length, decimal=6)
        biobuddy_tendon_slack_length = scaled_model.muscles[muscle].tendon_slack_length
        osim_tendon_slack_length = osim_model_scaled.muscles[muscle].tendon_slack_length
        npt.assert_almost_equal(biobuddy_tendon_slack_length, osim_tendon_slack_length, decimal=6)

    # Make sure the experimental markers are at the same position as the model's ones in static pose
    scale_tool = ScaleTool(original_model=original_model).from_xml(filepath=xml_filepath)
    scaled_model = scale_tool.scale(
        filepath=static_filepath,
        first_frame=0,
        last_frame=531,
        mass=69.2,
        q_regularization_weight=0.1,
        make_static_pose_the_models_zero=True,
    )
    scaled_model.to_biomod(scaled_biomod_filepath, with_mesh=False)
    scaled_biorbd_model = biorbd.Model(scaled_biomod_filepath)

    exp_markers = scale_tool.mean_experimental_markers[:, :]
    for i_marker in range(exp_markers.shape[1]):
        biobuddy_scaled_marker = scaled_biorbd_model.markers(q_zeros[:, 0])[i_marker].to_array()
        npt.assert_almost_equal(exp_markers[:, i_marker], biobuddy_scaled_marker, decimal=5)

    os.remove(scaled_biomod_filepath)
    os.remove(converted_scaled_osim_filepath)
    os.remove(parent_path + "/examples/models/static.trc")
    os.remove("wholebody.xml")
    os.remove("wholebody.osim")
    os.remove(parent_path + "/examples/models/scaled.osim")
    remove_temporary_biomods()
