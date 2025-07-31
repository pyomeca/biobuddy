import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import (
    MarkerReal,
    MeshFileReal,
    SegmentCoordinateSystemReal,
    AxisReal,
    ContactReal,
    MeshReal,
    SegmentReal,
    InertialMeasurementUnitReal,
    ViaPointReal,
    MuscleReal,
    MuscleGroupReal,
    MuscleType,
    MuscleStateType,
    RotoTransMatrix,
    Translations,
    Rotations,
    RangeOfMotion,
    Ranges,
    ViaPoint,
)
from test_utils import MockC3dData


# ------- MarkerReal ------- #
def test_init_marker_real():
    # Test initialization with minimal parameters
    marker = MarkerReal(name="test_marker", parent_name="segment1")
    assert marker.name == "test_marker"
    assert marker.parent_name == "segment1"
    assert marker.is_technical is True
    assert marker.is_anatomical is False

    # Test initialization with all parameters
    position = np.array([[1.0], [2.0], [3.0], [1.0]])
    marker = MarkerReal(
        name="test_marker", parent_name="segment1", position=position, is_technical=False, is_anatomical=True
    )

    assert marker.name == "test_marker"
    assert marker.parent_name == "segment1"
    npt.assert_array_equal(marker.position, position)
    assert marker.is_technical is False
    assert marker.is_anatomical is True


def test_marker_real_mean_position():
    # Test with single position
    position = np.array([[1.0], [2.0], [3.0], [1.0]])
    marker = MarkerReal(name="test_marker", parent_name="segment1", position=position)
    npt.assert_array_equal(marker.mean_position, np.array([1.0, 2.0, 3.0, 1.0]))

    # Test with multiple positions (time series)
    positions = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [1.0, 1.0, 1.0]])
    marker = MarkerReal(name="test_marker", parent_name="segment1", position=positions)
    npt.assert_array_equal(marker.mean_position, np.array([2.0, 3.0, 4.0, 1.0]))

    # Test with invalid shape
    with pytest.raises(
        RuntimeError,
        match=r"The marker must be a np.ndarray of shape \(3,\), \(3, x\) \(4,\) or \(4, x\), but received: \(3, 3, 3\)",
    ):
        marker.position = np.ones((3, 3, 3))
        marker.mean_position


def test_marker_real_to_biomod():
    # Create a marker
    position = np.array([[1.0], [2.0], [3.0], [1.0]])
    marker = MarkerReal(
        name="test_marker", parent_name="segment1", position=position, is_technical=True, is_anatomical=False
    )

    # Generate biomod string
    biomod_str = marker.to_biomod()

    # Check the content
    expected_str = (
        "marker\ttest_marker\n"
        "\tparent\tsegment1\n"
        "\tposition\t1.00000000\t2.00000000\t3.00000000\n"
        "\ttechnical\t1\n"
        "\tanatomical\t0\n"
        "endmarker\n"
    )
    assert biomod_str == expected_str


def test_marker_real_arithmetic():
    # Create markers
    position1 = np.array([[1.0], [2.0], [3.0], [1.0]])
    marker1 = MarkerReal(name="marker1", parent_name="segment1", position=position1)

    position2 = np.array([[2.0], [3.0], [4.0], [1.0]])
    marker2 = MarkerReal(name="marker2", parent_name="segment1", position=position2)

    # Test addition with another marker
    result = marker1 + marker2
    npt.assert_array_equal(result.position, np.array([[3.0], [5.0], [7.0], [1.0]]))

    # Test addition with a numpy array
    offset = np.array([0.5, 0.5, 0.5, 0])
    result = marker1 + offset
    npt.assert_array_equal(result.position, np.array([[1.5], [2.5], [3.5], [1.0]]))

    # Test addition with a tuple
    result = marker1 + (0.5, 0.5, 0.5)
    npt.assert_array_equal(result.position, np.array([[1.5], [2.5], [3.5], [1.0]]))

    # Test subtraction with another marker
    result = marker2 - marker1
    npt.assert_array_equal(result.position, np.array([[1.0], [1.0], [1.0], [1.0]]))

    # Test subtraction with a numpy array
    result = marker1 - offset
    npt.assert_array_equal(result.position, np.array([[0.5], [1.5], [2.5], [1.0]]))

    # Test subtraction with a tuple
    result = marker1 - (0.5, 0.5, 0.5)
    npt.assert_array_equal(result.position, np.array([[0.5], [1.5], [2.5], [1.0]]))

    # Test unsupported operation
    with pytest.raises(NotImplementedError, match=r"The addition for \<class \'str\'\> is not implemented"):
        marker1 + "invalid"


# ------- MeshFileReal ------- #
def test_init_mesh_file_real():
    # Test initialization with minimal parameters
    mesh_file = MeshFileReal(mesh_file_name="test.obj")
    assert mesh_file.mesh_file_name == "test.obj"
    assert mesh_file.mesh_color is None
    npt.assert_array_equal(mesh_file.mesh_scale, np.ones((4, 1)))
    npt.assert_array_equal(mesh_file.mesh_rotation, np.zeros((4, 1)))
    npt.assert_array_equal(mesh_file.mesh_translation, np.zeros((4, 1)))

    # Test initialization with all parameters
    mesh_color = np.array([1.0, 0.0, 0.0])
    mesh_scale = np.array([2.0, 2.0, 2.0])
    mesh_rotation = np.array([0.1, 0.2, 0.3])
    mesh_translation = np.array([1.0, 2.0, 3.0])

    mesh_file = MeshFileReal(
        mesh_file_name="test.obj",
        mesh_color=mesh_color,
        mesh_scale=mesh_scale,
        mesh_rotation=mesh_rotation,
        mesh_translation=mesh_translation,
    )

    assert mesh_file.mesh_file_name == "test.obj"
    npt.assert_array_equal(mesh_file.mesh_color, mesh_color)
    npt.assert_array_equal(mesh_file.mesh_scale, np.array([[2.0], [2.0], [2.0], [1.0]]))
    npt.assert_array_equal(mesh_file.mesh_rotation, np.array([[0.1], [0.2], [0.3], [1.0]]))
    npt.assert_array_equal(mesh_file.mesh_translation, np.array([[1.0], [2.0], [3.0], [1.0]]))


def test_mesh_file_real_to_biomod():
    # Create a mesh file
    mesh_file = MeshFileReal(
        mesh_file_name="test.obj",
        mesh_color=np.array([1.0, 0.0, 0.0]),
        mesh_scale=np.array([2.0, 2.0, 2.0]),
        mesh_rotation=np.array([0.1, 0.2, 0.3]),
        mesh_translation=np.array([1.0, 2.0, 3.0]),
    )

    # Generate biomod string
    biomod_str = mesh_file.to_biomod()

    # Check the content
    expected_str = (
        "\tmeshfile\ttest.obj\n"
        "\tmeshcolor\t1.0\t0.0\t0.0\n"
        "\tmeshscale\t2.0\t2.0\t2.0\n"
        "\tmeshrt\t0.1\t0.2\t0.3\txyz\t1.0\t2.0\t3.0\n"
    )
    assert biomod_str == expected_str

    # Test with missing rotation or translation
    mesh_file = MeshFileReal(
        mesh_file_name="test.obj",
        mesh_color=np.array([1.0, 0.0, 0.0]),
        mesh_scale=np.array([2.0, 2.0, 2.0]),
        mesh_rotation=np.array([0.1, 0.2, 0.3]),
        mesh_translation=None,
    )

    expected_str = (
        "\tmeshfile\ttest.obj\n"
        "\tmeshcolor\t1.0\t0.0\t0.0\n"
        "\tmeshscale\t2.0\t2.0\t2.0\n"
        "\tmeshrt\t0.1\t0.2\t0.3\txyz\t0.0\t0.0\t0.0\n"
    )
    biomod_str = mesh_file.to_biomod()
    assert biomod_str == expected_str


# ------- AxisReal ------- #
def test_init_axis_real():
    # Create markers for the axis
    start_marker = MarkerReal(name="start", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    end_marker = MarkerReal(name="end", parent_name="segment1", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    # Test initialization
    axis = AxisReal(name=AxisReal.Name.X, start=start_marker, end=end_marker)

    assert axis.name == AxisReal.Name.X
    assert axis.start_point == start_marker
    assert axis.end_point == end_marker


def test_axis_real_axis():
    # Create markers for the axis
    start_marker = MarkerReal(name="start", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    end_marker = MarkerReal(name="end", parent_name="segment1", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    # Create axis
    axis = AxisReal(name=AxisReal.Name.X, start=start_marker, end=end_marker)

    # Test axis vector calculation
    axis_vector = axis.axis()
    npt.assert_array_equal(axis_vector, np.array([[1.0], [0.0], [0.0], [0.0]]))

    # Test with different positions
    start_marker = MarkerReal(name="start", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    end_marker = MarkerReal(name="end", parent_name="segment1", position=np.array([[0.0], [1.0], [0.0], [1.0]]))

    axis = AxisReal(name=AxisReal.Name.Y, start=start_marker, end=end_marker)
    axis_vector = axis.axis()
    npt.assert_array_equal(axis_vector, np.array([[0.0], [1.0], [0.0], [0.0]]))


# ------- SegmentCoordinateSystemReal ------- #
def test_init_segment_coordinate_system_real():
    # Test initialization with default values
    scs = SegmentCoordinateSystemReal()
    assert isinstance(scs.scs, RotoTransMatrix)
    assert scs.is_in_global is True
    assert scs.is_in_local is False

    # Test initialization with custom values
    rt_matrix = RotoTransMatrix()
    rt_matrix.from_euler_angles_and_translation(
        angle_sequence="xyz", angles=np.array([0.1, 0.2, 0.3]), translation=np.array([1.0, 2.0, 3.0])
    )

    scs = SegmentCoordinateSystemReal(scs=rt_matrix, is_scs_local=True)

    assert scs.scs == rt_matrix
    assert scs.is_in_global is False
    assert scs.is_in_local is True


def test_segment_coordinate_system_real_from_rt_matrix():
    # Create an RT matrix
    rt_matrix = np.eye(4)
    rt_matrix[:3, 3] = [1.0, 2.0, 3.0]

    # Create SCS from RT matrix
    scs = SegmentCoordinateSystemReal.from_rt_matrix(rt_matrix=rt_matrix, is_scs_local=True)

    # Test the resulting SCS
    assert isinstance(scs, SegmentCoordinateSystemReal)
    assert scs.is_in_global is False
    npt.assert_array_equal(scs.scs.rt_matrix, rt_matrix)


def test_segment_coordinate_system_real_from_euler_and_translation():
    # Create Euler angles and translation
    angles = np.array([0.1, 0.2, 0.3])
    translation = np.array([1.0, 2.0, 3.0])

    # Create SCS from Euler angles and translation
    scs = SegmentCoordinateSystemReal.from_euler_and_translation(
        angles=angles, angle_sequence="xyz", translation=translation, is_scs_local=True
    )

    # Test the resulting SCS
    assert isinstance(scs, SegmentCoordinateSystemReal)
    assert scs.is_in_global is False

    # The exact RT matrix would need to be calculated for comparison
    # This is a simplified check
    npt.assert_array_equal(scs.scs.rt_matrix[3, :], np.array([0.0, 0.0, 0.0, 1.0]))


def test_segment_coordinate_system_real_inverse():
    # Create an RT matrix
    rt_matrix = np.eye(4)
    rt_matrix[:3, 3] = [1.0, 2.0, 3.0]

    # Create SCS from RT matrix
    scs = SegmentCoordinateSystemReal.from_rt_matrix(rt_matrix=rt_matrix)

    # Get inverse
    inverse_scs = scs.inverse

    # Test the inverse
    assert isinstance(inverse_scs, SegmentCoordinateSystemReal)

    # The inverse of a translation matrix should have negative translations
    expected_inverse = np.eye(4)
    expected_inverse[:3, 3] = [-1.0, -2.0, -3.0]
    npt.assert_array_equal(inverse_scs.scs.rt_matrix, expected_inverse)


def test_segment_coordinate_system_real_to_biomod():
    # Create an RT matrix
    rt_matrix = np.eye(4)
    rt_matrix[:3, 3] = [1.0, 2.0, 3.0]

    # Create SCS from RT matrix
    scs = SegmentCoordinateSystemReal.from_rt_matrix(rt_matrix=rt_matrix)

    # Generate biomod string
    biomod_str = scs.to_biomod()

    # Check the content
    expected_str = (
        "\tRTinMatrix\t1\n"
        "\tRT\n"
        "\t\t1.000000\t0.000000\t0.000000\t1.000000\n"
        "\t\t0.000000\t1.000000\t0.000000\t2.000000\n"
        "\t\t0.000000\t0.000000\t1.000000\t3.000000\n"
        "\t\t0.000000\t0.000000\t0.000000\t1.000000\n"
    )
    assert biomod_str == expected_str


# ------- ContactReal ------- #
def test_init_contact_real():
    # Test initialization with minimal parameters
    contact = ContactReal(name="test_contact", parent_name="segment1")
    assert contact.name == "test_contact"
    assert contact.parent_name == "segment1"
    assert contact.axis is None

    # Test initialization with all parameters
    position = np.array([[1.0], [2.0], [3.0], [1.0]])
    contact = ContactReal(name="test_contact", parent_name="segment1", position=position, axis=Translations.XYZ)

    assert contact.name == "test_contact"
    assert contact.parent_name == "segment1"
    npt.assert_array_equal(contact.position, position)
    assert contact.axis == Translations.XYZ


def test_contact_real_to_biomod():
    # Create a contact
    position = np.array([[1.0], [2.0], [3.0], [1.0]])
    contact = ContactReal(name="test_contact", parent_name="segment1", position=position, axis=Translations.XYZ)

    # Generate biomod string
    biomod_str = contact.to_biomod()

    # Check the content
    expected_str = (
        "contact\ttest_contact\n" "\tparent\tsegment1\n" "\tposition\t1.0\t2.0\t3.0\n" "\taxis\txyz\n" "endcontact\n"
    )
    assert biomod_str == expected_str


# ------- MeshReal ------- #
def test_init_mesh_real():
    # Test initialization with no positions
    mesh = MeshReal()
    assert mesh.positions.shape == (4, 0)

    # Test initialization with positions
    positions = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [1.0, 1.0, 1.0]])
    mesh = MeshReal(positions=positions)
    npt.assert_array_equal(mesh.positions, positions)


def test_mesh_real_add_positions():
    # Create a mesh with initial positions
    initial_positions = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1.0, 1.0]])
    mesh = MeshReal(positions=initial_positions)

    # Add more positions
    additional_positions = np.array([[3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [1.0, 1.0]])
    mesh.add_positions(additional_positions)

    # Check the result
    expected_positions = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [1.0, 1.0, 1.0, 1.0]]
    )
    npt.assert_array_equal(mesh.positions, expected_positions)


def test_mesh_real_to_biomod():
    # Create a mesh with positions
    positions = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1.0, 1.0]])
    mesh = MeshReal(positions=positions)

    # Generate biomod string
    biomod_str = mesh.to_biomod()

    # Check the content
    expected_str = "\tmesh\t1.000000\t2.000000\t3.000000\n" "\tmesh\t2.000000\t3.000000\t4.000000\n"
    assert biomod_str == expected_str

    # Test with nan values
    mesh.positions = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0], [1.0, 1.0]])

    with pytest.raises(RuntimeError):
        mesh.to_biomod()


# ------- InertialMeasurementUnitReal ------- #
def test_init_inertial_measurement_unit_real():
    # Test initialization with minimal parameters
    imu = InertialMeasurementUnitReal(name="test_imu", parent_name="segment1")
    assert imu.name == "test_imu"
    assert imu.parent_name == "segment1"
    assert isinstance(imu.scs, RotoTransMatrix)
    assert imu.is_technical is True
    assert imu.is_anatomical is False

    # Test initialization with all parameters
    rt_matrix = RotoTransMatrix()
    rt_matrix.from_euler_angles_and_translation(
        angle_sequence="xyz", angles=np.array([0.1, 0.2, 0.3]), translation=np.array([1.0, 2.0, 3.0])
    )

    imu = InertialMeasurementUnitReal(
        name="test_imu", parent_name="segment1", scs=rt_matrix, is_technical=False, is_anatomical=True
    )

    assert imu.name == "test_imu"
    assert imu.parent_name == "segment1"
    assert imu.scs == rt_matrix
    assert imu.is_technical is False
    assert imu.is_anatomical is True


def test_inertial_measurement_unit_real_to_biomod():
    # Create an IMU
    rt_matrix = RotoTransMatrix()
    rt_matrix.from_euler_angles_and_translation(
        angle_sequence="xyz", angles=np.array([0.1, 0.2, 0.3]), translation=np.array([1.0, 2.0, 3.0])
    )

    imu = InertialMeasurementUnitReal(
        name="test_imu", parent_name="segment1", scs=rt_matrix, is_technical=True, is_anatomical=False
    )

    # Generate biomod string
    biomod_str = imu.to_biomod()

    # Check the content (simplified check)
    assert "imu\ttest_imu" in biomod_str
    assert "\tparent\tsegment1" in biomod_str
    assert "\tRTinMatrix\t1" in biomod_str
    assert "\ttechnical\t1" in biomod_str
    assert "\tanatomical\t0" in biomod_str
    assert "endimu" in biomod_str


# ------- SegmentReal ------- #
def test_init_segment_real():
    # Test initialization with minimal parameters
    segment = SegmentReal(name="test_segment")
    assert segment.name == "test_segment"
    assert segment.parent_name == "base"
    assert isinstance(segment.segment_coordinate_system, SegmentCoordinateSystemReal)
    assert segment.translations == Translations.NONE
    assert segment.rotations == Rotations.NONE
    assert len(segment.dof_names) == 0
    assert segment.q_ranges is None
    assert segment.qdot_ranges is None
    assert len(segment.markers) == 0
    assert len(segment.contacts) == 0
    assert len(segment.imus) == 0
    assert segment.inertia_parameters is None
    assert segment.mesh is None
    assert segment.mesh_file is None

    # Test initialization with custom parameters
    scs = SegmentCoordinateSystemReal()
    q_ranges = RangeOfMotion(Ranges.Q, [-1, -1, -1], [1, 1, 1])
    qdot_ranges = RangeOfMotion(Ranges.Qdot, [-10, -10, -10], [10, 10, 10])

    segment = SegmentReal(
        name="test_segment",
        parent_name="parent_segment",
        segment_coordinate_system=scs,
        translations=Translations.XYZ,
        rotations=Rotations.XYZ,
        dof_names=["dof1", "dof2", "dof3", "dof4", "dof5", "dof6"],
        q_ranges=q_ranges,
        qdot_ranges=qdot_ranges,
    )

    assert segment.name == "test_segment"
    assert segment.parent_name == "parent_segment"
    assert segment.segment_coordinate_system == scs
    assert segment.translations == Translations.XYZ
    assert segment.rotations == Rotations.XYZ
    assert segment.dof_names == ["dof1", "dof2", "dof3", "dof4", "dof5", "dof6"]
    assert segment.q_ranges == q_ranges
    assert segment.qdot_ranges == qdot_ranges


def test_segment_real_dof_names_auto_generation():
    # Test auto-generation of dof_names
    segment = SegmentReal(name="test_segment", translations=Translations.XY, rotations=Rotations.Z)

    expected_dof_names = ["test_segment_transX", "test_segment_transY", "test_segment_rotZ"]
    assert segment.dof_names == expected_dof_names

    # Test mismatch between dof_names length and actual DoFs
    with pytest.raises(RuntimeError):
        SegmentReal(
            name="test_segment",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            dof_names=["dof1"],  # Only one name for 6 DoFs
        )


def test_segment_real_add_remove_marker():
    # Create a segment
    segment = SegmentReal(name="test_segment")

    # Create a marker
    marker = MarkerReal(name="test_marker", parent_name=None)

    # Add marker to segment
    segment.add_marker(marker)

    # Verify marker was added and parent_name was set
    assert len(segment.markers) == 1
    assert marker.parent_name == "test_segment"

    # Create a marker with matching parent_name
    marker2 = MarkerReal(name="test_marker2", parent_name="test_segment")
    segment.add_marker(marker2)
    assert len(segment.markers) == 2

    # Create a marker with non-matching parent_name
    marker3 = MarkerReal(name="test_marker3", parent_name="other_segment")
    with pytest.raises(ValueError):
        segment.add_marker(marker3)

    # Remove a marker
    segment.remove_marker(marker.name)
    assert len(segment.markers) == 1
    assert segment.markers[0].name == "test_marker2"


def test_segment_real_add_remove_contact():
    # Create a segment
    segment = SegmentReal(name="test_segment")

    # Create a contact
    contact = ContactReal(name="test_contact", parent_name="test_segment")

    # Add contact to segment
    segment.add_contact(contact)

    # Verify contact was added
    assert len(segment.contacts) == 1
    assert contact.parent_name == "test_segment"

    # Create a contact with no parent_name
    contact2 = ContactReal(name="test_contact2", parent_name=None)
    with pytest.raises(ValueError):
        segment.add_contact(contact2)

    # Create a contact with non-matching parent_name
    contact3 = ContactReal(name="test_contact3", parent_name="other_segment")
    with pytest.raises(ValueError):
        segment.add_contact(contact3)

    # Remove a contact
    segment.remove_contact(contact.name)
    assert len(segment.contacts) == 0


def test_segment_real_add_remove_imu():
    # Create a segment
    segment = SegmentReal(name="test_segment")

    # Create an IMU
    imu = InertialMeasurementUnitReal(name="test_imu", parent_name="test_segment")

    # Add IMU to segment
    segment.add_imu(imu)

    # Verify IMU was added
    assert len(segment.imus) == 1
    assert imu.parent_name == "test_segment"

    # Create an IMU with no parent_name
    imu2 = InertialMeasurementUnitReal(name="test_imu2", parent_name=None)
    with pytest.raises(RuntimeError):
        segment.add_imu(imu2)

    # Remove an IMU
    segment.remove_imu(imu.name)
    assert len(segment.imus) == 0


def test_segment_real_rt_from_local_q():
    # Create a segment with translations and rotations
    segment = SegmentReal(name="test_segment", translations=Translations.XYZ, rotations=Rotations.XYZ)

    # Test with correct q vector size
    local_q = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    rt = segment.rt_from_local_q(local_q)

    assert isinstance(rt, RotoTransMatrix)

    # Test with incorrect q vector size
    with pytest.raises(RuntimeError):
        segment.rt_from_local_q(np.array([1.0, 2.0]))


def test_segment_real_to_biomod():
    # Create a segment with various components
    segment = SegmentReal(
        name="test_segment", parent_name="parent_segment", translations=Translations.XYZ, rotations=Rotations.XYZ
    )

    # Add a marker
    marker = MarkerReal(name="test_marker", parent_name="test_segment", position=np.array([[1.0], [2.0], [3.0], [1.0]]))
    segment.add_marker(marker)

    # Add a contact
    contact = ContactReal(
        name="test_contact", parent_name="test_segment", position=np.array([[1.0], [2.0], [3.0], [1.0]])
    )
    segment.add_contact(contact)

    # Exporting without an axis is not allowed
    with pytest.raises(RuntimeError, match="The axis of the contact must be defined before exporting to biomod."):
        segment.to_biomod(with_mesh=False)

    # Generate biomod string
    segment.remove_contact("test_contact")
    contact.axis = Translations.XYZ  # Set an axis for the contact
    segment.add_contact(contact)
    biomod_str = segment.to_biomod(with_mesh=True)

    # Check the content (simplified check)
    assert "segment\ttest_segment" in biomod_str
    assert "\tparent\tparent_segment" in biomod_str
    assert "\ttranslations\txyz" in biomod_str
    assert "\trotations\txyz" in biomod_str
    assert "endsegment" in biomod_str
    assert "marker\ttest_marker" in biomod_str
    assert "contact\ttest_contact" in biomod_str


# ------- ViaPointReal ------- #
def test_init_via_point_real():
    # Test initialization with minimal parameters
    via_point = ViaPointReal(name="test_via_point", parent_name="segment1")

    assert via_point.name == "test_via_point"
    assert via_point.parent_name == "segment1"
    assert via_point.muscle_name is None
    assert via_point.muscle_group is None
    npt.assert_almost_equal(via_point.position, np.ndarray((4, 0)))
    assert via_point.condition is None
    assert via_point.movement is None

    # Test initialization with all parameters
    position = np.array([[1.0], [2.0], [3.0], [1.0]])
    via_point = ViaPointReal(
        name="test_via_point", parent_name="segment1", muscle_name="muscle1", muscle_group="group1", position=position
    )

    assert via_point.name == "test_via_point"
    assert via_point.parent_name == "segment1"
    assert via_point.muscle_name == "muscle1"
    assert via_point.muscle_group == "group1"
    npt.assert_array_equal(via_point.position, position)

    # Test with position as a tuple
    via_point = ViaPointReal(name="test_via_point", parent_name="segment1", position=np.array([[1.0], [2.0], [3.0]]))
    npt.assert_array_equal(via_point.position, np.array([[1.0], [2.0], [3.0], [1.0]]))

    # Test with both position and movement (should raise error)
    with pytest.raises(RuntimeError, match="You can only have either a position or a movement, not both."):
        ViaPointReal(
            name="test_via_point",
            parent_name="segment1",
            position=(1.0, 2.0, 3.0),
            movement=object(),  # Mock movement object
        )

    # Test with both condition and movement (should raise error)
    with pytest.raises(RuntimeError, match="You can only have either a condition or a movement, not both."):
        ViaPointReal(
            name="test_via_point",
            parent_name="segment1",
            condition=object(),  # Mock condition object
            movement=object(),  # Mock movement object
        )


def test_via_point_real_to_biomod():
    # Create a via point
    position = np.array([[1.0], [2.0], [3.0], [1.0]])
    via_point = ViaPointReal(
        name="test_via_point", parent_name="segment1", muscle_name="muscle1", muscle_group="group1", position=position
    )

    # Generate biomod string
    biomod_str = via_point.to_biomod()

    # Check the content
    expected_str = (
        "viapoint\ttest_via_point\n"
        "\tparent\tsegment1\n"
        "\tmuscle\tmuscle1\n"
        "\tmusclegroup\tgroup1\n"
        "\tposition\t1.0\t2.0\t3.0\n"
        "endviapoint\n"
        "\n\n"
    )
    assert biomod_str == expected_str

    # Test with condition (should return warning)
    via_point.condition = object()  # Mock condition object
    biomod_str = via_point.to_biomod()
    assert "WARNING: biorbd doe not support conditional via points" in biomod_str

    # Test with movement (should return warning)
    via_point.condition = None
    via_point.movement = object()  # Mock movement object
    biomod_str = via_point.to_biomod()
    assert "WARNING: biorbd doe not support moving via points" in biomod_str


# ------- MuscleReal ------- #
def test_init_muscle_real():
    # Create origin and insertion via points
    origin = ViaPointReal(name="origin", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))

    insertion = ViaPointReal(name="insertion", parent_name="segment2", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    # Test initialization with minimal parameters
    muscle = MuscleReal(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="group1",
        origin_position=origin,
        insertion_position=insertion,
    )

    assert muscle.name == "test_muscle"
    assert muscle.muscle_type == MuscleType.HILL
    assert muscle.state_type == MuscleStateType.DEGROOTE
    assert muscle.muscle_group == "group1"
    assert muscle.origin_position == origin
    assert muscle.insertion_position == insertion
    assert muscle.optimal_length is None
    assert muscle.maximal_force is None
    assert muscle.tendon_slack_length is None
    assert muscle.pennation_angle is None
    assert muscle.maximal_velocity is None
    assert muscle.maximal_excitation is None
    assert len(muscle.via_points) == 0

    # Test initialization with all parameters
    muscle = MuscleReal(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="group1",
        origin_position=origin,
        insertion_position=insertion,
        optimal_length=0.1,
        maximal_force=100.0,
        tendon_slack_length=0.2,
        pennation_angle=0.1,
        maximal_velocity=10.0,
        maximal_excitation=1.0,
    )

    assert muscle.name == "test_muscle"
    assert muscle.muscle_type == MuscleType.HILL
    assert muscle.state_type == MuscleStateType.DEGROOTE
    assert muscle.muscle_group == "group1"
    assert muscle.origin_position == origin
    assert muscle.insertion_position == insertion
    assert muscle.optimal_length == 0.1
    assert muscle.maximal_force == 100.0
    assert muscle.tendon_slack_length == 0.2
    assert muscle.pennation_angle == 0.1
    assert muscle.maximal_velocity == 10.0
    assert muscle.maximal_excitation == 1.0

    # Test with invalid parameters
    with pytest.raises(ValueError, match="The optimal length of the muscle must be greater than 0."):
        MuscleReal(
            name="test_muscle",
            muscle_type=MuscleType.HILL,
            state_type=MuscleStateType.DEGROOTE,
            muscle_group="group1",
            origin_position=origin,
            insertion_position=insertion,
            optimal_length=0.0,  # Invalid value
        )

    with pytest.raises(ValueError, match="The maximal force of the muscle must be greater than 0."):
        MuscleReal(
            name="test_muscle",
            muscle_type=MuscleType.HILL,
            state_type=MuscleStateType.DEGROOTE,
            muscle_group="group1",
            origin_position=origin,
            insertion_position=insertion,
            maximal_force=0.0,  # Invalid value
        )


def test_muscle_real_add_remove_via_point():
    # Create origin and insertion via points
    origin = ViaPointReal(name="origin", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))

    insertion = ViaPointReal(name="insertion", parent_name="segment2", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    # Create a muscle
    muscle = MuscleReal(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="group1",
        origin_position=origin,
        insertion_position=insertion,
    )

    # Create a via point
    via_point = ViaPointReal(name="via_point", parent_name="segment1", position=np.array([[0.5], [0.0], [0.0], [1.0]]))

    # Add via point to muscle
    muscle.add_via_point(via_point)

    # Verify via point was added and muscle_name was set
    assert len(muscle.via_points) == 1
    assert via_point.muscle_name == "test_muscle"

    # Create a via point with matching muscle_name
    via_point2 = ViaPointReal(
        name="via_point2",
        parent_name="segment1",
        muscle_name="test_muscle",
        position=np.array([[0.7], [0.0], [0.0], [1.0]]),
    )
    muscle.add_via_point(via_point2)
    assert len(muscle.via_points) == 2

    # Create a via point with non-matching muscle_name
    via_point3 = ViaPointReal(
        name="via_point3",
        parent_name="segment1",
        muscle_name="other_muscle",
        position=np.array([[0.8], [0.0], [0.0], [1.0]]),
    )
    with pytest.raises(ValueError, match="The via points's muscle .* should be the same as the muscle's name"):
        muscle.add_via_point(via_point3)

    # Remove a via point
    muscle.remove_via_point("via_point")
    assert len(muscle.via_points) == 1
    assert muscle.via_points[0].name == "via_point2"


def test_muscle_real_to_biomod():
    # Create origin and insertion via points
    origin = ViaPointReal(name="origin", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))

    insertion = ViaPointReal(name="insertion", parent_name="segment2", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    # Create a muscle
    muscle = MuscleReal(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="group1",
        origin_position=origin,
        insertion_position=insertion,
        optimal_length=0.1,
        maximal_force=100.0,
        tendon_slack_length=0.2,
        pennation_angle=0.1,
        maximal_velocity=10.0,
        maximal_excitation=1.0,
    )

    # Add a via point
    via_point = ViaPointReal(name="via_point", parent_name="segment1", position=np.array([[0.5], [0.0], [0.0], [1.0]]))
    muscle.add_via_point(via_point)

    # Generate biomod string
    biomod_str = muscle.to_biomod()

    # Check the content
    assert "muscle\ttest_muscle" in biomod_str
    assert "\ttype\thill" in biomod_str
    assert "\tstatetype\tdegroote" in biomod_str
    assert "\tmusclegroup\tgroup1" in biomod_str
    assert "\toriginposition\t0.0\t0.0\t0.0" in biomod_str
    assert "\tinsertionposition\t1.0\t0.0\t0.0" in biomod_str
    assert "\toptimallength\t0.1000" in biomod_str
    assert "\tmaximalforce\t100.0000" in biomod_str
    assert "\ttendonslacklength\t0.2000" in biomod_str
    assert "\tpennationangle\t0.1000" in biomod_str
    assert "\tmaxvelocity\t10.0000" in biomod_str
    assert "\tmaxexcitation\t1.0000" in biomod_str
    assert "endmuscle" in biomod_str
    assert "viapoint\tvia_point" in biomod_str


# ------- MuscleGroupReal ------- #
def test_init_muscle_group_real():
    # Test initialization
    muscle_group = MuscleGroupReal(name="test_group", origin_parent_name="segment1", insertion_parent_name="segment2")

    assert muscle_group.name == "test_group"
    assert muscle_group.origin_parent_name == "segment1"
    assert muscle_group.insertion_parent_name == "segment2"
    assert len(muscle_group.muscles) == 0

    # Test with same origin and insertion parent names (should raise error)
    with pytest.raises(ValueError, match="The origin and insertion parent names cannot be the same."):
        MuscleGroupReal(name="test_group", origin_parent_name="segment1", insertion_parent_name="segment1")


def test_muscle_group_real_add_remove_muscle():
    # Create a muscle group
    muscle_group = MuscleGroupReal(name="test_group", origin_parent_name="segment1", insertion_parent_name="segment2")

    # Create origin and insertion via points
    origin = ViaPointReal(name="origin", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))

    insertion = ViaPointReal(name="insertion", parent_name="segment2", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    # Create a muscle with no muscle_group
    muscle = MuscleReal(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group=None,
        origin_position=origin,
        insertion_position=insertion,
    )

    # Add muscle to group
    muscle_group.add_muscle(muscle)

    # Verify muscle was added and muscle_group was set
    assert len(muscle_group.muscles) == 1
    assert muscle.muscle_group == "test_group"

    # Create a muscle with matching muscle_group
    origin = ViaPointReal(name="origin", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    insertion = ViaPointReal(name="insertion", parent_name="segment2", position=np.array([[1.0], [0.0], [0.0], [1.0]]))
    muscle2 = MuscleReal(
        name="test_muscle2",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="test_group",
        origin_position=origin,
        insertion_position=insertion,
    )
    muscle_group.add_muscle(muscle2)
    assert len(muscle_group.muscles) == 2

    # Create a muscle with non-matching muscle_group
    origin = ViaPointReal(name="origin", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    insertion = ViaPointReal(name="insertion", parent_name="segment2", position=np.array([[1.0], [0.0], [0.0], [1.0]]))
    muscle3 = MuscleReal(
        name="test_muscle3",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="other_group",
        origin_position=origin,
        insertion_position=insertion,
    )
    with pytest.raises(ValueError, match="The muscle's muscle_group should be the same as the 'key'"):
        muscle_group.add_muscle(muscle3)

    # Remove a muscle
    muscle_group.remove_muscle("test_muscle")
    assert len(muscle_group.muscles) == 1
    assert muscle_group.muscles[0].name == "test_muscle2"


def test_muscle_group_real_properties():
    # Create a muscle group
    muscle_group = MuscleGroupReal(name="test_group", origin_parent_name="segment1", insertion_parent_name="segment2")

    # Create origin and insertion via points
    origin = ViaPointReal(name="origin", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))

    insertion = ViaPointReal(name="insertion", parent_name="segment2", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    # Create muscles
    muscle1 = MuscleReal(
        name="test_muscle1",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group=None,
        origin_position=origin,
        insertion_position=insertion,
    )

    origin = ViaPointReal(name="origin", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    insertion = ViaPointReal(name="insertion", parent_name="segment2", position=np.array([[1.0], [0.0], [0.0], [1.0]]))
    muscle2 = MuscleReal(
        name="test_muscle2",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group=None,
        origin_position=origin,
        insertion_position=insertion,
    )

    # Add muscles to group
    muscle_group.add_muscle(muscle1)
    muscle_group.add_muscle(muscle2)

    # Test properties
    assert muscle_group.nb_muscles == 2
    assert muscle_group.muscle_names == ["test_muscle1", "test_muscle2"]


def test_muscle_group_real_to_biomod():
    # Create a muscle group
    muscle_group = MuscleGroupReal(name="test_group", origin_parent_name="segment1", insertion_parent_name="segment2")

    # Create origin and insertion via points
    origin = ViaPointReal(name="origin", parent_name="segment1", position=np.array([[0.0], [0.0], [0.0], [1.0]]))

    insertion = ViaPointReal(name="insertion", parent_name="segment2", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    # Create a muscle
    muscle = MuscleReal(
        name="test_muscle",
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group=None,
        origin_position=origin,
        insertion_position=insertion,
        maximal_force=100.0,
    )

    # Add muscle to group
    muscle_group.add_muscle(muscle)

    # Generate biomod string
    biomod_str = muscle_group.to_biomod()

    # Check the content
    assert "musclegroup\ttest_group" in biomod_str
    assert "\tOriginParent\tsegment1" in biomod_str
    assert "\tInsertionParent\tsegment2" in biomod_str
    assert "endmusclegroup" in biomod_str
    assert "muscle\ttest_muscle" in biomod_str
