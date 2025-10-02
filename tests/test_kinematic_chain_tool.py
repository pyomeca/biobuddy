import pytest
import numpy as np
import numpy.testing as npt
from deepdiff import DeepDiff

from biobuddy import (
    ModifyKinematicChainTool,
    ChangeFirstSegment,
    SegmentReal,
    Translations,
    Rotations,
    RotoTransMatrix,
    InertiaParametersReal,
    InertialMeasurementUnitReal,
)
from test_utils import create_simple_model
from biobuddy.utils.linear_algebra import point_from_local_to_global


def test_change_first_segment_init():
    """Test the initialization of ChangeFirstSegment"""

    # Without merged_origin_name
    segment_merge = ChangeFirstSegment(first_segment_name="UPPER_ARMS")
    assert segment_merge.first_segment_name == "UPPER_ARMS"
    assert segment_merge.new_segment_name == "LINK"

    segment_merge = ChangeFirstSegment(first_segment_name="UPPER_ARMS", new_segment_name="PELVIS")
    assert segment_merge.new_segment_name == "PELVIS"


def test_kinematic_chain_tool_init():
    """
    Test the initialization of the ModifyKinematicChainTool
    """
    original_model = create_simple_model()

    merge_tool = ModifyKinematicChainTool(original_model)
    assert DeepDiff(merge_tool.original_model, original_model, ignore_order=True) == {}


def test_kinematic_chain_tool_modify():
    """Test the merge functionality of ModifyKinematicChainTool"""

    original_model = create_simple_model()

    kinematic_chain_modifier = ModifyKinematicChainTool(original_model)
    kinematic_chain_modifier.add(ChangeFirstSegment(first_segment_name="child", new_segment_name="PELVIS"))
    modified_model = kinematic_chain_modifier.modify()

    # Check the number of elements
    assert modified_model.nb_segments == 4
    assert modified_model.nb_muscles == 1
    assert modified_model.nb_via_points == 1
    assert modified_model.nb_markers == 4
    assert modified_model.nb_contacts == 2

    # Check the segment's name
    assert modified_model.segment_names == ["root", "child", "parent", "PELVIS"]

    # Check the segment's parent name
    assert modified_model.segments["child"].parent_name == "root"
    assert modified_model.segments["parent"].parent_name == "child"
    assert modified_model.segments["PELVIS"].parent_name == "parent"

    # Check the segment's dofs
    assert modified_model.segments["child"].translations == Translations.XYZ
    assert modified_model.segments["child"].rotations == Rotations.XYZ
    assert modified_model.segments["parent"].translations == Translations.NONE
    assert modified_model.segments["parent"].rotations == original_model.segments["child"].rotations
    assert modified_model.segments["PELVIS"].translations == Translations.NONE
    assert modified_model.segments["PELVIS"].rotations == Rotations.NONE

    # TODO: another segment should be added to test ranges
    assert modified_model.segments["child"].q_ranges is None
    assert modified_model.segments["parent"].q_ranges is None
    assert modified_model.segments["PELVIS"].q_ranges is None

    # Check the segment's scs
    expected_new_root_origin = RotoTransMatrix()
    expected_new_root_origin.translation = original_model.segment_com_in_global("child")
    npt.assert_almost_equal(
        modified_model.segments["child"].segment_coordinate_system.scs.rt_matrix, expected_new_root_origin.rt_matrix
    )
    npt.assert_almost_equal(
        modified_model.segment_coordinate_system_in_global("parent").rt_matrix,
        original_model.segment_coordinate_system_in_global("child").rt_matrix,
    )
    npt.assert_almost_equal(
        modified_model.segment_coordinate_system_in_global("PELVIS").rt_matrix,
        original_model.segment_coordinate_system_in_global("parent").rt_matrix,
    )

    # Check the segment's mass
    npt.assert_almost_equal(
        modified_model.segments["child"].inertia_parameters.mass,
        original_model.segments["child"].inertia_parameters.mass,
    )
    npt.assert_almost_equal(
        modified_model.segments["parent"].inertia_parameters.mass,
        original_model.segments["parent"].inertia_parameters.mass,
    )
    npt.assert_almost_equal(modified_model.segments["PELVIS"].inertia_parameters.mass, 0)

    # Check that the segment's com did not move in the global
    npt.assert_almost_equal(
        modified_model.segments["child"].inertia_parameters.center_of_mass.reshape(
            4,
        ),
        np.array([0, 0, 0, 1]),
    )
    npt.assert_almost_equal(
        modified_model.segment_com_in_global("child"), original_model.segment_com_in_global("child")
    )
    npt.assert_almost_equal(
        modified_model.segments["parent"].inertia_parameters.center_of_mass.reshape(
            4,
        ),
        np.array([0, 0, -0.5, 1]),
    )
    npt.assert_almost_equal(
        modified_model.segment_com_in_global("parent"), original_model.segment_com_in_global("parent")
    )
    npt.assert_almost_equal(modified_model.total_com_in_global(), original_model.total_com_in_global())

    # Check the merged segment's inertia
    npt.assert_almost_equal(
        modified_model.segments["child"].inertia_parameters.inertia,
        original_model.segments["child"].inertia_parameters.inertia,
    )
    npt.assert_almost_equal(
        modified_model.segments["parent"].inertia_parameters.inertia,
        original_model.segments["parent"].inertia_parameters.inertia,
    )
    npt.assert_almost_equal(
        modified_model.segments["PELVIS"].inertia_parameters.inertia,
        np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )

    # Check the segment's mesh
    npt.assert_almost_equal(
        modified_model.segments["parent"].mesh.positions, np.array([[0.0, 0.2], [0.0, 0.1], [-1.0, -0.7], [1.0, 1.0]])
    )
    npt.assert_almost_equal(
        modified_model.segments["child"].mesh.positions, np.array([[0.0, 0.2], [-0.1, 0.0], [0.0, 0.3], [1.0, 1.0]])
    )
    assert modified_model.segments["PELVIS"].mesh is None

    # Check that the markers did not move in the global (but the order changed)
    npt.assert_almost_equal(
        modified_model.markers_in_global()[:, :, 0], original_model.markers_in_global()[:, [2, 3, 0, 1], 0]
    )

    # Check that the contacts did not move in the global (but the order changed)
    npt.assert_almost_equal(
        modified_model.contacts_in_global()[:, :, 0], original_model.contacts_in_global()[:, [1, 0], 0]
    )

    # Test Via point positions
    for i_muscle, muscle_name in enumerate(modified_model.muscle_names):
        npt.assert_almost_equal(
            modified_model.via_points_in_global(muscle_name)[:, :, 0],
            original_model.via_points_in_global(muscle_name)[:, :, 0],
        )

        # Origin
        origin_position_modified_global = (
            modified_model.muscle_groups["parent_to_child"]
            .muscles["muscle1"]
            .origin_position.position.reshape(
                4,
            )
        )
        origin_position_modified = point_from_local_to_global(
            origin_position_modified_global, modified_model.segment_coordinate_system_in_global("parent")
        )
        origin_position_original_global = (
            original_model.muscle_groups["parent_to_child"]
            .muscles["muscle1"]
            .origin_position.position.reshape(
                4,
            )
        )
        origin_position_original = point_from_local_to_global(
            origin_position_original_global, original_model.segment_coordinate_system_in_global("parent")
        )
        npt.assert_almost_equal(origin_position_modified, origin_position_original)

        # Insertion
        insertion_position_modified_global = (
            modified_model.muscle_groups["parent_to_child"]
            .muscles["muscle1"]
            .insertion_position.position.reshape(
                4,
            )
        )
        insertion_position_modified = point_from_local_to_global(
            insertion_position_modified_global, modified_model.segment_coordinate_system_in_global("child")
        )
        insertion_position_original_global = (
            original_model.muscle_groups["parent_to_child"]
            .muscles["muscle1"]
            .insertion_position.position.reshape(
                4,
            )
        )
        insertion_position_original = point_from_local_to_global(
            insertion_position_original_global, original_model.segment_coordinate_system_in_global("child")
        )
        npt.assert_almost_equal(insertion_position_modified, insertion_position_original)


def test_kinematic_chain_tool_errors():

    # More than one child of "root"
    with pytest.raises(
        NotImplementedError,
        match=r"Only inversion of kinematic chains with one first segment \(only one segment is the child of root\) is implemented",
    ):
        simple_model = create_simple_model()
        simple_model.add_segment(SegmentReal("bad_segment", parent_name="root"))
        kinematic_chain_modifier = ModifyKinematicChainTool(simple_model)
        kinematic_chain_modifier.add(ChangeFirstSegment(first_segment_name="child", new_segment_name="PELVIS"))
        modified_model = kinematic_chain_modifier.modify()

    # Segments without center of mass
    with pytest.raises(
        RuntimeError,
        match="The first segment must have inertia parameters with a center of mass defined, as the root segment will be defined at this point in the global reference frame.",
    ):
        simple_model = create_simple_model()
        simple_model.add_segment(SegmentReal("grand_child", parent_name="child"))
        kinematic_chain_modifier = ModifyKinematicChainTool(simple_model)
        kinematic_chain_modifier.add(ChangeFirstSegment(first_segment_name="grand_child", new_segment_name="PELVIS"))
        modified_model = kinematic_chain_modifier.modify()

    # Segments with rotations between them
    with pytest.raises(NotImplementedError, match=r"The rotation of inertia matrix is not implemented yet."):
        simple_model = create_simple_model()
        simple_model.add_segment(
            SegmentReal(
                "grand_child",
                parent_name="child",
                inertia_parameters=InertiaParametersReal(
                    mass=1.0, center_of_mass=np.array([0.1, 0.2, 0.3, 1.0]), inertia=np.eye(4)
                ),
            )
        )
        rt_matrix = RotoTransMatrix()
        rt_matrix.from_euler_angles_and_translation("xyz", np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))
        simple_model.segments["child"].segment_coordinate_system.scs = rt_matrix
        kinematic_chain_modifier = ModifyKinematicChainTool(simple_model)
        kinematic_chain_modifier.add(ChangeFirstSegment(first_segment_name="grand_child", new_segment_name="PELVIS"))
        modified_model = kinematic_chain_modifier.modify()

    # Segments with IMU
    with pytest.raises(
        NotImplementedError,
        match="This piece of code bellow was not tested yet, but if you encounter this error and observe that the code works, please open a PR on GitHub.",
    ):
        simple_model = create_simple_model()
        simple_model.segments["child"].add_imu(InertialMeasurementUnitReal("IMU"))
        kinematic_chain_modifier = ModifyKinematicChainTool(simple_model)
        kinematic_chain_modifier.add(ChangeFirstSegment(first_segment_name="child", new_segment_name="PELVIS"))
        modified_model = kinematic_chain_modifier.modify()
