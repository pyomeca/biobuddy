import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path
import os

from biobuddy import (
    ViaPointReal,
    PathPointCondition,
    PathPointMovement,
    SimmSpline,
    SegmentReal,
    MeshFileReal,
    MeshReal,
    RangeOfMotion,
    Ranges,
    Translations,
    Rotations,
    RotoTransMatrix,
    InertiaParametersReal,
    LigamentReal,
)
from biobuddy.components.ligament_utils import LigamentType
from test_utils import create_simple_model


def test_fix_via_points_errors():

    # create a simple model
    model = create_simple_model()

    # Check that it does not have any conditional or moving via points
    for via_point in model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points + [
        model.muscle_groups["parent_to_child"].muscles["muscle1"].origin_position,
        model.muscle_groups["parent_to_child"].muscles["muscle1"].insertion_position,
    ]:
        assert via_point.condition is None
        assert via_point.movement is None

    # Check that it is not allowed to add a condition on origin/insertion
    with pytest.raises(RuntimeError, match="Muscle origin cannot be conditional."):
        model.muscle_groups["parent_to_child"].muscles["muscle1"].origin_position = ViaPointReal(
            name="origin_muscle1",
            parent_name="parent",
            position=np.array([0.0, 0.1, 0.0, 1.0]),
            condition=PathPointCondition(dof_name=f"child_rotX", range_min=0, range_max=np.pi / 2),
        )
        model.validate_model()
    model.muscle_groups["parent_to_child"].muscles["muscle1"].origin_position.condition = None
    with pytest.raises(RuntimeError, match="Muscle insertion cannot be conditional."):
        model.muscle_groups["parent_to_child"].muscles["muscle1"].insertion_position = ViaPointReal(
            name="origin_muscle1",
            parent_name="child",
            position=np.array([0.0, 0.1, 0.0, 1.0]),
            condition=PathPointCondition(dof_name=f"child_rotX", range_min=0, range_max=np.pi / 2),
        )
        model.validate_model()
    model.muscle_groups["parent_to_child"].muscles["muscle1"].insertion_position.condition = None


def test_fix_conditional_via_points_true():

    # create a simple model
    model = create_simple_model()

    # Add a conditional via point and fix it
    model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points["via_point1"].condition = PathPointCondition(
        dof_name=f"child_rotX", range_min=0, range_max=np.pi / 2
    )
    model.fix_via_points(np.ones((model.nb_q,)) * 0.1)

    # Check that the condition is fixed
    assert model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points["via_point1"].condition is None
    npt.assert_almost_equal(
        model.muscle_groups["parent_to_child"]
        .muscles["muscle1"]
        .via_points["via_point1"]
        .position.reshape(
            4,
        ),
        np.array([0.2, 0.3, 0.4, 1.0]),
    )


def test_fix_conditional_via_points_false():
    # create a simple model
    model = create_simple_model()

    # Add a conditional via point and fix it
    model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points["via_point1"].condition = PathPointCondition(
        dof_name=f"child_rotX", range_min=0, range_max=np.pi / 2
    )
    model.fix_via_points(np.ones((model.nb_q,)) * -0.1)

    # Check that the condition is fixed
    assert "via_point1" not in model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points


def test_fix_moving_via_points_errors():

    # create a simple model
    model = create_simple_model()

    # Bad sizes
    with pytest.raises(RuntimeError, match=r"dof_names must be a list of 3 dof_names \(x, y, x\)."):
        PathPointMovement(
            dof_names=["child_rotX"],
            locations=[
                SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6]))
            ],
        )
    with pytest.raises(RuntimeError, match=r"locations must be a list of 3 Functions \(x, y, x\)."):
        PathPointMovement(
            dof_names=["child_rotX", "child_rotX", "child_rotX"],
            locations=[
                SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6]))
            ],
        )

    # Bad type
    with pytest.raises(RuntimeError, match="All locations must be instances of Functions."):
        PathPointMovement(
            dof_names=["child_rotX", "child_rotY", "child_rotZ"],
            locations=[
                SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])),
                None,
                None,
            ],
        )


def test_fix_moving_via_points():

    # create a simple model
    model = create_simple_model()

    # Add a moving via point and fix it
    model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points["via_point1"].movement = PathPointMovement(
        dof_names=["child_rotX", "child_rotX", "child_rotX"],
        locations=[
            SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])),
            SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])),
            SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])),
        ],
    )

    # Check that it is not allowed to have position and movement
    with pytest.raises(
        RuntimeError,
        match="A via point can either have a position or a movement, but not both at the same time, via_point1 has both.",
    ):
        model.validate_model()

    # But if we remove the position, it is fine
    model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points["via_point1"].position = None
    model.validate_model()

    # But not possible to have a condition and a movement
    with pytest.raises(
        RuntimeError,
        match="A via point can either have a movement or a condition, but not both at the same time, via_point1 has both.",
    ):
        model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points["via_point1"].condition = (
            PathPointCondition(dof_name=f"child_rotX", range_min=0, range_max=np.pi / 2)
        )
        model.validate_model()

    # If we remove it, it's fine again
    model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points["via_point1"].condition = None
    model.validate_model()

    # Fix the via points
    model.fix_via_points(np.ones((model.nb_q,)) * 0.15)

    # Check that the position is fixed
    assert model.muscle_groups["parent_to_child"].muscles["muscle1"].via_points["via_point1"].movement is None
    expected_value = SimmSpline(
        x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    ).evaluate(0.15)[0]
    npt.assert_almost_equal(expected_value, 0.25)
    npt.assert_almost_equal(
        model.muscle_groups["parent_to_child"]
        .muscles["muscle1"]
        .via_points["via_point1"]
        .position.reshape(
            4,
        ),
        np.array([0.25, 0.25, 0.25, 1.0]),
    )


def test_fix_moving_origin():

    # create a simple model
    model = create_simple_model()

    # Add a moving origin and fix it
    model.muscle_groups["parent_to_child"].muscles["muscle1"].origin_position.movement = PathPointMovement(
        dof_names=["child_rotX", "child_rotX", "child_rotX"],
        locations=[
            SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])),
            SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])),
            SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])),
        ],
    )

    # Check that it is not allowed to have position and movement
    with pytest.raises(
        RuntimeError,
        match="A via point can either have a position or a movement, but not both at the same time, origin_muscle1 has both.",
    ):
        model.validate_model()

    # But if we remove the position, it is fine
    model.muscle_groups["parent_to_child"].muscles["muscle1"].origin_position.position = None
    model.validate_model()

    # Fix the via points
    model.fix_via_points(np.ones((model.nb_q,)) * 0.15)

    # Check that the position is fixed
    assert model.muscle_groups["parent_to_child"].muscles["muscle1"].origin_position.movement is None
    expected_value = SimmSpline(
        x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    ).evaluate(0.15)[0]
    npt.assert_almost_equal(expected_value, 0.25)
    npt.assert_almost_equal(
        model.muscle_groups["parent_to_child"]
        .muscles["muscle1"]
        .origin_position.position.reshape(
            4,
        ),
        np.array([0.25, 0.25, 0.25, 1.0]),
    )


def test_fix_moving_insertion():

    # create a simple model
    model = create_simple_model()

    # Add a moving insertion and fix it
    model.muscle_groups["parent_to_child"].muscles["muscle1"].insertion_position.movement = PathPointMovement(
        dof_names=["child_rotX", "child_rotX", "child_rotX"],
        locations=[
            SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])),
            SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])),
            SimmSpline(x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])),
        ],
    )

    # Check that it is not allowed to have position and movement
    with pytest.raises(
        RuntimeError,
        match="A via point can either have a position or a movement, but not both at the same time, insertion_muscle1 has both.",
    ):
        model.validate_model()

    # But if we remove the position, it is fine
    model.muscle_groups["parent_to_child"].muscles["muscle1"].insertion_position.position = None
    model.validate_model()

    # Fix the via points
    model.fix_via_points(np.ones((model.nb_q,)) * 0.15)

    # Check that the position is fixed
    assert model.muscle_groups["parent_to_child"].muscles["muscle1"].insertion_position.movement is None
    expected_value = SimmSpline(
        x_points=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_points=np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    ).evaluate(0.15)[0]
    npt.assert_almost_equal(expected_value, 0.25)
    npt.assert_almost_equal(
        model.muscle_groups["parent_to_child"]
        .muscles["muscle1"]
        .insertion_position.position.reshape(
            4,
        ),
        np.array([0.25, 0.25, 0.25, 1.0]),
    )


def test_check_kinematic_chain_loop():
    # create a simple model
    model = create_simple_model()

    # Check that the model is valid
    model.validate_model()

    # Check that it is not allowed to have a closed-loop
    model.add_segment(
        SegmentReal(
            name="grand-child",
            parent_name="child",
        )
    )
    model.segments["child"].parent_name = "grand-child"
    with pytest.raises(
        RuntimeError,
        match="The segment child was caught up in a kinematic chain loop, which is not permitted. Please verify the parent-child relationships in yor model.",
    ):
        model.validate_model()


def test_dof_ranges():
    # create a simple model
    model = create_simple_model()

    # Add some DoF ranges
    model.segments["parent"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q, min_bound=[-np.pi / 4] * 6, max_bound=[np.pi / 4] * 6
    )
    model.segments["child"].q_ranges = RangeOfMotion(range_type=Ranges.Q, min_bound=[-np.pi / 2], max_bound=[np.pi / 2])
    assert model.segments["parent"].q_ranges.min_bound == [-np.pi / 4] * 6
    assert model.segments["parent"].q_ranges.max_bound == [np.pi / 4] * 6
    assert model.segments["child"].q_ranges.min_bound == [-np.pi / 2]
    assert model.segments["child"].q_ranges.max_bound == [np.pi / 2]

    ranges = model.get_dof_ranges()
    assert ranges.shape == (2, 7)
    npt.assert_almost_equal(
        ranges,
        np.array(
            [
                [-0.78539816, -0.78539816, -0.78539816, -0.78539816, -0.78539816, -0.78539816, -1.57079633],
                [0.78539816, 0.78539816, 0.78539816, 0.78539816, 0.78539816, 0.78539816, 1.57079633],
            ]
        ),
    )


def test_change_mesh_directories():
    # create a simple model
    model = create_simple_model()

    # Add some mesh files
    model.segments["parent"].mesh_file = MeshFileReal(
        mesh_file_name="parent_mesh.obj", mesh_file_directory="old_geometry"
    )
    model.segments["child"].mesh_file = MeshFileReal(
        mesh_file_name="child_mesh.obj", mesh_file_directory="old_geometry"
    )
    assert model.segments["parent"].mesh_file.mesh_file_directory == "old_geometry"
    assert model.segments["child"].mesh_file.mesh_file_directory == "old_geometry"

    # Change mesh directories
    model.change_mesh_directories(new_directory="new_geometry")

    # Check that the mesh directories have been changed
    assert model.segments["parent"].mesh_file.mesh_file_directory == "new_geometry"
    assert model.segments["child"].mesh_file.mesh_file_directory == "new_geometry"


def test_get_full_segment_chain():
    # Make sure one segment works
    model = create_simple_model()
    segment_chain = model.get_full_segment_chain(segment_name="child")
    assert segment_chain == ["child"]

    # Test for a logical chain of segments
    model = create_simple_model()
    model.add_segment(
        SegmentReal(
            name="new_parent_offset",
            parent_name="child",
        )
    )
    model.add_segment(
        SegmentReal(
            name="new_translation",
            parent_name="new_parent_offset",
        )
    )
    model.add_segment(
        SegmentReal(
            name="new_rotation",
            parent_name="new_translation",
        )
    )
    model.add_segment(
        SegmentReal(
            name="new_geom1",
            parent_name="new_rotation",
        )
    )
    model.add_segment(
        SegmentReal(
            name="new_geom2",
            parent_name="new_rotation",
        )
    )
    model.add_segment(
        SegmentReal(
            name="new_reset_axis",
            parent_name="new_geom1",
        )
    )
    model.add_segment(
        SegmentReal(
            name="new",
            parent_name="new_reset_axis",
        )
    )
    segment_chain = model.get_full_segment_chain(segment_name="new")
    assert segment_chain == [
        "new_parent_offset",
        "new_translation",
        "new_rotation",
        "new_geom1",
        "new_geom2",
        "new_reset_axis",
        "new",
    ]

    # Test for a not supported chain of segments
    model = create_simple_model()
    model.add_segment(
        SegmentReal(
            name="new_parent_offset",
            parent_name="child",
        )
    )
    model.add_segment(
        SegmentReal(
            name="bad_segment",
            parent_name="parent",
        )
    )
    model.add_segment(
        SegmentReal(
            name="new_rotation",
            parent_name="new_parent_offset",
        )
    )
    model.add_segment(
        SegmentReal(
            name="new",
            parent_name="new_rotation",
        )
    )
    with pytest.raises(
        NotImplementedError,
        match="The segments in the model are not in the correct order to get the full segment chain for new.",
    ):
        model.get_full_segment_chain(segment_name="new")


def test_has_mesh():
    # create a simple model
    model = create_simple_model()

    # Initially, two segments have a mesh
    assert model.has_meshes

    # Remove meshes
    model.segments["parent"].mesh = None
    model.segments["child"].mesh = None
    assert not model.has_meshes


def test_has_mesh_file():
    # create a simple model
    model = create_simple_model()

    # Initially, no segments have mesh file files
    assert not model.has_mesh_files

    # Add a mesh file to one segment
    model.segments["parent"].mesh_file = MeshFileReal(mesh_file_name="parent_mesh.obj", mesh_file_directory="geometry")
    assert model.has_mesh_files


def test_write_graphviz():
    from examples.create_graph_from_bioMod_file import create_graph_from_biomod_file

    # Create the dot and png
    create_graph_from_biomod_file()

    # Check that the files have been created
    current_path_file = Path(__file__).parent
    base_name = "arm26_allbiceps_1dof"
    output_path = f"{current_path_file}/../examples/data/{base_name}"
    assert os.path.isfile(f"{output_path}.dot")
    assert os.path.isfile(f"{output_path}.png")

    # Check that the dot is the same as the reference
    with open(f"{output_path}_reference.dot", "r") as f:
        reference_dot = f.read()

    with open(f"{output_path}.dot", "r") as f:
        output_dot = f.read()

    assert reference_dot == output_dot


def test_segment_has_ghost_parents():
    # Create a simple model
    model = create_simple_model()

    # Initially, no segments have ghost parents
    assert not model.segment_has_ghost_parents("parent")
    assert not model.segment_has_ghost_parents("child")

    # Add a segment with a ghost parent
    model.add_segment(
        SegmentReal(
            name="test_segment_parent_offset",
            parent_name="child",
        )
    )
    assert model.segment_has_ghost_parents("test_segment")

    # Add other types of ghost parents
    model.add_segment(
        SegmentReal(
            name="test_segment2_translation",
            parent_name="child",
        )
    )
    assert model.segment_has_ghost_parents("test_segment2")

    model.add_segment(
        SegmentReal(
            name="test_segment3_rotation_transform",
            parent_name="child",
        )
    )
    assert model.segment_has_ghost_parents("test_segment3")

    model.add_segment(
        SegmentReal(
            name="test_segment4_reset_axis",
            parent_name="child",
        )
    )
    assert model.segment_has_ghost_parents("test_segment4")


def test_children_segment_names():
    # Create a simple model
    model = create_simple_model()

    # Test getting children of parent segment
    children = model.children_segment_names("parent")
    assert children == ["child"]

    # Test getting children of child segment (should be empty)
    children = model.children_segment_names("child")
    assert children == []

    # Add more children to test multiple children
    model.add_segment(
        SegmentReal(
            name="child2",
            parent_name="parent",
        )
    )
    model.add_segment(
        SegmentReal(
            name="child3",
            parent_name="parent",
        )
    )

    children = model.children_segment_names("parent")
    assert set(children) == {"child", "child2", "child3"}


def test_get_chain_between_segments():
    # Create a simple model
    model = create_simple_model()

    # Test chain from parent to child
    chain = model.get_chain_between_segments("parent", "child")
    assert chain == ["parent", "child"]

    # Add more segments to create a longer chain
    model.add_segment(
        SegmentReal(
            name="grandchild",
            parent_name="child",
        )
    )
    model.add_segment(
        SegmentReal(
            name="great_grandchild",
            parent_name="grandchild",
        )
    )

    # Test longer chain
    chain = model.get_chain_between_segments("parent", "great_grandchild")
    assert chain == ["parent", "child", "grandchild", "great_grandchild"]

    # Test chain from child to grandchild
    chain = model.get_chain_between_segments("child", "grandchild")
    assert chain == ["child", "grandchild"]

    # Test when segments are not in the same chain (should return empty list)
    model.add_segment(
        SegmentReal(
            name="unrelated_segment",
            parent_name="base",
        )
    )
    chain = model.get_chain_between_segments("parent", "unrelated_segment")
    assert chain == []

    # Test same segment
    chain = model.get_chain_between_segments("child", "child")
    assert chain == ["child"]


def test_get_real_parent_name():
    # Create a simple model
    model = create_simple_model()

    # Test with no ghost parents
    real_parent = model.get_real_parent_name("child")
    assert real_parent == "parent"

    # Add ghost parents
    model.add_segment(
        SegmentReal(
            name="test_segment_parent_offset",
            parent_name="child",
        )
    )
    model.add_segment(
        SegmentReal(
            name="test_segment_translation",
            parent_name="test_segment_parent_offset",
        )
    )
    model.add_segment(
        SegmentReal(
            name="test_segment",
            parent_name="test_segment_translation",
        )
    )

    # Test getting real parent through ghost parents
    real_parent = model.get_real_parent_name("test_segment")
    assert real_parent == "test_segment_translation"


def test_has_parent_offset():
    # Create a simple model
    model = create_simple_model()

    # Initially, no segments have parent offset
    assert not model.has_parent_offset("parent")
    assert not model.has_parent_offset("child")

    # Add a segment with parent offset
    model.add_segment(
        SegmentReal(
            name="test_segment_parent_offset",
            parent_name="child",
        )
    )

    # Now test_segment should have a parent offset
    assert model.has_parent_offset("test_segment")


def test_dof_parent_segment_name():
    # Create a simple model
    model = create_simple_model()

    # Test getting parent segment for a DOF
    parent_segment = model.dof_parent_segment_name("child_rotX")
    assert parent_segment == "child"

    # Test with parent segment DOFs
    parent_segment = model.dof_parent_segment_name("parent_transX")
    assert parent_segment == "parent"

    # Test with non-existent DOF
    with pytest.raises(ValueError, match="Degree of freedom non_existent_dof not found in the model"):
        model.dof_parent_segment_name("non_existent_dof")


def test_remove_muscles():
    # Create a simple model
    model = create_simple_model()

    # Verify initial state
    assert "muscle1" in model.muscle_groups["parent_to_child"].muscle_names

    # Remove a muscle
    model.remove_muscles(["muscle1"])

    # Verify muscle was removed
    assert "muscle1" not in model.muscle_groups["parent_to_child"].muscle_names

    # Test removing non-existent muscle (should not raise error)
    model.remove_muscles(["non_existent_muscle"])


def test_update_muscle_groups():
    # Create a simple model
    model = create_simple_model()

    # Remove all muscles from a muscle group
    model.remove_muscles(["muscle1"])

    # Verify muscle group still exists
    assert "parent_to_child" in model.muscle_group_names

    # Update muscle groups to remove empty ones
    model.update_muscle_groups()

    # Verify empty muscle group was removed
    assert "parent_to_child" not in model.muscle_group_names


def test_update_segments():
    # Create a simple model
    model = create_simple_model()

    # Add an empty segment (no markers, contacts, imus, dofs, mesh, inertia)
    model.add_segment(
        SegmentReal(
            name="empty_segment",
            parent_name="child",
            translations=Translations.NONE,
            rotations=Rotations.NONE,
        )
    )

    # Add a child to the empty segment
    model.add_segment(
        SegmentReal(
            name="child_of_empty",
            parent_name="empty_segment",
            translations=Translations.XYZ,  # Add something to the segment so that it is kept
        )
    )

    # Verify empty segment exists
    assert "empty_segment" in model.segment_names

    # Update segments to remove empty ones
    model.update_segments()

    # Verify empty segment was removed
    assert "empty_segment" not in model.segment_names

    # Verify kinematic chain was updated
    assert model.segments["child_of_empty"].parent_name == "child"


def test_modify_model_static_pose():
    # Create a simple model
    model = create_simple_model()

    # Get initial segment coordinate systems
    initial_scs = model.segments["child"].segment_coordinate_system.scs.rt_matrix.copy()

    # Create a new static pose
    q_static = np.ones((model.nb_q,)) * 0.1

    # Modify the static pose
    model.modify_model_static_pose(q_static)

    # Verify segment coordinate systems were updated
    new_scs = model.segments["child"].segment_coordinate_system.scs.rt_matrix
    assert not np.allclose(initial_scs, new_scs)

    # Test with wrong shape
    with pytest.raises(RuntimeError, match="The shape of q_static must be"):
        model.modify_model_static_pose(np.ones((model.nb_q + 1,)))


def test_dof_indices():
    # Create a simple model
    model = create_simple_model()

    # Test getting DOF indices for parent segment
    parent_indices = model.dof_indices("parent")
    assert parent_indices == [0, 1, 2, 3, 4, 5]

    # Test getting DOF indices for child segment
    child_indices = model.dof_indices("child")
    assert child_indices == [6]

    # Test with non-existent segment
    with pytest.raises(ValueError, match="Segment non_existent not found in the model"):
        model.dof_indices("non_existent")


def test_markers_indices():
    # Create a simple model
    model = create_simple_model()

    # Get marker indices
    marker_names = model.marker_names
    indices = model.markers_indices(marker_names[:2])

    # Verify indices are correct
    assert indices == [0, 1]


def test_contact_indices():
    # Create a simple model with contacts
    model = create_simple_model()

    # Add contacts to test
    from biobuddy import ContactReal

    model.segments["parent"].add_contact(
        ContactReal(name="contact1", parent_name="parent", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    )
    model.segments["child"].add_contact(
        ContactReal(name="contact2", parent_name="child", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    )

    # Get contact indices
    contact_names = model.contact_names
    indices = model.contact_indices(contact_names)

    # Verify indices are correct
    assert indices == [0, 1, 2, 3]
    assert model.contact_indices(["contact1"]) == [1]


def test_imu_indices():
    # Create a simple model with IMUs
    model = create_simple_model()

    # Add IMUs to test
    from biobuddy import InertialMeasurementUnitReal

    model.segments["parent"].add_imu(InertialMeasurementUnitReal(name="imu1", parent_name="parent"))
    model.segments["child"].add_imu(InertialMeasurementUnitReal(name="imu2", parent_name="child"))

    # Get IMU indices
    imu_names = model.imu_names
    indices = model.imu_indices(imu_names)

    # Verify indices are correct
    assert indices == [0, 1]


def test_dofs_property():
    # Create a simple model
    model = create_simple_model()

    # Get DOFs
    dofs = model.dofs

    # Verify DOFs are correct
    assert len(dofs) == 3
    assert dofs[0] == Translations.XYZ
    assert dofs[1] == Rotations.XYZ
    assert dofs[2] == Rotations.X


def test_muscle_group_origin_insertion_parent_names():
    # Create a simple model
    model = create_simple_model()

    # Get muscle group origin parent names
    origin_names = model.muscle_group_origin_parent_names
    assert origin_names == ["parent"]

    # Get muscle group insertion parent names
    insertion_names = model.muscle_group_insertion_parent_names
    assert insertion_names == ["child"]


def test_ligament_origin_insertion_parent_names():
    # Create a simple model
    model = create_simple_model()

    # Add a ligament
    from biobuddy import LigamentReal, ViaPointReal
    from biobuddy.components.ligament_utils import LigamentType

    origin = ViaPointReal(name="lig_origin", parent_name="parent", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    insertion = ViaPointReal(name="lig_insertion", parent_name="child", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    ligament = LigamentReal(
        name="test_ligament",
        ligament_type=LigamentType.LINEAR_SPRING,
        origin_position=origin,
        insertion_position=insertion,
        ligament_slack_length=0.1,
        stiffness=1000.0,
    )

    model.add_ligament(ligament)

    # Get ligament origin parent names
    origin_names = model.ligament_origin_parent_names
    assert origin_names == ["parent"]

    # Get ligament insertion parent names
    insertion_names = model.ligament_insertion_parent_names
    assert insertion_names == ["child"]


def test_segment_coordinate_system_in_local():
    # Create a simple model
    model = create_simple_model()

    # Test with base segment
    scs_local = model.segment_coordinate_system_in_local("base")
    assert scs_local.is_identity

    # Test with segment already in local -> does not chance
    model.segments["parent"].segment_coordinate_system.scs = RotoTransMatrix.from_euler_angles_and_translation(
        "xyz",
        angles=np.array([0.3, 0.2, 0.1]),
        translation=np.array([0.5, 0.0, 0.5])
    )
    model.segments["child"].segment_coordinate_system.scs = RotoTransMatrix.from_euler_angles_and_translation(
        "xyz",
        angles=np.array([0.1, 0.2, 0.3]),
        translation=np.array([0.25, 0.5, 0.75])
    )
    model.segments["child"].segment_coordinate_system.is_in_global = False
    scs_local = model.segment_coordinate_system_in_local("child")
    assert scs_local == model.segments["child"].segment_coordinate_system.scs
    npt.assert_almost_equal(scs_local.rt_matrix, np.array([[ 0.93629336, -0.28962948,  0.19866933,  0.25      ],
                                                   [ 0.31299183,  0.94470249, -0.0978434 ,  0.5       ],
                                                   [-0.15934508,  0.153792  ,  0.97517033,  0.75      ],
                                                   [ 0.        ,  0.        ,  0.        ,  1.        ]]))

    # Test with segment in global (needs transformation)
    model.segments["child"].segment_coordinate_system.is_in_global = True
    new_scs_local = model.segment_coordinate_system_in_local("child")
    # The result should be different from the original global SCS
    assert not np.allclose(new_scs_local.rt_matrix, scs_local.rt_matrix)
    npt.assert_almost_equal(new_scs_local.rt_matrix, np.array([[ 0.986572  , -0.16165639,  0.02330031, -0.20673285],
                                   [ 0.15420033,  0.96893676,  0.19334896,  0.57506005],
                                   [-0.05383262, -0.18715975,  0.98085334,  0.03959127],
                                   [ 0.        ,  0.        ,  0.        ,  1.        ]]))

def test_segment_coordinate_system_in_global():
    # Create a simple model
    model = create_simple_model()

    # Test with base segment
    scs_global = model.segment_coordinate_system_in_global("base")
    assert scs_global.is_identity

    # Test with segment already in global
    model.segments["child"].segment_coordinate_system.is_in_global = True
    scs_global = model.segment_coordinate_system_in_global("child")
    assert scs_global == model.segments["child"].segment_coordinate_system.scs

    # Test with segment in local (needs transformation)
    model.segments["child"].segment_coordinate_system.is_in_global = False
    scs_global = model.segment_coordinate_system_in_global("child")
    # The result should include parent transformations
    assert isinstance(scs_global, RotoTransMatrix)


def test_rt_from_parent_offset_to_real_segment():
    # Create a simple model
    model = create_simple_model()

    # Test with no parent offset
    rt = model.rt_from_parent_offset_to_real_segment("child")
    assert rt.is_identity

    # Add a parent offset segment
    model.add_segment(
        SegmentReal(
            name="test_segment_parent_offset",
            parent_name="child",
        )
    )
    model.add_segment(
        SegmentReal(
            name="test_segment",
            parent_name="test_segment_parent_offset",
        )
    )

    # Test with parent offset
    rt = model.rt_from_parent_offset_to_real_segment("test_segment")
    assert isinstance(rt, RotoTransMatrix)


def test_segment_com_in_global():
    # Create a simple model
    model = create_simple_model()

    # Add inertia parameters to a segment
    model.segments["child"].inertia_parameters = InertiaParametersReal(
        mass=1.0, center_of_mass=np.array([[0.1], [0.2], [0.3], [1.0]]), inertia=np.eye(4)
    )

    # Test with default q (zeros)
    com_position = model.segment_com_in_global("child")
    assert com_position.shape == (4, 1)
    assert com_position[3, 0] == 1.0

    # Test with custom q
    q = np.ones((model.nb_q, 1)) * 0.1
    com_position = model.segment_com_in_global("child", q)
    assert com_position.shape == (4, 1)

    # Test with segment without inertia
    model.segments["parent"].inertia_parameters = None
    com_position = model.segment_com_in_global("parent")
    assert com_position is None


def test_muscle_origin_in_global():
    # Create a simple model
    model = create_simple_model()

    # Test with default q
    origin_position = model.muscle_origin_in_global("muscle1")
    assert origin_position.shape == (4, 1)
    assert origin_position[3, 0] == 1.0

    # Test with custom q
    q = np.ones((model.nb_q, 1)) * 0.1
    origin_position = model.muscle_origin_in_global("muscle1", q)
    assert origin_position.shape == (4, 1)

    # Test with multiple frames
    q = np.ones((model.nb_q, 5)) * 0.1
    origin_position = model.muscle_origin_in_global("muscle1", q)
    assert origin_position.shape == (4, 5)


def test_muscle_insertion_in_global():
    # Create a simple model
    model = create_simple_model()

    # Test with default q
    insertion_position = model.muscle_insertion_in_global("muscle1")
    assert insertion_position.shape == (4, 1)
    assert insertion_position[3, 0] == 1.0

    # Test with custom q
    q = np.ones((model.nb_q, 1)) * 0.1
    insertion_position = model.muscle_insertion_in_global("muscle1", q)
    assert insertion_position.shape == (4, 1)

    # Test with multiple frames
    q = np.ones((model.nb_q, 5)) * 0.1
    insertion_position = model.muscle_insertion_in_global("muscle1", q)
    assert insertion_position.shape == (4, 5)


def test_via_points_in_global():
    # Create a simple model
    model = create_simple_model()

    # Test with default q
    via_points_position = model.via_points_in_global("muscle1")
    assert via_points_position.shape[0] == 4
    assert via_points_position.shape[2] == 1

    # Test with custom q
    q = np.ones((model.nb_q, 1)) * 0.1
    via_points_position = model.via_points_in_global("muscle1", q)
    assert via_points_position.shape[0] == 4

    # Test with multiple frames
    q = np.ones((model.nb_q, 5)) * 0.1
    via_points_position = model.via_points_in_global("muscle1", q)
    assert via_points_position.shape[2] == 5


def test_total_com_in_global():
    # Create a simple model
    model = create_simple_model()

    # Add inertia parameters to segments
    model.segments["parent"].inertia_parameters = InertiaParametersReal(
        mass=2.0, center_of_mass=np.array([[0.1], [0.2], [0.3], [1.0]]), inertia=np.eye(4)
    )
    model.segments["child"].inertia_parameters = InertiaParametersReal(
        mass=1.0, center_of_mass=np.array([[0.2], [0.3], [0.4], [1.0]]), inertia=np.eye(4)
    )

    # Test with default q
    com_position = model.total_com_in_global()
    assert com_position.shape == (4, 1)
    assert com_position[3, 0] == 1.0

    # Test with custom q
    q = np.ones((model.nb_q, 1)) * 0.1
    com_position = model.total_com_in_global(q)
    assert com_position.shape == (4, 1)

    # Test with multiple frames
    q = np.ones((model.nb_q, 5)) * 0.1
    com_position = model.total_com_in_global(q)
    assert com_position.shape == (4, 5)


def test_muscle_tendon_length():
    # Create a simple model
    model = create_simple_model()

    # Test with default q
    length = model.muscle_tendon_length("muscle1")
    assert length.shape == (1,)
    assert length[0] > 0

    # Test with custom q
    q = np.ones((model.nb_q, 1)) * 0.1
    length = model.muscle_tendon_length("muscle1", q)
    assert length.shape == (1,)

    # Test with multiple frames
    q = np.ones((model.nb_q, 5)) * 0.1
    length = model.muscle_tendon_length("muscle1", q)
    assert length.shape == (5,)

    # Test with non-existent muscle
    with pytest.raises(RuntimeError, match="The muscle non_existent was not found in the model."):
        model.muscle_tendon_length("non_existent")


def test_ligament_length():
    # Create a simple model
    model = create_simple_model()

    # Add a ligament
    origin = ViaPointReal(name="lig_origin", parent_name="parent", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    insertion = ViaPointReal(name="lig_insertion", parent_name="child", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    ligament = LigamentReal(
        name="test_ligament",
        ligament_type=LigamentType.LINEAR_SPRING,
        origin_position=origin,
        insertion_position=insertion,
        ligament_slack_length=0.1,
        stiffness=1000.0,
    )

    model.add_ligament(ligament)

    # Test with default q
    length = model.ligament_length("test_ligament")
    assert length.shape == (1,)
    assert length[0] > 0

    # Test with custom q
    q = np.ones((model.nb_q, 1)) * 0.1
    length = model.ligament_length("test_ligament", q)
    assert length.shape == (1,)

    # Test with multiple frames
    q = np.ones((model.nb_q, 5)) * 0.1
    length = model.ligament_length("test_ligament", q)
    assert length.shape == (5,)


def test_get_ligament_length_ranges():
    # Create a simple model
    model = create_simple_model()

    # Add DoF ranges
    model.segments["parent"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q, min_bound=[-np.pi / 4] * 6, max_bound=[np.pi / 4] * 6
    )
    model.segments["child"].q_ranges = RangeOfMotion(range_type=Ranges.Q, min_bound=[-np.pi / 2], max_bound=[np.pi / 2])

    # Add a ligament
    origin = ViaPointReal(name="lig_origin", parent_name="parent", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    insertion = ViaPointReal(name="lig_insertion", parent_name="child", position=np.array([[1.0], [0.0], [0.0], [1.0]]))

    ligament = LigamentReal(
        name="test_ligament",
        ligament_type=LigamentType.LINEAR_SPRING,
        origin_position=origin,
        insertion_position=insertion,
        ligament_slack_length=0.1,
        stiffness=1000.0,
    )

    model.add_ligament(ligament)

    # Get ligament length ranges
    ranges = model.get_ligament_length_ranges()

    # Verify the ranges
    assert "test_ligament" in ranges
    assert len(ranges["test_ligament"]) == 2
    assert ranges["test_ligament"][0] == 0.1  # slack length
    npt.assert_almost_equal(ranges["test_ligament"][1], 1.41421356, decimal=6)  # max length


def test_forward_kinematics_errors():
    # Create a simple model
    model = create_simple_model()

    # Test with wrong shape
    with pytest.raises(RuntimeError, match="q must be of shape"):
        model.forward_kinematics(np.ones((model.nb_q, 5, 3)))


def test_markers_in_global_errors():
    # Create a simple model
    model = create_simple_model()

    # Test with wrong shape
    with pytest.raises(RuntimeError, match="q must be of shape"):
        model.markers_in_global(np.ones((model.nb_q, 5, 3)))


def test_contacts_in_global():
    # Create a simple model with contacts
    model = create_simple_model()

    from biobuddy import ContactReal

    model.segments["parent"].add_contact(
        ContactReal(name="contact1", parent_name="parent", position=np.array([[0.0], [0.0], [0.0], [1.0]]))
    )

    # Test with default q
    contact_positions = model.contacts_in_global()
    assert contact_positions.shape[0] == 4
    assert contact_positions.shape[2] == 1

    # Test with custom q
    q = np.ones((model.nb_q, 1)) * 0.1
    contact_positions = model.contacts_in_global(q)
    assert contact_positions.shape[0] == 4

    # Test with multiple frames
    q = np.ones((model.nb_q, 5)) * 0.1
    contact_positions = model.contacts_in_global(q)
    assert contact_positions.shape[2] == 5


def test_requires_initialization_decorator():
    # Create a model without initialization
    from biobuddy.components.real.model_dynamics import ModelDynamics

    model = ModelDynamics()

    # Test that methods raise error when not initialized
    with pytest.raises(RuntimeError, match="cannot be called because the object is not initialized"):
        model.segment_coordinate_system_in_local("test")

    with pytest.raises(RuntimeError, match="cannot be called because the object is not initialized"):
        model.segment_coordinate_system_in_global("test")

    with pytest.raises(RuntimeError, match="cannot be called because the object is not initialized"):
        model.forward_kinematics()
