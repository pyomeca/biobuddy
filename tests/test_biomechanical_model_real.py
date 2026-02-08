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
)
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
    ).evaluate(0.15)
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
    ).evaluate(0.15)
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
    ).evaluate(0.15)
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
