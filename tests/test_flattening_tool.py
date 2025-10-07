import numpy as np
import pytest
import numpy.testing as npt
from pathlib import Path
import os

from biobuddy import (
    BiomechanicalModelReal,
    Translations,
    FlatteningTool,
    InertialMeasurementUnitReal,
)
from biobuddy.model_modifiers.flattening_tool import AXIS_TO_INDEX
from biobuddy.utils.linear_algebra import RotoTransMatrix
from test_utils import create_simple_model, compare_models


def test_flattening_tool_initialization():
    """Test the initialization of the FlatteningTool"""
    model = create_simple_model()
    flattening_tool = FlatteningTool(model, Translations.Y)

    assert flattening_tool.original_model is model
    assert flattening_tool.axis == Translations.Y
    assert flattening_tool.flattened_model is not model  # Should be a deep copy


def test_check_model():
    """Test the _check_model method"""
    model = create_simple_model()
    flattening_tool = FlatteningTool(model, Translations.Y)

    # This model is aligned so it should be OK
    flattening_tool._check_model()

    # Now, rotate the RT to create an error
    model.segments[0].segment_coordinate_system.scs.rotation_matrix = np.array(
        [[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]]
    )  # 30 degree rotation around Z

    flattening_tool = FlatteningTool(model, Translations.Y)

    # This should raise an error
    with pytest.raises(
        ValueError,
        match="Segment root has a rotated coordinate system. Flattening is only possible if all segment coordinate systems are aligned.",
    ):
        flattening_tool._check_model()


def test_modify_jcs():
    """Test the _modify_jcs method"""
    model = create_simple_model()
    flattening_tool = FlatteningTool(model, Translations.Z)

    # Before modification there is a translation on Z axis
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[2].segment_coordinate_system.scs.translation, np.array([0.0, 0.0, 1.0])
    )

    flattening_tool._modify_jcs()

    # After modification, Z should be 0
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[2].segment_coordinate_system.scs.translation, np.array([0.0, 0.0, 0.0])
    )

    # Original model should be unchanged
    npt.assert_almost_equal(
        flattening_tool.original_model.segments[2].segment_coordinate_system.scs.translation, np.array([0.0, 0.0, 1.0])
    )


def test_modify_com():
    """Test the _modify_com method"""
    model = create_simple_model()
    flattening_tool = FlatteningTool(model, Translations.Y)

    # Before modification
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[2].inertia_parameters.center_of_mass[:3].reshape((3,)),
        np.array([0.0, 0.1, 0.0]),
    )

    flattening_tool._modify_com()

    # After modification, Y should be 0
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[2].inertia_parameters.center_of_mass[:3].reshape((3,)),
        np.array([0.0, 0.0, 0.0]),
    )

    # Original model should be unchanged
    npt.assert_almost_equal(
        flattening_tool.original_model.segments[2].inertia_parameters.center_of_mass[:3].reshape((3,)),
        np.array([0.0, 0.1, 0.0]),
    )


def test_modify_markers():
    """Test the _modify_markers method"""
    model = create_simple_model()
    flattening_tool = FlatteningTool(model, Translations.X)

    # Before modification
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[1].markers[0].position[:3].reshape((3,)), np.array([0.1, 0.2, 0.3])
    )

    flattening_tool._modify_markers()

    # After modification, X should be 0
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[1].markers[0].position[:3].reshape((3,)), np.array([0.0, 0.2, 0.3])
    )

    # Original model should be unchanged
    npt.assert_almost_equal(
        flattening_tool.original_model.segments[1].markers[0].position[:3].reshape((3,)), np.array([0.1, 0.2, 0.3])
    )


def test_modify_contacts():
    """Test the _modify_contacts method"""
    model = create_simple_model()
    flattening_tool = FlatteningTool(model, Translations.Y)

    # Before modification
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[2].contacts[0].position[:3].reshape((3,)), np.array([-0.05, 0.5, 0.35])
    )

    flattening_tool._modify_contacts()

    # After modification, Y should be 0
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[2].contacts[0].position[:3].reshape((3,)), np.array([-0.05, 0.0, 0.35])
    )

    # Original model should be unchanged
    npt.assert_almost_equal(
        flattening_tool.original_model.segments[2].contacts[0].position[:3].reshape((3,)), np.array([-0.05, 0.5, 0.35])
    )


def test_modify_imus():
    """Test the _modify_imus method"""
    model = create_simple_model()
    model.segments["child"].add_imu(imu=InertialMeasurementUnitReal(name="test_imu", parent_name="child"))

    flattening_tool = FlatteningTool(model, Translations.Y)

    with pytest.raises(
        NotImplementedError,
        match="This feature was never tested. If you encounter this error, please contact the developers.",
    ):
        flattening_tool._modify_imus()


def test_modify_muscles():
    """Test the _modify_muscles method"""
    model = create_simple_model()
    flattening_tool = FlatteningTool(model, Translations.Y)

    # Before modification
    npt.assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].origin_position.position[:3].reshape((3,)),
        np.array([0.0, 0.1, 0.0]),
    )
    npt.assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].via_points[0].position[:3].reshape((3,)),
        np.array([0.2, 0.3, 0.4]),
    )
    npt.assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].insertion_position.position[:3].reshape((3,)),
        np.array([0.5, 0.4, 0.3]),
    )

    flattening_tool._modify_muscles()

    # After modification, Y should be 0
    npt.assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].origin_position.position[:3].reshape((3,)),
        np.array([0.0, 0.0, 0.0]),
    )
    npt.assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].via_points[0].position[:3].reshape((3,)),
        np.array([0.2, 0.0, 0.4]),
    )
    npt.assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].insertion_position.position[:3].reshape((3,)),
        np.array([0.5, 0.0, 0.3]),
    )

    # Original model should be unchanged
    npt.assert_almost_equal(
        flattening_tool.original_model.muscle_groups[0].muscles[0].origin_position.position[:3].reshape((3,)),
        np.array([0.0, 0.1, 0.0]),
    )
    npt.assert_almost_equal(
        flattening_tool.original_model.muscle_groups[0].muscles[0].via_points[0].position[:3].reshape((3,)),
        np.array([0.2, 0.3, 0.4]),
    )
    npt.assert_almost_equal(
        flattening_tool.original_model.muscle_groups[0].muscles[0].insertion_position.position[:3].reshape((3,)),
        np.array([0.5, 0.4, 0.3]),
    )


def test_flatten():
    """Test the flatten method"""
    model = create_simple_model()
    model.segments["root"].segment_coordinate_system.scs = RotoTransMatrix()
    flattening_tool = FlatteningTool(model, Translations.Y)

    # Before flattening
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[2].segment_coordinate_system.scs.translation[:3].reshape((3,)),
        np.array([0.0, 0.0, 1.0]),
    )
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[2].markers[0].position[:3].reshape((3,)), np.array([0.4, 0.5, 0.6])
    )
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[2].inertia_parameters.center_of_mass[:3].reshape((3,)),
        np.array([0.0, 0.1, 0.0]),
    )
    npt.assert_almost_equal(
        flattening_tool.flattened_model.segments[2].contacts[0].position[:3].reshape((3,)), np.array([-0.05, 0.5, 0.35])
    )
    npt.assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].origin_position.position[:3].reshape((3,)),
        np.array([0.0, 0.1, 0.0]),
    )
    npt.assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].via_points[0].position[:3].reshape((3,)),
        np.array([0.2, 0.3, 0.4]),
    )
    npt.assert_almost_equal(
        flattening_tool.flattened_model.muscle_groups[0].muscles[0].insertion_position.position[:3].reshape((3,)),
        np.array([0.5, 0.4, 0.3]),
    )

    flattened_model = flattening_tool.flatten()

    # After flattening, Y should be 0 everywhere
    npt.assert_almost_equal(
        flattened_model.segments[2].segment_coordinate_system.scs.translation[:3].reshape((3,)),
        np.array([0.0, 0.0, 1.0]),
    )
    npt.assert_almost_equal(
        flattened_model.segments[2].markers[0].position[:3].reshape((3,)), np.array([0.4, 0.0, 0.6])
    )
    npt.assert_almost_equal(
        flattened_model.segments[2].inertia_parameters.center_of_mass[:3].reshape((3,)), np.array([0.0, 0.0, 0.0])
    )
    npt.assert_almost_equal(
        flattened_model.segments[2].contacts[0].position[:3].reshape((3,)), np.array([-0.05, 0.0, 0.35])
    )
    npt.assert_almost_equal(
        flattened_model.muscle_groups[0].muscles[0].origin_position.position[:3].reshape((3,)),
        np.array([0.0, 0.0, 0.0]),
    )
    npt.assert_almost_equal(
        flattened_model.muscle_groups[0].muscles[0].via_points[0].position[:3].reshape((3,)), np.array([0.2, 0.0, 0.4])
    )
    npt.assert_almost_equal(
        flattened_model.muscle_groups[0].muscles[0].insertion_position.position[:3].reshape((3,)),
        np.array([0.5, 0.0, 0.3]),
    )

    # Muscle parameters should be changed to reflect new lengths
    original_muscle_length = flattening_tool.original_model.muscle_tendon_length(
        flattening_tool.original_model.muscle_groups[0].muscles[0].name
    )
    flattened_muscle_length = flattening_tool.flattened_model.muscle_tendon_length(
        flattening_tool.original_model.muscle_groups[0].muscles[0].name
    )
    ratio = flattened_muscle_length / original_muscle_length

    original_optimal_length = flattening_tool.original_model.muscle_groups[0].muscles[0].optimal_length
    flattened_optimal_length = flattening_tool.flattened_model.muscle_groups[0].muscles[0].optimal_length
    optimal_length_ratio = flattened_optimal_length / original_optimal_length

    original_tendon_slack_length = flattening_tool.original_model.muscle_groups[0].muscles[0].tendon_slack_length
    flattened_tendon_slack_length = flattening_tool.flattened_model.muscle_groups[0].muscles[0].tendon_slack_length
    tendon_slack_length_ratio = flattened_tendon_slack_length / original_tendon_slack_length

    npt.assert_almost_equal(ratio, optimal_length_ratio)
    npt.assert_almost_equal(ratio, tendon_slack_length_ratio)


def test_axis_to_index():
    """Test the AXIS_TO_INDEX mapping"""
    assert AXIS_TO_INDEX[Translations.X] == 0
    assert AXIS_TO_INDEX[Translations.Y] == 1
    assert AXIS_TO_INDEX[Translations.Z] == 2


def test_an_empty_model():
    model = BiomechanicalModelReal()
    flattening_tool = FlatteningTool(model, Translations.Y)
    flattened_model = flattening_tool.flatten()


def test_simplify_an_arm_model():
    from examples.applied_examples.simplify_an_arm_model import create_planar_model

    # Paths
    current_path_file = Path(__file__).parent
    biomod_filepath = f"{current_path_file}/../examples/models/simple_arm_model.bioMod"
    osim_filepath = f"{current_path_file}/../examples/models/MOBL_ARMS_41.osim"
    geometry_path = f"{current_path_file}/../examples/models/Geometry_cleaned"

    create_planar_model(
        osim_filepath=osim_filepath,
        biomod_filepath=biomod_filepath,
        geometry_path=geometry_path,
        with_mesh=True,
    )

    model = BiomechanicalModelReal().from_biomod(biomod_filepath)
    model_reference = BiomechanicalModelReal().from_biomod(biomod_filepath.replace(".bioMod", "_reference.bioMod"))
    compare_models(model, model_reference)

    os.remove(biomod_filepath)
