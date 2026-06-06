import numpy as np
import pytest

from biobuddy import DictData
from biobuddy.components.generic.rigidbody.axis import Axis
from biobuddy.gui.model_builder import AxisSpec, MarkerEndpointSpec
from biobuddy.gui.virtual_points import (
    compute_virtual_axes,
    compute_virtual_points,
    example_predictive_hip_cor,
    example_predictive_shoulder_cor,
    global_linear_regression_virtual_point,
    global_vector_virtual_axis,
    local_frame_virtual_axis,
    local_frame_regression_virtual_point,
    marker_data_with_virtual_axes,
    marker_data_with_virtual_features,
    marker_data_with_virtual_points,
    marker_pair_virtual_axis,
    marker_mean_virtual_point,
    point_pair_virtual_axis,
    pointing_virtual_point,
    sara_virtual_axis_placeholder,
)


def test_pointing_and_marker_mean_virtual_points_are_evaluated():
    data = _simple_marker_data()
    pointing = pointing_virtual_point("PointedHip", "A")
    midpoint = marker_mean_virtual_point("MidAB", ("A", "B"))

    points = compute_virtual_points(data, (pointing, midpoint))

    np.testing.assert_allclose(points["PointedHip"][:3, 0], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(points["MidAB"][:3, 0], np.array([0.5, 0.0, 0.0]))


def test_global_linear_regression_virtual_point_uses_marker_weights():
    data = _simple_marker_data()
    regression = global_linear_regression_virtual_point(
        name="Weighted",
        marker_weights={"A": 0.25, "B": 0.75},
        intercept=(0.0, 0.0, 0.1),
    )

    point = regression.evaluate(data)

    np.testing.assert_allclose(point[:3, 0], np.array([0.75, 0.0, 0.1]))


def test_marker_pair_virtual_axis_is_evaluated_from_marker_groups():
    data = _simple_marker_data()
    axis = marker_pair_virtual_axis("ABAxis", ("A",), ("B",))

    axes = compute_virtual_axes(data, (axis,))
    start, end = axes["ABAxis"]

    np.testing.assert_allclose(start[:3, 0], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(end[:3, 0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(axis.vector(data)[:, 0], np.array([1.0, 0.0, 0.0]))


def test_point_pair_virtual_axis_uses_two_virtual_points():
    data = _simple_marker_data()
    start_point = marker_mean_virtual_point("Start", ("A", "C"))
    end_point = marker_mean_virtual_point("End", ("B", "C"))
    axis = point_pair_virtual_axis("MeanAxis", start_point, end_point)

    start, end = axis.evaluate(data)

    np.testing.assert_allclose(start[:3, 0], np.array([0.0, 0.5, 0.0]))
    np.testing.assert_allclose(end[:3, 0], np.array([0.5, 0.5, 0.0]))


def test_global_vector_virtual_axis_uses_normalized_direction_and_length():
    data = _simple_marker_data()
    origin = pointing_virtual_point("Origin", "A")
    axis = global_vector_virtual_axis("GlobalZ", origin, vector=(0.0, 0.0, 2.0), length=0.5)

    start, end = axis.evaluate(data)

    np.testing.assert_allclose(start[:3, 0], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(end[:3, 0], np.array([0.0, 0.0, 0.5]))


def test_local_frame_virtual_axis_projects_local_direction_to_global():
    data = _simple_marker_data()
    axis = local_frame_virtual_axis(
        name="LocalY",
        origin=MarkerEndpointSpec(("A",)),
        first_axis=AxisSpec.from_markers(Axis.Name.X, "A", "B"),
        second_axis=AxisSpec.from_markers(Axis.Name.Y, "A", "C"),
        axis_to_keep=Axis.Name.X,
        local_direction=(0.0, 1.0, 0.0),
        length=0.25,
    )

    start, end = axis.evaluate(data)

    np.testing.assert_allclose(start[:3, 0], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(end[:3, 0], np.array([0.0, 0.25, 0.0]))


def test_marker_data_with_virtual_axes_appends_axis_endpoints():
    data = _simple_marker_data()
    augmented = marker_data_with_virtual_axes(data, (marker_pair_virtual_axis("ABAxis", ("A",), ("B",)),))

    assert "ABAxis_start" in augmented.marker_names
    assert "ABAxis_end" in augmented.marker_names
    np.testing.assert_allclose(augmented.get_position(["ABAxis_end"])[:3, 0, 0], np.array([1.0, 0.0, 0.0]))


def test_marker_data_with_virtual_features_appends_points_and_axis_endpoints():
    data = _simple_marker_data()
    augmented = marker_data_with_virtual_features(
        data,
        point_definitions=(marker_mean_virtual_point("MidAB", ("A", "B")),),
        axis_definitions=(marker_pair_virtual_axis("ABAxis", ("A",), ("B",)),),
    )

    assert "MidAB" in augmented.marker_names
    assert "ABAxis_start" in augmented.marker_names
    assert "ABAxis_end" in augmented.marker_names
    np.testing.assert_allclose(augmented.get_position(["MidAB"])[:3, 0, 0], np.array([0.5, 0.0, 0.0]))


def test_sara_virtual_axis_placeholder_declares_required_markers_and_raises():
    data = _simple_marker_data()
    condyle_axis = marker_pair_virtual_axis("Condyles", ("A",), ("B",))
    axis = sara_virtual_axis_placeholder(
        "ElbowAxis",
        parent_marker_names=("A", "B"),
        child_marker_names=("B", "C"),
        condyle_axis=condyle_axis,
    )

    assert axis.required_markers == ("A", "B", "C")
    with pytest.raises(NotImplementedError, match="SARA virtual axis"):
        axis.evaluate(data)


def test_local_frame_regression_virtual_point_projects_offsets_to_global():
    data = _simple_marker_data()
    regression = local_frame_regression_virtual_point(
        name="LocalOffset",
        origin=MarkerEndpointSpec(("A",)),
        first_axis=AxisSpec.from_markers(Axis.Name.X, "A", "B"),
        second_axis=AxisSpec.from_markers(Axis.Name.Y, "A", "C"),
        axis_to_keep=Axis.Name.X,
        local_offset=(0.1, 0.2, 0.3),
    )

    point = regression.evaluate(data)

    np.testing.assert_allclose(point[:3, 0], np.array([0.1, 0.2, 0.3]))


def test_marker_data_with_virtual_points_appends_computed_markers():
    data = _simple_marker_data()
    augmented = marker_data_with_virtual_points(data, (marker_mean_virtual_point("MidAB", ("A", "B")),))

    assert "MidAB" in augmented.marker_names
    np.testing.assert_allclose(augmented.get_position(["MidAB"])[:3, 0, 0], np.array([0.5, 0.0, 0.0]))


def test_virtual_point_reports_missing_required_markers():
    data = _simple_marker_data()
    definition = marker_mean_virtual_point("MissingMean", ("A", "DOES_NOT_EXIST"))

    with pytest.raises(ValueError, match="DOES_NOT_EXIST"):
        definition.evaluate(data)


def test_predictive_joint_center_examples_define_expected_required_markers():
    hip = example_predictive_hip_cor("D")
    shoulder = example_predictive_shoulder_cor("G")

    assert {"EIASD", "EIASG", "EIPSD", "EIPSG"}.issubset(hip.required_markers)
    assert {"CLAV1G", "ACRANTG", "ACRPOSTG"}.issubset(shoulder.required_markers)


def _simple_marker_data() -> DictData:
    marker_dict = {}
    points = {
        "A": (0.0, 0.0, 0.0),
        "B": (1.0, 0.0, 0.0),
        "C": (0.0, 1.0, 0.0),
    }
    for marker_name, position in points.items():
        marker = np.ones((4, 2))
        marker[:3, :] = np.asarray(position, dtype=float)[:, np.newaxis]
        marker_dict[marker_name] = marker
    return DictData(marker_dict)
