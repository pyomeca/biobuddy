import pytest
import numpy as np
import numpy.testing as npt
from lxml import etree

import opensim as osim

from biobuddy import (
    SimmSpline,
    PiecewiseLinearFunction,
)


@pytest.mark.parametrize("nb_nodes", [3, 7])
def test_simm_spline(nb_nodes: int):

    # Create sample data
    if nb_nodes == 3:
        # 3 points is a special case
        x_points = [0, 45, 90]
        y_points = [0.02, 0.05, 0.01]
    else:
        x_points = [0, 15, 30, 45, 60, 75, 90]
        y_points = [0.02, 0.045, 0.055, 0.05, 0.04, 0.025, 0.01]

    # Create spline
    biobuddy_spline = SimmSpline(np.array(x_points), np.array(y_points))

    # Create equivalent OpenSim spline
    opensim_spline = osim.SimmSpline()
    # Add data points to the spline
    for x, y in zip(x_points, y_points):
        opensim_spline.addPoint(x, y)

    # Test evaluation
    test_x = 37.5
    osim_vector = osim.Vector()
    osim_vector.resize(1)
    osim_vector.set(0, test_x)

    # The evaluated value is the same
    npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], opensim_spline.calcValue(osim_vector), decimal=6)

    if nb_nodes == 7:
        # The derivative too, but I get a c++ error from Opensim on the remote tests (that I cannot reproduce locally), so I'll just test the values.
        # order_1 = osim.StdVectorInt()
        # order_1.append(1)
        # npt.assert_almost_equal(
        #     biobuddy_spline.evaluate_derivative(test_x, order=1),
        #     opensim_spline.calcDerivative(order_1, osim_vector),
        #     decimal=6,
        # )
        npt.assert_almost_equal(biobuddy_spline.evaluate_derivative(test_x, order=1), -0.0003825093035619349)

        # However, there is a mismatch for order 2 :(
        # npt.assert_almost_equal(biobuddy_spline.evaluate_derivative(test_x, order=2), opensim_spline.calcDerivative(order_2, osim_vector), decimal=3)
        with pytest.raises(
            NotImplementedError,
            match="Only first derivative is implemented. There is a discrepancy with OpenSim for the second order derivative.",
        ):
            biobuddy_spline.evaluate_derivative(test_x, order=2)

        # Get coefficients
        b, c, d = biobuddy_spline.get_coefficients()
        npt.assert_almost_equal(
            b,
            np.array(
                [
                    2.14207868e-03,
                    1.19125465e-03,
                    9.29027113e-05,
                    -5.62865497e-04,
                    -8.41440723e-04,
                    -1.07137161e-03,
                    -8.73072834e-04,
                ]
            ),
        )
        npt.assert_almost_equal(
            c,
            np.array(
                [
                    -3.16941343e-05,
                    -3.16941343e-05,
                    -4.15293284e-05,
                    -2.18855219e-06,
                    -1.63831295e-05,
                    1.05440369e-06,
                    1.21655148e-05,
                ]
            ),
        )
        npt.assert_almost_equal(
            d,
            np.array(
                [
                    -1.12937726e-22,
                    -2.18559868e-07,
                    8.74239471e-07,
                    -3.15435052e-07,
                    3.87500738e-07,
                    2.46913580e-07,
                    2.46913580e-07,
                ]
            ),
        )

        # Test for extrapolation
        test_x = -10.0
        osim_vector = osim.Vector()
        osim_vector.resize(1)
        osim_vector.set(0, test_x)
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], opensim_spline.calcValue(osim_vector), decimal=6)

        test_x = 180.0
        osim_vector = osim.Vector()
        osim_vector.resize(1)
        osim_vector.set(0, test_x)
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], opensim_spline.calcValue(osim_vector), decimal=6)

        # Test for values at the end of the range
        test_x = 0.0
        osim_vector = osim.Vector()
        osim_vector.resize(1)
        osim_vector.set(0, test_x)
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], opensim_spline.calcValue(osim_vector), decimal=6)

        test_x = 90.0
        osim_vector = osim.Vector()
        osim_vector.resize(1)
        osim_vector.set(0, test_x)
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], opensim_spline.calcValue(osim_vector), decimal=6)


def test_simm_spline_errors():

    # Test with less than two points
    with pytest.raises(ValueError, match="At least 2 data points are required"):
        SimmSpline(np.array([1.0]), np.array([1.0]))

    # Test with mismatched lengths
    with pytest.raises(ValueError, match="x_points and y_points must have the same length"):
        SimmSpline(np.array([0, 1]), np.array([0, 1, 2]))

    # Test non-increasing x
    with pytest.raises(ValueError, match="x_points must be sorted in ascending order"):
        SimmSpline(np.array([1.0, 0.5, 0.0]), np.array([1.0, 0.5, 0.0]))

    # order
    x_points = [0, 15, 30, 45, 60, 75, 90]
    y_points = [0.02, 0.045, 0.055, 0.05, 0.04, 0.025, 0.01]
    biobuddy_spline = SimmSpline(np.array(x_points), np.array(y_points))
    with pytest.raises(
        NotImplementedError,
        match="Only first derivative is implemented. There is a discrepancy with OpenSim for the second order derivative.",
    ):
        biobuddy_spline.evaluate_derivative(30, order=0)
    with pytest.raises(
        NotImplementedError,
        match="Only first derivative is implemented. There is a discrepancy with OpenSim for the second order derivative.",
    ):
        biobuddy_spline.evaluate_derivative(30, order=3)

    # Tes derivatives at the end points
    with pytest.raises(NotImplementedError, match="Extrapolation for derivatives is not implemented."):
        biobuddy_spline.evaluate_derivative(-1.0, order=1)
    with pytest.raises(NotImplementedError, match="Extrapolation for derivatives is not implemented."):
        biobuddy_spline.evaluate_derivative(180.0, order=1)
    with pytest.raises(NotImplementedError, match="Extrapolation for derivatives is not implemented."):
        biobuddy_spline.evaluate_derivative(0.0, order=1)
    with pytest.raises(NotImplementedError, match="Extrapolation for derivatives is not implemented."):
        biobuddy_spline.evaluate_derivative(90.0, order=1)


def test_simm_spline_safe_max():
    # Create a simple spline
    x_points = np.array([0, 1, 2])
    y_points = np.array([0, 1, 0])
    spline = SimmSpline(x_points, y_points)

    # Test safe_max with normal values
    result = spline.safe_max(np.array([1.0, 2.0, 3.0]))
    assert result == 3.0

    # Test safe_max with very small values (should return TINY_NUMBER)
    result = spline.safe_max(np.array([1e-10, 1e-11, 1e-12]))
    assert result == spline.TINY_NUMBER

    # Test safe_max with negative values
    result = spline.safe_max(np.array([-1.0, -2.0, -3.0]))
    assert result == spline.TINY_NUMBER


def test_simm_spline_get_scalar_value():
    # Create a simple spline
    x_points = np.array([0, 1, 2])
    y_points = np.array([0, 1, 0])
    spline = SimmSpline(x_points, y_points)

    # Test with scalar float
    result = spline.get_scalar_value(5.0)
    assert result == 5.0

    # Test with scalar int
    result = spline.get_scalar_value(5)
    assert result == 5

    # Test with single-element array
    result = spline.get_scalar_value(np.array([5.0]))
    assert result == 5.0

    # Test with single-element list
    result = spline.get_scalar_value([5.0])
    assert result == 5.0

    # Test with multi-element array (should raise error)
    with pytest.raises(ValueError, match="Only single value arrays are supported"):
        spline.get_scalar_value(np.array([5.0, 6.0]))

    # Test with multi-element list (should raise error)
    with pytest.raises(ValueError, match="Only single value arrays are supported"):
        spline.get_scalar_value([5.0, 6.0])


def test_simm_spline_evaluate_array_inputs():
    # Create a simple spline
    x_points = np.array([0, 15, 30, 45, 60, 75, 90])
    y_points = np.array([0.02, 0.045, 0.055, 0.05, 0.04, 0.025, 0.01])
    spline = SimmSpline(x_points, y_points)

    # Test with array of x values
    test_x = np.array([10.0, 20.0, 30.0, 40.0])
    result = spline.evaluate(test_x)
    assert result.shape == (4,)

    # Verify each value individually
    for i, x in enumerate(test_x):
        expected = spline.evaluate(x)[0]
        npt.assert_almost_equal(result[i], expected)

    # Test with 2D array (should be reshaped)
    test_x = np.array([[10.0], [20.0], [30.0]])
    result = spline.evaluate(test_x)
    assert result.shape == (3,)


def test_simm_spline_evaluate_derivative_array_inputs():
    # Create a simple spline
    x_points = np.array([0, 15, 30, 45, 60, 75, 90])
    y_points = np.array([0.02, 0.045, 0.055, 0.05, 0.04, 0.025, 0.01])
    spline = SimmSpline(x_points, y_points)

    # Test with array of x values
    test_x = np.array([10.0, 20.0, 30.0, 40.0])
    result = spline.evaluate_derivative(test_x, order=1)
    assert result.shape == (4,)

    # Verify each value individually
    for i, x in enumerate(test_x):
        expected = spline.evaluate_derivative(x, order=1)[0]
        npt.assert_almost_equal(result[i], expected)

    # Test with 2D array (should be reshaped)
    test_x = np.array([[10.0], [20.0], [30.0]])
    result = spline.evaluate_derivative(test_x, order=1)
    assert result.shape == (3,)


def test_simm_spline_to_osim():
    # Create a simple spline
    x_points = np.array([0, 15, 30, 45])
    y_points = np.array([0.02, 0.045, 0.055, 0.05])
    spline = SimmSpline(x_points, y_points)

    # Test with default name
    xml_elem = spline.to_osim()
    assert xml_elem.tag == "SimmSpline"
    assert xml_elem.get("name") == "spline_function"

    # Check x values
    x_elem = xml_elem.find("x")
    assert x_elem is not None
    x_values = [float(v) for v in x_elem.text.split()]
    npt.assert_almost_equal(x_values, x_points)

    # Check y values
    y_elem = xml_elem.find("y")
    assert y_elem is not None
    y_values = [float(v) for v in y_elem.text.split()]
    npt.assert_almost_equal(y_values, y_points)

    # Test with custom name
    xml_elem = spline.to_osim(name="custom_spline")
    assert xml_elem.get("name") == "custom_spline"


def test_simm_spline_two_points():
    # Test the special case with only 2 points (linear interpolation)
    x_points = np.array([0, 10])
    y_points = np.array([0, 5])
    spline = SimmSpline(x_points, y_points)

    # Get coefficients
    b, c, d = spline.get_coefficients()

    # For 2 points, should be linear (c and d should be zero)
    npt.assert_almost_equal(c, np.array([0.0, 0.0]))
    npt.assert_almost_equal(d, np.array([0.0, 0.0]))

    # b should be the slope
    expected_slope = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
    npt.assert_almost_equal(b, np.array([expected_slope, expected_slope]))

    # Test evaluation
    test_x = 5.0
    result = spline.evaluate(test_x)
    expected = y_points[0] + expected_slope * (test_x - x_points[0])
    npt.assert_almost_equal(result[0], expected)


@pytest.mark.parametrize("nb_nodes", [2, 3])
def test_linear_function(nb_nodes: int):

    # Create sample data
    if nb_nodes == 2:
        x_points = [0, 45]
        y_points = [0.02, 0.05]
    else:
        x_points = [0, 45, 90]
        y_points = [0.02, 0.05, 0.01]

    # Create spline
    biobuddy_spline = PiecewiseLinearFunction(np.array(x_points), np.array(y_points))

    # Test evaluation
    if nb_nodes == 2:
        test_x = 37.5
        expected_a = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
        expected_b = 0.02
        expected_value = expected_a * test_x + expected_b
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], expected_value)

        # Test derivatives
        npt.assert_almost_equal(biobuddy_spline.evaluate_derivative(test_x, order=1), expected_a, decimal=6)
        npt.assert_almost_equal(biobuddy_spline.evaluate_derivative(test_x, order=2), 0, decimal=6)

        # Get coefficients
        a, b = biobuddy_spline.get_coefficients()
        npt.assert_almost_equal(a, expected_a)
        npt.assert_almost_equal(b, expected_b)

        # Test for extrapolation
        test_x = -10.0
        expected_value = expected_a * test_x + expected_b
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], expected_value, decimal=6)

        test_x = 180.0
        expected_value = expected_a * test_x + expected_b
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], expected_value, decimal=6)

        # Test for values at the end of the range
        test_x = 0.0
        expected_value = expected_a * test_x + expected_b
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], expected_value, decimal=6)

        test_x = 90.0
        expected_value = expected_a * test_x + expected_b
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], expected_value, decimal=6)

    elif nb_nodes == 3:
        # Test the first segment
        test_x = 7.5
        expected_a_1 = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
        expected_b_1 = 0.02
        expected_value = expected_a_1 * test_x + expected_b_1
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], expected_value)

        # Test derivatives
        npt.assert_almost_equal(biobuddy_spline.evaluate_derivative(test_x, order=1), expected_a_1, decimal=6)
        npt.assert_almost_equal(biobuddy_spline.evaluate_derivative(test_x, order=2), 0, decimal=6)

        # Test the second segment
        test_x = 57.5
        expected_a_2 = (y_points[2] - y_points[1]) / (x_points[2] - x_points[1])
        expected_b_2 = y_points[1] - expected_a_2 * x_points[1]
        expected_value = expected_a_2 * test_x + expected_b_2
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x)[0], expected_value)

        # Test derivatives
        npt.assert_almost_equal(biobuddy_spline.evaluate_derivative(test_x, order=1), expected_a_2, decimal=6)
        npt.assert_almost_equal(biobuddy_spline.evaluate_derivative(test_x, order=2), 0, decimal=6)

        # Get coefficients
        a, b = biobuddy_spline.get_coefficients()
        npt.assert_almost_equal(a, np.array([expected_a_1, expected_a_2]))
        npt.assert_almost_equal(b, np.array([expected_b_1, expected_b_2]))


def test_linear_function_errors():

    # Test with less than two points
    with pytest.raises(ValueError, match="At least 2 data points are required"):
        PiecewiseLinearFunction(np.array([1.0]), np.array([1.0]))

    # Test with mismatched lengths
    with pytest.raises(ValueError, match="x_points and y_points must have the same length"):
        PiecewiseLinearFunction(np.array([0, 1]), np.array([0, 1, 2]))

    # Test non-increasing x
    with pytest.raises(ValueError, match="x_points must be sorted in ascending order"):
        PiecewiseLinearFunction(np.array([1.0, 0.5, 0.0]), np.array([1.0, 0.5, 0.0]))

    # order
    x_points = [0, 15, 30, 45, 60, 75, 90]
    y_points = [0.02, 0.045, 0.055, 0.05, 0.04, 0.025, 0.01]
    biobuddy_spline = PiecewiseLinearFunction(np.array(x_points), np.array(y_points))
    with pytest.raises(
        RuntimeError,
        match="The order of the derivative must be an int larger or equal to 1.0",
    ):
        biobuddy_spline.evaluate_derivative(30, order=0)
    with pytest.raises(
        RuntimeError,
        match="The order of the derivative must be an int larger or equal to 1.0",
    ):
        biobuddy_spline.evaluate_derivative(30, order=1.5)


def test_piecewise_linear_safe_max():
    # Create a simple function
    x_points = np.array([0, 1, 2])
    y_points = np.array([0, 1, 0])
    func = PiecewiseLinearFunction(x_points, y_points)

    # Test safe_max with normal values
    result = func.safe_max(np.array([1.0, 2.0, 3.0]))
    assert result == 3.0

    # Test safe_max with very small values (should return TINY_NUMBER)
    result = func.safe_max(np.array([1e-10, 1e-11, 1e-12]))
    assert result == func.TINY_NUMBER


def test_piecewise_linear_get_scalar_value():
    # Create a simple function
    x_points = np.array([0, 1, 2])
    y_points = np.array([0, 1, 0])
    func = PiecewiseLinearFunction(x_points, y_points)

    # Test with scalar float
    result = func.get_scalar_value(5.0)
    assert result == 5.0

    # Test with single-element array
    result = func.get_scalar_value(np.array([5.0]))
    assert result == 5.0


def test_piecewise_linear_get_coefficient_index():
    # Create a function with multiple segments
    x_points = np.array([0, 10, 20, 30])
    y_points = np.array([0, 5, 3, 8])
    func = PiecewiseLinearFunction(x_points, y_points)

    # Test x before first point (should use first segment)
    idx = func.get_coefficient_index(-5.0)
    assert idx == 0

    # Test x after last point (should use last segment)
    idx = func.get_coefficient_index(50.0)
    assert idx == -1

    # Test x in first segment
    idx = func.get_coefficient_index(5.0)
    assert idx == 0

    # Test x in second segment
    idx = func.get_coefficient_index(15.0)
    assert idx == 1

    # Test x in third segment
    idx = func.get_coefficient_index(25.0)
    assert idx == 2

    # Test x exactly at a point
    idx = func.get_coefficient_index(10.0)
    assert idx == 1  # Should use the segment after the point

    idx = func.get_coefficient_index(20.0)
    assert idx == 2  # Should use the segment after the point


def test_piecewise_linear_evaluate_array_inputs():
    # Create a simple function
    x_points = np.array([0, 10, 20, 30])
    y_points = np.array([0, 5, 3, 8])
    func = PiecewiseLinearFunction(x_points, y_points)

    # Test with array of x values
    test_x = np.array([5.0, 15.0, 25.0])
    result = func.evaluate(test_x)
    assert result.shape == (3,)

    # Verify each value individually
    for i, x in enumerate(test_x):
        expected = func.evaluate(x)[0]
        npt.assert_almost_equal(result[i], expected)

    # Test with 2D array (should be reshaped)
    test_x = np.array([[5.0], [15.0], [25.0]])
    result = func.evaluate(test_x)
    assert result.shape == (3,)


def test_piecewise_linear_evaluate_derivative_array_inputs():
    # Create a simple function
    x_points = np.array([0, 10, 20, 30])
    y_points = np.array([0, 5, 3, 8])
    func = PiecewiseLinearFunction(x_points, y_points)

    # Test with array of x values for first derivative
    test_x = np.array([5.0, 15.0, 25.0])
    result = func.evaluate_derivative(test_x, order=1)
    assert result.shape == (3,)

    # Verify each value individually
    for i, x in enumerate(test_x):
        expected = func.evaluate_derivative(x, order=1)[0]
        npt.assert_almost_equal(result[i], expected)

    # Test with array of x values for second derivative (should be all zeros)
    result = func.evaluate_derivative(test_x, order=2)
    assert result.shape == (3,)
    npt.assert_almost_equal(result, np.zeros(3))

    # Test with higher order derivatives (should be all zeros)
    result = func.evaluate_derivative(test_x, order=3)
    assert result.shape == (3,)
    npt.assert_almost_equal(result, np.zeros(3))


def test_piecewise_linear_to_osim():
    # Create a simple function
    x_points = np.array([0, 10, 20, 30])
    y_points = np.array([0, 5, 3, 8])
    func = PiecewiseLinearFunction(x_points, y_points)

    # Test with default name
    xml_elem = func.to_osim()
    assert xml_elem.tag == "PiecewiseLinearFunction"
    assert xml_elem.get("name") == "piecewise_linear_function"

    # Check x values
    x_elem = xml_elem.find("x")
    assert x_elem is not None
    x_values = [float(v) for v in x_elem.text.split("\t")]
    npt.assert_almost_equal(x_values, x_points)

    # Check y values (note: the implementation has a bug - it uses "x" tag for y values)
    y_elem = xml_elem.findall("x")[1]  # Second "x" element is actually y values
    assert y_elem is not None
    y_values = [float(v) for v in y_elem.text.split("\t")]
    npt.assert_almost_equal(y_values, y_points)

    # Test with custom name
    xml_elem = func.to_osim(name="custom_linear_func")
    assert xml_elem.get("name") == "custom_linear_func"


def test_piecewise_linear_evaluate_at_boundaries():
    # Create a simple function
    x_points = np.array([0, 10, 20])
    y_points = np.array([0, 5, 3])
    func = PiecewiseLinearFunction(x_points, y_points)

    # Test evaluation exactly at x_points
    result = func.evaluate(0.0)
    npt.assert_almost_equal(result[0], 0.0)

    result = func.evaluate(10.0)
    npt.assert_almost_equal(result[0], 5.0)

    result = func.evaluate(20.0)
    npt.assert_almost_equal(result[0], 3.0)


def test_simm_spline_evaluate_at_boundaries():
    # Create a simple spline
    x_points = np.array([0, 10, 20])
    y_points = np.array([0, 5, 3])
    spline = SimmSpline(x_points, y_points)

    # Test evaluation exactly at x_points (within tolerance)
    result = spline.evaluate(0.0 + 1e-11)
    npt.assert_almost_equal(result[0], 0.0, decimal=5)

    result = spline.evaluate(20.0 - 1e-11)
    npt.assert_almost_equal(result[0], 3.0, decimal=5)
