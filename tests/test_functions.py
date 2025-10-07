import pytest
import numpy as np
import numpy.testing as npt

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
    npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), opensim_spline.calcValue(osim_vector), decimal=6)

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
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), opensim_spline.calcValue(osim_vector), decimal=6)

        test_x = 180.0
        osim_vector = osim.Vector()
        osim_vector.resize(1)
        osim_vector.set(0, test_x)
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), opensim_spline.calcValue(osim_vector), decimal=6)

        # Test for values at the end of the range
        test_x = 0.0
        osim_vector = osim.Vector()
        osim_vector.resize(1)
        osim_vector.set(0, test_x)
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), opensim_spline.calcValue(osim_vector), decimal=6)

        test_x = 90.0
        osim_vector = osim.Vector()
        osim_vector.resize(1)
        osim_vector.set(0, test_x)
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), opensim_spline.calcValue(osim_vector), decimal=6)


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
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), expected_value)

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
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), expected_value, decimal=6)

        test_x = 180.0
        expected_value = expected_a * test_x + expected_b
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), expected_value, decimal=6)

        # Test for values at the end of the range
        test_x = 0.0
        expected_value = expected_a * test_x + expected_b
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), expected_value, decimal=6)

        test_x = 90.0
        expected_value = expected_a * test_x + expected_b
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), expected_value, decimal=6)

    elif nb_nodes == 3:
        # Test the first segment
        test_x = 7.5
        expected_a_1 = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
        expected_b_1 = 0.02
        expected_value = expected_a_1 * test_x + expected_b_1
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), expected_value)

        # Test derivatives
        npt.assert_almost_equal(biobuddy_spline.evaluate_derivative(test_x, order=1), expected_a_1, decimal=6)
        npt.assert_almost_equal(biobuddy_spline.evaluate_derivative(test_x, order=2), 0, decimal=6)

        # Test the second segment
        test_x = 57.5
        expected_a_2 = (y_points[2] - y_points[1]) / (x_points[2] - x_points[1])
        expected_b_2 = y_points[1] - expected_a_2 * x_points[1]
        expected_value = expected_a_2 * test_x + expected_b_2
        npt.assert_almost_equal(biobuddy_spline.evaluate(test_x), expected_value)

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
