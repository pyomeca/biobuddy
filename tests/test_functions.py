import pytest
import numpy as np
import numpy.testing as npt

import opensim as osim

from biobuddy import (
    SimmSpline,
)


def test_simm_spline():

    # Create sample data (could represent muscle moment arm vs. joint angle)
    x_points = [0, 15, 30, 45, 60, 75, 90]  # Joint angle in degrees
    y_points = [0.02, 0.045, 0.055, 0.05, 0.04, 0.025, 0.01]  # Moment arm in meters

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

    # The derivative too, but I get a c++ error from Opensim on the remote tests (that I cannot reproduce locally), so I'll just test the values.
    # order_1 = osim.StdVectorInt()
    # order_1.append(1)
    # npt.assert_almost_equal(
    #     biobuddy_spline.evaluate_derivative(test_x, order=1),
    #     opensim_spline.calcDerivative(1, osim_vector),
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
