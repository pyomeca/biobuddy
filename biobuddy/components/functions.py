from typing import TypeAlias
import numpy as np


class SimmSpline:
    """
    Python implementation of SIMM (Software for Interactive Musculoskeletal Modeling) cubic spline interpolation.
    Translated from opensim-core/OpenSim/Common/SimmSpline.cpp
    """
    def __init__(self, x_points: np.ndarray, y_points: np.ndarray):
        """
        Initialize the SimmSpline with x and y data points.

        Parameters
        ----------
        x_points
            The x coordinates of the data points (must be sorted in ascending order).
        y_points
            The y coordinates of the data points.
        """
        nb_nodes = x_points.shape[0]
        if nb_nodes < 2:
            raise ValueError("At least 2 data points are required")
        if len(y_points) != nb_nodes:
            raise ValueError("x_points and y_points must have the same length")
        if not np.all(x_points[:-1] <= x_points[1:]):
            raise ValueError("x_points must be sorted in ascending order")

        self.nb_nodes = nb_nodes
        self.x_points = x_points
        self.y_points = y_points

        # Calculate spline coefficients
        self.b = None
        self.c = None
        self.d = None
        self.TINY_NUMBER = 0.0000001  # Defined in opensim-core/OpenSim/Common/SimmMacros.h
        self._calculate_coefficients()  # Will set b, c, and d

    def safe_max(self, array: np.ndarray) -> float:
        out = np.max(array)
        out = self.TINY_NUMBER if self.TINY_NUMBER > out else out
        return out

    def _calculate_coefficients(self):
        """Calculate the spline coefficients."""

        if self.nb_nodes < 2:
            return

        # Initialize coefficient arrays
        self.b = np.zeros((self.nb_nodes, ))
        self.c = np.zeros((self.nb_nodes, ))
        self.d = np.zeros((self.nb_nodes, ))

        # Handle the case with only 2 points (linear interpolation)
        if self.nb_nodes == 2:
            t = self.safe_max(self.x_points[1] - self.x_points[0])
            self.b[0] = self.b[1] = (self.y_points[1] - self.y_points[0]) / t
            self.c[0] = self.c[1] = 0.0
            self.d[0] = self.d[1] = 0.0
            return

        nm1 = self.nb_nodes - 1
        nm2 = self.nb_nodes - 2

        # Set up tridiagonal system:
        # b = diagonal, d = offdiagonal, c = right-hand side
        self.d[0] = self.safe_max(self.x_points[1] - self.x_points[0])
        self.c[1] = (self.y_points[1] - self.y_points[0]) / self.d[0]

        for i in range(1, nm1):
            self.d[i] = self.safe_max(self.x_points[i + 1] - self.x_points[i])
            self.b[i] = 2.0 * (self.d[i - 1] + self.d[i])
            self.c[i + 1] = (self.y_points[i + 1] - self.y_points[i]) / self.d[i]
            self.c[i] = self.c[i + 1] - self.c[i]

        # End conditions. Third derivatives at x[0] and x[self.nb_nodes-1]
        # are obtained from divided differences.
        self.b[0] = -self.d[0]
        self.b[nm1] = -self.d[nm2]
        self.c[0] = 0.0
        self.c[nm1] = 0.0

        if self.nb_nodes > 3:
            d31 = self.safe_max(self.x_points[3] - self.x_points[1])
            d20 = self.safe_max(self.x_points[2] - self.x_points[0])
            d1 = self.safe_max(self.x_points[nm1] - self.x_points[self.nb_nodes - 3])
            d2 = self.safe_max(self.x_points[nm2] - self.x_points[self.nb_nodes - 4])
            d30 = self.safe_max(self.x_points[3] - self.x_points[0])
            d3 = self.safe_max(self.x_points[nm1] - self.x_points[self.nb_nodes - 4])

            self.c[0] = self.c[2] / d31 - self.c[1] / d20
            self.c[nm1] = self.c[nm2] / d1 - self.c[self.nb_nodes - 3] / d2
            self.c[0] = self.c[0] * self.d[0] * self.d[0] / d30
            self.c[nm1] = -self.c[nm1] * self.d[nm2] * self.d[nm2] / d3

        # Forward elimination
        for i in range(1, self.nb_nodes):
            t = self.d[i - 1] / self.b[i - 1]
            self.b[i] -= t * self.d[i - 1]
            self.c[i] -= t * self.c[i - 1]

        # Back substitution
        self.c[nm1] /= self.b[nm1]
        for j in range(nm1):
            i = nm2 - j
            self.c[i] = (self.c[i] - self.d[i] * self.c[i + 1]) / self.b[i]

        # Compute polynomial coefficients
        self.b[nm1] = ((self.y_points[nm1] - self.y_points[nm2]) / self.d[nm2] +
                        self.d[nm2] * (self.c[nm2] + 2.0 * self.c[nm1]))

        for i in range(nm1):
            self.b[i] = ((self.y_points[i + 1] - self.y_points[i]) / self.d[i] -
                          self.d[i] * (self.c[i + 1] + 2.0 * self.c[i]))
            self.d[i] = (self.c[i + 1] - self.c[i]) / self.d[i]
            self.c[i] *= 3.0

        self.c[nm1] *= 3.0
        self.d[nm1] = self.d[nm2]

    def get_coefficients(self):
        """Return the calculated coefficients."""
        return self.b.copy(), self.c.copy(), self.d.copy()

    def evaluate(self, x: float) -> float:
        """
        Calculate the spline value at a given x coordinate.

        Parameters
        ----------
        x
            The x coordinate to evaluate the spline at
        """
        # Handle both scalar and array inputs
        if hasattr(x, '__len__') and not isinstance(x, str):
            aX = x[0]  # Use first element if array-like
        else:
            aX = x

        # Check for valid data
        if (len(self.y_points) == 0 or len(self.b) == 0 or
                len(self.c) == 0 or len(self.d) == 0):
            return float('nan')

        n = len(self.x_points)

        # Handle out-of-range extrapolation using slope at endpoints
        if aX < self.x_points[0]:
            return self.y_points[0] + (aX - self.x_points[0]) * self.b[0]
        elif aX > self.x_points[n - 1]:
            return self.y_points[n - 1] + (aX - self.x_points[n - 1]) * self.b[n - 1]

        # Check if close to endpoints (within numerical tolerance)
        tolerance = 1e-10
        if abs(aX - self.x_points[0]) < tolerance:
            return self.y_points[0]
        elif abs(aX - self.x_points[n - 1]) < tolerance:
            return self.y_points[n - 1]

        # Find the appropriate interval using binary search
        if n < 3:
            k = 0
        else:
            i = 0
            j = n
            while True:
                k = (i + j) // 2
                if aX < self.x_points[k]:
                    j = k
                elif aX > self.x_points[k + 1]:
                    i = k
                else:
                    break

        # Evaluate the cubic polynomial using Horner's method
        dx = aX - self.x_points[k]
        return self.y_points[k] + dx * (self.b[k] + dx * (self.c[k] + dx * self.d[k]))

    def evaluate_derivative(self, x: float, order: int = 1) -> float:
        """
        Calculate the derivative of the spline at a given x coordinate.

        Parameters
        ----------
        x
            The x coordinate to evaluate at
        order
            The order of the derivative (1 or 2)
        """
        if order != 1.0:
            raise NotImplementedError("Only first derivative is implemented. There is a discrepancy with OpenSim for the second order derivative.")

        # Handle both scalar and array inputs
        if hasattr(x, '__len__') and not isinstance(x, str):
            aX = x[0]  # Use first element if array-like
        else:
            aX = x

        # Check for valid data
        if (len(self.y_points) == 0 or len(self.b) == 0 or
                len(self.c) == 0 or len(self.d) == 0):
            return float('nan')

        if order < 1 or order > 2:
            raise ValueError("Derivative order must be 1 or 2")

        n = len(self.x_points)

        # Handle out-of-range cases
        if aX < self.x_points[0]:
            if order == 1:
                return self.b[0]
            else:
                return 0.0
        elif aX > self.x_points[n - 1]:
            if order == 1:
                return self.b[n - 1]
            else:
                return 0.0

        # Check if close to endpoints (within numerical tolerance)
        tolerance = 1e-10
        if abs(aX - self.x_points[0]) < tolerance:
            if order == 1:
                return self.b[0]
            else:
                return 2.0 * self.c[0]
        elif abs(aX - self.x_points[n - 1]) < tolerance:
            if order == 1:
                return self.b[n - 1]
            else:
                return 2.0 * self.c[n - 1]

        # Find the appropriate interval using binary search
        if n < 3:
            k = 0
        else:
            i = 0
            j = n
            while True:
                k = (i + j) // 2
                if aX < self.x_points[k]:
                    j = k
                elif aX > self.x_points[k + 1]:
                    i = k
                else:
                    break

        dx = aX - self.x_points[k]

        if order == 1:
            # First derivative: b + 2*c*dx + 3*d*dx^2
            return self.b[k] + dx * (2.0 * self.c[k] + 3.0 * dx * self.d[k])
        else:
            # Second derivative: 2*c + 6*d*dx
            return 2.0 * self.c[k] + 6.0 * dx * self.d[k]


Functions: TypeAlias = SimmSpline