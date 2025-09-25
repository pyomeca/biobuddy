from typing import TypeAlias
from abc import ABC, abstractmethod

import numpy as np


class InterpolationFunction(ABC):

    def __init__(self, x_points: np.ndarray, y_points: np.ndarray):
        if len(y_points) != len(x_points):
            raise ValueError("x_points and y_points must have the same length")
        if not np.all(x_points[:-1] <= x_points[1:]):
            raise ValueError("x_points must be sorted in ascending order")

        nb_nodes = x_points.shape[0]
        if nb_nodes < 2:
            raise ValueError("At least 2 data points are required")

        self.nb_nodes = nb_nodes
        self.x_points = x_points
        self.y_points = y_points
        self.TINY_NUMBER = 0.0000001  # Defined in opensim-core/OpenSim/Common/SimmMacros.h

    def safe_max(self, array: np.ndarray) -> float:
        out = np.max(array)
        out = self.TINY_NUMBER if self.TINY_NUMBER > out else out
        return out

    @staticmethod
    def get_scalar_value(x: float) -> float:
        """Handle both scalar and array inputs."""
        if hasattr(x, "__len__") and not isinstance(x, str):
            if len(x) != 1:
                raise ValueError("Only single value arrays are supported")
            else:
                return x[0]  # Use first element if array-like
        else:
            return x

    @abstractmethod
    def _calculate_coefficients(self):
        """Calculate the polynomial coefficients."""
        pass

    @abstractmethod
    def get_coefficients(self):
        """Return the calculated coefficients."""

    @abstractmethod
    def evaluate(self, x: float) -> float:
        """
        Calculate the polynomial at a given x coordinate.

        Parameters
        ----------
        x
            The x coordinate to evaluate the spline at
        """
        pass

    @abstractmethod
    def evaluate_derivative(self, x: float, order: int = 1) -> float:
        """
        Calculate the derivative of the polynomial at a given x coordinate.

        Parameters
        ----------
        x
            The x coordinate to evaluate at
        order
            The order of the derivative (1 or 2)
        """
        pass


class SimmSpline(InterpolationFunction):
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
        super().__init__(x_points, y_points)

        # Calculate spline coefficients
        self.b = None
        self.c = None
        self.d = None
        self._calculate_coefficients()  # Will set b, c, and d

    def _calculate_coefficients(self):
        """Calculate the spline coefficients."""

        # Initialize coefficient arrays
        self.b = np.zeros((self.nb_nodes,))
        self.c = np.zeros((self.nb_nodes,))
        self.d = np.zeros((self.nb_nodes,))

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
        self.b[nm1] = (self.y_points[nm1] - self.y_points[nm2]) / self.d[nm2] + self.d[nm2] * (
            self.c[nm2] + 2.0 * self.c[nm1]
        )

        for i in range(nm1):
            self.b[i] = (self.y_points[i + 1] - self.y_points[i]) / self.d[i] - self.d[i] * (
                self.c[i + 1] + 2.0 * self.c[i]
            )
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
        x_scalar = self.get_scalar_value(x)

        # Handle out-of-range extrapolation using slope at endpoints
        if x_scalar < self.x_points[0]:
            return self.y_points[0] + (x_scalar - self.x_points[0]) * self.b[0]
        elif x_scalar > self.x_points[self.nb_nodes - 1]:
            return (
                self.y_points[self.nb_nodes - 1]
                + (x_scalar - self.x_points[self.nb_nodes - 1]) * self.b[self.nb_nodes - 1]
            )

        # Check if close to endpoints (within numerical tolerance)
        tolerance = 1e-10
        if abs(x_scalar - self.x_points[0]) < tolerance:
            return self.y_points[0]
        elif abs(x_scalar - self.x_points[self.nb_nodes - 1]) < tolerance:
            return self.y_points[self.nb_nodes - 1]

        # Find the appropriate interval using binary search
        if self.nb_nodes < 3:
            k = 0
        else:
            i = 0
            j = self.nb_nodes
            while True:
                k = (i + j) // 2
                if x_scalar < self.x_points[k]:
                    j = k
                elif x_scalar > self.x_points[k + 1]:
                    i = k
                else:
                    break

        # Evaluate the cubic polynomial using Horner's method
        dx = x_scalar - self.x_points[k]
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
            raise NotImplementedError(
                "Only first derivative is implemented. There is a discrepancy with OpenSim for the second order derivative."
            )
        # if order < 1 or order > 2:
        #     raise ValueError("Derivative order must be 1 or 2")

        x_scalar = self.get_scalar_value(x)

        # Handle out-of-range cases
        if x_scalar < self.x_points[0]:
            raise NotImplementedError("Extrapolation for derivatives is not implemented.")
            # if order == 1:
            #     return self.b[0]
            # else:
            #     return 0.0
        elif x_scalar > self.x_points[self.nb_nodes - 1]:
            raise NotImplementedError("Extrapolation for derivatives is not implemented.")
            # if order == 1:
            #     return self.b[self.nb_nodes - 1]
            # else:
            #     return 0.0

        # Check if close to endpoints (within numerical tolerance)
        tolerance = 1e-10
        if abs(x_scalar - self.x_points[0]) < tolerance:
            raise NotImplementedError("Extrapolation for derivatives is not implemented.")
            # if order == 1:
            #     return self.b[0]
            # else:
            #     return 2.0 * self.c[0]
        elif abs(x_scalar - self.x_points[self.nb_nodes - 1]) < tolerance:
            raise NotImplementedError("Extrapolation for derivatives is not implemented.")
            # if order == 1:
            #     return self.b[self.nb_nodes - 1]
            # else:
            #     return 2.0 * self.c[self.nb_nodes - 1]

        # Find the appropriate interval using binary search
        if self.nb_nodes < 3:
            k = 0
        else:
            i = 0
            j = self.nb_nodes
            while True:
                k = (i + j) // 2
                if x_scalar < self.x_points[k]:
                    j = k
                elif x_scalar > self.x_points[k + 1]:
                    i = k
                else:
                    break

        dx = x_scalar - self.x_points[k]

        if order == 1:
            # First derivative: b + 2*c*dx + 3*d*dx^2
            return self.b[k] + dx * (2.0 * self.c[k] + 3.0 * dx * self.d[k])
        else:
            # Second derivative: 2*c + 6*d*dx
            return 2.0 * self.c[k] + 6.0 * dx * self.d[k]


class PiecewiseLinearFunction(InterpolationFunction):
    """
    Python implementation of linear interpolation between each pair of points.
    """

    def __init__(self, x_points: np.ndarray, y_points: np.ndarray):
        """
        Initialize the PieceWiseLinearFunction with x and y data points.

        Parameters
        ----------
        x_points
            The x coordinates of the data points (must be sorted in ascending order).
        y_points
            The y coordinates of the data points.
        """
        super().__init__(x_points, y_points)

        # Calculate the coefficients
        self.a = None
        self.b = None
        self._calculate_coefficients()  # Will set a and b

    def _calculate_coefficients(self):
        """Calculate the spline coefficients."""
        self.a = np.zeros((self.nb_nodes - 1,))
        self.b = np.zeros((self.nb_nodes - 1,))
        for i_node in range(self.nb_nodes - 1):
            self.a[i_node] = (self.y_points[i_node + 1] - self.y_points[i_node]) / (
                self.x_points[i_node + 1] - self.x_points[i_node]
            )
            self.b[i_node] = self.y_points[i_node] - self.a[i_node] * self.x_points[i_node]

    def get_coefficients(self):
        """Return the calculated coefficients."""
        return self.a.copy(), self.b.copy()

    def get_coefficient_index(self, x: float) -> int:
        # Get which coefficients to use
        if x <= self.x_points[0]:
            linear_piece_idx = 0
        elif x >= self.x_points[-1]:
            linear_piece_idx = -1
        else:
            linear_piece_idx = np.where(x < self.x_points)[0][0] - 1
        return linear_piece_idx

    def evaluate(self, x: float) -> float:
        """
        Calculate the linear interpolation value at a given x coordinate.

        Parameters
        ----------
        x
            The x coordinate to evaluate the line at
        """
        x_scalar = self.get_scalar_value(x)
        linear_piece_idx = self.get_coefficient_index(x_scalar)
        y = self.a[linear_piece_idx] * x_scalar + self.b[linear_piece_idx]
        return y

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
        if not isinstance(order, int) or order < 1:
            raise RuntimeError("The order of the derivative must be an int larger or equal to 1.0")

        if order == 1.0:
            x_scalar = self.get_scalar_value(x)
            linear_piece_idx = self.get_coefficient_index(x_scalar)
            return self.a[linear_piece_idx]
        else:
            return 0.0


Functions: TypeAlias = SimmSpline | PiecewiseLinearFunction
