import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from biobuddy.utils.aliases import point_to_array, points_to_array, inertia_to_array


class TestAliases(unittest.TestCase):
    def test_point_to_array_none(self):
        """Test that None returns an empty array"""
        result = point_to_array(None)
        self.assertEqual(result.shape, (4, 0))

    def test_point_to_array_list(self):
        """Test conversion from list to array"""
        result = point_to_array([1, 2, 3])
        expected = np.array([[1], [2], [3], [1]])
        assert_array_equal(result, expected)

    def test_point_to_array_numpy(self):
        """Test conversion from numpy array"""
        result = point_to_array(np.array([1, 2, 3]))
        expected = np.array([[1], [2], [3], [1]])
        assert_array_equal(result, expected)

    def test_point_to_array_with_homogeneous(self):
        """Test conversion with homogeneous coordinate already present"""
        result = point_to_array(np.array([1, 2, 3, 1]))
        expected = np.array([[1], [2], [3], [1]])
        assert_array_equal(result, expected)

    def test_point_to_array_scalar_error(self):
        """Test that scalar input raises error"""
        with self.assertRaises(RuntimeError):
            point_to_array(5)

    def test_point_to_array_wrong_shape_error(self):
        """Test that wrong shape raises error"""
        with self.assertRaises(RuntimeError):
            point_to_array(np.array([[1, 2], [3, 4]]))

    def test_points_to_array_none(self):
        """Test that None returns an empty array"""
        result = points_to_array(None, "test_points")
        self.assertEqual(result.shape, (4, 0))

    def test_points_to_array_list(self):
        """Test conversion from list of lists"""
        points = [[1, 2, 3], [4, 5, 6]]
        result = points_to_array(points, "test_points")
        expected = np.array([[1, 4], [2, 5], [3, 6], [1, 1]])
        assert_array_equal(result, expected)

    def test_points_to_array_numpy(self):
        """Test conversion from numpy array"""
        points = np.array([[1, 2, 3], [4, 5, 6]])
        result = points_to_array(points, "test_points")
        expected = np.array([[1, 4], [2, 5], [3, 6], [1, 1]])
        assert_array_equal(result, expected)

    def test_points_to_array_transposed(self):
        """Test conversion from transposed array"""
        points = np.array([[1, 4], [2, 5], [3, 6]])
        result = points_to_array(points, "test_points")
        expected = np.array([[1, 4], [2, 5], [3, 6], [1, 1]])
        assert_array_equal(result, expected)

    def test_points_to_array_with_homogeneous(self):
        """Test conversion with homogeneous coordinate already present"""
        points = np.array([[1, 4], [2, 5], [3, 6], [1, 1]])
        result = points_to_array(points, "test_points")
        expected = np.array([[1, 4], [2, 5], [3, 6], [1, 1]])
        assert_array_equal(result, expected)

    def test_points_to_array_wrong_type_error(self):
        """Test that wrong type raises error"""
        with self.assertRaises(RuntimeError):
            points_to_array("not a list or array", "test_points")

    def test_points_to_array_wrong_shape_error(self):
        """Test that wrong shape raises error"""
        with self.assertRaises(RuntimeError):
            points_to_array(np.array([[[1, 2], [3, 4]]]), "test_points")

    def test_inertia_to_array_none(self):
        """Test that None returns an empty 4x4 array"""
        result = inertia_to_array(None)
        self.assertEqual(result.shape, (4, 4))

    def test_inertia_to_array_diagonal_list(self):
        """Test conversion from diagonal list to matrix"""
        inertia = [1, 2, 3]
        result = inertia_to_array(inertia)
        expected = np.array([
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 1]
        ])
        assert_array_equal(result, expected)

    def test_inertia_to_array_diagonal_numpy(self):
        """Test conversion from diagonal numpy array to matrix"""
        inertia = np.array([1, 2, 3])
        result = inertia_to_array(inertia)
        expected = np.array([
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 1]
        ])
        assert_array_equal(result, expected)

    def test_inertia_to_array_matrix(self):
        """Test conversion from 3x3 matrix to 4x4 matrix"""
        inertia = np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6]
        ])
        result = inertia_to_array(inertia)
        expected = np.array([
            [1, 2, 3, 0],
            [2, 4, 5, 0],
            [3, 5, 6, 0],
            [0, 0, 0, 1]
        ])
        assert_array_equal(result, expected)

    def test_inertia_to_array_wrong_type_error(self):
        """Test that wrong type raises error"""
        with self.assertRaises(RuntimeError):
            inertia_to_array("not a list or array")

    def test_inertia_to_array_wrong_shape_error(self):
        """Test that wrong shape raises error"""
        with self.assertRaises(RuntimeError):
            inertia_to_array(np.array([[[1, 2], [3, 4]]]))


if __name__ == "__main__":
    unittest.main()
