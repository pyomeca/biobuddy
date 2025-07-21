import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import (Muscle, ViaPoint, MuscleGroup)
from biobuddy.utils.named_list import NamedList
from test_utils import MockC3dData


# ------- Via Point ------- #
def test_init_via_points():

    # Test initialization with default values
    via_point = ViaPoint(name="test_via_point")
    assert via_point.name == "test_via_point"
    assert via_point.parent_name is None
    via_point.parent_name = "parent1"
    assert via_point.parent_name == "parent1"
    assert via_point.muscle_name is None
    via_point.muscle_name = "muscle1"
    assert via_point.muscle_name == "muscle1"
    assert via_point.muscle_group is None
    via_point.muscle_group = "group1"
    assert via_point.muscle_group == "group1"

    # Test with string position function
    via_point = ViaPoint(name="test_via_point", position_function="marker1")
    # Call the position function with a mock marker dictionary
    markers = {"marker1": np.array([1, 2, 3])}
    result = via_point.position_function(markers, None)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    # Test with callable position function
    custom_func = lambda m, bio: np.array([4, 5, 6])
    via_point = ViaPoint(name="test_via_point", position_function=custom_func)
    result = via_point.position_function(None, None)
    np.testing.assert_array_equal(result, np.array([4, 5, 6]))


def test_to_via_point():
    # Mock the ViaPointReal class and from_data method
    mock_data = MockC3dData()
    mock_model = None

    # Crete a via point
    via_point = ViaPoint(
        name="test_via_point",
        parent_name="parent1",
        muscle_name="muscle1",
        muscle_group="group1",
    )
    # Not possible to evaluate the via point without a position function
    with pytest.raises(RuntimeError, match="You must provide a position function to evaluate the ViaPoint into a ViaPointReal."):
        via_point_real = via_point.to_via_point(mock_data, mock_model)

    # Set the function
    via_point.position_function = lambda m, bio: m["HV"]

    # Call to_via_point
    via_point_real = via_point.to_via_point(mock_data, mock_model)
    npt.assert_almost_equal(np.mean(via_point.position_function(mock_data.values, mock_model), axis=1).reshape(4, ),
                           np.array([0.5758053 , 0.60425486, 1.67896849, 1.        ]))
    npt.assert_almost_equal(np.mean(via_point_real.position, axis=1).reshape(4, ), np.array([0.5758053 , 0.60425486, 1.67896849, 1.        ]))


    # Set the marker name
    via_point.position_function = "HV"

    # Call to_via_point
    via_point_real = via_point.to_via_point(mock_data, mock_model)
    npt.assert_almost_equal(np.mean(via_point_real.position, axis=1).reshape(4, ), np.array([0.5758053 , 0.60425486, 1.67896849, 1.        ]))


