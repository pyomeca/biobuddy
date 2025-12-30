import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path
import os
import pandas as pd

from biobuddy.utils.marker_data import MarkerData, CsvData, ReferenceFrame


def test_csv_data_initialization():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)

    assert marker_data.csv_path == csv_path
    assert marker_data.first_frame == 0
    assert marker_data.last_frame == 27
    assert marker_data.nb_frames == 28
    assert marker_data.nb_markers == 21
    assert len(marker_data.marker_names) == 21
    marker_names = marker_data.marker_names
    for marker in marker_names:
        if marker not in [
            'WRA',
            'WRB',
            'RU_1',
            'RU_2',
            'RU_3',
            'RU_4',
            'ELB_M',
            'ELB_L',
            'H_1',
            'H_2',
            'H_3',
            'H_4',
            'H_5',
            'H_6',
            'SA_1',
            'SA_2',
            'SA_3',
            'CS_1',
            'CS_2',
            'CS_3',
            'CS_4',
        ]:
            raise AssertionError(f"Unexpected marker name: {marker}")

        # Load the csv file and make sure it matches
        csv_data_frame = pd.read_csv(csv_path)
        # Test the first marker
        npt.assert_almost_equal(np.array(csv_data_frame[" WRA"])[1:].astype(float), marker_data.all_marker_positions[0, 0, :] * 100)  # Convert back to cm for comparison
        npt.assert_almost_equal(np.array(csv_data_frame["Unnamed: 1"])[1:].astype(float), marker_data.all_marker_positions[1, 0, :] * 100)  # Convert back to cm for comparison
        npt.assert_almost_equal(np.array(csv_data_frame["Unnamed: 2"])[1:].astype(float), marker_data.all_marker_positions[2, 0, :] * 100)  # Convert back to cm for comparison
        npt.assert_almost_equal(np.ones((marker_data.nb_frames, )), marker_data.all_marker_positions[3, 0, :])  # Convert back to cm for comparison
        # Test the 9th marker
        npt.assert_almost_equal(np.array(csv_data_frame["H_1"])[1:].astype(float), marker_data.all_marker_positions[0, 8, :] * 100)
        npt.assert_almost_equal(np.array(csv_data_frame["Unnamed: 25"])[1:].astype(float), marker_data.all_marker_positions[1, 8, :] * 100)
        npt.assert_almost_equal(np.array(csv_data_frame["Unnamed: 26"])[1:].astype(float), marker_data.all_marker_positions[2, 8, :] * 100)
        npt.assert_almost_equal(np.ones((marker_data.nb_frames, )), marker_data.all_marker_positions[3, 8, :])  # Convert back to cm for comparison


def test_csv_data_initialization_with_frame_range():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path, first_frame=5, last_frame=15)

    assert marker_data.first_frame == 5
    assert marker_data.last_frame == 15
    assert marker_data.nb_frames == 11

def test_csv_data_marker_index():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)

    wra_index = marker_data.marker_index("WRA")
    assert wra_index == 0

    wrb_index = marker_data.marker_index("WRB")
    assert wrb_index == 1


def test_csv_data_marker_indices():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)

    indices = marker_data.marker_indices(["WRA", "WRB", "ELB_M"])
    assert isinstance(indices, tuple)
    assert len(indices) == 3
    assert indices[0] == 0
    assert indices[1] == 1


def test_csv_data_get_position_single_marker():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    position = marker_data.get_position(["WRA"])

    assert position.shape == (4, 1, 28)
    npt.assert_almost_equal(position[0, 0, :], np.array([2.63645, 2.63645, 2.63645, 2.63645, 2.63646, 2.63646, 2.63647,
       2.63646, 2.63648, 2.63648, 2.63645, 2.63647, 2.63647, 2.63649,
       2.63651, 2.63648, 2.6365 , 2.63648, 2.6365 , 2.6365 , 2.6365 ,
       2.63649, 2.6365 , 2.63656, 2.63651, 2.63652, 2.63644, 2.63654]))


def test_csv_data_get_position_multiple_markers():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    position = marker_data.get_position(["WRA", "WRB", "ELB_M"])

    assert position.shape == (4, 3, 28)
    npt.assert_almost_equal(position[0, 0, :], np.array([2.63645, 2.63645, 2.63645, 2.63645, 2.63646, 2.63646, 2.63647,
       2.63646, 2.63648, 2.63648, 2.63645, 2.63647, 2.63647, 2.63649,
       2.63651, 2.63648, 2.6365 , 2.63648, 2.6365 , 2.6365 , 2.6365 ,
       2.63649, 2.6365 , 2.63656, 2.63651, 2.63652, 2.63644, 2.63654]))
    npt.assert_almost_equal(position[1, 2, :], np.array([5.85784, 5.85783, 5.85776, 5.85794, 5.85784, 5.85779, 5.85789,
       5.8577 , 5.85786, 5.85787, 5.85797, 5.85828, 5.85799, 5.85802,
       5.85804, 5.85805, 5.85818, 5.8581 , 5.85812, 5.85808, 5.8581 ,
       5.85815, 5.8581 , 5.85758, 5.85757, 5.85761, 5.85801, 5.85804]))


def test_csv_data_get_position_with_frame_range():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path, first_frame=5, last_frame=15)
    position = marker_data.get_position(["WRA"])

    assert position.shape == (4, 1, 11)
    npt.assert_almost_equal(position[0, 0, :], np.array([2.63646, 2.63647, 2.63646, 2.63648, 2.63648, 2.63645, 2.63647,
       2.63647, 2.63649, 2.63651, 2.63648]))


def test_csv_data_all_marker_positions():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    all_positions = marker_data.all_marker_positions

    assert all_positions.shape == (4, 21, 28)
    npt.assert_almost_equal(all_positions[0, 0, :], np.array([2.63645, 2.63645, 2.63645, 2.63645, 2.63646, 2.63646, 2.63647,
       2.63646, 2.63648, 2.63648, 2.63645, 2.63647, 2.63647, 2.63649,
       2.63651, 2.63648, 2.6365 , 2.63648, 2.6365 , 2.6365 , 2.6365 ,
       2.63649, 2.6365 , 2.63656, 2.63651, 2.63652, 2.63644, 2.63654]))
    npt.assert_almost_equal(all_positions[1, 2, :], np.array([5.09335, 5.09332, 5.09315, 5.09313, 5.09311, 5.09308, 5.09303,
       5.09302, 5.09299, 5.09297, 5.09282, 5.09282, 5.09278, 5.09277,
       5.09273, 5.09271, 5.09262, 5.0926 , 5.09252, 5.09248, 5.09248,
       5.09244, 5.09239, 5.09235, 5.09231, 5.09218, 5.09215, 5.0921]))


def test_csv_data_all_marker_positions_setter():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    original_positions = marker_data.all_marker_positions.copy()

    # Modify positions
    new_positions = original_positions.copy()
    new_positions[0, 0, 0] = 999.0  # cm

    marker_data.all_marker_positions = new_positions

    # Verify the change
    updated_positions = marker_data.all_marker_positions
    assert updated_positions[0, 0, 0] == 9.99  # m


def test_csv_data_all_marker_positions_setter_wrong_shape():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)

    with pytest.raises(ValueError, match=r"Expected shape \(4, 21, 28\), got \(3, 21, 28\)."):
        marker_data.all_marker_positions = np.zeros((3, 21, 28))

    with pytest.raises(ValueError, match=r"Expected shape \(4, 21, 28\), got \(4, 10, 28\)."):
        marker_data.all_marker_positions = np.zeros((4, 10, 28))

    with pytest.raises(ValueError, match=r"Expected shape \(4, 21, 28\), got \(4, 21, 10\)."):
        marker_data.all_marker_positions = np.zeros((4, 21, 10))

def test_csv_data_markers_center_position():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    center = marker_data.markers_center_position(["WRA", "WRB"])

    expected_center = np.nanmean(marker_data.get_position(["WRA", "WRB"]), axis=1)
    assert center.shape == (4, 28)
    npt.assert_almost_equal(center, expected_center)


def test_csv_data_mean_marker_position():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    mean_pos = marker_data.mean_marker_position("WRA")
    expected_mean = np.nanmean(marker_data.get_position(["WRA"]), axis=2)

    assert mean_pos.shape == (4, 1)
    npt.assert_almost_equal(mean_pos, expected_mean)


def test_csv_data_std_marker_position():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    std_pos = marker_data.std_marker_position("WRA")
    expected_std = np.nanstd(marker_data.get_position(["WRA"]), axis=2)

    assert std_pos.shape == (4, 1)
    npt.assert_almost_equal(std_pos, expected_std)


def test_csv_data_change_ref_frame_z_up_to_y_up():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    original_positions = marker_data.all_marker_positions.copy()

    marker_data.change_ref_frame(ReferenceFrame.Z_UP, ReferenceFrame.Y_UP)
    new_positions = marker_data.all_marker_positions

    # X should stay the same
    npt.assert_array_almost_equal(new_positions[0, :, :], original_positions[0, :, :])
    # Y should become Z
    npt.assert_array_almost_equal(new_positions[1, :, :], original_positions[2, :, :])
    # Z should become -Y
    npt.assert_array_almost_equal(new_positions[2, :, :], -original_positions[1, :, :])
    # Should have ones on the last row
    npt.assert_array_almost_equal(new_positions[3, :, :], np.ones_like(new_positions[3, :, :]))


def test_csv_data_change_ref_frame_y_up_to_z_up():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    original_positions = marker_data.all_marker_positions.copy()

    marker_data.change_ref_frame(ReferenceFrame.Y_UP, ReferenceFrame.Z_UP)
    new_positions = marker_data.all_marker_positions

    # X should stay the same
    npt.assert_array_almost_equal(new_positions[0, :, :], original_positions[0, :, :])
    # Y should become -Z
    npt.assert_array_almost_equal(new_positions[1, :, :], -original_positions[2, :, :])
    # Z should become Y
    npt.assert_array_almost_equal(new_positions[2, :, :], original_positions[1, :, :])
    # Should have ones on the last row
    npt.assert_array_almost_equal(new_positions[3, :, :], np.ones_like(new_positions[3, :, :]))



def test_csv_data_change_ref_frame_same_frame():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    original_positions = marker_data.all_marker_positions.copy()

    marker_data.change_ref_frame(ReferenceFrame.Z_UP, ReferenceFrame.Z_UP)
    new_positions = marker_data.all_marker_positions

    npt.assert_array_equal(new_positions, original_positions)


def test_csv_data_change_ref_frame_invalid():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)

    # This should raise an error for unsupported conversion
    # Since only Z_UP <-> Y_UP are supported
    with pytest.raises(ValueError, match="Cannot change from bad_value to ReferenceFrame.Z_UP."):
        # Create a mock invalid conversion by trying something not implemented
        marker_data.change_ref_frame("bad_value", ReferenceFrame.Z_UP)
        # Actually, same frame returns early, so let's not test this way


def test_csv_data_save():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"
    tmp_path = csv_path.replace(".csv", "_temp.csv")

    # Read and save file
    marker_data = CsvData(csv_path=csv_path)
    marker_data.save(tmp_path)

    # Load the saved file and compare (marker names and positions is enough)
    loaded_marker_data = CsvData(csv_path=tmp_path)
    npt.assert_array_almost_equal(marker_data.all_marker_positions, loaded_marker_data.all_marker_positions)
    assert marker_data.marker_names == loaded_marker_data.marker_names

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

