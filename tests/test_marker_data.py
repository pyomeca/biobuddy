import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path
import os
import pandas as pd

from biobuddy.utils.marker_data import MarkerData, CsvData, C3dData, ReferenceFrame


# ------- CsvData ------- #
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


# ------- C3dData ------- #
def test_c3d_data_initialization():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)

    assert marker_data.c3d_path == c3d_path
    assert marker_data.first_frame == 0
    assert marker_data.last_frame == 48
    assert marker_data.nb_frames == 49
    assert marker_data.nb_markers == 49  # Should have taken a c3d with different number of frames and markers... --'
    assert len(marker_data.marker_names) == marker_data.nb_markers
    marker_names = marker_data.marker_names
    expected_marker_names = ['HV',
         'OCC',
         'LTEMP',
         'RTEMP',
         'SEL',
         'C7',
         'T10',
         'SUP',
         'STR',
         'LA',
         'LLHE',
         'LMHE',
         'LUS',
         'LRS',
         'LHMH5',
         'LHMH2',
         'LFT3',
         'RA',
         'RLHE',
         'RMHE',
         'RUS',
         'RRS',
         'RHMH5',
         'RHMH2',
         'RFT3',
         'LPSIS',
         'RPSIS',
         'LASIS',
         'RASIS',
         'LGT',
         'LLFE',
         'LMFE',
         'LLM',
         'LSPH',
         'LCAL',
         'LMFH5',
         'LMFH1',
         'LTT2',
         'RGT',
         'RLFE',
         'RMFE',
         'RLM',
         'RSPH',
         'RCAL',
         'RMFH5',
         'RMFH1',
         'RTT2',
         'LATT',
         'RATT']
    for marker in marker_names:
        if marker not in expected_marker_names:
            raise AssertionError(f"Unexpected marker name: {marker}")

    # TODO: Verify actual marker positions match ezc3d data
    # Test the first marker
    npt.assert_almost_equal(marker_data.all_marker_positions[0, 0, :], np.array([0.62772119, 0.62763147, 0.62759515, 0.62746539, 0.62742102,
       0.62736273, 0.6273045 , 0.62723438, 0.62726038, 0.62727234,
       0.62732269, 0.62734711, 0.6274082 , 0.62748688, 0.62757477,
       0.6276698 , 0.62775873, 0.62783307, 0.62797827, 0.62812335,
       0.62829315, 0.62847589, 0.62866486, 0.62887134, 0.62920728,
       0.62948212, 0.62973914, 0.62992456, 0.63016418, 0.63034698,
       0.63047266, 0.63062195, 0.63072186, 0.63073834, 0.63082123,
       0.63087988, 0.63090247, 0.63096954, 0.63097424, 0.63098871,
       0.63100397, 0.6309682 , 0.63095044, 0.63095587, 0.63096045,
       0.63096857, 0.63098773, 0.63098718, 0.63097058]))
    npt.assert_almost_equal(marker_data.all_marker_positions[1, 0, :], np.array([0.50385559, 0.50386957, 0.5038974 , 0.50399402, 0.50400955,
       0.50404926, 0.50409882, 0.50413892, 0.50417731, 0.50421225,
       0.50426355, 0.50431174, 0.50433096, 0.50436084, 0.50439487,
       0.50443781, 0.50450854, 0.50456119, 0.50461545, 0.50471045,
       0.50480426, 0.50490176, 0.50496494, 0.50503503, 0.50514926,
       0.50526385, 0.50535223, 0.50541306, 0.50544537, 0.50555276,
       0.50563837, 0.50568887, 0.50574948, 0.50574014, 0.50578812,
       0.50577527, 0.505776  , 0.50577127, 0.50574524, 0.5056835 ,
       0.50566284, 0.50562708, 0.50553378, 0.50546817, 0.50537665,
       0.50525867, 0.50516443, 0.50501077, 0.50483649]))
    npt.assert_almost_equal(marker_data.all_marker_positions[2, 0, :], np.array([1.72424866, 1.72414758, 1.72402979, 1.72394043, 1.7238457 ,
       1.72370789, 1.72364148, 1.7235    , 1.72342798, 1.72334106,
       1.72329639, 1.72323108, 1.72318909, 1.72313757, 1.72306689,
       1.72301453, 1.72296509, 1.72293298, 1.7229054 , 1.7228894 ,
       1.72290869, 1.7228894 , 1.72290759, 1.72292456, 1.72298157,
       1.72300708, 1.72304846, 1.72309521, 1.72315503, 1.72317212,
       1.72318958, 1.72317615, 1.72314709, 1.72313135, 1.72313098,
       1.72312598, 1.72311414, 1.7231283 , 1.72316882, 1.7231554 ,
       1.72316638, 1.72318652, 1.7231908 , 1.72319336, 1.72321033,
       1.72323242, 1.72324817, 1.72325732, 1.72329163]))
    npt.assert_almost_equal(np.ones((marker_data.nb_frames, )), marker_data.all_marker_positions[3, 0, :])
    # Test the 5th marker
    npt.assert_almost_equal(marker_data.all_marker_positions[0, 4, :], np.array([0.78093298, 0.78105139, 0.78116998, 0.78119513, 0.78128748,
       0.7813526 , 0.78140692, 0.78144946, 0.78152026, 0.78156091,
       0.78168811, 0.78176794, 0.78191138, 0.78200116, 0.78208508,
       0.78224469, 0.78239398, 0.78258331, 0.78263568, 0.78275806,
       0.7828382 , 0.78306732, 0.78319971, 0.78336176, 0.7835152 ,
       0.7837085 , 0.783909  , 0.78399841, 0.78409406, 0.78436292,
       0.7845368 , 0.78457288, 0.78469489, 0.78478796, 0.78485388,
       0.78494348, 0.78496973, 0.78511755, 0.78521783, 0.78528912,
       0.78534229, 0.78541187, 0.78536627, 0.78541565, 0.78545178,
       0.78544543, 0.78550525, 0.78552118, 0.78554889]))
    npt.assert_almost_equal(marker_data.all_marker_positions[1, 4, :], np.array([0.49789438, 0.4978494 , 0.497841  , 0.49786414, 0.49783557,
       0.49782831, 0.49784103, 0.49784625, 0.49784198, 0.49785229,
       0.49787149, 0.49785919, 0.49789655, 0.49790607, 0.4979577 ,
       0.49798233, 0.49804218, 0.49814801, 0.4981532 , 0.49818637,
       0.49820813, 0.49832016, 0.49832648, 0.49835083, 0.49837164,
       0.49836182, 0.49831262, 0.49835263, 0.49839551, 0.49829611,
       0.49834128, 0.49838818, 0.49836359, 0.49832999, 0.49839352,
       0.49839777, 0.49842072, 0.49848523, 0.49841803, 0.49836716,
       0.49834155, 0.4982858 , 0.4980517 , 0.49798172, 0.49785782,
       0.49772229, 0.49757025, 0.49738596, 0.49726416]))
    npt.assert_almost_equal(marker_data.all_marker_positions[2, 4, :], np.array([1.65562048, 1.65575232, 1.65593945, 1.65608435, 1.65625916,
       1.65644641, 1.65662085, 1.65676746, 1.65691077, 1.65701318,
       1.65711536, 1.65717798, 1.65725977, 1.65730518, 1.65737878,
       1.6574071 , 1.65746301, 1.65748975, 1.65749182, 1.65749133,
       1.65746143, 1.65741541, 1.65732935, 1.65725256, 1.65714929,
       1.6570426 , 1.65695044, 1.65683459, 1.65677539, 1.65668481,
       1.65663831, 1.65662378, 1.65663794, 1.65666589, 1.65668005,
       1.65672571, 1.65679297, 1.65683667, 1.65688013, 1.65695239,
       1.65705359, 1.65712402, 1.65721521, 1.6573269 , 1.65742908,
       1.65748779, 1.65757422, 1.65763159, 1.65770081]))
    npt.assert_almost_equal(np.ones((marker_data.nb_frames, )), marker_data.all_marker_positions[3, 4, :])


def test_c3d_data_initialization_with_frame_range():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path, first_frame=5, last_frame=15)

    assert marker_data.first_frame == 5
    assert marker_data.last_frame == 15
    assert marker_data.nb_frames == 11


def test_c3d_data_marker_index():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)

    first_marker_name = "HV"
    fifth_marker_name = "SEL"

    first_marker_index = marker_data.marker_index(first_marker_name)
    assert first_marker_index == 0

    second_marker_index = marker_data.marker_index(fifth_marker_name)
    assert second_marker_index == 4


def test_c3d_data_marker_indices():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)

    marker_names_to_test = ["HV", "SEL", "LA"]
    indices = marker_data.marker_indices(marker_names_to_test)
    assert isinstance(indices, tuple)
    assert len(indices) == 3
    assert indices[0] == 0
    assert indices[1] == 4
    assert indices[2] == 9


def test_c3d_data_get_position_single_marker():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    marker_name = "SEL"
    position = marker_data.get_position([marker_name])

    expected_nb_frames = 49
    assert position.shape == (4, 1, expected_nb_frames)
    npt.assert_almost_equal(position[0, 0, :], np.array([0.78093298, 0.78105139, 0.78116998, 0.78119513, 0.78128748,
       0.7813526 , 0.78140692, 0.78144946, 0.78152026, 0.78156091,
       0.78168811, 0.78176794, 0.78191138, 0.78200116, 0.78208508,
       0.78224469, 0.78239398, 0.78258331, 0.78263568, 0.78275806,
       0.7828382 , 0.78306732, 0.78319971, 0.78336176, 0.7835152 ,
       0.7837085 , 0.783909  , 0.78399841, 0.78409406, 0.78436292,
       0.7845368 , 0.78457288, 0.78469489, 0.78478796, 0.78485388,
       0.78494348, 0.78496973, 0.78511755, 0.78521783, 0.78528912,
       0.78534229, 0.78541187, 0.78536627, 0.78541565, 0.78545178,
       0.78544543, 0.78550525, 0.78552118, 0.78554889]))


def test_c3d_data_get_position_multiple_markers():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    marker_names = ["HV", "SEL", "LA"]
    position = marker_data.get_position(marker_names)

    expected_nb_frames = 49
    assert position.shape == (4, 3, expected_nb_frames)
    npt.assert_almost_equal(position[0, 0, :], np.array([0.62772119, 0.62763147, 0.62759515, 0.62746539, 0.62742102,
       0.62736273, 0.6273045 , 0.62723438, 0.62726038, 0.62727234,
       0.62732269, 0.62734711, 0.6274082 , 0.62748688, 0.62757477,
       0.6276698 , 0.62775873, 0.62783307, 0.62797827, 0.62812335,
       0.62829315, 0.62847589, 0.62866486, 0.62887134, 0.62920728,
       0.62948212, 0.62973914, 0.62992456, 0.63016418, 0.63034698,
       0.63047266, 0.63062195, 0.63072186, 0.63073834, 0.63082123,
       0.63087988, 0.63090247, 0.63096954, 0.63097424, 0.63098871,
       0.63100397, 0.6309682 , 0.63095044, 0.63095587, 0.63096045,
       0.63096857, 0.63098773, 0.63098718, 0.63097058]))
    npt.assert_almost_equal(position[1, 2, :], np.array([0.67651031, 0.67648486, 0.67646112, 0.67642487, 0.67640863,
       0.67635858, 0.67633325, 0.67631879, 0.67631824, 0.67630554,
       0.67630945, 0.67633221, 0.67630988, 0.67632159, 0.67634033,
       0.67634406, 0.6763703 , 0.6763573 , 0.67642688, 0.67645538,
       0.67646204, 0.67653992, 0.67657965, 0.67661469, 0.67663702,
       0.67661847, 0.67664728, 0.67669061, 0.67667139, 0.67672546,
       0.67673395, 0.67675684, 0.67674725, 0.67675745, 0.6767522 ,
       0.67674017, 0.67670679, 0.67674292, 0.67668848, 0.67667224,
       0.67666522, 0.67665216, 0.67664502, 0.67660596, 0.67659705,
       0.67662067, 0.67658752, 0.67659558, 0.67654651]))


def test_c3d_data_get_position_with_frame_range():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path, first_frame=5, last_frame=15)
    marker_name = "SEL"
    position = marker_data.get_position([marker_name])

    assert position.shape == (4, 1, 11)
    npt.assert_almost_equal(position[0, 0, :], np.array([0.7813526 , 0.78140692, 0.78144946, 0.78152026, 0.78156091,
       0.78168811, 0.78176794, 0.78191138, 0.78200116, 0.78208508,
       0.78224469]))


def test_c3d_data_all_marker_positions():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    all_positions = marker_data.all_marker_positions

    expected_nb_markers = 49
    expected_nb_frames = 49
    assert all_positions.shape == (4, expected_nb_markers, expected_nb_frames)
    npt.assert_almost_equal(all_positions[0, 0, :], np.array([0.62772119, 0.62763147, 0.62759515, 0.62746539, 0.62742102,
       0.62736273, 0.6273045 , 0.62723438, 0.62726038, 0.62727234,
       0.62732269, 0.62734711, 0.6274082 , 0.62748688, 0.62757477,
       0.6276698 , 0.62775873, 0.62783307, 0.62797827, 0.62812335,
       0.62829315, 0.62847589, 0.62866486, 0.62887134, 0.62920728,
       0.62948212, 0.62973914, 0.62992456, 0.63016418, 0.63034698,
       0.63047266, 0.63062195, 0.63072186, 0.63073834, 0.63082123,
       0.63087988, 0.63090247, 0.63096954, 0.63097424, 0.63098871,
       0.63100397, 0.6309682 , 0.63095044, 0.63095587, 0.63096045,
       0.63096857, 0.63098773, 0.63098718, 0.63097058]))
    npt.assert_almost_equal(all_positions[1, 2, :], np.array([0.60282379, 0.60277985, 0.60279041, 0.60274902, 0.60273944,
       0.60270013, 0.6027843 , 0.60281647, 0.60283545, 0.60284607,
       0.6028811 , 0.60287976, 0.60292841, 0.60298163, 0.60299268,
       0.60302374, 0.60308289, 0.60310071, 0.60315674, 0.60325293,
       0.6033208 , 0.60336694, 0.60343579, 0.60349103, 0.60355536,
       0.60358478, 0.60358026, 0.60363757, 0.60367401, 0.60372205,
       0.60374109, 0.60376562, 0.60377551, 0.60377997, 0.60373254,
       0.60374493, 0.60371643, 0.60368073, 0.60367902, 0.60363666,
       0.60359674, 0.60357587, 0.60346722, 0.60343268, 0.6032973 ,
       0.60321051, 0.60305823, 0.60296912, 0.60289569]))


def test_c3d_data_all_marker_positions_setter():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    original_positions = marker_data.all_marker_positions.copy()

    # Modify positions
    new_positions = original_positions.copy()
    new_positions[0, 0, 0] = 999.0 * 1000

    marker_data.all_marker_positions = new_positions

    # Verify the change
    updated_positions = marker_data.all_marker_positions
    assert updated_positions[0, 0, 0] == 999.0


def test_c3d_data_all_marker_positions_setter_wrong_shape():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)

    with pytest.raises(ValueError, match=rf"Expected shape \(4, 49, 49\), got \(3, 49, 49\)."):
        marker_data.all_marker_positions = np.zeros((3, 49, 49))

    with pytest.raises(ValueError, match=rf"Expected shape \(4, 49, 49\), got \(4, 10, 49\)."):
        marker_data.all_marker_positions = np.zeros((4, 10, 49))

    with pytest.raises(ValueError, match=rf"Expected shape \(4, 49, 49\), got \(4, 49, 10\)."):
        marker_data.all_marker_positions = np.zeros((4, 49, 10))


def test_c3d_data_markers_center_position():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    marker_names = ["HV", "SEL"]
    center = marker_data.markers_center_position(marker_names)

    expected_center = np.nanmean(marker_data.get_position(marker_names), axis=1)
    expected_nb_frames = 49
    assert center.shape == (4, expected_nb_frames)
    npt.assert_almost_equal(center, expected_center)


def test_c3d_data_mean_marker_position():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    marker_name = "SEL"
    mean_pos = marker_data.mean_marker_position(marker_name)
    expected_mean = np.nanmean(marker_data.get_position([marker_name]), axis=2)

    assert mean_pos.shape == (4, 1)
    npt.assert_almost_equal(mean_pos, expected_mean)


def test_c3d_data_std_marker_position():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    marker_name = "SEL"
    std_pos = marker_data.std_marker_position(marker_name)
    expected_std = np.nanstd(marker_data.get_position([marker_name]), axis=2)

    assert std_pos.shape == (4, 1)
    npt.assert_almost_equal(std_pos, expected_std)


def test_c3d_data_change_ref_frame_z_up_to_y_up():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
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


def test_c3d_data_change_ref_frame_y_up_to_z_up():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
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


def test_c3d_data_change_ref_frame_same_frame():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    original_positions = marker_data.all_marker_positions.copy()

    marker_data.change_ref_frame(ReferenceFrame.Z_UP, ReferenceFrame.Z_UP)
    new_positions = marker_data.all_marker_positions

    npt.assert_array_equal(new_positions, original_positions)


def test_c3d_data_change_ref_frame_invalid():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)

    # This should raise an error for unsupported conversion
    with pytest.raises(ValueError, match="Cannot change from bad_value to ReferenceFrame.Z_UP."):
        marker_data.change_ref_frame("bad_value", ReferenceFrame.Z_UP)


def test_c3d_data_save():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"
    tmp_path = c3d_path.replace(".c3d", "_temp.c3d")

    # Read and save file
    marker_data = C3dData(c3d_path=c3d_path)
    marker_data.save(tmp_path)

    # Load the saved file and compare (marker names and positions is enough)
    loaded_marker_data = C3dData(c3d_path=tmp_path)
    npt.assert_array_almost_equal(marker_data.all_marker_positions, loaded_marker_data.all_marker_positions)
    assert marker_data.marker_names == loaded_marker_data.marker_names

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
