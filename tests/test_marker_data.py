import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path
import os
import pandas as pd
import pickle

from biobuddy.utils.marker_data import MarkerData, CsvData, C3dData, DictData, ReferenceFrame


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
            "WRA",
            "WRB",
            "RU_1",
            "RU_2",
            "RU_3",
            "RU_4",
            "ELB_M",
            "ELB_L",
            "H_1",
            "H_2",
            "H_3",
            "H_4",
            "H_5",
            "H_6",
            "SA_1",
            "SA_2",
            "SA_3",
            "CS_1",
            "CS_2",
            "CS_3",
            "CS_4",
        ]:
            raise AssertionError(f"Unexpected marker name: {marker}")

        # Load the csv file and make sure it matches
        csv_data_frame = pd.read_csv(csv_path)
        # Test the first marker
        npt.assert_almost_equal(
            np.array(csv_data_frame[" WRA"])[1:].astype(float), marker_data.all_marker_positions[0, 0, :] * 100
        )  # Convert back to cm for comparison
        npt.assert_almost_equal(
            np.array(csv_data_frame["Unnamed: 1"])[1:].astype(float), marker_data.all_marker_positions[1, 0, :] * 100
        )  # Convert back to cm for comparison
        npt.assert_almost_equal(
            np.array(csv_data_frame["Unnamed: 2"])[1:].astype(float), marker_data.all_marker_positions[2, 0, :] * 100
        )  # Convert back to cm for comparison
        npt.assert_almost_equal(
            np.ones((marker_data.nb_frames,)), marker_data.all_marker_positions[3, 0, :]
        )  # Convert back to cm for comparison
        # Test the 9th marker
        npt.assert_almost_equal(
            np.array(csv_data_frame["H_1"])[1:].astype(float), marker_data.all_marker_positions[0, 8, :] * 100
        )
        npt.assert_almost_equal(
            np.array(csv_data_frame["Unnamed: 25"])[1:].astype(float), marker_data.all_marker_positions[1, 8, :] * 100
        )
        npt.assert_almost_equal(
            np.array(csv_data_frame["Unnamed: 26"])[1:].astype(float), marker_data.all_marker_positions[2, 8, :] * 100
        )
        npt.assert_almost_equal(
            np.ones((marker_data.nb_frames,)), marker_data.all_marker_positions[3, 8, :]
        )  # Convert back to cm for comparison


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
    npt.assert_almost_equal(
        position[0, 0, :],
        np.array(
            [
                2.63645,
                2.63645,
                2.63645,
                2.63645,
                2.63646,
                2.63646,
                2.63647,
                2.63646,
                2.63648,
                2.63648,
                2.63645,
                2.63647,
                2.63647,
                2.63649,
                2.63651,
                2.63648,
                2.6365,
                2.63648,
                2.6365,
                2.6365,
                2.6365,
                2.63649,
                2.6365,
                2.63656,
                2.63651,
                2.63652,
                2.63644,
                2.63654,
            ]
        ),
    )


def test_csv_data_get_position_multiple_markers():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    position = marker_data.get_position(["WRA", "WRB", "ELB_M"])

    assert position.shape == (4, 3, 28)
    npt.assert_almost_equal(
        position[0, 0, :],
        np.array(
            [
                2.63645,
                2.63645,
                2.63645,
                2.63645,
                2.63646,
                2.63646,
                2.63647,
                2.63646,
                2.63648,
                2.63648,
                2.63645,
                2.63647,
                2.63647,
                2.63649,
                2.63651,
                2.63648,
                2.6365,
                2.63648,
                2.6365,
                2.6365,
                2.6365,
                2.63649,
                2.6365,
                2.63656,
                2.63651,
                2.63652,
                2.63644,
                2.63654,
            ]
        ),
    )
    npt.assert_almost_equal(
        position[1, 2, :],
        np.array(
            [
                5.85784,
                5.85783,
                5.85776,
                5.85794,
                5.85784,
                5.85779,
                5.85789,
                5.8577,
                5.85786,
                5.85787,
                5.85797,
                5.85828,
                5.85799,
                5.85802,
                5.85804,
                5.85805,
                5.85818,
                5.8581,
                5.85812,
                5.85808,
                5.8581,
                5.85815,
                5.8581,
                5.85758,
                5.85757,
                5.85761,
                5.85801,
                5.85804,
            ]
        ),
    )


def test_csv_data_get_position_with_frame_range():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path, first_frame=5, last_frame=15)
    position = marker_data.get_position(["WRA"])

    assert position.shape == (4, 1, 11)
    npt.assert_almost_equal(
        position[0, 0, :],
        np.array([2.63646, 2.63647, 2.63646, 2.63648, 2.63648, 2.63645, 2.63647, 2.63647, 2.63649, 2.63651, 2.63648]),
    )


def test_csv_data_all_marker_positions():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)
    all_positions = marker_data.all_marker_positions

    assert all_positions.shape == (4, 21, 28)
    npt.assert_almost_equal(
        all_positions[0, 0, :],
        np.array(
            [
                2.63645,
                2.63645,
                2.63645,
                2.63645,
                2.63646,
                2.63646,
                2.63647,
                2.63646,
                2.63648,
                2.63648,
                2.63645,
                2.63647,
                2.63647,
                2.63649,
                2.63651,
                2.63648,
                2.6365,
                2.63648,
                2.6365,
                2.6365,
                2.6365,
                2.63649,
                2.6365,
                2.63656,
                2.63651,
                2.63652,
                2.63644,
                2.63654,
            ]
        ),
    )
    npt.assert_almost_equal(
        all_positions[1, 2, :],
        np.array(
            [
                5.09335,
                5.09332,
                5.09315,
                5.09313,
                5.09311,
                5.09308,
                5.09303,
                5.09302,
                5.09299,
                5.09297,
                5.09282,
                5.09282,
                5.09278,
                5.09277,
                5.09273,
                5.09271,
                5.09262,
                5.0926,
                5.09252,
                5.09248,
                5.09248,
                5.09244,
                5.09239,
                5.09235,
                5.09231,
                5.09218,
                5.09215,
                5.0921,
            ]
        ),
    )


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


def test_csv_data_get_partial_dict_data():
    current_path_file = Path(__file__).parent
    csv_path = f"{current_path_file}/../examples/data/static.csv"

    marker_data = CsvData(csv_path=csv_path)

    # Get partial data with subset of markers
    partial_data = marker_data.get_partial_dict_data(["WRA", "WRB", "ELB_M"])

    assert isinstance(partial_data, DictData)
    assert partial_data.nb_markers == 3
    assert partial_data.nb_frames == 28
    assert partial_data.marker_names == ["WRA", "WRB", "ELB_M"]

    # Verify the positions match
    for marker_name in ["WRA", "WRB", "ELB_M"]:
        original_pos = marker_data.get_position([marker_name])
        partial_pos = partial_data.get_position([marker_name])
        npt.assert_array_almost_equal(original_pos.squeeze(), partial_pos.squeeze())


# ------- C3dData ------- #
def test_c3d_data_initialization():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)

    assert marker_data.c3d_path == c3d_path
    assert marker_data.first_frame == 0
    assert marker_data.last_frame == 137
    assert marker_data.nb_frames == 138
    assert marker_data.nb_markers == 49
    assert len(marker_data.marker_names) == marker_data.nb_markers
    marker_names = marker_data.marker_names
    expected_marker_names = [
        "HV",
        "OCC",
        "LTEMP",
        "RTEMP",
        "SEL",
        "C7",
        "T10",
        "SUP",
        "STR",
        "LA",
        "LLHE",
        "LMHE",
        "LUS",
        "LRS",
        "LHMH5",
        "LHMH2",
        "LFT3",
        "RA",
        "RLHE",
        "RMHE",
        "RUS",
        "RRS",
        "RHMH5",
        "RHMH2",
        "RFT3",
        "LPSIS",
        "RPSIS",
        "LASIS",
        "RASIS",
        "LGT",
        "LLFE",
        "LMFE",
        "LLM",
        "LSPH",
        "LCAL",
        "LMFH5",
        "LMFH1",
        "LTT2",
        "RGT",
        "RLFE",
        "RMFE",
        "RLM",
        "RSPH",
        "RCAL",
        "RMFH5",
        "RMFH1",
        "RTT2",
        "LATT",
        "RATT",
    ]
    for marker in marker_names:
        if marker not in expected_marker_names:
            raise AssertionError(f"Unexpected marker name: {marker}")

    # Test the first marker
    npt.assert_almost_equal(
        marker_data.all_marker_positions[0, 0, :],
        np.array(
            [
                0.62772119,
                0.62763147,
                0.62759515,
                0.62746539,
                0.62742102,
                0.62736273,
                0.6273045,
                0.62723438,
                0.62726038,
                0.62727234,
                0.62732269,
                0.62734711,
                0.6274082,
                0.62748688,
                0.62757477,
                0.6276698,
                0.62775873,
                0.62783307,
                0.62797827,
                0.62812335,
                0.62829315,
                0.62847589,
                0.62866486,
                0.62887134,
                0.62920728,
                0.62948212,
                0.62973914,
                0.62992456,
                0.63016418,
                0.63034698,
                0.63047266,
                0.63062195,
                0.63072186,
                0.63073834,
                0.63082123,
                0.63087988,
                0.63090247,
                0.63096954,
                0.63097424,
                0.63098871,
                0.63100397,
                0.6309682,
                0.63095044,
                0.63095587,
                0.63096045,
                0.63096857,
                0.63098773,
                0.63098718,
                0.63097058,
                0.63102661,
                0.63101013,
                0.63100909,
                0.63103131,
                0.6309823,
                0.63099103,
                0.63107715,
                0.63107635,
                0.63106635,
                0.6310589,
                0.63103192,
                0.63098816,
                0.63094031,
                0.63091418,
                0.63092224,
                0.63081482,
                0.63068958,
                0.63062012,
                0.6305498,
                0.6304671,
                0.63038989,
                0.63034644,
                0.63031079,
                0.63026202,
                0.63024902,
                0.63017798,
                0.63014008,
                0.63008704,
                0.63006982,
                0.62988776,
                0.62971106,
                0.62969611,
                0.62961884,
                0.6295119,
                0.62944904,
                0.62935669,
                0.62932703,
                0.62926422,
                0.62921808,
                0.62918402,
                0.62907233,
                0.62899689,
                0.62894916,
                0.6289776,
                0.6289541,
                0.62889178,
                0.62892566,
                0.62891736,
                0.62890466,
                0.62887537,
                0.62892163,
                0.62895026,
                0.62908185,
                0.62911871,
                0.62918298,
                0.62930267,
                0.62928888,
                0.62937891,
                0.62942108,
                0.62947827,
                0.62955115,
                0.62961566,
                0.62966089,
                0.62983173,
                0.62984033,
                0.62987744,
                0.62995929,
                0.63004303,
                0.63012506,
                0.63018524,
                0.63024316,
                0.63041864,
                0.63044586,
                0.6305321,
                0.63058972,
                0.63066364,
                0.63071619,
                0.63082251,
                0.63085303,
                0.63089252,
                0.63096741,
                0.63097302,
                0.63098163,
                0.63103412,
                0.63101508,
                0.63101117,
                0.63094604,
                0.63100586,
                0.63098798,
            ]
        ),
    )
    npt.assert_almost_equal(
        marker_data.all_marker_positions[1, 0, :],
        np.array(
            [
                0.50385559,
                0.50386957,
                0.5038974,
                0.50399402,
                0.50400955,
                0.50404926,
                0.50409882,
                0.50413892,
                0.50417731,
                0.50421225,
                0.50426355,
                0.50431174,
                0.50433096,
                0.50436084,
                0.50439487,
                0.50443781,
                0.50450854,
                0.50456119,
                0.50461545,
                0.50471045,
                0.50480426,
                0.50490176,
                0.50496494,
                0.50503503,
                0.50514926,
                0.50526385,
                0.50535223,
                0.50541306,
                0.50544537,
                0.50555276,
                0.50563837,
                0.50568887,
                0.50574948,
                0.50574014,
                0.50578812,
                0.50577527,
                0.505776,
                0.50577127,
                0.50574524,
                0.5056835,
                0.50566284,
                0.50562708,
                0.50553378,
                0.50546817,
                0.50537665,
                0.50525867,
                0.50516443,
                0.50501077,
                0.50483649,
                0.50468338,
                0.50453793,
                0.50441461,
                0.50425317,
                0.50415527,
                0.50400793,
                0.50380753,
                0.50371463,
                0.50357278,
                0.50346497,
                0.50337701,
                0.5033082,
                0.50319226,
                0.50314682,
                0.50305731,
                0.5029863,
                0.5028811,
                0.50279001,
                0.50270313,
                0.5026055,
                0.50253268,
                0.50242215,
                0.50224518,
                0.5021282,
                0.50200519,
                0.50183664,
                0.50171942,
                0.50153824,
                0.50137564,
                0.50115649,
                0.5009559,
                0.50073898,
                0.50062033,
                0.5005047,
                0.50035626,
                0.50022107,
                0.5000834,
                0.49997638,
                0.49985623,
                0.4997218,
                0.49960858,
                0.49947702,
                0.49932358,
                0.49917822,
                0.4990932,
                0.49900537,
                0.49892297,
                0.49887088,
                0.49872528,
                0.49859726,
                0.49851263,
                0.4984259,
                0.4983512,
                0.49828485,
                0.49824326,
                0.49822339,
                0.49818732,
                0.49816324,
                0.4981756,
                0.49816266,
                0.49823987,
                0.49825693,
                0.49826736,
                0.49824088,
                0.49833286,
                0.49835654,
                0.49840686,
                0.49843805,
                0.49848062,
                0.49855167,
                0.49860214,
                0.49868784,
                0.49871521,
                0.49876346,
                0.49881015,
                0.49883456,
                0.49885068,
                0.49885852,
                0.49889966,
                0.49887836,
                0.49888675,
                0.49892871,
                0.49898245,
                0.4990119,
                0.49906641,
                0.49911136,
                0.49919168,
                0.49925702,
                0.49936157,
            ]
        ),
    )
    npt.assert_almost_equal(
        marker_data.all_marker_positions[2, 0, :],
        np.array(
            [
                1.72424866,
                1.72414758,
                1.72402979,
                1.72394043,
                1.7238457,
                1.72370789,
                1.72364148,
                1.7235,
                1.72342798,
                1.72334106,
                1.72329639,
                1.72323108,
                1.72318909,
                1.72313757,
                1.72306689,
                1.72301453,
                1.72296509,
                1.72293298,
                1.7229054,
                1.7228894,
                1.72290869,
                1.7228894,
                1.72290759,
                1.72292456,
                1.72298157,
                1.72300708,
                1.72304846,
                1.72309521,
                1.72315503,
                1.72317212,
                1.72318958,
                1.72317615,
                1.72314709,
                1.72313135,
                1.72313098,
                1.72312598,
                1.72311414,
                1.7231283,
                1.72316882,
                1.7231554,
                1.72316638,
                1.72318652,
                1.7231908,
                1.72319336,
                1.72321033,
                1.72323242,
                1.72324817,
                1.72325732,
                1.72329163,
                1.72327478,
                1.72329797,
                1.72330701,
                1.72331262,
                1.72332788,
                1.72332397,
                1.72334802,
                1.72336316,
                1.72338501,
                1.72340161,
                1.72343848,
                1.72347534,
                1.72351794,
                1.72357788,
                1.72361377,
                1.72365576,
                1.72366943,
                1.72370728,
                1.72371338,
                1.72373157,
                1.72373047,
                1.72375146,
                1.72377783,
                1.72380237,
                1.72382507,
                1.72384485,
                1.72383081,
                1.72383643,
                1.72387085,
                1.7238739,
                1.72386609,
                1.72385913,
                1.7238916,
                1.72389185,
                1.72389746,
                1.72388062,
                1.72387903,
                1.72390295,
                1.72389355,
                1.7238916,
                1.72392688,
                1.72391052,
                1.72391638,
                1.72392419,
                1.72390308,
                1.72389465,
                1.72391541,
                1.72389563,
                1.72389319,
                1.72392737,
                1.72395349,
                1.72398193,
                1.72401794,
                1.72403247,
                1.72404797,
                1.72406201,
                1.72408325,
                1.72408679,
                1.72408203,
                1.72408276,
                1.72406665,
                1.72405762,
                1.72404456,
                1.72405957,
                1.72405334,
                1.72405127,
                1.7240719,
                1.72407129,
                1.72409534,
                1.72410278,
                1.72410205,
                1.72410571,
                1.72412927,
                1.72414722,
                1.72415479,
                1.72417737,
                1.72416382,
                1.72417993,
                1.7241991,
                1.72420178,
                1.72421704,
                1.72420105,
                1.72421545,
                1.72421069,
                1.72424719,
                1.72425098,
                1.7242467,
                1.72424756,
                1.72426208,
            ]
        ),
    )
    npt.assert_almost_equal(np.ones((marker_data.nb_frames,)), marker_data.all_marker_positions[3, 0, :])
    # Test the 5th marker
    npt.assert_almost_equal(
        marker_data.all_marker_positions[0, 4, :],
        np.array(
            [
                0.78093298,
                0.78105139,
                0.78116998,
                0.78119513,
                0.78128748,
                0.7813526,
                0.78140692,
                0.78144946,
                0.78152026,
                0.78156091,
                0.78168811,
                0.78176794,
                0.78191138,
                0.78200116,
                0.78208508,
                0.78224469,
                0.78239398,
                0.78258331,
                0.78263568,
                0.78275806,
                0.7828382,
                0.78306732,
                0.78319971,
                0.78336176,
                0.7835152,
                0.7837085,
                0.783909,
                0.78399841,
                0.78409406,
                0.78436292,
                0.7845368,
                0.78457288,
                0.78469489,
                0.78478796,
                0.78485388,
                0.78494348,
                0.78496973,
                0.78511755,
                0.78521783,
                0.78528912,
                0.78534229,
                0.78541187,
                0.78536627,
                0.78541565,
                0.78545178,
                0.78544543,
                0.78550525,
                0.78552118,
                0.78554889,
                0.78558484,
                0.78562,
                0.7858067,
                0.78580615,
                0.78582819,
                0.78578876,
                0.78581128,
                0.78579797,
                0.78580634,
                0.78578583,
                0.78568121,
                0.78564935,
                0.78558783,
                0.78556183,
                0.7854726,
                0.7854472,
                0.78541003,
                0.78533923,
                0.78534381,
                0.78529279,
                0.78522931,
                0.78520605,
                0.78514227,
                0.78509619,
                0.78500977,
                0.78499048,
                0.78485931,
                0.78483472,
                0.78468616,
                0.78460443,
                0.78450336,
                0.78439862,
                0.78434155,
                0.78427893,
                0.78419373,
                0.78417273,
                0.78411884,
                0.78403357,
                0.78399158,
                0.78390186,
                0.78380145,
                0.78381726,
                0.78380438,
                0.7838233,
                0.78383893,
                0.78383807,
                0.78386481,
                0.78384894,
                0.78383917,
                0.78385376,
                0.78392133,
                0.78393658,
                0.78401733,
                0.78409174,
                0.78411896,
                0.7842088,
                0.78422119,
                0.78423199,
                0.78431537,
                0.784349,
                0.78441626,
                0.78442694,
                0.78448877,
                0.78458417,
                0.78478204,
                0.78481409,
                0.78481836,
                0.78491406,
                0.7848927,
                0.78497247,
                0.78499622,
                0.78515796,
                0.78527612,
                0.78534247,
                0.78542285,
                0.78543433,
                0.78544623,
                0.78554852,
                0.78558398,
                0.78569019,
                0.78573627,
                0.78576056,
                0.78578162,
                0.7858125,
                0.78585175,
                0.78586804,
                0.78589832,
                0.78588715,
                0.78600812,
            ]
        ),
    )
    npt.assert_almost_equal(
        marker_data.all_marker_positions[1, 4, :],
        np.array(
            [
                0.49789438,
                0.4978494,
                0.497841,
                0.49786414,
                0.49783557,
                0.49782831,
                0.49784103,
                0.49784625,
                0.49784198,
                0.49785229,
                0.49787149,
                0.49785919,
                0.49789655,
                0.49790607,
                0.4979577,
                0.49798233,
                0.49804218,
                0.49814801,
                0.4981532,
                0.49818637,
                0.49820813,
                0.49832016,
                0.49832648,
                0.49835083,
                0.49837164,
                0.49836182,
                0.49831262,
                0.49835263,
                0.49839551,
                0.49829611,
                0.49834128,
                0.49838818,
                0.49836359,
                0.49832999,
                0.49839352,
                0.49839777,
                0.49842072,
                0.49848523,
                0.49841803,
                0.49836716,
                0.49834155,
                0.4982858,
                0.4980517,
                0.49798172,
                0.49785782,
                0.49772229,
                0.49757025,
                0.49738596,
                0.49726416,
                0.49712524,
                0.49695337,
                0.4967019,
                0.49659525,
                0.49647119,
                0.49639148,
                0.49630408,
                0.49620636,
                0.49612427,
                0.49602972,
                0.4958671,
                0.49578574,
                0.49568753,
                0.49556744,
                0.4954975,
                0.49544104,
                0.4953223,
                0.49523758,
                0.49515909,
                0.49505447,
                0.49495621,
                0.49486005,
                0.49476227,
                0.49464496,
                0.49456583,
                0.49446802,
                0.49428659,
                0.49422742,
                0.49402383,
                0.49393884,
                0.49384012,
                0.49377701,
                0.49367459,
                0.49357162,
                0.49352911,
                0.49340234,
                0.49332629,
                0.49325116,
                0.49320673,
                0.49309048,
                0.49289084,
                0.492836,
                0.49277219,
                0.49271301,
                0.49262,
                0.49258456,
                0.49220377,
                0.49214215,
                0.49200739,
                0.4919212,
                0.49183777,
                0.49179303,
                0.49169708,
                0.49164276,
                0.49158496,
                0.49154874,
                0.49154324,
                0.49154105,
                0.49155399,
                0.49152899,
                0.49155518,
                0.49158728,
                0.49161203,
                0.49161752,
                0.49180084,
                0.491858,
                0.49189575,
                0.49189606,
                0.49195587,
                0.49196411,
                0.49202667,
                0.49222339,
                0.49228192,
                0.49234174,
                0.49239999,
                0.4924978,
                0.49256226,
                0.4926001,
                0.49270676,
                0.49270804,
                0.49278516,
                0.49283481,
                0.49285635,
                0.49293616,
                0.492953,
                0.49297467,
                0.4930535,
                0.49307242,
                0.49321469,
            ]
        ),
    )
    npt.assert_almost_equal(
        marker_data.all_marker_positions[2, 4, :],
        np.array(
            [
                1.65562048,
                1.65575232,
                1.65593945,
                1.65608435,
                1.65625916,
                1.65644641,
                1.65662085,
                1.65676746,
                1.65691077,
                1.65701318,
                1.65711536,
                1.65717798,
                1.65725977,
                1.65730518,
                1.65737878,
                1.6574071,
                1.65746301,
                1.65748975,
                1.65749182,
                1.65749133,
                1.65746143,
                1.65741541,
                1.65732935,
                1.65725256,
                1.65714929,
                1.6570426,
                1.65695044,
                1.65683459,
                1.65677539,
                1.65668481,
                1.65663831,
                1.65662378,
                1.65663794,
                1.65666589,
                1.65668005,
                1.65672571,
                1.65679297,
                1.65683667,
                1.65688013,
                1.65695239,
                1.65705359,
                1.65712402,
                1.65721521,
                1.6573269,
                1.65742908,
                1.65748779,
                1.65757422,
                1.65763159,
                1.65770081,
                1.65777026,
                1.65781201,
                1.65785535,
                1.65795203,
                1.65797925,
                1.65803589,
                1.65806311,
                1.65809082,
                1.6581189,
                1.65813232,
                1.65816211,
                1.65819202,
                1.65824585,
                1.65827625,
                1.65833459,
                1.65839063,
                1.65841931,
                1.65847754,
                1.65850574,
                1.65856409,
                1.65860413,
                1.6586333,
                1.65867688,
                1.65869006,
                1.65873364,
                1.65873303,
                1.65874878,
                1.65877454,
                1.65880615,
                1.65883411,
                1.65884937,
                1.65889209,
                1.65892114,
                1.65896155,
                1.65897913,
                1.65900793,
                1.65904993,
                1.65908069,
                1.65910815,
                1.65912305,
                1.65915369,
                1.65916736,
                1.65916541,
                1.65916492,
                1.65918774,
                1.65914685,
                1.65913269,
                1.65915906,
                1.65915796,
                1.65915503,
                1.65914087,
                1.65913477,
                1.65910742,
                1.65910522,
                1.6590907,
                1.65908838,
                1.65908679,
                1.6591012,
                1.65908496,
                1.65909924,
                1.65909814,
                1.65913867,
                1.65912659,
                1.65911365,
                1.6591239,
                1.65914038,
                1.65913928,
                1.65914124,
                1.65914209,
                1.65914612,
                1.65911865,
                1.65912744,
                1.65909827,
                1.65909973,
                1.65908582,
                1.65909973,
                1.65908484,
                1.65909888,
                1.65905798,
                1.65907227,
                1.65909973,
                1.65908667,
                1.65907568,
                1.65908838,
                1.65911902,
                1.65913379,
                1.65913269,
                1.65913428,
                1.65914465,
            ]
        ),
    )
    npt.assert_almost_equal(np.ones((marker_data.nb_frames,)), marker_data.all_marker_positions[3, 4, :])


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

    expected_nb_frames = 138
    assert position.shape == (4, 1, expected_nb_frames)
    npt.assert_almost_equal(
        position[0, 0, :],
        np.array(
            [
                0.78093298,
                0.78105139,
                0.78116998,
                0.78119513,
                0.78128748,
                0.7813526,
                0.78140692,
                0.78144946,
                0.78152026,
                0.78156091,
                0.78168811,
                0.78176794,
                0.78191138,
                0.78200116,
                0.78208508,
                0.78224469,
                0.78239398,
                0.78258331,
                0.78263568,
                0.78275806,
                0.7828382,
                0.78306732,
                0.78319971,
                0.78336176,
                0.7835152,
                0.7837085,
                0.783909,
                0.78399841,
                0.78409406,
                0.78436292,
                0.7845368,
                0.78457288,
                0.78469489,
                0.78478796,
                0.78485388,
                0.78494348,
                0.78496973,
                0.78511755,
                0.78521783,
                0.78528912,
                0.78534229,
                0.78541187,
                0.78536627,
                0.78541565,
                0.78545178,
                0.78544543,
                0.78550525,
                0.78552118,
                0.78554889,
                0.78558484,
                0.78562,
                0.7858067,
                0.78580615,
                0.78582819,
                0.78578876,
                0.78581128,
                0.78579797,
                0.78580634,
                0.78578583,
                0.78568121,
                0.78564935,
                0.78558783,
                0.78556183,
                0.7854726,
                0.7854472,
                0.78541003,
                0.78533923,
                0.78534381,
                0.78529279,
                0.78522931,
                0.78520605,
                0.78514227,
                0.78509619,
                0.78500977,
                0.78499048,
                0.78485931,
                0.78483472,
                0.78468616,
                0.78460443,
                0.78450336,
                0.78439862,
                0.78434155,
                0.78427893,
                0.78419373,
                0.78417273,
                0.78411884,
                0.78403357,
                0.78399158,
                0.78390186,
                0.78380145,
                0.78381726,
                0.78380438,
                0.7838233,
                0.78383893,
                0.78383807,
                0.78386481,
                0.78384894,
                0.78383917,
                0.78385376,
                0.78392133,
                0.78393658,
                0.78401733,
                0.78409174,
                0.78411896,
                0.7842088,
                0.78422119,
                0.78423199,
                0.78431537,
                0.784349,
                0.78441626,
                0.78442694,
                0.78448877,
                0.78458417,
                0.78478204,
                0.78481409,
                0.78481836,
                0.78491406,
                0.7848927,
                0.78497247,
                0.78499622,
                0.78515796,
                0.78527612,
                0.78534247,
                0.78542285,
                0.78543433,
                0.78544623,
                0.78554852,
                0.78558398,
                0.78569019,
                0.78573627,
                0.78576056,
                0.78578162,
                0.7858125,
                0.78585175,
                0.78586804,
                0.78589832,
                0.78588715,
                0.78600812,
            ]
        ),
    )


def test_c3d_data_get_position_multiple_markers():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    marker_names = ["HV", "SEL", "LA"]
    position = marker_data.get_position(marker_names)

    expected_nb_frames = 138
    assert position.shape == (4, 3, expected_nb_frames)
    npt.assert_almost_equal(
        position[0, 0, :],
        np.array(
            [
                0.62772119,
                0.62763147,
                0.62759515,
                0.62746539,
                0.62742102,
                0.62736273,
                0.6273045,
                0.62723438,
                0.62726038,
                0.62727234,
                0.62732269,
                0.62734711,
                0.6274082,
                0.62748688,
                0.62757477,
                0.6276698,
                0.62775873,
                0.62783307,
                0.62797827,
                0.62812335,
                0.62829315,
                0.62847589,
                0.62866486,
                0.62887134,
                0.62920728,
                0.62948212,
                0.62973914,
                0.62992456,
                0.63016418,
                0.63034698,
                0.63047266,
                0.63062195,
                0.63072186,
                0.63073834,
                0.63082123,
                0.63087988,
                0.63090247,
                0.63096954,
                0.63097424,
                0.63098871,
                0.63100397,
                0.6309682,
                0.63095044,
                0.63095587,
                0.63096045,
                0.63096857,
                0.63098773,
                0.63098718,
                0.63097058,
                0.63102661,
                0.63101013,
                0.63100909,
                0.63103131,
                0.6309823,
                0.63099103,
                0.63107715,
                0.63107635,
                0.63106635,
                0.6310589,
                0.63103192,
                0.63098816,
                0.63094031,
                0.63091418,
                0.63092224,
                0.63081482,
                0.63068958,
                0.63062012,
                0.6305498,
                0.6304671,
                0.63038989,
                0.63034644,
                0.63031079,
                0.63026202,
                0.63024902,
                0.63017798,
                0.63014008,
                0.63008704,
                0.63006982,
                0.62988776,
                0.62971106,
                0.62969611,
                0.62961884,
                0.6295119,
                0.62944904,
                0.62935669,
                0.62932703,
                0.62926422,
                0.62921808,
                0.62918402,
                0.62907233,
                0.62899689,
                0.62894916,
                0.6289776,
                0.6289541,
                0.62889178,
                0.62892566,
                0.62891736,
                0.62890466,
                0.62887537,
                0.62892163,
                0.62895026,
                0.62908185,
                0.62911871,
                0.62918298,
                0.62930267,
                0.62928888,
                0.62937891,
                0.62942108,
                0.62947827,
                0.62955115,
                0.62961566,
                0.62966089,
                0.62983173,
                0.62984033,
                0.62987744,
                0.62995929,
                0.63004303,
                0.63012506,
                0.63018524,
                0.63024316,
                0.63041864,
                0.63044586,
                0.6305321,
                0.63058972,
                0.63066364,
                0.63071619,
                0.63082251,
                0.63085303,
                0.63089252,
                0.63096741,
                0.63097302,
                0.63098163,
                0.63103412,
                0.63101508,
                0.63101117,
                0.63094604,
                0.63100586,
                0.63098798,
            ]
        ),
    )
    npt.assert_almost_equal(
        position[1, 2, :],
        np.array(
            [
                0.67651031,
                0.67648486,
                0.67646112,
                0.67642487,
                0.67640863,
                0.67635858,
                0.67633325,
                0.67631879,
                0.67631824,
                0.67630554,
                0.67630945,
                0.67633221,
                0.67630988,
                0.67632159,
                0.67634033,
                0.67634406,
                0.6763703,
                0.6763573,
                0.67642688,
                0.67645538,
                0.67646204,
                0.67653992,
                0.67657965,
                0.67661469,
                0.67663702,
                0.67661847,
                0.67664728,
                0.67669061,
                0.67667139,
                0.67672546,
                0.67673395,
                0.67675684,
                0.67674725,
                0.67675745,
                0.6767522,
                0.67674017,
                0.67670679,
                0.67674292,
                0.67668848,
                0.67667224,
                0.67666522,
                0.67665216,
                0.67664502,
                0.67660596,
                0.67659705,
                0.67662067,
                0.67658752,
                0.67659558,
                0.67654651,
                0.67655151,
                0.67654608,
                0.67649554,
                0.67648102,
                0.67642493,
                0.67639471,
                0.67635443,
                0.67632599,
                0.67623895,
                0.67617865,
                0.67608282,
                0.67599689,
                0.67591296,
                0.67584998,
                0.67575305,
                0.6756485,
                0.67556866,
                0.67546814,
                0.67538116,
                0.67527222,
                0.67517706,
                0.67508948,
                0.6749837,
                0.67488788,
                0.67477893,
                0.67471906,
                0.67458215,
                0.67449799,
                0.674401,
                0.67432867,
                0.67422388,
                0.67413885,
                0.67405457,
                0.67396191,
                0.67386511,
                0.67378888,
                0.673703,
                0.67362079,
                0.67352405,
                0.67346606,
                0.67340887,
                0.67331683,
                0.67328485,
                0.67320483,
                0.67313409,
                0.67307025,
                0.67299677,
                0.67296729,
                0.67289471,
                0.67285736,
                0.67289844,
                0.67287811,
                0.67286938,
                0.67283795,
                0.67280127,
                0.67274634,
                0.67274005,
                0.67269598,
                0.67263654,
                0.67277161,
                0.67275653,
                0.67265314,
                0.67266077,
                0.67263867,
                0.67260809,
                0.67261005,
                0.67264697,
                0.67262805,
                0.67271277,
                0.67264771,
                0.67260266,
                0.67266284,
                0.67267419,
                0.67263983,
                0.67272101,
                0.67271814,
                0.6727699,
                0.67278082,
                0.67281427,
                0.67285895,
                0.67286292,
                0.67291626,
                0.67294202,
                0.67297357,
                0.67300116,
                0.67298651,
                0.67303613,
                0.67303113,
                0.67305762,
            ]
        ),
    )


def test_c3d_data_get_position_with_frame_range():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path, first_frame=5, last_frame=15)
    marker_name = "SEL"
    position = marker_data.get_position([marker_name])

    assert position.shape == (4, 1, 11)
    npt.assert_almost_equal(
        position[0, 0, :],
        np.array(
            [
                0.7813526,
                0.78140692,
                0.78144946,
                0.78152026,
                0.78156091,
                0.78168811,
                0.78176794,
                0.78191138,
                0.78200116,
                0.78208508,
                0.78224469,
            ]
        ),
    )


def test_c3d_data_all_marker_positions():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    all_positions = marker_data.all_marker_positions

    expected_nb_markers = 49
    expected_nb_frames = 138
    assert all_positions.shape == (4, expected_nb_markers, expected_nb_frames)
    npt.assert_almost_equal(
        all_positions[0, 0, :],
        np.array(
            [
                0.62772119,
                0.62763147,
                0.62759515,
                0.62746539,
                0.62742102,
                0.62736273,
                0.6273045,
                0.62723438,
                0.62726038,
                0.62727234,
                0.62732269,
                0.62734711,
                0.6274082,
                0.62748688,
                0.62757477,
                0.6276698,
                0.62775873,
                0.62783307,
                0.62797827,
                0.62812335,
                0.62829315,
                0.62847589,
                0.62866486,
                0.62887134,
                0.62920728,
                0.62948212,
                0.62973914,
                0.62992456,
                0.63016418,
                0.63034698,
                0.63047266,
                0.63062195,
                0.63072186,
                0.63073834,
                0.63082123,
                0.63087988,
                0.63090247,
                0.63096954,
                0.63097424,
                0.63098871,
                0.63100397,
                0.6309682,
                0.63095044,
                0.63095587,
                0.63096045,
                0.63096857,
                0.63098773,
                0.63098718,
                0.63097058,
                0.63102661,
                0.63101013,
                0.63100909,
                0.63103131,
                0.6309823,
                0.63099103,
                0.63107715,
                0.63107635,
                0.63106635,
                0.6310589,
                0.63103192,
                0.63098816,
                0.63094031,
                0.63091418,
                0.63092224,
                0.63081482,
                0.63068958,
                0.63062012,
                0.6305498,
                0.6304671,
                0.63038989,
                0.63034644,
                0.63031079,
                0.63026202,
                0.63024902,
                0.63017798,
                0.63014008,
                0.63008704,
                0.63006982,
                0.62988776,
                0.62971106,
                0.62969611,
                0.62961884,
                0.6295119,
                0.62944904,
                0.62935669,
                0.62932703,
                0.62926422,
                0.62921808,
                0.62918402,
                0.62907233,
                0.62899689,
                0.62894916,
                0.6289776,
                0.6289541,
                0.62889178,
                0.62892566,
                0.62891736,
                0.62890466,
                0.62887537,
                0.62892163,
                0.62895026,
                0.62908185,
                0.62911871,
                0.62918298,
                0.62930267,
                0.62928888,
                0.62937891,
                0.62942108,
                0.62947827,
                0.62955115,
                0.62961566,
                0.62966089,
                0.62983173,
                0.62984033,
                0.62987744,
                0.62995929,
                0.63004303,
                0.63012506,
                0.63018524,
                0.63024316,
                0.63041864,
                0.63044586,
                0.6305321,
                0.63058972,
                0.63066364,
                0.63071619,
                0.63082251,
                0.63085303,
                0.63089252,
                0.63096741,
                0.63097302,
                0.63098163,
                0.63103412,
                0.63101508,
                0.63101117,
                0.63094604,
                0.63100586,
                0.63098798,
            ]
        ),
    )
    npt.assert_almost_equal(
        all_positions[1, 2, :],
        np.array(
            [
                0.60282379,
                0.60277985,
                0.60279041,
                0.60274902,
                0.60273944,
                0.60270013,
                0.6027843,
                0.60281647,
                0.60283545,
                0.60284607,
                0.6028811,
                0.60287976,
                0.60292841,
                0.60298163,
                0.60299268,
                0.60302374,
                0.60308289,
                0.60310071,
                0.60315674,
                0.60325293,
                0.6033208,
                0.60336694,
                0.60343579,
                0.60349103,
                0.60355536,
                0.60358478,
                0.60358026,
                0.60363757,
                0.60367401,
                0.60372205,
                0.60374109,
                0.60376562,
                0.60377551,
                0.60377997,
                0.60373254,
                0.60374493,
                0.60371643,
                0.60368073,
                0.60367902,
                0.60363666,
                0.60359674,
                0.60357587,
                0.60346722,
                0.60343268,
                0.6032973,
                0.60321051,
                0.60305823,
                0.60296912,
                0.60289569,
                0.60275244,
                0.60268872,
                0.60260052,
                0.602435,
                0.60231165,
                0.60226178,
                0.6021889,
                0.60209546,
                0.60200708,
                0.60191705,
                0.60185559,
                0.60175421,
                0.60168097,
                0.60160168,
                0.60148096,
                0.60141608,
                0.601323,
                0.60123975,
                0.60115228,
                0.6010329,
                0.60094006,
                0.60077728,
                0.60066626,
                0.60056171,
                0.60044232,
                0.60034479,
                0.60020081,
                0.60007275,
                0.59990546,
                0.59971729,
                0.5995979,
                0.59947644,
                0.5994267,
                0.59933777,
                0.59925677,
                0.59918274,
                0.5991059,
                0.599026,
                0.59883246,
                0.59871851,
                0.59856573,
                0.59848053,
                0.59837701,
                0.59822693,
                0.59815576,
                0.59806433,
                0.59797034,
                0.59793384,
                0.59782623,
                0.59778394,
                0.59772949,
                0.59768835,
                0.59764069,
                0.59762122,
                0.59759332,
                0.5975022,
                0.59748816,
                0.59750061,
                0.59748254,
                0.5974834,
                0.59746887,
                0.59747357,
                0.59749084,
                0.59751813,
                0.59754126,
                0.59757227,
                0.59758466,
                0.59767987,
                0.59770532,
                0.59774841,
                0.59776929,
                0.5977901,
                0.59783514,
                0.59787189,
                0.59790179,
                0.59792285,
                0.59800012,
                0.59801843,
                0.59806726,
                0.59809985,
                0.59814325,
                0.59818591,
                0.5982392,
                0.59824634,
                0.59832922,
                0.59835382,
                0.59841888,
                0.59843677,
                0.59848993,
            ]
        ),
    )


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

    with pytest.raises(ValueError, match=rf"Expected shape \(4, 49, 138\), got \(3, 49, 138\)."):
        marker_data.all_marker_positions = np.zeros((3, 49, 138))

    with pytest.raises(ValueError, match=rf"Expected shape \(4, 49, 138\), got \(4, 10, 138\)."):
        marker_data.all_marker_positions = np.zeros((4, 10, 138))

    with pytest.raises(ValueError, match=rf"Expected shape \(4, 49, 138\), got \(4, 49, 10\)."):
        marker_data.all_marker_positions = np.zeros((4, 49, 10))


def test_c3d_data_markers_center_position():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)
    marker_names = ["HV", "SEL"]
    center = marker_data.markers_center_position(marker_names)

    expected_center = np.nanmean(marker_data.get_position(marker_names), axis=1)
    expected_nb_frames = 138
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


def test_c3d_data_get_partial_dict_data():
    current_path_file = Path(__file__).parent
    c3d_path = f"{current_path_file}/../examples/data/static.c3d"

    marker_data = C3dData(c3d_path=c3d_path)

    # Get partial data with subset of markers
    partial_data = marker_data.get_partial_dict_data(["HV", "SEL", "LA"])

    assert isinstance(partial_data, DictData)
    assert partial_data.nb_markers == 3
    assert partial_data.nb_frames == 138
    assert partial_data.marker_names == ["HV", "SEL", "LA"]

    # Verify the positions match
    for marker_name in ["HV", "SEL", "LA"]:
        original_pos = marker_data.get_position([marker_name])
        partial_pos = partial_data.get_position([marker_name])
        npt.assert_array_almost_equal(original_pos.squeeze(), partial_pos.squeeze())


# ------- DictData ------- #
def test_dict_data_initialization():
    # Create a simple marker dictionary
    marker_dict = {
        "marker1": np.array([[1.0], [2.0], [3.0], [1.0]]),
        "marker2": np.array([[4.0], [5.0], [6.0], [1.0]]),
        "marker3": np.array([[7.0], [8.0], [9.0], [1.0]]),
    }

    marker_data = DictData(marker_dict=marker_dict)

    assert marker_data.first_frame == 0
    assert marker_data.last_frame == 0
    assert marker_data.nb_frames == 1
    assert marker_data.nb_markers == 3
    assert len(marker_data.marker_names) == 3
    assert marker_data.marker_names == ["marker1", "marker2", "marker3"]


def test_dict_data_initialization_multiple_frames():
    # Create marker dictionary with multiple frames
    nb_frames = 10
    marker_dict = {
        "marker1": np.random.randn(4, nb_frames),
        "marker2": np.random.randn(4, nb_frames),
        "marker3": np.random.randn(4, nb_frames),
    }
    # Set last row to 1
    for key in marker_dict:
        marker_dict[key][3, :] = 1.0

    marker_data = DictData(marker_dict=marker_dict)

    assert marker_data.first_frame == 0
    assert marker_data.last_frame == 9
    assert marker_data.nb_frames == 10
    assert marker_data.nb_markers == 3


def test_dict_data_initialization_with_frame_range():
    # Create marker dictionary with multiple frames
    nb_frames = 20
    marker_dict = {
        "marker1": np.random.randn(4, nb_frames),
        "marker2": np.random.randn(4, nb_frames),
    }
    # Set last row to 1
    for key in marker_dict:
        marker_dict[key][3, :] = 1.0

    marker_data = DictData(marker_dict=marker_dict, first_frame=5, last_frame=15)

    assert marker_data.first_frame == 5
    assert marker_data.last_frame == 15
    assert marker_data.nb_frames == 11


def test_dict_data_initialization_wrong_shape():
    # Test with wrong first dimension
    marker_dict = {
        "marker1": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # Only 3 rows instead of 4
    }

    with pytest.raises(
        ValueError,
        match=r"Data for marker 'marker1' should have shape \(4, nb_frames\), but has shape \(3, 2\).",
    ):
        DictData(marker_dict=marker_dict)


def test_dict_data_initialization_inconsistent_frames():
    # Test with inconsistent number of frames
    marker_dict = {
        "marker1": np.random.randn(4, 10),
        "marker2": np.random.randn(4, 15),  # Different number of frames
    }

    with pytest.raises(
        ValueError,
        match=r"All markers should have the same number of frames. Marker 'marker2' has 15 frames, expected 10.",
    ):
        DictData(marker_dict=marker_dict)


def test_dict_data_marker_index():
    marker_dict = {
        "marker1": np.random.randn(4, 5),
        "marker2": np.random.randn(4, 5),
        "marker3": np.random.randn(4, 5),
    }
    for key in marker_dict:
        marker_dict[key][3, :] = 1.0

    marker_data = DictData(marker_dict=marker_dict)

    assert marker_data.marker_index("marker1") == 0
    assert marker_data.marker_index("marker2") == 1
    assert marker_data.marker_index("marker3") == 2


def test_dict_data_marker_indices():
    marker_dict = {
        "marker1": np.random.randn(4, 5),
        "marker2": np.random.randn(4, 5),
        "marker3": np.random.randn(4, 5),
    }
    for key in marker_dict:
        marker_dict[key][3, :] = 1.0

    marker_data = DictData(marker_dict=marker_dict)

    indices = marker_data.marker_indices(["marker1", "marker3"])
    assert isinstance(indices, tuple)
    assert len(indices) == 2
    assert indices[0] == 0
    assert indices[1] == 2


def test_dict_data_get_position_single_marker():
    marker_dict = {
        "marker1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]]),
        "marker2": np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [1.0, 1.0, 1.0]]),
    }

    marker_data = DictData(marker_dict=marker_dict)
    position = marker_data.get_position(["marker1"])

    assert position.shape == (4, 1, 3)
    npt.assert_array_equal(position[:, 0, :], marker_dict["marker1"])


def test_dict_data_get_position_multiple_markers():
    marker_dict = {
        "marker1": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 1.0]]),
        "marker2": np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [1.0, 1.0]]),
        "marker3": np.array([[13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [1.0, 1.0]]),
    }

    marker_data = DictData(marker_dict=marker_dict)
    position = marker_data.get_position(["marker1", "marker3"])

    assert position.shape == (4, 2, 2)
    npt.assert_array_equal(position[:, 0, :], marker_dict["marker1"])
    npt.assert_array_equal(position[:, 1, :], marker_dict["marker3"])


def test_dict_data_get_position_invalid_marker():
    marker_dict = {
        "marker1": np.random.randn(4, 5),
        "marker2": np.random.randn(4, 5),
    }
    for key in marker_dict:
        marker_dict[key][3, :] = 1.0

    marker_data = DictData(marker_dict=marker_dict)

    with pytest.raises(ValueError, match=r"Marker name 'invalid_marker' not found in the marker dictionary."):
        marker_data.get_position(["invalid_marker"])


def test_dict_data_all_marker_positions():
    marker_dict = {
        "marker1": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 1.0]]),
        "marker2": np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [1.0, 1.0]]),
    }

    marker_data = DictData(marker_dict=marker_dict)
    all_positions = marker_data.all_marker_positions

    assert all_positions.shape == (4, 2, 2)
    npt.assert_array_equal(all_positions[:, 0, :], marker_dict["marker1"])
    npt.assert_array_equal(all_positions[:, 1, :], marker_dict["marker2"])


def test_dict_data_markers_center_position():
    marker_dict = {
        "marker1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]]),
        "marker2": np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [1.0, 1.0, 1.0]]),
    }

    marker_data = DictData(marker_dict=marker_dict)
    center = marker_data.markers_center_position(["marker1", "marker2"])

    expected_center = np.nanmean(marker_data.get_position(["marker1", "marker2"]), axis=1)
    assert center.shape == (4, 3)
    npt.assert_array_equal(center, expected_center)


def test_dict_data_mean_marker_position():
    marker_dict = {
        "marker1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]]),
    }

    marker_data = DictData(marker_dict=marker_dict)
    mean_pos = marker_data.mean_marker_position("marker1")

    expected_mean = np.array([[2.0], [5.0], [8.0], [1.0]])
    assert mean_pos.shape == (4, 1)
    npt.assert_array_equal(mean_pos, expected_mean)


def test_dict_data_std_marker_position():
    marker_dict = {
        "marker1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]]),
    }

    marker_data = DictData(marker_dict=marker_dict)
    std_pos = marker_data.std_marker_position("marker1")

    expected_std = np.nanstd(marker_dict["marker1"], axis=1).reshape(4, 1)
    assert std_pos.shape == (4, 1)
    npt.assert_array_almost_equal(std_pos, expected_std)


def test_dict_data_save():
    marker_dict = {
        "marker1": np.random.randn(4, 10),
        "marker2": np.random.randn(4, 10),
        "marker3": np.random.randn(4, 10),
    }
    for key in marker_dict:
        marker_dict[key][3, :] = 1.0

    marker_data = DictData(marker_dict=marker_dict)

    # Save to a temporary file
    tmp_path = "test_dict_data_temp.pkl"
    marker_data.save(tmp_path)

    # Load the saved file
    with open(tmp_path, "rb") as f:
        loaded_dict = pickle.load(f)

    # Verify the loaded data matches
    assert len(loaded_dict) == len(marker_dict)
    for key in marker_dict:
        npt.assert_array_equal(loaded_dict[key], marker_dict[key])

    # Clean up
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_dict_data_get_partial_dict_data():
    marker_dict = {
        "marker1": np.random.randn(4, 10),
        "marker2": np.random.randn(4, 10),
        "marker3": np.random.randn(4, 10),
        "marker4": np.random.randn(4, 10),
    }
    for key in marker_dict:
        marker_dict[key][3, :] = 1.0

    marker_data = DictData(marker_dict=marker_dict)

    # Get partial data with subset of markers
    partial_data = marker_data.get_partial_dict_data(["marker1", "marker3"])

    assert isinstance(partial_data, DictData)
    assert partial_data.nb_markers == 2
    assert partial_data.nb_frames == 10
    assert partial_data.marker_names == ["marker1", "marker3"]

    # Verify the positions match
    for marker_name in ["marker1", "marker3"]:
        original_pos = marker_data.get_position([marker_name])
        partial_pos = partial_data.get_position([marker_name])
        npt.assert_array_equal(original_pos.squeeze(), partial_pos.squeeze())


def test_dict_data_single_frame_auto_expansion():
    # Test that single frame data (1D arrays) are automatically expanded to 2D
    marker_dict = {
        "marker1": np.array([1.0, 2.0, 3.0, 1.0]),  # 1D array
        "marker2": np.array([4.0, 5.0, 6.0, 1.0]),  # 1D array
    }

    marker_data = DictData(marker_dict=marker_dict)

    assert marker_data.nb_frames == 1
    assert marker_data.nb_markers == 2

    # Verify the data was expanded correctly
    position = marker_data.get_position(["marker1"])
    assert position.shape == (4, 1, 1)
    npt.assert_array_equal(position[:, 0, 0], np.array([1.0, 2.0, 3.0, 1.0]))
