from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from copy import deepcopy

from ..utils.aliases import Points


class ReferenceFrame(Enum):
    """
    The reference frame for the C3D data.
    """

    Z_UP = "z-up"
    Y_UP = "y-up"


class MarkerData(ABC):
    """
    Abstract class to handle marker data.
    """

    def __init__(self, first_frame: int | None, last_frame: int | None, total_nb_frames: int):

        # Fix the value of the first and last frames for easy accessing
        if first_frame is None:
            self.first_frame = 0
        else:
            self.first_frame = first_frame

        if last_frame is None:
            self.last_frame = total_nb_frames - 1
        else:
            self.last_frame = last_frame

        self.marker_names = self.set_marker_names()
        self.nb_frames = self.set_nb_frames()
        self.nb_markers = self.set_nb_markers()

    @abstractmethod
    def set_marker_names(self) -> list[str]:
        pass

    @abstractmethod
    def set_nb_markers(self) -> int:
        pass

    def set_nb_frames(self) -> int:
        return self.last_frame + 1 - self.first_frame

    def set_values(self) -> dict[str, Points]:
        values = {}
        for marker_name in self.marker_names:
            values[marker_name] = self.get_position((marker_name,)).squeeze()
        return values

    def marker_index(self, from_markers: str) -> int:
        return self.marker_names.index(from_markers)

    def marker_indices(self, from_markers: tuple[str, ...] | list[str]) -> tuple[int, ...]:
        return tuple(self.marker_names.index(n) for n in from_markers)

    @property
    def all_marker_positions(self) -> np.ndarray:
        return self.get_position(marker_names=self.marker_names)

    def markers_center_position(self, marker_names: tuple[str, ...] | list[str]) -> np.ndarray:
        """Get the geometrical center position between markers"""
        marker_position = self.get_position(marker_names)
        if marker_position.size == 0:
            raise RuntimeError(
                f"The marker position is empty (shape: {marker_position.shape}), cannot compute marker center position."
            )
        return np.nanmean(marker_position, axis=1)

    def mean_marker_position(self, marker_name: str) -> np.ndarray:
        """Get the mean position of a marker"""
        marker_position = self.get_position((marker_name,))
        if marker_position.size == 0:
            raise RuntimeError(f"The marker position is empty (shape: {marker_position.shape}), cannot compute mean.")
        return np.nanmean(marker_position, axis=2)

    def std_marker_position(self, marker_name: str) -> np.ndarray:
        """Get the std from the position of a marker"""
        marker_position = self.get_position((marker_name,))
        if marker_position.size == 0:
            raise RuntimeError(f"The marker position is empty (shape: {marker_position.shape}), cannot compute std.")
        return np.nanstd(marker_position, axis=2)

    @abstractmethod
    def get_position(self, marker_names: tuple[str, ...] | list[str]):
        pass

    @abstractmethod
    def save(self, new_path: str):
        pass


class C3dData(MarkerData):
    """
    Handles .c3d files.
    """

    def __init__(self, c3d_path: str, first_frame: int | None = None, last_frame: int | None = None):

        try:
            import ezc3d
        except ImportError:
            raise ImportError(
                "The ezc3d package is required to read C3D files. Please install it via 'conda install -c conda-forge ezc3d'."
            )

        self.c3d_path = c3d_path
        self.ezc3d_data = ezc3d.c3d(c3d_path)
        total_nb_frames = self.ezc3d_data["data"]["points"].shape[1]

        super().__init__(first_frame, last_frame, total_nb_frames)

        self.values = MarkerData.set_values(self)

    def set_marker_names(self) -> list[str]:
        return self.ezc3d_data["parameters"]["POINT"]["LABELS"]["value"]

    def set_nb_markers(self) -> int:
        return self.ezc3d_data["data"]["points"].shape[1]

    @property
    def all_marker_positions(self) -> np.ndarray:
        return self.get_position(marker_names=self.marker_names)

    @all_marker_positions.setter
    def all_marker_positions(self, value: np.ndarray):
        if value.shape != (4, self.nb_markers, self.nb_frames):
            raise ValueError(f"Expected shape (4, {self.nb_markers}, {self.nb_frames}), got {value.shape}.")
        self.ezc3d_data["data"]["points"][:, :, self.first_frame : self.last_frame + 1] = value

    def get_position(self, marker_names: tuple[str, ...] | list[str]):
        return self._to_meter(
            self.ezc3d_data["data"]["points"][
                :, self.marker_indices(marker_names), self.first_frame : self.last_frame + 1
            ]
        )

    def _to_meter(self, data: np.array) -> np.ndarray:
        units = self.ezc3d_data["parameters"]["POINT"]["UNITS"]["value"]
        units = units[0] if len(units) > 0 else units

        if units == "mm":
            factor = 1000
        elif units == "m":
            factor = 1
        else:
            raise RuntimeError(f"The unit {units} is not recognized (current options are mm of m).")

        data /= factor
        data[3] = 1
        return data

    def change_ref_frame(self, ref_from: ReferenceFrame, ref_to: ReferenceFrame) -> None:
        """
        Change the reference frame of the data.
        """
        if ref_from == ref_to:
            return

        if ref_from == ReferenceFrame.Z_UP and ref_to == ReferenceFrame.Y_UP:
            temporary_data = self.ezc3d_data["data"]["points"].copy()
            self.ezc3d_data["data"]["points"][0, self.first_frame : self.last_frame + 1, :] = temporary_data[
                0, self.first_frame : self.last_frame + 1, :
            ]  # X = X
            self.ezc3d_data["data"]["points"][1, self.first_frame : self.last_frame + 1, :] = temporary_data[
                2, self.first_frame : self.last_frame + 1, :
            ]  # Y = Z
            self.ezc3d_data["data"]["points"][2, self.first_frame : self.last_frame + 1, :] = -temporary_data[
                1, self.first_frame : self.last_frame + 1, :
            ]  # Z = -Y

        elif ref_from == ReferenceFrame.Y_UP and ref_to == ReferenceFrame.Z_UP:
            temporary_data = self.ezc3d_data["data"]["points"].copy()
            self.ezc3d_data["data"]["points"][0, self.first_frame : self.last_frame + 1, :] = temporary_data[
                0, self.first_frame : self.last_frame + 1, :
            ]  # X = X
            self.ezc3d_data["data"]["points"][1, self.first_frame : self.last_frame + 1, :] = -temporary_data[
                2, self.first_frame : self.last_frame + 1, :
            ]  # Y = -Z
            self.ezc3d_data["data"]["points"][2, self.first_frame : self.last_frame + 1, :] = temporary_data[
                1, self.first_frame : self.last_frame + 1, :
            ]  # Z = Y

        else:
            raise ValueError(f"Cannot change from {ref_from} to {ref_to}.")

    def save(self, new_path: str):
        """
        Save the changes made to the C3D file.
        """
        if "meta_points" in self.ezc3d_data["data"]:
            # Remove meta points if they exist as it might cause issues with some C3D writer
            del self.ezc3d_data["data"]["meta_points"]
        self.ezc3d_data.write(new_path)


class CsvData(MarkerData):
    """
    Handles .c3d files.
    """

    def __init__(self, csv_path: str, first_frame: int | None = None, last_frame: int | None = None):

        self.csv_path = csv_path
        pd_csv_data = pd.read_csv(csv_path)
        self.column_titles = list(pd_csv_data.columns)
        self.axis_titles = np.array(pd_csv_data)[0, :].tolist()
        self.csv_array = np.array(pd_csv_data)[1:, :].astype(float)
        total_nb_frames = self.csv_array.shape[0]

        super().__init__(first_frame, last_frame, total_nb_frames)

        self.csv_data = self.finalize_marker_data()
        self.values = MarkerData.set_values(self)

    def finalize_marker_data(self):

        csv_data = np.ones((4, self.nb_markers, self.nb_frames))  # This shape mocks a c3d point field
        axes = ["X", "Y", "Z"]
        i_ax = 0
        i_marker = 0
        for i_col in range(len(self.column_titles)):
            if i_ax == 0:
                marker_name = self.column_titles[i_col]
                if marker_name.startswith("Unnamed:"):
                    raise RuntimeError(
                        "The first row of your .csv file should contain the name of each marker. "
                        "Please see the readme to build a proper .csv file."
                    )

            if self.axis_titles[i_col] != axes[i_ax]:
                raise RuntimeError(
                    "The second row of your csv file should contain the coordinate name 'X', 'Y', 'Z' is order."
                    "Here, it should be 'X'."
                    "Please see the readme to build a proper .csv file."
                )
            csv_data[i_ax, i_marker, :] = self.csv_array[self.first_frame : self.last_frame + 1, i_marker * 3 + i_ax]
            i_ax += 1
            if i_ax == 3:
                i_ax = 0
                i_marker += 1

        return csv_data

    def set_marker_names(self) -> list[str]:
        marker_names = []
        for marker in self.column_titles[0::3]:
            marker_names += [marker.strip()]
        return marker_names

    def set_nb_markers(self) -> int:
        if self.csv_array.shape[1] % 3 != 0:
            raise RuntimeError(
                f"The .csv file should contain nb_markers x 3 component rows. "
                f"You have {self.csv_array.shape[1]} rows, which is not divisible by 3."
            )
        return int(self.csv_array.shape[1] / 3)

    @property
    def all_marker_positions(self) -> np.ndarray:
        return self.get_position(marker_names=self.marker_names)

    @all_marker_positions.setter
    def all_marker_positions(self, value: np.ndarray):
        if value.shape != (4, self.nb_markers, self.nb_frames):
            raise ValueError(f"Expected shape (4, {self.nb_markers}, {self.nb_frames}), got {value.shape}.")

        for i_marker in range(self.nb_markers):
            for i_ax in range(3):
                self.csv_array[self.first_frame : self.last_frame + 1, i_marker * 3 + i_ax] = value[i_ax, i_marker, :]
                i_ax += 1

    def get_position(self, marker_names: tuple[str, ...] | list[str]):
        nb_markers = len(marker_names)
        positions = np.ones((4, nb_markers, self.nb_frames))
        for i_marker in range(nb_markers):
            marker_index = self.marker_index(marker_names[i_marker])
            positions[:3, i_marker, :] = self._to_meter(
                self.csv_array[self.first_frame : self.last_frame + 1, marker_index * 3 : (marker_index + 1) * 3].T
            )
        return positions

    def _to_meter(self, data: np.array) -> np.ndarray:
        """
        The data is expected to be expressed in cm in the csv files.
        """
        if data.shape[0] != 3:
            raise RuntimeError(f"The data array should be shape (3, nb_frames), you have {data.shape}")
        factor = 100  # cm
        return data / factor

    def change_ref_frame(self, ref_from: ReferenceFrame, ref_to: ReferenceFrame) -> None:
        """
        Change the reference frame of the data.
        """
        if ref_from == ref_to:
            return

        if ref_from == ReferenceFrame.Z_UP and ref_to == ReferenceFrame.Y_UP:
            temporary_data = deepcopy(self.csv_array[self.first_frame : self.last_frame + 1, :])
            # X = X
            self.csv_array[self.first_frame : self.last_frame + 1, 0::3] = temporary_data[
                self.first_frame : self.last_frame + 1, 0::3
            ]
            # Y = Z
            self.csv_array[self.first_frame : self.last_frame + 1, 1::3] = temporary_data[
                self.first_frame : self.last_frame + 1, 2::3
            ]
            # Z = -Y
            self.csv_array[self.first_frame : self.last_frame + 1, 2::3] = -temporary_data[
                self.first_frame : self.last_frame + 1, 1::3
            ]

        elif ref_from == ReferenceFrame.Y_UP and ref_to == ReferenceFrame.Z_UP:
            temporary_data = deepcopy(self.csv_array[self.first_frame : self.last_frame + 1, :])
            # X = X
            self.csv_array[self.first_frame : self.last_frame + 1, 0::3] = temporary_data[
                self.first_frame : self.last_frame + 1, 0::3
            ]
            # Y = -Z
            self.csv_array[self.first_frame : self.last_frame + 1, 1::3] = -temporary_data[
                self.first_frame : self.last_frame + 1, 2::3
            ]
            # Z = Y
            self.csv_array[self.first_frame : self.last_frame + 1, 2::3] = temporary_data[
                self.first_frame : self.last_frame + 1, 1::3
            ]
        else:
            raise ValueError(f"Cannot change from {ref_from} to {ref_to}.")

    def save(self, new_path: str):
        """
        Save the changes made to the CSV file.
        """

        # Column titles
        column_titles = []
        i_unnamed = 0
        for marker in self.marker_names:
            column_titles += [marker, f"Unnamed: {i_unnamed + 1}", f"Unnamed: {i_unnamed + 2}"]
            i_unnamed += 2

        # Axis titles
        axis_titles = ["X", "Y", "Z"] * self.nb_markers
        axis_row = pd.DataFrame([axis_titles], columns=column_titles)

        # Data rows
        data_df = pd.DataFrame(self.csv_array, columns=column_titles)

        # Combine them
        pd_csv_data = pd.concat([axis_row, data_df], ignore_index=True)

        # Save the output file
        pd_csv_data.to_csv(new_path, index=False)
