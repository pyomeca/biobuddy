from enum import Enum
import numpy as np
import ezc3d


class ReferenceFrame(Enum):
    """
    The reference frame for the C3D data.
    """

    Z_UP = "z-up"
    Y_UP = "y-up"


class C3dData:
    """
    Implementation of the `Data` protocol from model_creation
    """

    def __init__(self, c3d_path, first_frame: int | None = 0, last_frame: int | None = -1):

        if first_frame is None:
            first_frame = 0
        if last_frame is None:
            last_frame = -1

        self.first_frame = first_frame
        self.last_frame = last_frame
        self.c3d_path = c3d_path
        self.ezc3d_data = ezc3d.c3d(c3d_path)
        self.marker_names = self.ezc3d_data["parameters"]["POINT"]["LABELS"]["value"]

        if self.ezc3d_data["data"]["points"].shape[2] == 1 and self.last_frame == -1:
            self.last_frame = 2  # This is a bug otherwise since data[:, :, 0:-1] returns nothing

        self.values = {}
        for marker_name in self.marker_names:
            self.values[marker_name] = self.get_position((marker_name,)).squeeze()

        # Not a property to avoid recomputing each time
        self.nb_frames = self.ezc3d_data["data"]["points"][:, :, self.first_frame : self.last_frame].shape[2]
        self.nb_markers = self.ezc3d_data["data"]["points"].shape[1]

    @property
    def all_marker_positions(self) -> np.ndarray:
        return self.get_position(marker_names=self.marker_names)

    @all_marker_positions.setter
    def all_marker_positions(self, value: np.ndarray):
        if value.shape != (4, self.nb_markers, self.nb_frames):
            raise ValueError(f"Expected shape (4, {self.nb_markers}, {self.nb_frames}), got {value.shape}.")
        self.ezc3d_data["data"]["points"][:, :, self.first_frame : self.last_frame] = value

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

    def _indices_in_c3d(self, from_markers: tuple[str, ...] | list[str]) -> tuple[int, ...]:
        return tuple(self.ezc3d_data["parameters"]["POINT"]["LABELS"]["value"].index(n) for n in from_markers)

    def get_position(self, marker_names: tuple[str, ...] | list[str]):
        return self._to_meter(
            self.ezc3d_data["data"]["points"][:, self._indices_in_c3d(marker_names), self.first_frame : self.last_frame]
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
            self.ezc3d_data["data"]["points"][0, self.first_frame : self.last_frame, :] = temporary_data[
                0, self.first_frame : self.last_frame, :
            ]  # X = X
            self.ezc3d_data["data"]["points"][1, self.first_frame : self.last_frame, :] = temporary_data[
                2, self.first_frame : self.last_frame, :
            ]  # Y = Z
            self.ezc3d_data["data"]["points"][2, self.first_frame : self.last_frame, :] = -temporary_data[
                1, self.first_frame : self.last_frame, :
            ]  # Z = -Y

        elif ref_from == ReferenceFrame.Y_UP and ref_to == ReferenceFrame.Z_UP:
            temporary_data = self.ezc3d_data["data"]["points"].copy()
            self.ezc3d_data["data"]["points"][0, self.first_frame : self.last_frame, :] = temporary_data[
                0, self.first_frame : self.last_frame, :
            ]  # X = X
            self.ezc3d_data["data"]["points"][1, self.first_frame : self.last_frame, :] = -temporary_data[
                2, self.first_frame : self.last_frame, :
            ]  # Y = -Z
            self.ezc3d_data["data"]["points"][2, self.first_frame : self.last_frame, :] = temporary_data[
                1, self.first_frame : self.last_frame, :
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
