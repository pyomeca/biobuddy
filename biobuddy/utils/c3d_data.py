import numpy as np
import ezc3d


class C3dData:
    """
    Implementation of the `Data` protocol from model_creation
    """

    def __init__(self, c3d_path, first_frame: int = 0, last_frame: int = -1):
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.ezc3d_data = ezc3d.c3d(c3d_path)
        self.marker_names = self.ezc3d_data["parameters"]["POINT"]["LABELS"]["value"]

        if self.ezc3d_data["data"]["points"].shape[2] == 1 and self.last_frame == -1:
            self.last_frame = 2  # This is a bug otherwise since data[:, :, 0:-1] returns nothing

        self.values = {}
        for marker_name in self.marker_names:
            self.values[marker_name] = self.get_position((marker_name,)).squeeze()

    @property
    def all_marker_positions(self) -> np.ndarray:
        return self.get_position(marker_names=self.marker_names)

    def mean_marker_positions(self, marker_names: tuple[str, ...]) -> np.ndarray:
        return np.mean(np.nanmean(self.get_position(marker_names), axis=2), axis=1)

    def _indices_in_c3d(self, from_markers: tuple[str, ...]) -> tuple[int, ...]:
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
