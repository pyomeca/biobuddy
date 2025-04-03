import numpy as np
from typing import TypeAlias

import biorbd

from .marker_real import MarkerReal
from ....utils.translations import Translations


class MeanMarker:
    def __init__(self, marker_names: list[str]):
        self.marker_names = marker_names

    def get_position(self, markers_list: list[MarkerReal]) -> np.ndarray:
        position_mean = np.zeros((3, 1))
        for i_marker in range(len(self.marker_names)):
            position_mean += markers_list[i_marker].position
        position_mean /= len(self.marker_names)
        return position_mean


class AxisWiseScaling:
    def __init__(self,
                 axis: list[Translations],
                 marker_pairs: list[list[[str, str], ...]]):
        """
        A scaling factor is applied to each axis from each segment.
        Each marker pair is used to compute a scaling factor used to scale the segment on the axis specified by axis.

        Parameters
        ----------
        axis
            The axis on which to scale the segment
        marker_pairs
            The pairs of markers used to compute the averaged scaling factor
        """

        # Checks for the marker axis definition
        if not isinstance(marker_pairs, list):
            raise RuntimeError("marker_pairs must be a list of list of marker names.")

        if len(axis) != 3:
            raise RuntimeError("All three axis must be specified for the AxisWise scaling.")
        if len(marker_pairs) != 3:
            raise RuntimeError("A marker pair must be specified for each of the three axis for the AxisWise scaling.")

        for ax in axis:
            if ax not in [Translations.X, Translations.Y, Translations.Z]:
                raise RuntimeError("One axis must be specified at a time.")

        for i in range(3):
            for pair in marker_pairs[i]:
                if len(pair) != 2:
                    raise RuntimeError("Scaling with more than 2 markers is not possible for SegmentWiseScaling.")

        self.axis = axis
        self.marker_pairs = marker_pairs

    def compute_scale_factors(self, marker_positions: np.ndarray, marker_names: list[str], original_model: biorbd.Model) -> dict[str, float]:
        raise NotImplementedError("AxisWiseScaling is not implemented yet.")
        # scale_factor_per_axis["mass"] = mean_scale_factor based on volume difference


class SegmentWiseScaling:
    def __init__(self, axis: Translations, marker_pairs: list[list[str, str]]):
        """
        One scaling factor is applied per segment.
        This method is equivalent to OpenSim's method.
        Each marker pair is used to compute a scaling factor and the average of all scaling factors is used to scale the segment on the axis specified by axis.

        Parameters
        ----------
        axis
            The axis on which to scale the segment
        marker_pairs
            The pairs of markers used to compute the averaged scaling factor
        """

        # Checks for the marker axis definition
        if not isinstance(marker_pairs, list):
            raise RuntimeError("marker_pairs must be a list of marker names.")
        for pair in marker_pairs:
            if len(pair) != 2:
                raise RuntimeError("Scaling with more than 2 markers is not possible for SegmentWiseScaling.")

        self.axis = axis
        self.marker_pairs = marker_pairs

    def compute_scale_factors(self, marker_positions: np.ndarray, marker_names: list[str], original_model: biorbd.Model) -> dict[str, float]:

        original_marker_names = [m.to_string() for m in original_model.markerNames()]
        q_zeros = np.zeros((original_model.nbQ(), ))

        scale_factor = []
        for marker_pair in self.marker_pairs:

            # Distance between the marker pairs in the static file
            marker1_position_subject = marker_positions[:, marker_names.index(marker_pair[0]), :]
            marker2_position_subject = marker_positions[:, marker_names.index(marker_pair[1]), :]
            mean_distance_subject = np.nanmean(np.linalg.norm(marker2_position_subject - marker1_position_subject, axis=0))

            # Distance between the marker pairs in the original model
            marker1_position_original = original_model.markers(q_zeros)[original_marker_names.index(marker_pair[0])].to_array()
            marker2_position_original = original_model.markers(q_zeros)[original_marker_names.index(marker_pair[1])].to_array()
            distance_original = np.linalg.norm(marker2_position_original - marker1_position_original)

            scale_factor += [mean_distance_subject / distance_original]

        mean_scale_factor = np.mean(scale_factor)

        scale_factor_per_axis = {}
        for ax in ["x", "y", "z"]:
            if ax in self.axis.value:
                scale_factor_per_axis[ax] = mean_scale_factor
            else:
                scale_factor_per_axis[ax] = 1.0

        scale_factor_per_axis["mass"] = mean_scale_factor

        return scale_factor_per_axis


class BodyWiseScaling:
    def __init__(self, height: float):
        """
        One scaling factor is applied for the whole body based on the total height.
        It scales all segments on all three axis with one global scaling factor.

        Parameters
        ----------
        height
            The height of the subject
        """
        self.height = height

    def compute_scale_factors(self, marker_positions: np.ndarray, marker_names: list[str], original_model: biorbd.Model) -> dict[str, float]:
        raise NotImplementedError("BodyWiseScaling is not implemented yet.")


ScalingType: TypeAlias = AxisWiseScaling | SegmentWiseScaling | BodyWiseScaling

class SegmentScaling:
    def __init__(
        self,
        name: str,
        scaling_type: ScalingType,
    ):

        # Checks for scaling_type
        if not isinstance(scaling_type, SegmentWiseScaling):
            raise NotImplementedError("Only the SegmentWiseScaling scaling is implemented yet.")

        self.name = name
        self.scaling_type = scaling_type

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def scaling_type(self) -> ScalingType:
        return self._scaling_type

    @scaling_type.setter
    def scaling_type(self, value: ScalingType):
        self._scaling_type = value

    def compute_scaling_factors(self, marker_positions: np.ndarray, marker_names: list[str], original_model_biorbd: biorbd.Model) -> dict[str, float]:
        return self.scaling_type.compute_scale_factors(marker_positions, marker_names, original_model_biorbd)

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        raise NotImplementedError("TODO: implement SegmentScaling.to_biomod()")

    def to_xml(self):
        # Define the print function, so it automatically formats things in the file properly
        raise NotImplementedError("TODO: implement SegmentScaling.to_xml()")