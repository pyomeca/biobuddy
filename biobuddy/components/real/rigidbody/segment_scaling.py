import xml.etree.cElementTree as ET
import numpy as np
from typing import TypeAlias

from .marker_real import MarkerReal
from ....utils.enums import Translations


class MeanMarker:
    def __init__(self, marker_names: list[str]):
        self.marker_names = marker_names

    def get_position(self, markers_list: list[MarkerReal]) -> np.ndarray:
        position_mean = np.zeros((3, 1))
        for i_marker in range(len(self.marker_names)):
            position_mean += markers_list[i_marker].position
        position_mean /= len(self.marker_names)
        return position_mean


class ScaleFactor:
    def __init__(self, x: float = None, y: float = None, z: float = None, mass: float = None):
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass

    def __setitem__(self, key, value):
        if key == "x":
            self.x = value
        elif key == "y":
            self.y = value
        elif key == "z":
            self.z = value
        elif key == "mass":
            self.mass = value
        else:
            raise KeyError(f"Invalid key: {key}")

    def to_vector(self) -> np.ndarray:
        return np.hstack((np.array([self.x, self.y, self.z]), 1.0)).reshape(4, 1)


class AxisWiseScaling:
    def __init__(self, segment_name: str, axis: list[Translations], marker_pairs: list[list[[str, str], ...]]):
        """
        A scaling factor is applied to each axis from each segment.
        Each marker pair is used to compute a scaling factor used to scale the segment on the axis specified by axis.

        Parameters
        ----------
        segment_name
            The name of the segment to scale
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

    def compute_scale_factors(
        self,
        segment_name: str,
        original_model: "BiomechanicalModelReal",
        marker_positions: np.ndarray,
        marker_names: list[str],
    ) -> ScaleFactor:
        raise NotImplementedError("AxisWiseScaling is not implemented yet.")
        # scale_factor_per_axis["mass"] = mean_scale_factor based on volume difference

    def to_biomod(self):
        out_string = ""
        out_string += "scalingtype\taxiswisescaling\n"
        out_string += f"\taxis\t{self.axis.value}\n"
        for i_ax, ax in enumerate(self.axis.value):
            for marker_pair in self.marker_pairs[i_ax]:
                out_string += f"\t{ax}markerpair\t{marker_pair[0]}\t{marker_pair[1]}\n"
        return out_string


class SegmentWiseScaling:
    def __init__(self, segment_name: str, axis: Translations, marker_pairs: list[list[str, str]]):
        """
        One scaling factor is applied per segment.
        This method is equivalent to OpenSim's method.
        Each marker pair is used to compute a scaling factor and the average of all scaling factors is used to scale the segment on the axis specified by axis.

        Parameters
        ----------
        segment_name
            The name of the segment to scale
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

    def compute_scale_factors(
        self,
        segment_name: str,
        original_model: "BiomechanicalModelReal",
        marker_positions: np.ndarray,
        marker_names: list[str],
    ) -> ScaleFactor:

        original_marker_names = original_model.marker_names
        q_zeros = np.zeros((original_model.nb_q, 1))
        markers = original_model.markers_in_global(q_zeros)

        scale_factor = []
        for marker_pair in self.marker_pairs:

            # Distance between the marker pairs in the static file
            marker1_position_subject = marker_positions[:, marker_names.index(marker_pair[0]), :]
            marker2_position_subject = marker_positions[:, marker_names.index(marker_pair[1]), :]
            mean_distance_subject = np.nanmean(
                np.linalg.norm(marker2_position_subject - marker1_position_subject, axis=0)
            )

            # Distance between the marker pairs in the original model
            marker1_position_original = markers[:3, original_marker_names.index(marker_pair[0]), 0]
            marker2_position_original = markers[:3, original_marker_names.index(marker_pair[1]), 0]
            distance_original = np.linalg.norm(marker2_position_original - marker1_position_original)

            scale_factor += [mean_distance_subject / distance_original]

        mean_scale_factor = np.mean(scale_factor)

        scale_factor_per_axis = ScaleFactor()
        for ax in ["x", "y", "z"]:
            if ax in self.axis.value:
                scale_factor_per_axis[ax] = mean_scale_factor
            else:
                scale_factor_per_axis[ax] = 1.0

        scale_factor_per_axis["mass"] = mean_scale_factor

        return scale_factor_per_axis

    def to_biomod(self):
        out_string = ""
        out_string += "\tscalingtype\tsegmentwisescaling\n"
        out_string += f"\taxis\t{self.axis.value}\n"
        for marker_pair in self.marker_pairs:
            out_string += f"\tmarkerpair\t{marker_pair[0]}\t{marker_pair[1]}\n"
        return out_string

    def to_xml(self, marker_objects: ET.Element):
        for marker_pair in self.marker_pairs:
            pair = ET.SubElement(marker_objects, "MarkerPair")
            ET.SubElement(pair, "markers").text = f" {marker_pair[0]} {marker_pair[1]}"


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

    def compute_scale_factors(
        self,
        segment_name: str,
        original_model: "BiomechanicalModelReal",
        marker_positions: np.ndarray,
        marker_names: list[str],
    ) -> ScaleFactor:
        raise NotImplementedError("BodyWiseScaling is not implemented yet.")

    def to_biomod(self):
        raise NotImplementedError("BodyWiseScaling to_biomod is not implemented yet.")


ScalingType: TypeAlias = AxisWiseScaling | SegmentWiseScaling | BodyWiseScaling


class SegmentScaling:
    def __init__(
        self,
        name: str,
        scaling_type: ScalingType,
    ):

        # Checks for scaling_type
        if scaling_type is not None and not isinstance(scaling_type, SegmentWiseScaling):
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

    def compute_scaling_factors(
        self, original_model: "BiomechanicalModelReal", marker_positions: np.ndarray, marker_names: list[str]
    ) -> ScaleFactor:
        return self.scaling_type.compute_scale_factors(self.name, original_model, marker_positions, marker_names)

    def to_biomod(self):
        out_string = ""
        out_string += f"scalingsegment\t{self.name}\n"
        out_string += self.scaling_type.to_biomod()
        out_string += f"endscalingsegment\n\n\n"
        return out_string

    def to_xml(self, objects: ET.Element):

        # Create the Measurement element for "pelvis"
        measurement = ET.SubElement(objects, "Measurement", name=self.name)
        ET.SubElement(measurement, "apply").text = "true"

        # Create the MarkerPairSet element and its MarkerPair elements
        marker_pair_set = ET.SubElement(measurement, "MarkerPairSet")
        marker_objects = ET.SubElement(marker_pair_set, "objects")

        self.scaling_type.to_xml(marker_objects)

        # Create the BodyScaleSet element and its BodyScale element
        body_scale_set = ET.SubElement(measurement, "BodyScaleSet")
        body_scale_objects = ET.SubElement(body_scale_set, "objects")
        body_scale = ET.SubElement(body_scale_objects, "BodyScale", name=self.name)
        ET.SubElement(body_scale, "axes").text = " ".join(
            f"{self.scaling_type.axis.value.upper()[i]}" for i in range(3)
        )
