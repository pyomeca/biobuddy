import xml.etree.cElementTree as ET
import numpy as np
from typing import TypeAlias

from .marker_real import MarkerReal
from ....utils.enums import Translations


class ScaleFactor:
    def __init__(self, x: float = 1.0, y: float = 1.0, z: float = 1.0, mass: float = 1.0):
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
    def __init__(self, marker_pairs: dict[Translations, list[list[str]]]):
        """
        A scaling factor is applied to each axis from each segment.
        Each marker pair is used to compute a scaling factor used to scale the segment on the axis specified by axis.
        The mass scaling factor is computed as the cubic root of the product of the scaling factors on each axis.

        Parameters
        ----------
        marker_pairs
            The pairs of markers used to compute the averaged scaling factor

        *WARNING*: This method was only tested on mock data/ mock model. If you use it on a real world application,
        please contact the developers on GitHub so that we make sure together that it behaves as you expect.
        """

        # Checks for the marker axis definition
        if not isinstance(marker_pairs, dict):
            raise RuntimeError("marker_pairs must be a dict of {Translations: list of marker names}.")

        for key in marker_pairs:
            if key not in [Translations.X, Translations.Y, Translations.Z]:
                raise RuntimeError("One axis must be specified at a time.")
            if not isinstance(marker_pairs[key], (list, tuple)):
                raise RuntimeError("marker_pairs must be a dict of {Translations: list of marker names}.")
            for pair in marker_pairs[key]:
                if len(pair) != 2:
                    raise RuntimeError("Scaling with more than 2 markers is not possible for AxisWiseScaling.")

        self.marker_pairs = marker_pairs

    def compute_scale_factors(
        self,
        original_model: "BiomechanicalModelReal",
        marker_positions: np.ndarray,
        marker_names: list[str],
    ) -> ScaleFactor:

        original_marker_names = original_model.marker_names
        q_zeros = np.zeros((original_model.nb_q, 1))
        markers = original_model.markers_in_global(q_zeros)
        scale_factor_per_axis = ScaleFactor()

        for axis in self.marker_pairs.keys():
            scale_factor = []
            for marker_pair in self.marker_pairs[axis]:

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
            scale_factor_per_axis[axis.value] = mean_scale_factor

        # Compute the mass scale factor based on the volume difference
        scale_factor_per_axis["mass"] = (
            scale_factor_per_axis.x * scale_factor_per_axis.y * scale_factor_per_axis.z
        ) ** (1 / 3)

        return scale_factor_per_axis

    def to_biomod(self):
        out_string = ""
        out_string += "scalingtype\taxiswisescaling\n"
        for axis in self.marker_pairs.keys():
            out_string += f"\taxis\t{axis.value}\n"
            for marker_pair in self.marker_pairs[axis]:
                out_string += f"\t{axis.value}markerpair\t{marker_pair[0]}\t{marker_pair[1]}\n"
        return out_string


class SegmentWiseScaling:
    def __init__(self, axis: Translations, marker_pairs: list[list[str]]):
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

    def compute_scale_factors(
        self,
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
    def __init__(self, subject_height: float):
        """
        One scaling factor is applied for the whole body based on the total height.
        It scales all segments on all three axis with one global scaling factor.

        Parameters
        ----------
        subject_height
            The height of the subject

        *WARNING*: This method was only tested on mock data/ mock model. If you use it on a real world application,
        please contact the developers on GitHub so that we make sure together that it behaves as you expect.
        """
        self.subject_height = subject_height

    def compute_scale_factors(
        self,
        original_model: "BiomechanicalModelReal",
        marker_positions: np.ndarray,
        marker_names: list[str],
    ) -> ScaleFactor:

        if original_model.height is None:
            raise RuntimeError(
                f"The original model height must be set to use BodyWiseScaling. you can set it using `original_model.height = height`."
            )
        scale_factor = self.subject_height / original_model.height

        scale_factor_per_axis = ScaleFactor()
        for ax in ["x", "y", "z"]:
            scale_factor_per_axis[ax] = scale_factor
        scale_factor_per_axis["mass"] = scale_factor

        return scale_factor_per_axis

    def to_biomod(self):
        raise NotImplementedError("BodyWiseScaling to_biomod is not implemented yet.")


ScalingType: TypeAlias = AxisWiseScaling | SegmentWiseScaling | BodyWiseScaling


class SegmentScaling:
    def __init__(
        self,
        name: str,
        scaling_type: ScalingType,
    ):
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
        return self.scaling_type.compute_scale_factors(original_model, marker_positions, marker_names)

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
