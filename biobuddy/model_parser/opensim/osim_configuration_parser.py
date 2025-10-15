from time import strftime

from lxml import etree

from biobuddy import BiomechanicalModelReal
from ..utils_xml import is_element_empty, find_in_tree, match_tag, match_text, str_to_bool
from ...components.real.rigidbody.segment_scaling import SegmentScaling, SegmentWiseScaling
from ...components.real.rigidbody.marker_weight import MarkerWeight
from ...model_modifiers.scale_tool import ScaleTool
from ...utils.enums import Translations


def _get_file_version(model: etree.ElementTree) -> int:
    return int(model.getroot().attrib["Version"])


class OsimConfigurationParser:
    """
    This xml parser assumes that the scaling configuration was created using OpenSim.
    This means that the
    """

    def __init__(self, filepath: str, original_model: "BiomechanicalModelReal"):
        """
        Reads and converts OpenSim configuration files (.xml) to a generic configuration.

        Parameters
        ----------
        filepath : str
            Path to the OpenSim configuration.xml file to read

        Raises
        ------
        RuntimeError
            If file version is too old or units are not meters/newtons
        """
        # Initial attributes
        self.filepath = filepath

        # Attributes needed to read the xml configuration file
        self.header = (
            "This scaling configuration was created by BioBuddy on "
            + strftime("%Y-%m-%d %H:%M:%S")
            + f"\nIt is based on the original file {filepath}.\n"
        )
        self.configuration = etree.parse(filepath)
        self.model_scaler = None
        self.marker_placer = None
        self.warnings = ""

        for element in self.configuration.getroot()[0]:

            if isinstance(element, etree._Comment):
                # Skip comments
                continue

            elif match_tag(element, "Mass"):
                self.original_mass = float(element.text)  # in kg
                if self.original_mass <= 0:
                    raise NotImplementedError(f"The mass of the original model must be positive.")

            elif match_tag(element, "Height") or match_tag(element, "Age"):
                # These tags are ignored by opensim too.
                continue

            elif match_tag(element, "Notes"):
                self.header += element.text + "\n"

            elif match_tag(element, "GenericModelMaker"):
                if match_text(element, "Unassigned"):
                    self.warnings += "Biobuddy does not handle GenericModelMaker it uses the model specified in the original_model specified as 'scale_tool.scale(original_model=original_model)'.\n"
                    # Note to the devs: In this tag, MakerSet might be useful to modify the original_model marker set specifically for scaling.

            elif match_tag(element, "ModelScaler"):
                self.model_scaler = element

            elif match_tag(element, "MarkerPlacer"):
                self.marker_placer = element

            else:
                raise RuntimeError(
                    f"Element {element.tag} not recognize. Please verify your xml file or send an issue"
                    f" in the github repository."
                )

        # Initialize and fill the scaling configuration
        self.scale_tool = ScaleTool(original_model)  # TODO: this is weird !
        self._read()

    def _read(self):
        """Parse the xml scaling configuration file and populate the output scale tool.

        Processes:
        - Model scaler
        - Marker placer

        Raises
        ------
        RuntimeError
            If critical scaling components are missing or invalid

        Note
        ----
        Modifies the scale_tool object in place by adding the configuration specified in the original xml file.
        """

        # Read model scaler
        if is_element_empty(self.model_scaler):
            raise RuntimeError("The 'ModelScaler' tag must be specified in the xml file.")
        else:
            for element in self.model_scaler:

                if isinstance(element, etree._Comment):
                    # Skipping comments
                    continue

                elif match_tag(element, "apply"):
                    if not match_text(element, "True"):
                        raise RuntimeError(
                            f"This scaling configuration does not do any scaling. Please verify your file {self.filepath}"
                        )

                elif match_tag(element, "preserve_mass_distribution"):
                    self.scale_tool.personalize_mass_distribution = not str_to_bool(element.text)

                elif match_tag(element, "scaling_order"):
                    if not match_text(element, "measurements"):
                        raise RuntimeError("Only 'measurements' based scaling is supported.")

                elif match_tag(element, "MeasurementSet"):
                    for obj in element.find("objects"):
                        if match_tag(obj, "measurement"):

                            name = obj.attrib.get("name", "").split("/")[-1]

                            apply_value = find_in_tree(element, "apply")
                            if apply_value is not None and not match_text(apply_value, "True"):
                                self.warnings += f"The scaling of segment {name} was ignored because the Apply tag is not set to True in the original xml file."

                            marker_pair_set = self._get_marker_pair_set(obj)
                            body_scale_set = self._get_body_scale_set(obj)
                            self.set_scaling_segment(name, marker_pair_set, body_scale_set)
                elif (
                    match_tag(element, "ScaleSet")
                    or match_tag(element, "marker_file")
                    or match_tag(element, "time_range")
                    or match_tag(element, "output_model_file")
                    or match_tag(element, "output_scale_file")
                ):
                    continue
                else:
                    raise RuntimeError(
                        f"Element {element.tag} not recognize. Please verify your xml file or send an issue"
                        f" in the github repository."
                    )

        # Read marker placer
        if is_element_empty(self.marker_placer):
            raise RuntimeError("The 'MarkerPlacer' tag must be specified in the xml file.")
        else:
            for element in self.marker_placer:

                if isinstance(element, etree._Comment):
                    # Skipping comments
                    continue

                elif match_tag(element, "apply"):
                    if match_text(element, "False"):
                        raise NotImplementedError(
                            "The 'MarkerPlacer' tag is set to False. Biobuddy considers that markers should be replaced on the scale model to match the experimental position of the marker on the subject's segments."
                        )

                elif match_tag(element, "max_marker_movement"):
                    max_marker_movement = float(element.text)
                    self.scale_tool.max_marker_movement = float(element.text) if max_marker_movement > 0 else None

                elif match_tag(element, "IKTaskSet"):
                    for obj in element.find("objects"):
                        if match_tag(obj, "IKMarkerTask"):
                            marker_name = obj.attrib.get("name", "").split("/")[-1]
                            apply = str_to_bool(find_in_tree(obj, "apply"))
                            weight = float(find_in_tree(obj, "weight"))
                            self.set_marker_weights(marker_name, apply, weight)

                elif (
                    match_tag(element, "marker_file")
                    or match_tag(element, "coordinate_file")
                    or match_tag(element, "time_range")
                    or match_tag(element, "output_motion_file")
                    or match_tag(element, "output_model_file")
                    or match_tag(element, "output_marker_file")
                ):
                    continue

                else:
                    raise RuntimeError(
                        f"Element {element.tag} not recognize. Please verify your xml file or send an issue"
                        f" in the github repository."
                    )

    @staticmethod
    def _get_marker_pair_set(obj):
        marker_pair_set = obj.find("MarkerPairSet")
        marker_pairs = []
        if marker_pair_set is not None:
            marker_objects = marker_pair_set.find("objects")
            if marker_objects is not None:
                for marker_pair in marker_objects:
                    markers_elem = marker_pair.find("markers")
                    if markers_elem is not None:
                        markers = markers_elem.text.strip().split()
                        marker_pairs.append(markers)
        return marker_pairs

    @staticmethod
    def _get_body_scale_set(obj):
        body_scale_set = obj.find("BodyScaleSet")
        scaling_axis = None
        if body_scale_set is not None:
            scale_objects = body_scale_set.find("objects")
            if scale_objects is not None:
                for scale in scale_objects:
                    if scaling_axis is not None:
                        raise RuntimeError("The scaling axis were already defined.")
                    scaling_elem = scale.find("axes")
                    if scaling_elem is not None:
                        scaling_axis = Translations(scaling_elem.text.replace(" ", "").lower())
        return scaling_axis

    def set_scaling_segment(
        self, segment_name: str, marker_pair_set: list[list[str, str]], body_scale_set: Translations
    ):
        self.scale_tool.add_scaling_segment(
            SegmentScaling(
                name=segment_name,
                scaling_type=SegmentWiseScaling(axis=body_scale_set, marker_pairs=marker_pair_set),
            )
        )

    def set_marker_weights(self, marker_name: str, apply: bool = True, weight: float = 1):
        if apply:
            self.scale_tool.add_marker_weight(MarkerWeight(marker_name, weight))
