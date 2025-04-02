from time import strftime

from xml.etree import ElementTree

from .utils import _is_element_empty, find_in_tree, match_tag, match_text
from ...components.real.rigidbody.segment_scaling import SegmentScaling, SegmentWiseScaling
from ...model_modifiers.scale_tool import ScaleTool
from ...utils.translations import Translations


def _get_file_version(model: ElementTree) -> int:
    return int(model.getroot().attrib["Version"])


class OsimConfigurationParser:
    """
    This xml parser assumes that the scaling configuration was created using OpenSim.
    This means that the
    """
    def __init__(
        self,
        filepath: str,
    ):
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
        self.header = ("This scaling configuration was created by BioBuddy on " + strftime("%Y-%m-%d %H:%M:%S") +
                       f"\nIt is based on the original file {filepath}.\n")
        self.configuration = ElementTree.parse(filepath)
        self.model_scaler = None
        self.marker_placer = None
        self.warnings = ""

        for element in self.configuration.getroot()[0]:

            if match_tag(element, "Mass"):
                self.original_mass = float(element.text)  # in kg

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

        # Create the biomechanical model
        self.scale_tool = ScaleTool()
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
        if _is_element_empty(self.model_scaler):
            raise RuntimeError("The 'ModelScaler' tag must be specified in the xml file.")
        else:
            for element in self.model_scaler:

                if match_tag(element, "apply"):
                    if not match_text(element, "True"):
                        raise RuntimeError(f"This scaling configuration does not do any scaling. Please verify your file {self.filepath}")

                elif match_tag(element, "preserve_mass_distribution"):  # TODO : implement all four OpenSim cases
                    self.preserve_mass_distribution = bool(element.text)

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

        # Read marker placer
        if _is_element_empty(self.marker_placer):
            raise RuntimeError("The 'MarkerPlacer' tag must be specified in the xml file.")
        else:
            for element in self.marker_placer:
                if element.tag.upper() == "apply".upper():
                    self.scale_tool.preserve_mass_distribution

    @staticmethod
    def _get_marker_pair_set(obj):
        marker_pair_set = obj.find("MarkerPairSet")
        if marker_pair_set is not None:
            marker_pairs = []
            marker_objects = marker_pair_set.find("objects")
            if marker_objects is not None:
                for marker_pair in marker_objects:
                    markers_elem = marker_pair.find("markers")
                    if markers_elem is not None:
                        markers = markers_elem.text.strip().split()
                        marker_pairs.append(markers)
        return marker_pair_set

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

    def set_scaling_segment(self, segment_name: str, marker_pair_set: list[list[str, str]], body_scale_set: Translations):
        self.scale_tool.scaling_segments.append(
            SegmentScaling(segment_name=segment_name,
                            scaling_type=SegmentWiseScaling(axis=body_scale_set, marker_pairs=marker_pair_set)))
