from enum import Enum

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.segment_scaling import SegmentScaling
from ..utils.named_list import NamedList


class MassDistributionType(Enum):
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"

class InertialCharacteristics(Enum):
    DE_LEVA = "de_leva"

class ScaleTool:
    def __init__(self,
                 preserve_mass_distribution: bool = True,
                 max_marker_movement: float = 0.1):

        self.original_model = BiomechanicalModelReal()
        self.header = ""
        self.original_mass = None
        self.preserve_mass_distribution = preserve_mass_distribution
        self.max_marker_movement = max_marker_movement
        self.scaling_segments = NamedList[SegmentScaling]()
        self.marker_weightings = {}
        self.warnings = ""

    def scale(self, original_model: BiomechanicalModelReal,
                    static_trial: str,
                    time_range: range,
                    mass: float):
        """
        Scale the model using the configuration defined in the ScaleTool.

        Parameters
        ----------
        original_model
            The original model to scale to the subjects anthropometry
        static_trial
            The .c3d or .trc file of the static trial to use for the scaling
        time_range
            The range of frames to consider for the scaling
        mass
            The mass of the subject
        """

        # Check file format
        if not static_trial.endswith(".c3d"):
            if static_trial.endswith(".trc"):
                raise NotImplementedError(".trc files cannot be read yet.")
            else:
                raise RuntimeError("The static_trial must be a .c3d file in a static posture.")

        # Check the weights
        for marker in self.marker_weightings:
            if self.marker_weightings[marker] < 0:
                raise RuntimeError(f"The weight of marker {marker} is negative. It must be positive.")

        # Check that the marker indeed do not move too much in the static trial
        raise NotImplementedError("TODO: < self.max_marker_movement")

        # Scale the model
        # Change the muscle characteristics


    @staticmethod
    def from_biomod(
            filepath: str,
    ):
        """
        Create a biomechanical model from a biorbd model
        """
        from ..model_parser.biorbd import BiomodConfigurationParser

        return BiomodConfigurationParser(filepath=filepath)

    @staticmethod
    def from_xml(
            filepath: str,
    ):
        """
        Read an xml file from OpenSim and extract the scaling configuration.

        Parameters
        ----------
        filepath: str
            The path to the xml file to read from
        """
        from ..model_parser.opensim import OsimConfigurationParser
        configuration = OsimConfigurationParser(filepath=filepath)
        return configuration.scale_tool
