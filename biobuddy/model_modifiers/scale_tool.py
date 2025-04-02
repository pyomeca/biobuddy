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
    def __init__(self, preserve_mass_distribution: bool = True):

        self.original_model = BiomechanicalModelReal()
        self.header = ""
        self.original_mass = None
        self.scaling_segments = NamedList[SegmentScaling]()  # TODO
        self.preserve_mass_distribution = preserve_mass_distribution
        self.warnings = ""

    def scale(self, original_model=model,
                                    static_trial=static_file_path,
                                    time_range=time_range,
                                    mass=mass):


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

        return OsimConfigurationParser(filepath=filepath)
