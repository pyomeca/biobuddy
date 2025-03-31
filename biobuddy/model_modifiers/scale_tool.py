
from biobuddy import BiomechanicalModelReal


class ScaleTool:
    def __init__(self):
        self.original_model = BiomechanicalModelReal()

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