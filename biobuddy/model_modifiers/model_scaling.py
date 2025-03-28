from typing import Self

from xml.etree import ElementTree

from biobuddy import BiomechanicalModelReal


class ScaleTool:
    def __init__(self):
        self.original_model = BiomechanicalModelReal()
        self.scaling_configuration = ScalingConfiguration()

    @staticmethod
    def from_biomod(
            filepath: str,
    ) -> Self:
        """
        Create a biomechanical model from a biorbd model
        """
        from ...model_parser.biorbd import BiomodModelParser

        return BiomodConfigurationParser(
            filepath=filepath
        ).to_real()

    @staticmethod
    def from_osim(
        filepath: str,
        muscle_type: MuscleType = MuscleType.HILL_DE_GROOTE,
        muscle_state_type: MuscleStateType = MuscleStateType.DEGROOTE,
        mesh_dir: str = None,
    ) -> Self:
        """
        Read an osim file and create both a generic biomechanical model and a personalized model.

        Parameters
        ----------
        filepath: str
            The path to the osim file to read from
        muscle_type: MuscleType
            The type of muscle to assume when interpreting the osim model
        muscle_state_type : MuscleStateType
            The muscle state type to assume when interpreting the osim model
        mesh_dir: str
            The directory where the meshes are located
        """
        from ...model_parser.opensim import OsimModelParser

        return OsimModelParser(
            filepath=filepath, muscle_type=muscle_type, muscle_state_type=muscle_state_type, mesh_dir=mesh_dir
        ).to_real()