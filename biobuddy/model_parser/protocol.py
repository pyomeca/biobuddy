from typing import Protocol


from ..components.real.biomechanical_model_real import BiomechanicalModelReal


class ModelParser(Protocol):
    def __init__(self, filepath: str):
        """
        Load the model from the filepath

        Parameters
        ----------
        filepath
            The path to the model to load
        """

    def to_real(self) -> BiomechanicalModelReal:
        """
        Convert the model to a BiomechanicalModelReal
        """
