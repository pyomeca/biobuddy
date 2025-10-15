from abc import ABC, abstractmethod

from ..components.real.biomechanical_model_real import BiomechanicalModelReal


class AbstractModelParser(ABC):
    def __init__(self, filepath: str):
        """
        The path where the model should be read

        Parameters
        ----------
        filepath
            The path to the model to parse
        """
        self.filepath = filepath

    @abstractmethod
    def to_real(self) -> BiomechanicalModelReal:
        """
        convert the model to BiomechanicalModelReal

        Parameters
        ----------
        model
            The model to print to the file
        """
