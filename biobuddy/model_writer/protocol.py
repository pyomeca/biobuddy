from typing import Protocol


class ModelWriter(Protocol):
    def __init__(self, filepath: str, with_mesh: bool = False):
        """
        The path where the model should be printed

        Parameters
        ----------
        filepath
            The path to the model to write
        with_mesh
            If the mesh files should be added to the model to write
        """

    def write(self, model: "BiomechanicalModelReal") -> None:
        """
        Writes the BiomechanicalModelReal into a text file of a specific format

        Parameters
        ----------
        model
            The model to print to the file
        """
