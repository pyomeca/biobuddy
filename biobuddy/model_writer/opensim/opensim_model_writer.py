class OpensimModelWriter:
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
        self.filepath = filepath
        self.with_mesh = with_mesh

    def write(self, model: "BiomechanicalModelReal") -> None:
        """
        Writes the BiomechanicalModelReal into a text file of format .osim
        """
        raise NotImplementedError("TODO ;P")

        # removing any character that is not ascii readable from the out_string before writing the model
        cleaned_string = out_string.encode("ascii", "ignore").decode()

        # Write it to the .osim file
        with open(filepath, "w") as file:
            file.write(cleaned_string)
