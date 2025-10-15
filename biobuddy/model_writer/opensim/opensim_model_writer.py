from ..abstract_model_writer import AbstractModelWriter

class OpensimModelWriter(AbstractModelWriter):

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
