from ..abstract_model_writer import AbstractModelWriter


class BiorbdModelWriter(AbstractModelWriter):

    def write(self, model: "BiomechanicalModelReal") -> None:
        """
        Writes the BiomechanicalModelReal into a text file of formal .bioMod
        """

        # Collect the text to write
        out_string = "version 4\n\n"

        out_string += model.header

        if model.gravity is not None:
            out_string += f"gravity\t{model.gravity[0, 0]}\t{model.gravity[1, 0]}\t{model.gravity[2, 0]}\n\n"

        out_string += "\n\n\n"
        out_string += "// --------------------------------------------------------------\n"
        out_string += "// SEGMENTS\n"
        out_string += "// --------------------------------------------------------------\n\n"
        for segment in model.segments:
            if segment.segment_coordinate_system.is_in_global:
                raise RuntimeError(
                    f"Something went wrong, the segment coordinate system of segment {segment.name} is expressed in the global."
                )
            out_string += segment.to_biomod(with_mesh=self.with_mesh)
            out_string += "\n\n\n"  # Give some space between segments

        if model.muscle_groups:
            out_string += "// --------------------------------------------------------------\n"
            out_string += "// MUSCLES\n"
            out_string += "// --------------------------------------------------------------\n\n"
            for muscle_group in model.muscle_groups:
                out_string += muscle_group.to_biomod()
                out_string += "\n"
            out_string += "\n\n\n"  # Give some space after muscle groups

        if model.warnings:
            out_string += "\n/*-------------- WARNINGS---------------\n"
            for warning in model.warnings:
                out_string += "\n" + warning
            out_string += "\n\n*/\n"

        # removing any character that is not ascii readable from the out_string before writing the model
        cleaned_string = out_string.encode("ascii", "ignore").decode()

        # Write it to the .bioMod file
        with open(self.filepath, "w") as file:
            file.write(cleaned_string)
