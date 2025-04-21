

class BiorbdModelWriter:
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
        Writes the BiomechanicalModelReal into a text file of formal .bioMod
        """

        # Collect the text to write
        out_string = "version 4\n\n"

        out_string += model.header

        out_string += "\n\n\n"
        out_string += "// --------------------------------------------------------------\n"
        out_string += "// SEGMENTS\n"
        out_string += "// --------------------------------------------------------------\n\n"
        for segment in model.segments:
            if segment.segment_coordinate_system.is_in_global:
                raise RuntimeError(f"Something went wrong, the segment coordinate system of segment {segment.name} is expressed in the global.")
            out_string += segment.to_biomod(with_mesh=self.with_mesh)
            out_string += "\n\n\n"  # Give some space between segments

        if model.muscle_groups:
            out_string += "// --------------------------------------------------------------\n"
            out_string += "// MUSCLE GROUPS\n"
            out_string += "// --------------------------------------------------------------\n\n"
            for muscle_group in model.muscle_groups:
                out_string += muscle_group.to_biomod()
                out_string += "\n"
            out_string += "\n\n\n"  # Give some space after muscle groups

        if model.muscles:
            out_string += "// --------------------------------------------------------------\n"
            out_string += "// MUSCLES\n"
            out_string += "// --------------------------------------------------------------\n\n"
            for muscle in model.muscles:
                out_string += muscle.to_biomod()
                out_string += "\n\n\n"  # Give some space between muscles

        if model.via_points:
            out_string += "// --------------------------------------------------------------\n"
            out_string += "// MUSCLES VIA POINTS\n"
            out_string += "// --------------------------------------------------------------\n\n"
            for via_point in model.via_points:
                out_string += via_point.to_biomod()
                out_string += "\n\n\n"  # Give some space between via points

        if model.warnings:
            out_string += "\n/*-------------- WARNINGS---------------\n"
            for warning in model.warnings:
                out_string += "\n" + warning
            out_string += "*/\n"

        # removing any character that is not ascii readable from the out_string before writing the model
        cleaned_string = out_string.encode("ascii", "ignore").decode()

        # Write it to the .bioMod file
        with open(self.filepath, "w") as file:
            file.write(cleaned_string)
