class BiomechanicalModelReal:
    def __init__(self):
        from biobuddy.components.segment_real import SegmentReal  # Imported here to prevent from circular imports
        from biobuddy.components.muscle_group import MuscleGroup
        from biobuddy.components.muscle_real import MuscleReal
        from biobuddy.components.via_point_real import ViaPointReal

        self.gravity = None
        self.segments: dict[str:SegmentReal, ...] = {}
        # From Pythom 3.7 the insertion order in a dict is preserved. This is important because when writing a new
        # .bioMod file, the order of the segment matters
        self.muscle_groups: dict[str:MuscleGroup, ...] = {}
        self.muscles: dict[str:MuscleReal, ...] = {}
        self.via_points: dict[str:ViaPointReal, ...] = {}

    def to_biomod(self, file_path: str):
        """
        Write the bioMod file.

        Parameters
        ----------
        file_path
            The path to save the bioMod
        """

        # Collect the text to write
        out_string = "version 4\n\n"



        out_string += "// --------------------------------------------------------------\n"
        out_string += "// SEGMENTS\n"
        out_string += "// --------------------------------------------------------------\n\n"
        for name in self.segments:
            out_string += self.segments[name].to_biomod
            out_string += "\n\n\n"  # Give some space between segments

        out_string += "// --------------------------------------------------------------\n"
        out_string += "// MUSCLE GROUPS\n"
        out_string += "// --------------------------------------------------------------\n\n"
        for name in self.muscle_groups:
            out_string += self.muscle_groups[name].to_biomod
            out_string += "\n"
        out_string += "\n\n\n"  # Give some space after muscle groups

        out_string += "// --------------------------------------------------------------\n"
        out_string += "// MUSCLES\n"
        out_string += "// --------------------------------------------------------------\n\n"
        for name in self.muscles:
            out_string += self.muscles[name].to_biomod
            out_string += "\n\n\n"  # Give some space between muscles

        out_string += "// --------------------------------------------------------------\n"
        out_string += "// MUSCLES VIA POINTS\n"
        out_string += "// --------------------------------------------------------------\n\n"
        for name in self.via_points:
            out_string += self.via_points[name].to_biomod
            out_string += "\n\n\n"  # Give some space between via points


        # Actually write it to the .bioMod file
        with open(file_path, "w") as file:
            file.write(out_string)
