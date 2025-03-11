class BiomechanicalModelReal:
    def __init__(self):
        from biobuddy.components.segment_real import SegmentReal  # Imported here to prevent from circular imports
        from biobuddy.components.muscle_group import MuscleGroup
        from biobuddy.components.muscle_real import MuscleReal
        from biobuddy.components.via_point_real import ViaPointReal

        self.gravity = None
        self.header = ""
        self.warnings = ""
        self.segments: dict[str:SegmentReal, ...] = {}
        # From Pythom 3.7 the insertion order in a dict is preserved. This is important because when writing a new
        # .bioMod file, the order of the segment matters
        self.muscle_groups: dict[str:MuscleGroup, ...] = {}
        self.muscles: dict[str:MuscleReal, ...] = {}
        self.via_points: dict[str:ViaPointReal, ...] = {}


    def remove_segment(self, segment_name: str):
        """
        Remove a segment from the model

        Parameters
        ----------
        segment_name
            The name of the segment to remove
        """
        self.segments = [segment for segment in self.segments if segment.name != segment_name]

    def remove_muscle_group(self, muscle_group_name: str):
        """
        Remove a muscle group from the model

        Parameters
        ----------
        muscle_group_name
            The name of the muscle group to remove
        """
        self.muscle_groups = [muscle_group for muscle_group in self.muscle_groups if
                              muscle_group.name != muscle_group_name]


    def remove_muscle(self, muscle_name: str):
        """
        Remove a muscle from the model

        Parameters
        ----------
        muscle_name
            The name of the muscle to remove
        """
        self.muscles = [muscle for muscle in self.muscles if muscle.name != muscle_name]


    def remove_via_point(self, via_point_name: str):
        """
        Remove a via point from the model

        Parameters
        ----------
        via_point_name
            The name of the via point to remove
        """
        self.via_points = [via_point for via_point in self.via_points if via_point.name != via_point_name]


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

        out_string += self.header

        out_string += "\n\n\n"
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

        if self.warnings:
            out_string += "\n/*-------------- WARNINGS---------------\n"
            for warning in self.warnings:
                out_string += "\n" + warning
            out_string +="*/\n"


        # Actually write it to the .bioMod file
        with open(file_path, "w") as file:
            file.write(out_string)
