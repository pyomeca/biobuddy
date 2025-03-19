from typing import Self

from .muscle.muscle_real import MuscleType, MuscleStateType
from ...utils.aliases import Point, point_to_array


class BiomechanicalModelReal:
    def __init__(self, gravity: Point = None):
        # Imported here to prevent from circular imports
        from ..generic.muscle.muscle_group import MuscleGroup
        from .muscle.muscle_real import MuscleReal
        from .muscle.via_point_real import ViaPointReal
        from .rigidbody.segment_real import SegmentReal

        self.gravity = None if gravity is None else point_to_array("gravity", gravity)
        self.segments: dict[str, SegmentReal] = {}
        self.muscle_groups: dict[str, MuscleGroup] = {}
        self.muscles: dict[str, MuscleReal] = {}
        self.via_points: dict[str, ViaPointReal] = {}

    def remove_segment(self, segment_name: str):
        """
        Remove a segment from the model

        Parameters
        ----------
        segment_name
            The name of the segment to remove
        """
        self.segments.pop(segment_name)

    def remove_muscle_group(self, muscle_group_name: str):
        """
        Remove a muscle group from the model

        Parameters
        ----------
        muscle_group_name
            The name of the muscle group to remove
        """
        self.muscle_groups.pop(muscle_group_name)

    def remove_muscle(self, muscle_name: str):
        """
        Remove a muscle from the model

        Parameters
        ----------
        muscle_name
            The name of the muscle to remove
        """
        self.muscles.pop(muscle_name)

    def remove_via_point(self, via_point_name: str):
        """
        Remove a via point from the model

        Parameters
        ----------
        via_point_name
            The name of the via point to remove
        """
        self.via_points.pop(via_point_name)

    @staticmethod
    def from_biomod() -> Self:
        """
        Create a biomechanical model from a biorbd model
        """

    @staticmethod
    def from_osim(
        osim_path: str,
        muscle_type: MuscleType = MuscleType.HILL_DE_GROOTE,
        muscle_state_type: MuscleStateType = MuscleStateType.DEGROOTE,
        mesh_dir: str = None,
    ) -> Self:
        """
        Read an osim file and create both a generic biomechanical model and a personalized model.

        Parameters
        ----------
        osim_path: str
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
            osim_path=osim_path, muscle_type=muscle_type, muscle_state_type=muscle_state_type, mesh_dir=mesh_dir
        ).to_real()

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
            out_string += "*/\n"

        # Write it to the .bioMod file
        with open(file_path, "w") as file:
            file.write(out_string)

    def to_osim(self, save_path: str, header: str = "", print_warnings: bool = True):
        """
        Write the .osim file

        Parameters
        ----------
        save_path
            The path to save the osim to
        print_warnings
            If the function should print warnings or not in the osim output file if problems are encountered
        """
        raise NotImplementedError("meh")
