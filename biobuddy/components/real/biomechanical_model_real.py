import os
from copy import deepcopy

# from typing import Self

from .muscle.muscle_real import MuscleType, MuscleStateType
from ...utils.translations import Translations
from ...utils.rotations import Rotations
from ...utils.aliases import Point, point_to_array
from ...utils.named_list import NamedList
from ...utils.c3d_data import C3dData
from .biomechanical_model_real_utils import segment_coordinate_system_in_local


class BiomechanicalModelReal:
    def __init__(self, gravity: Point = None):
        # Imported here to prevent from circular imports
        from ..generic.muscle.muscle_group import MuscleGroup
        from .muscle.muscle_real import MuscleReal
        from .muscle.via_point_real import ViaPointReal
        from .rigidbody.segment_real import SegmentReal

        self.header = ""
        self.gravity = None if gravity is None else point_to_array("gravity", gravity)
        self.segments = NamedList[SegmentReal]()
        self.muscle_groups = NamedList[MuscleGroup]()
        self.muscles = NamedList[MuscleReal]()
        self.via_points = NamedList[ViaPointReal]()
        self.warnings = ""

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

    @property
    def marker_names(self):
        list_marker_names = []
        for segment in self.segments:
            for marker in segment.markers:
                list_marker_names += [marker.name]
        return list_marker_names

    @property
    def nb_markers(self):
        return sum(segment.nb_markers for segment in self.segments)

    @property
    def nb_q(self):
        return sum(segment.nb_q for segment in self.segments)

    def dof_indices(self, segment_name):
        """
        Get the indices of the degrees of freedom of a segment

        Parameters
        ----------
        segment_name
            The name of the segment to get the indices for
        """
        nb_dof = 0
        for segment in self.segments:
            if segment != segment_name:
                if segment.translations != Translations.NONE:
                    nb_dof += len(segment.translations.value)
                if segment.rotations != Rotations.NONE:
                    nb_dof += len(segment.rotations.value)
            else:
                nb_translations = len(segment.translations.value) if segment.translations != Translations.NONE else 0
                nb_rotations = len(segment.rotations.value) if segment.rotations != Rotations.NONE else 0
                return list(range(nb_dof, nb_dof + nb_translations + nb_rotations))


    @staticmethod
    def from_biomod(
        filepath: str,
    ) -> "Self":
        """
        Create a biomechanical model from a biorbd model
        """
        from ...model_parser.biorbd import BiomodModelParser

        return BiomodModelParser(filepath=filepath).to_real()

    @staticmethod
    def from_osim(
        filepath: str,
        muscle_type: MuscleType = MuscleType.HILL_DE_GROOTE,
        muscle_state_type: MuscleStateType = MuscleStateType.DEGROOTE,
        mesh_dir: str = None,
    ) -> "Self":
        """
        Read an osim file and create both a generic biomechanical model and a personalized model.

        Parameters
        ----------
        filepath: str
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
            filepath=filepath, muscle_type=muscle_type, muscle_state_type=muscle_state_type, mesh_dir=mesh_dir
        ).to_real()

    def to_biomod(self, file_path: str, with_mesh: bool = True):
        """
        Write the bioMod file.

        Parameters
        ----------
        file_path
            The path to save the bioMod
        with_mesh
            If the mesh should be written to the bioMod file
        """

        # Collect the text to write
        out_string = "version 4\n\n"

        out_string += self.header

        out_string += "\n\n\n"
        out_string += "// --------------------------------------------------------------\n"
        out_string += "// SEGMENTS\n"
        out_string += "// --------------------------------------------------------------\n\n"
        for segment in self.segments:
            # Make sure the scs are in the local reference frame before printing
            from ..real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal

            segment.segment_coordinate_system = SegmentCoordinateSystemReal(
                scs=deepcopy(segment_coordinate_system_in_local(self, segment.name)),
                is_scs_local=True,
            )
            out_string += segment.to_biomod(with_mesh=with_mesh)
            out_string += "\n\n\n"  # Give some space between segments

        if self.muscle_groups:
            out_string += "// --------------------------------------------------------------\n"
            out_string += "// MUSCLE GROUPS\n"
            out_string += "// --------------------------------------------------------------\n\n"
            for muscle_group in self.muscle_groups:
                out_string += muscle_group.to_biomod()
                out_string += "\n"
            out_string += "\n\n\n"  # Give some space after muscle groups

        if self.muscles:
            out_string += "// --------------------------------------------------------------\n"
            out_string += "// MUSCLES\n"
            out_string += "// --------------------------------------------------------------\n\n"
            for muscle in self.muscles:
                out_string += muscle.to_biomod()
                out_string += "\n\n\n"  # Give some space between muscles

        if self.via_points:
            out_string += "// --------------------------------------------------------------\n"
            out_string += "// MUSCLES VIA POINTS\n"
            out_string += "// --------------------------------------------------------------\n\n"
            for via_point in self.via_points:
                out_string += via_point.to_biomod()
                out_string += "\n\n\n"  # Give some space between via points

        if self.warnings:
            out_string += "\n/*-------------- WARNINGS---------------\n"
            for warning in self.warnings:
                out_string += "\n" + warning
            out_string += "*/\n"

        # removing any character that is not ascii readable from the out_string before writing the model
        cleaned_string = out_string.encode("ascii", "ignore").decode()

        # Write it to the .bioMod file
        with open(file_path, "w") as file:
            file.write(cleaned_string)

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

        # removing any character that is not ascii readable from the out_string before writing the model
        cleaned_string = out_string.encode("ascii", "ignore").decode()

        # Write it to the .osim file
        with open(file_path, "w") as file:
            file.write(cleaned_string)
