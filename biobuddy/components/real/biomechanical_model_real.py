from copy import deepcopy

# from typing import Self

from .muscle.muscle_real import MuscleType, MuscleStateType
from ...utils.translations import Translations
from ...utils.rotations import Rotations
from ...utils.aliases import Point, point_to_array
from ...utils.named_list import NamedList
from .model_dynamics import ModelDynamics


class BiomechanicalModelReal(ModelDynamics):
    def __init__(self, gravity: Point = None):

        # Imported here to prevent from circular imports
        from ..generic.muscle.muscle_group import MuscleGroup
        from .muscle.muscle_real import MuscleReal
        from .muscle.via_point_real import ViaPointReal
        from .rigidbody.segment_real import SegmentReal

        super().__init__()
        self.is_initialized = True  # So we can now use the ModelDynamics functions

        self.header = ""
        self.gravity = None if gravity is None else point_to_array(gravity, "gravity")
        self.segments = NamedList[SegmentReal]()
        self.muscle_groups = NamedList[MuscleGroup]()
        self.muscles = NamedList[MuscleReal]()
        self.via_points = NamedList[ViaPointReal]()
        self.warnings = ""

    def add_segment(self, segment: "SegmentReal") -> None:
        """
        Add a segment to the model

        Parameters
        ----------
        segment
            The segment to add
        """
        if segment.parent_name != "base" and segment.parent_name not in self.segment_names:
            raise ValueError(
                f"Parent segment should be declared before the child segments. "
                f"Please declare the parent {segment.parent_name} before declaring the child segment {segment.name}."
            )
        self.segments.append(segment)

    def remove_segment(self, segment_name: str) -> None:
        """
        Remove a segment from the model

        Parameters
        ----------
        segment_name
            The name of the segment to remove
        """
        self.segments.remove(segment_name)

    def add_muscle_group(self, muscle_group: "MuscleGroup") -> None:
        """
        Add a muscle group to the model

        Parameters
        ----------
        muscle_group
            The muscle group to add
        """
        if muscle_group.origin_parent_name not in self.segment_names:
            raise ValueError(
                f"The origin segment of a muscle group must be declared before the muscle group."
                f"Please declare the segment {muscle_group.origin_parent_name} before declaring the muscle group {muscle_group.name}."
            )
        if muscle_group.insertion_parent_name not in self.segment_names:
            raise ValueError(
                f"The insertion segment of a muscle group must be declared before the muscle group."
                f"Please declare the segment {muscle_group.insertion_parent_name} before declaring the muscle group {muscle_group.name}."
            )
        self.muscle_groups.append(muscle_group)

    def remove_muscle_group(self, muscle_group_name: str) -> None:
        """
        Remove a muscle group from the model

        Parameters
        ----------
        muscle_group_name
            The name of the muscle group to remove
        """
        self.muscle_groups.remove(muscle_group_name)

    def add_muscle(self, muscle: "MuscleReal") -> None:
        """
        Add a muscle to the model

        Parameters
        ----------
        muscle
            The muscle to add
        """
        if muscle.muscle_group not in self.muscle_group_names:
            raise ValueError(
                f"The muscle group must be declared before the muscle."
                f"Please declare the muscle_group  {muscle.muscle_group} before declaring the muscle {muscle.name}."
            )
        self.muscles.append(muscle)

    def remove_muscle(self, muscle_name: str) -> None:
        """
        Remove a muscle from the model

        Parameters
        ----------
        muscle_name
            The name of the muscle to remove
        """
        self.muscles.remove(muscle_name)

    def add_via_point(self, via_point: "ViaPointReal") -> None:
        """
        Add a via point to the model

        Parameters
        ----------
        via_point
            The via point to add
        """
        if via_point.parent_name not in self.segment_names:
            raise ValueError(
                f"The parent segment of a via point must be declared before the via point."
                f"Please declare the segment {via_point.parent_name} before declaring the via point {via_point.name}."
            )
        elif via_point.muscle_group not in self.muscle_group_names:
            raise ValueError(
                f"The muscle group of a via point must be declared before the via point."
                f"Please declare the muscle group {via_point.muscle_group} before declaring the via point {via_point.name}."
            )
        elif via_point.muscle_name not in self.muscle_names:
            raise ValueError(
                f"The muscle of a via point must be declared before the via point."
                f"Please declare the muscle {via_point.muscle_name} before declaring the via point {via_point.name}."
            )

        self.via_points.append(via_point)

    def remove_via_point(self, via_point_name: str) -> None:
        """
        Remove a via point from the model

        Parameters
        ----------
        via_point_name
            The name of the via point to remove
        """
        self.via_points.remove(via_point_name)

    @property
    def segment_names(self) -> list[str]:
        """
        Get the names of the segments in the model
        """
        return list(self.segments.keys())

    @property
    def marker_names(self) -> list[str]:
        list_marker_names = []
        for segment in self.segments:
            for marker in segment.markers:
                list_marker_names += [marker.name]
        return list_marker_names

    @property
    def contact_names(self) -> list[str]:
        list_contact_names = []
        for segment in self.segments:
            for contact in segment.contacts:
                list_contact_names += [contact.name]
        return list_contact_names

    @property
    def imu_names(self) -> list[str]:
        list_imu_names = []
        for segment in self.segments:
            for imu in segment.imus:
                list_imu_names += [imu.name]
        return list_imu_names

    @property
    def muscle_group_names(self) -> list[str]:
        """
        Get the names of the muscle groups in the model
        """
        return list(self.muscle_groups.keys())

    @property
    def muscle_names(self) -> list[str]:
        """
        Get the names of the muscles in the model
        """
        return list(self.muscles.keys())

    @property
    def via_point_names(self) -> list[str]:
        """
        Get the names of the via points in the model
        """
        return list(self.via_points.keys())

    def has_parent_offset(self, segment_name: str) -> bool:
        """True if the segment segment_name has an offset parent."""
        return segment_name + "_parent_offset" in self.segment_names

    def children_segment_names(self, parent_name: str):
        children = []
        for segment_name in self.segments.keys():
            if self.segments[segment_name].parent_name == parent_name:
                children.append(segment_name)
        return children

    def get_chain_between_segments(self, first_segment_name: str, last_segment_name: str) -> list[str]:
        """
        Get the name of the segments in the kinematic chain between first_segment_name and last_segment_name
        """
        chain = []
        this_segment = last_segment_name
        while this_segment != first_segment_name:
            chain.append(this_segment)
            this_segment = self.segments[this_segment].parent_name
        chain.append(first_segment_name)
        chain.reverse()
        return chain

    @property
    def nb_segments(self) -> int:
        return len(self.segments)

    @property
    def nb_markers(self) -> int:
        return sum(segment.nb_markers for segment in self.segments)

    @property
    def nb_contacts(self) -> int:
        return sum(segment.nb_contacts for segment in self.segments)

    @property
    def nb_imus(self) -> int:
        return sum(segment.nb_imus for segment in self.segments)

    @property
    def nb_muscle_groups(self) -> int:
        return len(self.muscle_groups)

    @property
    def nb_muscles(self) -> int:
        return len(self.muscles)

    @property
    def nb_via_points(self) -> int:
        return len(self.via_points)

    @property
    def nb_q(self) -> int:
        return sum(segment.nb_q for segment in self.segments)

    def segment_index(self, segment_name: str) -> int:
        return list(self.segments.keys()).index(segment_name)

    def dof_indices(self, segment_name: str) -> list[int]:
        """
        Get the indices of the degrees of freedom from the model

        Parameters
        ----------
        segment_name
            The name of the segment to get the indices for
        """
        nb_dof = 0
        for segment in self.segments:
            if segment.name != segment_name:
                if segment.translations != Translations.NONE:
                    nb_dof += len(segment.translations.value)
                if segment.rotations != Rotations.NONE:
                    nb_dof += len(segment.rotations.value)
            else:
                nb_translations = len(segment.translations.value) if segment.translations != Translations.NONE else 0
                nb_rotations = len(segment.rotations.value) if segment.rotations != Rotations.NONE else 0
                return list(range(nb_dof, nb_dof + nb_translations + nb_rotations))
        raise ValueError(f"Segment {segment_name} not found in the model")

    def markers_indices(self, marker_names: list[str]) -> list[int]:
        """
        Get the indices of the markers of the model

        Parameters
        ----------
        marker_names
            The name of the markers to get the indices for
        """
        return [self.marker_names.index(marker) for marker in marker_names]

    def contact_indices(self, contact_names: list[str]) -> list[int]:
        """
        Get the indices of the contacts of the model

        Parameters
        ----------
        contact_names
            The name of the contacts to get the indices for
        """
        return [self.contact_names.index(contact) for contact in contact_names]

    def imu_indices(self, imu_names: list[str]) -> list[int]:
        """
        Get the indices of the imus of the model

        Parameters
        ----------
        imu_names
            The name of the imu to get the indices for
        """
        return [self.imu_names.index(imu) for imu in imu_names]

    @property
    def mass(self) -> float:
        """
        Get the mass of the model
        """
        total_mass = 0.0
        for segment in self.segments:
            if segment.inertia_parameters is not None:
                total_mass += segment.inertia_parameters.mass
        return total_mass

    def segments_rt_to_local(self):
        """
        Make sure all scs are expressed in the local reference frame before moving on to the next step.
        This method should be called everytime a model is returned to the user to avoid any confusion.
        """
        from ..real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal

        for segment in self.segments:
            segment.segment_coordinate_system = SegmentCoordinateSystemReal(
                scs=deepcopy(self.segment_coordinate_system_in_local(segment.name)),
                is_scs_local=True,
            )

    def muscle_origin_on_this_segment(self, segment_name: str) -> list[str]:
        """
        Get the names of the muscles which have an insertion on this segment.
        """
        muscle_names = []
        for muscle in self.muscles:
            if self.muscle_groups[muscle.muscle_group].origin_parent_name == segment_name:  # TODO: This is wack !
                muscle_names += [muscle.name]
        return muscle_names

    def muscle_insertion_on_this_segment(self, segment_name: str) -> list[str]:
        """
        Get the names of the muscles which have an insertion on this segment.
        """
        muscle_names = []
        for muscle in self.muscles:
            if self.muscle_groups[muscle.muscle_group].insertion_parent_name == segment_name:  # TODO: This is wack !
                muscle_names += [muscle.name]
        return muscle_names

    def via_points_on_this_segment(self, segment_name: str) -> list[str]:
        """
        Get the names of the via point which have this segment as a parent.
        """
        return [via_point.name for via_point in self.via_points if via_point.parent_name == segment_name]

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

        model = OsimModelParser(
            filepath=filepath, muscle_type=muscle_type, muscle_state_type=muscle_state_type, mesh_dir=mesh_dir
        ).to_real()
        model.segments_rt_to_local()
        return model

    def to_biomod(self, filepath: str, with_mesh: bool = True) -> None:
        """
        Write the bioMod file.

        Parameters
        ----------
        filepath
            The path to save the bioMod
        with_mesh
            If the mesh should be written to the bioMod file
        """
        from ...model_writer.biorbd.biorbd_model_writer import BiorbdModelWriter

        writer = BiorbdModelWriter(filepath=filepath, with_mesh=with_mesh)
        writer.write(self)

    def to_osim(self, filepath: str, with_mesh: bool = False) -> None:
        """
        Write the .osim file
        """
        from ...model_writer.opensim.opensim_model_writer import OpensimModelWriter

        writer = OpensimModelWriter(filepath=filepath, with_mesh=with_mesh)
        writer.write(self)
