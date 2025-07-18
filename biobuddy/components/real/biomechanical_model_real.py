from copy import deepcopy

# from typing import Self

from .muscle.muscle_real import MuscleType, MuscleStateType
from ...utils.aliases import Point, point_to_array
from ...utils.named_list import NamedList
from ..model_utils import ModelUtils
from .model_dynamics import ModelDynamics


class BiomechanicalModelReal(ModelDynamics, ModelUtils):
    def __init__(self, gravity: Point = None):

        # Imported here to prevent from circular imports
        from ..generic.muscle.muscle_group import MuscleGroup
        from .muscle.muscle_real import MuscleReal
        from .muscle.via_point_real import ViaPointReal
        from .rigidbody.segment_real import SegmentReal

        ModelDynamics.__init__(self)
        ModelUtils.__init__(self)
        self.is_initialized = True  # So we can now use the ModelDynamics functions

        # Model core attributes
        self.header = ""
        self.gravity = None if gravity is None else point_to_array(gravity, "gravity")
        self.segments = NamedList[SegmentReal]()
        self.muscle_groups = NamedList[MuscleGroup]()
        self.muscles = NamedList[MuscleReal]()
        self.via_points = NamedList[ViaPointReal]()
        self.warnings = ""

        # Meta-data
        self.filepath = None  # The path to the file from which the model was read, if any
        self.height = None

    def add_segment(self, segment: "SegmentReal") -> None:
        """
        Add a segment to the model

        Parameters
        ----------
        segment
            The segment to add
        """
        # If there is no root segment, declare one before adding other segments
        from ..real.rigidbody.segment_real import SegmentReal

        if len(self.segments) == 0 and segment.name != "root":
            self.segments._append(SegmentReal(name="root"))
            segment.parent_name = "root"

        if segment.parent_name != "base" and segment.parent_name not in self.segment_names:
            raise ValueError(
                f"Parent segment should be declared before the child segments. "
                f"Please declare the parent {segment.parent_name} before declaring the child segment {segment.name}."
            )
        self.segments._append(segment)

    def remove_segment(self, segment_name: str) -> None:
        """
        Remove a segment from the model

        Parameters
        ----------
        segment_name
            The name of the segment to remove
        """
        self.segments._remove(segment_name)

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
        self.muscle_groups._append(muscle_group)

    def remove_muscle_group(self, muscle_group_name: str) -> None:
        """
        Remove a muscle group from the model

        Parameters
        ----------
        muscle_group_name
            The name of the muscle group to remove
        """
        self.muscle_groups._remove(muscle_group_name)

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
        self.muscles._append(muscle)

    def remove_muscle(self, muscle_name: str) -> None:
        """
        Remove a muscle from the model

        Parameters
        ----------
        muscle_name
            The name of the muscle to remove
        """
        self.muscles._remove(muscle_name)

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

        self.via_points._append(via_point)

    def remove_via_point(self, via_point_name: str) -> None:
        """
        Remove a via point from the model

        Parameters
        ----------
        via_point_name
            The name of the via point to remove
        """
        self.via_points._remove(via_point_name)

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

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        if value is not None and not isinstance(value, float):
            raise ValueError("height must be a float.")
        self._height = value

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

    def from_biomod(
        self,
        filepath: str,
    ) -> "Self":
        """
        Create a biomechanical model from a biorbd model
        """
        from ...model_parser.biorbd import BiomodModelParser

        self.filepath = filepath
        return BiomodModelParser(filepath=filepath).to_real()

    def from_osim(
        self,
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

        self.filepath = filepath
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
        self.segments_rt_to_local()
        writer.write(self)

    def to_osim(self, filepath: str, with_mesh: bool = False) -> None:
        """
        Write the .osim file
        """
        from ...model_writer.opensim.opensim_model_writer import OpensimModelWriter

        writer = OpensimModelWriter(filepath=filepath, with_mesh=with_mesh)
        self.segments_rt_to_local()
        writer.write(self)
