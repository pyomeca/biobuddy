from copy import deepcopy
import numpy as np

# from typing import Self

from .muscle.muscle_real import MuscleType, MuscleStateType
from ...utils.aliases import Point, point_to_array, points_to_array
from ...utils.named_list import NamedList
from ..model_utils import ModelUtils
from .model_dynamics import ModelDynamics


class BiomechanicalModelReal(ModelDynamics, ModelUtils):
    def __init__(self, gravity: Point = None):

        # Imported here to prevent from circular imports
        from ..generic.muscle.muscle_group import MuscleGroup
        from .rigidbody.segment_real import SegmentReal

        ModelDynamics.__init__(self)
        ModelUtils.__init__(self)
        self.is_initialized = True  # So we can now use the ModelDynamics functions

        # Model core attributes
        self.header = ""
        self.gravity = None if gravity is None else point_to_array(gravity, "gravity")
        self.segments = NamedList[SegmentReal]()
        self.muscle_groups = NamedList[MuscleGroup]()
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

    def validate_parents(self):
        """
        Validate that all via points have a valid parent segment.
        """
        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:
                if muscle.origin_position.parent_name != muscle_group.origin_parent_name:
                    raise ValueError(
                        f"The origin position of the muscle {muscle.name} must be the same as the origin parent segment {muscle_group.origin_parent_name}."
                    )
                if muscle.insertion_position.parent_name != muscle_group.insertion_parent_name:
                    raise ValueError(
                        f"The insertion position of the muscle {muscle.name} must be the same as the insertion parent segment {muscle_group.insertion_parent_name}."
                    )
                if muscle.origin_position.condition is not None:
                    raise RuntimeError("Muscle origin cannot be conditional.")
                if muscle.insertion_position.condition is not None:
                    raise RuntimeError("Muscle insertion cannot be conditional.")

                for via_point in muscle.via_points:
                    if via_point.parent_name not in self.segment_names:
                        raise ValueError(
                            f"The via point {via_point.name} has a parent segment that does not exist in the model {via_point.parent_name}. "
                        )

    def validate_moving_via_points(self):
        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:
                for via_point in muscle.via_points + [muscle.origin_position, muscle.insertion_position]:
                    if via_point.movement is not None and via_point.position.size != 0:
                        raise RuntimeError(
                            f"A via point can either have a position or a movement, but not both at the same time, {via_point.name} has both."
                        )

    def validate_model(self):
        self.segments_rt_to_local()
        self.validate_parents()
        self.validate_moving_via_points()

    def muscle_origin_on_this_segment(self, segment_name: str) -> list[str]:
        """
        Get the names of the muscles which have an insertion on this segment.
        """
        muscle_names = []
        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:
                if self.muscle_groups[muscle.muscle_group].origin_parent_name == segment_name:
                    muscle_names += [muscle.name]
        return muscle_names

    def muscle_insertion_on_this_segment(self, segment_name: str) -> list[str]:
        """
        Get the names of the muscles which have an insertion on this segment.
        """
        muscle_names = []
        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:
                if self.muscle_groups[muscle.muscle_group].insertion_parent_name == segment_name:
                    muscle_names += [muscle.name]
        return muscle_names

    def via_points_on_this_segment(self, segment_name: str) -> list[str]:
        """
        Get the names of the via point which have this segment as a parent.
        """
        return [via_point.name for via_point in self.via_points if via_point.parent_name == segment_name]

    def fix_via_points(self, q: np.ndarray = None) -> None:
        """
        This function allows to fix conditional and moving via points on the model. This is useful to reduce modeling complexity if the via point do not change much over the range of motion used. It is also useful when using biorbd as these features are not implement in biorbd yet.
        Note: This is a destructive operation: once the conditional and moving via points are fixed, they cannot be reverted.
        """
        if q is None:
            q = np.zeros((self.nb_q, 1))
        elif len(q.shape) == 2:
            if q.shape[1] != 1:
                raise RuntimeError(
                    "fix_via_points is only possible for one configuration (q of shape (nb_q,) or (nb_q, 1)."
                )
        elif len(q.shape) == 1:
            q = q[:, np.newaxis]
        else:
            raise RuntimeError(
                "fix_via_points is only possible for one configuration (q of shape (nb_q,) or (nb_q, 1)."
            )

        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:

                # Moving origin
                if muscle.origin_position.movement is not None:
                    # Get the position of the via point in this configuration
                    muscle.origin_position.position = muscle.origin_position.movement.evaluate(q)
                    muscle.origin_position.movement = None

                # Moving insertion
                if muscle.insertion_position.movement is not None:
                    # Get the position of the via point in this configuration
                    muscle.insertion_position.position = muscle.insertion_position.movement.evaluate(q)
                    muscle.insertion_position.movement = None

                #  Via points
                original_via_points = deepcopy(muscle.via_points)
                for via_point in original_via_points:

                    # Conditional via points
                    if via_point.condition is not None:
                        dof_index = self.dof_index(via_point.condition.dof_name)
                        if via_point.condition.evaluate(q[dof_index]):
                            # The via point is active in this configuration so we keep it
                            muscle.via_points[via_point.name].condition = None
                        else:
                            # The via point is not activa, so we remove it
                            muscle.remove_via_point(via_point.name)

                    # Moving via points
                    elif via_point.movement is not None:
                        # Get the position of the via point in this configuration
                        dof_indices = [self.dof_index(name) for name in via_point.movement.dof_names]
                        muscle.via_points[via_point.name].position = via_point.movement.evaluate(q[dof_indices])
                        muscle.via_points[via_point.name].movement = None

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
        model.validate_model()
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
        self.validate_model()
        writer.write(self)

    def to_osim(self, filepath: str, with_mesh: bool = False) -> None:
        """
        Write the .osim file
        """
        from ...model_writer.opensim.opensim_model_writer import OpensimModelWriter

        writer = OpensimModelWriter(filepath=filepath, with_mesh=with_mesh)
        self.validate_model()
        writer.write(self)
