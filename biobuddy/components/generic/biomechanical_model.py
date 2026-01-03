from .muscle.muscle_group import MuscleGroup
from .rigidbody.segment import Segment
from ..model_utils import ModelUtils
from ..real.biomechanical_model_real import BiomechanicalModelReal
from ...utils.aliases import Point, point_to_array
from ...utils.linear_algebra import RotoTransMatrix
from ...utils.named_list import NamedList
from ...utils.marker_data import MarkerData


class BiomechanicalModel(ModelUtils):
    def __init__(self):
        super().__init__()
        self.segments = NamedList[Segment]()
        self.muscle_groups = NamedList[MuscleGroup]()

    def add_segment(self, segment: Segment):
        """
        Add a segment to the model

        Parameters
        ----------
        segment
            The segment to add
        """
        # If there is no root segment, declare one before adding other segments
        if len(self.segments) == 0 and segment.name != "root":
            self.segments._append(Segment(name="root"))
            segment.parent_name = "root"

        if segment.parent_name != "base" and segment.parent_name not in self.segment_names:
            raise ValueError(
                f"Parent segment should be declared before the child segments. "
                f"Please declare the parent {segment.parent_name} before declaring the child segment {segment.name}."
            )
        self.segments._append(segment)

    def remove_segment(self, segment_name: str):
        """
        Remove a segment from the model

        Parameters
        ----------
        segment_name
            The name of the segment to remove
        """
        self.segments._remove(segment_name)

    def add_muscle_group(self, muscle_group: MuscleGroup):
        """
        Add a muscle group to the model

        Parameters
        ----------
        muscle_group
            The muscle group to add
        """
        self.muscle_groups._append(muscle_group)

    def remove_muscle_group(self, muscle_group_name: str):
        """
        Remove a muscle group from the model

        Parameters
        ----------
        muscle_group_name
            The name of the muscle group to remove
        """
        self.muscle_groups._remove(muscle_group_name)

    def to_real(self, data: MarkerData, gravity: Point = None) -> BiomechanicalModelReal:
        """
        Collapse the model to an actual personalized biomechanical model based on the generic model and the data
        file (usually a static trial)

        Parameters
        ----------
        data
            The data to collapse the model from
        """
        from ..real.muscle.muscle_group_real import MuscleGroupReal
        from ..real.rigidbody.segment_real import SegmentReal
        from ..real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal

        gravity = None if gravity is None else point_to_array(gravity, "gravity")
        model = BiomechanicalModelReal(gravity=gravity)

        for segment in self.segments:
            try:
                scs = SegmentCoordinateSystemReal()
                scs_global = RotoTransMatrix()
                if segment.segment_coordinate_system is not None:
                    parent_scs_global = model.segment_coordinate_system_in_global(segment.parent_name)
                    scs = segment.segment_coordinate_system.to_scs(data, model, parent_scs_global)
                    scs_global = parent_scs_global @ scs.scs

                inertia_parameters = None
                if segment.inertia_parameters is not None:
                    inertia_parameters = segment.inertia_parameters.to_inertia(data, model, scs_global)

                mesh = None
                if segment.mesh is not None:
                    mesh = segment.mesh.to_mesh(data, model, scs_global)

                mesh_file = None
                if segment.mesh_file is not None:
                    mesh_file = segment.mesh_file.to_mesh_file(data, model)

                model.add_segment(
                    SegmentReal(
                        name=segment.name,
                        parent_name=segment.parent_name,
                        segment_coordinate_system=scs,
                        translations=segment.translations,
                        rotations=segment.rotations,
                        q_ranges=segment.q_ranges,
                        qdot_ranges=segment.qdot_ranges,
                        inertia_parameters=inertia_parameters,
                        mesh=mesh,
                        mesh_file=mesh_file,
                    )
                )

                for marker in segment.markers:
                    try:
                        model.segments[marker.parent_name].add_marker(marker.to_marker(data, model, scs_global))
                    except Exception as e:  # Marker try
                        raise RuntimeError(
                            f"The segment '{segment.name}' could not be collided: \n"
                            + f"\tThe marker '{marker.name}' could not be collided: \n\t\t{e}"
                        )

                for contact in segment.contacts:
                    try:
                        model.segments[contact.parent_name].add_contact(contact.to_contact(data, model, scs_global))
                    except Exception as e:  # Contact try
                        raise RuntimeError(
                            f"The segment '{segment.name}' could not be collided: \n"
                            + f"\tThe contact '{contact.name}' could not be collided: \n\t\t{e}"
                        )

            except Exception as e:  # Segment try
                raise RuntimeError(f"The following error occurred when collapsing the segment '{segment.name}': {e}")

        for muscle_group in self.muscle_groups:
            try:
                model.add_muscle_group(
                    MuscleGroupReal(
                        name=muscle_group.name,
                        origin_parent_name=muscle_group.origin_parent_name,
                        insertion_parent_name=muscle_group.insertion_parent_name,
                    )
                )

                for muscle in muscle_group.muscles:
                    try:
                        model.muscle_groups[muscle_group.name].add_muscle(muscle.to_muscle(data, model, scs_global))
                    except Exception as e:  # Muscle try
                        raise RuntimeError(
                            f"The muscle group '{muscle_group.name}' could not be collided: \n"
                            + f"\tThe muscle '{muscle.name}' could not be collided: \n\t\t{e}"
                        )

                    for via_point in muscle.via_points:
                        try:
                            model.muscle_groups[muscle_group.name].muscles[muscle.name].add_via_point(
                                via_point.to_via_point(data, model, scs_global)
                            )
                        except Exception as e:  # Via Point try
                            raise RuntimeError(
                                f"The muscle group '{muscle_group.name}' could not be collided: \n"
                                + f"\tThe muscle '{muscle.name}' could not be collided: \n"
                                f"\t\tThe via point '{via_point.name}' could not be collided: \t\t\t{e}"
                            )

            except Exception as e:  # Muscle Group try
                raise RuntimeError(
                    f"The following error occurred when collapsing the muscle group '{muscle_group.name}': {e}"
                )
        model.validate_model()
        return model
