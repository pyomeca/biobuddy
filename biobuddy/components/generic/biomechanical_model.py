from copy import deepcopy

from .muscle.muscle_group import MuscleGroup
from .muscle.muscle import Muscle
from .muscle.via_point import ViaPoint
from .rigidbody.segment import Segment
from ..real.biomechanical_model_real import BiomechanicalModelReal
from ..real.rigidbody.segment_real import SegmentReal
from ..real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ...utils.aliases import Point, point_to_array
from ...utils.named_list import NamedList
from ...utils.protocols import Data


class BiomechanicalModel:
    def __init__(self):
        self.segments = NamedList[Segment]()
        self.muscle_groups = NamedList[MuscleGroup]()
        self.muscles = NamedList[Muscle]()
        self.via_points = NamedList[ViaPoint]()


    def add_segment(self, segment: "SegmentReal"):
        """
        Add a segment to the model

        Parameters
        ----------
        segment
            The segment to add
        """
        self.segments.append(segment)

    def remove_segment(self, segment_name: str):
        """
        Remove a segment from the model

        Parameters
        ----------
        segment_name
            The name of the segment to remove
        """
        self.segments.remove(segment_name)

    def add_muscle_group(self, muscle_group: "MuscleGroup"):
        """
        Add a muscle group to the model

        Parameters
        ----------
        muscle_group
            The muscle group to add
        """
        self.muscle_groups.append(muscle_group)

    def remove_muscle_group(self, muscle_group_name: str):
        """
        Remove a muscle group from the model

        Parameters
        ----------
        muscle_group_name
            The name of the muscle group to remove
        """
        self.muscle_groups.remove(muscle_group_name)

    def add_muscle(self, muscle: "MuscleReal"):
        """
        Add a muscle to the model

        Parameters
        ----------
        muscle
            The muscle to add
        """
        self.muscles.append(muscle)

    def remove_muscle(self, muscle_name: str):
        """
        Remove a muscle from the model

        Parameters
        ----------
        muscle_name
            The name of the muscle to remove
        """
        self.muscles.remove(muscle_name)

    def add_via_point(self, via_point: "ViaPointReal"):
        """
        Add a via point to the model

        Parameters
        ----------
        via_point
            The via point to add
        """
        self.via_points.append(via_point)

    def remove_via_point(self, via_point_name: str):
        """
        Remove a via point from the model

        Parameters
        ----------
        via_point_name
            The name of the via point to remove
        """
        self.via_points.remove(via_point_name)

    def to_real(self, data: Data, gravity: Point = None) -> BiomechanicalModelReal:
        """
        Collapse the model to an actual personalized biomechanical model based on the generic model and the data
        file (usually a static trial)

        Parameters
        ----------
        data
            The data to collapse the model from
        """
        gravity = None if gravity is None else point_to_array(gravity, "gravity")
        model = BiomechanicalModelReal(gravity=gravity)

        for segment in self.segments:
            scs = SegmentCoordinateSystemReal()
            if segment.segment_coordinate_system is not None:
                parent_scs = None
                if segment.parent_name is not None and segment.parent_name not in ["base"]:
                    parent_scs = deepcopy(model.segments[segment.parent_name].segment_coordinate_system)
                scs = segment.segment_coordinate_system.to_scs(
                    data,
                    model,
                    parent_scs,
                )

            inertia_parameters = None
            if segment.inertia_parameters is not None:
                inertia_parameters = segment.inertia_parameters.to_real(data, model, scs)

            mesh = None
            if segment.mesh is not None:
                mesh = segment.mesh.to_mesh(data, model, scs)

            mesh_file = None
            if segment.mesh_file is not None:
                mesh_file = segment.mesh_file.to_mesh_file(data)

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
                model.segments[marker.parent_name].add_marker(marker.to_marker(data, model, scs))

            for contact in segment.contacts:
                model.segments[contact.parent_name].add_contact(contact.to_contact(data))

        for muscle_group in self.muscle_groups:
            model.add_muscle_group(
                MuscleGroup(
                    name=muscle_group.name,
                    origin_parent_name=muscle_group.origin_parent_name,
                    insertion_parent_name=muscle_group.insertion_parent_name,
                )
            )

        for muscle in self.muscles:
            if muscle.muscle_group not in model.muscle_groups.keys():
                raise RuntimeError(
                    f"Please create the muscle group {muscle.muscle_group} before putting the muscle {muscle.name} in it."
                )

            model.add_muscle(muscle.to_muscle(model, data))

        for via_point in self.via_points:
            if via_point.muscle_name not in model.muscles.keys():
                raise RuntimeError(
                    f"Please create the muscle {via_point.muscle_name} before putting the via point {via_point.name} in it."
                )

            if via_point.muscle_group not in model.muscle_groups.keys():
                raise RuntimeError(
                    f"Please create the muscle group {via_point.muscle_group} before putting the via point {via_point.name} in it."
                )

            model.add_via_point(via_point.to_via_point(data, model))

        return model
