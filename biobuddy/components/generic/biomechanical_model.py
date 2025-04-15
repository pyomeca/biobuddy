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

    def to_real(self, data: Data, gravity: Point = None) -> BiomechanicalModelReal:
        """
        Collapse the model to an actual personalized biomechanical model based on the generic model and the data
        file (usually a static trial)

        Parameters
        ----------
        data
            The data to collapse the model from
        """
        gravity = None if gravity is None else point_to_array("gravity", gravity)
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

            model.segments.append(
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
            model.muscle_groups.append(
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

            model.muscles.append(muscle.to_muscle(model, data))

        for via_point in self.via_points:
            if via_point.muscle_name not in model.muscles.keys():
                raise RuntimeError(
                    f"Please create the muscle {via_point.muscle_name} before putting the via point {via_point.name} in it."
                )

            if via_point.muscle_group not in model.muscle_groups.keys():
                raise RuntimeError(
                    f"Please create the muscle group {via_point.muscle_group} before putting the via point {via_point.name} in it."
                )

            model.via_points.append(via_point.to_via_point(data, model))

        return model
