from .muscle.muscle_group import MuscleGroup
from .muscle.muscle import Muscle
from .muscle.via_point import ViaPoint
from .rigidbody.segment import Segment
from ..real.biomechanical_model_real import BiomechanicalModelReal
from ..real.rigidbody.segment_real import SegmentReal
from ..real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ...utils.aliases import Point, point_to_array
from ...utils.protocols import Data


class BiomechanicalModel:
    def __init__(self):
        self.segments: dict[str, Segment] = {}
        self.muscle_groups: dict[str, MuscleGroup] = {}
        self.muscles: dict[str, Muscle] = {}
        self.via_points: dict[str, ViaPoint] = {}

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

        for name in self.segments:
            s = self.segments[name]

            scs = SegmentCoordinateSystemReal()
            if s.segment_coordinate_system is not None:
                scs = s.segment_coordinate_system.to_scs(
                    data,
                    model,
                    model.segments[s.parent_name].segment_coordinate_system if s.parent_name else None,
                )

            inertia_parameters = None
            if s.inertia_parameters is not None:
                inertia_parameters = s.inertia_parameters.to_real(data, model, scs)

            mesh = None
            if s.mesh is not None:
                mesh = s.mesh.to_mesh(data, model, scs)

            mesh_file = None
            if s.mesh_file is not None:
                mesh_file = s.mesh_file.to_mesh_file(data)

            model.segments[s.name] = SegmentReal(
                name=s.name,
                parent_name=s.parent_name,
                segment_coordinate_system=scs,
                translations=s.translations,
                rotations=s.rotations,
                q_ranges=s.q_ranges,
                qdot_ranges=s.qdot_ranges,
                inertia_parameters=inertia_parameters,
                mesh=mesh,
                mesh_file=mesh_file,
            )

            for marker in s.markers:
                model.segments[name].add_marker(marker.to_marker(data, model, scs))

            for contact in s.contacts:
                model.segments[name].add_contact(contact.to_contact(data))

        for name in self.muscle_groups:
            mg = self.muscle_groups[name]

            model.muscle_groups[mg.name] = MuscleGroup(
                name=mg.name,
                origin_parent_name=mg.origin_parent_name,
                insertion_parent_name=mg.insertion_parent_name,
            )

        for name in self.muscles:
            m = self.muscles[name]

            if m.muscle_group not in model.muscle_groups:
                raise RuntimeError(
                    f"Please create the muscle group {m.muscle_group} before putting the muscle {m.name} in it."
                )

            model.muscles[m.name] = m.to_muscle(model, data)

        for name in self.via_points:
            vp = self.via_points[name]

            if vp.muscle_name not in model.muscles:
                raise RuntimeError(
                    f"Please create the muscle {vp.muscle_name} before putting the via point {vp.name} in it."
                )

            if vp.muscle_group not in model.muscle_groups:
                raise RuntimeError(
                    f"Please create the muscle group {vp.muscle_group} before putting the via point {vp.name} in it."
                )

            model.via_points[vp.name] = vp.to_via_point(data)

        return model
