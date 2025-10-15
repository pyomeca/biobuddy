from copy import deepcopy
import logging
import numpy as np

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.segment_real import SegmentReal
from ..components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ..components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ..components.real.rigidbody.mesh_real import MeshReal
from ..components.real.rigidbody.mesh_file_real import MeshFileReal
from ..components.real.rigidbody.marker_real import MarkerReal
from ..components.real.rigidbody.contact_real import ContactReal
from ..components.real.rigidbody.inertial_measurement_unit_real import InertialMeasurementUnitReal
from ..components.generic.rigidbody.range_of_motion import RangeOfMotion
from ..utils.named_list import NamedList
from ..utils.linear_algebra import (
    RotoTransMatrix,
    RotationMatrix,
    point_from_global_to_local,
    point_from_local_to_global,
    local_rt_between_global_rts,
)
from ..utils.enums import Translations, Rotations
from ..utils.aliases import points_to_array, Point


_logger = logging.getLogger(__name__)

# TODO: add the possibility to choose the position of the  new root in the segment's coordinate system


class ChangeFirstSegment:
    def __init__(self, first_segment_name: str, new_segment_name: str = "LINK"):
        """
        Initialize a ChangeFirstSegment configuration.
        This is used to change the root segment of a model, and invert all the segments in the kinematic chain between
        the new first segment and the old first segment. Please note that the new first segment will be granted 6 DoFs,
        but this can be modified afterward easily.

        Parameters
        ----------
        first_segment_name
            The name of the new segment that will be the new first segment.
        new_segment_name
            The name of the new segment that will be created at the end of the inversion. Please note that this fake
            segment only serve to join the inverted kinematic chain to the untouched kinematic chain, thus it does not hold inertia.
        """
        self.first_segment_name = first_segment_name
        self.new_segment_name = new_segment_name

    def get_segments_to_invert(
        self, original_model: BiomechanicalModelReal
    ) -> tuple[list[SegmentReal], list[SegmentReal]]:
        """
        Get all the segments between the new first segment (self.first_segment_name) and the old first segment (the child of 'root').
        """
        old_first_segment = original_model.children_segment_names("root")
        if len(old_first_segment) != 1:
            raise NotImplementedError(
                "Only inversion of kinematic chains with one first segment (only one segment is the child of root) is implemented."
            )
        segment_names = original_model.get_chain_between_segments(old_first_segment[0], self.first_segment_name)
        segments_to_invert = []
        remaining_segments = []
        for segment_name in original_model.segment_names:
            if segment_name in segment_names:
                segments_to_invert += [original_model.segments[segment_name]]
            elif segment_name not in ["base", "root"]:
                remaining_segments += [original_model.segments[segment_name]]
        segments_to_invert.reverse()
        return segments_to_invert, remaining_segments

    @staticmethod
    def get_modified_dofs(
        original_model: BiomechanicalModelReal, current_parent: str
    ) -> tuple[Translations, Rotations, list[str], RangeOfMotion, RangeOfMotion]:
        if current_parent == "root":
            modified_translations = Translations.XYZ
            modified_rotations = Rotations.XYZ
            modified_dof_names = None
            modified_q_ranges = None
            modified_qdot_ranges = None
        else:
            modified_translations = original_model.segments[current_parent].translations
            modified_rotations = original_model.segments[current_parent].rotations
            modified_dof_names = original_model.segments[current_parent].dof_names
            modified_q_ranges = original_model.segments[current_parent].q_ranges
            modified_qdot_ranges = original_model.segments[current_parent].qdot_ranges
        return modified_translations, modified_rotations, modified_dof_names, modified_q_ranges, modified_qdot_ranges

    @staticmethod
    def get_modified_inertia(
        original_model: BiomechanicalModelReal, segment_name: str, current_scs_global: RotoTransMatrix
    ) -> InertiaParametersReal:

        # The mass is the same
        mass = deepcopy(original_model.segments[segment_name].inertia_parameters.mass)

        # Center of mass
        com_in_global = original_model.segment_com_in_global(segment_name)
        modified_com = point_from_global_to_local(com_in_global, current_scs_global)

        # Inertia stays the same, as it is expressed around the com, but is rotated if needed
        rt_to_new_scs = original_model.segments[segment_name].segment_coordinate_system.scs.inverse @ current_scs_global
        if np.any(rt_to_new_scs.rotation_matrix != RotationMatrix().rotation_matrix):
            raise NotImplementedError("The rotation of inertia matrix is not implemented yet.")
        modified_inertia = deepcopy(original_model.segments[segment_name].inertia_parameters.inertia)

        modified_inertia_parameters = InertiaParametersReal(
            mass=mass, center_of_mass=modified_com, inertia=modified_inertia
        )
        return modified_inertia_parameters

    @staticmethod
    def get_modified_scs_local(
        modified_model: BiomechanicalModelReal, current_parent: str, current_scs_global: RotoTransMatrix
    ) -> SegmentCoordinateSystemReal:
        """
        Get the modified segment coordinate system in local coordinates.
        """
        parent_scs_global = modified_model.segment_coordinate_system_in_global(current_parent)
        local_scs = local_rt_between_global_rts(parent_scs_global, current_scs_global)
        modified_scs = SegmentCoordinateSystemReal(scs=local_scs, is_scs_local=True)
        return modified_scs

    @staticmethod
    def get_modified_mesh(
        original_model: BiomechanicalModelReal, segment_name: str, current_scs_global: RotoTransMatrix
    ) -> MeshReal | None:

        if original_model.segments[segment_name].mesh is None:
            return None
        else:
            mesh_points = original_model.segments[segment_name].mesh.positions
            segment_scs_global = original_model.segment_coordinate_system_in_global(segment_name)
            modified_mesh_points = points_to_array(None, name="modified_mesh_points")

            for i_mesh in range(mesh_points.shape[1]):
                point_in_global = point_from_local_to_global(mesh_points[:, i_mesh], segment_scs_global)
                point_in_new_local = point_from_global_to_local(point_in_global, current_scs_global)
                modified_mesh_points = np.hstack((modified_mesh_points, point_in_new_local))

            modified_mesh = MeshReal(modified_mesh_points)
            return modified_mesh

    @staticmethod
    def get_modified_mesh_file(
        original_model: BiomechanicalModelReal, segment_name: str, current_scs_global: RotoTransMatrix
    ) -> MeshFileReal | None:

        mesh_file = original_model.segments[segment_name].mesh_file
        if mesh_file is None:
            return None
        else:
            modified_mesh_translation = mesh_file.mesh_translation
            modified_mesh_rotation = mesh_file.mesh_rotation
            modified_mesh_scale = mesh_file.mesh_scale

            # TODO: finalize flipping
            modified_mesh_file = deepcopy(original_model.segments[segment_name].mesh_file)
            modified_mesh_file.mesh_translation = modified_mesh_translation
            modified_mesh_file.mesh_rotation = modified_mesh_rotation
            modified_mesh_file.mesh_scale = modified_mesh_scale
            return modified_mesh_file

    @staticmethod
    def add_modified_markers(
        original_model: BiomechanicalModelReal,
        modified_model: BiomechanicalModelReal,
        segment_name: str,
        current_scs_global: RotoTransMatrix,
    ) -> BiomechanicalModelReal:

        modified_markers = NamedList[MarkerReal]()

        for marker in original_model.segments[segment_name].markers:
            segment_global_scs = original_model.segment_coordinate_system_in_global(segment_name)
            marker_in_global = point_from_local_to_global(marker.position, segment_global_scs)
            marker_in_new_local = point_from_global_to_local(marker_in_global, current_scs_global)
            modified_markers._append(
                MarkerReal(
                    name=deepcopy(marker.name),
                    parent_name=segment_name,
                    position=marker_in_new_local,
                    is_technical=deepcopy(marker.is_technical),
                    is_anatomical=deepcopy(marker.is_anatomical),
                )
            )

        modified_model.segments[segment_name].markers = modified_markers
        return modified_model

    @staticmethod
    def add_modified_contacts(
        original_model: BiomechanicalModelReal,
        modified_model: BiomechanicalModelReal,
        segment_name: str,
        current_scs_global: RotoTransMatrix,
    ) -> BiomechanicalModelReal:

        modified_contacts = NamedList[ContactReal]()

        for contact in original_model.segments[segment_name].contacts:
            segment_global_scs = original_model.segment_coordinate_system_in_global(segment_name)
            contact_in_global = point_from_local_to_global(contact.position, segment_global_scs)
            contact_in_new_local = point_from_global_to_local(contact_in_global, current_scs_global)
            modified_contacts._append(
                ContactReal(
                    name=deepcopy(contact.name),
                    parent_name=segment_name,
                    position=contact_in_new_local,
                    axis=deepcopy(contact.axis),
                )
            )

        modified_model.segments[segment_name].contacts = modified_contacts
        return modified_model

    @staticmethod
    def add_modified_imus(
        original_model: BiomechanicalModelReal,
        modified_model: BiomechanicalModelReal,
        segment_name: str,
        current_scs_global: RotoTransMatrix,
    ) -> BiomechanicalModelReal:

        modified_imus = NamedList[InertialMeasurementUnitReal]()

        for imu in original_model.segments[segment_name].imus:
            raise NotImplementedError(
                "This piece of code bellow was not tested yet, but if you encounter this error and"
                " observe that the code works, please open a PR on GitHub."
            )
            global_scs = original_model.segment_coordinate_system_in_global(segment_name)
            imu_in_global = global_scs @ imu.scs
            imu_in_new_local = current_scs_global.inverse @ imu_in_global
            modified_imus._append(
                InertialMeasurementUnitReal(
                    name=deepcopy(imu.name),
                    parent_name=modified_segment_name,
                    scs=imu_in_new_local,
                    is_technical=deepcopy(marker.is_technical),
                    is_anatomical=deepcopy(marker.is_anatomical),
                )
            )

        modified_model.segments[segment_name].imus = modified_imus
        return modified_model

    @staticmethod
    def transform_point_to_modified_coordinate_system(
        point_position: np.ndarray,
        original_model: BiomechanicalModelReal,
        parent_name: str,
        segment_name: str,
        current_scs_global: RotoTransMatrix,
    ) -> np.ndarray:
        """
        Transform a point from its original parent coordinate system to the new coordinate system.

        Parameters
        -----------
        point_position
            The position of the point in its original parent's local coordinates
        original_model
            The original biomechanical model, before modifications
        parent_name
            The name of the parent segment
        segment_name
            The name of the segment being modified
        current_scs_global
            The curent coordinate system in the global reference frame

        Returns
        ----------
            The point position in the modified coordinate system's local coordinates
        """
        if parent_name == original_model.segments[segment_name].name:
            original_scs_global = original_model.segment_coordinate_system_in_global(parent_name)
            point_in_global = point_from_local_to_global(point_position, original_scs_global)
            return point_from_global_to_local(point_in_global, current_scs_global)
        else:
            # Return unchanged if not one of the merge segments
            return point_position

    @staticmethod
    def update_muscle_attachment_point(
        modified_model: BiomechanicalModelReal,
        muscle_group_name: str,
        muscle_name: str,
        point_type: str,
        new_position: np.ndarray,
        segment_name: str,
    ):
        """
        Update a specific attachment point (origin or insertion) of a muscle.

        Parameters
        ----------
        modified_model
            The biomechanical model being modified
        muscle_group_name
            Name of the muscle group
        muscle_name
            Name of the muscle
        point_type
            Either 'origin' or 'insertion'
        new_position
            The new position for the attachment point
        segment_name
            The name of the segment being modified
        """
        muscle = modified_model.muscle_groups[muscle_group_name].muscles[muscle_name]
        if point_type == "origin":
            origin_via_point = deepcopy(muscle.origin_position)
            origin_via_point.position = new_position
            origin_via_point.parent_name = segment_name
            muscle.origin_position = origin_via_point
        elif point_type == "insertion":
            insertion_via_point = deepcopy(muscle.insertion_position)
            insertion_via_point.position = new_position
            insertion_via_point.parent_name = segment_name
            muscle.insertion_position = insertion_via_point
        return modified_model

    def add_modified_muscles(
        self,
        original_model: BiomechanicalModelReal,
        modified_model: BiomechanicalModelReal,
        segment_name: str,
        current_scs_global: RotoTransMatrix,
    ):
        """
        Modify all muscles by transforming their attachment points and via points to the new coordinate system.
        """
        for muscle_group in original_model.muscle_groups:
            for muscle in muscle_group.muscles:

                # Transform origin point
                if muscle_group.origin_parent_name == segment_name:
                    new_origin = self.transform_point_to_modified_coordinate_system(
                        point_position=muscle.origin_position.position,
                        original_model=original_model,
                        parent_name=muscle_group.origin_parent_name,
                        segment_name=segment_name,
                        current_scs_global=current_scs_global,
                    )
                    modified_model = self.update_muscle_attachment_point(
                        modified_model=modified_model,
                        muscle_group_name=muscle_group.name,
                        muscle_name=muscle.name,
                        point_type="origin",
                        new_position=new_origin,
                        segment_name=segment_name,
                    )

                # Transform insertion point
                if muscle_group.insertion_parent_name == segment_name:
                    new_insertion = self.transform_point_to_modified_coordinate_system(
                        point_position=muscle.insertion_position.position,
                        original_model=original_model,
                        parent_name=muscle_group.insertion_parent_name,
                        segment_name=segment_name,
                        current_scs_global=current_scs_global,
                    )
                    modified_model = self.update_muscle_attachment_point(
                        modified_model=modified_model,
                        muscle_group_name=muscle_group.name,
                        muscle_name=muscle.name,
                        point_type="insertion",
                        new_position=new_insertion,
                        segment_name=segment_name,
                    )

                # Transform via points
                for via_point in muscle.via_points:
                    if via_point.parent_name == segment_name:
                        new_via_position = self.transform_point_to_modified_coordinate_system(
                            point_position=via_point.position,
                            original_model=original_model,
                            parent_name=via_point.parent_name,
                            segment_name=segment_name,
                            current_scs_global=current_scs_global,
                        )
                        modified_via_point = deepcopy(
                            original_model.muscle_groups[muscle_group.name]
                            .muscles[muscle.name]
                            .via_points[via_point.name]
                        )
                        modified_via_point.position = new_via_position
                        modified_via_point.parent_name = segment_name
                        modified_model.muscle_groups[muscle_group.name].muscles[muscle.name].via_points[
                            via_point.name
                        ] = modified_via_point
        return modified_model

    def modify(self, original_model: BiomechanicalModelReal) -> BiomechanicalModelReal:

        # Copy the original model to modify
        modified_model = deepcopy(original_model)

        # Get the segments to invert
        segments_to_invert, remaining_segments = self.get_segments_to_invert(original_model)

        current_parent = "root"
        # By default, the position of the new first segment is the center of mass of the first segment
        # in the global reference frame
        if (
            original_model.segments[self.first_segment_name].inertia_parameters is None
            or original_model.segments[self.first_segment_name].inertia_parameters.center_of_mass is None
        ):
            raise RuntimeError(
                "The first segment must have inertia parameters with a center of mass defined, as the root segment will be defined at this point in the global reference frame."
            )
        current_scs_global = RotoTransMatrix()
        current_scs_global.translation = modified_model.segment_com_in_global(self.first_segment_name)

        # Invert the segments
        for segment in segments_to_invert:

            # Remove them from the model
            modified_model.remove_segment(segment.name)

            (modified_translations, modified_rotations, modified_dof_names, modified_q_ranges, modified_qdot_ranges) = (
                self.get_modified_dofs(original_model, current_parent)
            )

            modified_inertia_parameters = self.get_modified_inertia(original_model, segment.name, current_scs_global)

            modified_scs_local = self.get_modified_scs_local(modified_model, current_parent, current_scs_global)

            modified_mesh = self.get_modified_mesh(original_model, segment.name, current_scs_global)

            modified_mesh_file = self.get_modified_mesh_file(original_model, segment.name, current_scs_global)

            modified_segment = SegmentReal(
                name=segment.name,
                parent_name=current_parent,
                translations=modified_translations,
                rotations=modified_rotations,
                dof_names=modified_dof_names,
                q_ranges=modified_q_ranges,
                qdot_ranges=modified_qdot_ranges,
                segment_coordinate_system=modified_scs_local,
                inertia_parameters=modified_inertia_parameters,
                mesh=modified_mesh,
                mesh_file=modified_mesh_file,
            )

            # Add the merged segment to the new model
            modified_model.add_segment(modified_segment)

            # Add components
            modified_model = self.add_modified_markers(original_model, modified_model, segment.name, current_scs_global)
            modified_model = self.add_modified_contacts(
                original_model, modified_model, segment.name, current_scs_global
            )
            modified_model = self.add_modified_imus(original_model, modified_model, segment.name, current_scs_global)
            modified_model = self.add_modified_muscles(original_model, modified_model, segment.name, current_scs_global)

            # Advance the current scs one segment
            current_scs_global = original_model.segment_coordinate_system_in_global(segment.name)
            current_parent = segment.name

        # We must now add a last segment between the last inverted segment and the old first segment
        modified_scs_local = self.get_modified_scs_local(modified_model, current_parent, current_scs_global)

        modified_segment = SegmentReal(
            name=self.new_segment_name,
            parent_name=current_parent,
            translations=Translations.NONE,
            rotations=Rotations.NONE,
            dof_names=None,
            q_ranges=None,
            qdot_ranges=None,
            segment_coordinate_system=modified_scs_local,
            inertia_parameters=InertiaParametersReal(
                mass=0, center_of_mass=np.array([0, 0, 0]), inertia=np.zeros((3, 3))
            ),
            mesh=None,
            mesh_file=None,
        )

        # Add the merged segment to the new model
        modified_model.add_segment(modified_segment)

        # Once the chain is inverted, the remaining segments should be added at the end of the chain
        for segment in remaining_segments:

            # Remove them from the beginning of the model
            modified_model.remove_segment(segment.name)

            # And readd them at the end
            modified_model.add_segment(segment)

            if segment.parent_name == current_parent:
                # The first remaining segment must be attached to the new intermediary segment
                modified_model.segments[segment.name].parent_name = self.new_segment_name

        return modified_model


class ModifyKinematicChainTool:
    def __init__(
        self,
        original_model: BiomechanicalModelReal,
    ):
        """
        Initialize the kinematic chain modifier tool.

        Parameters
        ----------
        original_model
            The original model to modify by changing the kinematic chain.
        """

        # Original attributes
        self.original_model = original_model

        # Extended attributes to be filled
        self.modified_model = BiomechanicalModelReal()
        self.kinematic_chain_changes: list[ChangeFirstSegment] = []

    def add(self, kinematic_chain_change: ChangeFirstSegment):
        self.kinematic_chain_changes += [kinematic_chain_change]

    def modify(
        self,
    ) -> BiomechanicalModelReal:
        """
        Modify the kinematic chain of the model using the configuration defined in the ModifyKinematicChainTool.
        """
        # Copy the original model
        self.modified_model = deepcopy(self.original_model)

        # Then, modify it based on the modification tasks
        for merge_task in self.kinematic_chain_changes:
            self.modified_model = merge_task.modify(self.modified_model)

        return self.modified_model
