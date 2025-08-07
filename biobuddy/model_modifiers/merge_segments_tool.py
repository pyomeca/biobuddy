from copy import deepcopy
import logging
import numpy as np

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.segment_real import SegmentReal
from ..components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ..components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ..components.real.rigidbody.mesh_real import MeshReal
from ..components.real.rigidbody.marker_real import MarkerReal
from ..components.real.rigidbody.contact_real import ContactReal
from ..components.real.rigidbody.inertial_measurement_unit_real import InertialMeasurementUnitReal
from ..components.generic.rigidbody.range_of_motion import RangeOfMotion
from ..utils.named_list import NamedList
from ..utils.linear_algebra import RotoTransMatrix, point_from_global_to_local, point_from_local_to_global
from ..utils.enums import Translations, Rotations
from ..utils.aliases import points_to_array


_logger = logging.getLogger(__name__)


class SegmentMerge:
    def __init__(self, name: str, first_segment_name: str, second_segment_name: str, merged_origin_name: str = None):
        """
        Initialize a segment merge configuration.
        This is used to merge segments together in the model.

        Parameters
        ----------
        name
            The name of the new segment that will be created by merging the two segments.
        first_segment_name
            The name of the first segment to merge
        second_segment_name
            The name of the second segment to merge
        merged_origin_name
            The name of the segment that will be the origin of the merged segments.
            If None, the origin is the mean of the origin of the first and second segments.
        """
        if merged_origin_name is not None and merged_origin_name not in [first_segment_name, second_segment_name]:
            raise RuntimeError(
                "The merged origin name must be one of the two segments being merged or None if you want it to be the mean of both origins."
            )
        self.name = name
        self.first_segment_name = first_segment_name
        self.second_segment_name = second_segment_name
        self.merged_origin_name = merged_origin_name


class MergeSegmentsTool:
    def __init__(
        self,
        original_model: BiomechanicalModelReal,
    ):
        """
        Initialize the segment merger tool.

        Parameters
        ----------
        original_model
            The original model to modify by merging segments together.
        """

        # Original attributes
        self.original_model = original_model

        # Extended attributes to be filled
        self.merged_model = BiomechanicalModelReal()
        self.segments_to_merge = NamedList[SegmentMerge]()

    def add(self, merge_segment: SegmentMerge):
        self.segments_to_merge._append(merge_segment)

    def get_merged_parent(
        self, first_segment: SegmentReal, second_segment: SegmentReal, merged_origin_name: str
    ) -> str:

        if merged_origin_name is None:
            if first_segment.parent_name != second_segment.parent_name:
                raise ValueError("You cannot use merged_origin_name=None if the two segments have different parents.")
            else:
                merged_parent = first_segment.parent_name
        elif merged_origin_name == first_segment.name:
            merged_parent = first_segment.parent_name
        elif merged_origin_name == second_segment.name:
            merged_parent = second_segment.parent_name
        else:
            raise RuntimeError(
                "The merged origin name must be one of the two segments being merged or None if you want it to be the mean of both origins."
            )

        return merged_parent

    def get_merged_dofs(
        self, first_segment: SegmentReal, second_segment: SegmentReal, merged_origin_name: str
    ) -> tuple[Translations, Rotations, list[str], RangeOfMotion, RangeOfMotion]:

        first_translations = first_segment.translations
        first_rotations = first_segment.rotations
        first_dof_names = first_segment.dof_names
        first_q_ranges = first_segment.q_ranges
        first_qdot_ranges = first_segment.qdot_ranges
        second_translations = second_segment.translations
        second_rotations = second_segment.rotations
        second_dof_names = second_segment.dof_names
        second_q_ranges = second_segment.q_ranges
        second_qdot_ranges = second_segment.qdot_ranges

        if merged_origin_name is None:
            if first_translations != second_translations or first_rotations != second_rotations:
                raise NotImplementedError(
                    "You cannot use merged_origin_name=None if the two segments have different degrees of freedom."
                )
            else:
                merged_translations = first_translations
                merged_rotations = first_rotations
                merged_dof_names = first_dof_names
                merged_q_ranges = first_q_ranges
                merged_qdot_ranges = first_qdot_ranges
        elif merged_origin_name == first_segment.name:
            merged_translations = first_translations
            merged_rotations = first_rotations
            merged_dof_names = first_dof_names
            merged_q_ranges = first_q_ranges
            merged_qdot_ranges = first_qdot_ranges
        elif merged_origin_name == second_segment.name:
            merged_translations = second_translations
            merged_rotations = second_rotations
            merged_dof_names = second_dof_names
            merged_q_ranges = second_q_ranges
            merged_qdot_ranges = second_qdot_ranges
        else:
            raise RuntimeError(
                "The merged origin name must be one of the two segments being merged or None if you want it to be the mean of both origins."
            )

        return merged_translations, merged_rotations, merged_dof_names, merged_q_ranges, merged_qdot_ranges

    def get_merged_scs(
        self, first_segment: SegmentReal, second_segment: SegmentReal, merged_origin_name: str
    ) -> SegmentCoordinateSystemReal:
        # Get the new origin
        if merged_origin_name is None:
            first_scs = first_segment.segment_coordinate_system.scs
            second_scs = second_segment.segment_coordinate_system.scs
            merged_scs = RotoTransMatrix()
            merged_scs.translation = (first_scs.translation + second_scs.translation) / 2
            first_euler = first_scs.euler_angles("xyz")
            second_euler = second_scs.euler_angles("xyz")
            if any(first_euler != second_euler):
                raise NotImplementedError(
                    "You want to merge segments that are not aligned in orientation. This should be allowed, but is not implemented yet."
                )
            else:
                merged_scs.rotation_matrix = first_scs.rotation_matrix
        else:
            merged_scs = self.original_model.segments[merged_origin_name].segment_coordinate_system.scs

        merged_scs = SegmentCoordinateSystemReal(scs=merged_scs, is_scs_local=True)
        return merged_scs

    @staticmethod
    def transport_inertia(com_distance: np.ndarray, mass: float) -> np.ndarray:
        """
        Computes the moment of inertia due to the transport of the inertia tensor from the original center of mass to the new center of mass.
        """
        inertia = np.identity(4)

        a = com_distance[0]
        b = com_distance[1]
        c = com_distance[2]

        inertia[0, 0] = mass * (b**2 + c**2)
        inertia[0, 1] = mass * (-a * b)
        inertia[0, 2] = mass * (-a * c)
        inertia[1, 0] = mass * (-a * b)
        inertia[1, 1] = mass * (c**2 + a**2)
        inertia[1, 2] = mass * (-b * c)
        inertia[2, 0] = mass * (-a * c)
        inertia[2, 1] = mass * (-b * c)
        inertia[2, 2] = mass * (a**2 + b**2)

        return inertia

    def get_merged_inertia_parameters(
        self, first_segment: SegmentReal, second_segment: SegmentReal, merged_scs_in_global: RotoTransMatrix
    ) -> InertiaParametersReal:

        # Mass (sum)
        merged_mass = first_segment.inertia_parameters.mass + second_segment.inertia_parameters.mass

        # Center of mass (mean position weighted by the mass of the segments)
        first_segment_scs_global = self.original_model.segment_coordinate_system_in_global(first_segment.name)
        second_segment_scs_global = self.original_model.segment_coordinate_system_in_global(second_segment.name)
        first_com_in_global = point_from_local_to_global(
            first_segment.inertia_parameters.center_of_mass, first_segment_scs_global
        )
        second_com_in_global = point_from_local_to_global(
            second_segment.inertia_parameters.center_of_mass, second_segment_scs_global
        )
        merged_com_in_global = (
            first_com_in_global * first_segment.inertia_parameters.mass
            + second_com_in_global * second_segment.inertia_parameters.mass
        ) / merged_mass
        merged_com = point_from_global_to_local(merged_com_in_global, merged_scs_in_global)

        # Inertia (sum of both inertia + transport inertia)
        first_com_distance = merged_com_in_global - first_com_in_global
        second_com_distance = merged_com_in_global - second_com_in_global
        first_inertia = first_segment.inertia_parameters.inertia + self.transport_inertia(
            first_com_distance, first_segment.inertia_parameters.mass
        )
        second_inertia = second_segment.inertia_parameters.inertia + self.transport_inertia(
            second_com_distance, second_segment.inertia_parameters.mass
        )
        merged_inertia = first_inertia + second_inertia

        merged_inertia_parameters = InertiaParametersReal(
            mass=merged_mass, center_of_mass=merged_com, inertia=merged_inertia
        )

        return merged_inertia_parameters

    def get_merged_mesh(
        self, first_segment: SegmentReal, second_segment: SegmentReal, merged_scs_global: RotoTransMatrix
    ) -> MeshReal:
        first_mesh_points = first_segment.mesh.positions
        second_mesh_points = second_segment.mesh.positions
        first_segment_scs_global = self.original_model.segment_coordinate_system_in_global(first_segment.name)
        second_segment_scs_global = self.original_model.segment_coordinate_system_in_global(second_segment.name)
        merged_mesh_points = points_to_array(None, name="merged_mesh_points")

        for i_mesh in range(first_mesh_points.shape[1]):
            point_in_global = point_from_local_to_global(first_mesh_points[:, i_mesh], first_segment_scs_global)
            point_in_new_local = point_from_global_to_local(point_in_global, merged_scs_global)
            merged_mesh_points = np.hstack((merged_mesh_points, point_in_new_local))

        for i_mesh in range(second_mesh_points.shape[1]):
            point_in_global = point_from_local_to_global(second_mesh_points[:, i_mesh], second_segment_scs_global)
            point_in_new_local = point_from_global_to_local(point_in_global, merged_scs_global)
            merged_mesh_points = np.hstack((merged_mesh_points, point_in_new_local))

        merged_mesh = MeshReal(merged_mesh_points)
        return merged_mesh

    def add_merged_markers(
        self,
        first_segment: SegmentReal,
        second_segment: SegmentReal,
        merged_scs_global: RotoTransMatrix,
        merged_segment_name: str,
    ):
        """
        Get the merged markers from the two segments.
        The markers are transformed to the new segment coordinate system.
        """
        merged_markers = NamedList[MarkerReal]()

        for marker in first_segment.markers:
            global_scs = self.original_model.segment_coordinate_system_in_global(first_segment.name)
            marker_in_global = point_from_local_to_global(marker.position, global_scs)
            marker_in_new_local = point_from_global_to_local(marker_in_global, merged_scs_global)
            merged_markers._append(
                MarkerReal(
                    name=deepcopy(marker.name),
                    parent_name=merged_segment_name,
                    position=marker_in_new_local,
                    is_technical=deepcopy(marker.is_technical),
                    is_anatomical=deepcopy(marker.is_anatomical),
                )
            )

        for marker in second_segment.markers:
            global_scs = self.original_model.segment_coordinate_system_in_global(second_segment.name)
            marker_in_global = point_from_local_to_global(marker.position, global_scs)
            marker_in_new_local = point_from_global_to_local(marker_in_global, merged_scs_global)
            merged_markers._append(
                MarkerReal(
                    name=deepcopy(marker.name),
                    parent_name=merged_segment_name,
                    position=marker_in_new_local,
                    is_technical=deepcopy(marker.is_technical),
                    is_anatomical=deepcopy(marker.is_anatomical),
                )
            )

        self.merged_model.segments[merged_segment_name].markers = merged_markers
        return

    def add_merged_contacts(
        self,
        first_segment: SegmentReal,
        second_segment: SegmentReal,
        merged_scs_global: RotoTransMatrix,
        merged_segment_name: str,
    ):
        """
        Get the merged contacts from the two segments.
        The contacts are transformed to the new segment coordinate system.
        """
        merged_contacts = NamedList[ContactReal]()

        for contact in first_segment.contacts:
            global_scs = self.original_model.segment_coordinate_system_in_global(first_segment.name)
            contact_in_global = point_from_local_to_global(contact.position, global_scs)
            contact_in_new_local = point_from_global_to_local(contact_in_global, merged_scs_global)
            merged_contacts._append(
                ContactReal(
                    name=deepcopy(contact.name),
                    parent_name=merged_segment_name,
                    position=contact_in_new_local,
                    axis=deepcopy(contact.axis),
                )
            )

        for contact in second_segment.contacts:
            global_scs = self.original_model.segment_coordinate_system_in_global(second_segment.name)
            contact_in_global = point_from_local_to_global(contact.position, global_scs)
            contact_in_new_local = point_from_global_to_local(contact_in_global, merged_scs_global)
            merged_contacts._append(
                ContactReal(
                    name=deepcopy(contact.name),
                    parent_name=merged_segment_name,
                    position=contact_in_new_local,
                    axis=deepcopy(contact.axis),
                )
            )

        self.merged_model.segments[merged_segment_name].contacts = merged_contacts
        return

    def add_merged_imus(
        self,
        first_segment: SegmentReal,
        second_segment: SegmentReal,
        merged_scs_global: RotoTransMatrix,
        merged_segment_name: str,
    ):
        """
        Get the merged inertial measurement units from the two segments.
        The imus are transformed to the new segment coordinate system.
        """
        merged_imus = NamedList[InertialMeasurementUnitReal]()

        for imu in first_segment.imus:
            raise NotImplementedError(
                "This piece of code bellow was not tested yet, but if you encounter this error and"
                " observe that the code works, please open a PR on GitHub."
            )
            global_scs = self.original_model.segment_coordinate_system_in_global(first_segment.name)
            imu_in_global = global_scs @ imu.scs
            imu_in_new_local = merged_scs_global.inverse @ imu_in_global
            merged_imus._append(
                InertialMeasurementUnitReal(
                    name=deepcopy(imu.name),
                    parent_name=merged_segment_name,
                    scs=imu_in_new_local,
                    is_technical=deepcopy(marker.is_technical),
                    is_anatomical=deepcopy(marker.is_anatomical),
                )
            )

        for imu in second_segment.imus:
            raise NotImplementedError(
                "This piece of code bellow was not tested yet, but if you encounter this error and observe that the code works, please open a PR on GitHub."
            )
            global_scs = self.original_model.segment_coordinate_system_in_global(second_segment.name)
            imu_in_global = global_scs @ imu.scs
            imu_in_new_local = merged_scs_global.inverse @ imu_in_global
            merged_imus._append(
                InertialMeasurementUnitReal(
                    name=deepcopy(imu.name),
                    parent_name=merged_segment_name,
                    scs=imu_in_new_local,
                    is_technical=deepcopy(marker.is_technical),
                    is_anatomical=deepcopy(marker.is_anatomical),
                )
            )

        self.merged_model.segments[merged_segment_name].imus = merged_imus
        return

    def transform_point_to_merged_coordinate_system(
        self,
        point_position: np.ndarray,
        parent_name: str,
        first_segment: SegmentReal,
        second_segment: SegmentReal,
        merged_scs_global: RotoTransMatrix,
    ) -> np.ndarray:
        """
        Transform a point from its original parent coordinate system to the merged coordinate system.

        Parameters
        -----------
        point_position
            The position of the point in its original parent's local coordinates
        parent_name
            The name of the parent segment
        first_segment
            The first segment being merged
        second_segment
            The second segment being merged
        merged_scs_global
            The global coordinate system of the merged segment

        Returns
        ----------
            The point position in the merged coordinate system's local coordinates
        """
        if parent_name == first_segment.name or parent_name == second_segment.name:
            original_scs_global = self.original_model.segment_coordinate_system_in_global(parent_name)
            point_in_global = point_from_local_to_global(point_position, original_scs_global)
            return point_from_global_to_local(point_in_global, merged_scs_global)
        else:
            # Return unchanged if not one of the merge segments
            return point_position

    def update_muscle_attachment_point(
        self,
        muscle_group_name: str,
        muscle_name: str,
        point_type: str,
        new_position: np.ndarray,
        merged_segment_name: str,
    ):
        """
        Update a specific attachment point (origin or insertion) of a muscle.

        Parameters
        ----------
        muscle_group_name
            Name of the muscle group
        muscle_name
            Name of the muscle
        point_type
            Either 'origin' or 'insertion'
        new_position
            The new position for the attachment point
        """
        muscle = self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name]
        if point_type == "origin":
            origin_via_point = deepcopy(muscle.origin_position)
            origin_via_point.position = new_position
            origin_via_point.parent_name = merged_segment_name
            muscle.origin_position = origin_via_point
        elif point_type == "insertion":
            insertion_via_point = deepcopy(muscle.insertion_position)
            insertion_via_point.position = new_position
            insertion_via_point.parent_name = merged_segment_name
            muscle.insertion_position = insertion_via_point

    def add_merged_muscles(
        self,
        first_segment: SegmentReal,
        second_segment: SegmentReal,
        merged_scs_global: RotoTransMatrix,
        merged_segment_name: str,
    ):
        """
        Modify all muscles by transforming their attachment points and via points to the merged coordinate system.

        Parameters
        ----------
        first_segment
            The first segment being merged
        second_segment
            The second segment being merged
        merged_scs_global
            The global coordinate system of the merged segment
        merged_segment_name
            The name of the merged segment to which the muscles will be attached
        """
        for muscle_group in self.original_model.muscle_groups:
            for muscle in muscle_group.muscles:

                # Transform origin point
                if muscle_group.origin_parent_name in [first_segment.name, second_segment.name]:
                    new_origin = self.transform_point_to_merged_coordinate_system(
                        muscle.origin_position.position,
                        muscle_group.origin_parent_name,
                        first_segment,
                        second_segment,
                        merged_scs_global,
                    )
                    self.update_muscle_attachment_point(
                        muscle_group.name, muscle.name, "origin", new_origin, merged_segment_name
                    )

                # Transform insertion point
                if muscle_group.insertion_parent_name in [first_segment.name, second_segment.name]:
                    new_insertion = self.transform_point_to_merged_coordinate_system(
                        muscle.insertion_position.position,
                        muscle_group.insertion_parent_name,
                        first_segment,
                        second_segment,
                        merged_scs_global,
                    )
                    self.update_muscle_attachment_point(
                        muscle_group.name, muscle.name, "insertion", new_insertion, merged_segment_name
                    )

                # Transform via points
                for via_point in muscle.via_points:
                    if via_point.parent_name in [first_segment.name, second_segment.name]:
                        new_via_position = self.transform_point_to_merged_coordinate_system(
                            via_point.position, via_point.parent_name, first_segment, second_segment, merged_scs_global
                        )
                        merged_via_point = deepcopy(
                            self.original_model.muscle_groups[muscle_group.name]
                            .muscles[muscle.name]
                            .via_points[via_point.name]
                        )
                        merged_via_point.position = new_via_position
                        merged_via_point.parent_name = merged_segment_name
                        self.merged_model.muscle_groups[muscle_group.name].muscles[muscle.name].via_points[
                            via_point.name
                        ] = merged_via_point

    def add_merged_children(
        self,
        first_segment: SegmentReal,
        second_segment: SegmentReal,
        merge_task: SegmentMerge,
        merged_scs_local: SegmentCoordinateSystemReal,
    ):

        # Switch the child segments' parent
        first_children = self.merged_model.children_segment_names(merge_task.first_segment_name)
        for child in first_children:
            # Get the new segment coordinate system for the child segment
            global_scs = (
                first_segment.segment_coordinate_system.scs
                @ self.merged_model.segments[child].segment_coordinate_system.scs
            )
            local_scs = merged_scs_local.scs.inverse @ global_scs
            # Modify the child segment
            self.merged_model.segments[child].parent_name = merge_task.name
            self.merged_model.segments[child].scs = local_scs

        second_children = self.merged_model.children_segment_names(merge_task.second_segment_name)
        for child in second_children:
            # Get the new segment coordinate system for the child segment
            global_scs = (
                second_segment.segment_coordinate_system.scs
                @ self.merged_model.segments[child].segment_coordinate_system.scs
            )
            local_scs = merged_scs_local.scs.inverse @ global_scs
            # Modify the child segment
            self.merged_model.segments[child].parent_name = merge_task.name
            self.merged_model.segments[child].scs = local_scs
            return

    def merge(
        self,
    ) -> BiomechanicalModelReal:
        """
        Merge the model's segments using the configuration defined in the MergeSegmentTool.
        """
        # Copy the original model
        self.merged_model = deepcopy(self.original_model)

        # Then, modify it based on the merge tasks
        for merge_task in self.segments_to_merge:

            if (
                merge_task.first_segment_name not in self.merged_model.segment_names
                or merge_task.second_segment_name not in self.merged_model.segment_names
            ):
                raise RuntimeError(
                    f"Segments {merge_task.first_segment_name} and/or {merge_task.second_segment_name} "
                    "not found in the original model."
                )

            # Get the segments to merge
            first_segment = deepcopy(self.merged_model.segments[merge_task.first_segment_name])
            second_segment = deepcopy(self.merged_model.segments[merge_task.second_segment_name])

            # Remove them from the segments
            self.merged_model.remove_segment(merge_task.first_segment_name)
            self.merged_model.remove_segment(merge_task.second_segment_name)

            merged_parent_name = self.get_merged_parent(first_segment, second_segment, merge_task.merged_origin_name)
            (merged_translations, merged_rotations, merged_dof_names, merged_q_range, merged_qdot_range) = (
                self.get_merged_dofs(first_segment, second_segment, merge_task.merged_origin_name)
            )
            merged_scs_local = self.get_merged_scs(first_segment, second_segment, merge_task.merged_origin_name)
            merged_scs_global = (
                self.merged_model.segment_coordinate_system_in_global(merged_parent_name) @ merged_scs_local.scs
            )
            merged_inertia_parameters = self.get_merged_inertia_parameters(
                first_segment, second_segment, merged_scs_global
            )
            merged_mesh = self.get_merged_mesh(first_segment, second_segment, merged_scs_global)

            merged_segment = SegmentReal(
                name=merge_task.name,
                parent_name=merged_parent_name,
                translations=merged_translations,
                rotations=merged_rotations,
                dof_names=merged_dof_names,
                q_ranges=merged_q_range,
                qdot_ranges=merged_qdot_range,
                segment_coordinate_system=merged_scs_local,
                inertia_parameters=merged_inertia_parameters,
                mesh=merged_mesh,
                mesh_file=None,  # It is not possible for now to have multiple meshes in a segment, so we set it to None
            )

            # Add the merged segment to the new model
            self.merged_model.add_segment(merged_segment)

            # Add components
            self.add_merged_markers(
                first_segment, second_segment, merged_scs_global, merged_segment_name=merge_task.name
            )
            self.add_merged_contacts(
                first_segment, second_segment, merged_scs_global, merged_segment_name=merge_task.name
            )
            self.add_merged_imus(first_segment, second_segment, merged_scs_global, merged_segment_name=merge_task.name)
            self.add_merged_muscles(
                first_segment, second_segment, merged_scs_global, merged_segment_name=merge_task.name
            )

            # Modify the children
            self.add_merged_children(first_segment, second_segment, merge_task, merged_scs_local)

        return self.merged_model
