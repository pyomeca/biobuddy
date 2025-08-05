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
from ..utils.linear_algebra import RotoTransMatrix, RotationMatrix, point_from_global_to_local, point_from_local_to_global
from ..utils.enums import Translations, Rotations
from ..utils.aliases import points_to_array, Point


_logger = logging.getLogger(__name__)


class ChangeFirstSegment:
    def __init__(self,
                 first_segment_name: str,
                 origin_position: Point,
                 first_scs: RotoTransMatrix = None):
        """
        Initialize a ChangeFirstSegment configuration.
        This is used to change the root segment of a model, and invert all the segments in the kinematic chain between
        the new first segment and the old first segment. Please note that the new first segment will be granted 6 DoFs,
        but this can be modified afterward easily.

        Parameters
        ----------
        first_segment_name
            The name of the new segment that will be the new first segment.
        origin_position
            The position of the new first segment's origin in the old first segment segment coordinate system
            reference frame
        first_scs
            The segment coordinate system of the new first segment in the global reference frame. If None, the old
            segment coordinate system will be used
        """
        self.first_segment_name = first_segment_name
        self.origin_position = origin_position
        self.first_scs = first_scs

    def get_segments_to_invert(self, modified_model: BiomechanicalModelReal) -> list[SegmentReal]:
        """
        Get all the segments between the new first segment (self.first_segment_name) and the old first segment (the child of 'root').
        """
        old_first_segment = modified_model.children_segment_names("root")
        if len(old_first_segment) != 1:
            raise NotImplementedError("Only inversion of kinematic chains with one first segment (only one segment is the child of root) is implemented.")
        segment_names = modified_model.get_chain_between_segments(old_first_segment[0], self.first_segment_name)
        segments_to_invert = []
        for segment_name in segment_names:
            segments_to_invert += [modified_model.segments[segment_name]]
        segments_to_invert.reverse()
        return segments_to_invert

    def get_modified_dofs(self, original_model: BiomechanicalModelReal, current_parent: str) -> tuple[Translations, Rotations, list[str], RangeOfMotion, RangeOfMotion]:
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

    def get_modified_inertia(self, original_model: BiomechanicalModelReal, segment_name: str, current_scs_global: RotoTransMatrix) -> InertiaParametersReal:

        # The mass is the same
        mass = deepcopy(original_model.segments[segment_name].inertia_parameters.mass)

        # Center of mass
        com_in_global = original_model.segment_com_in_global(segment_name)
        modified_com = point_from_global_to_local(com_in_global, current_scs_global)

        # Inertia stays the same, as it is expressed around the com, but is rotated if needed
        rt_to_new_scs = original_model.segments[segment_name].segment_coordinate_system.scs.rt_matrix.inverse @ current_scs_global
        if rt_to_new_scs.rotation_matrix != RotationMatrix():
            raise NotImplementedError("The rotation of inertia matrix is not implemented yet.")
        modified_inertia = deepcopy(original_model.segments[segment_name].inertia_parameters.inertia)

        modified_inertia_parameters = InertiaParametersReal(
            mass=mass, center_of_mass=modified_com, inertia=modified_inertia
        )
        return modified_inertia_parameters

    def get_modified_scs_local(self, current_scs_global: RotoTransMatrix) -> SegmentCoordinateSystemReal:
        """
        Get the modified segment coordinate system in local coordinates.
        """
        merged_scs = SegmentCoordinateSystemReal(scs=current_scs_global.inverse, is_scs_local=True)
        return merged_scs

    def get_modified_mesh(self, original_model: BiomechanicalModelReal, segment_name: str, current_scs_global: RotoTransMatrix) -> MeshReal:

        mesh_points = original_model.segments[segment_name].mesh.positions
        segment_scs_global = original_model.segment_coordinate_system_in_global(segment_name)
        modified_mesh_points = points_to_array(None, name="modified_mesh_points")

        for i_mesh in range(mesh_points.shape[1]):
            point_in_global = point_from_local_to_global(mesh_points[:, i_mesh], segment_scs_global)
            point_in_new_local = point_from_global_to_local(point_in_global, current_scs_global)
            modified_mesh_points = np.hstack((modified_mesh_points, point_in_new_local))

        modified_mesh = MeshReal(modified_mesh_points)
        return modified_mesh

    def get_modified_mesh(self, original_model: BiomechanicalModelReal, segment_name: str,
                          current_scs_global: RotoTransMatrix) -> MeshReal:

        mesh_points = original_model.segments[segment_name].mesh.positions
        segment_scs_global = original_model.segment_coordinate_system_in_global(segment_name)
        modified_mesh_points = points_to_array(None, name="modified_mesh_points")

        for i_mesh in range(mesh_points.shape[1]):
            point_in_global = point_from_local_to_global(mesh_points[:, i_mesh], segment_scs_global)
            point_in_new_local = point_from_global_to_local(point_in_global, current_scs_global)
            modified_mesh_points = np.hstack((modified_mesh_points, point_in_new_local))

        modified_mesh = MeshReal(modified_mesh_points)
        return modified_mesh
    
    def modify(self, original_model: BiomechanicalModelReal, modified_model: BiomechanicalModelReal) -> BiomechanicalModelReal:

        # Get the segments to invert
        segment_to_invert = self.get_segments_to_invert(modified_model)

        # Invert the segments
        current_parent = "root"
        if self.first_scs is None:
            current_scs_global = modified_model.segment_coordinate_system_in_global(self.first_segment_name)
        else:
            current_scs_global = self.first_scs

        for segment in segment_to_invert:

            # Remove them from the segments
            modified_model.remove_segment(segment.name)

            (modified_translations,
             modified_rotations,
             modified_dof_names,
             modified_q_ranges,
             modified_qdot_ranges) = self.get_modified_dofs(original_model, current_parent)

            modified_inertia_parameters = self.get_modified_inertia(original_model, segment.name, current_scs_global)

            modified_scs_local = self.get_modified_scs_local(current_scs_global)

            modified_mesh = self.get_modified_mesh(original_model, segment.name, current_scs_global)

            modified_mesh_file = self.get_modified_mesh_file(original_model, segment.name, current_scs_global)

            merged_segment = SegmentReal(
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

            current_scs = current_scs

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

        return self.merged_model
