from copy import deepcopy
import logging
import numpy as np

import biorbd

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.biomechanical_model_real_utils import (
    inverse_kinematics,
    segment_coordinate_system_in_global,
    markers_in_global,
    contacts_in_global,
    point_from_global_to_local,
)
from ..components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ..utils.c3d_data import C3dData

_logger = logging.getLogger(__name__)

class Score:
    def __init__(self,
                 file_path: str,
                 parent_name: str,
                 child_name: str,
                 parent_marker_names: list[str],
                 child_marker_names: list[str],
                 first_frame: int,
                 last_frame: int):
        """
        Initializes the Score class which will find the position of the joint center using functional movements.
        The SCoRE algorithm considers that both segments are rigid bodies and that the joint center is located at the
        intersection of the two segments.
        TODO: Add algo ref link.

        Parameters
        ----------
        file_path
            The path to the .c3d file containing the functional trial.
        parent_name
            The name of the joint's parent segment.
        child_name
            The name of the joint's child segment.
        parent_marker_names
            The name of the markers in the parent segment to consider during the SCoRE algorithm.
        child_marker_names
            The name of the markers in the child segment to consider during the SCoRE algorithm.
        first_frame
            The first frame to consider in the functional trial.
        last_frame
            The last frame to consider in the functional trial.
        """

        # Original attributes
        self.file_path = file_path
        self.parent_name = parent_name
        self.child_name = child_name
        self.parent_marker_names = parent_marker_names
        self.child_marker_names = child_marker_names

        # Check file format
        if file_path.endswith(".c3d"):
            self.c3d_data = C3dData(file_path, first_frame, last_frame)
        else:
            if file_path.endswith(".trc"):
                raise NotImplementedError(".trc files cannot be read yet.")
            else:
                raise RuntimeError("The file_path (static trial) must be a .c3d file in a static posture.")

    def _rt_from_trial(self, original_model_biorbd: biorbd.Model) -> tuple[np.ndarray, np.ndarray]:

        optimal_q = inverse_kinematics(
            original_model_biorbd,
            marker_positions=self.c3d_data.all_marker_positions,
            marker_names=self.c3d_data.marker_names)

        segment_names = [s.name.to_string() for s in original_model_biorbd.segments()]
        nb_frames = self.c3d_data.all_marker_positions.shape[2]
        rt_parent = np.zeros((4, 4, nb_frames))
        rt_child = np.zeros((4, 4, nb_frames))
        parent_idx = segment_names.index(self.parent_name)
        child_idx = segment_names.index(self.child_name)
        for i_frame in range(nb_frames):
            rt_parent[:, :, i_frame] = original_model_biorbd.globalJCS(optimal_q[:, i_frame], parent_idx, True).to_array()
            rt_child[:, :, i_frame] = original_model_biorbd.globalJCS(optimal_q[:, i_frame], child_idx, True).to_array()

        return rt_parent, rt_child


    def _score_algorithm(self,
                         parent_markers: np.ndarray,
                         child_markers: np.ndarray,
                         rt_parent: np.ndarray,
                         rt_child: np.ndarray,
                         ):

        def apply_transform(rt, cor_part):
            homog_cor = np.append(cor_part, 1)
            return np.einsum('ijk,k->ij', rt, homog_cor)

        nb_frames = parent_markers.shape[2]
        a_matrix = np.full((3 * nb_frames, 6), np.nan)
        b_vector = np.full((3 * nb_frames, 1), np.nan)

        for i_frame in range(nb_frames):
            a_matrix[i_frame * 3:i_frame * 3 + 3, :] = np.hstack((
                rt_child[:3, :3, i_frame],
                -rt_parent[:3, :3, i_frame]
            ))
            b_vector[i_frame * 3:i_frame * 3 + 3, :] = (
                    rt_parent[:3, 3, i_frame].reshape(3, 1) - rt_child[:3, 3, i_frame].reshape(3, 1)
            )

        valid_rows = ~np.isnan(a_matrix[:, 0])
        U, S, Vt = np.linalg.svd(a_matrix[valid_rows, :], full_matrices=False)
        V = Vt.T
        dS = S

        cor_in_local = V @ np.diag(1.0 / dS) @ U.T @ b_vector[valid_rows].flatten()

        parent_current_cor_in_global = apply_transform(rt_parent, cor_in_local[3:6])
        child_current_cor_in_global = apply_transform(rt_child, cor_in_local[0:3])
        residual = np.sqrt(np.sum((parent_current_cor_in_global - child_current_cor_in_global) ** 2, axis=0))

        cor_in_global = 0.5 * (parent_current_cor_in_global + child_current_cor_in_global)

        # Compute distances of markers to the center of rotation
        diff_parent_markers = parent_markers - cor_in_global[:3].reshape(3, 1, -1)
        cor_parent_markers_distance = np.sqrt(np.sum(diff_parent_markers ** 2, axis=0)).T

        diff_child_markers = child_markers - cor_in_global[:3].reshape(3, 1, -1)
        cor_child_markers_distance = np.sqrt(np.sum(diff_child_markers ** 2, axis=0)).T

        _logger.info(f"The std of markers position is {cor_parent_markers_distance, cor_child_markers_distance}")
        if residual > 0.25:
            raise RuntimeError(
                f"The distance between the parent {self.parent_name} and the child {self.child_name} CoR position is too high. Please make sure that the maker do not move on the segments during the functional trials and that there are enough markers on each segments.")

        _logger.info(f"\nThere is a residual distance between the parent's and the child's CoR position of : {residual}")
        if residual > 0.25:
            raise RuntimeError(f"The distance between the parent {self.parent_name} and the child {self.child_name} CoR position is too high. Please make sure that the maker do not move on the segments during the functional trials and that there are enough markers on each segments.")

        return cor_in_global


    def perform_task(self, original_model_biorbd: biorbd.Model, original_model: BiomechanicalModelReal, new_model: BiomechanicalModelReal):

        # Reconstruct the trial using the current model to identify the orientation of the segments
        rt_parent, rt_child = self._rt_from_trial(original_model_biorbd)

        # Apply the algo to identify the joint center
        parent_markers = self.c3d_data.get_position(self.parent_marker_names)
        child_markers = self.c3d_data.get_position(self.child_marker_names)
        cor_in_global = self._score_algorithm(parent_markers=parent_markers,
                              child_markers=child_markers,
                              rt_parent=rt_parent,
                              rt_child=rt_child)

        # Replace the model components in the new local reference frame
        parent_cor_position_in_global = segment_coordinate_system_in_global(new_model, self.parent_name)[:3, 3, 0]
        scs_in_local = deepcopy(new_model.segments[self.child_name].segment_coordinate_system.scs)
        scs_in_local[:3, 3] = cor_in_global[:3] - parent_cor_position_in_global

        # Segment RT
        new_model.segments[self.child_name].segment_coordinate_system = SegmentCoordinateSystemReal(
            scs=scs_in_local,
            is_scs_local=True,
        )
        # Markers
        marker_positions = markers_in_global(original_model)
        for i_marker, marker_name in new_model.segments[self.child_name].markers:
            new_model.segments[self.child_name].markers[marker_name].position = point_from_global_to_local(marker_positions[i_marker], cor_in_global)
        # Contacts
        contact_positions = contacts_in_global(original_model)
        for i_contact, contact_name in new_model.segments[self.child_name].contacts:
            new_model.segments[self.child_name].contacts[contact_name].position = point_from_global_to_local(contact_positions[i_contact], cor_in_global)
        # IMUs
        # Muscles origin, insertion, via points


class Sara:
    def __init__(self, file_path: str, parent_name: str, child_name: str, first_frame: int, last_frame: int):
        """
        Initializes the Sara class which will find the position of the joint center using functional movements.
        The algorithm considers that both segments are rigid bodies and that the joint center is located at the
        intersection of the two segments.
        TODO: Add algo ref link.

        Parameters
        ----------
        file_path
            The path to the .c3d file containing the functional trial.
        parent_name
            The name of the joint's parent segment.
        child_name
            The name of the joint's child segment.
        first_frame
            The first frame to consider in the functional trial.
        last_frame
            The last frame to consider in the functional trial.
        """

        # Original attributes
        self.file_path = file_path
        self.parent_name = parent_name
        self.child_name = child_name

        # Check file format
        if file_path.endswith(".c3d"):
            # Load the c3d file
            c3d_data = C3dData(file_path, first_frame, last_frame)
            self.marker_names = c3d_data.marker_names
            self.marker_positions = c3d_data.all_marker_positions[:3, :, :]
        else:
            if file_path.endswith(".trc"):
                raise NotImplementedError(".trc files cannot be read yet.")
            else:
                raise RuntimeError("The file_path (static trial) must be a .c3d file in a static posture.")

        raise NotImplementedError("The SARA algorithm is not implemented yet.")


class JointCenterTool:
    def __init__(self, original_model: BiomechanicalModelReal):

        # Original attributes
        self.original_model = original_model
        self.original_model_biorbd = self.original_model.get_biorbd_model  # TODO: remove

        # Extended attributes to be filled
        self.joint_center_tasks = []  # Not a NamedList because nothing in BioBuddy refer to joints (only segments)
        self.new_model = deepcopy(original_model)

    def add(self, jcs_identifier: Score | Sara):
        """
        Add a joint center identification task to the pipeline.

        Parameters
        ----------
        jcs_identifier
            The type of algorithm to use to identify the joint center (and the parameters necessary for computation).
        """

        # Check that the jcs_identifier is a Score or Sara object
        if isinstance(jcs_identifier, Score):
            self.joint_center_tasks.append(jcs_identifier)
        elif isinstance(jcs_identifier, Sara):
            self.joint_center_tasks.append(jcs_identifier)
        else:
            raise RuntimeError("The joint center must be a Score or Sara object.")

        # Check that there is really a link between parent and child segments
        if self.original_model.segments[jcs_identifier.child_name].parent_name != jcs_identifier.parent_name:
            raise RuntimeError(
                f"The segment {jcs_identifier.child_name} is not the child of the segment {jcs_identifier.parent_name}."
            )


    def replace_joint_centers(self) -> BiomechanicalModelReal:

        for task in self.joint_center_tasks:
            task.perfrom_task(self.original_model_biorbd, self.new_model)

        return self.new_model