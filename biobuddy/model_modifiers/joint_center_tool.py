from copy import deepcopy
import logging
import numpy as np

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.biomechanical_model_real_utils import (
    segment_coordinate_system_in_global,
    markers_in_global,
    contacts_in_global,
    point_from_global_to_local,
    forward_kinematics,
)
from ..components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ..utils.c3d_data import C3dData
from ..utils.linear_algebra import RotoTransMatrix, get_rt_aligning_markers_in_global

_logger = logging.getLogger(__name__)


class Score:
    def __init__(
        self,
        file_path: str,
        parent_name: str,
        child_name: str,
        parent_marker_names: list[str],
        child_marker_names: list[str],
        first_frame: int,
        last_frame: int,
    ):
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

        illegal_names = ["_parent_offset", "_translation", "_rotation_transform", "_reset_axis"]
        for name in illegal_names:
            if name in parent_name:
                raise RuntimeError(
                    f"The names {name} are not allowed in the parent or child names. Please change the segment named {parent_name} from the Score configuration."
                )
            if name in child_name:
                raise RuntimeError(
                    f"The names {name} are not allowed in the parent or child names. Please change the segment named {child_name} from the Score configuration."
                )

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

    def _rt_from_trial(self, original_model: "BiomechanicalModelReal") -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the rigid transformation matrices rt (4×4×N) that align local marker positions to global marker positions over time.

        Parameters
        ----------
        original_model
            The scaled model

        Returns
        ----------

        """
        # Get the segment RT in static pose to compute the marker position in the local reference frame
        parent_jcs_in_global = RotoTransMatrix()
        parent_jcs_in_global.rt_matrix = segment_coordinate_system_in_global(original_model, self.parent_name)
        parent_markers_local = parent_jcs_in_global.inverse @ self.c3d_data.mean_marker_positions(
            self.parent_marker_names
        )

        child_jcs_in_global = RotoTransMatrix()
        child_jcs_in_global.rt_matrix = segment_coordinate_system_in_global(original_model, self.child_name)
        child_markers_local = child_jcs_in_global.inverse @ self.c3d_data.mean_marker_positions(self.child_marker_names)

        # Marker positions in the global
        parent_markers_global = self.c3d_data.get_position(self.parent_marker_names)
        child_markers_global = self.c3d_data.get_position(self.child_marker_names)

        # Centroid of local marker set (constant)
        parent_local_centroid = np.mean(parent_markers_local, axis=1, keepdims=True)
        parent_local_centered = parent_markers_local - parent_local_centroid
        child_local_centroid = np.mean(child_markers_local, axis=1, keepdims=True)
        child_local_centered = child_markers_local - child_local_centroid

        nb_frames = self.c3d_data.all_marker_positions.shape[2]
        rt_parent = np.zeros((4, 4, nb_frames))
        rt_child = np.zeros((4, 4, nb_frames))
        for i_frame in range(nb_frames):
            # Finding the RT allowing to align the segments' markers
            rt_parent[:, :, i_frame] = get_rt_aligning_markers_in_global(
                parent_markers_global[:, :, i_frame], parent_local_centered, parent_local_centroid
            )
            rt_child[:, :, i_frame] = get_rt_aligning_markers_in_global(
                child_markers_global[:, :, i_frame], child_local_centered, child_local_centroid
            )

        return rt_parent, rt_child

    def _score_algorithm(
        self, parent_rt: np.ndarray, child_rt: np.ndarray, recursive_outlier_removal: bool = True
    ) -> np.ndarray:
        """
        Estimate the center of rotation (CoR) using the SCoRE algorithm (Ehrig et al., 2006).

        Parameters
        ----------
        parent_rt : np.ndarray, shape (4, 4, N)
            Homogeneous transformations of the parent segment (e.g., pelvis)
        child_rt : np.ndarray, shape (4, 4, N)
            Homogeneous transformations of the child segment (e.g., femur)
        recursive_outlier_removal : bool
            If True, performs 95th percentile residual filtering and recomputes the center.

        Returns
        -------
        CoR_global : np.ndarray, shape (3,)
            Estimated global position of the center of rotation.
        """
        nb_frames = parent_rt.shape[2]

        # Build linear system A x = b to solve for CoR positions in child and parent segment frames
        A = np.zeros((3 * nb_frames, 6))
        b = np.zeros((3 * nb_frames,))

        for i in range(nb_frames):
            parent_rot = parent_rt[:3, :3, i]
            child_rot = child_rt[:3, :3, i]
            parent_trans = parent_rt[:3, 3, i]
            child_trans = child_rt[:3, 3, i]

            A[3 * i : 3 * (i + 1), 0:3] = child_rot
            A[3 * i : 3 * (i + 1), 3:6] = -parent_rot
            b[3 * i : 3 * (i + 1)] = parent_trans - child_trans

        # Solve via least squares: A x = b → x = [CoR2_local; CoR1_local]
        x, residuals_ls, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cor_child_local = x[:3]
        cor_parent_local = x[3:]

        # Compute transformed CoR positions in global frame
        cor_parent_global = np.einsum("ijk,k->ij", parent_rt, np.append(cor_parent_local, 1))  # shape (4, N)
        cor_child_global = np.einsum("ijk,k->ij", child_rt, np.append(cor_child_local, 1))

        residuals = np.linalg.norm(cor_parent_global[:3, :] - cor_child_global[:3, :], axis=0)

        if recursive_outlier_removal:
            threshold = np.percentile(residuals, 95)
            valid = residuals < threshold
            if np.sum(valid) < nb_frames:
                return self._score_algorithm(parent_rt[:, :, valid], child_rt[:, :, valid], recursive_outlier_removal)

        # Final output
        cor_mean_global = 0.5 * (np.mean(cor_parent_global[:3, :], axis=1) + np.mean(cor_child_global[:3, :], axis=1))

        _logger.info(
            f"\nThere is a residual distance between the parent's and the child's CoR position of : {residuals}"
        )
        return cor_mean_global

    def perform_task(self, original_model: BiomechanicalModelReal, new_model: BiomechanicalModelReal):

        # Reconstruct the trial using the current model to identify the orientation of the segments
        rt_parent, rt_child = self._rt_from_trial(original_model)

        # Apply the algo to identify the joint center
        cor_in_global = self._score_algorithm(rt_parent, rt_child)

        # Replace the model components in the new local reference frame
        parent_cor_position_in_global = segment_coordinate_system_in_global(new_model, self.parent_name)[:3, 3, 0]

        if (
            new_model.segments[self.child_name].segment_coordinate_system is None
            or new_model.segments[self.child_name].segment_coordinate_system.is_in_global
        ):
            raise RuntimeError(
                "The child segment is not in local reference frame. Please set it to local before using the SCoRE algorithm."
            )
        scs_in_local = deepcopy(new_model.segments[self.child_name].segment_coordinate_system.scs)
        scs_in_local[:3, 3] = cor_in_global[:3] - parent_cor_position_in_global

        # Segment RT
        if self.child_name + "_parent_offset" in new_model.segment_names:
            segment_to_move_rt_from = self.child_name + "_parent_offset"
        else:
            segment_to_move_rt_from = self.child_name
        new_model.segments[segment_to_move_rt_from].segment_coordinate_system = SegmentCoordinateSystemReal(
            scs=scs_in_local,
            is_scs_local=True,
        )
        # Markers
        marker_positions = markers_in_global(original_model)
        for i_marker, marker_name in new_model.segments[self.child_name].markers:
            new_model.segments[self.child_name].markers[marker_name].position = point_from_global_to_local(
                marker_positions[i_marker], cor_in_global
            )
        # Contacts
        contact_positions = contacts_in_global(original_model)
        for i_contact, contact_name in new_model.segments[self.child_name].contacts:
            new_model.segments[self.child_name].contacts[contact_name].position = point_from_global_to_local(
                contact_positions[i_contact], cor_in_global
            )
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
        current_segment = deepcopy(self.original_model.segments[jcs_identifier.child_name])
        while current_segment.parent_name != jcs_identifier.parent_name:
            current_segment = deepcopy(self.original_model.segments[current_segment.parent_name])
            if (
                current_segment.parent_name == ""
                or current_segment.parent_name == "base"
                or current_segment.parent_name is None
            ):
                raise RuntimeError(
                    f"The segment {jcs_identifier.child_name} is not the child of the segment {jcs_identifier.parent_name}. Please check the kinematic chain again"
                )

    def replace_joint_centers(self) -> BiomechanicalModelReal:

        for task in self.joint_center_tasks:
            task.perform_task(self.original_model, self.new_model)

        return self.new_model
