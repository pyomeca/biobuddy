from copy import deepcopy
import logging
import numpy as np
from scipy import optimize
from functools import wraps

from ...utils.linear_algebra import RotoTransMatrix, get_closest_rotation_matrix, point_from_local_to_global

_logger = logging.getLogger(__name__)


def requires_initialization(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.is_initialized:
            raise RuntimeError(f"{method.__name__} cannot be called because the object is not initialized.")
        return method(self, *args, **kwargs)

    return wrapper


class ModelDynamics:
    def __init__(self):

        # This tag makes sure the ModelDynamics cannot be called alone without BiomechanicalModelReal
        self.is_initialized = False

        # Attributes that will be filled by BiomechanicalModelReal
        self.segments = None
        self.muscle_groups = None
        self.muscles = None
        self.via_points = None

    # TODO: The two following functions should be handled differently
    @requires_initialization
    def segment_coordinate_system_in_local(self, segment_name: str) -> np.ndarray:
        """
        Transforms a SegmentCoordinateSystemReal expressed in the global reference frame into a SegmentCoordinateSystemReal expressed in the local reference frame.

        Parameters
        ----------
        segment_name
            The name of the segment whose SegmentCoordinateSystemReal should be expressed in the local

        Returns
        -------
        The SegmentCoordinateSystemReal in local reference frame
        """

        if segment_name == "base":
            return np.identity(4)
        elif self.segments[segment_name].segment_coordinate_system.is_in_local:
            return self.segments[segment_name].segment_coordinate_system.scs[:, :, 0]
        else:

            parent_name = self.segments[segment_name].parent_name
            parent_scs = RotoTransMatrix()
            parent_scs.rt_matrix = self.segment_coordinate_system_in_global(segment_name=parent_name)
            inv_parent_scs = parent_scs.inverse
            scs_in_local = inv_parent_scs @ self.segments[segment_name].segment_coordinate_system.scs[:, :, 0]
            return get_closest_rotation_matrix(scs_in_local)[:, :, np.newaxis]

    @requires_initialization
    def segment_coordinate_system_in_global(self, segment_name: str) -> np.ndarray:
        """
        Transforms a SegmentCoordinateSystemReal expressed in the local reference frame into a SegmentCoordinateSystemReal expressed in the global reference frame.

        Parameters
        ----------
        segment_name
            The name of the segment whose SegmentCoordinateSystemReal should be expressed in the global

        Returns
        -------
        The SegmentCoordinateSystemReal in global reference frame
        """

        if segment_name == "base":
            return np.identity(4)
        elif self.segments[segment_name].segment_coordinate_system.is_in_global:
            return self.segments[segment_name].segment_coordinate_system.scs[:, :, 0]

        else:

            current_segment = self.segments[segment_name]
            rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0]
            while current_segment.segment_coordinate_system.is_in_local:
                current_parent_name = current_segment.parent_name
                if (
                    current_parent_name == "base" or current_parent_name is None
                ):  # @pariterre : is this really hardcoded in biorbd ? I thought it was "root"
                    break
                current_segment = self.segments[current_parent_name]
                rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0] @ rt_to_global

            return get_closest_rotation_matrix(rt_to_global)[:, :, np.newaxis]

    @staticmethod
    def _marker_residual(
        model: "BiomechanicalModelReal" or "biorbd.Model",
        q_regularization_weight: float,
        q_target: np.ndarray,
        q: np.ndarray,
        experimental_markers: np.ndarray,
        with_biorbd: bool,
    ) -> np.ndarray:

        nb_markers = experimental_markers.shape[1]
        vect_pos_markers = np.zeros(3 * nb_markers)

        if with_biorbd:
            markers_model = np.zeros((3, nb_markers, 1))
            for i_marker in range(nb_markers):
                markers_model[:, i_marker, 0] = model.marker(q, i_marker, True).to_array()
        else:
            markers_model = np.array(model.markers_in_global(q))

        for i_marker in range(nb_markers):
            vect_pos_markers[i_marker * 3 : (i_marker + 1) * 3] = (
                markers_model[:3, i_marker, 0] - experimental_markers[:3, i_marker]
            )
        # TODO: setup the IKTask to set the "q_ref" to something else than zero.
        out = np.hstack(
            (
                vect_pos_markers,
                q_regularization_weight
                * (
                    q
                    - q_target.reshape(
                        -1,
                    )
                ),
            )
        )
        return out

    @staticmethod
    def _marker_jacobian(
        model: "BiomechanicalModelReal" or "biorbd.Model",
        q_regularization_weight: float,
        q: np.ndarray,
        with_biorbd: bool,
    ) -> np.ndarray:
        nb_q = q.shape[0]
        nb_markers = model.nbMarkers() if with_biorbd else model.nb_markers
        vec_jacobian = np.zeros((3 * nb_markers + nb_q, nb_q))

        if with_biorbd:
            jacobian_matrix = np.zeros((3, nb_markers, nb_q))
            for i_marker in range(nb_markers):
                jacobian_matrix[:, i_marker, :] = model.markersJacobian(q)[i_marker].to_array()
        else:
            jacobian_matrix = np.array(model.markers_jacobian(q))

        for i_marker in range(nb_markers):
            vec_jacobian[i_marker * 3 : (i_marker + 1) * 3, :] = jacobian_matrix[:, i_marker, :]

        for i_q in range(nb_q):
            vec_jacobian[nb_markers * 3 + i_q, i_q] = q_regularization_weight

        return vec_jacobian

    @requires_initialization
    def inverse_kinematics(
        self,
        marker_positions: np.ndarray,
        marker_names: list[str],
        q_regularization_weight: float = None,
        q_target: np.ndarray = None,
    ) -> np.ndarray:
        """
        Solve the inverse kinematics problem using least squares optimization.
        The objective is to match the experimental marker positions with the model marker positions.
        There is also a regularization term matching a predefined posture q_target weighted using q_regularization_weight.
        By default, the q_target is zero, and there is no weight on the regularization term.

        Parameters
        ----------
        marker_positions
            The experimental marker positions
        marker_names
            The names of the experimental markers (the names must match the marker names in the model).
        q_regularization_weight
            The weight of the regularization term. If None, no regularization is applied.
        q_target
            The target posture to match. If None, the target posture is set to zero.
        """

        try:
            # biorbd (in c++) is quicker than this custom Python code, which makes a large difference here
            import biorbd

            self.to_biomod("temporary.bioMod", with_mesh=False)
            with_biorbd = True
            model_to_use = biorbd.Model("temporary.bioMod")

            _logger.info(f"Using biorbd for the inverse kinematics as it is faster")
        except:
            with_biorbd = False
            model_to_use = deepcopy(self)
            _logger.info(
                f"Using slower Python code for the inverse kinematics as either biorbd is not installed or the model is not compatible with biorbd."
            )

        marker_indices = [marker_names.index(m) for m in self.marker_names]
        markers_real = marker_positions[:, marker_indices, :]

        nb_q = self.nb_q
        nb_frames = marker_positions.shape[2]

        init = np.ones((nb_q,)) * 0.0001
        if q_target is not None:
            init[:] = q_target
        else:
            q_target = np.zeros((self.nb_q, 1))

        if q_regularization_weight is None:
            q_regularization_weight = 0.0

        optimal_q = np.zeros((self.nb_q, nb_frames))
        for f in range(nb_frames):
            sol = optimize.least_squares(
                fun=lambda q: self._marker_residual(
                    model_to_use, q_regularization_weight, q_target, q, markers_real[:, :, f], with_biorbd
                ),
                jac=lambda q: self._marker_jacobian(model_to_use, q_regularization_weight, q, with_biorbd),
                x0=init,
                method="lm",
                xtol=1e-6,
                tr_options=dict(disp=False),
            )
            optimal_q[:, f] = sol.x

        return optimal_q

    @requires_initialization
    def forward_kinematics(self, q: np.ndarray = None) -> dict[str, np.ndarray]:
        """
        Applied the generalized coordinates to move find the position and orientation of the model's segments.
        Here, we assume that the parent is always defined before the child in the model.
        """
        if q is None:
            q = np.zeros((self.nb_q, 1))
        elif len(q.shape) == 1:
            q = q[:, np.newaxis]
        elif len(q.shape) > 2:
            raise RuntimeError("q must be of shape (nb_q, ) or (nb_q, nb_frames).")
        nb_frames = q.shape[1]

        segment_rt_in_global = {}
        for segment_name in self.segments.keys():

            if not self.segments[segment_name].segment_coordinate_system.is_in_local:
                raise NotImplementedError(
                    "The function forward_kinematics is not implemented yet for global rt. They should be converted to local."
                )

            segment_rt_in_global[segment_name] = np.ones((4, 4, nb_frames))
            for i_frame in range(nb_frames):
                segment_rt = self.segments[segment_name].segment_coordinate_system.scs[:, :, 0]
                parent_name = self.segments[segment_name].parent_name
                if parent_name == "base":
                    parent_rt = np.identity(4)
                else:
                    parent_rt = segment_rt_in_global[parent_name][:, :, i_frame]

                if self.segments[segment_name].nb_q == 0:
                    segment_rt_in_global[segment_name][:, :, i_frame] = parent_rt @ segment_rt
                else:
                    local_q = q[self.dof_indices(segment_name), i_frame]
                    rt_caused_by_q = self.segments[segment_name].rt_from_local_q(local_q)
                    segment_rt_in_global[segment_name][:, :, i_frame] = parent_rt @ segment_rt @ rt_caused_by_q

        return segment_rt_in_global

    @requires_initialization
    def markers_in_global(self, q: np.ndarray = None) -> np.ndarray:

        q = np.zeros((self.nb_q, 1)) if q is None else q
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        elif len(q.shape) > 2:
            raise RuntimeError("q must be of shape (nb_q, ) or (nb_q, nb_frames).")

        nb_frames = q.shape[1]

        marker_positions = np.ones((4, self.nb_markers, nb_frames))
        jcs_in_global = self.forward_kinematics(q)
        for i_frame in range(nb_frames):
            i_marker = 0
            for i_segment, segment in enumerate(self.segments):
                for marker in segment.markers:
                    marker_in_global = point_from_local_to_global(
                        point_in_local=marker.position, jcs_in_global=jcs_in_global[segment.name][:, :, i_frame]
                    )
                    marker_positions[:, i_marker] = marker_in_global
                    i_marker += 1

        return marker_positions

    @requires_initialization
    def contacts_in_global(self, q: np.ndarray = None) -> np.ndarray:

        q = np.zeros((self.nb_q, 1)) if q is None else q
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        elif len(q.shape) > 2:
            raise RuntimeError("q must be of shape (nb_q, ) or (nb_q, nb_frames).")

        nb_frames = q.shape[1]

        jcs_in_global = self.forward_kinematics(q)

        contact_positions = np.ones((4, self.nb_contacts, nb_frames))
        for i_frame in range(nb_frames):
            i_contact = 0
            for i_segment, segment in enumerate(self.segments):
                for contact in segment.contacts:
                    contact_in_global = point_from_local_to_global(
                        point_in_local=contact.position, jcs_in_global=jcs_in_global[segment][:, :, i_frame]
                    )
                    contact_positions[:, i_contact] = contact_in_global
                    i_contact += 1

        return contact_positions

    @requires_initialization
    def markers_jacobian(self, q: np.ndarray, epsilon: float = 0.0001) -> np.ndarray:
        """
        Numerically compute the Jacobian of marker position with respect to q.

        Parameters
        ----------
        q : np.ndarray
            Generalized coordinates (nb_q, 1).
        epsilon : float
            Perturbation size for finite difference.

        Returns
        -------
        np.ndarray
            Jacobian of shape (3, nb_q)
        """
        nb_q = self.nb_q
        nb_markers = self.nb_markers
        jac = np.zeros((3, nb_markers, nb_q))
        f0 = self.markers_in_global(q)[:3, :, 0]

        for i_q in range(nb_q):
            dq = np.zeros_like(q)
            dq[i_q] = epsilon
            f1 = self.markers_in_global(q + dq)[:3, :, 0]
            for i_marker in range(nb_markers):
                jac[:, i_marker, i_q] = (f1[:, i_marker] - f0[:, i_marker]) / epsilon

        return jac

    @requires_initialization
    def muscle_length(self, muscle_name: str, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Please implement the muscle_length function.")
