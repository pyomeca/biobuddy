from enum import Enum
from copy import deepcopy
import logging
import numpy as np
from scipy import optimize
from functools import wraps

from ...utils.linear_algebra import (
    RotoTransMatrix,
    RotoTransMatrixTimeSeries,
    point_from_local_to_global,
)
from ...utils.enums import ViewAs

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

    # TODO: The two following functions should be handled differently
    @requires_initialization
    def segment_coordinate_system_in_local(self, segment_name: str) -> RotoTransMatrix:
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
            return RotoTransMatrix()
        elif self.segments[segment_name].segment_coordinate_system.is_in_local:
            # Already in local
            return self.segments[segment_name].segment_coordinate_system.scs
        else:
            # In global -> need to transform it into local coordinates
            parent_name = self.segments[segment_name].parent_name
            parent_scs = self.segment_coordinate_system_in_global(segment_name=parent_name)
            scs_in_local = parent_scs.inverse @ self.segments[segment_name].segment_coordinate_system.scs
            return scs_in_local

    @requires_initialization
    def segment_coordinate_system_in_global(self, segment_name: str) -> RotoTransMatrix:
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
            return RotoTransMatrix()
        elif self.segments[segment_name].segment_coordinate_system.is_in_global:
            # Already in global
            return self.segments[segment_name].segment_coordinate_system.scs
        else:
            # In local -> need to transform it into global coordinates
            current_segment = self.segments[segment_name]
            rt_to_global = current_segment.segment_coordinate_system.scs
            while current_segment.segment_coordinate_system.is_in_local:
                current_parent_name = current_segment.parent_name
                if current_parent_name == "base":
                    break
                current_segment = self.segments[current_parent_name]
                rt_to_global = current_segment.segment_coordinate_system.scs @ rt_to_global

            return rt_to_global

    @requires_initialization
    def rt_from_parent_offset_to_real_segment(self, segment_name: str) -> RotoTransMatrix:
        """
        Computes the RotoTransMatrix from the [segment_name]_parent_offset to the real [segment_name] segment.
        """
        parent_name = self.segments[segment_name].parent_name
        parent_offset_name = segment_name + "_parent_offset"
        if parent_offset_name not in self.segment_names:
            if parent_name.startswith(segment_name):
                raise NotImplementedError(
                    f"The segment {segment_name} does not have a parent offset, but is attached another ghost segments. If you run into this error, please notify the developers by opening an issue on GitHub."
                )
            else:
                out_rt = RotoTransMatrix()
                out_rt.from_rt_matrix(np.identity(4))
                return out_rt
        else:
            rt = self.segments[segment_name].segment_coordinate_system.scs @ RotoTransMatrix()
            while parent_name != parent_offset_name:
                if parent_name == "base":
                    raise RuntimeError(f"The parent offset of segment {segment_name} was not found.")
                rt = self.segments[parent_name].segment_coordinate_system.scs @ rt
                parent_name = self.segments[parent_name].parent_name

            return rt

    def segment_has_ghost_parents(self, segment_name: str) -> bool:
        """
        Check if the segment has ghost parents.
        A ghost parent is a segment that does not hold inertia, but is used to define the segment's coordinate system.
        """
        ghost_keys = ["_parent_offset", "_translation", "_rotation_transform", "_reset_axis"]
        for key in ghost_keys:
            if segment_name + key in self.segments.keys():
                return True
        return False

    @staticmethod
    def _marker_residual(
        model: "BiomechanicalModelReal" or "biorbd.Model",
        q_regularization_weight: float,
        q_target: np.ndarray,
        q: np.ndarray,
        marker_names: list[str],
        experimental_markers: np.ndarray,
        marker_weights_reordered: np.ndarray,
        with_biorbd: bool,
    ) -> np.ndarray:

        nb_markers = experimental_markers.shape[1]
        vect_pos_markers = np.zeros(3 * nb_markers)

        if with_biorbd:
            markers_model = np.zeros((3, nb_markers, 1))
            for i_marker in range(nb_markers):
                if model.markerNames()[i_marker].to_string() in marker_names:
                    markers_model[:, i_marker, 0] = model.marker(q, i_marker, True).to_array()
        else:
            markers_model = np.array(model.markers_in_global(q))

        for i_marker in range(nb_markers):
            vect_pos_markers[i_marker * 3 : (i_marker + 1) * 3] = (
                markers_model[:3, i_marker, 0] - experimental_markers[:3, i_marker]
            ) * marker_weights_reordered[i_marker]
        # TODO: setup the IKTask from osim to set the "q_ref" to something else than zero.
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
        # Replace NaN with 0.0
        out[np.where(np.isnan(out))] = 0.0
        return out

    @staticmethod
    def _marker_distance(
        model: "BiomechanicalModelReal" or "biorbd.Model",
        q: np.ndarray,
        marker_names: list[str],
        experimental_markers: np.ndarray,
        with_biorbd: bool,
    ) -> np.ndarray:

        nb_markers = experimental_markers.shape[1]
        vect_pos_markers = np.zeros((nb_markers,))

        if with_biorbd:
            markers_model = np.zeros((3, nb_markers, 1))
            for i_marker in range(nb_markers):
                if model.markerNames()[i_marker].to_string() in marker_names:
                    markers_model[:, i_marker, 0] = model.marker(q, i_marker, True).to_array()
        else:
            markers_model = np.array(model.markers_in_global(q))

        for i_marker in range(nb_markers):
            vect_pos_markers[i_marker] = np.linalg.norm(
                (markers_model[:3, i_marker, 0] - experimental_markers[:3, i_marker])
            )

        return vect_pos_markers

    @staticmethod
    def _marker_jacobian(
        model: "BiomechanicalModelReal" or "biorbd.Model",
        q_regularization_weight: np.ndarray[float],
        q: np.ndarray,
        marker_names: list[str],
        marker_weights_reordered: np.ndarray,
        with_biorbd: bool,
    ) -> np.ndarray:
        nb_q = q.shape[0]
        nb_markers = marker_weights_reordered.shape[0]
        vec_jacobian = np.zeros((3 * nb_markers + nb_q, nb_q))

        if with_biorbd:
            jacobian_matrix = np.zeros((3, nb_markers, nb_q))
            for i_marker in range(nb_markers):
                if model.markerNames()[i_marker].to_string() in marker_names:
                    jacobian_matrix[:, i_marker, :] = (
                        model.markersJacobian(q)[i_marker].to_array() * marker_weights_reordered[i_marker]
                    )
        else:
            jacobian_matrix = np.array(model.markers_jacobian(q)) * marker_weights_reordered

        for i_marker in range(nb_markers):
            marker_name = model.markerNames()[i_marker].to_string() if with_biorbd else model.marker_names[i_marker]
            if marker_name in marker_names:
                vec_jacobian[i_marker * 3 : (i_marker + 1) * 3, :] = jacobian_matrix[:, i_marker, :]

        for i_q in range(nb_q):
            vec_jacobian[nb_markers * 3 + i_q, i_q] = q_regularization_weight[i_q]

        return vec_jacobian

    @requires_initialization
    def inverse_kinematics(
        self,
        marker_positions: np.ndarray,
        marker_names: list[str],
        q_regularization_weight: float | np.ndarray[float] = None,
        q_target: np.ndarray = None,
        marker_weights: "NamedList[MarkerWeight]" = None,
        method: str = "lm",
        animate_reconstruction: bool = False,
        compute_residual_distance: bool = False,
        verbose: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        marker_weights
            The weights of each marker to consider during the least squares. If None, all markers are equally weighted.
        method
            The least square method to use. By default, the Levenberg-Marquardt method is used.
        animate_reconstruction
            Weather to animate the reconstruction
            verbose
        The verbosity level of the optimization algorithm [0, 1, 2]. Default is 0 (no output).
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

        nb_q = self.nb_q
        nb_markers = len(marker_names)
        if len(marker_positions.shape) == 2:
            marker_positions = marker_positions[:, :, np.newaxis]
        if len(marker_positions.shape) != 3:
            raise RuntimeError(
                f"The marker_positions must be of shape (3, nb_markers, nb_frames). Here the shape provided is {marker_positions.shape}"
            )

        if marker_positions.shape[0] == 4:
            marker_positions = marker_positions[:3, :, :]  # Remove the homogeneous coordinate if present
        if marker_positions.shape[0] != 3:
            raise RuntimeError(
                f"The marker_positions must be of shape (3, nb_markers, nb_frames). Here the shape provided is {marker_positions.shape}"
            )

        nb_frames = marker_positions.shape[2]

        marker_indices = []
        marker_names_reordered = []
        for m in self.marker_names:
            if m in marker_names:
                marker_indices += [marker_names.index(m)]
                marker_names_reordered += [m]
        markers_real = marker_positions[:, marker_indices, :]

        marker_weights_reordered = np.ones((nb_markers,))
        if marker_weights is not None:
            for marker_name in marker_names_reordered:
                if marker_name not in marker_weights.keys():
                    raise ValueError(
                        f"Marker {marker_name} not found in marker_weights. Please provide a weight to each markers or None of them."
                    )
            for i_marker in range(nb_markers):
                marker_weights_reordered[i_marker] = marker_weights[marker_names_reordered[i_marker]].weight

        init = np.ones((nb_q,)) * 0.0001
        if q_target is not None:
            init[:] = q_target
        else:
            q_target = np.zeros((self.nb_q, 1))

        if q_regularization_weight is None:
            q_regularization_weight = np.zeros((self.nb_q,))  # No regularization by default
        elif isinstance(q_regularization_weight, (int, float)):
            q_regularization_weight = np.ones((self.nb_q,)) * q_regularization_weight
        else:
            if len(q_regularization_weight) != self.nb_q:
                raise RuntimeError(
                    f"The q_regularization_weight must be of shape (nb_q, ). Here the shape provided is {q_regularization_weight.shape}"
                )

        optimal_q = np.zeros((self.nb_q, nb_frames))
        residuals = None
        if compute_residual_distance:
            residuals = np.zeros((nb_markers, nb_frames))
        for i_frame in range(nb_frames):

            if i_frame % 100 == 0 and i_frame != 0:
                print(f"{i_frame}/{nb_frames} frames")

            sol = optimize.least_squares(
                fun=lambda q: self._marker_residual(
                    model_to_use,
                    q_regularization_weight,
                    q_target,
                    q,
                    marker_names_reordered,
                    markers_real[:, :, i_frame],
                    marker_weights_reordered,
                    with_biorbd,
                ),
                jac=lambda q: self._marker_jacobian(
                    model_to_use,
                    q_regularization_weight,
                    q,
                    marker_names_reordered,
                    marker_weights_reordered,
                    with_biorbd,
                ),
                x0=init,
                method=method,
                xtol=1e-5,
                ftol=1e-5,
                tr_options=dict(disp=False),
                verbose=verbose,
            )
            optimal_q[:, i_frame] = sol["x"]
            if compute_residual_distance:
                residuals[:, i_frame] = self._marker_distance(
                    model_to_use,
                    optimal_q[:, i_frame],
                    marker_names_reordered,
                    markers_real[:, :, i_frame],
                    with_biorbd=with_biorbd,
                )

        if animate_reconstruction:
            if not with_biorbd:
                raise RuntimeError(
                    "To animate the inverse kinematics reconstruction, your model should be to_biomod-able."
                )
            else:

                # Compare the result visually
                import pyorerun

                t = np.linspace(0, 1, optimal_q.shape[1])
                viz = pyorerun.PhaseRerun(t)

                # Add the experimental markers from the static trial
                pyomarkers = pyorerun.PyoMarkers(data=markers_real, channels=marker_names_reordered, show_labels=False)
                viz_scaled_model = pyorerun.BiorbdModel("temporary.bioMod")
                viz_scaled_model.options.transparent_mesh = False
                viz_scaled_model.options.show_gravity = True
                viz_scaled_model.options.show_marker_labels = False
                viz_scaled_model.options.show_center_of_mass_labels = False
                viz.add_animated_model(viz_scaled_model, optimal_q, tracked_markers=pyomarkers)
                viz.rerun_by_frame("Model output")

        return optimal_q, residuals

    @requires_initialization
    def forward_kinematics(self, q: np.ndarray = None) -> dict[str, RotoTransMatrixTimeSeries]:
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

            segment_rt_in_global[segment_name] = RotoTransMatrixTimeSeries(nb_frames)
            for i_frame in range(nb_frames):
                segment_rt = self.segments[segment_name].segment_coordinate_system.scs
                parent_name = self.segments[segment_name].parent_name
                if parent_name == "base":
                    parent_rt = RotoTransMatrix()
                else:
                    parent_rt = segment_rt_in_global[parent_name][i_frame]

                if self.segments[segment_name].nb_q == 0:
                    segment_rt_in_global[segment_name][i_frame] = parent_rt @ segment_rt
                else:
                    local_q = q[self.dof_indices(segment_name), i_frame]
                    rt_caused_by_q = self.segments[segment_name].rt_from_local_q(local_q)
                    segment_rt_in_global[segment_name][i_frame] = parent_rt @ segment_rt @ rt_caused_by_q

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
                        point_in_local=marker.position, jcs_in_global=jcs_in_global[segment.name][i_frame]
                    )
                    marker_positions[:, i_marker, i_frame] = marker_in_global.reshape(
                        -1,
                    )
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
                        point_in_local=contact.position, jcs_in_global=jcs_in_global[segment.name][i_frame]
                    )
                    contact_positions[:, i_contact, i_frame] = contact_in_global.reshape(
                        -1,
                    )
                    i_contact += 1

        return contact_positions

    @requires_initialization
    def segment_com_in_global(self, segment_name: str, q: np.ndarray = None) -> np.ndarray:
        q = np.zeros((self.nb_q, 1)) if q is None else q
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        elif len(q.shape) > 2:
            raise RuntimeError("q must be of shape (nb_q, ) or (nb_q, nb_frames).")

        nb_frames = q.shape[1]

        if self.segments[segment_name].inertia_parameters is None:
            return None
        else:
            com_position = np.ones((4, nb_frames))
            jcs_in_global = self.forward_kinematics(q)
            for i_frame in range(nb_frames):
                segment_com_in_global = point_from_local_to_global(
                    point_in_local=self.segments[segment_name].inertia_parameters.center_of_mass,
                    jcs_in_global=jcs_in_global[segment_name][i_frame],
                )
                com_position[:, i_frame] = segment_com_in_global.reshape(
                    -1,
                )

        return com_position

    @requires_initialization
    def muscle_origin_in_global(self, muscle_name: str, q: np.ndarray = None) -> np.ndarray:
        q = np.zeros((self.nb_q, 1)) if q is None else q
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        elif len(q.shape) > 2:
            raise RuntimeError("q must be of shape (nb_q, ) or (nb_q, nb_frames).")

        nb_frames = q.shape[1]

        origin_position = np.ones((4, nb_frames))
        jcs_in_global = self.forward_kinematics(q)
        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:
                if muscle.name == muscle_name:
                    for i_frame in range(nb_frames):
                        origin_position[:, i_frame] = point_from_local_to_global(
                            point_in_local=muscle.origin_position.position,
                            jcs_in_global=jcs_in_global[muscle_group.origin_parent_name][i_frame],
                        ).reshape(
                            -1,
                        )

        return origin_position

    @requires_initialization
    def muscle_insertion_in_global(self, muscle_name: str, q: np.ndarray = None) -> np.ndarray:
        q = np.zeros((self.nb_q, 1)) if q is None else q
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        elif len(q.shape) > 2:
            raise RuntimeError("q must be of shape (nb_q, ) or (nb_q, nb_frames).")

        nb_frames = q.shape[1]

        insertion_position = np.ones((4, nb_frames))
        jcs_in_global = self.forward_kinematics(q)
        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:
                if muscle.name == muscle_name:
                    for i_frame in range(nb_frames):
                        insertion_position[:, i_frame] = point_from_local_to_global(
                            point_in_local=muscle.insertion_position.position,
                            jcs_in_global=jcs_in_global[muscle_group.insertion_parent_name][i_frame],
                        ).reshape(
                            -1,
                        )

        return insertion_position

    @requires_initialization
    def via_points_in_global(self, muscle_name: str, q: np.ndarray = None) -> np.ndarray:
        q = np.zeros((self.nb_q, 1)) if q is None else q
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        elif len(q.shape) > 2:
            raise RuntimeError("q must be of shape (nb_q, ) or (nb_q, nb_frames).")

        nb_frames = q.shape[1]

        via_points_position = np.ones((4, 0, nb_frames))
        jcs_in_global = self.forward_kinematics(q)
        for muscle_group in self.muscle_groups:
            for muscle in muscle_group.muscles:
                if muscle.name == muscle_name:
                    for via_point in muscle.via_points:
                        this_via_point = np.ones((4, nb_frames))
                        for i_frame in range(nb_frames):
                            this_via_point[:, i_frame] = point_from_local_to_global(
                                point_in_local=via_point.position,
                                jcs_in_global=jcs_in_global[via_point.parent_name][i_frame],
                            ).reshape(
                                -1,
                            )
                        via_points_position = np.concatenate(
                            (via_points_position, this_via_point[:, np.newaxis, :]), axis=1
                        )

        return via_points_position

    @requires_initialization
    def total_com_in_global(self, q: np.ndarray = None) -> np.ndarray:
        q = np.zeros((self.nb_q, 1)) if q is None else q
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        elif len(q.shape) > 2:
            raise RuntimeError("q must be of shape (nb_q, ) or (nb_q, nb_frames).")

        nb_frames = q.shape[1]

        com_position = np.zeros((4, nb_frames))
        for segment_name in self.segments.keys():
            this_segment_com = self.segment_com_in_global(segment_name, q=q)
            if this_segment_com is not None:
                com_position += this_segment_com * self.segments[segment_name].inertia_parameters.mass

        com_position[:3, :] /= self.mass
        com_position[3, :] = 1.0  # Set the homogeneous coordinate to 1

        return com_position

    @requires_initialization
    def markers_jacobian(self, q: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
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
        # TODO: Watch out there i problem with this implementation

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
    def muscle_tendon_length(self, muscle_name: str, q: np.ndarray = None) -> np.ndarray:
        """
        Computes the length of the muscle + tendon unit.
        Please note that the muscle trajectory is computed based on the order of declaration of the via points in the model.
        """
        if q is None:
            q = np.zeros((self.nb_q, 1))
        elif len(q.shape) == 1:
            q = q[:, np.newaxis]
        elif len(q.shape) > 2:
            raise RuntimeError("q must be of shape (nb_q, ) or (nb_q, nb_frames).")

        muscle_origin_parent_name, muscle_insertion_parent_name, muscle_origin, muscle_insertion = (
            None,
            None,
            None,
            None,
        )
        nb_frames = q.shape[1]
        muscle_tendon_length = np.zeros((nb_frames,))
        global_jcs = self.forward_kinematics(q)
        for i_frame in range(nb_frames):
            muscle_found = False
            # Get all the points composing the muscle
            muscle_via_points = []
            for muscle_group in self.muscle_groups:
                for muscle in muscle_group.muscles:
                    if muscle.name == muscle_name:
                        for via_point in muscle.via_points:
                            rt = global_jcs[via_point.parent_name][i_frame]
                            muscle_via_points += [rt @ via_point.position]
                        muscle_origin = muscle.origin_position.position
                        muscle_origin_parent_name = muscle.origin_position.parent_name
                        muscle_insertion = muscle.insertion_position.position
                        muscle_insertion_parent_name = muscle.insertion_position.parent_name
                        muscle_found = True
                        break
                if muscle_found:
                    break

            if (
                muscle_origin_parent_name is None
                or muscle_insertion_parent_name is None
                or muscle_origin is None
                or muscle_insertion is None
            ):
                raise RuntimeError(f"The muscle {muscle_name} was not found in the model.")

            origin_position = global_jcs[muscle_origin_parent_name][i_frame] @ muscle_origin
            insertion_position = global_jcs[muscle_insertion_parent_name][i_frame] @ muscle_insertion

            muscle_trajectory = [origin_position] + muscle_via_points + [insertion_position]
            muscle_norm = 0
            for i_point in range(len(muscle_trajectory) - 1):
                muscle_norm += np.linalg.norm(muscle_trajectory[i_point + 1][:3] - muscle_trajectory[i_point][:3])
            muscle_tendon_length[i_frame] = muscle_norm

        return muscle_tendon_length

    def animate(self, view_as: ViewAs = ViewAs.BIORBD, model_path: str = None):

        if view_as == ViewAs.BIORBD:
            try:
                import pyorerun

                if model_path is None or not model_path.endswith(".bioMod"):
                    model_path = "temporary.bioMod"
                    self.to_biomod(model_path, with_mesh=False)

                animation = pyorerun.LiveModelAnimation(model_path, with_q_charts=True)
                animation.options.set_all_labels(False)
                animation.rerun()
            except ImportError:
                _logger.error("pyorerun is not installed. Cannot animate the model.")

        else:
            raise NotImplementedError(
                f"The viewer {view_as} is not implemented yet. Please use view_as=ViewAs.BIORBD for now."
            )

    # TODO: implement tendons
    # @requires_initialization
    # def tendon_length(self, muscle_name: str) -> np.ndarray:
    #     """
    #     Returns the length of the tendon only.
    #     *WARNING* For now, the tendons are assumed rigid, but the tendon length should be variable and thus computed here.
    #     """
    #     return self.muscles[muscle_name].tendon_slack_length
    #
    # @requires_initialization
    # def muscle_length(self, muscle_name: str, q: np.ndarray = None) -> np.ndarray:
    #     """
    #     Computes the length of the muscle only (without tendon).
    #     Please note that the muscle trajectory is computed based on the order of declaration of the via points in the model.
    #     """
    #     return self.muscle_tendon_length(muscle_name, q) - self.tendon_length(muscle_name)
