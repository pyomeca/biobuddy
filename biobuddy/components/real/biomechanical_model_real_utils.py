import numpy as np
from scipy import optimize
import biorbd

from ...utils.linear_algebra import RotoTransMatrix, get_closest_rotation_matrix


def segment_coordinate_system_in_local(model: "BiomechanicalModelReal", segment_name: str) -> np.ndarray:
    """
    Transforms a SegmentCoordinateSystemReal expressed in the global reference frame into a SegmentCoordinateSystemReal expressed in the local reference frame.

    Parameters
    ----------
    model
        The model to use
    segment_name
        The name of the segment whose SegmentCoordinateSystemReal should be expressed in the local

    Returns
    -------
    The SegmentCoordinateSystemReal in local reference frame
    """

    if segment_name == "base":
        return np.eye(4)
    elif model.segments[segment_name].segment_coordinate_system.is_in_local:
        return model.segments[segment_name].segment_coordinate_system.scs[:, :, 0]
    else:

        parent_name = model.segments[segment_name].parent_name
        parent_scs = RotoTransMatrix()
        parent_scs.from_rt_matrix(segment_coordinate_system_in_global(model=model, segment_name=parent_name))
        inv_parent_scs = parent_scs.inverse
        scs_in_local = inv_parent_scs @ model.segments[segment_name].segment_coordinate_system.scs[:, :, 0]
        return get_closest_rotation_matrix(scs_in_local)[:, :, np.newaxis]


def segment_coordinate_system_in_global(model: "BiomechanicalModelReal", segment_name: str) -> np.ndarray:
    """
    Transforms a SegmentCoordinateSystemReal expressed in the local reference frame into a SegmentCoordinateSystemReal expressed in the global reference frame.

    Parameters
    ----------
    model
        The model to use
    segment_name
        The name of the segment whose SegmentCoordinateSystemReal should be expressed in the global

    Returns
    -------
    The SegmentCoordinateSystemReal in global reference frame
    """

    if segment_name == "base":
        return np.eye(4)
    elif model.segments[segment_name].segment_coordinate_system.is_in_global:
        return model.segments[segment_name].segment_coordinate_system.scs[:, :, 0]

    else:

        current_segment = model.segments[segment_name]
        rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0]
        while current_segment.segment_coordinate_system.is_in_local:
            current_parent_name = current_segment.parent_name
            if (
                current_parent_name == "base" or current_parent_name is None
            ):  # @pariterre : is this really hardcoded in biorbd ? I thought it was "root"
                break
            current_segment = model.segments[current_parent_name]
            rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0] @ rt_to_global

        return get_closest_rotation_matrix(rt_to_global)[:, :, np.newaxis]


def _marker_residual(model_biorbd: biorbd.Model,
                    q_regularization_weight: float,
                    q_target: np.ndarray,
                    q: np.ndarray,
                    experimental_markers: np.ndarray) -> np.ndarray:
    markers_model = np.array(model_biorbd.markers(q))
    nb_marker = experimental_markers.shape[1]
    vect_pos_markers = np.zeros(3 * nb_marker)
    for m, value in enumerate(markers_model):
        vect_pos_markers[m * 3 : (m + 1) * 3] = value.to_array()
    # TODO: setup the IKTask to set the "q_ref" to something else than zero.
    out = np.hstack(
        (vect_pos_markers - np.reshape(experimental_markers.T, (3 * nb_marker,)), q_regularization_weight * (q - q_target))
    )
    return out

def _marker_jacobian(model_biorbd: biorbd.Model, q_regularization_weight: float, q: np.ndarray) -> np.ndarray:
    nb_q = q.shape[0]
    jacobian_matrix = np.array(model_biorbd.markersJacobian(q))
    nb_marker = jacobian_matrix.shape[0]
    vec_jacobian = np.zeros((3 * nb_marker + nb_q, nb_q))
    for m, value in enumerate(jacobian_matrix):
        vec_jacobian[m * 3 : (m + 1) * 3, :] = value.to_array()
    for i_q in range(nb_q):
        vec_jacobian[nb_marker * 3 + i_q, i_q] = q_regularization_weight
    return vec_jacobian


def inverse_kinematics(
        model_biorbd: biorbd.Model,
        marker_positions: np.ndarray,
       marker_names: list[str],
       q_regularization_weight: float = None,
       q_target: np.ndarray = None) -> np.ndarray:
    """
    Solve the inverse kinematics problem using least squares optimization.
    The objective is to match the experimental marker positions with the model marker positions.
    There is also a regularization term matching a predefined posture q_target weighted using q_regularization_weight.
    By default, the q_target is zero, and there is no weight on the regularization term.

    Parameters
    ----------
    model_biorbd
        The biorbd model to use for the inverse kinematic reconstruction.
    marker_positions
        The experimental marker positions
    marker_names
        The names of the experimental markers (the names must match the marker names in the biorbd_model).
    q_regularization_weight
        The weight of the regularization term. If None, no regularization is applied.
    q_target
        The target posture to match. If None, the target posture is set to zero.
    """

    marker_indices = [marker_names.index(m.to_string()) for m in model_biorbd.markerNames()]
    markers_real = marker_positions[:, marker_indices, :]
    nb_frames = marker_positions.shape[2]
    nb_q = model_biorbd.nbQ()

    init = np.ones((nb_q,)) * 0.0001
    if q_target is not None:
        init[:] = q_target
    else:
        q_target = np.zeros((nb_q,))

    if q_regularization_weight is None:
        q_regularization_weight = 0.0

    optimal_q = np.zeros((nb_q, nb_frames))
    for f in range(nb_frames):
        sol = optimize.least_squares(
            fun=lambda q: _marker_residual(model_biorbd, q_regularization_weight, q_target, q, markers_real[:, :, f]),
            jac=lambda q: _marker_jacobian(model_biorbd, q_regularization_weight, q),
            x0=init,
            method="lm",
            xtol=1e-6,
            tr_options=dict(disp=False),
        )
        optimal_q[:, f] = sol.x

    return optimal_q

def point_from_global_to_local(point_in_global, jcs_in_global):
    rt_matrix = RotoTransMatrix()
    rt_matrix.from_rt_matrix(jcs_in_global)
    return rt_matrix.inverse @ np.hstack((point_in_global, 1))

def point_from_local_to_global(point_in_local, jcs_in_global):
    rt_matrix = RotoTransMatrix()
    rt_matrix.from_rt_matrix(jcs_in_global)
    return rt_matrix @ np.hstack((point_in_local, 1))

def markers_in_global(model: "BiomechanicalModelReal", q: np.ndarray = None) -> np.ndarray:

    if q is not None:
        # nb_frames = q.shape[2]
        raise NotImplementedError("The function markers_in_global is not implemented yet for q, please implement it ;)")

    i_marker = 0
    marker_positions = np.ones((4, model.nb_markers))
    for i_segment, segment in enumerate(model.segments):
        jcs_in_global = segment_coordinate_system_in_global(model=model, segment_name=segment.name)
        for marker in segment.markers:
            marker_in_local = point_from_local_to_global(point_in_local=marker.position, jcs_in_global=jcs_in_global)
            marker_positions[:, i_marker] = marker_in_local
            i_marker += 1

    return marker_positions


def contacts_in_global(model: "BiomechanicalModelReal", q: np.ndarray = None) -> np.ndarray:
    if q is not None:
        # nb_frames = q.shape[2]
        raise NotImplementedError("The function contacts_in_global is not implemented yet for q, please implement it ;)")

    i_contact = 0
    contact_positions = np.ones((4, model.nb_contacts))
    for i_segment, segment in enumerate(model.segments):
        jcs_in_global = segment_coordinate_system_in_global(model=model, segment_name=segment.name)
        for contact in segment.contacts:
            contact_in_local = point_from_local_to_global(point_in_local=contact.position, jcs_in_global=jcs_in_global)
            contact_positions[:, i_contact] = contact_in_local
            i_contact += 1

    return contact_positions
