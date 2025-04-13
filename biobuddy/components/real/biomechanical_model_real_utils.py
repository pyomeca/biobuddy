import numpy as np

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
            if current_parent_name == "base" or current_parent_name is None:  # @pariterre : is this really hardcoded in biorbd ? I thought it was "root"
                break
            current_segment = model.segments[current_parent_name]
            rt_to_global = current_segment.segment_coordinate_system.scs[:, :, 0] @ rt_to_global

        return get_closest_rotation_matrix(rt_to_global)[:, :, np.newaxis]
