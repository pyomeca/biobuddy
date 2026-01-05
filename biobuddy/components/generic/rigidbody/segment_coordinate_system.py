from functools import partial
import hashlib
from typing import Callable

import numpy as np

from .axis import Axis
from .marker import Marker
from ...real.biomechanical_model_real import BiomechanicalModelReal
from ...real.rigidbody.axis_real import AxisReal
from ...real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ....utils.marker_data import MarkerData, DictData
from ....utils.linear_algebra import RotoTransMatrixTimeSeries, RotoTransMatrix
from ....model_modifiers.joint_center_tool import Score, Sara


class SegmentCoordinateSystem:
    def __init__(
        self,
        origin: Callable[[MarkerData, "BiomechanicalModelReal"], np.ndarray] | str | Marker,
        first_axis: Axis = Axis(Axis.Name.X),
        second_axis: Axis = Axis(Axis.Name.Y),
        axis_to_keep: AxisReal.Name = Axis.Name.X,
    ):
        """
        Set the SegmentCoordinateSystemReal matrix of the segment. To compute the third axis, a first cross product of
        the first_axis with the second_axis is performed. All the axes are then normalized. Then, either the first or
        second axis (depending on [axis_to_keep]) is recomputed with a cross product to get an
        orthonormal system of axes. The system is finally moved to the origin

        Parameters
        ----------
        origin
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the
            origin of the reference frame.
            If a str is provided, the position of the corresponding marker is used
        first_axis
            The first axis defining the segment_coordinate_system
        second_axis
            The second axis defining the segment_coordinate_system
        axis_to_keep
            The Axis.Name of the axis to keep while recomputing the reference frame. It must be the same as either
            first_axis.name or second_axis.name
        """
        self.origin = origin
        self.first_axis = first_axis
        self.second_axis = second_axis
        self.axis_to_keep = axis_to_keep

    @property
    def origin(self) -> Marker:
        """
        The origin of the segment coordinate system
        """
        return self._origin

    @origin.setter
    def origin(self, value: Marker | str | Callable):
        """
        Setter for the origin of the segment coordinate system
        """
        if isinstance(value, str):
            value = Marker(name=value)
        elif isinstance(value, Marker):
            value = value
        elif callable(value):
            value = Marker(function=value)
        else:
            raise RuntimeError(f"The origin must be a Marker, a str or a Callable, not {type(value)}")
        self._origin = value

    def get_axes(
        self, data: MarkerData, model: BiomechanicalModelReal, parent_scs: RotoTransMatrix
    ) -> tuple[AxisReal, AxisReal, AxisReal.Name]:

        # Find the two adjacent axes and reorder accordingly (assuming right-hand RT)
        if self.first_axis.name == self.second_axis.name:
            raise ValueError("The two axes cannot be the same axis")

        first_axis, second_axis = self.first_axis, self.second_axis
        if self.first_axis.name == AxisReal.Name.X:
            third_axis_name = AxisReal.Name.Y if self.second_axis.name == AxisReal.Name.Z else AxisReal.Name.Z
            if self.second_axis.name == AxisReal.Name.Z:
                first_axis, second_axis = self.second_axis, self.first_axis
        elif self.first_axis.name == AxisReal.Name.Y:
            third_axis_name = AxisReal.Name.Z if self.second_axis.name == AxisReal.Name.X else AxisReal.Name.X
            if self.second_axis.name == AxisReal.Name.X:
                first_axis, second_axis = self.second_axis, self.first_axis
        elif self.first_axis.name == AxisReal.Name.Z:
            third_axis_name = AxisReal.Name.X if self.second_axis.name == AxisReal.Name.Y else AxisReal.Name.Y
            if self.second_axis.name == AxisReal.Name.Y:
                first_axis, second_axis = self.second_axis, self.first_axis
        else:
            raise ValueError("first_axis should be an X, Y or Z axis")

        first_axis = first_axis.to_axis(data, model, parent_scs)
        second_axis = second_axis.to_axis(data, model, parent_scs)

        return first_axis, second_axis, third_axis_name

    def get_axes_vectors(
        self, first_axis: AxisReal, second_axis: AxisReal
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Compute the third axis and recompute one of the previous two
        first_axis_vector = first_axis.axis()[:3, :]
        second_axis_vector = second_axis.axis()[:3, :]
        third_axis_vector = np.cross(first_axis_vector, second_axis_vector, axis=0)
        if self.axis_to_keep == first_axis.name:
            second_axis_vector = np.cross(third_axis_vector, first_axis_vector, axis=0)
        elif self.axis_to_keep == second_axis.name:
            first_axis_vector = np.cross(second_axis_vector, third_axis_vector, axis=0)
        else:
            raise ValueError("Name of axis to keep should be one of the two axes")

        return first_axis_vector, second_axis_vector, third_axis_vector

    def get_scs_from_vectors(
        self,
        first_axis: AxisReal,
        second_axis: AxisReal,
        third_axis_name: AxisReal.Name,
        first_axis_vector: np.ndarray,
        second_axis_vector: np.ndarray,
        third_axis_vector: np.ndarray,
        origin: np.ndarray,
    ) -> RotoTransMatrix:
        # Dispatch the result into a matrix
        n_frames = max(first_axis_vector.shape[1], second_axis_vector.shape[1])
        rt = np.zeros((4, 4, n_frames))
        rt[:3, first_axis.name, :] = first_axis_vector / np.linalg.norm(first_axis_vector, axis=0)
        rt[:3, second_axis.name, :] = second_axis_vector / np.linalg.norm(second_axis_vector, axis=0)
        rt[:3, third_axis_name, :] = third_axis_vector / np.linalg.norm(third_axis_vector, axis=0)
        rt[:3, 3, :] = origin[:3, :]
        rt[3, 3, :] = 1
        all_scs = RotoTransMatrixTimeSeries.from_rt_matrix(rt)
        scs = all_scs.mean_homogenous_matrix()

        return scs

    def to_scs(
        self, data: MarkerData, model: BiomechanicalModelReal, parent_scs: RotoTransMatrix
    ) -> SegmentCoordinateSystemReal:
        """
        This constructs a SegmentCoordinateSystemReal by evaluating the function that defines the marker to get an
        actual position

        Parameters
        ----------
        data
            The data to pick the data from
        model
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        parent_scs
            The segment coordinate system in which the marker is defined. If None, the marker is assumed to be in the global
            coordinate system.
        """
        first_axis, second_axis, third_axis_name = self.get_axes(data, model, parent_scs)
        first_axis_vector, second_axis_vector, third_axis_vector = self.get_axes_vectors(first_axis, second_axis)
        origin = self.origin.to_marker(data, model, parent_scs).position
        scs = self.get_scs_from_vectors(
            first_axis, second_axis, third_axis_name, first_axis_vector, second_axis_vector, third_axis_vector, origin
        )

        return SegmentCoordinateSystemReal(scs=scs, is_scs_local=True)


class SegmentCoordinateSystemUtils:
    @staticmethod
    def rigidify(functional_data: MarkerData, static_data: MarkerData = None) -> RotoTransMatrixTimeSeries:
        """
        Compute the rigid body transformation matrices from a set of markers

        Parameters
        ----------
        functional_data
            The data containing the markers
        static_data
            The static data containing the markers, only the first frame is considered.
            If None is provided, the first frame of data is used as reference. Please note, the resulting rt won't
            correspond to another trial with a different initial pose.

        Returns
        -------
        The rigid body transformation matrices as a RotoTransMatrixTimeSeries (4x4xT)
        """
        # Determine a static
        static_markers = (
            static_data.get_position(functional_data.marker_names)[:, :, 0:1]
            if static_data is not None
            else functional_data.get_position(functional_data.marker_names)[:, :, 0:1]
        )

        reference_pts = static_markers[:3, :, 0]  # 3 x N at frame 0
        reference_centroid = np.mean(reference_pts, axis=1, keepdims=True)
        reference_pts_centered = reference_pts - reference_centroid

        markers = functional_data.all_marker_positions
        rt_matrices = RotoTransMatrixTimeSeries(functional_data.nb_frames)
        for i_frame in range(functional_data.nb_frames):
            pts = markers[:3, :, i_frame]
            centroid: np.ndarray = np.mean(pts, axis=1, keepdims=True)
            pts_centered = pts - centroid

            h = pts_centered @ reference_pts_centered.T
            try:
                u, _, vh = np.linalg.svd(h, full_matrices=False)
            except np.linalg.LinAlgError as e:
                rt_matrices[i_frame] = RotoTransMatrix.from_rt_matrix(np.ndarray((4, 4, 1)) * np.nan)
                continue
            r = vh.T @ u.T

            # Check for reflection (instead of rotation) and correct if needed
            if np.linalg.det(r) < 0:
                vh[-1, :] *= -1
                r = vh.T @ u.T

            t = centroid.flatten()
            rt_matrices[i_frame] = RotoTransMatrix.from_rt_matrix(np.vstack((np.hstack((r, t[:, None])), [0, 0, 0, 1])))

        return rt_matrices

    @staticmethod
    def mean_markers(marker_names: tuple[str, ...] | list[str]) -> Callable:
        """
        Compute the mean position of a set of markers

        Parameters
        ----------
        marker_names
            The names of the markers to compute the mean position from

        Returns
        -------
        A lambda function that can be called during the to_real process
        TODO: Move in MarkerData class
        """

        return lambda m, bio: np.nanmean(m.markers_center_position(marker_names), axis=1)

    @staticmethod
    def score(
        functional_data: MarkerData,
        parent_marker_names: tuple[str, ...] | list[str],
        child_marker_names: tuple[str, ...] | list[str],
        visualize: bool = False,
    ) -> Callable:
        """
        Compute the SCoRE (Symmetrical Center of Rotation Estimation) between two sets of markers

        Parameters
        ----------
        parent_marker_names
            The names of the markers on the parent segment to compute the score point from
        child_marker_names
            The names of the markers on the child segment to compute the score point from
        visualize
            If True, a 3D visualization of the score point computation will be shown. Plotly is required for this.

        Returns
        -------
        A lambda function that can be called during the to_real process
        """

        score_cache = {}  # We only need to perform score once. So we store the result here.

        def collapse(static_markers: MarkerData, _: BiomechanicalModelReal, visualize: bool) -> np.ndarray:
            static_markers_hash = _markers_fingerprint(static_markers)

            is_in_cache = static_markers_hash in score_cache
            if not is_in_cache:

                # Check that the markers are in the static
                for name in parent_marker_names + child_marker_names:
                    if name not in static_markers.marker_names:
                        raise RuntimeError(f"The marker {name} is not present in the static markers.")

                # Rigidify the parent segment at static markers
                parent_static_marker_data = static_markers.get_partial_dict_data(parent_marker_names)
                rt_parent_static = SegmentCoordinateSystemUtils.rigidify(
                    functional_data=parent_static_marker_data,
                )

                # Rigidify functional data
                parent_functional_marker_data = functional_data.get_partial_dict_data(parent_marker_names)
                rt_parent_func = SegmentCoordinateSystemUtils.rigidify(
                    functional_data=parent_functional_marker_data,
                    static_data=static_markers,
                )
                child_functional_marker_data = functional_data.get_partial_dict_data(child_marker_names)
                rt_child_func = SegmentCoordinateSystemUtils.rigidify(
                    functional_data=child_functional_marker_data,
                    static_data=static_markers,
                )

                # Compute the SCoRE point
                _, cor_parent_local, _, _, _ = Score.score_algorithm(rt_parent_func, rt_child_func)
                score_cache[static_markers_hash] = [rt_parent_static, rt_parent_func, rt_child_func, cor_parent_local]

            rt_parent_static = score_cache[static_markers_hash][0]
            cor_in_local = np.hstack((score_cache[static_markers_hash][3], 1))

            # Project the optimal point into the static parent segment
            frame_count_static = len(rt_parent_static)
            cor_static = np.zeros((4, frame_count_static))
            for i_frame in range(frame_count_static):
                cor_static[:, i_frame] = (rt_parent_static[i_frame] @ cor_in_local).reshape(4)

            if visualize and not is_in_cache:  # Do not show twice the same visualization
                # rt_child_static = SegmentCoordinateSystemUtils.rigidify(
                #     _InternalData({name: static_markers[name] for name in child_marker_names})
                # )
                _visualize_score(static_markers, rt_parent_static, rt_child_static, cor_static)

                rt_parent_func = score_cache[static_markers_hash][1]
                rt_child_func = score_cache[static_markers_hash][2]
                frame_count_func = len(rt_parent_func)
                cor_func = np.zeros((4, frame_count_func))
                for i_frame in range(frame_count_func):
                    cor_func[:, i_frame] = (rt_parent_func[i_frame] @ cor_in_local).reshape(4)
                _visualize_score(functional_data, rt_parent_func, rt_child_func, cor_func)

            # Collapse across frames
            return np.nanmean(cor_static, axis=1)

        return partial(collapse, visualize=visualize)

    @staticmethod
    def sara(
        name: int,
        functional_data: MarkerData,
        parent_marker_names: tuple[str, ...] | list[str],
        child_marker_names: tuple[str, ...] | list[str],
        visualize: bool = False,
    ) -> Axis:
        """
        Compute the SARA (Symmetrical Axis of Rotation Approach) between two sets of markers

        Parameters
        ----------
        parent_marker_names
            The names of the markers on the parent segment to compute the SARA axis from
        child_marker_names
            The names of the markers on the child segment to compute the SARA axis from
        visualize
            If True, a 3D visualization of the SARA axis computation will be shown. Plotly is required for this.

        Returns
        -------
        A lambda function that can be called during the to_real process
        """

        sara_cache = {}  # We only need to perform SARA once. So we store the result here.

        def collapse(
            static_markers: MarkerData, _: BiomechanicalModelReal, visualize: bool
        ) -> tuple[np.ndarray, np.ndarray]:
            static_markers_hash = _markers_fingerprint(static_markers)

            is_in_cache = static_markers_hash in sara_cache
            if not is_in_cache:
                # Rigidify the parent segment at static markers
                parent_static_marker_data = static_markers.get_partial_dict_data(parent_marker_names)
                rt_parent_static = SegmentCoordinateSystemUtils.rigidify(
                    functional_data=parent_static_marker_data,
                )

                # Rigidify functional data
                parent_functional_marker_data = functional_data.get_partial_dict_data(parent_marker_names)
                rt_parent_func = SegmentCoordinateSystemUtils.rigidify(
                    functional_data=parent_functional_marker_data,
                    static_data=static_markers,
                )
                child_functional_marker_data = functional_data.get_partial_dict_data(child_marker_names)
                rt_child_func = SegmentCoordinateSystemUtils.rigidify(
                    functional_data=child_functional_marker_data,
                    static_data=static_markers,
                )

                # Compute the SARA axis
                _, aor_parent, _, _, cor_parent, _, _, _ = Sara.sara_algorithm(rt_parent_func, rt_child_func)
                sara_cache[static_markers_hash] = [
                    rt_parent_static,
                    rt_parent_func,
                    rt_child_func,
                    aor_parent,
                    cor_parent,
                ]

            rt_parent_static = sara_cache[static_markers_hash][0]
            aor_parent = sara_cache[static_markers_hash][3]
            cor_parent = sara_cache[static_markers_hash][4]

            # Project the optimal point into the static parent segment
            frame_count_static = len(rt_parent_static)
            end_aor_static = np.ones((4, frame_count_static))
            start_aor_static = np.ones((4, frame_count_static))
            for i_frame in range(frame_count_static):
                end_aor_static[:, i_frame] = (rt_parent_static[i_frame] @ aor_parent).reshape(4)
                start_aor_static[:, i_frame] = (rt_parent_static[i_frame] @ cor_parent).reshape(4)

            if visualize and not is_in_cache:  # Do not show twice the same visualization
                child_static_marker_data = static_markers.get_partial_dict_data(child_marker_names)
                rt_child_static = SegmentCoordinateSystemUtils.rigidify(
                    functional_data=child_static_marker_data,
                )
                _visualize_score(static_markers, rt_parent_static, rt_child_static, [start_aor_static, end_aor_static])

                rt_parent_func = sara_cache[static_markers_hash][1]
                rt_child_func = sara_cache[static_markers_hash][2]
                frame_count_func = len(rt_parent_func)
                end_aor_func = np.zeros((4, frame_count_func))
                start_aor_func = np.zeros((4, frame_count_func))
                for i_frame in range(frame_count_func):
                    end_aor_func[:, i_frame] = (rt_parent_func[i_frame] @ aor_parent).reshape(4)
                    start_aor_func[:, i_frame] = (rt_parent_func[i_frame] @ cor_parent).reshape(4)
                _visualize_score(functional_data, rt_parent_func, rt_child_func, [start_aor_func, end_aor_func])

            # Collapse across frames
            return np.nanmean(start_aor_static, axis=1), np.nanmean(end_aor_static, axis=1)

        return Axis(
            name=name,
            start=lambda x, model: collapse(x, model, visualize=visualize)[0],
            end=lambda x, model: collapse(x, model, visualize=visualize)[1],
        )


def _visualize_score(
    data: MarkerData,
    rt_parent: RotoTransMatrixTimeSeries,
    rt_child: RotoTransMatrixTimeSeries,
    cor_global: np.ndarray | tuple[np.ndarray, np.ndarray],
):
    import plotly.graph_objects as go

    data_np = data.all_marker_positions

    frame_count = len(rt_parent)
    frame_data = []
    frames = []
    scaling = 0.05
    for k in range(frame_count):
        parent_origin = rt_parent[k].translation
        parent_rt_points = rt_parent[k] @ np.array([[scaling, 0, 0], [0, scaling, 0], [0, 0, scaling], [1, 1, 1]])

        child_origin = rt_child[k].translation
        child_rt_points = rt_child[k] @ np.array([[scaling, 0, 0], [0, scaling, 0], [0, 0, scaling], [1, 1, 1]])
        frame_data.append(
            [
                go.Scatter3d(
                    x=[parent_origin[0], parent_rt_points[0, 0]],
                    y=[parent_origin[1], parent_rt_points[1, 0]],
                    z=[parent_origin[2], parent_rt_points[2, 0]],
                    mode="lines",
                    line=dict(color="red", width=5),
                ),
                go.Scatter3d(
                    x=[parent_origin[0], parent_rt_points[0, 1]],
                    y=[parent_origin[1], parent_rt_points[1, 1]],
                    z=[parent_origin[2], parent_rt_points[2, 1]],
                    mode="lines",
                    line=dict(color="green", width=5),
                ),
                go.Scatter3d(
                    x=[parent_origin[0], parent_rt_points[0, 2]],
                    y=[parent_origin[1], parent_rt_points[1, 2]],
                    z=[parent_origin[2], parent_rt_points[2, 2]],
                    mode="lines",
                    line=dict(color="blue", width=5),
                ),
                go.Scatter3d(
                    x=[child_origin[0], child_rt_points[0, 0]],
                    y=[child_origin[1], child_rt_points[1, 0]],
                    z=[child_origin[2], child_rt_points[2, 0]],
                    mode="lines",
                    line=dict(color="red", width=5),
                ),
                go.Scatter3d(
                    x=[child_origin[0], child_rt_points[0, 1]],
                    y=[child_origin[1], child_rt_points[1, 1]],
                    z=[child_origin[2], child_rt_points[2, 1]],
                    mode="lines",
                    line=dict(color="green", width=5),
                ),
                go.Scatter3d(
                    x=[child_origin[0], child_rt_points[0, 2]],
                    y=[child_origin[1], child_rt_points[1, 2]],
                    z=[child_origin[2], child_rt_points[2, 2]],
                    mode="lines",
                    line=dict(color="blue", width=5),
                ),
                go.Scatter3d(
                    x=data_np[:, 0, k],
                    y=data_np[:, 1, k],
                    z=data_np[:, 2, k],
                    mode="markers",
                    marker=dict(size=2, color="blue"),
                ),
            ]
        )
        if isinstance(cor_global, list) or isinstance(cor_global, tuple):
            frame_data[-1] += [
                go.Scatter3d(
                    x=np.array([cor_global[0][0, k], cor_global[1][0, k]]),
                    y=np.array([cor_global[0][1, k], cor_global[1][1, k]]),
                    z=np.array([cor_global[0][2, k], cor_global[1][2, k]]),
                    mode="lines",
                    line=dict(width=5, color="red"),
                )
            ]

        else:
            frame_data[-1] += [
                go.Scatter3d(
                    x=[cor_global[0, k]],
                    y=[cor_global[1, k]],
                    z=[cor_global[2, k]],
                    mode="markers",
                    marker=dict(size=5, color="red"),
                )
            ]
        frames.append(go.Frame(data=frame_data[-1], name=str(k)))

    fig = go.Figure(data=frame_data[0], frames=frames)
    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Frame: "},
            pad={"t": 50},
            steps=[
                dict(
                    method="animate",
                    args=[[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    label=str(k),
                )
                for k in range(len(frames))
            ],
        )
    ]
    fig.update_layout(
        sliders=sliders,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        title="Score Point Visualization",
    )
    fig.show()
    
    return fig


def _markers_fingerprint(markers: MarkerData) -> str:
    h = hashlib.sha256()

    for name in sorted(markers.marker_names):  # order-independent
        arr = markers.get_position([name])

        h.update(name.encode("utf-8"))
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())

    return h.hexdigest()
