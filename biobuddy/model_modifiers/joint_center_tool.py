from copy import deepcopy
import logging
import numpy as np
import itertools

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ..utils.c3d_data import C3dData
from ..utils.linear_algebra import RotoTransMatrix, unit_vector

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

    # def _rt_from_trial(self, original_model: "BiomechanicalModelReal") -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Estimate the rigid transformation matrices rt (4×4×N) that align local marker positions to global marker positions over time.
    #
    #     Parameters
    #     ----------
    #     original_model
    #         The scaled model
    #
    #     Returns
    #     ----------
    #
    #     """
    #     nb_frames = self.c3d_data.all_marker_positions.shape[2]
    #
    #     # Marker positions in the global
    #     parent_markers_global = self.c3d_data.get_position(self.parent_marker_names)
    #     child_markers_global = self.c3d_data.get_position(self.child_marker_names)
    #
    #     # Get the segment RT in static pose to compute the marker position in the local reference frame
    #     parent_jcs_in_global = RotoTransMatrix()
    #     parent_jcs_in_global.rt_matrix = original_model.segment_coordinate_system_in_global(self.parent_name)[:, :, 0]
    #     parent_markers_local = np.einsum('ij,jkf->ikf', parent_jcs_in_global.inverse, parent_markers_global)
    #
    #     child_jcs_in_global = RotoTransMatrix()
    #     child_jcs_in_global.rt_matrix = original_model.segment_coordinate_system_in_global(self.child_name)[:, :, 0]
    #     child_markers_local = np.einsum('ij,jkf->ikf', child_jcs_in_global.inverse, child_markers_global)
    #
    #     # Centroid of local marker set (constant)
    #     parent_local_centroid = np.nanmean(parent_markers_local, axis=(1, 2))
    #     child_local_centroid = np.nanmean(child_markers_local, axis=(1, 2))
    #
    #     # Fixed (same for all frames)
    #     parent_local_centered = np.nanmean(parent_markers_local, axis=2) - parent_local_centroid[:, np.newaxis]
    #     child_local_centered = np.nanmean(child_markers_local, axis=2) - child_local_centroid[:, np.newaxis]
    #
    #     rt_parent = np.zeros((4, 4, nb_frames))
    #     rt_child = np.zeros((4, 4, nb_frames))
    #     for i_frame in range(nb_frames):
    #         # Finding the RT allowing to align the segments' markers
    #         rt_parent[:, :, i_frame] = get_rt_aligning_markers_in_global(
    #             parent_markers_global[:, :, i_frame], parent_local_centered, parent_local_centroid
    #         )
    #         rt_child[:, :, i_frame] = get_rt_aligning_markers_in_global(
    #             child_markers_global[:, :, i_frame], child_local_centered, child_local_centroid
    #         )
    #
    #     return rt_parent, rt_child

    # def get_local_segment(self, conf, Mstat):
    #     nb0 = [seg['n'] for seg in conf['S']]
    #     nb = nb0
    #     n_segments = conf['segments']
    #     max_n = max(nb)
    #     n_frames = Mstat.shape[2]
    #
    #     MS = np.full((4, max_n + 4, n_frames, n_segments), np.nan)
    #     MS[3, :, :, :] = 1  # homogeneous coordinate row
    #     Group = np.zeros((n_segments, max_n + 3))
    #     R_t0 = np.full((3, 3, n_segments), np.nan)
    #
    #     for s in range(n_segments):
    #         a, b, c = conf['num'][s, :3]
    #         d = b - a + 1  # total markers
    #         e = c - a + 1  # technical markers
    #
    #         tp1 = Mstat[:, a:c + 1, :]  # technical markers
    #
    #         # Use frame 0 for static alignment
    #         group_result = four_groups(np.mean(tp1[:, :, 0], axis=1, keepdims=True))
    #         if np.sum(group_result) != 0:
    #             tp3 = use_4_groups(np.mean(tp1[:, :, 0], axis=1), group_result)
    #         elif tp1.shape[1] == 2:
    #             raise NotImplementedError("2-marker segment not yet handled in Python version.")
    #         else:
    #             tp3 = np.eye(3)  # fallback
    #
    #         MS[0:3, 0:d, :, s] = Mstat[:, a:b + 1, :]
    #         Group[s, 0:e + 4 if e == 3 else e + 3] = group_result
    #         conf['S'][s]['group'] = group_result
    #         R_t0[:, :, s] = tp3
    #
    #         print(f"segment {s + 1} {conf['S'][s]['name']} ... {c} markers, from {a} to {b}")
    #
    #     return R_t0, MS

    def _four_groups(self, markers: np.ndarray):
        """
        Find 4 groups of markers to define 2 near-orthogonal axes.

        Parameters
        ----------
        markers : np.ndarray, shape (3, n_markers)
            Marker positions in local segment frame.

        Returns
        -------
        best_combo : np.ndarray
            Indices and NaNs separating groups.
        """

        nb_markers = markers.shape[1]
        if nb_markers == 3:
            return np.array([0, np.nan, 1, np.nan, 0, np.nan, 2])

        best_score = 0
        best_combo = np.zeros((nb_markers + 3,))
        half_combos = list(itertools.combinations(range(nb_markers), nb_markers // 2))

        for i in range(len(half_combos) // 2):
            A = set(half_combos[i])
            B = set(range(nb_markers)) - A
            A = list(A)
            B = list(B)
            for a1 in itertools.combinations(A, len(A) // 2):
                a2 = list(set(A) - set(a1))
                for b1 in itertools.combinations(B, len(B) // 2):
                    b2 = list(set(B) - set(b1))

                    Xa = np.nanmean(markers[:3, a1], axis=1) - np.nanmean(markers[:3, a2], axis=1)
                    Yb = np.nanmean(markers[:3, b1], axis=1) - np.nanmean(markers[:3, b2], axis=1)
                    cp = np.sum(np.cross(Xa, Yb) ** 2)

                    if cp > best_score:
                        best_score = cp
                        best_combo = list(a1) + [np.nan] + list(a2) + [np.nan] + list(b1) + [np.nan] + list(b2)
        return np.array(best_combo)

    def _use_4_groups(self, markers: np.ndarray, groups: np.ndarray):
        """
        Build orthonormal basis using grouped markers.

        Parameters
        ----------
        markers : np.ndarray, shape (3, n_markers)
            Marker positions
        groups : np.ndarray
            Group indices with NaNs separating subgroups

        Returns
        -------
            Local coordinate system (X, Y, Z axes)
        """
        nan_indices = np.where(np.isnan(groups))[0]
        a1 = groups[0 : nan_indices[0]].astype(int)
        a2 = groups[nan_indices[0] + 1 : nan_indices[1]].astype(int)
        b1 = groups[nan_indices[1] + 1 : nan_indices[2]].astype(int)
        b2 = groups[nan_indices[2] + 1 :].astype(int)

        x_axis = np.mean(markers[:3, a1], axis=1) - np.mean(markers[:3, a2], axis=1)
        y_axis = np.mean(markers[:3, b1], axis=1) - np.mean(markers[:3, b2], axis=1)
        z_axis = np.cross(x_axis, y_axis)
        y_axis = np.cross(z_axis, x_axis)

        return np.stack([unit_vector(x_axis), unit_vector(y_axis), unit_vector(z_axis)], axis=1)

    def _optimal_local_rt(self, markers: np.ndarray, static_markers: np.ndarray, rotation_init: np.ndarray):

        def inv_ppvect(x):
            return np.array([x[2, 1], x[0, 2], x[1, 0]])

        def ppvect_mat(x):
            q = x.shape[-1]
            out = np.zeros((3, 3))
            out[0, 1] = -x[2]
            out[0, 2] = x[1]
            out[1, 0] = x[2]
            out[1, 2] = -x[0]
            out[2, 0] = -x[1]
            out[2, 1] = x[0]
            return out

        markers = markers[:3, :, :]
        static_markers = static_markers[:3, :, :]
        nb_markers = markers.shape[1]
        nb_frames = markers.shape[2]

        mean_markers = np.mean(markers, axis=1, keepdims=True)
        mean_static_markers = np.mean(static_markers, axis=1, keepdims=True)

        current_center = markers - mean_markers
        static_center = static_markers - mean_static_markers

        m1 = np.sqrt((1 + np.trace(rotation_init[:3, :3])) / 4)
        M1 = inv_ppvect((rotation_init[:3, :3] - rotation_init[:3, :3].T) / (4 * m1))
        # T0 = np.mean(markers[:, :, 0], axis=1, keepdims=True)

        F = np.zeros((3, 3, nb_frames))
        for i_marker in range(nb_markers):
            static_j = static_center[:3, i_marker, 0]
            for i_frame in range(nb_frames):
                current_jk = current_center[:3, i_marker, i_frame]
                F[:, :, i_frame] += np.outer(current_jk, static_j)

        S = 0.5 * (F + np.transpose(F, (1, 0, 2)))
        W = 0.5 * (F - np.transpose(F, (1, 0, 2)))
        W_vec = np.array([W[2, 1, :], W[0, 2, :], W[1, 0, :]]).reshape(3, 1, nb_frames)

        Q = np.zeros((4, 4, nb_frames))
        I = np.eye(3).reshape(3, 3, 1)
        trace_S = np.trace(S, axis1=0, axis2=1).reshape(1, 1, nb_frames)
        Q[:3, :3, :] = 2 * S - trace_S * I
        Q[:3, 3, :] = W_vec[:, 0, :]
        Q[3, :3, :] = np.transpose(W_vec, (1, 0, 2))
        Q[3, 3, :] = trace_S.flatten()

        Y = np.ones((4, nb_frames))
        YtQY = np.zeros((nb_frames,))
        for i_frame in range(nb_frames):
            YtQY[i_frame] = Y[:, i_frame].T @ Q[:, :, i_frame] @ Y[:, i_frame]

        G = np.zeros((4, nb_frames))
        for i_frame in range(nb_frames):
            G[:, i_frame] = 0.5 * (Q[:, :, i_frame] @ Y[:, i_frame] - YtQY[i_frame] / 4 * Y[:, i_frame])

        Yj, Zj, Gj = Y.copy(), G.copy(), G.copy()
        for _ in range(200):
            cond = np.linalg.norm(Gj - 0, axis=1).flatten() > 1e-15
            if not np.any(cond):
                break

            Yi, Zi, Gi = Yj.copy(), Zj.copy(), Gj.copy()

            ZZi = np.zeros((nb_frames,))
            YYi = np.zeros((nb_frames,))
            YZi = np.zeros((nb_frames,))
            ZiQ = np.zeros((4, nb_frames))
            YiQ = np.zeros((4, nb_frames))
            ZiQZi = np.zeros((nb_frames,))
            dot_ZiQ_Yi = np.zeros((nb_frames,))
            dot_YiQ_Yi = np.zeros((nb_frames,))
            for i_frame in range(nb_frames):
                ZZi[i_frame] = np.dot(Zi[:, i_frame], Zi[:, i_frame])
                YYi[i_frame] = np.dot(Yi[:, i_frame], Yi[:, i_frame])
                YZi[i_frame] = np.dot(Yi[:, i_frame], Zi[:, i_frame])
                ZiQ[:, i_frame] = Zi[:, i_frame] @ Q[:, :, i_frame]
                YiQ[:, i_frame] = Yi[:, i_frame] @ Q[:, :, i_frame]
                ZiQZi[i_frame] = np.dot(ZiQ[:, i_frame], Zi[:, i_frame])
                dot_ZiQ_Yi[i_frame] = np.dot(ZiQ[:, i_frame], Yi[:, i_frame])
                dot_YiQ_Yi[i_frame] = np.dot(YiQ[:, i_frame], Yi[:, i_frame])

            a = YZi * ZiQZi - ZZi * dot_ZiQ_Yi
            b = YYi * ZiQZi - ZZi * dot_YiQ_Yi
            c = YYi * dot_ZiQ_Yi - YZi * dot_YiQ_Yi
            delta = ((YYi * ZZi - YZi**2) * b**2 + (YYi * a - ZZi * c) ** 2) / (YYi * ZZi)
            mu = (-b - np.sqrt(delta)) / (2 * a)
            Yj = Yi + mu * Zi

            Gj = np.zeros((4, nb_frames))
            for i_frame in range(nb_frames):
                yk = Yj[:, i_frame]
                qk = Q[:, :, i_frame]

                yk_norm_sq = np.dot(yk, yk)
                qy = qk @ yk
                yq_y = np.dot(yk, qy)

                Gj[:, i_frame] = 2 / yk_norm_sq * (qy - (yq_y / yk_norm_sq) * yk)

            nu = np.zeros((nb_frames,))
            Zj = np.zeros((4, nb_frames))
            for i_frame in range(nb_frames):
                numerator = np.dot(Gj[:, i_frame], Gj[:, i_frame] - Gi[:, i_frame])
                denominator = np.dot(Gi[:, i_frame], Gi[:, i_frame])
                nu_k = numerator / denominator if denominator != 0 else 0
                nu[i_frame] = nu_k
                Zj[:, i_frame] = Gj[:, i_frame] + nu_k * Zi[:, i_frame]

        # Final pose
        X = np.zeros_like(Yj)
        for i_frame in range(nb_frames):
            norm_yk = np.linalg.norm(Yj[:, i_frame])
            if norm_yk != 0:
                X[:, i_frame] = Yj[:, i_frame] / norm_yk

        md = X[3, :]
        Md = X[:3, :]

        # PP_Md = ppvect_mat(Md)
        # MdT_Md = np.einsum("ijk,ijk->1jk", np.transpose(Md, (1, 0, 2)), Md)

        # tp1 = (md ** 2 - MdT_Md) * np.eye(3)[:, :, None] + \
        #       2 * np.einsum("ijk,jik->ijk", Md, np.transpose(Md, (1, 0, 2))) + \
        #       2 * md * PP_Md

        # tp2 = mean_markers + tp1 @ (-T0)

        # Rd = np.zeros((4, 4, nb_frames))
        # Rd[:3, :3, :] = tp1
        # Rd[:3, 3, :] = tp2.squeeze()
        # Rd[3, 3, :] = 1

        m = np.zeros((nb_frames,))
        M = np.zeros((3, nb_frames))
        for i_frame in range(nb_frames):
            m[i_frame] = md[i_frame] * m1 - np.dot(Md[:, i_frame], M1)
            M[:, i_frame] = md[i_frame] * M1 + m1 * Md[:, i_frame] + np.cross(Md[:, i_frame], M1)

        R = np.zeros((3, 3, nb_frames))
        for i_frame in range(nb_frames):
            M_k = M[:, i_frame]
            MtM = M_k @ M_k.T
            MMt = np.dot(M_k.T, M_k).item()
            R[:, :, i_frame] = (m[i_frame] ** 2 - MMt) * np.eye(3) + 2 * MtM + 2 * m[i_frame] * ppvect_mat(M_k)

        # Fill final RT
        local_rt = np.zeros((4, 4, nb_frames))
        local_rt[:3, :3, :] = R
        local_rt[:3, 3, :] = mean_markers.squeeze()
        local_rt[3, 3, :] = 1

        residual = np.full((nb_frames, nb_markers), np.nan)
        for i_marker in range(nb_markers):
            static_local = rotation_init[:3, :3].T @ (static_markers[:, i_marker] - mean_static_markers.squeeze())
            for i_frame in range(nb_frames):
                current_local = R[:3, :, i_frame].T @ (
                    markers[:, i_marker, i_frame] - np.nanmean(mean_markers, axis=2)[:, 0]
                )
                residual[i_frame, i_marker] = np.linalg.norm(static_local - current_local)

        return local_rt  # , Rd, residual

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
        static_markers = original_model.markers_in_global(np.zeros((original_model.nb_q,)))

        parent_static_markers = static_markers[:, original_model.markers_indices(self.parent_marker_names)]
        child_static_markers = static_markers[:, original_model.markers_indices(self.child_marker_names)]

        # Marker positions in the global
        parent_markers_global = self.c3d_data.get_position(self.parent_marker_names)
        child_markers_global = self.c3d_data.get_position(self.child_marker_names)

        # Marker positions in the local
        parent_jcs_in_global = RotoTransMatrix()
        parent_jcs_in_global.rt_matrix = original_model.segment_coordinate_system_in_global(self.parent_name)[:, :, 0]

        n_markers = parent_markers_global.shape[1]
        n_frames = parent_markers_global.shape[2]
        parent_markers_local = np.zeros((4, n_markers, n_frames))
        for i_frame in range(n_frames):
            for i_marker in range(n_markers):
                parent_markers_local[:, i_marker, i_frame] = parent_jcs_in_global.inverse @ parent_markers_global[:, i_marker, i_frame]
        mean_parent_markers_local = np.nanmean(parent_markers_local, axis=2)

        child_jcs_in_global = RotoTransMatrix()
        child_jcs_in_global.rt_matrix = original_model.segment_coordinate_system_in_global(self.child_name)[:, :, 0]
        n_markers = child_markers_global.shape[1]
        child_markers_local = np.zeros((4, n_markers, n_frames))
        for i_frame in range(n_frames):
            for i_marker in range(n_markers):
                child_markers_local[:, i_marker, i_frame] = child_jcs_in_global.inverse @ child_markers_global[:, i_marker, i_frame]
        mean_child_markers_local = np.nanmean(child_markers_local, axis=2)

        parent_marker_groups = self._four_groups(mean_parent_markers_local)
        child_marker_groups = self._four_groups(mean_child_markers_local)

        # parent_initial_rotation = np.zeros((3, 3, nb_frames))
        # child_initial_rotation = np.zeros((3, 3, nb_frames))
        # for i_frame in range(nb_frames):
        parent_initial_rotation = self._use_4_groups(np.nanmean(parent_markers_local, axis=2), parent_marker_groups)
        child_initial_rotation = self._use_4_groups(np.nanmean(child_markers_local, axis=2), child_marker_groups)

        rt_parent = self._optimal_local_rt(parent_markers_local, parent_static_markers, parent_initial_rotation)
        rt_child = self._optimal_local_rt(child_markers_local, child_static_markers, child_initial_rotation)

        return rt_parent, rt_child

    # def _compute_local_segment_reference_frames(self, model, c3d_data, parent_marker_sets):
    #     """
    #     Compute local segment rotation matrices from a static pose.
    #
    #     Parameters
    #     ----------
    #     model : BiomechanicalModelReal
    #         The scaled biomechanical model with segment info.
    #     c3d_data : C3DDataInterface
    #         Provides access to global marker positions.
    #     parent_marker_sets : dict[str, list[str]]
    #         Dictionary of {segment_name: [marker names]}.
    #
    #     Returns
    #     -------
    #     R_t0_dict : dict[str, np.ndarray]
    #         Local segment reference frames (3x3 rotation matrices).
    #     local_markers_dict : dict[str, np.ndarray]
    #         Local marker positions for each segment (3, n_markers).
    #     """
    #     R_t0_dict = {}
    #     local_markers_dict = {}
    #
    #     for segment_name, marker_names in parent_marker_sets.items():
    #         markers_global = c3d_data.get_static_position(marker_names)  # shape: (3, n_markers)
    #         if markers_global.shape[1] < 3:
    #             raise ValueError(f"Segment '{segment_name}' has fewer than 3 markers.")
    #
    #         segment_cs_global = segment_coordinate_system_in_global(model, segment_name)[:, :, 0]
    #         segment_cs_inv = np.linalg.inv(segment_cs_global)
    #
    #         # Transform global marker positions to local frame using static pose
    #         local_markers_hom = np.vstack((markers_global, np.ones((1, markers_global.shape[1]))))
    #         markers_local = segment_cs_inv @ local_markers_hom
    #         markers_local = markers_local[:3]  # discard homogeneous row
    #
    #         # Estimate local frame orientation
    #         marker_groups = four_groups(markers_local)
    #         R_t0 = use_4_groups(markers_local, marker_groups)
    #
    #         R_t0_dict[segment_name] = R_t0
    #         local_markers_dict[segment_name] = markers_local
    #
    #     return R_t0_dict, local_markers_dict

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
        cor_parent_global = np.zeros((4, parent_rt.shape[2]))
        cor_child_global = np.zeros((4, child_rt.shape[2]))
        for i in range(parent_rt.shape[2]):
            cor_parent_global[:, i] = parent_rt[:, :, i] @ np.hstack((cor_parent_local, 1))
            cor_child_global[:, i] = child_rt[:, :, i] @ np.hstack((cor_child_local, 1))

        residuals = np.linalg.norm(cor_parent_global[:3, :] - cor_child_global[:3, :], axis=0)

        if recursive_outlier_removal:
            threshold = np.mean(residuals) + 1.5 * np.std(residuals)
            valid = residuals < threshold
            if np.sum(valid) < nb_frames:
                return self._score_algorithm(parent_rt[:, :, valid], child_rt[:, :, valid], recursive_outlier_removal)

        # Final output
        cor_mean_global = 0.5 * (np.mean(cor_parent_global[:3, :], axis=1) + np.mean(cor_child_global[:3, :], axis=1))

        _logger.info(
            f"\nThere is a residual distance between the parent's and the child's CoR position of : {np.nanmean(residuals)} +- {np.nanstd(residuals)}"
        )
        return cor_mean_global

    def perform_task(self, original_model: BiomechanicalModelReal, new_model: BiomechanicalModelReal):

        # Reconstruct the trial using the current model to identify the orientation of the segments
        rt_parent, rt_child = self._rt_from_trial(original_model)

        # Apply the algo to identify the joint center
        cor_in_global = np.hstack((self._score_algorithm(rt_parent, rt_child), 1))

        # Replace the model components in the new local reference frame
        parent_jcs_in_global = RotoTransMatrix()
        parent_jcs_in_global.rt_matrix = new_model.segment_coordinate_system_in_global(self.parent_name)

        if (
            new_model.segments[self.child_name].segment_coordinate_system is None
            or new_model.segments[self.child_name].segment_coordinate_system.is_in_global
        ):
            raise RuntimeError(
                "The child segment is not in local reference frame. Please set it to local before using the SCoRE algorithm."
            )
        scs_in_local = deepcopy(new_model.segments[self.child_name].segment_coordinate_system.scs)
        scs_in_local[:, 3, 0] = parent_jcs_in_global.inverse @ cor_in_global

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
        marker_positions = original_model.markers_in_global()
        for i_marker, marker in enumerate(new_model.segments[self.child_name].markers):
            marker.position = parent_jcs_in_global.inverse @ marker_positions[:, i_marker, 0]
        # Contacts
        contact_positions = original_model.contacts_in_global()
        for i_contact, contact in enumerate(new_model.segments[self.child_name].contacts):
            contact.position = parent_jcs_in_global.inverse @ contact_positions[:, i_contact, 0]
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

        # Make sure that the scs ar in lical before starting
        for segment in original_model.segments:
            if segment.segment_coordinate_system.is_in_global:
                segment.segment_coordinate_system = SegmentCoordinateSystemReal(
                    scs=deepcopy(original_model.segment_coordinate_system_in_local(segment.name)),
                    is_scs_local=True,
                )

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
