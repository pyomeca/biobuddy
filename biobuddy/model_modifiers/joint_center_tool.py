from copy import deepcopy
import logging
import numpy as np
import itertools
from scipy import optimize

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ..utils.c3d_data import C3dData
from ..utils.linear_algebra import RotoTransMatrix, unit_vector, quaternion_to_rotation_matrix, get_closest_rt_matrix

_logger = logging.getLogger(__name__)


class Score:
    def __init__(
        self,
        filepath: str,
        parent_name: str,
        child_name: str,
        parent_marker_names: list[str],
        child_marker_names: list[str],
        first_frame: int,
        last_frame: int,
        method: str = "numerical",
    ):
        """
        Initializes the Score class which will find the position of the joint center using functional movements.
        The SCoRE algorithm considers that both segments are rigid bodies and that the joint center is located at the
        intersection of the two segments.
        TODO: Add algo ref link.

        Parameters
        ----------
        filepath
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
        method: "numerical" or "optimization"
            If the segments' rt should be estimated using constrained optimization or linear algebra
        """

        # Original attributes
        self.filepath = filepath
        self.parent_name = parent_name
        self.child_name = child_name
        self.parent_marker_names = parent_marker_names
        self.child_marker_names = child_marker_names
        self.method = method

        # Extended attributes
        self.parent_static_markers_in_global  = None
        self.child_static_markers_in_global  = None
        self.parent_static_markers_in_local  = None
        self.child_static_markers_in_local  = None
        self.parent_markers_global = None
        self.child_markers_global = None

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

        # Check file format
        if filepath.endswith(".c3d"):
            self.c3d_data = C3dData(filepath, first_frame, last_frame)
        else:
            if filepath.endswith(".trc"):
                raise NotImplementedError(".trc files cannot be read yet.")
            else:
                raise RuntimeError("The filepath (static trial) must be a .c3d file in a static posture.")

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

    def _optimal_rt(
        self, markers: np.ndarray, static_markers_in_global: np.ndarray, rotation_init: np.ndarray, marker_names: list[str]
    ):

        def inv_ppvect(x):
            return np.array([x[2, 1], x[0, 2], x[1, 0]])

        def ppvect_mat(x):
            out = np.zeros((3, 3))
            out[0, 1] = -x[2]
            out[0, 2] = x[1]
            out[1, 0] = x[2]
            out[1, 2] = -x[0]
            out[2, 0] = -x[1]
            out[2, 1] = x[0]
            return out

        markers = markers[:3, :, :]
        static_markers_in_global = static_markers_in_global[:3, :, :]
        nb_markers, nb_frames, static_centered = self._check_optimal_rt_inputs(markers, static_markers_in_global, marker_names)

        mean_markers = np.mean(np.nanmean(markers, axis=1), axis=1)
        functional_centered = markers - mean_markers[:, np.newaxis, np.newaxis]

        static_quaternion_scalar = np.sqrt((1 + np.trace(rotation_init[:3, :3])) / 4)
        static_quaternion_vector = inv_ppvect((rotation_init[:3, :3] - rotation_init[:3, :3].T) / 4 / static_quaternion_scalar)

        F = np.zeros((3, 3, nb_frames))
        for i_marker, marker_name in enumerate(marker_names):
            current_static_marker_centered = static_centered[:3, i_marker, 0]
            for i_frame in range(nb_frames):
                current_functional_marker_centered = functional_centered[:3, i_marker, i_frame]
                F[:, :, i_frame] += np.dot(current_functional_marker_centered, current_static_marker_centered)

        S = 0.5 * (F + np.transpose(F, (1, 0, 2)))
        W = (F - np.transpose(F, (1, 0, 2)))
        W_vec = np.array([W[2, 1, :], W[0, 2, :], W[1, 0, :]])
        Q = np.zeros((4, 4, nb_frames))
        for i_frame in range(nb_frames):
            trace_S = np.trace(S[:, :, i_frame])
            Q[:3, :3, i_frame] = 2 * S[:, :, i_frame] - trace_S * np.identity(3)
            Q[:3, 3, i_frame] = W_vec[:, i_frame]
            Q[3, :3, i_frame] = np.transpose(W_vec[:, i_frame])
            Q[3, 3, i_frame] = trace_S

        Y = np.ones((4, nb_frames))
        G = np.zeros((4, nb_frames))
        for i_frame in range(nb_frames):
            G[:, i_frame] = 0.5 * (np.dot(Q[:, :, i_frame], Y[:, i_frame]) - np.dot(np.dot(np.dot(Y[:, i_frame].T, Q[:, :, i_frame]), Y[:, i_frame]) / 4, Y[:, i_frame]))

        Yj, Zj, Gj = Y.copy(), G.copy(), G.copy()
        for _ in range(200):
            cond = np.linalg.norm(Gj - 0, axis=1).flatten() > 1e-10
            if not np.any(cond):
                break

            Yi, Zi, Gi = Yj.copy(), Zj.copy(), Gj.copy()

            for i_frame in range(nb_frames):
                ZZi = np.dot(Zi[:, i_frame].T, Zi[:, i_frame])
                YYi = np.dot(Yi[:, i_frame].T, Yi[:, i_frame])
                YZi = np.dot(Yi[:, i_frame].T, Zi[:, i_frame])
                ZiQ = np.dot(Zi[:, i_frame].T, Q[:, :, i_frame])
                YiQ = np.dot(Yi[:, i_frame].T, Q[:, :, i_frame])
                ZiQZi = np.dot(ZiQ, Zi[:, i_frame])
                dot_ZiQ_Yi = np.dot(ZiQ, Yi[:, i_frame])
                dot_YiQ_Yi = np.dot(YiQ, Yi[:, i_frame])

                a = np.dot(YZi, ZiQZi) - np.dot(ZZi, dot_ZiQ_Yi)
                b = np.dot(YYi, ZiQZi) - np.dot(ZZi, dot_YiQ_Yi)
                c = np.dot(YYi, dot_ZiQ_Yi) - np.dot(YZi, dot_YiQ_Yi)

                delta = ((np.dot(YYi, ZZi) - YZi**2) * b**2 + (YYi * a - ZZi * c) ** 2) / (YYi * ZZi)
                mu = (-b - np.sqrt(delta)) / (2 * a)
                Yj[:, i_frame] = Yi[:, i_frame] + mu * Zi[:, i_frame]
                Gj[:, i_frame] = np.dot(2 / np.dot(Yj[:, i_frame].T, Yj[:, i_frame]), np.dot(Q[:, :, i_frame], Yj[:, i_frame]) - np.dot(np.dot(np.dot(Yj[:, i_frame].T, Q[:, :, i_frame]), Yj[:, i_frame]) / np.dot(Yj[:, i_frame].T, Yj[:, i_frame]), Yj[:, i_frame]))

                numerator = np.dot(Gj[:, i_frame], Gj[:, i_frame] - Gi[:, i_frame])
                denominator = np.dot(Gi[:, i_frame], Gi[:, i_frame])
                nu = numerator / denominator if denominator != 0 else 0
                Zj[:, i_frame] = Gj[:, i_frame] + nu * Zi[:, i_frame]

        # Final pose
        X = np.zeros_like(Yj)
        for i_frame in range(nb_frames):
            norm_yk = np.sqrt(np.dot(Yj[:, i_frame].T, Yj[:, i_frame]))
            if norm_yk != 0:
                X[:, i_frame] = Yj[:, i_frame] / norm_yk

        functional_quaternion_scalar = X[3, :]
        functional_quaternion_vector = X[:3, :]

        quaternion_real_scalar = np.zeros((nb_frames,))
        quaternion_vector = np.zeros((3, nb_frames))
        rotation = np.zeros((3, 3, nb_frames))
        for i_frame in range(nb_frames):

            # Compute the quaternion
            quaternion_real_scalar[i_frame] = functional_quaternion_scalar[i_frame] * static_quaternion_scalar - np.dot(functional_quaternion_vector[:, i_frame].T, static_quaternion_vector)
            quaternion_vector[:, i_frame] = functional_quaternion_scalar[i_frame] * static_quaternion_vector + static_quaternion_scalar * functional_quaternion_vector[:, i_frame] + np.dot(ppvect_mat(functional_quaternion_vector[:, i_frame]), static_quaternion_vector)

            # Renormalization of the quaternion to make sure it lies in SO(3)
            quaternion_norm = np.linalg.norm(np.hstack((quaternion_real_scalar[i_frame], quaternion_vector[:, i_frame])))
            if np.abs(1 - quaternion_norm) > 1e-3:
                raise RuntimeError("The quaternion norm is not close to 1.")
            quaternion_real_scalar[i_frame] /= quaternion_norm
            quaternion_vector[:, i_frame] /= quaternion_norm

            # Transforming into a rotation matrix
            rotation[:, :, i_frame] = quaternion_to_rotation_matrix(quaternion_real_scalar[i_frame], quaternion_vector[:, i_frame])

        # Fill final RT
        optimal_rt = np.zeros((4, 4, nb_frames))
        optimal_rt[:3, :3, :] = rotation
        optimal_rt[:3, 3, :] = mean_markers[:, np.newaxis]
        optimal_rt[3, 3, :] = 1

        residual = np.full((nb_frames, nb_markers), np.nan)
        for i_marker in range(nb_markers):
            static_local = rotation_init[:3, :3].T @ (static_markers_in_global[:, i_marker] - mean_static_markers_in_global.squeeze())
            for i_frame in range(nb_frames):
                current_local = rotation[:3, :, i_frame].T @ (
                    markers[:, i_marker, i_frame] - mean_markers
                )
                residual[i_frame, i_marker] = np.linalg.norm(static_local - current_local)

        return optimal_rt


    def _check_optimal_rt_inputs(self, markers: np.ndarray, static_markers: np.ndarray, marker_names: list[str]) -> tuple[int, int, np.ndarray]:

        nb_markers = markers.shape[1]
        nb_frames = markers.shape[2]

        if len(marker_names) != nb_markers:
            raise RuntimeError(f"The marker_names {marker_names} do not match the number of markers {nb_markers}.")

        mean_static_markers = np.mean(static_markers, axis=1, keepdims=True)
        static_centered = static_markers - mean_static_markers

        functional_mean_markers_each_frame = np.nanmean(markers, axis=1)
        for i_marker, marker_name in enumerate(marker_names):
            for i_frame in range(nb_frames):
                current_functional_marker_centered = markers[:, i_marker, i_frame] - functional_mean_markers_each_frame[:,
                                                                                     i_frame]
                if (
                        np.abs(
                            np.linalg.norm(static_centered[:, i_marker])
                            - np.linalg.norm(current_functional_marker_centered)
                        )
                        > 0.05
                ):
                    raise RuntimeError(
                        f"The marker {marker_name} seem to move during the functional trial."
                        f"The distance between the center and this marker is "
                        f"{np.linalg.norm(static_centered)} during the static trial and "
                        f"{np.linalg.norm(current_functional_marker_centered)} during the functional trial."
                    )
            return nb_markers, nb_frames, static_centered


    def _marker_residual(
            self,
            rt: np.ndarray,
            static_markers_in_local: np.ndarray,
            functional_markers: np.ndarray,
    ) -> float:
        nb_markers = static_markers_in_local.shape[1]
        vect_pos_markers = np.zeros(4 * nb_markers)
        rt_matrix = rt.reshape(4, 4)
        for i_marker in range(nb_markers):
            vect_pos_markers[i_marker * 4 : (i_marker + 1) * 4] = (rt_matrix @ static_markers_in_local[:, i_marker] - functional_markers[:, i_marker]) ** 2
        return np.sum(vect_pos_markers)

    def _rt_constraints(self, rt: np.ndarray) -> np.ndarray:
        rt_matrix = rt.reshape(4, 4)
        g = rt_matrix.T @ rt_matrix - np.identity(4)
        return g.flatten()

    def _scipy_optimal_rt(
        self, markers: np.ndarray, static_markers_in_local: np.ndarray, rotation_init: np.ndarray, marker_names: list[str]
    ):

        nb_markers, nb_frames, _ = self._check_optimal_rt_inputs(markers, static_markers_in_local[:3, :], marker_names)

        optimal_rt = np.zeros((4, 4, nb_frames))
        for i_frame in range(nb_frames):

            init = np.eye(4)
            init[:3, :3] = rotation_init
            init[:, 3] = np.nanmean(markers[:, :, i_frame], axis=1)
            init = init.flatten()

            lbx = np.ones((4, 4)) * -5
            ubx = np.ones((4, 4)) * 5
            lbx[:3, :3] = -1
            ubx[:3, :3] = 1
            lbx[3, :] = [0, 0, 0, 1]
            ubx[3, :] = [0, 0, 0, 1]

            sol = optimize.minimize(
                fun=lambda rt: self._marker_residual(rt, static_markers_in_local, markers[:, :, i_frame],
                ),
                x0=init,
                method="SLSQP",
                constraints={"type": "eq", "fun": lambda rt: self._rt_constraints(rt)},
                bounds=optimize.Bounds(lbx.flatten(), ubx.flatten()),
            )
            optimal_rt[:, :, i_frame] = np.reshape(sol.x, (4, 4))

        return optimal_rt

    def _rt_from_trial(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the rigid transformation matrices rt (4×4×N) that align local marker positions to global marker positions over time.
        """
        parent_marker_groups = self._four_groups(self.parent_static_markers_in_global[:, :, 0])
        child_marker_groups = self._four_groups(self.child_static_markers_in_global[:, :, 0])
        parent_initial_rotation = self._use_4_groups(self.parent_static_markers_in_global[:, :, 0], parent_marker_groups)
        child_initial_rotation = self._use_4_groups(self.child_static_markers_in_global[:, :, 0], child_marker_groups)

        if self.method == "numerical":
            # rt are in the global but are positioned at the center of the markers
            rt_parent = self._optimal_rt(
                self.parent_markers_global,
                self.parent_static_markers_in_global,
                parent_initial_rotation,
                marker_names=self.parent_marker_names,
            )
            rt_child = self._optimal_rt(
                self.child_markers_global,
                self.child_static_markers_in_global,
                child_initial_rotation,
                marker_names=self.child_marker_names,
            )
        elif self.method == "optimization":
            # rt are in the global but are positioned at the old joint center
            rt_parent_functional = self._scipy_optimal_rt(
                self.parent_markers_global,
                self.parent_static_markers_in_local,
                parent_initial_rotation,
                marker_names=self.parent_marker_names,
            )
            rt_child_functional = self._scipy_optimal_rt(
                self.child_markers_global,
                self.child_static_markers_in_local,
                child_initial_rotation,
                marker_names=self.child_marker_names,
            )
            rt_parent_static = self._scipy_optimal_rt(
                self.parent_static_markers_in_global,
                self.parent_static_markers_in_local,
                parent_initial_rotation,
                marker_names=self.parent_marker_names,
            )
            rt_child_static = self._scipy_optimal_rt(
                self.child_static_markers_in_global,
                self.child_static_markers_in_local,
                child_initial_rotation,
                marker_names=self.child_marker_names,
            )
        else:
            raise RuntimeError(f"The method {self.method} is not recognized.")

        return rt_parent_functional, rt_child_functional, rt_parent_static, rt_child_static


    def _score_algorithm(
        self, parent_rt: np.ndarray, child_rt: np.ndarray, recursive_outlier_removal: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return cor_mean_global, cor_parent_local, cor_child_local


    def perform_task(self, original_model: BiomechanicalModelReal, new_model: BiomechanicalModelReal):

        # Reconstruct the trial using the current model to identify the orientation of the segments
        # rt_parent, rt_child = self._rt_from_trial()
        rt_parent_functional, rt_child_functional, rt_parent_static, rt_child_static = self._rt_from_trial()

        cor_in_global, cor_in_parent, cor_in_child = self._score_algorithm(rt_parent_functional, rt_child_functional)
        cor_global_static = 0.5 * (rt_parent_static[:, :, 0] @ np.hstack((cor_in_parent, 1)) + rt_child_static[:, :, 0] @ np.hstack((cor_in_child, 1)))

        # rt_from_functional_to_static
        # Apply the algo to identify the joint center
        # cor_in_parent = np.hstack((self._score_algorithm(rt_parent, rt_child), 1))
        # old_parent_jcs_in_global = new_model.segment_coordinate_system_in_global(self.parent_name)
        # rt_from_functional_to_static = rt_parent[] @
        # if self.method == "numerical":
        #     # The CoR is expressed at the center of the markers during the functional trial
        #     parent_mean_position = np.nanmean(self.parent_markers_global, axis=(1, 2))
        #     global_rt_parent = deepcopy(rt_parent)
        #     global_rt_parent[:3, 3] += parent_mean_position
        #     cor_in_global = global_rt_parent @ np.hstack((cor_in_parent, 1))
        # elif self.method == "optimization":
        #     # The CoR is expressed at the old joint center during the functional trial
        #     cor_in_global = old_parent_jcs_in_global @ np.hstack((cor_in_parent, 1))
        # else:
        #     raise RuntimeError(f"The method {self.method} is not recognized.")

        # cor_in_global = old_parent_jcs_in_global[:, :, 0] @ cor_in_parent

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

        # Segment RT
        reset_axis_rt = RotoTransMatrix()
        reset_axis_rt.rt_matrix = np.eye(4)
        if self.child_name + "_parent_offset" in new_model.segment_names:
            segment_to_move_rt_from = self.child_name + "_parent_offset"
            if self.child_name + "_reset_axis" in new_model.segment_names:
                reset_axis_rt.rt_matrix = deepcopy(new_model.segments[self.child_name + "_reset_axis"].segment_coordinate_system.scs)
        else:
            segment_to_move_rt_from = self.child_name
        scs_in_local = deepcopy(new_model.segments[segment_to_move_rt_from].segment_coordinate_system.scs)
        scs_in_local[:, 3, 0] = parent_jcs_in_global.inverse @ cor_global_static
        new_model.segments[segment_to_move_rt_from].segment_coordinate_system = SegmentCoordinateSystemReal(
            scs=scs_in_local,
            is_scs_local=True,
        )

        # New position of the child jsc after replacing the parent_offset segment
        new_child_jcs_in_global = RotoTransMatrix()
        new_child_jcs_in_global.rt_matrix = new_model.segment_coordinate_system_in_global(self.child_name)

        # Markers
        marker_positions = original_model.markers_in_global()
        for i_marker, marker in enumerate(new_model.segments[self.child_name].markers):
            marker_index = original_model.markers_indices([marker.name])
            marker.position = new_child_jcs_in_global.inverse @ marker_positions[:, marker_index, 0]
        # Contacts
        contact_positions = original_model.contacts_in_global()
        for i_contact, contact in enumerate(new_model.segments[self.child_name].contacts):
            contact_index = original_model.markers_indices([contact.name])
            contact.position = new_child_jcs_in_global.inverse @ contact_positions[:, contact_index, 0]
        # IMUs
        # Muscles origin, insertion, via points


class Sara:
    def __init__(self, filepath: str, parent_name: str, child_name: str, first_frame: int, last_frame: int):
        """
        Initializes the Sara class which will find the position of the joint center using functional movements.
        The algorithm considers that both segments are rigid bodies and that the joint center is located at the
        intersection of the two segments.
        TODO: Add algo ref link.

        Parameters
        ----------
        filepath
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
        self.filepath = filepath
        self.parent_name = parent_name
        self.child_name = child_name

        # Extended attributes
        self.parent_static_markers_in_global = None
        self.child_static_markers_in_global = None
        self.parent_static_markers_in_local = None
        self.child_static_markers_in_local = None
        self.parent_markers_global = None
        self.child_markers_global = None

        # Check file format
        if filepath.endswith(".c3d"):
            # Load the c3d file
            c3d_data = C3dData(filepath, first_frame, last_frame)
            self.marker_names = c3d_data.marker_names
            self.marker_positions = c3d_data.all_marker_positions[:3, :, :]
        else:
            if filepath.endswith(".trc"):
                raise NotImplementedError(".trc files cannot be read yet.")
            else:
                raise RuntimeError("The filepath (static trial) must be a .c3d file in a static posture.")

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

        # Check that there is a functional movement in the trial (aka the markers really move)
        std = []
        for marker_name in jcs_identifier.parent_marker_names + jcs_identifier.child_marker_names:
            std += jcs_identifier.c3d_data.std_marker_position(marker_name)
        if all(np.array(std) < 0.01):
            raise RuntimeError(
                f"The markers {jcs_identifier.parent_marker_names + jcs_identifier.child_marker_names} are not moving in the functional trial (markers std = {std}). "
                f"Please check the trial again."
            )

    def _check_marker_positions(self, task):
        """
        Check that the markers are positioned at the same place on the subject between the static trial and the current functional trial.
        """
        # Parent
        for marker_name_1 in task.parent_marker_names:
            for marker_name_2 in task.parent_marker_names:
                if marker_name_1 != marker_name_2:
                    distance_trial = np.linalg.norm(
                        task.parent_static_markers_in_global[:, task.parent_marker_names.index(marker_name_1), 0]
                        - task.parent_static_markers_in_global[:, task.parent_marker_names.index(marker_name_2), 0]
                    )
                    distance_static = np.linalg.norm(
                        task.parent_markers_global[:, task.parent_marker_names.index(marker_name_1), 0]
                        - task.parent_markers_global[:, task.parent_marker_names.index(marker_name_2), 0]
                    )
                    if np.abs(distance_static - distance_trial) > 0.05:
                        raise RuntimeError(
                            f"There is a difference in marker placement of more than 1cm between the static trial and the functional trial for markers {marker_name_1} and {marker_name_2}. Please make sure that the markers do not move on the subjects segments."
                        )
        # Child
        for marker_name_1 in task.child_marker_names:
            for marker_name_2 in task.child_marker_names:
                if marker_name_1 != marker_name_2:
                    distance_trial = np.linalg.norm(
                        task.child_static_markers_in_global[:3, task.child_marker_names.index(marker_name_1), 0]
                        - task.child_static_markers_in_global[:3, task.child_marker_names.index(marker_name_2), 0]
                    )
                    distance_static = np.linalg.norm(
                        task.child_markers_global[:3, task.child_marker_names.index(marker_name_1), 0]
                        - task.child_markers_global[:3, task.child_marker_names.index(marker_name_2), 0]
                    )
                    if np.abs(distance_static - distance_trial) > 0.05:
                        raise RuntimeError(
                            f"There is a difference in marker placement of more than 1cm between the static trial and the functional trial for markers {marker_name_1} and {marker_name_2}. Please make sure that the markers do not move on the subjects segments."
                        )


    def replace_joint_centers(self) -> BiomechanicalModelReal:

        static_markers_in_global = self.original_model.markers_in_global(np.zeros((self.original_model.nb_q,)))
        for task in self.joint_center_tasks:

            # Marker positions in the global from the static trial
            task.parent_static_markers_in_global = static_markers_in_global[
                :, self.original_model.markers_indices(task.parent_marker_names)
            ]
            task.child_static_markers_in_global = static_markers_in_global[:, self.original_model.markers_indices(task.child_marker_names)]

            # Marker positions in the local from the static trial
            task.parent_static_markers_in_local = np.zeros((4, len(task.parent_marker_names)))
            for i_marker, marker_name in enumerate(task.parent_marker_names):
                task.parent_static_markers_in_local[:, i_marker] = self.original_model.segments[task.parent_name].markers[marker_name].position[:, 0]
            task.child_static_markers_in_local = np.zeros((4, len(task.child_marker_names)))
            for i_marker, marker_name in enumerate(task.child_marker_names):
                task.child_static_markers_in_local[:, i_marker] = \
                self.original_model.segments[task.child_name].markers[marker_name].position[:, 0]

            # Marker positions in the global from this functional trial
            task.parent_markers_global = task.c3d_data.get_position(task.parent_marker_names)
            task.child_markers_global = task.c3d_data.get_position(task.child_marker_names)

            # Replace the joint center in the new model
            self._check_marker_positions(task)
            task.perform_task(self.original_model, self.new_model)

        self.new_model.segments_rt_to_local()
        return self.new_model
