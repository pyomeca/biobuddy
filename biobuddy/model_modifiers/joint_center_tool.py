import biorbd
from copy import deepcopy
import logging
import numpy as np
import itertools
from scipy import optimize

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.segment_real import SegmentReal
from ..components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ..utils.translations import Translations
from ..utils.rotations import Rotations
from ..utils.c3d_data import C3dData
from ..utils.linear_algebra import (
    RotoTransMatrix,
    unit_vector,
    quaternion_to_rotation_matrix,
    get_closest_rt_matrix,
    mean_unit_vector,
    RotoTransMatrixTimeSeries,
    point_from_local_to_global,
)

_logger = logging.getLogger(__name__)


class RigidSegmentIdentification:
    def __init__(
        self,
        filepath: str,
        parent_name: str,
        child_name: str,
        parent_marker_names: list[str],
        child_marker_names: list[str],
        first_frame: int,
        last_frame: int,
        initialize_whole_trial_reconstruction: bool = False,
        animate_rt: bool = False,
    ):
        """
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
        initialize_whole_trial_reconstruction
            If True, the whole trial is reconstructed using whole body inverse kinematics to initialize the segments' rt in the global reference frame.
        animate_rt
            If True, it animates the segment rt reconstruction using pyomeca and pyorerun.
        """

        # Original attributes
        self.filepath = filepath
        self.parent_name = parent_name
        self.child_name = child_name
        self.parent_marker_names = parent_marker_names
        self.child_marker_names = child_marker_names
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.initialize_whole_trial_reconstruction = initialize_whole_trial_reconstruction
        self.animate_rt = animate_rt

        # Extended attributes
        self.parent_static_markers_in_global: np.ndarray = None
        self.child_static_markers_in_global: np.ndarray = None
        self.parent_static_markers_in_local: np.ndarray = None
        self.child_static_markers_in_local: np.ndarray = None
        self.parent_markers_global: np.ndarray = None
        self.child_markers_global: np.ndarray = None
        self.c3d_data: C3dData = None
        self.marker_name: list[str] = None
        self.marker_positions: np.ndarray = None

        self._check_segment_names()
        self._check_c3d_functional_trial_file()

    def _check_segment_names(self):
        illegal_names = ["_parent_offset", "_translation", "_rotation_transform", "_reset_axis"]
        for name in illegal_names:
            if name in self.parent_name:
                raise RuntimeError(
                    f"The names {name} are not allowed in the parent or child names. Please change the segment named {self.parent_name} from the Score configuration."
                )
            if name in self.child_name:
                raise RuntimeError(
                    f"The names {name} are not allowed in the parent or child names. Please change the segment named {self.child_name} from the Score configuration."
                )

    def _check_c3d_functional_trial_file(self):
        """
        Check that the file format is appropriate and that there is a functional movement in the trial (aka the markers really move).
        """
        # Check file format
        if self.filepath.endswith(".c3d"):
            # Load the c3d file
            self.c3d_data = C3dData(self.filepath, self.first_frame, self.last_frame)
            self.marker_names = self.c3d_data.marker_names
            self.marker_positions = self.c3d_data.all_marker_positions[:3, :, :]
        else:
            if self.filepath.endswith(".trc"):
                raise NotImplementedError(".trc files cannot be read yet.")
            else:
                raise RuntimeError("The filepath (static trial) must be a .c3d file in a static posture.")

        # Check that the markers move
        std = []
        for marker_name in self.parent_marker_names + self.child_marker_names:
            std += self.c3d_data.std_marker_position(marker_name)
        if all(np.array(std) < 0.01):
            raise RuntimeError(
                f"The markers {self.parent_marker_names + self.child_marker_names} are not moving in the functional trial (markers std = {std}). "
                f"Please check the trial again."
            )

    def remove_offset_from_optimal_rt(
        self, original_model: BiomechanicalModelReal, rt_parent_functional: np.ndarray, rt_child_functional: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, RotoTransMatrix]:

        if original_model.has_parent_offset(self.parent_name):
            parent_offset_rt = original_model.rt_from_parent_offset_to_real_segment(self.parent_name)
            rt_parent_functional_offsetted = np.zeros_like(rt_parent_functional)
            for i_frame in range(rt_parent_functional.shape[2]):
                rt_parent_functional_offsetted[:, :, i_frame] = (
                    rt_parent_functional[:, :, i_frame] @ parent_offset_rt.inverse
                )
        else:
            rt_parent_functional_offsetted = rt_parent_functional

        if original_model.has_parent_offset(self.child_name):
            child_offset_rt = original_model.rt_from_parent_offset_to_real_segment(self.child_name)
            rt_child_functional_offsetted = np.zeros_like(rt_child_functional)
            for i_frame in range(rt_parent_functional.shape[2]):
                rt_child_functional_offsetted[:, :, i_frame] = (
                    rt_child_functional[:, :, i_frame] @ child_offset_rt.inverse
                )
        else:
            child_offset_rt = RotoTransMatrix()
            child_offset_rt.from_rt_matrix(np.identity(4))
            rt_child_functional_offsetted = rt_child_functional

        return rt_parent_functional_offsetted, rt_child_functional_offsetted, child_offset_rt

    def animate_the_segment_reconstruction(
        self,
        original_model: BiomechanicalModelReal,
        rt_parent: np.ndarray,
        rt_child: np.ndarray,
        without_exp_markers: bool = False,
    ):

        def setup_segments_for_animation(segment_name: str):
            if original_model.has_parent_offset(segment_name):

                segment_list = original_model.get_chain_between_segments(segment_name + "_parent_offset", segment_name)

                # Set rotations and translations to the parent offset
                parent_offset = original_model.segments[segment_list[0]]
                joint_model.add_segment(
                    SegmentReal(
                        name=parent_offset.name,
                        parent_name="ground",
                        segment_coordinate_system=SegmentCoordinateSystemReal(scs=np.identity(4), is_scs_local=True),
                        translations=Translations.XYZ,
                        rotations=Rotations.XYZ,
                        mesh_file=original_model.segments[parent_offset.name].mesh_file,
                    )
                )

                for i_segment, segment_name in enumerate(segment_list[1:]):
                    joint_model.add_segment(
                        SegmentReal(
                            name=segment_name,
                            parent_name=segment_list[i_segment],
                            segment_coordinate_system=deepcopy(
                                original_model.segments[segment_name].segment_coordinate_system
                            ),
                            mesh_file=deepcopy(original_model.segments[segment_name].mesh_file),
                        )
                    )

                    # Modify the markers from the real segment to leave only the functional trial markers
                    for marker in original_model.segments[segment_name].markers:
                        if marker.name in self.parent_marker_names + self.child_marker_names:
                            joint_model.segments[segment_name].add_marker(marker)

            else:
                if original_model.segments[segment_name].parent_name.startswith(segment_name):
                    raise NotImplementedError(
                        "The parent segment does not have a parent offset, but has other ghost segments as parent. This is not implemented yet."
                    )

                joint_model.add_segment(
                    SegmentReal(
                        name=segment_name,
                        parent_name="ground",
                        segment_coordinate_system=SegmentCoordinateSystemReal(scs=np.identity(4), is_scs_local=True),
                        translations=Translations.XYZ,
                        rotations=Rotations.XYZ,
                        mesh_file=original_model.segments[segment_name].mesh_file,
                    )
                )
                for marker in original_model.segments[segment_name].markers:
                    if marker.name in self.parent_marker_names + self.child_marker_names:
                        joint_model.segments[segment_name].add_marker(marker)

        joint_model = BiomechanicalModelReal()
        joint_model.add_segment(
            SegmentReal(
                name="ground",
                segment_coordinate_system=SegmentCoordinateSystemReal(scs=np.identity(4), is_scs_local=True),
            )
        )
        setup_segments_for_animation(self.parent_name)
        setup_segments_for_animation(self.child_name)

        nb_frames = rt_parent.shape[2]
        parent_trans = np.zeros((3, nb_frames))
        parent_rot = np.zeros((3, nb_frames))
        child_trans = np.zeros((3, nb_frames))
        child_rot = np.zeros((3, nb_frames))

        rt_parent_instance = RotoTransMatrixTimeSeries()
        rt_parent_instance.from_rt_matrix(rt_parent)
        rt_child_instance = RotoTransMatrixTimeSeries()
        rt_child_instance.from_rt_matrix(rt_child)

        for i_frame in range(nb_frames):
            parent_trans[:, i_frame] = rt_parent_instance[i_frame].translation
            parent_rot[:, i_frame] = rt_parent_instance[i_frame].euler_angles("xyz")
            child_trans[:, i_frame] = rt_child_instance[i_frame].translation
            child_rot[:, i_frame] = rt_child_instance[i_frame].euler_angles("xyz")

        q = np.vstack((parent_trans, parent_rot, child_trans, child_rot))

        try:
            import pyorerun
            from pyomeca import Markers
        except:
            raise ImportError("Please install pyorerun and pyomeca to visualize the segment reconstruction.")

        # Visualization
        t = np.linspace(0, 1, nb_frames)

        # Add the experimental markers from the static trial
        if not without_exp_markers:
            pyomarkers = Markers(
                data=np.concatenate((self.parent_markers_global, self.child_markers_global), axis=1),
                channels=self.parent_marker_names + self.child_marker_names,
            )

        joint_model.to_biomod("../models/temporary_rt.bioMod")

        viz_biomod_model = pyorerun.BiorbdModel("../models/temporary_rt.bioMod")
        viz_biomod_model.options.transparent_mesh = False
        viz_biomod_model.options.show_gravity = True

        viz = pyorerun.PhaseRerun(t)
        if not without_exp_markers:
            viz.add_animated_model(viz_biomod_model, q, tracked_markers=pyomarkers)
        else:
            viz.add_animated_model(viz_biomod_model, q)
        viz.rerun_by_frame("Segment RT animation")

    def replace_components_in_new_jcs(self, original_model: BiomechanicalModelReal, new_model: BiomechanicalModelReal):
        """
        Ather the SCS has been replaced in the model, the components from this segment must be replaced in the new JCS.
        TODO: Verify that this also works with non orthonormal rotation axes.
        """
        # New position of the child jsc after replacing the parent_offset segment
        new_child_jcs_in_global = RotoTransMatrix()
        new_child_jcs_in_global.from_rt_matrix(new_model.segment_coordinate_system_in_global(self.child_name))

        original_local_scs = RotoTransMatrix()
        original_local_scs.from_rt_matrix(original_model.segment_coordinate_system_in_local(self.child_name))
        new_local_scs = RotoTransMatrix()
        new_local_scs.from_rt_matrix(new_model.segment_coordinate_system_in_local(self.child_name))
        local_scs_transform = RotoTransMatrix()  # The transformation between the old local and the new local jcs
        local_scs_transform.from_rt_matrix(get_closest_rt_matrix(original_local_scs.inverse @ new_local_scs.rt_matrix))

        # Next JCS position
        child_names = original_model.children_segment_names(self.child_name)
        if len(child_names) > 0:
            next_child_name = child_names[0]
            new_model.segments[next_child_name].segment_coordinate_system = SegmentCoordinateSystemReal(
                scs=original_model.segment_coordinate_system_in_global(next_child_name),
                is_scs_local=False,
            )

        if original_model.segments[self.child_name].segment_coordinate_system.is_in_local:
            global_jcs = original_model.segment_coordinate_system_in_global(self.child_name)[:, :, 0]
        else:
            global_jcs = original_model.segments[self.child_name].segment_coordinate_system.scs

        # Meshes  # TODO: verify + test
        if original_model.segments[self.child_name].mesh is not None:
            new_model.segments[self.child_name].mesh.positions = (
                new_child_jcs_in_global.inverse
                @ point_from_local_to_global(original_model.segments[self.child_name].mesh.positions, global_jcs)
            )

        # # Mesh files  # TODO: go up the hierarchy to find the mesh file
        # if original_model.segments[self.child_name].mesh_file is not None:
        #     new_model.segments[self.child_name].mesh_file = None  # skipping this for now
        #     mesh_file = original_model.segments[self.child_name].mesh_file
        #
        #     # Construct transformation from mesh file's local frame to global
        #     rot_mesh_local = compute_matrix_rotation(mesh_file.mesh_rotation[:3, 0])
        #     mesh_local = np.eye(4)
        #     mesh_local[:3, :3] = rot_mesh_local
        #     mesh_local[:3, 3] = mesh_file.mesh_translation[:3, 0]
        #
        #     # Global pose of the mesh
        #     mesh_global = global_jcs @ mesh_local
        #
        #     # Express it in the new local frame
        #     new_rt_global = RotoTransMatrix()
        #     new_rt_global.from_rt_matrix(new_model.segment_coordinate_system_in_global(self.child_name)[:, :, 0])
        #     mesh_new_local = new_rt_global.inverse @ mesh_global
        #
        #     # Update mesh file's local rotation and translation
        #     new_model.segments[self.child_name].mesh_file.mesh_rotation = rot2eul(mesh_new_local[:3, :3])
        #     new_model.segments[self.child_name].mesh_file.mesh_translation = mesh_new_local[:3, 3]

        # Markers
        marker_positions = original_model.markers_in_global()
        for marker in new_model.segments[self.child_name].markers:
            marker_index = original_model.markers_indices([marker.name])
            marker.position = new_child_jcs_in_global.inverse @ marker_positions[:, marker_index, 0]

        # Contacts # TODO: verify + test
        contact_positions = original_model.contacts_in_global()
        for contact in new_model.segments[self.child_name].contacts:
            contact_index = original_model.contact_indices([contact.name])
            contact.position = new_child_jcs_in_global.inverse @ contact_positions[:, contact_index, 0]

        # IMUs # TODO: verify + test
        for imu in new_model.segments[self.child_name].imus:
            imu.scs = local_scs_transform.inverse @ imu.scs

        # Muscles (origin and insertion)
        for muscle_name in new_model.muscle_origin_on_this_segment(self.child_name):
            new_model.muscles[muscle_name].origin_position = (
                new_child_jcs_in_global.inverse
                @ point_from_local_to_global(original_model.muscles[muscle_name].origin_position, global_jcs)
            )
        for muscle_name in new_model.muscle_insertion_on_this_segment(self.child_name):
            new_model.muscles[muscle_name].insertion_position = (
                new_child_jcs_in_global.inverse
                @ point_from_local_to_global(original_model.muscles[muscle_name].insertion_position, global_jcs)
            )

        # Via points
        for via_point_name in new_model.via_points_on_this_segment(self.child_name):
            new_model.via_points[via_point_name].position = (
                new_child_jcs_in_global.inverse
                @ point_from_local_to_global(original_model.via_points[via_point_name].position, global_jcs)
            )

    def check_optimal_rt_inputs(
        self, markers: np.ndarray, static_markers: np.ndarray, marker_names: list[str]
    ) -> tuple[int, int, np.ndarray]:

        nb_markers = markers.shape[1]
        nb_frames = markers.shape[2]

        if len(marker_names) != nb_markers:
            raise RuntimeError(f"The marker_names {marker_names} do not match the number of markers {nb_markers}.")

        mean_static_markers = np.mean(static_markers[:3, :], axis=1, keepdims=True)
        static_centered = static_markers[:3, :] - mean_static_markers

        functional_mean_markers_each_frame = np.nanmean(markers[:3, :, :], axis=1)
        for i_marker, marker_name in enumerate(marker_names):
            for i_frame in range(nb_frames):
                current_functional_marker_centered = (
                    markers[:3, i_marker, i_frame] - functional_mean_markers_each_frame[:, i_frame]
                )
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
                        f"{np.linalg.norm(static_centered[:, i_marker])} during the static trial and "
                        f"{np.linalg.norm(current_functional_marker_centered)} during the functional trial."
                    )
            return nb_markers, nb_frames, static_centered

    def check_marker_positions(self):
        """
        Check that the markers are positioned at the same place on the subject between the static trial and the current functional trial.
        """
        # Parent
        for marker_name_1 in self.parent_marker_names:
            for marker_name_2 in self.parent_marker_names:
                if marker_name_1 != marker_name_2:
                    distance_trial = np.linalg.norm(
                        self.parent_static_markers_in_global[:, self.parent_marker_names.index(marker_name_1), 0]
                        - self.parent_static_markers_in_global[:, self.parent_marker_names.index(marker_name_2), 0]
                    )
                    distance_static = np.linalg.norm(
                        self.parent_markers_global[:, self.parent_marker_names.index(marker_name_1), 0]
                        - self.parent_markers_global[:, self.parent_marker_names.index(marker_name_2), 0]
                    )
                    if np.abs(distance_static - distance_trial) > 0.05:
                        raise RuntimeError(
                            f"There is a difference in marker placement of more than 1cm between the static trial and the functional trial for markers {marker_name_1} and {marker_name_2}. Please make sure that the markers do not move on the subjects segments."
                        )
        # Child
        for marker_name_1 in self.child_marker_names:
            for marker_name_2 in self.child_marker_names:
                if marker_name_1 != marker_name_2:
                    distance_trial = np.linalg.norm(
                        self.child_static_markers_in_global[:3, self.child_marker_names.index(marker_name_1), 0]
                        - self.child_static_markers_in_global[:3, self.child_marker_names.index(marker_name_2), 0]
                    )
                    distance_static = np.linalg.norm(
                        self.child_markers_global[:3, self.child_marker_names.index(marker_name_1), 0]
                        - self.child_markers_global[:3, self.child_marker_names.index(marker_name_2), 0]
                    )
                    if np.abs(distance_static - distance_trial) > 0.05:
                        raise RuntimeError(
                            f"There is a difference in marker placement of more than 1cm between the static trial and the functional trial for markers {marker_name_1} and {marker_name_2}. Please make sure that the markers do not move on the subjects segments."
                        )

    def marker_residual(
        self,
        optimal_rt: np.ndarray,
        static_markers_in_local: np.ndarray,
        functional_markers_in_global: np.ndarray,
    ) -> float:
        nb_markers = static_markers_in_local.shape[1]
        vect_pos_markers = np.zeros(4 * nb_markers)
        rt_matrix = optimal_rt.reshape(4, 4)
        for i_marker in range(nb_markers):
            vect_pos_markers[i_marker * 4 : (i_marker + 1) * 4] = (
                rt_matrix @ static_markers_in_local[:, i_marker] - functional_markers_in_global[:, i_marker]
            ) ** 2
        return np.sum(vect_pos_markers)

    def rt_constraints(self, optimal_rt: np.ndarray) -> np.ndarray:
        rt_matrix = optimal_rt.reshape(4, 4)
        R = rt_matrix[:3, :3]
        c1, c2, c3 = R[:, 0], R[:, 1], R[:, 2]
        constraints = np.array(
            [
                np.dot(c1, c1) - 1,
                np.dot(c2, c2) - 1,
                np.dot(c3, c3) - 1,
                np.dot(c1, c2),
                np.dot(c1, c3),
                np.dot(c2, c3),
            ]
        )
        return constraints

    def scipy_optimal_rt(
        self,
        markers_in_global: np.ndarray,
        static_markers_in_local: np.ndarray,
        rt_init: np.ndarray,
        marker_names: list[str],
    ):

        initialize_whole_trial_reconstruction = False if rt_init.shape[2] == 1 else True
        nb_markers, nb_frames, _ = self.check_optimal_rt_inputs(
            markers_in_global, static_markers_in_local, marker_names
        )

        rt_optimal = np.zeros((4, 4, nb_frames))
        init = rt_init[:, :, 0].reshape(4, 4)  # Initailize with the first frame
        for i_frame in range(nb_frames):
            init = init.flatten()

            lbx = np.ones((4, 4)) * -5
            ubx = np.ones((4, 4)) * 5
            lbx[:3, :3] = -1
            ubx[:3, :3] = 1
            lbx[3, :] = [0, 0, 0, 1]
            ubx[3, :] = [0, 0, 0, 1]

            sol = optimize.minimize(
                fun=lambda rt: self.marker_residual(
                    optimal_rt=rt,
                    static_markers_in_local=static_markers_in_local,
                    functional_markers_in_global=markers_in_global[:, :, i_frame],
                ),
                x0=init,
                method="SLSQP",
                constraints={"type": "eq", "fun": lambda rt: self.rt_constraints(optimal_rt=rt)},
                bounds=optimize.Bounds(lbx.flatten(), ubx.flatten()),
            )
            if sol.success:
                rt_optimal[:, :, i_frame] = np.reshape(sol.x, (4, 4))
                if initialize_whole_trial_reconstruction:
                    # Use the rt from the reconstruction of the whole trial at the current frame
                    frame = i_frame + 1 if i_frame + 1 < nb_frames else i_frame
                    init = rt_init[:, :, frame].reshape(4, 4)
                else:
                    # Use the optimal rt of the previous frame
                    init = rt_optimal[:, :, i_frame]
            else:
                init = np.nan
                print(f"The optimization failed: {sol.message}")

        return rt_optimal

    def rt_from_trial(self, parent_rt_init, child_rt_init) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the rigid transformation matrices rt (4×4×N) that align local marker positions to global marker positions over time.
        """
        rt_parent_functional = self.scipy_optimal_rt(
            markers_in_global=self.parent_markers_global,
            static_markers_in_local=self.parent_static_markers_in_local,
            rt_init=parent_rt_init,
            marker_names=self.parent_marker_names,
        )
        rt_child_functional = self.scipy_optimal_rt(
            markers_in_global=self.child_markers_global,
            static_markers_in_local=self.child_static_markers_in_local,
            rt_init=child_rt_init,
            marker_names=self.child_marker_names,
        )

        return rt_parent_functional, rt_child_functional


class Score(RigidSegmentIdentification):

    def _score_algorithm(
        self, rt_parent: np.ndarray, rt_child: np.ndarray, recursive_outlier_removal: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the center of rotation (CoR) using the SCoRE algorithm (Ehrig et al., 2006).

        Parameters
        ----------
        rt_parent : np.ndarray, shape (4, 4, N)
            Homogeneous transformations of the parent segment (e.g., pelvis)
        rt_child : np.ndarray, shape (4, 4, N)
            Homogeneous transformations of the child segment (e.g., femur)
        recursive_outlier_removal : bool
            If True, performs 95th percentile residual filtering and recomputes the center.

        Returns
        -------
        cor_global : np.ndarray, shape (3,)
            Estimated global position of the center of rotation.
        """
        nb_frames = rt_parent.shape[2]

        # Build linear system A x = b to solve for CoR positions in child and parent segment frames
        A = np.zeros((3 * nb_frames, 6))
        b = np.zeros((3 * nb_frames,))
        A[:, :] = np.nan
        b[:] = np.nan

        for i_frame in range(nb_frames):
            parent_rot = rt_parent[:3, :3, i_frame]
            child_rot = rt_child[:3, :3, i_frame]
            parent_trans = rt_parent[:3, 3, i_frame]
            child_trans = rt_child[:3, 3, i_frame]

            A[3 * i_frame : 3 * (i_frame + 1), 0:3] = child_rot
            A[3 * i_frame : 3 * (i_frame + 1), 3:6] = -parent_rot
            b[3 * i_frame : 3 * (i_frame + 1)] = parent_trans - child_trans

        # Remove nans
        valid_rows = ~np.isnan(A[:, 0])
        A_valid = A[valid_rows, :]
        b_valid = b[valid_rows]

        # Compute SVD
        U, S, Vt = np.linalg.svd(A_valid, full_matrices=False)

        # Compute pseudo-inverse solution
        S_inv = np.diag(1.0 / S)
        CoR = Vt.T @ S_inv @ U.T @ b_valid

        cor_child_local = CoR[:3]
        cor_parent_local = CoR[3:]

        # Compute transformed CoR positions in global frame
        cor_parent_global = np.zeros((4, rt_parent.shape[2]))
        cor_child_global = np.zeros((4, rt_child.shape[2]))
        for i_frame in range(rt_parent.shape[2]):
            cor_parent_global[:, i_frame] = rt_parent[:, :, i_frame] @ np.hstack((cor_parent_local, 1))
            cor_child_global[:, i_frame] = rt_child[:, :, i_frame] @ np.hstack((cor_child_local, 1))

        residuals = np.linalg.norm(cor_parent_global[:3, :] - cor_child_global[:3, :], axis=0)

        if recursive_outlier_removal:
            # The first time, remove the outliers
            threshold = np.mean(residuals) + 1.0 * np.std(residuals)
            valid = residuals < threshold
            if np.sum(valid) < nb_frames:
                _logger.info(f"\nRemoving {nb_frames - np.sum(valid)} frames")
                return self._score_algorithm(
                    rt_parent[:, :, valid], rt_child[:, :, valid], recursive_outlier_removal=False
                )

        # Final output
        cor_mean_global = 0.5 * (np.mean(cor_parent_global[:3, :], axis=1) + np.mean(cor_child_global[:3, :], axis=1))

        _logger.info(
            f"\nThere is a residual distance between the parent's and the child's CoR position of : {np.nanmean(residuals)} +- {np.nanstd(residuals)}"
        )
        return cor_mean_global, cor_parent_local, cor_child_local, rt_parent, rt_child

    def perform_task(
        self,
        original_model: BiomechanicalModelReal,
        new_model: BiomechanicalModelReal,
        parent_rt_init: np.ndarray,
        child_rt_init: np.ndarray,
    ):

        # Reconstruct the trial to identify the orientation of the segments
        rt_parent_functional, rt_child_functional = self.rt_from_trial(parent_rt_init, child_rt_init)

        rt_parent_functional_offsetted, rt_child_functional_offsetted, child_offset_rt = (
            self.remove_offset_from_optimal_rt(original_model, rt_parent_functional, rt_child_functional)
        )

        if self.animate_rt:
            self.animate_the_segment_reconstruction(
                original_model,
                rt_parent_functional_offsetted,
                rt_child_functional_offsetted,
            )

        # Identify center of rotation
        cor_mean_global, cor_parent_local, cor_child_local, rt_parent, rt_child = self._score_algorithm(
            rt_parent_functional_offsetted, rt_child_functional_offsetted, recursive_outlier_removal=True
        )

        scs_child_static = new_model.segments[self.child_name].segment_coordinate_system
        if scs_child_static.is_in_global:
            raise RuntimeError(
                "Something went wrong, the scs of the child segment in the new_model is in the global reference frame."
            )

        scs_of_child_in_local = scs_child_static.scs[:, :, 0]
        scs_of_child_in_local[:3, 3] = cor_parent_local[:3]
        scs_of_child_in_local = child_offset_rt.inverse @ scs_of_child_in_local @ child_offset_rt.rt_matrix

        # TODO: generalize + verify
        # Segment RT
        reset_axis_rt = RotoTransMatrix()
        reset_axis_rt.from_rt_matrix(np.eye(4))
        if self.child_name + "_parent_offset" in new_model.segment_names:
            segment_to_move_rt_from = self.child_name + "_parent_offset"
            if self.child_name + "_reset_axis" in new_model.segment_names:
                reset_axis_rt.from_rt_matrix(
                    deepcopy(new_model.segments[self.child_name + "_reset_axis"].segment_coordinate_system.scs)
                )
        else:
            segment_to_move_rt_from = self.child_name

        new_model.segments[segment_to_move_rt_from].segment_coordinate_system = SegmentCoordinateSystemReal(
            scs=scs_of_child_in_local,
            is_scs_local=True,
        )
        self.replace_components_in_new_jcs(original_model, new_model)


class Sara(RigidSegmentIdentification):
    def __init__(
        self,
        filepath: str,
        parent_name: str,
        child_name: str,
        parent_marker_names: list[str],
        child_marker_names: list[str],
        first_frame: int,
        last_frame: int,
        joint_center_markers: list[str],
        distal_markers: list[str],
        is_longitudinal_axis_from_jcs_to_distal_markers: bool,
        initialize_whole_trial_reconstruction: bool = False,
        animate_rt: bool = False,
    ):

        super(Sara, self).__init__(
            filepath=filepath,
            parent_name=parent_name,
            child_name=child_name,
            parent_marker_names=parent_marker_names,
            child_marker_names=child_marker_names,
            first_frame=first_frame,
            last_frame=last_frame,
            animate_rt=animate_rt,
            initialize_whole_trial_reconstruction=initialize_whole_trial_reconstruction,
        )

        self.joint_center_markers = joint_center_markers
        self.distal_markers = distal_markers
        self.longitudinal_axis_sign = 1 if is_longitudinal_axis_from_jcs_to_distal_markers else -1

    def _sara_algorithm(self, rt_parent: np.ndarray, rt_child: np.ndarray) -> np.ndarray:
        """
        Perform the SARA algorithm (Ehrig et al., 2007) to estimate the axis of rotation (AoR)
        between two segments over time using homogeneous transformation matrices.

        Parameters
        ----------
        rt_parent : ndarray (4, 4, N)
            Homogeneous transformation matrices from the global frame to the parent segment.
        rt_child : ndarray (4, 4, N)
            Homogeneous transformation matrices from the global frame to the child segment.

        Returns
        -------
        aor_global : ndarray (3, N)
            Orientation of the axis of rotation expressed in the global frame at each frame.
        """
        nb_frames = rt_parent.shape[2]

        # Build block matrix system R * [rCsi; rCsj] = (p_j - p_i)
        rot = np.zeros((3 * nb_frames, 6))
        trans = np.zeros((3 * nb_frames, 1))

        for i_frame in range(nb_frames):
            rotation_parent = rt_parent[:3, :3, i_frame]
            rotation_child = rt_child[:3, :3, i_frame]
            rot[3 * i_frame : 3 * i_frame + 3, :3] = rotation_parent
            rot[3 * i_frame : 3 * i_frame + 3, 3:] = -rotation_child
            trans[3 * i_frame : 3 * i_frame + 3, 0] = rt_child[:3, 3, i_frame] - rt_parent[:3, 3, i_frame]

        # SVD of the block matrix
        U, S, Vt = np.linalg.svd(rot, full_matrices=False)
        V = Vt.T  # Align with MATLAB's V

        # Axis orientations in local frames
        aor_local_parent = V[:3, -1]
        aor_local_child = V[3:, -1]
        aor_local_parent /= np.linalg.norm(aor_local_parent)
        aor_local_child /= np.linalg.norm(aor_local_child)

        # Compute axis direction in global frame over time
        a_parent_global = np.zeros((3, nb_frames))
        a_child_global = np.zeros((3, nb_frames))
        residual_angle = np.zeros((nb_frames,))
        for i_frame in range(nb_frames):
            a_parent_global[:, i_frame] = rt_parent[:3, :3, i_frame] @ aor_local_parent
            a_child_global[:, i_frame] = rt_child[:3, :3, i_frame] @ aor_local_child
            residual_angle[i_frame] = np.arccos(
                np.dot(a_parent_global[:, i_frame], a_child_global[:, i_frame])
                / (np.linalg.norm(a_parent_global[:, i_frame]) * np.linalg.norm(a_child_global[:, i_frame]))
            )

        aor_global = 0.5 * (a_parent_global + a_child_global)

        _logger.info(
            f"\nThere is a residual angle between the parent's and the child's AoR of : {np.nanmean(residual_angle)*180/np.pi} +- {np.nanstd(residual_angle)*180/np.pi} degrees."
        )

        return aor_global

    def _longitudinal_axis(self, original_model: BiomechanicalModelReal) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the longitudinal axis of the segment and the joint center.
        """
        segment_rt_in_global = original_model.forward_kinematics()
        parent_jcs_in_global = RotoTransMatrix()
        parent_jcs_in_global.from_rt_matrix(segment_rt_in_global[self.parent_name])

        joint_center_marker_index = original_model.markers_indices(self.joint_center_markers)
        joint_center_markers_in_global = original_model.markers_in_global()[:, joint_center_marker_index]
        joint_center_global = np.mean(joint_center_markers_in_global, axis=1)
        joint_center_local = parent_jcs_in_global.inverse @ joint_center_global

        distal_marker_index = original_model.markers_indices(self.distal_markers)
        distal_markers_in_global = original_model.markers_in_global()[:, distal_marker_index]
        distal_center_global = np.mean(distal_markers_in_global, axis=1)
        distal_center_in_local = parent_jcs_in_global.inverse @ distal_center_global

        longitudinal_axis_local = distal_center_in_local - joint_center_local
        longitudinal_axis_local[:3] *= self.longitudinal_axis_sign
        longitudinal_axis_local[:3] /= np.linalg.norm(longitudinal_axis_local[:3])
        longitudinal_axis_local[3] = 1

        return joint_center_local, longitudinal_axis_local

    def get_rotation_index(self, original_model):
        rot = original_model.segments[self.child_name].rotations.value
        if len(rot) != 1:
            raise RuntimeError(
                f"The Sara algorithm is meant to be used with a one DoF joint, you have defined rotations {original_model.segments[self.child_name].rotations} for segment {self.child_name}."
            )
        elif rot == "x":
            aor_index = 0
            perpendicular_index = 1
            longitudinal_index = 2
        elif rot == "y":
            raise NotImplementedError(
                "This axis combination has not been tested yet. Please make sure that the cross product make sense (correct order and correct sign)."
            )
            aor_index = 1
            perpendicular_index = 0
            longitudinal_index = 2
        elif rot == "z":
            aor_index = 2
            perpendicular_index = 0
            longitudinal_index = 1
        return aor_index, perpendicular_index, longitudinal_index

    def _extract_scs_from_axis(
        self,
        original_model: BiomechanicalModelReal,
        aor_local: np.ndarray,
        joint_center_local: np.ndarray,
        longitudinal_axis_local: np.ndarray,
    ) -> np.ndarray:
        """
        Extract the segment coordinate system (SCS) from the axis of rotation.
        """
        if original_model.has_parent_offset(self.child_name):
            raise NotImplementedError("Please implement generalization for ghost segments!")

        aor_index, perpendicular_index, longitudinal_index = self.get_rotation_index(original_model)

        # Extract an orthonormal basis
        perpendicular_axis = np.cross(aor_local[:3], longitudinal_axis_local[:3, 0])
        perpendicular_axis /= np.linalg.norm(perpendicular_axis)
        accurate_longitudinal_axis = np.cross(perpendicular_axis, aor_local[:3])
        accurate_longitudinal_axis /= np.linalg.norm(accurate_longitudinal_axis)

        scs_of_child_in_local = np.identity(4)
        scs_of_child_in_local[:3, aor_index] = aor_local[:3]
        scs_of_child_in_local[:3, perpendicular_index] = -perpendicular_axis
        scs_of_child_in_local[:3, longitudinal_index] = accurate_longitudinal_axis
        scs_of_child_in_local[:3, 3] = joint_center_local[:3, 0]
        scs_of_child_in_local[3, 3] = 1

        return scs_of_child_in_local

    def _check_aor(self, original_model: BiomechanicalModelReal, aor_global: np.ndarray) -> np.ndarray:

        def compute_angle_difference():
            angles = np.zeros((nb_frames,))
            for i_frame in range(nb_frames):
                angles[i_frame] = np.arccos(
                    np.dot(aor_global[:3, i_frame], original_axis[:3])
                    / (np.linalg.norm(aor_global[:3, i_frame]) * np.linalg.norm(original_axis[:3]))
                )
            return angles

        aor_index, _, _ = self.get_rotation_index(original_model)
        original_axis = original_model.forward_kinematics()[self.child_name][:, aor_index, 0]

        if aor_global.shape[0] == 3:
            aor_global = np.vstack((aor_global, np.ones((aor_global.shape[1]))))

        nb_frames = aor_global.shape[1]
        angles = compute_angle_difference()
        if np.abs(np.nanmean(angles) - np.pi) * 180 / np.pi < 30:
            aor_global[:3, :] = -aor_global[:3, :]
            angles = compute_angle_difference()
        if np.abs(np.nanmean(angles)) * 180 / np.pi > 30:
            raise RuntimeError(
                f"The optimal axis of rotation is more than 30° appart from the original axis. This is suspicious, please check the markers used for the sara algorithm."
            )
        if np.nanstd(angles) * 180 / np.pi > 30:
            raise RuntimeError(
                f"The optimal axis of rotation is not stable over time. This is suspicious, please check the markers used for the sara algorithm."
            )
        return aor_global

    def _get_aor_local(self, aor_global: np.ndarray, rt_parent_functional: np.ndarray) -> np.ndarray:
        """
        This function computes the axis of rotation in the local frame of the parent segment.
        It assumes that the axis or rotation does not move much over time in the local reference frame of the parent.
        """
        nb_frames = self.c3d_data.nb_frames
        aor_in_local = np.ones((4, self.c3d_data.nb_frames))
        for i_frame in range(nb_frames):
            if np.any(np.isnan(aor_global[:, i_frame])):
                aor_in_local[:, i_frame] = np.nan
            else:
                # Extract the axis of rotation in local frame
                parent_rt = RotoTransMatrix()
                parent_rt.from_rt_matrix(rt_parent_functional[:, :, i_frame])
                aor_in_local[:3, i_frame] = parent_rt.inverse[:3, :3] @ aor_global[:3, i_frame]
        mean_aor_in_local = mean_unit_vector(aor_in_local)
        return mean_aor_in_local

    def perform_task(
        self, original_model: BiomechanicalModelReal, new_model: BiomechanicalModelReal, parent_rt_init, child_rt_init
    ):

        # Reconstruct the trial to identify the orientation of the segments
        rt_parent_functional, rt_child_functional = self.rt_from_trial(
            parent_rt_init, child_rt_init
        )

        rt_parent_functional_offsetted, rt_child_functional_offsetted, child_offset_rt = (
            self.remove_offset_from_optimal_rt(original_model, rt_parent_functional, rt_child_functional)
        )

        if self.animate_rt:
            self.animate_the_segment_reconstruction(
                original_model,
                rt_parent_functional_offsetted,
                rt_child_functional_offsetted,
            )

        # Identify the approximate longitudinal axis of the segments
        joint_center_local, longitudinal_axis_local = self._longitudinal_axis(new_model)

        # Identify axis of rotation
        aor_global = self._sara_algorithm(rt_parent_functional_offsetted, rt_child_functional_offsetted)
        aor_global = self._check_aor(original_model, aor_global)
        aor_local = self._get_aor_local(aor_global, rt_parent_functional_offsetted)

        # Extract the joint coordinate system
        mean_scs_of_child_in_local = self._extract_scs_from_axis(
            original_model=original_model,
            aor_local=aor_local,
            joint_center_local=joint_center_local,
            longitudinal_axis_local=longitudinal_axis_local,
        )
        # Remove parent offset
        mean_scs_of_child_in_local = child_offset_rt.inverse @ mean_scs_of_child_in_local @ child_offset_rt.rt_matrix

        # Segment RT
        reset_axis_rt = RotoTransMatrix()
        reset_axis_rt.from_rt_matrix(np.eye(4))
        if self.child_name + "_parent_offset" in new_model.segment_names:
            segment_to_move_rt_from = self.child_name + "_parent_offset"
            if self.child_name + "_reset_axis" in new_model.segment_names:
                reset_axis_rt.from_rt_matrix(
                    deepcopy(new_model.segments[self.child_name + "_reset_axis"].segment_coordinate_system.scs)
                )
        else:
            segment_to_move_rt_from = self.child_name

        new_model.segments[segment_to_move_rt_from].segment_coordinate_system = SegmentCoordinateSystemReal(
            scs=mean_scs_of_child_in_local,
            is_scs_local=True,
        )
        self.replace_components_in_new_jcs(original_model, new_model)


class JointCenterTool:
    def __init__(self, original_model: BiomechanicalModelReal, animate_reconstruction: bool = False):

        # Make sure that the scs ar in local before starting
        for segment in original_model.segments:
            if segment.segment_coordinate_system.is_in_global:
                segment.segment_coordinate_system = SegmentCoordinateSystemReal(
                    scs=deepcopy(original_model.segment_coordinate_system_in_local(segment.name)),
                    is_scs_local=True,
                )

        # Original attributes
        self.original_model = original_model
        self.animate_reconstruction = animate_reconstruction

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

    def replace_joint_centers(self, marker_weights) -> BiomechanicalModelReal:

        static_markers_in_global = self.original_model.markers_in_global(np.zeros((self.original_model.nb_q,)))
        for task in self.joint_center_tasks:

            if task.initialize_whole_trial_reconstruction:
                # Reconstruct the whole trial to get a good initial rt for each frame
                q_init = self.original_model.inverse_kinematics(
                    marker_positions=task.c3d_data.get_position(self.original_model.marker_names)[:3, :, :],
                    marker_names=self.original_model.marker_names,
                    marker_weights=marker_weights,
                )
            else:
                # Reconstruct first frame to get an initial rt
                q_init = self.original_model.inverse_kinematics(
                    marker_positions=task.c3d_data.get_position(self.original_model.marker_names)[:3, :, 0],
                    marker_names=self.original_model.marker_names,
                    marker_weights=marker_weights,
                )
            segment_rt_in_global = self.original_model.forward_kinematics(q_init)
            parent_rt_init = segment_rt_in_global[task.parent_name]
            child_rt_init = segment_rt_in_global[task.child_name]
            # parent_rt_init = segment_rt_in_global[task.parent_name + "_parent_offset"]
            # child_rt_init = segment_rt_in_global[task.child_name + "_parent_offset"]

            # Marker positions in the global from the static trial
            task.parent_static_markers_in_global = static_markers_in_global[
                :, self.original_model.markers_indices(task.parent_marker_names)
            ]
            task.child_static_markers_in_global = static_markers_in_global[
                :, self.original_model.markers_indices(task.child_marker_names)
            ]

            # Marker positions in the local from the static trial
            task.parent_static_markers_in_local = np.zeros((4, len(task.parent_marker_names)))
            for i_marker, marker_name in enumerate(task.parent_marker_names):
                task.parent_static_markers_in_local[:, i_marker] = (
                    self.original_model.segments[task.parent_name].markers[marker_name].position[:, 0]
                )
            task.child_static_markers_in_local = np.zeros((4, len(task.child_marker_names)))
            for i_marker, marker_name in enumerate(task.child_marker_names):
                task.child_static_markers_in_local[:, i_marker] = (
                    self.original_model.segments[task.child_name].markers[marker_name].position[:, 0]
                )

            # Marker positions in the global from this functional trial
            task.parent_markers_global = task.c3d_data.get_position(task.parent_marker_names)
            task.child_markers_global = task.c3d_data.get_position(task.child_marker_names)

            if self.animate_reconstruction:
                task.animate_the_segment_reconstruction(
                    self.original_model,
                    parent_rt_init,
                    child_rt_init,
                )

            # Replace the joint center in the new model
            task.check_marker_positions()
            task.perform_task(self.original_model, self.new_model, parent_rt_init, child_rt_init)
            self.new_model.segments_rt_to_local()

        return self.new_model
