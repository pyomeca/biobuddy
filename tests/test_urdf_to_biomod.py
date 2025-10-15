"""
TODO: add MuJoCo muscle implementation
see https://github.com/MyoHub/myo_sim/blob/main/elbow/assets/myoelbow_2dof6muscles_body.xml
"""
from enum import Enum
import os
import biorbd
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import numpy.testing as npt

from biobuddy import MuscleType, MuscleStateType, BiomechanicalModelReal


class ModelEvaluation:
    def __init__(self, biomod, urdf_model):
        self.biomod_model = biorbd.Model(biomod)
        self.mujoco_model = mujoco.MjModel.from_xml_path(urdf_model)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

    def from_markers(self, markers: np.ndarray, marker_names: list = None, plot: bool = True):
        """
        Run test using markers data:
        1) inverse kinematic using biorbd
        2) apply the states on both model
        3) compare the markers positions during the movement

        Parameter:
        markers: np.ndarray()
            markers data (3, nb_markers, nb_frames) in the order of biomod model
        marker_names: list
            list of markers names in the same order as the biomod model
        plot: bool
            plot the markers position at the end of the evaluation

        Returns:
        markers_error: np.ndarray()
        """
        if markers.shape[1] != self.mujoco_model.nsite:
            raise RuntimeError("The number of markers in the model and the markers data must be the same.")
        elif markers.shape[0] != 3:
            raise RuntimeError("The markers data must be a 3D array of dim (3, n_markers, n_frames).")

        # 1) inverse kinematic using biorbd
        states = self._run_inverse_kin(markers)
        self.marker_names = marker_names
        self.markers = markers
        return self.from_states(states=states, plot=plot)

    def from_states(self, states, plot: bool = True) -> list:
        pass

    def test_segment_names(self):
        """Test that segment names and hierarchy match between MuJoCo and Biorbd"""

        # Test number of segments (bodies)
        nb_bodies = self.mujoco_model.nbody - 1  # Exclude world body
        biorbd_segment_names = [
            self.biomod_model.segment(i).name().to_string() for i in range(self.biomod_model.nbSegment())
        ]
        biorbd_parent_names = [
            self.biomod_model.segment(i).parent().to_string() for i in range(self.biomod_model.nbSegment())
        ]
        assert len(biorbd_segment_names) >= nb_bodies

        # Iterate through MuJoCo bodies (skip world body at index 0)
        for i_body in range(1, self.mujoco_model.nbody):
            # Test body names
            mujoco_body_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, i_body)
            assert mujoco_body_name in biorbd_segment_names

            # Test parent
            parent_id = self.mujoco_model.body_parentid[i_body]
            if parent_id > 0:  # If not attached to world
                mujoco_parent_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
                assert mujoco_parent_name in biorbd_parent_names

        # Test DoFs
        assert self.mujoco_model.nq == self.biomod_model.nbQ()

        # Get joint ranges from biorbd
        min_bound_biorbd = []
        max_bound_biorbd = []
        for segment in self.biomod_model.segments():
            this_range = segment.QRanges()
            for i in range(len(this_range)):
                min_bound_biorbd += [segment.QRanges()[i].min()]
                max_bound_biorbd += [segment.QRanges()[i].max()]

        # Compare joint ranges
        ordered_mujoco_idx = self._reorder_mujoco_joints()
        for i_dof_biomod, i_dof_mujoco in enumerate(ordered_mujoco_idx):
            # Test ranges
            jnt_id = self._get_joint_for_qpos(i_dof_mujoco)
            if jnt_id >= 0:
                min_bound_mujoco = self.mujoco_model.jnt_range[jnt_id, 0]
                max_bound_mujoco = self.mujoco_model.jnt_range[jnt_id, 1]
                npt.assert_almost_equal(min_bound_mujoco, min_bound_biorbd[i_dof_biomod], decimal=5)
                npt.assert_almost_equal(max_bound_mujoco, max_bound_biorbd[i_dof_biomod], decimal=5)

    def _plot_markers(
            self, default_nb_line: int, mujoco_marker_idx: list, mujoco_markers: np.ndarray, biorbd_markers: np.ndarray
    ):
        nb_markers = mujoco_markers.shape[1]
        var = ceil(nb_markers / default_nb_line)
        nb_line = var if var < default_nb_line else default_nb_line

        plt.figure("Markers (titles : (mujoco/biorbd))")
        list_labels = ["mujoco markers", "biorbd markers"]
        for m in range(nb_markers):
            plt.subplot(nb_line, ceil(nb_markers / nb_line), m + 1)
            for i in range(3):
                if self.markers is not None:
                    plt.plot(self.markers[i, m, :], "r--")
                    list_labels = ["experimental markers"] + list_labels
                plt.plot(mujoco_markers[i, m, :], "b")
                plt.plot(biorbd_markers[i, m, :], "g")

            mujoco_site_name = mujoco.mj_id2name(
                self.mujoco_model, mujoco.mjtObj.mjOBJ_SITE, mujoco_marker_idx[m]
            )
            plt.title(
                f"{mujoco_site_name}/"
                f"{self.biomod_model.markerNames()[m].to_string()}"
            )
            if m == 0:
                plt.legend(labels=list_labels)

    def _plot_states(self, default_nb_line: int, ordered_mujoco_idx: list, mujoco_states: np.ndarray,
                     states: np.ndarray):
        plt.figure("states (titles : (mujoco/biorbd))")
        var = ceil(states.shape[0] / default_nb_line)
        nb_line = var if var < default_nb_line else default_nb_line
        for i in range(states.shape[0]):
            plt.subplot(nb_line, ceil(states.shape[0] / nb_line), i + 1)
            plt.plot(mujoco_states[i, :], "b")
            plt.plot(states[i, :], "g")

            jnt_id = self._get_joint_for_qpos(ordered_mujoco_idx[i])
            mujoco_dof_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT,
                                                jnt_id) if jnt_id >= 0 else f"dof_{i}"
            plt.title(
                f"{mujoco_dof_name}/"
                f"{self.biomod_model.nameDof()[i].to_string()}"
            )
            if i == 0:
                plt.legend(labels=["mujoco states", "biorbd states"])
        plt.show()

    def _update_mujoco_model(self, states: np.ndarray, ordered_idx: list) -> np.ndarray:
        """
        Update the MuJoCo model to match the biomod model

        Parameters
        ----------
        states : np.ndarray
            The joint angles for 1 frame
        ordered_idx : list
            The list of the index of the joints in the MuJoCo model

        Returns
        -------
        np.array
            The mujoco_model_state for the current frame
        """
        mujoco_state = states.copy()
        for b in range(states.shape[0]):
            self.mujoco_data.qpos[ordered_idx[b]] = states[b]

        # Forward kinematics
        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)

        return mujoco_state

    def _reorder_mujoco_joints(self):
        """
        Reorder the coordinates to have rotation before translation like biorbd model
        MuJoCo uses qpos array which may have a different ordering
        """
        # For now, assume same ordering - this may need adjustment based on URDF structure
        ordered_idx = list(range(self.mujoco_model.nq))
        return ordered_idx

    def _get_joint_for_qpos(self, qpos_idx):
        """
        Get the joint ID corresponding to a qpos index
        """
        for jnt_id in range(self.mujoco_model.njnt):
            qpos_start = self.mujoco_model.jnt_qposadr[jnt_id]
            jnt_type = self.mujoco_model.jnt_type[jnt_id]

            # Determine number of qpos elements for this joint
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                nq = 7
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                nq = 4
            else:
                nq = 1

            if qpos_start <= qpos_idx < qpos_start + nq:
                return jnt_id
        return -1

    def _run_inverse_kin(self, markers: np.ndarray) -> np.ndarray:
        """
        Run biorbd inverse kinematics
        Parameters
        ----------
        markers: np.ndarray
            Markers data
        Returns
        -------
            states: np.ndarray
        """
        ik = biorbd.InverseKinematics(self.biomod_model, markers)
        ik.solve()
        return ik.q


class KinematicsTest(ModelEvaluation):
    def __init__(self, biomod: str, urdf_model: str):
        super(KinematicsTest, self).__init__(biomod, urdf_model)
        self.marker_names = None
        self.markers = None

    def from_states(self, states, plot: bool = True) -> list:
        """
        Run test using states data:
        1) apply the states on both model
        2) compare the markers positions during the movement

        Parameter:
        states: np.ndarray()
            states data (nb_dof, nb_frames) in the order of biomod model
        plot: bool
            plot the markers position at the end of the evaluation

        Returns:
        markers_error: list
        """
        nb_markers = self.mujoco_model.nsite
        nb_frame = states.shape[1]
        mujoco_markers = np.ndarray((3, nb_markers, nb_frame))
        biorbd_markers = np.ndarray((3, nb_markers, nb_frame))
        markers_error = []
        mujoco_marker_idx = []
        ordered_mujoco_idx = self._reorder_mujoco_joints()
        mujoco_state = np.copy(states)

        for i in range(nb_frame):
            mujoco_state[:, i] = self._update_mujoco_model(states[:, i], ordered_mujoco_idx)
            bio_markers_array = self.biomod_model.markers(states[:, i])

            # Build list of MuJoCo site names
            mujoco_site_names = [
                mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_SITE, s)
                for s in range(self.mujoco_model.nsite)
            ]

            mujoco_marker_idx = []
            for m in range(nb_markers):
                if self.marker_names and self.marker_names[m] != mujoco.mj_id2name(
                        self.mujoco_model, mujoco.mjtObj.mjOBJ_SITE, m
                ):
                    raise RuntimeError(
                        "Markers names are not the same between names and MuJoCo model."
                        " Place markers in the same order as the model."
                    )

                biomod_marker_name = self.biomod_model.markerNames()[m].to_string()
                mujoco_idx = mujoco_site_names.index(biomod_marker_name)
                mujoco_marker_idx.append(mujoco_idx)

                # Get marker position in MuJoCo (sites are in xpos)
                mujoco_markers[:, m, i] = self.mujoco_data.site_xpos[mujoco_idx].copy()
                biorbd_markers[:, m, i] = bio_markers_array[m].to_array()
                markers_error.append(np.mean(np.sqrt((mujoco_markers[:, m, i] - biorbd_markers[:, m, i]) ** 2)))

        if plot:
            default_nb_line = 5
            self._plot_markers(default_nb_line, mujoco_marker_idx, mujoco_markers, biorbd_markers)
            self._plot_states(default_nb_line, ordered_mujoco_idx, mujoco_state, states)
            plt.show()

        return markers_error

class VisualizeModel:
    def __init__(self, biomod_filepath):
        try:
            import pyorerun
        except ImportError:
            raise ImportError("pyorerun must be installed to visualize the model.")

        # Visualization
        t = np.linspace(0, 1, 10)
        viz = pyorerun.PhaseRerun(t)

        # Model output
        model = pyorerun.BiorbdModel(biomod_filepath)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        model.options.show_marker_labels = False
        model.options.show_center_of_mass_labels = False
        q = np.zeros((model.nb_q, 10))
        viz.add_animated_model(model, q)

        # Model reference
        reference_model = pyorerun.BiorbdModel(biomod_filepath.replace(".bioMod", "_reference.bioMod"))
        reference_model.options.transparent_mesh = False
        reference_model.options.show_gravity = True
        reference_model.options.show_marker_labels = False
        reference_model.options.show_center_of_mass_labels = False
        q_ref = np.zeros((reference_model.nb_q, 10))
        q_ref[0, :] = 0.5
        viz.add_animated_model(reference_model, q_ref)

        # Animate
        viz.rerun_by_frame("Model output")


def test_kinematics():
    """Test kinematics conversion from URDF to BioMod"""

    np.random.seed(42)

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    biomod_filepath = parent_path + f"/examples/models/flexiv_Rizon10s_kinematics.bioMod"
    urdf_filepath = parent_path + f"/examples/models/flexiv_Rizon10s_kinematics.urdf"

    # Delete the biomod file so we are sure to create it
    if os.path.exists(biomod_filepath):
        os.remove(biomod_filepath)

    # Convert URDF to biomod
    model = BiomechanicalModelReal().from_urdf(
        filepath=urdf_filepath,
    )
    model.to_biomod(biomod_filepath)

    # Test that the model created is valid
    biomod_model = biorbd.Model(biomod_filepath)
    nb_q = biomod_model.nbQ()

    # Test the marker position error
    kin_test = KinematicsTest(biomod=biomod_filepath, urdf_model=urdf_filepath)
    markers_error = kin_test.from_states(states=np.random.rand(nb_q, 20) * 0.2, plot=False)
    np.testing.assert_almost_equal(np.mean(markers_error), 0, decimal=4)

def test_translation_urdf_to_biomod():
    """Test comprehensive URDF to BioMod translation"""

    np.random.seed(42)

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_filepaths = [
        parent_path + f"/examples/models/flexiv_Rizon10s_kinematics.urdf",
        parent_path + f"/examples/models/kuka_lwr.urdf",
    ]

    for urdf_filepath in urdf_filepaths:
        biomod_filepath = urdf_filepath.replace(".urdf", ".bioMod")

        # Delete the biomod file so we are sure to create it
        if os.path.exists(biomod_filepath):
            os.remove(biomod_filepath)

        print(f" ******** Converting {urdf_filepath} ******** ")

        # Convert URDF to biomod
        model = BiomechanicalModelReal().from_urdf(
            filepath=urdf_filepath,
        )
        model.to_biomod(biomod_filepath, with_mesh=True)

        # Test that the model created is valid
        biomod_model = biorbd.Model(biomod_filepath)
        nb_q = biomod_model.nbQ()
        nb_markers = biomod_model.nbMarkers()
        nb_muscles = biomod_model.nbMuscles()

        # Test the components
        model_evaluation = ModelEvaluation(biomod=biomod_filepath, urdf_model=urdf_filepath)
        model_evaluation.test_segment_names()

        # Test the position of the markers
        if nb_markers > 0:
            kin_test = KinematicsTest(biomod=biomod_filepath, urdf_model=urdf_filepath)
            markers_error = kin_test.from_states(states=np.random.rand(nb_q, 1) * 0.2, plot=False)
            np.testing.assert_almost_equal(np.mean(markers_error), 0, decimal=4)

        if os.path.exists(biomod_filepath):
            os.remove(biomod_filepath)


if __name__ == "__main__":
    # Run tests
    test_kinematics()
    test_translation_urdf_to_biomod()