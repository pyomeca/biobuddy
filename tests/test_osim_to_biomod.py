import os
import biorbd
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import opensim as osim
import pytest

from biobuddy import MuscleType, MuscleStateType, BiomechanicalModelReal


class ModelEvaluation:
    def __init__(self, biomod, osim_model):
        self.biomod_model = biorbd.Model(biomod)
        self.osim_model = osim.Model(osim_model)

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
        if markers.shape[1] != self.osim_model.getMarkerSet().getSize():
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

    def _plot_markers(
        self, default_nb_line: int, osim_marker_idx: list, osim_markers: np.ndarray, biorbd_markers: np.ndarray
    ):
        nb_markers = osim_markers.shape[1]
        var = ceil(nb_markers / default_nb_line)
        nb_line = var if var < default_nb_line else default_nb_line

        plt.figure("Markers (titles : (osim/biorbd))")
        list_labels = ["osim markers", "biorbd markers"]
        for m in range(nb_markers):
            plt.subplot(nb_line, ceil(nb_markers / nb_line), m + 1)
            for i in range(3):
                if self.markers:
                    plt.plot(self.markers[i, m, :], "r--")
                    list_labels = ["experimental markers"] + list_labels
                plt.plot(osim_markers[i, m, :], "b")
                plt.plot(biorbd_markers[i, m, :], "g")
            plt.title(
                f"{self.osim_model.getMarkerSet().get(osim_marker_idx[m]).getName()}/"
                f"{self.biomod_model.markerNames()[m].to_string()}"
            )
            if m == 0:
                plt.legend(labels=list_labels)

    def _plot_states(self, default_nb_line: int, ordered_osim_idx: list, osim_states: np.ndarray, states: np.ndarray):
        plt.figure("states (titles : (osim/biorbd))")
        var = ceil(states.shape[0] / default_nb_line)
        nb_line = var if var < default_nb_line else default_nb_line
        for i in range(states.shape[0]):
            plt.subplot(nb_line, ceil(states.shape[0] / nb_line), i + 1)
            plt.plot(osim_states[i, :], "b")
            plt.plot(states[i, :], "g")
            plt.title(
                f"{self.osim_model.getCoordinateSet().get(ordered_osim_idx[i]).getName()}/"
                f"{self.biomod_model.nameDof()[i].to_string()}"
            )
            if i == 0:
                plt.legend(labels=["osim states (handle default value)", "states"])
        plt.show()

    def _plot_moment_arm(
        self,
        default_nb_line: int,
        osim_muscle_idx: list,
        ordered_osim_idx: list,
        osim_moment_arm: np.ndarray,
        biorbd_moment_arm: np.ndarray,
    ):
        nb_muscles = len(osim_muscle_idx)
        var = ceil(nb_muscles / default_nb_line)
        nb_line = var if var < default_nb_line else default_nb_line
        # plot osim marker and biomod markers in subplots
        for j in range(osim_moment_arm.shape[0]):
            osim_dof_name = self.osim_model.getCoordinateSet().get(ordered_osim_idx[j]).getName()
            biomod_dof_name = self.biomod_model.nameDof()[j].to_string()
            plt.figure(f"Lever arm osim:{osim_dof_name}/biomod: {biomod_dof_name}\n(titles : (osim/biorbd))")
            list_labels = ["osim lever arm", "biorbd lever arm"]
            for m in range(nb_muscles):
                plt.subplot(nb_line, ceil(nb_muscles / nb_line), m + 1)
                plt.plot(osim_moment_arm[j, m, :], "b")
                plt.plot(biorbd_moment_arm[j, m, :], "g")
                plt.title(
                    f"{self.osim_model.getMuscles().get(osim_muscle_idx[m]).getName()}/"
                    f"{self.biomod_model.muscleNames()[m].to_string()}"
                )
                if m == 0:
                    plt.legend(labels=list_labels)

    def _update_osim_model(
        self, my_state: osim.Model.initializeState, states: np.ndarray, ordered_idx: list
    ) -> np.array:
        """
        Update the osim model to match the biomod model

        Parameters
        ----------
        my_state : osim.Model.initializeState
            The state of the osim model
        states : np.ndarray
            The joint angle for 1 frame
        ordered_idx : list
            The list of the index of the joint in the osim model

        Returns
        -------
        np.array
            The osim_model_state for the curent frame
        """
        osim_state = states.copy()
        for b in range(states.shape[0]):
            if self.osim_model.getCoordinateSet().get(ordered_idx[b]).getDefaultValue() != 0:
                osim_state[b] = states[b] + self.osim_model.getCoordinateSet().get(ordered_idx[b]).getDefaultValue()
            self.osim_model.getCoordinateSet().get(ordered_idx[b]).setValue(my_state, osim_state[b])
        return osim_state

    def _reorder_osim_coordinate(self):
        """
        Reorder the coordinates to have translation after rotation like biorbd model
        """
        tot_idx = 0
        ordered_idx = []
        for i in range(self.osim_model.getJointSet().getSize()):
            translation_idx = []
            rotation_idx = []
            for j in range(self.osim_model.getJointSet().get(i).numCoordinates()):
                if not self.osim_model.getJointSet().get(i).get_coordinates(j).get_locked():
                    if self.osim_model.getJointSet().get(i).get_coordinates(j).getMotionType() == 1:
                        translation_idx.append(tot_idx + j)
                    elif self.osim_model.getJointSet().get(i).get_coordinates(j).getMotionType() == 3:
                        rotation_idx.append(tot_idx + j)
                    else:
                        raise RuntimeError("Unknown motionType.")
            tot_idx += self.osim_model.getJointSet().get(i).numCoordinates()
            ordered_idx += rotation_idx + translation_idx
        return ordered_idx

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
    def __init__(self, biomod: str, osim_model: str):
        super(KinematicsTest, self).__init__(biomod, osim_model)
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
        nb_markers = self.osim_model.getMarkerSet().getSize()
        nb_frame = states.shape[1]
        osim_markers = np.ndarray((3, nb_markers, nb_frame))
        biorbd_markers = np.ndarray((3, nb_markers, nb_frame))
        markers_error = []
        osim_marker_idx = []
        ordered_osim_idx = self._reorder_osim_coordinate()
        osim_state = np.copy(states)
        my_state = self.osim_model.initSystem()
        for i in range(nb_frame):
            osim_state[:, i] = self._update_osim_model(my_state, states[:, i], ordered_osim_idx)
            bio_markers_array = self.biomod_model.markers(states[:, i])
            osim_markers_names = [
                self.osim_model.getMarkerSet().get(m).toString()
                for m in range(self.osim_model.getMarkerSet().getSize())
            ]
            osim_marker_idx = []
            for m in range(nb_markers):
                if self.marker_names and self.marker_names[m] != self.osim_model.getMarkerSet().get(m).getName():
                    raise RuntimeError(
                        "Markers names are not the same between names and opensim model."
                        " Place markers in teh same order as the model."
                    )
                osim_idx = osim_markers_names.index(self.biomod_model.markerNames()[m].to_string())
                osim_marker_idx.append(osim_idx)
                osim_markers[:, m, i] = (
                    self.osim_model.getMarkerSet().get(osim_idx).getLocationInGround(my_state).to_numpy()
                )
                biorbd_markers[:, m, i] = bio_markers_array[m].to_array()
                markers_error.append(np.mean(np.sqrt((osim_markers[:, m, i] - biorbd_markers[:, m, i]) ** 2)))
        if plot:
            default_nb_line = 5
            self._plot_markers(default_nb_line, osim_marker_idx, osim_markers, biorbd_markers)
            self._plot_states(default_nb_line, ordered_osim_idx, osim_state, states)
            plt.show()
        return markers_error


class MomentArmTest(ModelEvaluation):
    def __init__(self, biomod: str, osim_model: str):
        super(MomentArmTest, self).__init__(biomod, osim_model)

    def from_states(self, states, plot: bool = True) -> list:
        """
        Run test using states data:
        1) apply the states on both model
        2) compare the lever arm during the movement

        Parameter:
        states: np.ndarray()
            states data (nb_dof, nb_frames) in the order of biomod model
        plot: bool
            plot the markers position at the end of the evaluation

        Returns:
        moment arm error: list
        """
        nb_muscles = self.osim_model.getMuscles().getSize()
        nb_dof = self.biomod_model.nbQ()
        nb_frame = states.shape[1]
        osim_moment_arm = np.ndarray((nb_dof, nb_muscles, nb_frame))
        biorbd_mament_arm = np.ndarray((nb_dof, nb_muscles, nb_frame))
        moment_arm_error = np.ndarray((nb_dof, nb_muscles))
        osim_muscle_idx = []
        ordered_osim_idx = self._reorder_osim_coordinate()
        osim_state = np.copy(states)
        osim_muscle_names = [
            self.osim_model.getMuscles().get(m).toString() for m in range(self.osim_model.getMuscles().getSize())
        ]
        my_state = self.osim_model.initSystem()
        for i in range(nb_frame):
            osim_state[:, i] = self._update_osim_model(my_state, states[:, i], ordered_osim_idx)
            bio_moment_arm_array = self.biomod_model.musclesLengthJacobian(states[:, i]).to_array()
            osim_muscle_idx = []
            for m in range(nb_muscles):
                osim_idx = osim_muscle_names.index(self.biomod_model.muscleNames()[m].to_string())
                osim_muscle_idx.append(osim_idx)
                for d in range(self.biomod_model.nbDof()):
                    osim_moment_arm[d, m, i] = (
                        -self.osim_model.getMuscles()
                        .get(osim_idx)
                        .computeMomentArm(my_state, self.osim_model.getCoordinateSet().get(ordered_osim_idx[d]))
                    )
                biorbd_mament_arm[:, m, i] = bio_moment_arm_array[m]
                moment_arm_error[:, m] = np.mean(np.sqrt((osim_moment_arm[:, m, i] - biorbd_mament_arm[:, m, i]) ** 2))
        if plot:
            default_nb_line = 5
            self._plot_moment_arm(
                default_nb_line, osim_muscle_idx, ordered_osim_idx, osim_moment_arm, biorbd_mament_arm
            )
            self._plot_states(default_nb_line, ordered_osim_idx, osim_state, states)
            plt.show()
        return moment_arm_error


class VisualizeModel:
    def __init__(self, biomod_file_path):
        try:
            import pyorerun
            import numpy as np
        except ImportError:
            raise ImportError("pyorerun must be installed to visualize the model. ")

        # Visualization
        t = np.linspace(0, 1, 10)
        viz = pyorerun.PhaseRerun(t)

        # Model output
        model = pyorerun.BiorbdModel(biomod_file_path)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        q = np.zeros((model.nb_q, 10))
        viz.add_animated_model(model, q)

        # Model reference
        reference_model = pyorerun.BiorbdModel(biomod_file_path.replace(".bioMod", "_reference.bioMod"))
        reference_model.options.transparent_mesh = False
        reference_model.options.show_gravity = True
        q_ref = np.zeros((reference_model.nb_q, 10))
        q_ref[0, :] = 0.5
        viz.add_animated_model(reference_model, q_ref)

        # Animate
        viz.rerun_by_frame("Model output")


def test_kinematics():

    # For ortho_norm_basis
    np.random.seed(42)

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    biomod_file_path = parent_path + f"/examples/models/Wu_Shoulder_Model_via_points.bioMod"
    osim_file_path = parent_path + f"/examples/models/Wu_Shoulder_Model_via_points.osim"

    # Delete the biomod file so we are sure to create it
    if os.path.exists(biomod_file_path):
        os.remove(biomod_file_path)

    # Convert osim to biomod
    model = BiomechanicalModelReal.from_osim(
        filepath = osim_file_path,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir="Geometry_cleaned"
    )
    model.to_biomod(biomod_file_path, with_mesh=False)

    # Test that the model created is valid
    biomod_model = biorbd.Model(biomod_file_path)
    nb_q = biomod_model.nbQ()

    # Test the kinematics
    kin_test = KinematicsTest(biomod=biomod_file_path, osim_model=osim_file_path)
    markers_error = kin_test.from_states(states=np.random.rand(nb_q, 20) * 0.2, plot=False)
    np.testing.assert_almost_equal(np.mean(markers_error), 0, decimal=4)


def test_moment_arm():
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    np.random.seed(42)
    biomod_model = parent_path + "/examples/models/Wu_Shoulder_Model_via_points.bioMod"
    osim_model = parent_path + "/examples/models/Wu_Shoulder_Model_via_points.osim"
    muscle_test = MomentArmTest(biomod=biomod_model, osim_model=osim_model)
    muscle_error = muscle_test.from_markers(markers=np.random.rand(3, 22, 20), plot=False)
    np.testing.assert_almost_equal(np.median(muscle_error), 0.0017046780530509224, decimal=4)

    muscle_test = MomentArmTest(biomod=biomod_model, osim_model=osim_model)
    print(muscle_test.from_markers(markers=np.random.rand(3, 22, 20), plot=False))
    kin_test = KinematicsTest(biomod=biomod_model, osim_model=osim_model)
    print(kin_test.from_states(states=np.random.rand(16, 20) * 0.2, plot=False))
