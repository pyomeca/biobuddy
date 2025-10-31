from enum import Enum
import os
import biorbd
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import opensim as osim
import pytest
import numpy.testing as npt
import lxml

from biobuddy import MuscleType, MuscleStateType, BiomechanicalModelReal
from test_utils import compare_models


class MotionType(Enum):
    TRANSLATION = 1
    TRANSLATION_AND_ROTATION = 2
    ROTATION = 3


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

    def test_segment_names(self):

        # Test number of segments
        nb_segments = self.osim_model.getNumBodies()
        biorbd_segment_names = [
            self.biomod_model.segment(i).name().to_string() for i in range(self.biomod_model.nbSegment())
        ]
        biorbd_parent_names = [
            self.biomod_model.segment(i).parent().to_string() for i in range(self.biomod_model.nbSegment())
        ]
        assert len(biorbd_segment_names) >= nb_segments

        for i_segment in range(nb_segments):

            # Test segment names
            current_segment = self.osim_model.get_BodySet().get(i_segment)
            osim_segment_name = current_segment.getName()
            assert osim_segment_name in biorbd_segment_names

            # Test parent
            socket_names = [current_segment.getSocketNames().get(i) for i in range(current_segment.getNumSockets())]
            if "parent_frame" in socket_names:
                osim_parent = current_segment.getSocket("parent_frame").getConnecteeName()

                # Find corresponding parent in Biorbd (only if it's in the list)
                assert osim_parent in biorbd_parent_names

        # TODO: Test meshes

        for i_joint in range(self.osim_model.getNumJoints()):
            current_joint = self.osim_model.getJointSet().get(i_joint)
            print(current_joint.getName())

        # Test DoFs
        osim_dofs = self.osim_model.getCoordinateSet()
        ordered_osim_idx = self._reorder_osim_coordinate()
        assert self.osim_model.getCoordinateSet().getSize() == self.biomod_model.nbQ()

        min_bound_biorbd = []
        max_bound_biorbd = []
        for segment in self.biomod_model.segments():
            this_range = segment.QRanges()
            for i in range(len(this_range)):
                min_bound_biorbd += [segment.QRanges()[i].min()]
                max_bound_biorbd += [segment.QRanges()[i].max()]

        for i_dof_biomod, i_dof_osim in enumerate(ordered_osim_idx):
            # Test ranges
            min_bound_osim = osim_dofs.get(i_dof_osim).get_range(0)
            max_bound_osim = osim_dofs.get(i_dof_osim).get_range(1)
            npt.assert_almost_equal(min_bound_osim, min_bound_biorbd[i_dof_biomod], decimal=5)
            npt.assert_almost_equal(max_bound_osim, max_bound_biorbd[i_dof_biomod], decimal=5)

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
    ) -> np.ndarray:
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

                    try:
                        motion_type = MotionType(
                            self.osim_model.getJointSet().get(i).get_coordinates(j).getMotionType()
                        )
                    except:
                        raise RuntimeError("Unknown motionType.")

                    if motion_type == MotionType.TRANSLATION:
                        translation_idx.append(tot_idx + j)
                    elif motion_type == MotionType.TRANSLATION_AND_ROTATION:
                        raise RuntimeError(f"TODO: {MotionType.TRANSLATION_AND_ROTATION.value} must be split in two.")
                    elif motion_type == MotionType.ROTATION:
                        rotation_idx.append(tot_idx + j)

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
    def __init__(self, biomod_filepath):
        try:
            import pyorerun
            import numpy as np
        except ImportError:
            raise ImportError("pyorerun must be installed to visualize the model. ")

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

    # For ortho_norm_basis
    np.random.seed(42)

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    biomod_filepath = parent_path + f"/examples/models/Wu_Shoulder_Model_via_points.bioMod"
    osim_filepath = parent_path + f"/examples/models/Wu_Shoulder_Model_via_points.osim"

    # Delete the biomod file so we are sure to create it
    if os.path.exists(biomod_filepath):
        os.remove(biomod_filepath)

    # Convert osim to biomod
    model = BiomechanicalModelReal().from_osim(
        filepath=osim_filepath,
        muscle_type=MuscleType.HILL_DE_GROOTE,
        muscle_state_type=MuscleStateType.DEGROOTE,
        mesh_dir="Geometry_cleaned",
    )
    model.fix_via_points()
    model.to_biomod(biomod_filepath, with_mesh=False)

    # Test that the model created is valid
    biomod_model = biorbd.Model(biomod_filepath)
    nb_q = biomod_model.nbQ()

    # Test the marker position error
    kin_test = KinematicsTest(biomod=biomod_filepath, osim_model=osim_filepath)
    markers_error = kin_test.from_states(states=np.random.rand(nb_q, 20) * 0.2, plot=False)
    np.testing.assert_almost_equal(np.mean(markers_error), 0, decimal=4)


def test_moment_arm():

    # For ortho_norm_basis
    np.random.seed(42)

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    biomod_model = parent_path + "/examples/models/Wu_Shoulder_Model_via_points.bioMod"
    osim_model = parent_path + "/examples/models/Wu_Shoulder_Model_via_points.osim"

    # Test the moment arm error
    muscle_test = MomentArmTest(biomod=biomod_model, osim_model=osim_model)
    muscle_error = muscle_test.from_markers(markers=np.random.rand(3, 22, 20), plot=False)
    np.testing.assert_array_less(np.max(muscle_error), 0.025)
    np.testing.assert_array_less(np.median(muscle_error), 0.0025)


def compare_osim_translated_model(model1: BiomechanicalModelReal, model2: BiomechanicalModelReal, decimal: int = 5):

    # Number of components
    assert model1.nb_q == model2.nb_q
    assert model1.nb_markers == model2.nb_markers
    assert model1.nb_contacts == model2.nb_contacts
    assert model1.nb_imus == model2.nb_imus
    assert model1.nb_muscle_groups == model2.nb_muscle_groups
    assert model1.nb_muscles == model2.nb_muscles
    assert model1.nb_via_points == model2.nb_via_points

    # Mass
    mass1 = model1.mass
    mass2 = model2.mass
    npt.assert_almost_equal(mass1, mass2, decimal=decimal)

    # CoM
    com1 = model1.total_com_in_global()
    com2 = model2.total_com_in_global()
    npt.assert_almost_equal(com1, com2, decimal=decimal)

    # Markers
    markers1 = model1.markers_in_global()
    markers2 = model1.markers_in_global()
    npt.assert_almost_equal(markers1, markers2, decimal=decimal)

    # Via points
    for muscle_name in model1.muscle_names:
        origin1 = model1.muscle_origin_in_global(muscle_name)
        origin2 = model2.muscle_origin_in_global(muscle_name)
        npt.assert_almost_equal(origin1, origin2, decimal=decimal)

        insertion1 = model1.muscle_insertion_in_global(muscle_name)
        insertion2 = model2.muscle_insertion_in_global(muscle_name)
        npt.assert_almost_equal(insertion1, insertion2, decimal=decimal)

        vp1 = model1.via_points_in_global(muscle_name)
        vp2 = model2.via_points_in_global(muscle_name)
        npt.assert_almost_equal(vp1, vp2, decimal=decimal)


def test_translation_osim_to_biomod():

    # For ortho_norm_basis
    np.random.seed(42)

    # Paths
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    successful_models = ["Arm26/arm26.osim", "Gait2354_Simbody/subject01_simbody.osim"]
    pin_joint_error_models = [
        "Pendulum/double_pendulum.osim",
        "DoublePendulum/double_pendulum.osim",
        "Rajagopal/RajagopalLaiUhlrich2023.osim",
        "Rajagopal/Rajagopal2016.osim",
        "Rajagopal_OpenSense/Rajagopal2015_opensense.osim",
        "Gait10dof18musc/subject01_metabolics_spring.osim",
        "Gait10dof18musc/subject01.osim",
        "Gait10dof18musc/gait10dof18musc.osim",
        "Gait10dof18musc/subject01_metabolics_path_spring.osim",
        "Gait10dof18musc/subject01_metabolics.osim",
        "Gait10dof18musc/subject01_metabolics_path_actuator.osim",
    ]
    slider_joint_error_models = ["Tug_of_War/Tug_of_War.osim", "Tug_of_War/Tug_of_War_Millard.osim"]
    lxml_synthax_error = [
        "ToyLanding/ToyLandingModel_activeAFO.osim",
        "ToyLanding/ToyLandingModel_AFO.osim",
        "ToyLanding/ToyLandingModel.osim",
        "SoccerKick/SoccerKickingModel.osim",
        "Jumper/DynamicJumperModel.osim",
        "BouncingBlock/bouncing_block.osim",
        "BouncingBlock/bouncing_block_weak_spring.osim",
        "WalkerModel/WalkerModel.osim",
        "Converting WalkerModel/WalkerModelTerrain.osim",
        "WalkerModel/WalkerModelTerrain.osim",
    ]  # HuntCrossleyForce::ContactParametersSet
    translation_and_rotation_dofs = [
        "Leg6Dof9Musc/leg6dof9musc.osim",
        "Gait2392_Simbody/gait2392_thelen2003muscle.osim",
        "Gait2392_Simbody/subject01_adjusted.osim",
        "Gait2392_Simbody/subject01.osim",
        "Gait2392_Simbody/subject01_simbody_adjusted.osim",
        "Gait2392_Simbody/gait2392_millard2012muscle.osim",
        "Leg39/leg39.osim",
        "Gait2354_Simbody/gait2354_simbody.osim",
        "Gait2354_Simbody/subject01_simbody.osim",
        "Hamner/FullBodyModel_Hamner2010_v2_0.osim",
    ]
    skipped = ["WristModel/wrist.osim"]  # To be verified

    # Test all OpenSim models
    osim_root_path = parent_path + "/external/opensim-models/Models"
    biomod_root_path = parent_path + "/examples/models"
    for root, dirs, files in os.walk(osim_root_path):
        for name in files:
            if name.endswith(".osim"):
                folder = root.split("/")[-1]
                osim_filepath = os.path.join(root, name)
                biomod_filepath = os.path.join(biomod_root_path, name.replace(".osim", ".bioMod"))

                if os.path.join(folder, name) in successful_models + translation_and_rotation_dofs:
                    # Delete the biomod file so we are sure to create it
                    if os.path.exists(biomod_filepath):
                        os.remove(biomod_filepath)

                    print(f" ******** Converting {os.path.join(folder, name)} ******** ")
                    # Convert osim to biomod
                    model = BiomechanicalModelReal().from_osim(
                        filepath=osim_filepath,
                        muscle_type=MuscleType.HILL_DE_GROOTE,
                        muscle_state_type=MuscleStateType.DEGROOTE,
                        mesh_dir=parent_path + "/examples/models/Geometry_cleaned",
                    )
                    model.fix_via_points()
                    model.to_biomod(biomod_filepath, with_mesh=False)

                    # Test that the model created is valid
                    biomod_model = biorbd.Model(biomod_filepath)
                    nb_q = biomod_model.nbQ()
                    nb_markers = biomod_model.nbMarkers()
                    nb_muscles = biomod_model.nbMuscles()

                    if os.path.join(folder, name) not in translation_and_rotation_dofs:
                        # Test the components
                        model_evaluation = ModelEvaluation(biomod=biomod_filepath, osim_model=osim_filepath)
                        model_evaluation.test_segment_names()

                        # Test the position of the markers
                        if nb_markers > 0:
                            kin_test = KinematicsTest(biomod=biomod_filepath, osim_model=osim_filepath)
                            markers_error = kin_test.from_states(states=np.random.rand(nb_q, 1) * 0.2, plot=False)
                            np.testing.assert_almost_equal(np.mean(markers_error), 0, decimal=4)

                        # Test the moment arm error
                        if nb_muscles > 0:
                            muscle_test = MomentArmTest(biomod=biomod_filepath, osim_model=osim_filepath)
                            muscle_error = muscle_test.from_markers(
                                markers=np.random.rand(3, nb_markers, 1), plot=False
                            )
                            np.testing.assert_array_less(np.max(muscle_error), 0.015)
                            np.testing.assert_array_less(np.median(muscle_error), 0.003)

                    # Test that the .biomod can be reconverted into .osim
                    model_from_biomod_2 = BiomechanicalModelReal().from_biomod(
                        filepath=biomod_filepath,
                    )
                    translated_osim_filepath = biomod_filepath.replace(".bioMod", "_translated.osim")
                    model_from_biomod_2.to_osim(filepath=translated_osim_filepath, with_mesh=True)
                    model_from_osim_2 = BiomechanicalModelReal().from_osim(
                        filepath=translated_osim_filepath,
                        muscle_type=MuscleType.HILL_DE_GROOTE,
                        muscle_state_type=MuscleStateType.DEGROOTE,
                        mesh_dir=parent_path + "/examples/models/Geometry_cleaned",
                    )
                    compare_osim_translated_model(model, model_from_osim_2, decimal=4)
                    # compare_models will not work because of the ghost segments
                    # compare_models(model, model_from_osim_2, decimal=5)

                    if os.path.exists(biomod_filepath):
                        os.remove(biomod_filepath)

                    if os.path.exists(translated_osim_filepath):
                        os.remove(translated_osim_filepath)

                elif os.path.join(folder, name) in pin_joint_error_models:
                    with pytest.raises(
                        RuntimeError,
                        match="Joint type PinJoint is not implemented yet. Allowed joint type are: WeldJoint CustomJoint Ground ",
                    ):
                        model = BiomechanicalModelReal().from_osim(
                            filepath=osim_filepath,
                            muscle_type=MuscleType.HILL_DE_GROOTE,
                            muscle_state_type=MuscleStateType.DEGROOTE,
                        )

                elif os.path.join(folder, name) in slider_joint_error_models:
                    with pytest.raises(
                        RuntimeError,
                        match="Joint type SliderJoint is not implemented yet. Allowed joint type are: WeldJoint CustomJoint Ground ",
                    ):
                        model = BiomechanicalModelReal().from_osim(
                            filepath=osim_filepath,
                            muscle_type=MuscleType.HILL_DE_GROOTE,
                            muscle_state_type=MuscleStateType.DEGROOTE,
                        )

                elif os.path.join(folder, name) in lxml_synthax_error:
                    pytest.raises(lxml.etree.XMLSyntaxError)

                else:
                    if os.path.join(folder, name) not in skipped:
                        raise RuntimeError(
                            f"OpenSim added a new model to their repository: {os.path.join(folder, name)}. Please check the model."
                        )
