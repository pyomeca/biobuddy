from enum import Enum
import numpy as np

from ..components.generic.rigidbody.inertia_parameters import InertiaParameters
from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.generic.rigidbody.segment import Segment
from ..components.generic.rigidbody.segment_coordinate_system import SegmentCoordinateSystem
from ..components.generic.rigidbody.mesh import Mesh
from ..utils.protocols import Data
from ..utils.enums import  Translations, Rotations


def point_on_vector_in_local(coef: float, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return coef * (end - start)


def point_on_vector_in_global(coef: float, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return start + coef * (end - start)


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"


class SegmentName(Enum):
    HEAD = "HEAD"
    TRUNK = "TRUNK"
    UPPER_ARM = "UPPER_ARM"
    LOWER_ARM = "LOWER_ARM"
    HAND = "HAND"
    THIGH = "THIGH"
    SHANK = "SHANK"
    FOOT = "FOOT"


class DeLevaTable:
    def __init__(self, total_mass: float, sex: Sex):
        """
        Implementation of the De Leva table (https://www.sciencedirect.com/science/article/pii/0021929095001786)
        for the inertial parameters of the segments of a human body.
        Please note that we have defined the segments from proximal to distal joints to match the kinematic chain.

        Parameters
        ----------
        total_mass
            The mass of the subject
        sex
            The sex ('male' or 'female') of the subject
        """
        self.sex = sex
        self.total_mass = total_mass

        # The following attributes will be set either by from_data or from_measurements
        self.inertial_table = None
        self.top_head_position = None
        self.shoulder_position = None
        self.pelvis_position = None
        self.elbow_position = None
        self.wrist_position = None
        self.finger_position = None
        self.knee_position = None
        self.ankle_position = None
        self.heel_position = None
        self.toes_position = None

    def define_inertial_table(self):
        """
        Define the inertial characteristics of the segments based on the De Leva table.
        """
        # TODO: Adapt to elderly with https://www.sciencedirect.com/science/article/pii/S0021929015004571?via%3Dihub
        # TODO: add Dumas et al. from https://www.sciencedirect.com/science/article/pii/S0021929006000728
        self.inertial_table = {
            Sex.MALE: {
                SegmentName.HEAD: InertiaParameters(
                    mass=lambda m, bio: 0.0694 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.5002), start=self.shoulder_position, end=self.top_head_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0694 * self.total_mass,
                        coef=(0.303, 0.315, 0.261),
                        start=self.top_head_position,
                        end=self.shoulder_position,
                    ),
                ),
                SegmentName.TRUNK: InertiaParameters(
                    mass=lambda m, bio: 0.4346 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.5138), start=self.pelvis_position, end=self.shoulder_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.4346 * self.total_mass,
                        coef=(0.328, 0.306, 0.169),
                        start=self.shoulder_position,
                        end=self.pelvis_position,
                    ),
                ),
                SegmentName.UPPER_ARM: InertiaParameters(
                    mass=lambda m, bio: 0.0271 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.5772), start=self.shoulder_position, end=self.elbow_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0271 * self.total_mass,
                        coef=(0.285, 0.269, 0.158),
                        start=self.shoulder_position,
                        end=self.elbow_position,
                    ),
                ),
                SegmentName.LOWER_ARM: InertiaParameters(
                    mass=lambda m, bio: 0.0162 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.4574), start=self.elbow_position, end=self.wrist_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0162 * self.total_mass,
                        coef=(0.276, 0.265, 0.121),
                        start=self.elbow_position,
                        end=self.wrist_position,
                    ),
                ),
                SegmentName.HAND: InertiaParameters(
                    mass=lambda m, bio: 0.0061 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.7900), start=self.wrist_position, end=self.finger_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0061 * self.total_mass,
                        coef=(0.628, 0.513, 0.401),
                        start=self.wrist_position,
                        end=self.finger_position,
                    ),
                ),
                SegmentName.THIGH: InertiaParameters(
                    mass=lambda m, bio: 0.1416 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4095, start=self.pelvis_position, end=self.knee_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.1416 * self.total_mass,
                        coef=(0.329, 0.329, 0.149),
                        start=self.pelvis_position,
                        end=self.knee_position,
                    ),
                ),
                SegmentName.SHANK: InertiaParameters(
                    mass=lambda m, bio: 0.0433 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4459, start=self.knee_position, end=self.ankle_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0433 * self.total_mass,
                        coef=(0.255, 0.249, 0.103),
                        start=self.knee_position,
                        end=self.ankle_position,
                    ),
                ),
                SegmentName.FOOT: InertiaParameters(
                    mass=lambda m, bio: 0.0137 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4415, start=self.heel_position, end=self.toes_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0137 * self.total_mass,
                        coef=(0.257, 0.245, 0.124),
                        start=self.heel_position,
                        end=self.toes_position,
                    ),
                ),
            },
            Sex.FEMALE: {
                SegmentName.HEAD: InertiaParameters(
                    mass=lambda m, bio: 0.0669 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.4841), start=self.shoulder_position, end=self.top_head_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0669 * self.total_mass,
                        coef=(0.271, 0.295, 0.261),
                        start=self.top_head_position,
                        end=self.shoulder_position,
                    ),
                ),
                SegmentName.TRUNK: InertiaParameters(
                    mass=lambda m, bio: 0.4257 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.4964), start=self.pelvis_position, end=self.shoulder_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.4257 * self.total_mass,
                        coef=(0.307, 0.292, 0.147),
                        start=self.shoulder_position,
                        end=self.pelvis_position,
                    ),
                ),
                SegmentName.UPPER_ARM: InertiaParameters(
                    mass=lambda m, bio: 0.0255 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.5754), start=self.shoulder_position, end=self.elbow_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0255 * self.total_mass,
                        coef=(0.278, 0.260, 0.148),
                        start=self.shoulder_position,
                        end=self.elbow_position,
                    ),
                ),
                SegmentName.LOWER_ARM: InertiaParameters(
                    mass=lambda m, bio: 0.0138 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.4559), start=self.elbow_position, end=self.wrist_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0138 * self.total_mass,
                        coef=(0.261, 0.257, 0.094),
                        start=self.elbow_position,
                        end=self.wrist_position,
                    ),
                ),
                SegmentName.HAND: InertiaParameters(
                    mass=lambda m, bio: 0.0056 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.7474), start=self.wrist_position, end=self.finger_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0056 * self.total_mass,
                        coef=(0.531, 0.454, 0.335),
                        start=self.wrist_position,
                        end=self.finger_position,
                    ),
                ),
                SegmentName.THIGH: InertiaParameters(
                    mass=lambda m, bio: 0.1478 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.3612, start=self.pelvis_position, end=self.knee_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.1478 * self.total_mass,
                        coef=(0.369, 0.364, 0.162),
                        start=self.pelvis_position,
                        end=self.knee_position,
                    ),
                ),
                SegmentName.SHANK: InertiaParameters(
                    mass=lambda m, bio: 0.0481 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4416, start=self.knee_position, end=self.ankle_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0481 * self.total_mass,
                        coef=(0.271, 0.267, 0.093),
                        start=self.knee_position,
                        end=self.ankle_position,
                    ),
                ),
                SegmentName.FOOT: InertiaParameters(
                    mass=lambda m, bio: 0.0129 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4014, start=self.heel_position, end=self.toes_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0129 * self.total_mass,
                        coef=(0.299, 0.279, 0.124),
                        start=self.heel_position,
                        end=self.toes_position,
                    ),
                ),
            },
        }

    def get_joint_position_from_measurements(self,
                                             total_height: float,
                                             pelvis_height: float,
                                             trunk_length: float,
                                             shoulder_width: float,
                                             upper_arm_length: float,
                                                lower_arm_length: float,
                                                hand_length: float,
                                             thigh_length: float,
                                             tibia_length: float,
                                             foot_length: float) -> None:



        self.pelvis_position = np.array([0.0, 0.0, pelvis_height, 1.0])
        self.shoulder_position = np.array([0.0, 0.0, shoulder_height, 1.0])
        self.top_head_position = np.array([0.0, 0.0, total_height, 1.0])
        self.elbow_position = self.shoulder_position - np.array([0.0, 0.0, upper_arm_length, 0.0])
        self.wrist_position = self.elbow_position - np.array([0.0, 0.0, lower_arm_length, 0.0])
        self.finger_position = self.wrist_position - np.array([0.0, 0.0, hand_length, 0.0])
        self.knee_position = np.array([0.0, 0.0, knee_height, 1.0])
        self.ankle_position = np.array([0.0, 0.0, ankle_height, 1.0])
        self.heel_position = np.array([0.0, 0.0, 0.0, 1.0])
        self.toes_position = np.array([0.0, foot_length, 0.0, 1.0])




    def from_data(self, data: Data):
        self.top_head_position = data.values["TOP_HEAD"]
        self.shoulder_position = data.values["SHOULDER"]
        self.pelvis_position = data.values["PELVIS"]
        self.elbow_position = data.values["ELBOW"]
        self.wrist_position = data.values["WRIST"]
        self.finger_position = data.values["FINGER"]
        self.knee_position = data.values["KNEE"]
        self.ankle_position = data.values["ANKLE"]
        self.heel_position = data.values["HEEL"]
        self.toes_position = data.values["TOE"]

        self.define_inertial_table()

    def from_measurements(
        self,
        total_height: float,
        ankle_height: float,
        knee_height: float,
        pelvis_height: float,
        shoulder_height: float,
        finger_span: float,
        wrist_span: float,
        elbow_span: float,
        shoulder_span: float,
        hip_width: float,
        foot_length: float,
    ):

        # Define some length from measurements
        self.total_height = total_height
        self.pelvis_height = pelvis_height
        self.trunk_length = shoulder_height - pelvis_height
        self.hand_length = (finger_span - wrist_span) / 2
        self.lower_arm_length = (wrist_span - elbow_span) / 2
        self.upper_arm_length = (elbow_span - shoulder_span) / 2
        self.shoulder_width = shoulder_span
        self.thigh_length = pelvis_height - knee_height
        self.tibia_length = knee_height - ankle_height
        self.foot_length = foot_length

        self.get_joint_position_from_measurements()
        self.define_inertial_table()

    def __getitem__(self, segment_name: SegmentName) -> InertiaParameters:
        """
        The inertial parameters for a particular segment

        Parameters
        ----------
        segment_name
            The name of the segment
        """
        return self.inertial_table[self.sex][segment_name]


    def to_simple_model(self) -> BiomechanicalModelReal:
        """
        Creates a simple BiomechanicalModelReal based on the measurements used to create the De Leva table.
        TODO: This could be handled differently so that the positions are already hadled properly in the from_ methods
        """

        # Generate the personalized kinematic model
        model_real = BiomechanicalModelReal()

        model_real.add_segment(Segment(name="Ground"))

        model_real.add_segment(
            Segment(
                name="TRUNK",
                parent_name="Ground",
                translations=Translations.XYZ,
                rotations=Rotations.XYZ,
                inertia_parameters=self.inertial_table[SegmentName.TRUNK],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=self.pelvis_position,
                ),
                mesh=Mesh([lambda m, model: np.array([0, 0, 0]),
                          lambda m, model: self.shoulder_position - self.pelvis_position], is_local=True),
            )
        )

        model_real.add_segment(
            Segment(
                name="RTHIGH",
                parent_name="TRUNK",
                rotations=Rotations.XY,
                inertia_parameters=self.inertial_table[SegmentName.THIGH],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=self.pelvis_position,
                ),
                mesh=Mesh([lambda m, model: np.array([0, 0, 0]),
                           lambda m, model: self.shoulder_position - self.pelvis_position], is_local=True),
            )
        )
        reduced_model.segments["RFemur"].add_marker(Marker("RLFE", is_technical=True, is_anatomical=True))
        reduced_model.segments["RFemur"].add_marker(Marker("RMFE", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="RTibia",
                parent_name="RFemur",
                rotations=Rotations.X,
                inertia_parameters=de_leva[SegmentName.SHANK],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["RMFE", "RLFE"]),
                    first_axis=Axis(name=Axis.Name.X, start="RSPH", end="RLM"),
                    second_axis=Axis(
                        name=Axis.Name.Z,
                        start=SegmentCoordinateSystemUtils.mean_markers(["RSPH", "RLM"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["RMFE", "RLFE"]),
                    ),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("RMFE", "RSPH", "RLM", "RLFE"), is_local=False),
            )
        )
        reduced_model.segments["RTibia"].add_marker(Marker("RLM", is_technical=True, is_anatomical=True))
        reduced_model.segments["RTibia"].add_marker(Marker("RSPH", is_technical=True, is_anatomical=True))

        # The foot is a special case since the position of the ankle relatively to the foot length is not given in De Leva
        # So here we assume that the foot com is in the middle of the three foot markers
        foot_inertia_parameters = de_leva[SegmentName.FOOT]
        rt_matrix = RotoTransMatrix()
        rt_matrix.from_euler_angles_and_translation(
            angle_sequence="y",
            angles=np.array([-np.pi / 2]),
            translation=np.array([0.0, 0.0, 0.0]),
        )
        foot_inertia_parameters.center_of_mass = lambda m, bio: rt_matrix.rt_matrix @ np.nanmean(
            np.nanmean(np.array([m[name] for name in ["LSPH", "LLM", "LTT2"]]), axis=0)
            - np.nanmean(np.array([m[name] for name in ["LSPH", "LLM"]]), axis=0),
            axis=1,
        )

        reduced_model.add_segment(
            Segment(
                name="RFoot",
                parent_name="RTibia",
                rotations=Rotations.X,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["RSPH", "RLM"]),
                    first_axis=Axis(
                        Axis.Name.Z, start=SegmentCoordinateSystemUtils.mean_markers(["RSPH", "RLM"]), end="RTT2"
                    ),
                    second_axis=Axis(Axis.Name.X, start="RSPH", end="RLM"),
                    axis_to_keep=Axis.Name.Z,
                ),
                inertia_parameters=foot_inertia_parameters,
                mesh=Mesh(("RLM", "RTT2", "RSPH", "RLM"), is_local=False),
            )
        )
        reduced_model.segments["RFoot"].add_marker(Marker("RTT2", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="LFemur",
                parent_name="Pelvis",
                rotations=Rotations.XY,
                inertia_parameters=de_leva[SegmentName.THIGH],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"])(
                        static_trial.values, None
                    )
                                          - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
                    first_axis=Axis(name=Axis.Name.X, start="LLFE", end="LMFE"),
                    second_axis=Axis(
                        name=Axis.Name.Z,
                        start=SegmentCoordinateSystemUtils.mean_markers(["LMFE", "LLFE"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"]),
                    ),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(
                    (
                        lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"])(
                            static_trial.values, None
                        )
                                       - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
                        "LMFE",
                        "LLFE",
                        lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"])(
                            static_trial.values, None
                        )
                                       - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
                    ),
                    is_local=False,
                ),
            )
        )
        reduced_model.segments["LFemur"].add_marker(Marker("LLFE", is_technical=True, is_anatomical=True))
        reduced_model.segments["LFemur"].add_marker(Marker("LMFE", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="LTibia",
                parent_name="LFemur",
                rotations=Rotations.X,
                inertia_parameters=de_leva[SegmentName.SHANK],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["LMFE", "LLFE"]),
                    first_axis=Axis(name=Axis.Name.X, start="LLM", end="LSPH"),
                    second_axis=Axis(
                        name=Axis.Name.Z,
                        start=SegmentCoordinateSystemUtils.mean_markers(["LSPH", "LLM"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["LMFE", "LLFE"]),
                    ),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("LMFE", "LSPH", "LLM", "LLFE"), is_local=False),
            )
        )
        reduced_model.segments["LTibia"].add_marker(Marker("LLM", is_technical=True, is_anatomical=True))
        reduced_model.segments["LTibia"].add_marker(Marker("LSPH", is_technical=True, is_anatomical=True))

        foot_inertia_parameters = de_leva[SegmentName.FOOT]
        rt_matrix = RotoTransMatrix()
        rt_matrix.from_euler_angles_and_translation(
            angle_sequence="y",
            angles=np.array([-np.pi / 2]),
            translation=np.array([0.0, 0.0, 0.0]),
        )
        foot_inertia_parameters.center_of_mass = lambda m, bio: rt_matrix.rt_matrix @ np.nanmean(
            np.nanmean(np.array([m[name] for name in ["LSPH", "LLM", "LTT2"]]), axis=0)
            - np.nanmean(np.array([m[name] for name in ["LSPH", "LLM"]]), axis=0),
            axis=1,
        )

        reduced_model.add_segment(
            Segment(
                name="LFoot",
                parent_name="LTibia",
                rotations=Rotations.X,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["LSPH", "LLM"]),
                    first_axis=Axis(
                        Axis.Name.Z, start=SegmentCoordinateSystemUtils.mean_markers(["LLM", "LSPH"]), end="LTT2"
                    ),
                    second_axis=Axis(Axis.Name.X, start="LLM", end="LSPH"),
                    axis_to_keep=Axis.Name.Z,
                ),
                inertia_parameters=foot_inertia_parameters,
                mesh=Mesh(("LLM", "LTT2", "LSPH", "LLM"), is_local=False),
            )
        )
        reduced_model.segments["LFoot"].add_marker(Marker("LTT2", is_technical=True, is_anatomical=True))


