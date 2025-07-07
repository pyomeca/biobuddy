from enum import Enum
import numpy as np

from ..components.generic.rigidbody.inertia_parameters import InertiaParameters
from ..utils.protocols import Data


def point_on_vector_in_local(coef: float, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return coef * (end - start)


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
        Implementation of the DeLeva table
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
        self.toes_position = None

    def define_inertial_table(self):
        """
        Define the inertial characteristics of the segments based on the De Leva table.
        """
        # TODO: @pariterre -> Bilateral segments (counts for the right and left sides) should be handled differently to allow for unilateral segments

        # TODO: Addapt to elderly with https://www.sciencedirect.com/science/article/pii/S0021929015004571?via%3Dihub
        # TODO: add Dumas et al. from https://www.sciencedirect.com/science/article/pii/S0021929006000728
        self.inertial_table = {
            Sex.MALE: {
                SegmentName.HEAD: InertiaParameters(
                    mass=lambda m, bio: 0.0694 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.5002, start=self.top_head_position, end=self.shoulder_position
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
                    center_of_mass=lambda m, bio: -point_on_vector_in_local(
                        0.5138, start=self.shoulder_position, end=self.pelvis_position
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
                        0.5772, start=self.shoulder_position, end=self.elbow_position
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
                        0.4574, start=self.elbow_position, end=self.wrist_position
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
                        0.7900, start=self.wrist_position, end=self.finger_position
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
                        0.4415, start=self.ankle_position, end=self.toes_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0137 * self.total_mass,
                        coef=(0.257, 0.245, 0.124),
                        start=self.ankle_position,
                        end=self.toes_position,
                    ),
                ),
            },
            Sex.FEMALE: {
                SegmentName.HEAD: InertiaParameters(
                    mass=lambda m, bio: 0.0669 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4841, start=self.top_head_position, end=self.shoulder_position
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
                    center_of_mass=lambda m, bio: -point_on_vector_in_local(
                        0.4964, start=self.shoulder_position, end=self.pelvis_position
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
                        0.5754, start=self.shoulder_position, end=self.elbow_position
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
                        0.4559, start=self.elbow_position, end=self.wrist_position
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
                        0.7474, start=self.wrist_position, end=self.finger_position
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
                        0.4014, start=self.ankle_position, end=self.toes_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0129 * self.total_mass,
                        coef=(0.299, 0.279, 0.124),
                        start=self.ankle_position,
                        end=self.toes_position,
                    ),
                ),
            },
        }

    def from_data(self, data: Data):
        self.top_head_position = data.values["TOP_HEAD"]
        self.shoulder_position = data.values["SHOULDER"]
        self.pelvis_position = data.values["PELVIS"]
        self.elbow_position = data.values["ELBOW"]
        self.wrist_position = data.values["WRIST"]
        self.finger_position = data.values["FINGER"]
        self.knee_position = data.values["KNEE"]
        self.ankle_position = data.values["ANKLE"]
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
        foot_length: float,
    ):

        # Define some length from measurements
        hand_length = (finger_span - wrist_span) / 2
        lower_arm_length = (wrist_span - elbow_span) / 2
        upper_arm_length = (elbow_span - shoulder_span) / 2

        self.pelvis_position = np.array([0.0, 0.0, pelvis_height, 1.0])
        self.shoulder_position = np.array([0.0, 0.0, shoulder_height, 1.0])
        self.top_head_position = np.array([0.0, 0.0, total_height, 1.0])
        self.elbow_position = self.shoulder_position - np.array([0.0, 0.0, upper_arm_length, 0.0])
        self.wrist_position = self.elbow_position - np.array([0.0, 0.0, lower_arm_length, 0.0])
        self.finger_position = self.wrist_position - np.array([0.0, 0.0, hand_length, 0.0])
        self.knee_position = np.array([0.0, 0.0, knee_height, 1.0])
        self.ankle_position = np.array([0.0, 0.0, ankle_height, 1.0])
        self.toes_position = np.array([0.0, 0.0, foot_length * 0.5, 0.0])  # TODO: check if foot length is ok to measure
        # Toes is sketchy because of the axis rotation

        self.define_inertial_table()

    def __getitem__(self, segment_name: SegmentName) -> InertiaParameters:
        """
        The inertia paremeters for a particular segment
        Parameters
        ----------
        segment_name
            The name of the segment
        """
        return self.inertial_table[self.sex][segment_name]
