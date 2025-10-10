from enum import Enum
import numpy as np

from ..components.generic.rigidbody.inertia_parameters import InertiaParameters
from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.generic.biomechanical_model import BiomechanicalModel
from ..components.generic.rigidbody.segment import Segment
from ..components.generic.rigidbody.segment_coordinate_system import SegmentCoordinateSystem
from ..components.generic.rigidbody.mesh import Mesh
from ..utils.protocols import Data
from ..utils.enums import Translations, Rotations


# TODO: Add a figure to the documentation to illustrate the segment definitions
# And create a simple table with all the values at the same place


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
        self.total_height = None
        self.pelvis_height = None
        self.trunk_length = None
        self.hand_length = None
        self.lower_arm_length = None
        self.upper_arm_length = None
        self.shoulder_width = None
        self.thigh_length = None
        self.shank_length = None
        self.hip_width = None
        self.foot_length = None

        # The following attributes will be set by get_joint_position_from_measurements
        self.pelvis_position = None
        self.neck_position = None
        self.top_head_position = None
        self.right_shoulder_position = None
        self.right_elbow_position = None
        self.right_wrist_position = None
        self.right_finger_position = None
        self.left_shoulder_position = None
        self.left_elbow_position = None
        self.left_wrist_position = None
        self.left_finger_position = None
        self.right_hip_position = None
        self.right_knee_position = None
        self.right_ankle_position = None
        self.right_toe_position = None
        self.left_hip_position = None
        self.left_knee_position = None
        self.left_ankle_position = None
        self.left_toe_position = None

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
                        (1 - 0.5002), start=self.neck_position, end=self.top_head_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0694 * self.total_mass,
                        coef=(0.303, 0.315, 0.261),
                        start=self.top_head_position,
                        end=self.neck_position,
                    ),
                ),
                SegmentName.TRUNK: InertiaParameters(
                    mass=lambda m, bio: 0.4346 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.5138), start=self.pelvis_position, end=self.neck_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.4346 * self.total_mass,
                        coef=(0.328, 0.306, 0.169),
                        start=self.neck_position,
                        end=self.pelvis_position,
                    ),
                ),
                SegmentName.UPPER_ARM: InertiaParameters(
                    mass=lambda m, bio: 0.0271 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.5772), start=self.right_shoulder_position, end=self.right_elbow_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0271 * self.total_mass,
                        coef=(0.285, 0.269, 0.158),
                        start=self.right_shoulder_position,
                        end=self.right_elbow_position,
                    ),
                ),
                SegmentName.LOWER_ARM: InertiaParameters(
                    mass=lambda m, bio: 0.0162 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.4574), start=self.right_elbow_position, end=self.right_wrist_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0162 * self.total_mass,
                        coef=(0.276, 0.265, 0.121),
                        start=self.right_elbow_position,
                        end=self.right_wrist_position,
                    ),
                ),
                SegmentName.HAND: InertiaParameters(
                    mass=lambda m, bio: 0.0061 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.3624, start=self.right_wrist_position, end=self.right_finger_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0061 * self.total_mass,
                        coef=(0.288, 0.235, 0.184),
                        start=self.right_wrist_position,
                        end=self.right_finger_position,
                    ),
                ),
                SegmentName.THIGH: InertiaParameters(
                    mass=lambda m, bio: 0.1416 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4095, start=self.right_hip_position, end=self.right_knee_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.1416 * self.total_mass,
                        coef=(0.329, 0.329, 0.149),
                        start=self.right_hip_position,
                        end=self.right_knee_position,
                    ),
                ),
                SegmentName.SHANK: InertiaParameters(
                    mass=lambda m, bio: 0.0433 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4459, start=self.right_knee_position, end=self.right_ankle_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0433 * self.total_mass,
                        coef=(0.255, 0.249, 0.103),
                        start=self.right_knee_position,
                        end=self.right_ankle_position,
                    ),
                ),
                SegmentName.FOOT: InertiaParameters(
                    mass=lambda m, bio: 0.0137 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4415, start=self.right_ankle_position, end=self.right_toe_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0137 * self.total_mass,
                        coef=(0.257, 0.245, 0.124),
                        start=self.right_ankle_position,
                        end=self.right_toe_position,
                    ),
                ),
            },
            Sex.FEMALE: {
                SegmentName.HEAD: InertiaParameters(
                    mass=lambda m, bio: 0.0669 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.4841), start=self.neck_position, end=self.top_head_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0669 * self.total_mass,
                        coef=(0.271, 0.295, 0.261),
                        start=self.top_head_position,
                        end=self.neck_position,
                    ),
                ),
                SegmentName.TRUNK: InertiaParameters(
                    mass=lambda m, bio: 0.4257 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.4964), start=self.pelvis_position, end=self.neck_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.4257 * self.total_mass,
                        coef=(0.307, 0.292, 0.147),
                        start=self.neck_position,
                        end=self.pelvis_position,
                    ),
                ),
                SegmentName.UPPER_ARM: InertiaParameters(
                    mass=lambda m, bio: 0.0255 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.5754), start=self.right_shoulder_position, end=self.right_elbow_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0255 * self.total_mass,
                        coef=(0.278, 0.260, 0.148),
                        start=self.right_shoulder_position,
                        end=self.right_elbow_position,
                    ),
                ),
                SegmentName.LOWER_ARM: InertiaParameters(
                    mass=lambda m, bio: 0.0138 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        (1 - 0.4559), start=self.right_elbow_position, end=self.right_wrist_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0138 * self.total_mass,
                        coef=(0.261, 0.257, 0.094),
                        start=self.right_elbow_position,
                        end=self.right_wrist_position,
                    ),
                ),
                SegmentName.HAND: InertiaParameters(
                    mass=lambda m, bio: 0.0056 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.3427, start=self.right_wrist_position, end=self.right_finger_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0056 * self.total_mass,
                        coef=(0.244, 0.208, 0.184),
                        start=self.right_wrist_position,
                        end=self.right_finger_position,
                    ),
                ),
                SegmentName.THIGH: InertiaParameters(
                    mass=lambda m, bio: 0.1478 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.3612, start=self.right_hip_position, end=self.right_knee_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.1478 * self.total_mass,
                        coef=(0.369, 0.364, 0.162),
                        start=self.right_hip_position,
                        end=self.right_knee_position,
                    ),
                ),
                SegmentName.SHANK: InertiaParameters(
                    mass=lambda m, bio: 0.0481 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4416, start=self.right_knee_position, end=self.right_ankle_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0481 * self.total_mass,
                        coef=(0.271, 0.267, 0.093),
                        start=self.right_knee_position,
                        end=self.right_ankle_position,
                    ),
                ),
                SegmentName.FOOT: InertiaParameters(
                    mass=lambda m, bio: 0.0129 * self.total_mass,
                    center_of_mass=lambda m, bio: point_on_vector_in_local(
                        0.4014, start=self.right_ankle_position, end=self.right_toe_position
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0129 * self.total_mass,
                        coef=(0.299, 0.279, 0.124),
                        start=self.right_ankle_position,
                        end=self.right_toe_position,
                    ),
                ),
            },
        }

    def get_joint_position_from_measurements(self) -> None:

        # Upper body
        self.pelvis_position = np.array([0.0, 0.0, self.pelvis_height, 1.0])
        self.neck_position = np.array([0.0, 0.0, self.pelvis_height + self.trunk_length, 1.0])
        self.top_head_position = np.array([0.0, 0.0, self.total_height, 1.0])

        # Right arm
        self.right_shoulder_position = np.array(
            [0.0, -self.shoulder_width / 2, self.pelvis_height + self.trunk_length, 1.0]
        )
        self.right_elbow_position = np.array(
            [0.0, -self.shoulder_width / 2, self.pelvis_height + self.trunk_length - self.upper_arm_length, 1.0]
        )
        self.right_wrist_position = np.array(
            [
                0.0,
                -self.shoulder_width / 2,
                self.pelvis_height + self.trunk_length - self.upper_arm_length - self.lower_arm_length,
                1.0,
            ]
        )
        self.right_finger_position = np.array(
            [
                0.0,
                -self.shoulder_width / 2,
                self.pelvis_height
                + self.trunk_length
                - self.upper_arm_length
                - self.lower_arm_length
                - self.hand_length,
                1.0,
            ]
        )

        # Left arm
        self.left_shoulder_position = np.array(
            [0.0, self.shoulder_width / 2, self.pelvis_height + self.trunk_length, 1.0]
        )
        self.left_elbow_position = np.array(
            [0.0, self.shoulder_width / 2, self.pelvis_height + self.trunk_length - self.upper_arm_length, 1.0]
        )
        self.left_wrist_position = np.array(
            [
                0.0,
                self.shoulder_width / 2,
                self.pelvis_height + self.trunk_length - self.upper_arm_length - self.lower_arm_length,
                1.0,
            ]
        )
        self.left_finger_position = np.array(
            [
                0.0,
                self.shoulder_width / 2,
                self.pelvis_height
                + self.trunk_length
                - self.upper_arm_length
                - self.lower_arm_length
                - self.hand_length,
                1.0,
            ]
        )

        # Right leg
        self.right_hip_position = np.array([0, -self.hip_width / 2, self.pelvis_height, 1])
        self.right_knee_position = np.array([0, -self.hip_width / 2, self.pelvis_height - self.thigh_length, 1])
        self.right_ankle_position = np.array(
            [0, -self.hip_width / 2, self.pelvis_height - self.thigh_length - self.shank_length, 1]
        )
        # Toes position is false due to the heel being behind the ankle
        self.right_toe_position = np.array(
            [self.foot_length, -self.hip_width / 2, self.pelvis_height - self.thigh_length - self.shank_length, 1]
        )

        # Left leg
        self.left_hip_position = np.array([0, self.hip_width / 2, self.pelvis_height, 1])
        self.left_knee_position = np.array([0, self.hip_width / 2, self.pelvis_height - self.thigh_length, 1])
        self.left_ankle_position = np.array(
            [0, self.hip_width / 2, self.pelvis_height - self.thigh_length - self.shank_length, 1]
        )
        # Toes position is false due to the heel being behind the ankle
        self.left_toe_position = np.array(
            [self.foot_length, self.hip_width / 2, self.pelvis_height - self.thigh_length - self.shank_length, 1]
        )

    def from_data(self, data: Data):
        """
        Create the De Leva table from a Data object containing the measurements of the subject.
        This is useful if you want to measure segment length using the markers from a static trial.
        Only the z (3rd) components are used to measure segment length, except for the foot length.
        """

        self.total_height = float(data.values["TOP_HEAD"][2])
        self.pelvis_height = float(data.values["PELVIS"][2])
        self.trunk_length = float(data.values["SHOULDER"][2] - data.values["PELVIS"][2])
        self.hand_length = float(data.values["WRIST"][2] - data.values["FINGER"][2])
        self.lower_arm_length = float(data.values["ELBOW"][2] - data.values["WRIST"][2])
        self.upper_arm_length = float(data.values["SHOULDER"][2] - data.values["ELBOW"][2])
        self.shoulder_width = 0  # Not defined
        self.thigh_length = float(data.values["PELVIS"][2] - data.values["KNEE"][2])
        self.shank_length = float(data.values["KNEE"][2] - data.values["ANKLE"][2])
        self.hip_width = 0  # Not defined
        self.foot_length = float(np.linalg.norm(data.values["HEEL"] - data.values["TOE"]))

        self.get_joint_position_from_measurements()
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
        """
        Create the De Leva table from a manual measurements of the subject.
        """

        # Define some length from measurements
        self.total_height = total_height
        self.pelvis_height = pelvis_height
        self.trunk_length = shoulder_height - pelvis_height
        self.hand_length = (finger_span - wrist_span) / 2
        self.lower_arm_length = (wrist_span - elbow_span) / 2
        self.upper_arm_length = (elbow_span - shoulder_span) / 2
        self.shoulder_width = shoulder_span
        self.thigh_length = pelvis_height - knee_height
        self.shank_length = knee_height - ankle_height
        self.hip_width = hip_width
        self.foot_length = foot_length

        self.get_joint_position_from_measurements()
        self.define_inertial_table()

    def from_height(
        self,
        total_height: float,
    ):
        """
        Create the De Leva table using standard body proportions.
        """

        length_ratios = {
            Sex.MALE: {
                "head_length": 0.2429 / 1.741,
                "trunk_length": 0.6033 / 1.741,
                "upper_arm_length": 0.2817 / 1.741,
                "lower_arm_length": 0.2689 / 1.741,
                "hand_length": 0.1879 / 1.741,
                "thigh_length": 0.4222 / 1.741,
                "shank_length": 0.434 / 1.741,
                "foot_length": 0.2581 / 1.741,
            },
            Sex.FEMALE: {
                "head_length": 0.2437 / 1.735,
                "trunk_length": 0.6148 / 1.735,
                "upper_arm_length": 0.2751 / 1.735,
                "lower_arm_length": 0.2643 / 1.735,
                "hand_length": 0.1701 / 1.735,
                "thigh_length": 0.3685 / 1.735,
                "shank_length": 0.4323 / 1.735,
                "foot_length": 0.2283 / 1.735,
            },
        }

        # Define some length from measurements
        self.total_height = total_height

        self.hip_width = 0
        self.shoulder_width = 0
        self.foot_length = length_ratios[self.sex]["foot_length"] * total_height
        self.shank_length = length_ratios[self.sex]["shank_length"] * total_height
        self.thigh_length = length_ratios[self.sex]["thigh_length"] * total_height
        self.trunk_length = length_ratios[self.sex]["trunk_length"] * total_height
        self.upper_arm_length = length_ratios[self.sex]["upper_arm_length"] * total_height
        self.lower_arm_length = length_ratios[self.sex]["lower_arm_length"] * total_height
        self.hand_length = length_ratios[self.sex]["hand_length"] * total_height

        # Approximate pelvis height as the sum of shank and thigh lengths
        self.pelvis_height = self.shank_length + self.thigh_length

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
        """

        # Define the position of the joints
        self.get_joint_position_from_measurements()

        # Generate the personalized kinematic model
        model = BiomechanicalModel()

        model.add_segment(
            Segment(
                name="TRUNK",
                translations=Translations.XYZ,
                rotations=Rotations.XYZ,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.TRUNK],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.pelvis_position,
                ),
                mesh=Mesh((lambda m, model: self.pelvis_position, lambda m, model: self.neck_position), is_local=False),
            )
        )

        model.add_segment(
            Segment(
                name="HEAD",
                parent_name="TRUNK",
                translations=Translations.NONE,
                rotations=Rotations.XYZ,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.HEAD],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.neck_position,
                ),
                mesh=Mesh(
                    (lambda m, model: self.neck_position, lambda m, model: self.top_head_position), is_local=False
                ),
            )
        )

        model.add_segment(
            Segment(
                name="R_THIGH",
                parent_name="TRUNK",
                rotations=Rotations.XY,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.THIGH],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.right_hip_position,
                ),
                mesh=Mesh(
                    (lambda m, model: self.right_hip_position, lambda m, model: self.right_knee_position),
                    is_local=False,
                ),
            )
        )

        model.add_segment(
            Segment(
                name="R_SHANK",
                parent_name="R_THIGH",
                rotations=Rotations.Y,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.SHANK],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.right_knee_position,
                ),
                mesh=Mesh(
                    (lambda m, model: self.right_knee_position, lambda m, model: self.right_ankle_position),
                    is_local=False,
                ),
            )
        )

        model.add_segment(
            Segment(
                name="R_FOOT",
                parent_name="R_SHANK",
                rotations=Rotations.Y,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.FOOT],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.right_ankle_position,
                ),
                mesh=Mesh(
                    (lambda m, model: np.array([0, 0, 0, 1]), lambda m, model: np.array([0, self.foot_length, 0, 1])),
                    is_local=True,
                ),
            )
        )

        model.add_segment(
            Segment(
                name="L_THIGH",
                parent_name="TRUNK",
                rotations=Rotations.XY,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.THIGH],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.left_hip_position,
                ),
                mesh=Mesh(
                    (lambda m, model: self.left_hip_position, lambda m, model: self.left_knee_position), is_local=False
                ),
            )
        )

        model.add_segment(
            Segment(
                name="L_SHANK",
                parent_name="L_THIGH",
                rotations=Rotations.Y,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.SHANK],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.left_knee_position,
                ),
                mesh=Mesh(
                    (lambda m, model: self.left_knee_position, lambda m, model: self.left_ankle_position),
                    is_local=False,
                ),
            )
        )

        model.add_segment(
            Segment(
                name="L_FOOT",
                parent_name="L_SHANK",
                rotations=Rotations.Y,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.FOOT],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.left_ankle_position,
                ),
                mesh=Mesh(
                    (lambda m, model: np.array([0, 0, 0, 1]), lambda m, model: np.array([0, self.foot_length, 0, 1])),
                    is_local=True,
                ),
            )
        )

        model.add_segment(
            Segment(
                name="R_UPPER_ARM",
                parent_name="TRUNK",
                rotations=Rotations.ZX,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.UPPER_ARM],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.right_shoulder_position,
                ),
                mesh=Mesh(
                    (lambda m, model: self.right_shoulder_position, lambda m, model: self.right_elbow_position),
                    is_local=False,
                ),
            )
        )

        model.add_segment(
            Segment(
                name="R_LOWER_ARM",
                parent_name="R_UPPER_ARM",
                rotations=Rotations.Y,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.LOWER_ARM],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.right_elbow_position,
                ),
                mesh=Mesh(
                    (lambda m, model: self.right_elbow_position, lambda m, model: self.right_wrist_position),
                    is_local=False,
                ),
            )
        )

        model.add_segment(
            Segment(
                name="R_HAND",
                parent_name="R_LOWER_ARM",
                rotations=Rotations.Y,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.HAND],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.right_wrist_position,
                ),
                mesh=Mesh(
                    (lambda m, model: np.array([0, 0, 0, 1]), lambda m, model: np.array([0, 0, -self.hand_length, 1])),
                    is_local=True,
                ),
            )
        )

        model.add_segment(
            Segment(
                name="L_UPPER_ARM",
                parent_name="TRUNK",
                rotations=Rotations.ZX,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.UPPER_ARM],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.left_shoulder_position,
                ),
                mesh=Mesh(
                    (lambda m, model: self.left_shoulder_position, lambda m, model: self.left_elbow_position),
                    is_local=False,
                ),
            )
        )

        model.add_segment(
            Segment(
                name="L_LOWER_ARM",
                parent_name="L_UPPER_ARM",
                rotations=Rotations.Y,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.LOWER_ARM],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.left_elbow_position,
                ),
                mesh=Mesh(
                    (lambda m, model: self.left_elbow_position, lambda m, model: self.left_wrist_position),
                    is_local=False,
                ),
            )
        )

        model.add_segment(
            Segment(
                name="L_HAND",
                parent_name="L_LOWER_ARM",
                rotations=Rotations.Y,
                inertia_parameters=self.inertial_table[self.sex][SegmentName.HAND],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, model: self.left_wrist_position,
                ),
                mesh=Mesh(
                    (lambda m, model: np.array([0, 0, 0, 1]), lambda m, model: np.array([0, 0, -self.hand_length, 1])),
                    is_local=True,
                ),
            )
        )

        model_real = model.to_real({})
        return model_real
