"""
In this example, we will build a full-body model as close as possible to the Plug-in Gait.
Please note that we are building a model with rigid segments linked together through idealized joints. This approach
contrasts with the original Plug-in Gait where the segments are identified at each frame of the experimental trial, thus
 the segments are not rigidified.

Credits: This example is adapted from https://github.com/cr-crme/walkerKinematicReconstruction/blob/main/walker/plugin_gait.py
"""

import numpy as np
from biobuddy import (
    Axis,
    BiomechanicalModel,
    BiomechanicalModelReal,
    SegmentCoordinateSystem,
    InertiaParameters,
    Mesh,
    Segment,
    Marker,
    Translations,
    Rotations,
    C3dData,
)


# Generic functions
def chord_function(
    offset: np.ndarray,
    known_center_of_rotation: np.ndarray,
    center_of_rotation_marker: np.ndarray,
    plane_marker: np.ndarray,
    direction: int = 1,
) -> np.ndarray:

    n_frames = center_of_rotation_marker.shape[1]

    # Create a coordinate system from the markers
    axis1 = plane_marker[:3, :] - known_center_of_rotation[:3, :]
    axis2 = center_of_rotation_marker[:3, :] - known_center_of_rotation[:3, :]
    axis3 = np.cross(axis1, axis2, axis=0)
    axis1 = np.cross(axis2, axis3, axis=0)
    axis1 /= np.linalg.norm(axis1, axis=0)
    axis2 /= np.linalg.norm(axis2, axis=0)
    axis3 /= np.linalg.norm(axis3, axis=0)
    rt = np.identity(4)
    rt = np.repeat(rt, n_frames, axis=1).reshape((4, 4, n_frames))
    rt[:3, 0, :] = axis1
    rt[:3, 1, :] = axis2
    rt[:3, 2, :] = axis3
    rt[:3, 3, :] = known_center_of_rotation[:3, :]

    # The point of interest is the chord from center_of_rotation_marker that has length 'offset' assuming
    # the diameter is the distance between center_of_rotation_marker and known_center_of_rotation.
    # To compute this, project in the rt knowing that by construction, known_center_of_rotation is at 0, 0, 0
    # and center_of_rotation_marker is at a diameter length on y
    diameter = np.linalg.norm(known_center_of_rotation[:3, :] - center_of_rotation_marker[:3, :], axis=0)
    x = offset * direction * np.sqrt(diameter**2 - offset**2) / diameter
    y = (diameter**2 - offset**2) / diameter

    # project the computed point in the global reference frame
    vect = np.concatenate((x[np.newaxis, :], y[np.newaxis, :], np.zeros((1, n_frames)), np.ones((1, n_frames))))

    def rt_times_vect(m1, m2):
        return np.einsum("ijk,jk->ik", m1, m2)

    return rt_times_vect(rt, vect)


def point_on_vector(coef: float, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Computes the 3d position of a point using this equation: start + coef * (end - start)

    Parameters
    ----------
    coef
        The coefficient of the length of the segment to use. It is given from the starting point
    start
        The starting point of the segment
    end
        The end point of the segment

    Returns
    -------
    The 3d position of the point
    """

    return start + coef * (end - start)


def project_point_on_line(start_line: np.ndarray, end_line: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Project a point on a line defined by to points (start_line and end_line)

    Parameters
    ----------
    start_line
        The starting point of the line
    end_line
        The ending point of the line
    point
        The point to project

    Returns
    -------
    The projected point
    -------

    """

    def dot(v1, v2):
        return np.einsum("ij,ij->j", v1, v2)

    sp = (point - start_line)[:3, :]
    line = (end_line - start_line)[:3, :]
    return start_line[:3, :] + dot(sp, line) / dot(line, line) * line


class PlugInGait(BiomechanicalModel):
    def __init__(
        self,
        body_mass: float,
        shoulder_offset: float = None,
        elbow_width: float = None,
        wrist_width: float = None,
        hand_thickness: float = None,
        leg_length: dict[str, float] = None,
        ankle_width: float = None,
        include_upper_body: bool = True,
    ):

        super().__init__()
        self.body_mass = body_mass
        self.shoulder_offset = shoulder_offset
        self.elbow_width = elbow_width
        self.wrist_width = wrist_width
        self.hand_thickness = hand_thickness
        self.leg_length = leg_length
        self.ankle_width = ankle_width
        self.include_upper_body = include_upper_body

    def get_lumbar_5(self, m: dict, bio: BiomechanicalModelReal):
        right_hip = self.get_hip_joint_center(m, bio, "R")
        left_hip = self.get_hip_joint_center(m, bio, "L")
        return np.nanmean((left_hip, right_hip), axis=0) + np.array((0.0, 0.0, 0.828, 0))[:, np.newaxis] * np.repeat(
            np.linalg.norm(left_hip - right_hip, axis=0)[np.newaxis, :], 4, axis=0
        )

    def get_pelvis_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        return (m["LPSI"] + m["RPSI"] + m["LASI"] + m["RASI"]) / 4

    def get_pelvis_center_of_mass(self, m: dict, bio: BiomechanicalModelReal) -> np.ndarray:
        """
        This computes the center of mass of the thorax

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        """
        right_hip = self.get_hip_joint_center(m, bio, "R")
        left_hip = self.get_hip_joint_center(m, bio, "L")
        p = self.get_pelvis_joint_center(m, bio)  # Make sure the center of mass is symmetric
        p[2, :] += 0.925 * (self.get_lumbar_5(m, bio) - np.nanmean((left_hip, right_hip), axis=0))[2, :]
        return p

    def get_thorax_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        return m["CLAV"]

    def get_thorax_center_of_mass(self, m: dict, bio: BiomechanicalModelReal) -> np.ndarray:
        """
        This computes the center of mass of the thorax

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        """
        com = point_on_vector(0.63, start=m["C7"], end=self.get_lumbar_5(m, bio))
        com[0, :] = self.get_thorax_joint_center(m, bio)[0, :]  # Make sur the center of mass is symmetric
        return com

    def get_head_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        return (m["LFHD"] + m["RFHD"]) / 2

    def get_head_center_of_mass(self, m: dict, bio: BiomechanicalModelReal):
        return point_on_vector(
            0.52,
            start=(m["LFHD"] + m["RFHD"]) / 2,
            end=(m["LBHD"] + m["RBHD"]) / 2,
        )

    def get_humerus_joint_center(self, m: dict, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        This is the implementation of the 'Shoulder joint center, p.69'.

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The position of the origin of the humerus
        """

        thorax_origin = bio.segments["Thorax"].segment_coordinate_system.scs[:, 3, :]
        thorax_x_axis = bio.segments["Thorax"].segment_coordinate_system.scs[:, 0, :]
        thorax_to_sho_axis = m[f"{side}SHO"] - thorax_origin
        shoulder_wand = np.cross(thorax_to_sho_axis[:3, :], thorax_x_axis[:3, :], axis=0)
        shoulder_offset = (
            self.shoulder_offset
            if self.shoulder_offset is not None
            else 0.17 * (m[f"{side}SHO"] - m[f"{side}ELB"])[2, :]
        )

        return chord_function(shoulder_offset, thorax_origin, m[f"{side}SHO"], shoulder_wand)

    def get_elbow_joint_center(self, m: dict, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the joint center of

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The position of the origin of the elbow
        """

        shoulder_origin = self.get_humerus_joint_center(m, bio, side)
        elbow_marker = m[f"{side}ELB"]
        wrist_marker = (m[f"{side}WRA"] + m[f"{side}WRB"]) / 2

        elbow_width = (
            self.elbow_width
            if self.elbow_width is not None
            else np.linalg.norm(m[f"{side}WRA"][:3, :] - m[f"{side}WRB"][:3, :], axis=0) * 1.15
        )
        elbow_offset = elbow_width / 2

        return chord_function(elbow_offset, shoulder_origin, elbow_marker, wrist_marker)

    def get_wrist_joint_center(self, m, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the segment coordinate system of the wrist. If wrist_width is not provided 2cm is assumed

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The SCS of the wrist
        """

        elbow_center = self.get_elbow_joint_center(m, bio, side)
        wrist_bar_center = project_point_on_line(m[f"{side}WRA"], m[f"{side}WRB"], elbow_center)
        offset_axis = np.cross(
            m[f"{side}WRA"][:3, :] - m[f"{side}WRB"][:3, :], elbow_center[:3, :] - wrist_bar_center, axis=0
        )
        offset_axis /= np.linalg.norm(offset_axis, axis=0)

        offset = (offset_axis * (self.wrist_width / 2)) if self.wrist_width is not None else 0.02 / 2
        return np.concatenate((wrist_bar_center + offset, np.ones((1, wrist_bar_center.shape[1]))))

    def get_hand_center(self, m, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the origin of the hand. If hand_thickness if not provided, it is assumed to be 1cm

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """

        elbow_center = self.get_elbow_joint_center(m, bio, side)
        wrist_joint = self.get_wrist_joint_center(m, bio, side)
        fin_marker = m[f"{side}FIN"]
        hand_offset = np.repeat(self.hand_thickness / 2 if self.hand_thickness else 0.01 / 2, fin_marker.shape[1])
        wrist_bar_center = project_point_on_line(m[f"{side}WRA"], m[f"{side}WRB"], elbow_center)

        return chord_function(hand_offset, wrist_joint, fin_marker, wrist_bar_center)

    def get_legs_length(self, m, bio: BiomechanicalModelReal) -> dict[str, float]:
        # TODO: Verify 95% makes sense  # @pariterre ?
        return {
            "R": self.leg_length["R"] if self.leg_length else np.nanmean(m[f"RASI"][2, :]) * 0.95,
            "L": self.leg_length["L"] if self.leg_length else np.nanmean(m[f"LASI"][2, :]) * 0.95,
        }

    def get_hip_joint_center(self, m, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the hip joint center. The LegLength is not provided, the height of the TROC is used (therefore assuming
        the subject is standing upright during the static trial)

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """

        inter_asis = np.nanmean(np.linalg.norm(m["LASI"][:3, :] - m["RASI"][:3, :], axis=0))
        legs_length_value = self.get_legs_length(m, bio)
        mean_legs_length = np.nanmean((legs_length_value["R"], legs_length_value["L"]))
        asis_troc_dist = 0.1288 * legs_length_value[side] - 0.04856

        c = mean_legs_length * 0.115 - 0.0153
        aa = inter_asis / 2
        theta = 0.5
        beta = 0.314
        x = c * np.cos(theta) * np.sin(beta) - asis_troc_dist * np.cos(beta)
        y = -(c * np.sin(theta) - aa)
        z = -c * np.cos(theta) * np.cos(beta) - asis_troc_dist * np.sin(beta)
        return m[f"{side}ASI"] + np.array((x, y, z, 0))[:, np.newaxis]

    def get_knee_axis(self, side) -> Axis:
        """
        Define the knee axis

        Parameters
        ----------
        side
            If the markers are from the right ("R") or left ("L") side
        """
        if side == "R":
            return Axis(Axis.Name.Y, start=f"{side}KNE", end=f"{side}KNM")
        elif side == "L":
            return Axis(Axis.Name.Y, start=f"{side}KNM", end=f"{side}KNE")
        else:
            raise ValueError("side should be 'R' or 'L'")

    def get_knee_joint_center(self, m, bio: BiomechanicalModelReal, side) -> np.ndarray:
        """
        Compute the knee joint center. This is a simplified version since the KNM exists

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """
        return (m[f"{side}KNM"] + m[f"{side}KNE"]) / 2

    def get_ankle_joint_center(self, m, bio: BiomechanicalModelReal, side) -> np.ndarray:
        """
        Compute the ankle joint center. This is a simplified version sie ANKM exists

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """

        return (m[f"{side}ANK"] + m[f"{side}ANKM"]) / 2

    def build_plug_in_gait(self) -> BiomechanicalModel:
        """
        Actually building the kinematic model.
        This is the implementation of the Plugin Gait (from Plug-in Gait Reference Guide
        https://docs.vicon.com/display/Nexus212/PDF+downloads+for+Vicon+Nexus)

        Since more markers are used in our version (namely Knee medial and ankle medial), the KJC and AJC were
        simplified to be the mean of these markers with their respective lateral markers. Hence, 'ankle_width'
        is no more useful
        """
        # TODO: @pariterre, could you document the following lines directly in the associated functions ?
        # Pelvis: verified, The radii of gyration were computed using InterHip normalisation
        # Thorax: verified
        # Head: verified
        # Humerus: verified
        # Radius: verified
        # Hand: Moved the hand joint center to WJC
        # Femur: verified
        # Knee: Used mid-point of 'KNM' and 'KNE' as KJC
        # Ankle: As for knee, we have access to a much easier medial marker (ANKM), so it was used instead

        self.add_segment(
            Segment(
                name="Pelvis",
                translations=Translations.XYZ,
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=self.get_pelvis_joint_center,
                    first_axis=Axis(name=Axis.Name.X, start=lambda m, bio: (m["LPSI"] + m["RPSI"]) / 2, end="RASI"),
                    second_axis=Axis(name=Axis.Name.Y, start="RASI", end="LASI"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("LPSI", "RPSI", "RASI", "LASI", "LPSI")),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.142 * self.body_mass,
                    center_of_mass=self.get_pelvis_center_of_mass,
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.142 * self.body_mass,
                        coef=(0.31, 0.31, 0.31),
                        start=self.get_pelvis_joint_center(m, bio),
                        end=self.get_pelvis_center_of_mass(m, bio),
                    ),
                ),
            )
        )

        # self.segments["Pelvis"].add_marker("Pelvis", "SACR", is_technical=False, is_anatomical=True)  # @pariterre: remove ?
        self.segments["Pelvis"].add_marker(Marker("LPSI", is_technical=True, is_anatomical=True))
        self.segments["Pelvis"].add_marker(Marker("RPSI", is_technical=True, is_anatomical=True))
        self.segments["Pelvis"].add_marker(Marker("LASI", is_technical=True, is_anatomical=True))
        self.segments["Pelvis"].add_marker(Marker("RASI", is_technical=True, is_anatomical=True))

        self.add_segment(
            Segment(
                name="Thorax",
                parent_name="Pelvis",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=self.get_thorax_joint_center,
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: (m["T10"] + m["STRN"]) / 2,
                        end=lambda m, bio: (m["C7"] + m["CLAV"]) / 2,
                    ),
                    second_axis=Axis(
                        Axis.Name.X,
                        start=lambda m, bio: (m["T10"] + m["C7"]) / 2,
                        end=lambda m, bio: (m["STRN"] + m["CLAV"]) / 2,
                    ),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("T10", "C7", "CLAV", "STRN", "T10")),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.355 * self.body_mass,
                    center_of_mass=self.get_thorax_center_of_mass,
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.355 * self.body_mass,
                        coef=(0.31, 0.31, 0.31),
                        start=m["C7"],
                        end=self.get_lumbar_5(m, bio),
                    ),
                ),
            )
        )
        self.segments["Thorax"].add_marker(Marker("T10", is_technical=True, is_anatomical=True))
        self.segments["Thorax"].add_marker(Marker("C7", is_technical=True, is_anatomical=True))
        self.segments["Thorax"].add_marker(Marker("STRN", is_technical=True, is_anatomical=True))
        self.segments["Thorax"].add_marker(Marker("CLAV", is_technical=True, is_anatomical=True))
        self.segments["Thorax"].add_marker(Marker("RBAK", is_technical=False, is_anatomical=False))

        self.add_segment(
            Segment(
                name="Head",
                parent_name="Thorax",
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=self.get_head_joint_center,
                    first_axis=Axis(
                        Axis.Name.X,
                        start=lambda m, bio: (m["LBHD"] + m["RBHD"]) / 2,
                        end=lambda m, bio: (m["LFHD"] + m["RFHD"]) / 2,
                    ),
                    second_axis=Axis(Axis.Name.Y, start="RFHD", end="LFHD"),
                    axis_to_keep=Axis.Name.X,
                ),
                mesh=Mesh(("LBHD", "RBHD", "RFHD", "LFHD", "LBHD")),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.081 * self.body_mass,
                    center_of_mass=self.get_head_center_of_mass,
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.081 * self.body_mass,
                        coef=(0.495, 0.495, 0.495),
                        start=self.get_head_center_of_mass(m, bio),
                        end=m["C7"][:3, :],
                    ),
                ),
            )
        )
        self.segments["Head"].add_marker(Marker("LBHD", is_technical=False, is_anatomical=True))
        self.segments["Head"].add_marker(Marker("RBHD", is_technical=False, is_anatomical=True))
        self.segments["Head"].add_marker(Marker("LFHD", is_technical=False, is_anatomical=True))
        self.segments["Head"].add_marker(Marker("RFHD", is_technical=False, is_anatomical=True))

        self.add_segment(
            Segment(
                name="RHumerus",
                parent_name="Thorax",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_humerus_joint_center(m, bio, "R"),
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: self.get_elbow_joint_center(m, bio, "R"),
                        end=lambda m, bio: self.get_humerus_joint_center(m, bio, "R"),
                    ),
                    second_axis=Axis(
                        Axis.Name.X,
                        start=lambda m, bio: self.get_elbow_joint_center(m, bio, "R"),
                        end=lambda m, bio: self.get_wrist_joint_center(m, bio, "R"),
                    ),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(
                    (
                        lambda m, bio: self.get_humerus_joint_center(m, bio, "R"),
                        lambda m, bio: self.get_elbow_joint_center(m, bio, "R"),
                    )
                ),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.028 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.564,
                        start=self.get_humerus_joint_center(m, bio, "R"),
                        end=self.get_elbow_joint_center(m, bio, "R"),
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.028 * self.body_mass,
                        coef=(0.322, 0.322, 0),
                        start=self.get_humerus_joint_center(m, bio, "R"),
                        end=self.get_elbow_joint_center(m, bio, "R"),
                    ),
                ),
            )
        )
        self.segments["RHumerus"].add_marker(Marker("RSHO", is_technical=self.include_upper_body, is_anatomical=True))
        self.segments["RHumerus"].add_marker(Marker("RELB", is_technical=self.include_upper_body, is_anatomical=True))
        self.segments["RHumerus"].add_marker(Marker("RHUM", is_technical=self.include_upper_body, is_anatomical=False))

        self.add_segment(
            Segment(
                name="RRadius",
                parent_name="RHumerus",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_elbow_joint_center(m, bio, "R"),
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: self.get_wrist_joint_center(m, bio, "R"),
                        end=lambda m, bio: self.get_elbow_joint_center(m, bio, "R"),
                    ),
                    second_axis=Axis(
                        Axis.Name.Y,
                        start=lambda m, bio: bio.segments["RHumerus"].segment_coordinate_system.scs[:, 3, :],
                        end=lambda m, bio: bio.segments["RHumerus"].segment_coordinate_system.scs[:, 1, :],
                    ),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(
                    (
                        lambda m, bio: self.get_elbow_joint_center(m, bio, "R"),
                        lambda m, bio: self.get_wrist_joint_center(m, bio, "R"),
                    )
                ),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.016 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.57,
                        start=self.get_elbow_joint_center(m, bio, "R"),
                        end=self.get_wrist_joint_center(m, bio, "R"),
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.016 * self.body_mass,
                        coef=(0.303, 0.303, 0),
                        start=self.get_elbow_joint_center(m, bio, "R"),
                        end=self.get_wrist_joint_center(m, bio, "R"),
                    ),
                ),
            )
        )
        self.segments["RRadius"].add_marker(Marker("RWRB", is_technical=self.include_upper_body, is_anatomical=True))
        self.segments["RRadius"].add_marker(Marker("RWRA", is_technical=self.include_upper_body, is_anatomical=True))

        self.add_segment(
            Segment(
                name="RHand",
                parent_name="RRadius",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_wrist_joint_center(m, bio, "R"),
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: self.get_hand_center(m, bio, "R"),
                        end=lambda m, bio: self.get_wrist_joint_center(m, bio, "R"),
                    ),
                    second_axis=Axis(Axis.Name.Y, start="RWRB", end="RWRA"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh((lambda m, bio: self.get_wrist_joint_center(m, bio, "R"), "RFIN")),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.006 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.6205,
                        start=self.get_wrist_joint_center(m, bio, "R"),
                        end=point_on_vector(1 / 0.75, start=self.get_wrist_joint_center(m, bio, "R"), end=m[f"RFIN"]),
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.006 * self.body_mass,
                        coef=(0.223, 0.223, 0),
                        start=self.get_wrist_joint_center(m, bio, "R"),
                        end=m[f"RFIN"],
                    ),
                ),
            )
        )
        self.segments["RHand"].add_marker(Marker("RFIN", is_technical=self.include_upper_body, is_anatomical=True))

        self.add_segment(
            Segment(
                name="LHumerus",
                parent_name="Thorax",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_humerus_joint_center(m, bio, "L"),
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: self.get_elbow_joint_center(m, bio, "L"),
                        end=lambda m, bio: self.get_humerus_joint_center(m, bio, "L"),
                    ),
                    second_axis=Axis(
                        Axis.Name.X,
                        start=lambda m, bio: self.get_elbow_joint_center(m, bio, "L"),
                        end=lambda m, bio: self.get_wrist_joint_center(m, bio, "L"),
                    ),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(
                    (
                        lambda m, bio: self.get_humerus_joint_center(m, bio, "L"),
                        lambda m, bio: self.get_elbow_joint_center(m, bio, "L"),
                    )
                ),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.028 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.564,
                        start=self.get_humerus_joint_center(m, bio, "L"),
                        end=self.get_elbow_joint_center(m, bio, "L"),
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.028 * self.body_mass,
                        coef=(0.322, 0.322, 0),
                        start=self.get_humerus_joint_center(m, bio, "L"),
                        end=self.get_elbow_joint_center(m, bio, "L"),
                    ),
                ),
            )
        )
        self.segments["LHumerus"].add_marker(Marker("LSHO", is_technical=self.include_upper_body, is_anatomical=True))
        self.segments["LHumerus"].add_marker(Marker("LELB", is_technical=self.include_upper_body, is_anatomical=True))
        # TODO: Add ELBM to define the axis  # @ pariterre ?
        self.segments["LHumerus"].add_marker(Marker("LHUM", is_technical=self.include_upper_body, is_anatomical=False))

        self.add_segment(
            Segment(
                name="LRadius",
                parent_name="LHumerus",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_elbow_joint_center(m, bio, "L"),
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: self.get_wrist_joint_center(m, bio, "L"),
                        end=lambda m, bio: self.get_elbow_joint_center(m, bio, "L"),
                    ),
                    second_axis=Axis(
                        Axis.Name.Y,
                        start=lambda m, bio: bio.segments["LHumerus"].segment_coordinate_system.scs[:, 3, :],
                        end=lambda m, bio: bio.segments["LHumerus"].segment_coordinate_system.scs[:, 1, :],
                    ),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(
                    (
                        lambda m, bio: self.get_elbow_joint_center(m, bio, "L"),
                        lambda m, bio: self.get_wrist_joint_center(m, bio, "L"),
                    )
                ),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.016 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.57,
                        start=self.get_elbow_joint_center(m, bio, "L"),
                        end=self.get_wrist_joint_center(m, bio, "L"),
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.016 * self.body_mass,
                        coef=(0.303, 0.303, 0),
                        start=self.get_elbow_joint_center(m, bio, "L"),
                        end=self.get_wrist_joint_center(m, bio, "L"),
                    ),
                ),
            )
        )
        self.segments["LRadius"].add_marker(Marker("LWRB", is_technical=self.include_upper_body, is_anatomical=True))
        self.segments["LRadius"].add_marker(Marker("LWRA", is_technical=self.include_upper_body, is_anatomical=True))

        self.add_segment(
            Segment(
                name="LHand",
                parent_name="LRadius",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_wrist_joint_center(m, bio, "L"),
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: self.get_hand_center(m, bio, "L"),
                        end=lambda m, bio: self.get_wrist_joint_center(m, bio, "L"),
                    ),
                    second_axis=Axis(Axis.Name.Y, start="LWRB", end="LWRA"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh((lambda m, bio: self.get_wrist_joint_center(m, bio, "L"), "LFIN")),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.006 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.6205, start=self.get_wrist_joint_center(m, bio, "L"), end=m[f"LFIN"]
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.006 * self.body_mass,
                        coef=(0.223, 0.223, 0),
                        start=self.get_wrist_joint_center(m, bio, "L"),
                        end=m[f"LFIN"],
                    ),
                ),
            )
        )
        self.segments["LHand"].add_marker(Marker("LFIN", is_technical=self.include_upper_body, is_anatomical=True))

        self.add_segment(
            Segment(
                name="RFemur",
                parent_name="Pelvis",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_hip_joint_center(m, bio, "R"),
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: self.get_knee_joint_center(m, bio, "R"),
                        end=lambda m, bio: self.get_hip_joint_center(m, bio, "R"),
                    ),
                    second_axis=self.get_knee_axis("R"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(
                    (
                        lambda m, bio: self.get_hip_joint_center(m, bio, "R"),
                        lambda m, bio: self.get_knee_joint_center(m, bio, "R"),
                    )
                ),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.1 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.567, start=self.get_hip_joint_center(m, bio, "R"), end=self.get_knee_joint_center(m, bio, "R")
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.1 * self.body_mass,
                        coef=(0.323, 0.323, 0),
                        start=self.get_hip_joint_center(m, bio, "R"),
                        end=self.get_knee_joint_center(m, bio, "R"),
                    ),
                ),
            )
        )
        self.segments["RFemur"].add_marker(Marker("RTROC", is_technical=True, is_anatomical=True))
        self.segments["RFemur"].add_marker(Marker("RKNE", is_technical=True, is_anatomical=True))
        self.segments["RFemur"].add_marker(Marker("RKNM", is_technical=False, is_anatomical=True))
        self.segments["RFemur"].add_marker(Marker("RTHI", is_technical=True, is_anatomical=False))
        self.segments["RFemur"].add_marker(Marker("RTHID", is_technical=True, is_anatomical=False))

        self.add_segment(
            Segment(
                name="RTibia",
                parent_name="RFemur",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_knee_joint_center(m, bio, "R"),
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: self.get_ankle_joint_center(m, bio, "R"),
                        end=lambda m, bio: self.get_knee_joint_center(m, bio, "R"),
                    ),
                    second_axis=self.get_knee_axis("R"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(
                    (
                        lambda m, bio: self.get_knee_joint_center(m, bio, "R"),
                        lambda m, bio: self.get_ankle_joint_center(m, bio, "R"),
                    )
                ),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.0465 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.567,
                        start=self.get_knee_joint_center(m, bio, "R"),
                        end=self.get_ankle_joint_center(m, bio, "R"),
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0465 * self.body_mass,
                        coef=(0.302, 0.302, 0),
                        start=self.get_knee_joint_center(m, bio, "R"),
                        end=self.get_ankle_joint_center(m, bio, "R"),
                    ),
                ),
            )
        )
        self.segments["RTibia"].add_marker(Marker("RANKM", is_technical=False, is_anatomical=True))
        self.segments["RTibia"].add_marker(Marker("RANK", is_technical=True, is_anatomical=True))
        self.segments["RTibia"].add_marker(Marker("RTIBP", is_technical=True, is_anatomical=False))
        self.segments["RTibia"].add_marker(Marker("RTIB", is_technical=True, is_anatomical=False))
        self.segments["RTibia"].add_marker(Marker("RTIBD", is_technical=True, is_anatomical=False))

        self.add_segment(
            Segment(
                name="RFoot",
                parent_name="RTibia",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_ankle_joint_center(m, bio, "R"),
                    first_axis=self.get_knee_axis("R"),
                    second_axis=Axis(Axis.Name.Z, start="RHEE", end="RTOE"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("RTOE", "R5MH", "RHEE", "RTOE")),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.0145 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.5, start=self.get_ankle_joint_center(m, bio, "R"), end=m[f"RTOE"]
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0145 * self.body_mass,
                        coef=(0.475, 0.475, 0),
                        start=self.get_ankle_joint_center(m, bio, "R"),
                        end=m[f"RTOE"],
                    ),
                ),
            )
        )
        self.segments["RFoot"].add_marker(Marker("RTOE", is_technical=True, is_anatomical=True))
        self.segments["RFoot"].add_marker(Marker("R5MH", is_technical=True, is_anatomical=True))
        self.segments["RFoot"].add_marker(Marker("RHEE", is_technical=True, is_anatomical=True))

        self.add_segment(
            Segment(
                name="LFemur",
                parent_name="Pelvis",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_hip_joint_center(m, bio, "L"),
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: self.get_knee_joint_center(m, bio, "L"),
                        end=lambda m, bio: self.get_hip_joint_center(m, bio, "L"),
                    ),
                    second_axis=self.get_knee_axis("L"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(
                    (
                        lambda m, bio: self.get_hip_joint_center(m, bio, "L"),
                        lambda m, bio: self.get_knee_joint_center(m, bio, "L"),
                    )
                ),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.1 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.567, start=self.get_hip_joint_center(m, bio, "L"), end=self.get_knee_joint_center(m, bio, "L")
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.1 * self.body_mass,
                        coef=(0.323, 0.323, 0),
                        start=self.get_hip_joint_center(m, bio, "L"),
                        end=self.get_knee_joint_center(m, bio, "L"),
                    ),
                ),
            )
        )
        self.segments["LFemur"].add_marker(Marker("LTROC", is_technical=True, is_anatomical=True))
        self.segments["LFemur"].add_marker(Marker("LKNE", is_technical=True, is_anatomical=True))
        self.segments["LFemur"].add_marker(Marker("LKNM", is_technical=False, is_anatomical=True))
        self.segments["LFemur"].add_marker(Marker("LTHI", is_technical=True, is_anatomical=False))
        self.segments["LFemur"].add_marker(Marker("LTHID", is_technical=True, is_anatomical=False))

        self.add_segment(
            Segment(
                name="LTibia",
                parent_name="LFemur",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_knee_joint_center(m, bio, "L"),
                    first_axis=Axis(
                        Axis.Name.Z,
                        start=lambda m, bio: self.get_ankle_joint_center(m, bio, "L"),
                        end=lambda m, bio: self.get_knee_joint_center(m, bio, "L"),
                    ),
                    second_axis=self.get_knee_axis("L"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(
                    (
                        lambda m, bio: self.get_knee_joint_center(m, bio, "L"),
                        lambda m, bio: self.get_ankle_joint_center(m, bio, "L"),
                    )
                ),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.0465 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.567,
                        start=self.get_knee_joint_center(m, bio, "L"),
                        end=self.get_ankle_joint_center(m, bio, "L"),
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0465 * self.body_mass,
                        coef=(0.302, 0.302, 0),
                        start=self.get_knee_joint_center(m, bio, "L"),
                        end=self.get_ankle_joint_center(m, bio, "L"),
                    ),
                ),
            )
        )
        self.segments["LTibia"].add_marker(Marker("LANKM", is_technical=False, is_anatomical=True))
        self.segments["LTibia"].add_marker(Marker("LANK", is_technical=True, is_anatomical=True))
        self.segments["LTibia"].add_marker(Marker("LTIBP", is_technical=True, is_anatomical=False))
        self.segments["LTibia"].add_marker(Marker("LTIB", is_technical=True, is_anatomical=False))
        self.segments["LTibia"].add_marker(Marker("LTIBD", is_technical=True, is_anatomical=False))

        self.add_segment(
            Segment(
                name="LFoot",
                parent_name="LTibia",
                rotations=Rotations.XYZ,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=lambda m, bio: self.get_ankle_joint_center(m, bio, "L"),
                    first_axis=self.get_knee_axis("L"),
                    second_axis=Axis(Axis.Name.Z, start="LHEE", end="LTOE"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("LTOE", "L5MH", "LHEE", "LTOE")),
                inertia_parameters=InertiaParameters(
                    mass=lambda m, bio: 0.0145 * self.body_mass,
                    center_of_mass=lambda m, bio: point_on_vector(
                        0.5, start=self.get_ankle_joint_center(m, bio, "L"), end=m[f"LTOE"]
                    ),
                    inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                        mass=0.0145 * self.body_mass,
                        coef=(0.475, 0.475, 0),
                        start=self.get_ankle_joint_center(m, bio, "L"),
                        end=m[f"LTOE"],
                    ),
                ),
            )
        )
        self.segments["LFoot"].add_marker(Marker("LTOE", is_technical=True, is_anatomical=True))
        self.segments["LFoot"].add_marker(Marker("L5MH", is_technical=True, is_anatomical=True))
        self.segments["LFoot"].add_marker(Marker("LHEE", is_technical=True, is_anatomical=True))


def main():

    # Create the model from user defined dimensions
    plugin_gait_model = PlugInGait(
        body_mass=70.0,  # Example body mass in kg
        shoulder_offset=0.1,  # Example shoulder offset in meters
        elbow_width=0.15,  # Example elbow width in meters
        wrist_width=0.02,  # Example wrist width in meters
        hand_thickness=0.01,  # Example hand thickness in meters
        leg_length={"R": 1.0, "L": 1.0},  # Example leg lengths in meters
        ankle_width=0.1,  # Example ankle width in meters
        include_upper_body=True,
    )
    plugin_gait_model.build_plug_in_gait()

    # Load the static trial
    static_trial_path = "../data/static_plugin_gait.c3d"
    static_data = C3dData(static_trial_path)

    # Evaluate the model to get a BiomechanicalModelReal personalized to the subject
    biomechanical_model_real = plugin_gait_model.to_real(static_data)

    # If you have downloaded pyorerun, you can animate the model to see the result
    # TODO: Charbie -> uncomment this line when my merge_segments branch is merged
    # biomechanical_model_real.animate()


if __name__ == "__main__":
    main()
