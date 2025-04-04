from copy import deepcopy
from enum import Enum
from scipy import optimize

import numpy as np
from ezc3d import c3d

from .. import SegmentCoordinateSystemReal
from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.segment_scaling import SegmentScaling
from ..components.real.rigidbody.segment_real import SegmentReal
from ..components.real.rigidbody.marker_real import MarkerReal
from ..components.real.rigidbody.contact_real import ContactReal
from ..components.real.rigidbody.inertial_measurement_unit_real import InertialMeasurementUnitReal
from ..components.real.rigidbody.mesh_file_real import MeshFileReal
from ..components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ..components.real.muscle.muscle_real import MuscleReal
from ..components.real.muscle.via_point_real import ViaPointReal
from ..utils.named_list import NamedList


class InertialCharacteristics(Enum):
    DE_LEVA = "de_leva"  # TODO


class ScaleTool:
    def __init__(self, personalize_mass_distribution: bool = True, max_marker_movement: float = 0.1):

        self.original_model = BiomechanicalModelReal()
        self.original_model_biorbd = None  # This original model is defined by self.scale
        self.scaled_model = BiomechanicalModelReal()
        self.scaled_model_biorbd = None  # This scaled model is defined later when the segment shape is defined

        self.header = ""
        self.original_mass = None
        self.personalize_mass_distribution = personalize_mass_distribution
        self.max_marker_movement = max_marker_movement
        self.scaling_segments = NamedList[SegmentScaling]()
        self.marker_weightings = {}
        self.warnings = ""

    def scale(self, original_model: BiomechanicalModelReal, static_trial: str, frame_range: range, mass: float):
        """
        Scale the model using the configuration defined in the ScaleTool.

        Parameters
        ----------
        original_model
            The original model to scale to the subjects anthropometry
        static_trial
            The .c3d or .trc file of the static trial to use for the scaling
        frame_range
            The range of frames to consider for the scaling
        mass
            The mass of the subject
        """

        # Initialize the original and scaled models
        self.original_model = original_model
        self.original_model_biorbd = self.original_model.get_biorbd_model

        # Check file format
        if not static_trial.endswith(".c3d"):
            if static_trial.endswith(".trc"):
                raise NotImplementedError(".trc files cannot be read yet.")
            else:
                raise RuntimeError("The static_trial must be a .c3d file in a static posture.")

        # Check the weights
        for marker in self.marker_weightings:
            if self.marker_weightings[marker] < 0:
                raise RuntimeError(f"The weight of marker {marker} is negative. It must be positive.")

        # Check the mass
        if mass <= 0:
            raise RuntimeError(f"The mass of the subject must be positive. The value given is {mass} kg.")

        # Load the c3d file
        c3d_file = c3d(static_trial)
        unit_multiplier = self.check_units(c3d_file)
        marker_names = c3d_file["parameters"]["POINT"]["LABELS"]["value"]
        marker_positions = c3d_file["data"]["points"][:3, :, frame_range] * unit_multiplier

        self.check_that_makers_do_not_move(marker_positions, marker_names)
        self.check_segments()

        self.scale_model_geometrically(marker_positions, marker_names, mass)
        self.scaled_model_biorbd = self.scaled_model.get_biorbd_model

        q_static = self.place_model_in_static_pose(marker_positions, marker_names)
        self.replace_markers_on_segments(q_static)
        self.modify_muscle_parameters()

    def check_units(self, c3d_file: c3d) -> float:
        unit = c3d_file["parameters"]["POINT"]["UNITS"]["value"][0]
        if unit == "mm":
            unit_multiplier = 0.001
        else:
            raise NotImplementedError(
                f"This unit of the marker position in your .c3d static file is {unit}, which is not implemented yet."
            )
        return unit_multiplier

    def check_that_makers_do_not_move(self, marker_positions, marker_names):
        """
        Check that the markers do not move too much in the static trial

        Parameters
        ----------
        marker_positions
            The position of the markers in the static trial (within the frame_range)
        marker_names
            The names of the marker labels in the c3d static file
        """

        if self.max_marker_movement is None:
            return

        else:
            for marker in self.marker_weightings:
                if marker not in marker_names:
                    raise RuntimeError(f"The marker {marker} is not in the c3d file.")
                marker_index = marker_names.index(marker)
                this_marker_position = marker_positions[:, marker_index, :]
                min_position = np.nanmin(this_marker_position, axis=1)
                max_position = np.nanmax(this_marker_position, axis=1)
                if np.linalg.norm(max_position - min_position) > self.max_marker_movement:
                    raise RuntimeError(
                        f"The marker {marker} moves of approximately {np.linalg.norm(max_position - min_position)} m during the static trial, which is above the maximal limit of {self.max_marker_movement} m."
                    )
            return

    def check_segments(self):

        # Check that all segments that bear mass are scaled.
        for segment_name in self.original_model.segments.keys():
            inertia = self.original_model.segments[segment_name].inertia_parameters
            if inertia is not None and inertia.mass > 0.1 and segment_name not in self.scaling_segments.keys():
                raise RuntimeError(
                    f"The segment {segment_name} has a positive mass of {self.original_model.segments[segment_name].inertia_parameters.mass}, but is not defined in the scaling configuration."
                )

        # Check that all scaled segments exist in the original model.
        for segment_name in self.scaling_segments.keys():
            if segment_name not in self.original_model.segments.keys():
                raise RuntimeError(
                    f"The segment {segment_name} has a scaling configuration, but does not exist in the original model."
                )

    def get_scaling_factors_and_masses(
        self,
        marker_positions: np.ndarray,
        marker_names: list[str],
        mass: float,
        original_mass: float,
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:

        scaling_factors = {}
        segment_masses = {}
        total_scaled_mass = 0
        for segment_name in self.scaling_segments.keys():
            # Compute the scale factors
            scaling_factors[segment_name] = self.scaling_segments[segment_name].compute_scaling_factors(
                marker_positions, marker_names, self.original_model_biorbd
            )
            # Get each segment's scaled mass
            if self.personalize_mass_distribution:
                segment_masses[segment_name] = (
                    self.original_model.segments[segment_name].inertia_parameters.mass
                    * scaling_factors[segment_name].to_vector()
                )
            else:
                segment_masses[segment_name] = (
                    self.original_model.segments[segment_name].inertia_parameters.mass * mass / original_mass
                )
            total_scaled_mass += segment_masses[segment_name]
        # Renormalize segment's mass to make sure the total mass is the mass of the subject
        for segment_name in self.scaling_segments.keys():
            segment_masses[segment_name] *= mass / total_scaled_mass

        return scaling_factors, segment_masses

    def scale_model_geometrically(self, marker_positions: np.ndarray, marker_names: list[str], mass: float):

        original_mass = self.original_model_biorbd.mass()

        scaling_factors, segment_masses = self.get_scaling_factors_and_masses(
            marker_positions, marker_names, mass, original_mass
        )

        self.scaled_model.header = self.original_model.header + f"\nModel scaled using Biobuddy.\n"
        self.scaled_model.gravity = self.original_model.gravity

        # Scale segments
        for segment_name in self.original_model.segments.keys():
            if segment_name not in self.scaling_segments.keys():
                # If the segment is not scaled, copy it to the scaled model
                self.scaled_model.segments[segment_name] = deepcopy(self.original_model.segments[segment_name])
            else:
                this_segment_scale_factor = scaling_factors[segment_name].to_vector()

                parent_name = self.original_model.segments[segment_name].parent_name
                if parent_name in self.scaling_segments.keys():
                    parent_scale_factor = scaling_factors[parent_name].to_vector()
                else:
                    parent_scale_factor = np.array([1.0, 1.0, 1.0])

                self.scaled_model.segments.append(
                    self.scale_segment(
                        self.original_model.segments[segment_name],
                        parent_scale_factor,
                        this_segment_scale_factor,
                        segment_masses[segment_name],
                    )
                )

                for marker in self.original_model.segments[segment_name].markers:
                    self.scaled_model.segments[segment_name].add_marker(
                        self.scale_marker(marker, this_segment_scale_factor)
                    )

                for contact in self.original_model.segments[segment_name].contacts:
                    self.scaled_model.segments[segment_name].add_contact(
                        self.scale_contact(contact, this_segment_scale_factor)
                    )

                for imu in self.original_model.segments[segment_name].imus:
                    self.scaled_model.segments[segment_name].add_imu(self.scale_imu(imu, this_segment_scale_factor))

        # Set muscle groups
        self.scaled_model.muscle_groups = deepcopy(self.original_model.muscle_groups)

        # Scale muscles
        for muscle_name in self.original_model.muscles.keys():

            muscle_group_name = self.original_model.muscles[muscle_name].muscle_group
            origin_parent_name = self.original_model.muscle_groups[muscle_group_name].origin_parent_name
            insertion_parent_name = self.original_model.muscle_groups[muscle_group_name].insertion_parent_name
            origin_scale_factor = scaling_factors[origin_parent_name].to_vector()
            insertion_scale_factor = scaling_factors[insertion_parent_name].to_vector()

            if (
                origin_parent_name not in self.scaling_segments.keys()
                and insertion_parent_name not in self.scaling_segments.keys()
            ):
                # If the muscle is not attached to a segment that is scaled, do not scale the muscle
                self.scaled_model.muscles.append(self.original_model.muscles[muscle_name])
            else:
                self.scaled_model.muscles.append(
                    self.scale_muscle(
                        self.original_model.muscles[muscle_name], origin_scale_factor, insertion_scale_factor
                    )
                )

        # Scale via points
        for via_point_name in self.original_model.via_points.keys():

            parent_name = self.original_model.via_points[via_point_name].parent_name
            parent_scale_factor = scaling_factors[parent_name].to_vector()

            if parent_name not in self.scaling_segments.keys():
                # If the via point is not attached to a segment that is scaled, do not scale the via point
                self.scaled_model.via_points.append(self.original_model.via_points[via_point_name])
            else:
                self.scaled_model.via_points.append(
                    self.scale_via_point(self.original_model.via_points[via_point_name], parent_scale_factor)
                )

        self.scaled_model.warnings = self.original_model.warnings

        return

    @staticmethod
    def scale_rt(rt: np.ndarray, scale_factor: np.ndarray) -> np.ndarray:
        rt_matrix = deepcopy(rt)
        rt_matrix[:3, 3] *= scale_factor.reshape(
            3,
        )
        return rt_matrix

    def scale_segment(
        self,
        original_segment: SegmentReal,
        parent_scale_factor: np.ndarray,
        scale_factor: np.ndarray,
        segment_mass: float,
    ) -> SegmentReal:
        """
        Only geometrical scaling is implemented.
        TODO: Implement scaling based on De Leva table.
        """

        if original_segment.segment_coordinate_system.is_in_global:
            raise NotImplementedError(
                "The segment_coordinate_system is not in the parent reference frame. This is not implemented yet."
            )

        segment_coordinate_system = SegmentCoordinateSystemReal(
            scs=self.scale_rt(original_segment.segment_coordinate_system.scs.reshape(4, 4), parent_scale_factor),
            is_scs_local=True,
        )

        original_radii_of_gyration = np.array(
            [
                np.sqrt(inertia / original_segment.inertia_parameters.mass)
                for inertia in original_segment.inertia_parameters.inertia
            ]
        )
        scaled_inertia = segment_mass * (original_radii_of_gyration * scale_factor) ** 2

        inertia_parameters = InertiaParametersReal(
            mass=segment_mass,
            center_of_mass=original_segment.inertia_parameters.center_of_mass * scale_factor,
            inertia=scaled_inertia,
        )

        mesh_file = (
            None
            if original_segment.mesh_file is None
            else MeshFileReal(
                mesh_file_name=original_segment.mesh_file.mesh_file_name,
                mesh_color=(
                    None if original_segment.mesh_file.mesh_color is None else original_segment.mesh_file.mesh_color
                ),
                mesh_scale=(
                    None
                    if original_segment.mesh_file.mesh_scale is None
                    else original_segment.mesh_file.mesh_scale * scale_factor
                ),
                mesh_rotation=(
                    None
                    if original_segment.mesh_file.mesh_rotation is None
                    else original_segment.mesh_file.mesh_rotation
                ),
                mesh_translation=(
                    None
                    if original_segment.mesh_file.mesh_translation is None
                    else original_segment.mesh_file.mesh_translation * scale_factor
                ),
            )
        )

        scaled_segment = SegmentReal(
            name=original_segment.name,
            parent_name=original_segment.parent_name,
            segment_coordinate_system=segment_coordinate_system,
            translations=original_segment.translations,
            rotations=original_segment.rotations,
            q_ranges=original_segment.q_ranges,
            qdot_ranges=original_segment.qdot_ranges,
            inertia_parameters=inertia_parameters,
            mesh_file=mesh_file,
        )

        return scaled_segment

    def scale_marker(self, original_marker: MarkerReal, scale_factor: np.ndarray) -> MarkerReal:
        scaled_marker = MarkerReal(
            name=original_marker.name,
            parent_name=original_marker.parent_name,
            position=original_marker.position.reshape(
                -1,
            )[:3]
            * scale_factor,
            is_technical=original_marker.is_technical,
            is_anatomical=original_marker.is_anatomical,
        )
        return scaled_marker

    def scale_contact(self, original_contact: ContactReal, scale_factor: np.ndarray) -> ContactReal:
        scaled_contact = ContactReal(
            name=original_contact.name,
            parent_name=original_contact.parent_name,
            position=original_contact.position.reshape(
                -1,
            )[:3]
            * scale_factor,
            axis=original_contact.axis,
        )
        return scaled_contact

    def scale_imu(
        self, original_imu: InertialMeasurementUnitReal, scale_factor: np.ndarray
    ) -> InertialMeasurementUnitReal:
        scaled_imu = InertialMeasurementUnitReal(
            name=original_imu.name,
            parent_name=original_imu.parent_name,
            scs=self.scale_rt(original_imu.scs, scale_factor),
            is_technical=original_imu.is_technical,
            is_anatomical=original_imu.is_anatomical,
        )
        return scaled_imu

    def scale_muscle(
        self, original_muscle: MuscleReal, origin_scale_factor: np.ndarray, insertion_scale_factor: np.ndarray
    ) -> MuscleReal:
        scaled_muscle = MuscleReal(
            name=original_muscle.name,
            muscle_type=original_muscle.muscle_type,
            state_type=original_muscle.state_type,
            muscle_group=original_muscle.muscle_group,
            origin_position=original_muscle.origin_position.reshape(
                -1,
            )[:3]
            * origin_scale_factor,
            insertion_position=original_muscle.insertion_position.reshape(
                -1,
            )[:3]
            * insertion_scale_factor,
            optimal_length=None,  # Will be set later
            maximal_force=original_muscle.maximal_force,
            tendon_slack_length=None,  # Will be set later
            pennation_angle=None,  # Will be set later
            maximal_excitation=original_muscle.maximal_excitation,
        )
        return scaled_muscle

    def scale_via_point(self, original_via_point: ViaPointReal, parent_scale_factor: np.ndarray) -> ViaPointReal:
        scaled_via_point = ViaPointReal(
            name=original_via_point.name,
            parent_name=original_via_point.parent_name,
            muscle_name=original_via_point.muscle_name,
            muscle_group=original_via_point.muscle_group,
            position=original_via_point.position * parent_scale_factor,
        )
        return scaled_via_point


    def place_model_in_static_pose(self, marker_positions: np.ndarray, marker_names: list[str]) -> np.ndarray:

        def marker_diff(q: np.ndarray, markers_real: np.ndarray) -> np.ndarray:
            markers_model = np.array(self.scaled_model_biorbd.markers(q))
            nb_marker = markers_real.shape[1]
            vect_pos_markers = np.zeros(3 * nb_marker)
            for m, value in enumerate(markers_model):
                vect_pos_markers[m * 3: (m + 1) * 3] = value.to_array()
            return vect_pos_markers - np.reshape(markers_real.T, (3 * nb_marker,))

        def marker_jacobian(q: np.ndarray) -> np.ndarray:
            nb_q = q.shape[0]
            jacobian_matrix = np.array(self.scaled_model_biorbd.markersJacobian(q))
            nb_marker = markers_real.shape[1]
            vec_jacobian = np.zeros((3 * nb_marker, nb_q))
            for m, value in enumerate(jacobian_matrix):
                vec_jacobian[m * 3: (m + 1) * 3, :] = value.to_array()
            return vec_jacobian

        marker_indices = [marker_names.index(m.to_string()) for m in self.scaled_model_biorbd.markerNames()]
        markers_real = marker_positions[:, marker_indices, :]
        nb_frames = marker_positions.shape[2]
        nb_q = self.scaled_model_biorbd.nbQ()
        min_bound = np.ones((nb_q, )) * -np.pi
        max_bound = np.ones((nb_q, )) * np.pi
        init = np.ones((nb_q, )) * 0.0001

        optimal_q = np.zeros((nb_q, nb_frames))
        for f in range(nb_frames):
            sol = optimize.least_squares(
                fun=lambda q, marker_real: marker_diff(q, marker_real),
                args=markers_real,
                bounds=(min_bound, max_bound),
                jac=lambda q: marker_jacobian(q),
                x0=init,
                method="lm",
                xtol=1e-6,
                tr_options=dict(disp=False),
            )
            optimal_q[:, f] = sol.x

        if np.std(optimal_q, axis=0) > 20 * 180 / np.pi:
            raise RuntimeError("The inverse kinematics shows more than 20° variance over the frame range specified."
                               "Please verify that the model and subject are not positioned close to singularities (gimbal lock).")

        return np.median(optimal_q, axis=0)


    def replace_markers_on_segments(self, q_static, marker_positions: np.ndarray, marker_names: list[str]):
        jcs = self.scaled_model_biorbd.globalJCS(q_static)
        for i_segment, segment_name in enumerate(self.scaled_model.segments.keys()):
            for marker in self.scaled_model.segments.markers:
                marker_name = marker.name
                marker_index = marker_names.index(marker_name)
                this_marker_position = np.nanmean(marker_positions[:, marker_index], axis=0)
                segment_jcs = jcs[i_segment].to_array()
                marker.position = segment_jcs @ this_marker_position


    def modify_muscle_parameters(self):
        """
        Modify the optimal length, tendon slack length and pennation angle of the muscles.
        """
        print("TODO")

    @staticmethod
    def from_biomod(
        filepath: str,
    ):
        """
        Create a biomechanical model from a biorbd model
        """
        from ..model_parser.biorbd import BiomodConfigurationParser

        return BiomodConfigurationParser(filepath=filepath)

    @staticmethod
    def from_xml(
        filepath: str,
    ):
        """
        Read an xml file from OpenSim and extract the scaling configuration.

        Parameters
        ----------
        filepath: str
            The path to the xml file to read from
        """
        from ..model_parser.opensim import OsimConfigurationParser

        configuration = OsimConfigurationParser(filepath=filepath)
        return configuration.scale_tool
