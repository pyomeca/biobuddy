import os
from copy import deepcopy
from enum import Enum

import numpy as np

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.segment_scaling import SegmentScaling
from ..components.real.rigidbody.segment_real import SegmentReal
from ..components.real.rigidbody.marker_real import MarkerReal
from ..components.real.rigidbody.contact_real import ContactReal
from ..components.real.rigidbody.inertial_measurement_unit_real import InertialMeasurementUnitReal
from ..components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ..components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ..components.real.muscle.muscle_real import MuscleReal
from ..components.real.muscle.via_point_real import ViaPointReal
from ..utils.linear_algebra import RotoTransMatrix
from ..utils.named_list import NamedList
from ..utils.c3d_data import C3dData


class InertialCharacteristics(Enum):
    DE_LEVA = "de_leva"  # TODO


class ScaleTool:
    def __init__(
        self,
        original_model: BiomechanicalModelReal,
        personalize_mass_distribution: bool = True,
        max_marker_movement: float = 0.1,
    ):
        """
        Initialize the scale tool.

        Parameters
        ----------
        original_model
            The original model to scale
        personalize_mass_distribution
            If True, the mass distribution of the mass across segments will be personalized based on the marker positions. Otherwise, the mass distribution across segments will be the same as the original model.
        max_marker_movement
            The maximum acceptable marker movement in the static trial to consider it "static".
        """

        # Original attributes
        self.original_model = original_model
        self.personalize_mass_distribution = personalize_mass_distribution
        self.max_marker_movement = max_marker_movement

        # Extended attributes to be filled
        self.scaled_model = BiomechanicalModelReal()
        self.mean_experimental_markers = None  # This field will be set when .scale is run

        self.header = ""
        self.original_mass = None
        self.scaling_segments = NamedList[SegmentScaling]()
        self.marker_weightings = {}
        self.warnings = ""

    def scale(
        self,
        filepath: str,
        first_frame: int,
        last_frame: int,
        mass: float,
        q_regularization_weight: float = None,
        initial_static_pose: np.ndarray = None,
        make_static_pose_the_models_zero: bool = True,
        visualize_optimal_static_pose: bool = False,
        method: str = "lm",
    ) -> BiomechanicalModelReal:
        """
        Scale the model using the configuration defined in the ScaleTool.

        Parameters
        ----------
        filepath
            The .c3d or .trc file of the static trial to use for the scaling
        first_frame
            The index of the first frame to use in the .c3d file.
        last_frame
            The index of the last frame to use in the .c3d file.
        mass
            The mass of the subject
        q_regularization_weight
            The weight of the regularization term in the inverse kinematics. If None, no regularization is applied.
        initial_static_pose
            The approximate posture (q) in which the subject will be during the static trial.
            Ideally, this should be zero so that the posture of the original model would be in the same posture as the subject during the static trial.
        make_static_pose_the_models_zero
            If True, the static posture of the model will be set to zero after scaling. Thus when a vector of zero is sent ot the model, it will be in the same posture as the subject during the static trisl.
        visualize_optimal_static_pose
            If True, the optimal static pose will be visualized using pyorerun. Itis always recommended to visually inspect the result of the scaling procedure to make sure it went all right.
        method
            The lease square method to use. (default: "lm", other options: "trf" or "dogbox")
        """

        # Check file format
        if filepath.endswith(".c3d"):
            # Load the c3d file
            c3d_data = C3dData(filepath, first_frame, last_frame)
            marker_names = c3d_data.marker_names
            marker_positions = c3d_data.all_marker_positions[:3, :, :]
        else:
            if filepath.endswith(".trc"):
                raise NotImplementedError(".trc files cannot be read yet.")
            else:
                raise RuntimeError("The filepath (static trial) must be a .c3d file in a static posture.")

        # Check the weights
        for marker in self.marker_weightings:
            if self.marker_weightings[marker] < 0:
                raise RuntimeError(f"The weight of marker {marker} is negative. It must be positive.")

        # Check the mass
        if mass <= 0:
            raise RuntimeError(f"The mass of the subject must be positive. The value given is {mass} kg.")

        # Check that a scaling configuration was set
        if len(self.scaling_segments) == 0:
            raise RuntimeError(
                "No scaling configuration was set. Please set a scaling configuration using ScaleTool.from_xml(filepath=filepath) or ScaleTool.from_biomod(filepath=filepath)."
            )

        self.check_that_makers_do_not_move(marker_positions, marker_names)
        self.check_segments()
        self.define_mean_experimental_markers(marker_positions, marker_names)

        self.scale_model_geometrically(marker_positions, marker_names, mass)

        # self.modify_muscle_parameters() # TODO !!!!!!!
        self.place_model_in_static_pose(
            marker_positions,
            marker_names,
            q_regularization_weight,
            initial_static_pose,
            make_static_pose_the_models_zero,
            visualize_optimal_static_pose,
            method,
        )

        self.scaled_model.segments_rt_to_local()
        return self.scaled_model

    def add_scaling_segment(self, scaling_segment: SegmentScaling):
        """
        Add a scaling segment to the scale tool.

        Parameters
        ----------
        scaling_segment
            The scaling segment to add
        """

        if not isinstance(scaling_segment, SegmentScaling):
            raise RuntimeError("The scaling segment must be of type SegmentScaling.")
        self.scaling_segments.append(scaling_segment)

    def remove_scaling_segment(self, segment_scaling_name: str):
        """
        Remove a scaling segment from the scale tool.

        Parameters
        ----------
        segment_scaling_name
            The name of the scaling segment to remove
        """
        self.scaling_segments.remove(segment_scaling_name)

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
            inertia = deepcopy(self.original_model.segments[segment_name].inertia_parameters)
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

    def define_mean_experimental_markers(self, marker_positions, marker_names):
        model_marker_names = self.original_model.marker_names
        self.mean_experimental_markers = np.zeros((3, len(model_marker_names)))
        for i_marker, name in enumerate(model_marker_names):
            marker_index = marker_names.index(name)
            this_marker_position = marker_positions[:, marker_index, :]
            self.mean_experimental_markers[:, i_marker] = np.nanmean(this_marker_position, axis=1)

    def get_scaling_factors_and_masses(
        self,
        marker_positions: np.ndarray,
        marker_names: list[str],
        mass: float,
        original_mass: float,
    ) -> tuple[dict[str, "ScaleFactor"], dict[str, np.ndarray]]:

        scaling_factors = {}
        segment_masses = {}
        total_scaled_mass = 0
        for segment_name in self.scaling_segments.keys():
            # Compute the scale factors
            scaling_factors[segment_name] = self.scaling_segments[segment_name].compute_scaling_factors(
                self.original_model, marker_positions, marker_names
            )
            # Get each segment's scaled mass
            if self.personalize_mass_distribution:
                segment_masses[segment_name] = (
                    deepcopy(self.original_model.segments[segment_name].inertia_parameters.mass)
                    * scaling_factors[segment_name].mass
                )
            else:
                segment_masses[segment_name] = (
                    deepcopy(self.original_model.segments[segment_name].inertia_parameters.mass) * mass / original_mass
                )
            total_scaled_mass += segment_masses[segment_name]

        # Renormalize segment's mass to make sure the total mass is the mass of the subject
        for segment_name in self.scaling_segments.keys():
            segment_masses[segment_name] *= mass / total_scaled_mass

        return scaling_factors, segment_masses

    def scale_model_geometrically(self, marker_positions: np.ndarray, marker_names: list[str], mass: float):

        original_mass = self.original_model.mass

        scaling_factors, segment_masses = self.get_scaling_factors_and_masses(
            marker_positions, marker_names, mass, original_mass
        )

        self.scaled_model.header = deepcopy(self.original_model.header) + f"\nModel scaled using Biobuddy.\n"
        self.scaled_model.gravity = deepcopy(self.original_model.gravity)

        for segment_name in self.original_model.segments.keys():

            # Check if the segments has a ghost parent
            if (
                self.original_model.segments[segment_name].name + "_parent_offset"
                in self.original_model.segments.keys()
            ):
                offset_parent = self.original_model.segments[segment_name + "_parent_offset"].parent_name
                if offset_parent in self.scaling_segments.keys():
                    # Apply scaling to the position of the offset parent segment instead of the current segment
                    offset_parent_scale_factor = scaling_factors[offset_parent].to_vector()
                    scs_scaled = SegmentCoordinateSystemReal(
                        scs=self.scale_rt(
                            deepcopy(
                                self.original_model.segments[
                                    segment_name + "_parent_offset"
                                ].segment_coordinate_system.scs[:, :, 0]
                            ),
                            offset_parent_scale_factor,
                        ),
                        is_scs_local=True,
                    )
                    self.scaled_model.segments[segment_name + "_parent_offset"].segment_coordinate_system = scs_scaled

                # Scale the meshes of the intermediary ghost segments
                looping_parent_name = self.original_model.segments[
                    segment_name
                ].parent_name  # The current segment's mesh will be scaled later
                scale_factor = scaling_factors[segment_name].to_vector()
                while "_parent_offset" not in looping_parent_name:
                    mesh_file = deepcopy(self.original_model.segments[looping_parent_name].mesh_file)
                    if mesh_file is not None:
                        mesh_file.mesh_scale *= scale_factor
                        mesh_file.mesh_translation *= scale_factor
                    self.scaled_model.segments[looping_parent_name].mesh_file = mesh_file
                    looping_parent_name = self.original_model.segments[looping_parent_name].parent_name

            # Apply scaling to the current segment
            if self.original_model.segments[segment_name].parent_name in self.scaling_segments.keys():
                parent_scale_factor = scaling_factors[
                    self.original_model.segments[segment_name].parent_name
                ].to_vector()
            else:
                parent_scale_factor = np.ones((4, 1))

            # Scale segments
            if segment_name in self.scaling_segments.keys():
                this_segment_scale_factor = scaling_factors[segment_name].to_vector()
                self.scaled_model.add_segment(
                    self.scale_segment(
                        deepcopy(self.original_model.segments[segment_name]),
                        parent_scale_factor,
                        this_segment_scale_factor,
                        segment_masses[segment_name],
                    )
                )

                for marker in deepcopy(self.original_model.segments[segment_name].markers):
                    self.scaled_model.segments[segment_name].remove_marker(marker.name)
                    self.scaled_model.segments[segment_name].add_marker(
                        self.scale_marker(marker, this_segment_scale_factor)
                    )

                for contact in deepcopy(self.original_model.segments[segment_name].contacts):
                    self.scaled_model.segments[segment_name].remove_contact(contact.name)
                    self.scaled_model.segments[segment_name].add_contact(
                        self.scale_contact(contact, this_segment_scale_factor)
                    )

                for imu in deepcopy(self.original_model.segments[segment_name].imus):
                    self.scaled_model.segments[segment_name].remove_imu(imu.name)
                    self.scaled_model.segments[segment_name].add_imu(self.scale_imu(imu, this_segment_scale_factor))

            else:
                self.scaled_model.segments[segment_name] = deepcopy(self.original_model.segments[segment_name])

        # Set muscle groups
        self.scaled_model.muscle_groups = deepcopy(self.original_model.muscle_groups)

        # Scale muscles
        for muscle_name in self.original_model.muscles.keys():

            muscle_group_name = deepcopy(self.original_model.muscles[muscle_name].muscle_group)
            origin_parent_name = deepcopy(self.original_model.muscle_groups[muscle_group_name].origin_parent_name)
            insertion_parent_name = deepcopy(self.original_model.muscle_groups[muscle_group_name].insertion_parent_name)
            origin_scale_factor = scaling_factors[origin_parent_name].to_vector()
            insertion_scale_factor = scaling_factors[insertion_parent_name].to_vector()

            if (
                origin_parent_name not in self.scaling_segments.keys()
                and insertion_parent_name not in self.scaling_segments.keys()
            ):
                # If the muscle is not attached to a segment that is scaled, do not scale the muscle
                self.scaled_model.add_muscle(deepcopy(self.original_model.muscles[muscle_name]))
            else:
                self.scaled_model.add_muscle(
                    self.scale_muscle(
                        deepcopy(self.original_model.muscles[muscle_name]), origin_scale_factor, insertion_scale_factor
                    )
                )

        # Scale via points
        for via_point_name in self.original_model.via_points.keys():

            parent_name = deepcopy(self.original_model.via_points[via_point_name].parent_name)
            parent_scale_factor = scaling_factors[parent_name].to_vector()

            if parent_name not in self.scaling_segments.keys():
                # If the via point is not attached to a segment that is scaled, do not scale the via point
                self.scaled_model.add_via_point(deepcopy(self.original_model.via_points[via_point_name]))
            else:
                self.scaled_model.add_via_point(
                    self.scale_via_point(deepcopy(self.original_model.via_points[via_point_name]), parent_scale_factor)
                )

        self.scaled_model.warnings = deepcopy(self.original_model.warnings)

        return

    @staticmethod
    def scale_rt(rt: np.ndarray, scale_factor: np.ndarray) -> np.ndarray:
        rt_matrix = deepcopy(rt)
        rt_matrix[:3, 3] *= scale_factor[:3].reshape(
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
            scs=self.scale_rt(original_segment.segment_coordinate_system.scs[:, :, 0], parent_scale_factor),
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

        mesh_file = deepcopy(original_segment.mesh_file)
        mesh_file.mesh_scale *= scale_factor
        mesh_file.mesh_translation *= scale_factor

        scaled_segment = deepcopy(original_segment)
        scaled_segment.segment_coordinate_system = segment_coordinate_system
        scaled_segment.inertia_parameters = inertia_parameters
        scaled_segment.mesh_file = mesh_file

        return scaled_segment

    def scale_marker(self, original_marker: MarkerReal, scale_factor: np.ndarray) -> MarkerReal:
        scaled_marker = deepcopy(original_marker)
        scaled_marker.position *= scale_factor
        return scaled_marker

    def scale_contact(self, original_contact: ContactReal, scale_factor: np.ndarray) -> ContactReal:
        scaled_contact = deepcopy(original_contact)
        scaled_contact.position *= scale_factor
        return scaled_contact

    def scale_imu(
        self, original_imu: InertialMeasurementUnitReal, scale_factor: np.ndarray
    ) -> InertialMeasurementUnitReal:
        scaled_imu = deepcopy(original_imu)
        scaled_imu.scs = self.scale_rt(original_imu.scs[:, :, 0], scale_factor)
        return scaled_imu

    def scale_muscle(
        self, original_muscle: MuscleReal, origin_scale_factor: np.ndarray, insertion_scale_factor: np.ndarray
    ) -> MuscleReal:
        scaled_muscle = deepcopy(original_muscle)
        scaled_muscle.origin_position *= origin_scale_factor
        scaled_muscle.insertion_position *= insertion_scale_factor
        scaled_muscle.optimal_length = (None,)  # Will be set later
        scaled_muscle.tendon_slack_length = (None,)  # Will be set later
        return scaled_muscle

    def scale_via_point(self, original_via_point: ViaPointReal, parent_scale_factor: np.ndarray) -> ViaPointReal:
        scaled_via_point = deepcopy(original_via_point)
        scaled_via_point.position *= parent_scale_factor
        return scaled_via_point

    def find_static_pose(
        self,
        marker_positions: np.ndarray,
        experimental_marker_names: list[str],
        q_regularization_weight: float | None,
        initial_static_pose: np.ndarray | None,
        visualize_optimal_static_pose: bool,
        method: str,
    ) -> np.ndarray:

        optimal_q = self.scaled_model.inverse_kinematics(
            marker_positions=marker_positions,
            marker_names=experimental_marker_names,
            q_regularization_weight=q_regularization_weight,
            q_target=initial_static_pose,
            method=method,
        )

        if visualize_optimal_static_pose:
            # Show the animation for debugging
            try:
                import pyorerun
                from pyomeca import Markers
            except ImportError:
                raise ImportError("You must install pyorerun and pyomeca to visualize the model")

            t = np.linspace(0, 1, marker_positions.shape[2])
            viz = pyorerun.PhaseRerun(t)

            debugging_model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../examples/models/temporary_model.bioMod")
            )
            self.scaled_model.to_biomod(debugging_model_path)
            viz_biomod_model = pyorerun.BiorbdModel(debugging_model_path)
            viz_biomod_model.options.transparent_mesh = False
            viz_biomod_model.options.show_gravity = True
            viz.add_animated_model(viz_biomod_model, optimal_q)

            model_marker_names = self.scaled_model.marker_names
            marker_indices = [experimental_marker_names.index(m) for m in model_marker_names]
            pyomarkers = Markers(data=marker_positions[:, marker_indices, :], channels=model_marker_names)
            viz.add_xp_markers(name=experimental_marker_names, markers=pyomarkers)
            # tracked_markers=pyomarkers
            viz.rerun_by_frame("Model output")

        if any(np.std(optimal_q, axis=1) > 20 * np.pi / 180):
            raise RuntimeError(
                "The inverse kinematics shows more than 20° variance over the frame range specified."
                "Please see the animation provided to verify that the subject does not move during the static trial."
                "If not, please make sure the model and subject are not positioned close to singularities (gimbal lock)."
            )

        return np.median(optimal_q, axis=1)

    def make_static_pose_the_zero(self, q_static: np.ndarray):
        jcs_in_global = self.scaled_model.forward_kinematics(q_static)
        for i_segment, segment_name in enumerate(self.scaled_model.segments.keys()):
            self.scaled_model.segments[segment_name].segment_coordinate_system = SegmentCoordinateSystemReal(
                scs=jcs_in_global[segment_name][:, :, 0],
                parent_scs=None,
                is_scs_local=(
                    segment_name == "base"
                ),  # joint coordinate system is now expressed in the global except for the base because it does not have a parent
            )

    def replace_markers_on_segments_global_scs(self, marker_positions: np.ndarray, marker_names: list[str]):
        for i_segment, segment in enumerate(self.scaled_model.segments):
            if segment.segment_coordinate_system is None or segment.segment_coordinate_system.is_in_local:
                raise RuntimeError(
                    "Something went wrong. Following make_static_pose_the_zero, the segment's coordinate system should be in the global."
                )
            for marker in segment.markers:
                marker_name = marker.name
                marker_index = marker_names.index(marker_name)
                this_marker_position = np.nanmean(marker_positions[:, marker_index], axis=1)
                rt = RotoTransMatrix()
                rt.rt_matrix = deepcopy(segment.segment_coordinate_system.scs[:, :, 0])
                marker.position = rt.inverse @ np.hstack((this_marker_position, 1))

    def replace_markers_on_segments_local_scs(
        self, marker_positions: np.ndarray, marker_names: list[str], q: np.ndarray
    ):
        jcs_in_global = self.scaled_model.forward_kinematics(q)
        for i_segment, segment in enumerate(self.scaled_model.segments):
            if segment.segment_coordinate_system is None or segment.segment_coordinate_system.is_in_global:
                raise RuntimeError(
                    "Something went wrong. Since make_static_pose_the_models_zero was set to False, the segment's coordinate system should be in the local reference frames."
                )
            for marker in segment.markers:
                marker_name = marker.name
                marker_index = marker_names.index(marker_name)
                this_marker_position = np.nanmean(marker_positions[:, marker_index], axis=1)
                rt = RotoTransMatrix()
                rt.rt_matrix = deepcopy(jcs_in_global[segment.name][:, :, 0])
                marker.position = rt.inverse @ np.hstack((this_marker_position, 1))

    def place_model_in_static_pose(
        self,
        marker_positions: np.ndarray,
        marker_names: list[str],
        q_regularization_weight: float | None,
        initial_static_pose: np.ndarray | None,
        make_static_pose_the_models_zero: bool,
        visualize_optimal_static_pose: bool,
        method: str,
    ):
        q_static = self.find_static_pose(
            marker_positions,
            marker_names,
            q_regularization_weight,
            initial_static_pose,
            visualize_optimal_static_pose,
            method,
        )
        if make_static_pose_the_models_zero:
            self.make_static_pose_the_zero(q_static)
            self.replace_markers_on_segments_global_scs(marker_positions, marker_names)
        else:
            self.replace_markers_on_segments_local_scs(marker_positions, marker_names, q_static)

    def modify_muscle_parameters(self):
        """
        Modify the optimal length, tendon slack length and pennation angle of the muscles.
        # TODO: compute muscle length !
        """
        muscle_names = self.original_model.muscle_names
        q_zeros = np.zeros((self.original_model.nb_q,))
        for muscle_name in self.original_model.muscles.keys():
            original_muscle_length = self.original_model.muscle_length(muscle_name, q_zeros)
            scaled_muscle_length = self.scaled_model.muscle_length(muscle_name, q_zeros)
            if self.original_model.muscles[muscle_name].optimal_length is None:
                print("sss")
            self.scaled_model.muscles[muscle_name].optimal_length = (
                deepcopy(self.original_model.muscles[muscle_name].optimal_length)
                * scaled_muscle_length
                / original_muscle_length
            )
            self.scaled_model.muscles[muscle_name].tendon_slack_length = (
                deepcopy(self.original_model.muscles[muscle_name].tendon_slack_length)
                * scaled_muscle_length
                / original_muscle_length
            )

    @staticmethod
    def from_biomod(
        filepath: str,
    ):
        """
        Create a biomechanical model from a biorbd model
        """
        from ..model_parser.biorbd import BiomodConfigurationParser

        return BiomodConfigurationParser(filepath=filepath)

    def from_xml(
        self,
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

        configuration = OsimConfigurationParser(filepath=filepath, original_model=self.original_model)
        return configuration.scale_tool
