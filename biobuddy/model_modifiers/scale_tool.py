import os
from copy import deepcopy
from enum import Enum
import logging
import xml.etree.cElementTree as ET
from xml.dom import minidom
import numpy as np

from .modifiers_utils import modify_muscle_parameters
from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.segment_scaling import SegmentScaling
from ..components.real.rigidbody.marker_weight import MarkerWeight
from ..components.real.rigidbody.segment_real import SegmentReal
from ..components.real.rigidbody.marker_real import MarkerReal
from ..components.real.rigidbody.contact_real import ContactReal
from ..components.real.rigidbody.inertial_measurement_unit_real import InertialMeasurementUnitReal
from ..components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ..components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ..components.real.muscle.muscle_real import MuscleReal
from ..components.real.muscle.muscle_group_real import MuscleGroupReal
from ..components.real.muscle.via_point_real import ViaPointReal
from ..utils.linear_algebra import RotoTransMatrix
from ..utils.named_list import NamedList
from ..utils.c3d_data import C3dData
from ..utils.enums import Translations
from ..utils.enums import Rotations
from ..utils.aliases import Point

_logger = logging.getLogger(__name__)


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
        self.marker_weights = NamedList[MarkerWeight]()
        self.warnings = ""

    def add_marker_weight(self, marker_weight: MarkerWeight):
        self.marker_weights._append(marker_weight)

    def remove_marker_weight(self, marker_name: str):
        self.marker_weights._remove(marker_name)

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
        self.scaling_segments._append(scaling_segment)

    def remove_scaling_segment(self, segment_scaling_name: str):
        """
        Remove a scaling segment from the scale tool.

        Parameters
        ----------
        segment_scaling_name
            The name of the scaling segment to remove
        """
        self.scaling_segments._remove(segment_scaling_name)

    def print_marker_weights(self):
        """
        Print the marker weights in a human-readable format for debugging purposes.
        """
        for marker_weight in self.marker_weights:
            print(f"{marker_weight.name} : {marker_weight.weight:.2f}")

    def scale(
        self,
        static_c3d: C3dData,
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
        static_c3d
            The .c3d or .trc file of the static trial to use for the scaling
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
        exp_marker_names = static_c3d.marker_names
        exp_marker_positions = static_c3d.all_marker_positions

        marker_indices = [idx for idx, m in enumerate(exp_marker_names) if m in self.original_model.marker_names]
        marker_names = [exp_marker_names[idx] for idx in marker_indices]
        marker_positions = exp_marker_positions[:, marker_indices, :]

        # Check the weights
        for marker_name in self.marker_weights.keys():
            if self.marker_weights[marker_name].weight < 0:
                raise RuntimeError(f"The weight of marker {marker_name} is negative. It must be positive.")

        # Check the mass
        if mass <= 0:
            raise RuntimeError(f"The mass of the subject must be positive. The value given is {mass} kg.")

        # Check that a scaling configuration was set
        if len(self.scaling_segments) == 0:
            raise RuntimeError(
                "No scaling configuration was set. Please set a scaling configuration using ScaleTool().from_xml(filepath=filepath) or ScaleTool().from_biomod(filepath=filepath)."
            )

        self.check_that_via_points_are_fixed()
        self.check_that_makers_do_not_move(marker_positions, marker_names)
        self.check_segments()
        self.define_mean_experimental_markers(marker_positions, marker_names)

        self.scale_model_geometrically(marker_positions, marker_names, mass)

        modify_muscle_parameters(self.original_model, self.scaled_model)
        self.place_model_in_static_pose(
            marker_positions,
            marker_names,
            q_regularization_weight,
            initial_static_pose,
            make_static_pose_the_models_zero,
            visualize_optimal_static_pose,
            method,
        )

        return self.scaled_model

    def check_that_via_points_are_fixed(self):
        """
        It is not possible yet to scale a model that has moving via points.
        """
        for muscle_group in self.original_model.muscle_groups:
            for muscle in muscle_group.muscles:
                if muscle.origin_position.movement is not None:
                    raise NotImplementedError(
                        f"The muscle {muscle.name} has a moving origin. Scaling models with moving via points is not implemented yet. Please run model.fix_via_points() before scaling the model."
                    )
                if muscle.insertion_position.movement is not None:
                    raise NotImplementedError(
                        f"The muscle {muscle.name} has a moving insertion. Scaling models with moving via points is not implemented yet. Please run model.fix_via_points() before scaling the model."
                    )
                for via_point in muscle.via_points:
                    if via_point.movement is not None:
                        if muscle.insertion_position.movement is not None:
                            raise NotImplementedError(
                                f"The muscle {muscle.name} has a moving via point. Scaling models with moving via points is not implemented yet. Please run model.fix_via_points() before scaling the model."
                            )

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
            for marker_name in self.marker_weights.keys():
                if self.marker_weights[marker_name].name not in marker_names:
                    raise RuntimeError(f"The marker {marker_name} is not in the c3d file.")
                marker_index = marker_names.index(marker_name)
                this_marker_position = marker_positions[:, marker_index, :]
                min_position = np.nanmin(this_marker_position, axis=1)
                max_position = np.nanmax(this_marker_position, axis=1)
                if np.linalg.norm(max_position - min_position) > self.max_marker_movement:
                    raise RuntimeError(
                        f"The marker {marker_name} moves of approximately {np.linalg.norm(max_position - min_position)} m during the static trial, which is above the maximal limit of {self.max_marker_movement} m."
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
        self.mean_experimental_markers = np.ones((4, len(model_marker_names)))
        for i_marker, name in enumerate(model_marker_names):
            marker_index = marker_names.index(name)
            this_marker_position = marker_positions[:3, marker_index, :]
            self.mean_experimental_markers[:3, i_marker] = np.nanmean(this_marker_position, axis=1)

    def get_scaling_factors_and_masses(
        self,
        marker_positions: np.ndarray,
        marker_names: list[str],
        mass: float,
        original_mass: float,
    ) -> tuple[dict[str, "ScaleFactor"], dict[str, float]]:

        scaling_factors = {}
        segment_masses = {}
        total_scaled_mass = 0
        for segment_name in self.original_model.segment_names:
            if segment_name in self.scaling_segments.keys():
                # Compute the scale factors
                scaling_factors[segment_name] = self.scaling_segments[segment_name].compute_scaling_factors(
                    self.original_model, marker_positions, marker_names
                )
                # Get each segment's scaled mass
                if self.personalize_mass_distribution:
                    # Personalized scaling for each segment based on the dimension of the segment
                    segment_masses[segment_name] = (
                        deepcopy(self.original_model.segments[segment_name].inertia_parameters.mass)
                        * scaling_factors[segment_name].mass
                    )
                else:
                    # Keeping the same mass distribution as the original model
                    segment_masses[segment_name] = (
                        deepcopy(self.original_model.segments[segment_name].inertia_parameters.mass)
                        * mass
                        / original_mass
                    )
            else:
                # If the segment is not scaled, keep its original mass
                if (
                    self.original_model.segments[segment_name].inertia_parameters is None
                    or self.original_model.segments[segment_name].inertia_parameters.mass == 0
                ):
                    segment_masses[segment_name] = 0
                else:
                    raise NotImplementedError(
                        f"You have a segment {segment_name} which has a non-null inertial parameter (mass {self.original_model.segments[segment_name].inertia_parameters.mass} kg) but is not scaled. The interpretation of this case is unclear."
                    )
                    # segment_masses[segment_name] = deepcopy(self.original_model.segments[segment_name].inertia_parameters.mass)

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
        _logger.info("Scaling factors for each segment:")
        for segment_name in scaling_factors.keys():
            _logger.info(f"  {segment_name}: {scaling_factors[segment_name].to_vector()}")

        self.scaled_model.header = deepcopy(self.original_model.header) + f"\n// Model scaled using Biobuddy.\n"
        self.scaled_model.gravity = deepcopy(self.original_model.gravity)

        for segment_name in self.original_model.segments.keys():

            # Check if the segments has ghost parents
            ghost_parent_names = ["_parent_offset", "_translation"]
            for ghost_key in ghost_parent_names:
                if self.original_model.segments[segment_name].name + ghost_key in self.original_model.segments.keys():
                    offset_parent = self.original_model.segments[segment_name + ghost_key].parent_name
                    if offset_parent in self.scaling_segments.keys():
                        # Apply scaling to the position of the offset parent segment instead of the current segment
                        offset_parent_scale_factor = scaling_factors[offset_parent].to_vector()
                        scs_scaled = SegmentCoordinateSystemReal(
                            scs=self.scale_rt(
                                deepcopy(
                                    self.original_model.segments[segment_name + ghost_key].segment_coordinate_system.scs
                                ),
                                offset_parent_scale_factor,
                            ),
                            is_scs_local=True,
                        )
                        self.scaled_model.segments[segment_name + ghost_key].segment_coordinate_system = scs_scaled

                    # Scale the meshes of the intermediary ghost segments
                    looping_parent_name = self.original_model.segments[
                        segment_name
                    ].parent_name  # The current segment's mesh will be scaled later
                    scale_factor = scaling_factors[segment_name].to_vector()
                    while ghost_key not in looping_parent_name:
                        mesh_file = deepcopy(self.original_model.segments[looping_parent_name].mesh_file)
                        if mesh_file is not None:
                            mesh_file.mesh_scale *= scale_factor
                            mesh_file.mesh_translation *= scale_factor
                        self.scaled_model.segments[looping_parent_name].mesh_file = mesh_file
                        looping_parent_name = self.original_model.segments[looping_parent_name].parent_name
                    # Apply it to only one of the ghost segments recognized
                    break

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
                    self.scaled_model.segments[segment_name].add_marker(
                        self.scale_marker(marker, this_segment_scale_factor)
                    )

                for contact in deepcopy(self.original_model.segments[segment_name].contacts):
                    self.scaled_model.segments[segment_name].add_contact(
                        self.scale_contact(contact, this_segment_scale_factor)
                    )

                for imu in deepcopy(self.original_model.segments[segment_name].imus):
                    self.scaled_model.segments[segment_name].add_imu(self.scale_imu(imu, this_segment_scale_factor))

            else:
                self.scaled_model.segments[segment_name] = deepcopy(self.original_model.segments[segment_name])

        # Scale muscles
        for muscle_group in self.original_model.muscle_groups:

            self.scaled_model.add_muscle_group(
                MuscleGroupReal(
                    name=deepcopy(muscle_group.name),
                    origin_parent_name=deepcopy(muscle_group.origin_parent_name),
                    insertion_parent_name=deepcopy(muscle_group.insertion_parent_name),
                )
            )
            for muscle in muscle_group.muscles:

                muscle_name = muscle.name
                muscle_group_name = deepcopy(muscle.muscle_group)
                origin_parent_name = muscle_group.origin_parent_name
                origin_scale_factor = scaling_factors[origin_parent_name].to_vector()
                insertion_parent_name = muscle_group.insertion_parent_name
                insertion_scale_factor = scaling_factors[insertion_parent_name].to_vector()

                if (
                    origin_parent_name not in self.scaling_segments.keys()
                    and insertion_parent_name not in self.scaling_segments.keys()
                ):
                    # If the muscle is not attached to a segment that is scaled, do not scale the muscle
                    self.scaled_model.muscle_groups[muscle_group_name].add_muscle(deepcopy(muscle))
                else:
                    self.scaled_model.muscle_groups[muscle_group_name].add_muscle(
                        self.scale_muscle(deepcopy(muscle), origin_scale_factor, insertion_scale_factor)
                    )

                # Scale via points
                for via_point in muscle.via_points:

                    via_point_name = via_point.name
                    parent_name = deepcopy(via_point.parent_name)
                    parent_scale_factor = scaling_factors[parent_name].to_vector()

                    if parent_name not in self.scaling_segments.keys():
                        # If the via point is not attached to a segment that is scaled, do not scale the via point
                        self.scaled_model.muscle_groups[muscle_group_name].muscles[muscle_name].add_via_point(
                            deepcopy(via_point)
                        )
                    else:
                        self.scaled_model.muscle_groups[muscle_group_name].muscles[muscle_name].add_via_point(
                            self.scale_via_point(deepcopy(via_point), parent_scale_factor)
                        )

        self.scaled_model.warnings = deepcopy(self.original_model.warnings)

        return

    @staticmethod
    def scale_rt(rt: RotoTransMatrix, scale_factor: Point) -> RotoTransMatrix:
        rt_matrix = rt.rt_matrix
        rt_matrix[:3, 3] *= scale_factor[:3].reshape(
            3,
        )
        out = RotoTransMatrix()
        out.from_rt_matrix(rt_matrix)
        return out

    def scale_segment(
        self,
        original_segment: SegmentReal,
        parent_scale_factor: Point,
        scale_factor: Point,
        segment_mass: float,
    ) -> SegmentReal:
        """
        Inertia is scaled using the following formula:
            I_new = m_new * (radii_of_gyration * scale_factor)**2
            radii_of_gyration = sqrt(I_original / m_original)
        Only geometrical scaling is implemented.
        TODO: Implement scaling based on De Leva table.
        """

        if original_segment.segment_coordinate_system.is_in_global:
            raise NotImplementedError(
                "The segment_coordinate_system is not in the parent reference frame. This is not implemented yet."
            )

        segment_coordinate_system_scaled = SegmentCoordinateSystemReal(
            scs=self.scale_rt(original_segment.segment_coordinate_system.scs, parent_scale_factor),
            is_scs_local=True,
        )

        original_radii_of_gyration = np.array(
            [
                np.sqrt(inertia / original_segment.inertia_parameters.mass)
                for inertia in original_segment.inertia_parameters.inertia[:3, :3]
            ]
        )
        scaled_inertia = segment_mass * (original_radii_of_gyration * scale_factor[:3]) ** 2

        inertia_parameters_scaled = InertiaParametersReal(
            mass=segment_mass,
            center_of_mass=original_segment.inertia_parameters.center_of_mass * scale_factor,
            inertia=scaled_inertia,
        )

        mesh_scaled = deepcopy(original_segment.mesh)
        if mesh_scaled is not None:
            mesh_scaled.positions *= scale_factor

        mesh_file_scaled = deepcopy(original_segment.mesh_file)
        if mesh_file_scaled is not None:
            mesh_file_scaled.mesh_scale *= scale_factor
            mesh_file_scaled.mesh_translation *= scale_factor

        return SegmentReal(
            name=deepcopy(original_segment.name),
            parent_name=deepcopy(original_segment.parent_name),
            segment_coordinate_system=segment_coordinate_system_scaled,
            translations=deepcopy(original_segment.translations),
            rotations=deepcopy(original_segment.rotations),
            q_ranges=deepcopy(original_segment.q_ranges),
            qdot_ranges=deepcopy(original_segment.qdot_ranges),
            inertia_parameters=inertia_parameters_scaled,
            mesh=mesh_scaled,
            mesh_file=mesh_file_scaled,
        )

    def scale_marker(self, original_marker: MarkerReal, scale_factor: Point) -> MarkerReal:
        return MarkerReal(
            name=deepcopy(original_marker.name),
            parent_name=deepcopy(original_marker.parent_name),
            position=deepcopy(original_marker.position) * scale_factor,
            is_technical=deepcopy(original_marker.is_technical),
            is_anatomical=deepcopy(original_marker.is_anatomical),
        )

    def scale_contact(self, original_contact: ContactReal, scale_factor: Point) -> ContactReal:
        return ContactReal(
            name=deepcopy(original_contact.name),
            parent_name=deepcopy(original_contact.parent_name),
            position=deepcopy(original_contact.position) * scale_factor,
            axis=deepcopy(original_contact.axis),
        )

    def scale_imu(self, original_imu: InertialMeasurementUnitReal, scale_factor: Point) -> InertialMeasurementUnitReal:
        return InertialMeasurementUnitReal(
            name=deepcopy(original_imu.name),
            parent_name=deepcopy(original_imu.parent_name),
            scs=self.scale_rt(original_imu.scs, scale_factor),
            is_technical=deepcopy(original_imu.is_technical),
            is_anatomical=deepcopy(original_imu.is_anatomical),
        )

    def scale_muscle(
        self, original_muscle: MuscleReal, origin_scale_factor: Point, insertion_scale_factor: Point
    ) -> MuscleReal:
        origin_position = deepcopy(original_muscle.origin_position)
        origin_position.position *= origin_scale_factor
        insertion_position = deepcopy(original_muscle.insertion_position)
        insertion_position.position *= insertion_scale_factor
        return MuscleReal(
            name=deepcopy(original_muscle.name),
            muscle_type=deepcopy(original_muscle.muscle_type),
            state_type=deepcopy(original_muscle.state_type),
            muscle_group=deepcopy(original_muscle.muscle_group),
            origin_position=origin_position,
            insertion_position=insertion_position,
            optimal_length=None,  # Will be set later
            maximal_force=deepcopy(original_muscle.maximal_force),
            tendon_slack_length=None,  # Will be set later
            pennation_angle=deepcopy(original_muscle.pennation_angle),
            maximal_excitation=deepcopy(original_muscle.maximal_excitation),
        )

    def scale_via_point(self, original_via_point: ViaPointReal, parent_scale_factor: Point) -> ViaPointReal:
        return ViaPointReal(
            name=deepcopy(original_via_point.name),
            parent_name=deepcopy(original_via_point.parent_name),
            muscle_name=deepcopy(original_via_point.muscle_name),
            muscle_group=deepcopy(original_via_point.muscle_group),
            position=original_via_point.position * parent_scale_factor,
        )

    def create_free_floating_base_model(self):
        model_6dof = deepcopy(self.scaled_model)
        if "root" in model_6dof.segment_names:
            model_6dof.segments["root"].translations = Translations.XYZ
            model_6dof.segments["root"].rotations = Rotations.XYZ
        else:
            raise NotImplementedError(
                "The model does not have a root segment. Creation of a temporary free-floating base model is not implemented, yet. Please notify the devs is you encounter this issue."
            )
        return model_6dof

    def find_static_pose(
        self,
        marker_positions: np.ndarray,
        experimental_marker_names: list[str],
        q_regularization_weight: float | None,
        initial_static_pose: np.ndarray | None,
        visualize_optimal_static_pose: bool,
        method: str,
    ) -> tuple[np.ndarray, BiomechanicalModelReal]:

        if self.scaled_model.root_segment.nb_q == 6 or self.scaled_model.dofs[:2] == [
            Translations.XYZ,
            Rotations.XYZ,
        ]:
            model_to_use = deepcopy(self.scaled_model)
        else:
            # If the model does not have a free-floating base, we need to create a temporary model with a free-floating base to match the experimental markers
            model_to_use = self.create_free_floating_base_model()

        optimal_q, _ = model_to_use.inverse_kinematics(
            marker_positions=marker_positions,
            marker_names=experimental_marker_names,
            q_regularization_weight=q_regularization_weight,
            q_target=initial_static_pose,
            marker_weights=self.marker_weights,
            method=method,
        )

        if visualize_optimal_static_pose:
            # Show the animation for debugging
            try:
                import pyorerun
            except ImportError:
                raise ImportError("You must install pyorerun to visualize the model")

            t = np.linspace(0, 1, marker_positions.shape[2])
            viz = pyorerun.PhaseRerun(t)

            debugging_model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../examples/models/temporary.bioMod")
            )
            model_to_use.to_biomod(debugging_model_path)
            viz_biomod_model = pyorerun.BiorbdModel(debugging_model_path)
            viz_biomod_model.options.transparent_mesh = False
            viz_biomod_model.options.show_gravity = True
            viz_biomod_model.options.show_marker_labels = False
            viz_biomod_model.options.show_center_of_mass_labels = False

            model_marker_names = model_to_use.marker_names
            marker_indices = [experimental_marker_names.index(m) for m in model_marker_names]
            pyomarkers = pyorerun.PyoMarkers(
                data=marker_positions[:, marker_indices, :], marker_names=model_marker_names, show_labels=False
            )
            viz.add_animated_model(viz_biomod_model, optimal_q, tracked_markers=pyomarkers)
            viz.rerun("Model output")

        if any(np.std(optimal_q, axis=1) > 20 * np.pi / 180):
            raise RuntimeError(
                "The inverse kinematics shows more than 20° variance over the frame range specified."
                "Please see the animation provided to verify that the subject does not move during the static trial."
                "If not, please make sure the model and subject are not positioned close to singularities (gimbal lock)."
            )

        q_static = np.median(optimal_q, axis=1)

        return q_static, model_to_use

    def make_static_pose_the_zero(self, q_static: np.ndarray):
        if q_static.shape != (self.scaled_model.nb_q,):
            raise RuntimeError(f"The shape of q_static must be (nb_q, ), you have {q_static.shape}.")

        # Remove the fake root degrees of freedom if needed
        if self.scaled_model.root_segment.nb_q == 6 or self.scaled_model.dofs[:2] == [
            Translations.XYZ,
            Rotations.XYZ,
        ]:
            q_original = q_static
        elif self.scaled_model.root_segment.nb_q == 0:
            q_original = q_static[6:]
        else:
            raise NotImplementedError(
                "Your model has between 1 and 5 degrees of freedom in the root segment. This is not implemented yet."
            )
        self.scaled_model.modify_model_static_pose(q_original)

    def replace_markers_on_segments_local_scs(self, q: np.ndarray, model_to_use: BiomechanicalModelReal):
        if q.shape != (self.scaled_model.nb_q,):
            raise RuntimeError(f"The shape of q must be (nb_q, ), you have {q.shape}.")

        model_marker_names = self.scaled_model.marker_names
        jcs_in_global = model_to_use.forward_kinematics(q)
        for i_segment, segment in enumerate(self.scaled_model.segments):
            if segment.segment_coordinate_system is None or segment.segment_coordinate_system.is_in_global:
                raise RuntimeError(
                    "Something went wrong. Since make_static_pose_the_models_zero was set to False, the segment's coordinate system should be in the local reference frames."
                )
            for marker in segment.markers:
                marker_index = model_marker_names.index(marker.name)
                this_marker_position = self.mean_experimental_markers[:, marker_index]
                rt = jcs_in_global[segment.name][0]  # We can take the 0th since there is just one frame in q
                marker.position = rt.inverse @ this_marker_position

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
        q_static, model_to_use = self.find_static_pose(
            marker_positions,
            marker_names,
            q_regularization_weight,
            initial_static_pose,
            visualize_optimal_static_pose,
            method,
        )

        if make_static_pose_the_models_zero:
            self.make_static_pose_the_zero(q_static)
            self.replace_markers_on_segments_local_scs(
                q=np.zeros((self.scaled_model.nb_q,)), model_to_use=self.scaled_model
            )
        else:
            self.replace_markers_on_segments_local_scs(q_static, model_to_use)

    def from_biomod(
        self,
        filepath: str,
    ):
        """
        Create a biomechanical model from a biorbd model
        """
        from ..model_parser.biorbd import BiomodConfigurationParser

        configuration = BiomodConfigurationParser(filepath=filepath, original_model=self.original_model)
        return configuration.scale_tool

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

    def to_biomod(self, filepath: str, append: bool = True):

        if os.path.exists(filepath) and append:
            file = open(filepath, "a")
        else:
            file = open(filepath, "w")

        out_string = ""
        out_string += "\n\n\n"
        out_string += "// --------------------------------------------------------------\n"
        out_string += "// SEGMENT SCALING CONFIGURATION\n"
        out_string += "// --------------------------------------------------------------\n\n"

        for segment_scaling in self.scaling_segments:
            out_string += segment_scaling.to_biomod()
        out_string += "\n\n\n"

        out_string += "// --------------------------------------------------------------\n"
        out_string += "// MARKER WEIGHTS\n"
        out_string += "// --------------------------------------------------------------\n\n"
        for marker_name in self.marker_weights.keys():
            out_string += f"markerweight\t{marker_name}\t{self.marker_weights[marker_name].weight}\n"

        file.write(out_string)
        file.close()

    def to_xml(self, filepath: str):

        # Create the root element
        root = ET.Element("OpenSimDocument", Version="40500")

        # Create the ScaleTool element and its children
        scale_tool = ET.SubElement(root, "ScaleTool", name="scale_tool")

        # Create the GenericModelMaker element and its children
        generic_model_maker = ET.SubElement(scale_tool, "GenericModelMaker")
        ET.SubElement(generic_model_maker, "model_file").text = self.original_model.filepath
        ET.SubElement(generic_model_maker, "marker_set_file").text = "Unassigned"

        # Create the ModelScaler element and its children
        model_scaler = ET.SubElement(scale_tool, "ModelScaler")
        ET.SubElement(model_scaler, "apply").text = "true"
        ET.SubElement(model_scaler, "scaling_order").text = "measurements"

        # Create the MeasurementSet element and its children
        measurement_set = ET.SubElement(model_scaler, "MeasurementSet")
        objects = ET.SubElement(measurement_set, "objects")

        for segment_scaling in self.scaling_segments:
            segment_scaling.to_xml(objects)

        # Create the MarkerPlacer element and its children
        marker_placer = ET.SubElement(scale_tool, "MarkerPlacer")

        # Add apply element
        ET.SubElement(marker_placer, "apply").text = "true"

        # Create the IKTaskSet element and its children
        ik_task_set = ET.SubElement(marker_placer, "IKTaskSet")
        ik_objects = ET.SubElement(ik_task_set, "objects")

        # Write the marker weights
        for marker_name in self.marker_weights.keys():
            ik_marker_task = ET.SubElement(ik_objects, "IKMarkerTask", name=marker_name)
            ET.SubElement(ik_marker_task, "apply").text = "true"
            ET.SubElement(ik_marker_task, "weight").text = str(self.marker_weights[marker_name].weight)

        # Write the XML string to a file with the usual indentation
        with open(filepath, "w", encoding="UTF-8") as f:
            xml_str = minidom.parseString(ET.tostring(root, "utf-8")).toprettyxml(indent="    ")
            f.write(xml_str)
