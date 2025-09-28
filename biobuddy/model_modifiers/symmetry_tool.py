import numpy as np

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..utils.enums import Translations

# NOTE: The inertia parameters are not modified !
# TODO: change muscle parameters !!

AXIS_TO_INDEX = {
    Translations.X: 0,
    Translations.Y: 1,
    Translations.Z: 2
}

class FlatteningTool:
    """
    A tool to flatten a model to make it planar (3D model -> 2D model).
    """
    def __init__(
            self,
            model: BiomechanicalModelReal,
            axis: Translations
    ):
        self.model = model
        self.axis = axis


    def _check_model(self):
        """
        Checks that alll joint coordinates systems are aligned, otherwise raises an error.
        This is temporary, as in the future, symmetrization could be performes on different axis for each segment.
        """
        for segment in self.model.segments:
            if np.any(np.abs(segment.segment_coordinate_system.scs.rotation_matrix - np.eye(3)) > 1e-6):
                raise ValueError(f"Segment {segment.name} has a rotated coordinate system. Symmetrization is only possible if all segment coordinate systems are aligned.")

    def _modify_jcs(self):
        """
        Modify the joint coordinate systems of the model to be centered on the specified axis.
        """
        for segment in self.model.segments:
            segment.segment_coordinate_system.scs.translation[AXIS_TO_INDEX[self.axis]] = 0

    def _modify_com(self):
        """
        Modify the center of mass of the model to be centered on the specified axis.
        """
        for segment in self.model.segments:
            if segment.inertia_parameters is not None:
                segment.inertia_parameters.center_of_mass[AXIS_TO_INDEX[self.axis]] = 0

    def _modify_markers(self):
        """
        Modify the markers of the model to be centered on the specified axis.
        """
        for segment in self.model.segments:
            for marker in segment.markers:
                marker.position[AXIS_TO_INDEX[self.axis]] = 0

    def _modify_contacts(self):
        """
        Modify the contacts of the model to be centered on the specified axis.
        """
        for segment in self.model.segments:
            for contact in segment.contacts:
                contact.position[AXIS_TO_INDEX[self.axis]] = 0

    def _modify_imus(self):
        """
        Modify the imus of the model to be centered on the specified axis.
        """
        for segment in self.model.segments:
            for imu in segment.imus:
               imu.scs.translation[AXIS_TO_INDEX[self.axis]] = 0

    def _modify_muscles(self):
        """
        Modify the muscles of the model to be centered on the specified axis.
        """
        for muscle_group in self.model.muscle_groups:
            for muscle in muscle_group.muscles:
                muscle.origin_position.position[AXIS_TO_INDEX[self.axis]] = 0
                for via_point in muscle.via_points:
                    via_point.position[AXIS_TO_INDEX[self.axis]] = 0
                muscle.insertion_position.position[AXIS_TO_INDEX[self.axis]] = 0

    def flatten(self) -> BiomechanicalModelReal:
        """
        Perform the symmetrization of the model, meaning that for each segment, the joint coordinate systems, markers,
        contacts, muscles, etc. are shifted to the zero on the axis specified.
        """
        self._check_model()
        self._modify_jcs()
        self._modify_com()
        self._modify_markers()
        self._modify_contacts()
        self._modify_imus()
        self._modify_muscles()
        return self.model