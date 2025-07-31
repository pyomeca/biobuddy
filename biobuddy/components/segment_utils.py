from .generic.rigidbody.range_of_motion import RangeOfMotion
from ..utils.enums import Rotations, Translations
from ..utils.named_list import NamedList


class SegmentUtils:
    def __init__(self):
        self.name = None
        self.parent_name = None
        self.segment_coordinate_system = None
        self.translations = None
        self.rotations = None
        self.q_ranges = None
        self.qdot_ranges = None
        self.markers = NamedList()
        self.contacts = NamedList()
        self.imus = NamedList()
        self.inertia_parameters = None
        self.mesh = None
        self.mesh_file = None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def parent_name(self) -> str:
        return self._parent_name

    @parent_name.setter
    def parent_name(self, value: str):
        self._parent_name = value

    @property
    def translations(self) -> Translations:
        return self._translations

    @translations.setter
    def translations(self, value: Translations | str | None):
        if value is None or isinstance(value, str):
            value = Translations(value)
        self._translations = value

    @property
    def rotations(self) -> Rotations:
        return self._rotations

    @rotations.setter
    def rotations(self, value: Rotations | str | None):
        if value is None or isinstance(value, str):
            value = Rotations(value)
        self._rotations = value

    @property
    def q_ranges(self) -> RangeOfMotion:
        return self._q_ranges

    @q_ranges.setter
    def q_ranges(self, value: RangeOfMotion):
        self._q_ranges = value

    @property
    def qdot_ranges(self) -> RangeOfMotion:
        return self._qdot_ranges

    @qdot_ranges.setter
    def qdot_ranges(self, value: RangeOfMotion):
        self._qdot_ranges = value

    @property
    def marker_names(self):
        return [marker.name for marker in self.markers]

    @property
    def contact_names(self):
        return [contact.name for contact in self.contacts]

    @property
    def imu_names(self):
        return [imu.name for imu in self.imus]

    @property
    def nb_markers(self):
        return len(self.markers)

    @property
    def nb_contacts(self):
        return len(self.contacts)

    @property
    def nb_imus(self):
        return len(self.imus)

    @property
    def nb_q(self):
        nb_translations = 0
        if self.translations is not None and self.translations != Translations.NONE:
            nb_translations = len(self.translations.value)
        nb_rotations = 0
        if self.rotations is not None and self.rotations != Rotations.NONE:
            nb_rotations = len(self.rotations.value)
        return nb_translations + nb_rotations

    def update_dof_names(self):
        """Update the dof_names property based on translations and rotations added after first initialization of the model."""
        self.dof_names = None
