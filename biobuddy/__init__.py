# Version
from .version import __version__

# The actual model to inherit from
from .biomechanical_model import BiomechanicalModel

# Some classes to define the BiomechanicalModel
from biobuddy.components.axis import Axis
from biobuddy.components.inertia_parameters import InertiaParameters
from biobuddy.components.marker import Marker
from biobuddy.components.contact import Contact
from biobuddy.components.muscle import Muscle
from biobuddy.components.muscle_group import MuscleGroup
from biobuddy.components.via_point import ViaPoint
from biobuddy.components.mesh import Mesh
from biobuddy.components.mesh_file import MeshFile
from biobuddy.components.rotations import Rotations
from biobuddy.components.range_of_motion import RangeOfMotion, Ranges
from biobuddy.components.segment import Segment
from biobuddy.components.segment_coordinate_system import SegmentCoordinateSystem
from biobuddy.components.translations import Translations

# Add also the "Real" version of classes to create models from values
from biobuddy.biomechanical_model_real import BiomechanicalModelReal
from biobuddy.components.axis_real import AxisReal
from biobuddy.components.marker_real import MarkerReal
from biobuddy.components.contact_real import ContactReal
from biobuddy.components.muscle_real import MuscleReal, MuscleType, MuscleStateType
from biobuddy.components.via_point_real import ViaPointReal
from biobuddy.components.mesh_real import MeshReal
from biobuddy.components.mesh_file_real import MeshFileReal
from biobuddy.components.segment_real import SegmentReal
from biobuddy.components.segment_coordinate_system_real import SegmentCoordinateSystemReal
from biobuddy.components.inertia_parameters_real import InertiaParametersReal

# The accepted data formating
from biobuddy.c3d_data import C3dData
from biobuddy.protocols import Data, GenericDynamicModel

# Segment predefined characteristics
from biobuddy.characteristics.de_leva import DeLevaTable

# Mesh modifications
from biobuddy.mesh_modifications.vtp_parser import VtpParser