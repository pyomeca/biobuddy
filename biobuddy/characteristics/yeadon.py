from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ..components.real.rigidbody.segment_coordinate_system_real import (
    SegmentCoordinateSystemReal,
)
from ..components.real.rigidbody.segment_real import SegmentReal
from ..utils.enums import Rotations, Translations, LengthUnits


class YeadonSegmentName(Enum):
    PELVIS = "P"
    THORAX = "T"
    CHEST_HEAD = "C"
    LEFT_UPPER_ARM = "A1"
    LEFT_FOREARM_HAND = "A2"
    RIGHT_UPPER_ARM = "B1"
    RIGHT_FOREARM_HAND = "B2"
    LEFT_THIGH = "J1"
    LEFT_SHANK_FOOT = "J2"
    RIGHT_THIGH = "K1"
    RIGHT_SHANK_FOOT = "K2"


class YeadonDensitySet(Enum):
    CHANDLER = "Chandler"
    CLAUSER = "Clauser"
    DEMPSTER = "Dempster"


class YeadonMeasures:
    def __init__(
        self,
        Ls1L: float,
        Ls2L: float,
        Ls3L: float,
        Ls4L: float,
        Ls5L: float,
        Ls6L: float,
        Ls7L: float,
        Ls8L: float,
        Ls0p: float,
        Ls1p: float,
        Ls2p: float,
        Ls3p: float,
        Ls5p: float,
        Ls6p: float,
        Ls7p: float,
        Ls0w: float,
        Ls1w: float,
        Ls2w: float,
        Ls3w: float,
        Ls4w: float,
        Ls4d: float,
        La2L: float,
        La3L: float,
        La4L: float,
        La5L: float,
        La6L: float,
        La7L: float,
        La0p: float,
        La1p: float,
        La2p: float,
        La3p: float,
        La4p: float,
        La5p: float,
        La6p: float,
        La7p: float,
        La4w: float,
        La5w: float,
        La6w: float,
        La7w: float,
        Lb2L: float,
        Lb3L: float,
        Lb4L: float,
        Lb5L: float,
        Lb6L: float,
        Lb7L: float,
        Lb0p: float,
        Lb1p: float,
        Lb2p: float,
        Lb3p: float,
        Lb4p: float,
        Lb5p: float,
        Lb6p: float,
        Lb7p: float,
        Lb4w: float,
        Lb5w: float,
        Lb6w: float,
        Lb7w: float,
        Lj1L: float,
        Lj3L: float,
        Lj4L: float,
        Lj5L: float,
        Lj6L: float,
        Lj8L: float,
        Lj9L: float,
        Lj1p: float,
        Lj2p: float,
        Lj3p: float,
        Lj4p: float,
        Lj5p: float,
        Lj6p: float,
        Lj7p: float,
        Lj8p: float,
        Lj9p: float,
        Lj8w: float,
        Lj9w: float,
        Lj6d: float,
        Lk1L: float,
        Lk3L: float,
        Lk4L: float,
        Lk5L: float,
        Lk6L: float,
        Lk8L: float,
        Lk9L: float,
        Lk1p: float,
        Lk2p: float,
        Lk3p: float,
        Lk4p: float,
        Lk5p: float,
        Lk6p: float,
        Lk7p: float,
        Lk8p: float,
        Lk9p: float,
        Lk8w: float,
        Lk9w: float,
        Lk6d: float,
        units: LengthUnits = LengthUnits.M,
    ):
        if units == LengthUnits.M:
            multiplier = 1
        elif units == LengthUnits.CM:
            multiplier = 0.01
        elif units == LengthUnits.MM:
            multiplier = 0.001
        else:
            raise ValueError(f"The units should be an LengthUnits enum, not {units}.")

        self.Ls1L = Ls1L * multiplier
        self.Ls2L = Ls2L * multiplier
        self.Ls3L = Ls3L * multiplier
        self.Ls4L = Ls4L * multiplier
        self.Ls5L = Ls5L * multiplier
        self.Ls6L = Ls6L * multiplier
        self.Ls7L = Ls7L * multiplier
        self.Ls8L = Ls8L * multiplier
        self.Ls0p = Ls0p * multiplier
        self.Ls1p = Ls1p * multiplier
        self.Ls2p = Ls2p * multiplier
        self.Ls3p = Ls3p * multiplier
        self.Ls5p = Ls5p * multiplier
        self.Ls6p = Ls6p * multiplier
        self.Ls7p = Ls7p * multiplier
        self.Ls0w = Ls0w * multiplier
        self.Ls1w = Ls1w * multiplier
        self.Ls2w = Ls2w * multiplier
        self.Ls3w = Ls3w * multiplier
        self.Ls4w = Ls4w * multiplier
        self.Ls4d = Ls4d * multiplier
        self.La2L = La2L * multiplier
        self.La3L = La3L * multiplier
        self.La4L = La4L * multiplier
        self.La5L = La5L * multiplier
        self.La6L = La6L * multiplier
        self.La7L = La7L * multiplier
        self.La0p = La0p * multiplier
        self.La1p = La1p * multiplier
        self.La2p = La2p * multiplier
        self.La3p = La3p * multiplier
        self.La4p = La4p * multiplier
        self.La5p = La5p * multiplier
        self.La6p = La6p * multiplier
        self.La7p = La7p * multiplier
        self.La4w = La4w * multiplier
        self.La5w = La5w * multiplier
        self.La6w = La6w * multiplier
        self.La7w = La7w * multiplier
        self.Lb2L = Lb2L * multiplier
        self.Lb3L = Lb3L * multiplier
        self.Lb4L = Lb4L * multiplier
        self.Lb5L = Lb5L * multiplier
        self.Lb6L = Lb6L * multiplier
        self.Lb7L = Lb7L * multiplier
        self.Lb0p = Lb0p * multiplier
        self.Lb1p = Lb1p * multiplier
        self.Lb2p = Lb2p * multiplier
        self.Lb3p = Lb3p * multiplier
        self.Lb4p = Lb4p * multiplier
        self.Lb5p = Lb5p * multiplier
        self.Lb6p = Lb6p * multiplier
        self.Lb7p = Lb7p * multiplier
        self.Lb4w = Lb4w * multiplier
        self.Lb5w = Lb5w * multiplier
        self.Lb6w = Lb6w * multiplier
        self.Lb7w = Lb7w * multiplier
        self.Lj1L = Lj1L * multiplier
        self.Lj3L = Lj3L * multiplier
        self.Lj4L = Lj4L * multiplier
        self.Lj5L = Lj5L * multiplier
        self.Lj6L = Lj6L * multiplier
        self.Lj8L = Lj8L * multiplier
        self.Lj9L = Lj9L * multiplier
        self.Lj1p = Lj1p * multiplier
        self.Lj2p = Lj2p * multiplier
        self.Lj3p = Lj3p * multiplier
        self.Lj4p = Lj4p * multiplier
        self.Lj5p = Lj5p * multiplier
        self.Lj6p = Lj6p * multiplier
        self.Lj7p = Lj7p * multiplier
        self.Lj8p = Lj8p * multiplier
        self.Lj9p = Lj9p * multiplier
        self.Lj8w = Lj8w * multiplier
        self.Lj9w = Lj9w * multiplier
        self.Lj6d = Lj6d * multiplier
        self.Lk1L = Lk1L * multiplier
        self.Lk3L = Lk3L * multiplier
        self.Lk4L = Lk4L * multiplier
        self.Lk5L = Lk5L * multiplier
        self.Lk6L = Lk6L * multiplier
        self.Lk8L = Lk8L * multiplier
        self.Lk9L = Lk9L * multiplier
        self.Lk1p = Lk1p * multiplier
        self.Lk2p = Lk2p * multiplier
        self.Lk3p = Lk3p * multiplier
        self.Lk4p = Lk4p * multiplier
        self.Lk5p = Lk5p * multiplier
        self.Lk6p = Lk6p * multiplier
        self.Lk7p = Lk7p * multiplier
        self.Lk8p = Lk8p * multiplier
        self.Lk9p = Lk9p * multiplier
        self.Lk8w = Lk8w * multiplier
        self.Lk9w = Lk9w * multiplier
        self.Lk6d = Lk6d * multiplier


@dataclass(frozen=True)
class YeadonMeasurementSpec:
    name: str
    group: str
    label: str
    description: str
    kind: str


YEADON_MEASUREMENT_NAMES = (
    "Ls1L",
    "Ls2L",
    "Ls3L",
    "Ls4L",
    "Ls5L",
    "Ls6L",
    "Ls7L",
    "Ls8L",
    "Ls0p",
    "Ls1p",
    "Ls2p",
    "Ls3p",
    "Ls5p",
    "Ls6p",
    "Ls7p",
    "Ls0w",
    "Ls1w",
    "Ls2w",
    "Ls3w",
    "Ls4w",
    "Ls4d",
    "La2L",
    "La3L",
    "La4L",
    "La5L",
    "La6L",
    "La7L",
    "La0p",
    "La1p",
    "La2p",
    "La3p",
    "La4p",
    "La5p",
    "La6p",
    "La7p",
    "La4w",
    "La5w",
    "La6w",
    "La7w",
    "Lb2L",
    "Lb3L",
    "Lb4L",
    "Lb5L",
    "Lb6L",
    "Lb7L",
    "Lb0p",
    "Lb1p",
    "Lb2p",
    "Lb3p",
    "Lb4p",
    "Lb5p",
    "Lb6p",
    "Lb7p",
    "Lb4w",
    "Lb5w",
    "Lb6w",
    "Lb7w",
    "Lj1L",
    "Lj3L",
    "Lj4L",
    "Lj5L",
    "Lj6L",
    "Lj8L",
    "Lj9L",
    "Lj1p",
    "Lj2p",
    "Lj3p",
    "Lj4p",
    "Lj5p",
    "Lj6p",
    "Lj7p",
    "Lj8p",
    "Lj9p",
    "Lj8w",
    "Lj9w",
    "Lj6d",
    "Lk1L",
    "Lk3L",
    "Lk4L",
    "Lk5L",
    "Lk6L",
    "Lk8L",
    "Lk9L",
    "Lk1p",
    "Lk2p",
    "Lk3p",
    "Lk4p",
    "Lk5p",
    "Lk6p",
    "Lk7p",
    "Lk8p",
    "Lk9p",
    "Lk8w",
    "Lk9w",
    "Lk6d",
)

_MEASUREMENT_GROUPS = {
    "Ls": ("Torso", "from the pelvis and trunk/head stadium levels"),
    "La": ("Left arm", "from the left shoulder, elbow, wrist, and hand stadium levels"),
    "Lb": (
        "Right arm",
        "from the right shoulder, elbow, wrist, and hand stadium levels",
    ),
    "Lj": (
        "Left leg",
        "from the left hip, knee, ankle, heel, ball, and toe stadium levels",
    ),
    "Lk": (
        "Right leg",
        "from the right hip, knee, ankle, heel, ball, and toe stadium levels",
    ),
}

_MEASUREMENT_KIND = {
    "L": ("length", "Longitudinal distance in meters"),
    "p": ("perimeter", "Body perimeter in meters"),
    "w": ("width", "Mediolateral width in meters"),
    "d": ("depth", "Anteroposterior depth in meters"),
}


def _measurement_spec(name: str) -> YeadonMeasurementSpec:
    group, group_description = _MEASUREMENT_GROUPS[name[:2]]
    kind_key = name[-1]
    kind, kind_description = _MEASUREMENT_KIND[kind_key]
    return YeadonMeasurementSpec(
        name=name,
        group=group,
        label=f"{name} ({kind})",
        description=f"{kind_description} {group_description}.",
        kind=kind,
    )


YEADON_MEASUREMENT_SPECS = tuple(_measurement_spec(name) for name in YEADON_MEASUREMENT_NAMES)


class YeadonTable:
    def __init__(
        self,
        symmetric: bool = True,
        density_set: YeadonDensitySet | str = YeadonDensitySet.DEMPSTER,
        total_mass: float | None = None,
    ):
        """
        Compute subject-specific segment inertia parameters using the Yeadon model.

        Parameters
        ----------
        symmetric
            If true, Yeadon averages left/right limb measurements before computing inertia parameters.
        density_set
            One of Chandler, Clauser, or Dempster density sets.
        total_mass
            Optional measured total mass in kilograms. When provided, Yeadon's density model is scaled to this mass.
        """
        self.symmetric = symmetric
        self.density_set = density_set.value if isinstance(density_set, YeadonDensitySet) else density_set
        self.total_mass = total_mass

        # The following attributes will be set by from_measurements
        self.human = None
        self.inertial_table: dict[YeadonSegmentName, InertiaParametersReal] = {}
        self.measures: YeadonMeasures = None

        # The following attributes will be set by get_joint_position_from_measurements
        self.pelvis_position: np.ndarray = None
        self.thorax_position: np.ndarray = None
        self.chest_head_position: np.ndarray = None
        self.top_head_position: np.ndarray = None
        self.right_shoulder_position: np.ndarray = None
        self.right_elbow_position: np.ndarray = None
        self.right_wrist_position: np.ndarray = None
        self.left_shoulder_position: np.ndarray = None
        self.left_elbow_position: np.ndarray = None
        self.left_wrist_position: np.ndarray = None
        self.right_hip_position: np.ndarray = None
        self.right_knee_position: np.ndarray = None
        self.right_ankle_position: np.ndarray = None
        self.left_hip_position: np.ndarray = None
        self.left_knee_position: np.ndarray = None
        self.left_ankle_position: np.ndarray = None

    def define_inertial_table(self):
        """
        Define the inertial characteristics of the segments based on the Yeadon anthropometric model.
        """
        yeadon = _import_yeadon()

        self.human = yeadon.Human(
            dict(vars(self.measures)),
            symmetric=self.symmetric,
            density_set=self.density_set,
        )
        if self.total_mass is not None:
            self.human.scale_human_by_mass(self.total_mass)

        self.inertial_table = self._inertial_table_from_human()

    def get_joint_position_from_measurements(self) -> None:
        """
        Define the position of the joint centers based on the Yeadon anthropometric model.
        This must be called after define_inertial_table since the joint centers are computed by the
        yeadon package while building the Human model.
        """
        self.pelvis_position = self.segment_origin(YeadonSegmentName.PELVIS)
        self.thorax_position = self.segment_origin(YeadonSegmentName.THORAX)
        self.chest_head_position = self.segment_origin(YeadonSegmentName.CHEST_HEAD)
        self.top_head_position = self.segment_end(YeadonSegmentName.CHEST_HEAD)

        # Right arm
        self.right_shoulder_position = self.segment_origin(YeadonSegmentName.RIGHT_UPPER_ARM)
        self.right_elbow_position = self.segment_origin(YeadonSegmentName.RIGHT_FOREARM_HAND)
        self.right_wrist_position = self.segment_end(YeadonSegmentName.RIGHT_FOREARM_HAND)

        # Left arm
        self.left_shoulder_position = self.segment_origin(YeadonSegmentName.LEFT_UPPER_ARM)
        self.left_elbow_position = self.segment_origin(YeadonSegmentName.LEFT_FOREARM_HAND)
        self.left_wrist_position = self.segment_end(YeadonSegmentName.LEFT_FOREARM_HAND)

        # Right leg
        self.right_hip_position = self.segment_origin(YeadonSegmentName.RIGHT_THIGH)
        self.right_knee_position = self.segment_origin(YeadonSegmentName.RIGHT_SHANK_FOOT)
        self.right_ankle_position = self.segment_end(YeadonSegmentName.RIGHT_SHANK_FOOT)

        # Left leg
        self.left_hip_position = self.segment_origin(YeadonSegmentName.LEFT_THIGH)
        self.left_knee_position = self.segment_origin(YeadonSegmentName.LEFT_SHANK_FOOT)
        self.left_ankle_position = self.segment_end(YeadonSegmentName.LEFT_SHANK_FOOT)

    def from_measurements(
        self,
        measures: YeadonMeasures,
    ) -> None:
        """
        Create the Yeadon inertial table from anthropometric measurements.

        Parameters
        ----------
        measures
            Yeadon's 95 measurements in meters
        """

        self.measures = measures

        self.define_inertial_table()
        self.get_joint_position_from_measurements()

    def from_file(
        self,
        filepath: str | Path,
    ) -> None:
        """
        Create the Yeadon inertial table from a Yeadon measurement text file.

        Parameters
        ----------
        filepath
            Path to a YAML-style measurement file, as produced by `to_file`: one `name: value` line
            per measurement, a `measurementconversionfactor` (e.g., 0.01 if the measurements are in
            centimeters), and an optional `totalmass` (in kilograms).
        """
        yeadon = _import_yeadon()

        self.human = yeadon.Human(
            str(filepath),
            symmetric=self.symmetric,
            density_set=self.density_set,
        )
        self.measures = YeadonMeasures(**{name: float(self.human.meas[name]) for name in YEADON_MEASUREMENT_NAMES})
        if self.total_mass is not None:
            self.human.scale_human_by_mass(self.total_mass)
        elif self.human.meas_mass > 0:
            self.total_mass = float(self.human.meas_mass)

        self.inertial_table = self._inertial_table_from_human()
        self.get_joint_position_from_measurements()

    def to_file(self, filepath: str | Path) -> None:
        """
        Export the current measurements to the YAML-style text format accepted by yeadon.
        """
        measurements = self._measurement_mapping_for_export()
        lines = [f"{name}: {_format_yeadon_file_value(measurements[name])}\n" for name in YEADON_MEASUREMENT_NAMES]
        total_mass = self._total_mass_for_export()
        if total_mass is not None:
            lines.append(f"totalmass: {_format_yeadon_file_value(total_mass)}\n")
        lines.append("measurementconversionfactor: 1\n")
        Path(filepath).write_text("".join(lines))

    @property
    def mass(self) -> float:
        return float(self.human.mass)

    @property
    def center_of_mass(self) -> np.ndarray:
        return np.asarray(self.human.center_of_mass, dtype=float)

    @property
    def inertia(self) -> np.ndarray:
        return np.asarray(self.human.inertia, dtype=float)

    @staticmethod
    def measurement_names() -> tuple[str, ...]:
        return YEADON_MEASUREMENT_NAMES

    def __getitem__(self, segment_name: YeadonSegmentName | str) -> InertiaParametersReal:
        return self.inertial_table[_coerce_segment_name(segment_name)]

    def segment_origin(self, segment_name: YeadonSegmentName | str) -> np.ndarray:
        segment = self._yeadon_segment(_coerce_segment_name(segment_name))
        return _homogeneous_point(segment.pos)

    def segment_end(self, segment_name: YeadonSegmentName | str) -> np.ndarray:
        segment = self._yeadon_segment(_coerce_segment_name(segment_name))
        return _homogeneous_point(segment.end_pos)

    def to_simple_model(self) -> BiomechanicalModelReal:
        model = BiomechanicalModelReal()
        for segment_name in YeadonSegmentName:
            yeadon_segment = self._yeadon_segment(segment_name)
            rt_matrix = np.eye(4)
            rt_matrix[:3, :3] = np.asarray(yeadon_segment.rot_mat, dtype=float)
            rt_matrix[:3, 3] = np.asarray(yeadon_segment.pos, dtype=float).reshape(3)
            model.add_segment(
                SegmentReal(
                    name=segment_name.value,
                    parent_name=_YEADON_PARENT_SEGMENTS[segment_name],
                    translations=_YEADON_TRANSLATIONS[segment_name],
                    rotations=_YEADON_ROTATIONS[segment_name],
                    inertia_parameters=self[segment_name],
                    segment_coordinate_system=SegmentCoordinateSystemReal.from_rt_matrix(
                        rt_matrix=rt_matrix,
                        is_scs_local=False,
                    ),
                )
            )
        model.segments_rt_to_local()
        return model

    def _yeadon_segment(self, segment_name: YeadonSegmentName) -> Any:
        return getattr(self.human, segment_name.value)

    def _inertial_table_from_human(self) -> dict[YeadonSegmentName, InertiaParametersReal]:
        return {
            segment_name: self._inertia_parameters_from_segment(self._yeadon_segment(segment_name))
            for segment_name in YeadonSegmentName
        }

    def _measurement_mapping_for_export(self) -> Mapping[str, float]:
        if self.measures is not None:
            measurements = vars(self.measures)
        elif self.human is not None and isinstance(getattr(self.human, "meas", None), Mapping):
            measurements = self.human.meas
        else:
            raise ValueError("Yeadon measurements are not available for export.")
        missing = [name for name in YEADON_MEASUREMENT_NAMES if name not in measurements]
        if missing:
            raise ValueError(f"Missing Yeadon measurements: {', '.join(missing)}.")
        return measurements

    def _total_mass_for_export(self) -> float | None:
        if self.total_mass is not None:
            return self.total_mass
        if self.human is not None:
            measured_mass = getattr(self.human, "meas_mass", None)
            if measured_mass is not None and measured_mass > 0:
                return measured_mass
        return None

    @staticmethod
    def _inertia_parameters_from_segment(segment: Any) -> InertiaParametersReal:
        return InertiaParametersReal(
            mass=float(segment.mass),
            center_of_mass=np.asarray(segment.rel_center_of_mass, dtype=float).reshape(3),
            inertia=np.asarray(segment.rel_inertia, dtype=float).reshape(3, 3),
        )


_YEADON_PARENT_SEGMENTS = {
    YeadonSegmentName.PELVIS: "base",
    YeadonSegmentName.THORAX: YeadonSegmentName.PELVIS.value,
    YeadonSegmentName.CHEST_HEAD: YeadonSegmentName.THORAX.value,
    YeadonSegmentName.LEFT_UPPER_ARM: YeadonSegmentName.CHEST_HEAD.value,
    YeadonSegmentName.LEFT_FOREARM_HAND: YeadonSegmentName.LEFT_UPPER_ARM.value,
    YeadonSegmentName.RIGHT_UPPER_ARM: YeadonSegmentName.CHEST_HEAD.value,
    YeadonSegmentName.RIGHT_FOREARM_HAND: YeadonSegmentName.RIGHT_UPPER_ARM.value,
    YeadonSegmentName.LEFT_THIGH: YeadonSegmentName.PELVIS.value,
    YeadonSegmentName.LEFT_SHANK_FOOT: YeadonSegmentName.LEFT_THIGH.value,
    YeadonSegmentName.RIGHT_THIGH: YeadonSegmentName.PELVIS.value,
    YeadonSegmentName.RIGHT_SHANK_FOOT: YeadonSegmentName.RIGHT_THIGH.value,
}

_YEADON_TRANSLATIONS = {
    YeadonSegmentName.PELVIS: Translations.XYZ,
    YeadonSegmentName.THORAX: Translations.NONE,
    YeadonSegmentName.CHEST_HEAD: Translations.NONE,
    YeadonSegmentName.LEFT_UPPER_ARM: Translations.NONE,
    YeadonSegmentName.LEFT_FOREARM_HAND: Translations.NONE,
    YeadonSegmentName.RIGHT_UPPER_ARM: Translations.NONE,
    YeadonSegmentName.RIGHT_FOREARM_HAND: Translations.NONE,
    YeadonSegmentName.LEFT_THIGH: Translations.NONE,
    YeadonSegmentName.LEFT_SHANK_FOOT: Translations.NONE,
    YeadonSegmentName.RIGHT_THIGH: Translations.NONE,
    YeadonSegmentName.RIGHT_SHANK_FOOT: Translations.NONE,
}

_YEADON_ROTATIONS = {
    YeadonSegmentName.PELVIS: Rotations.XYZ,
    YeadonSegmentName.THORAX: Rotations.XY,
    YeadonSegmentName.CHEST_HEAD: Rotations.XZ,
    YeadonSegmentName.LEFT_UPPER_ARM: Rotations.XYZ,
    YeadonSegmentName.LEFT_FOREARM_HAND: Rotations.X,
    YeadonSegmentName.RIGHT_UPPER_ARM: Rotations.XYZ,
    YeadonSegmentName.RIGHT_FOREARM_HAND: Rotations.X,
    YeadonSegmentName.LEFT_THIGH: Rotations.XY,
    YeadonSegmentName.LEFT_SHANK_FOOT: Rotations.X,
    YeadonSegmentName.RIGHT_THIGH: Rotations.XY,
    YeadonSegmentName.RIGHT_SHANK_FOOT: Rotations.X,
}


def _coerce_segment_name(segment_name: YeadonSegmentName | str) -> YeadonSegmentName:
    if isinstance(segment_name, YeadonSegmentName):
        return segment_name
    try:
        return YeadonSegmentName[segment_name]
    except KeyError:
        return YeadonSegmentName(segment_name)


def _homogeneous_point(point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=float).reshape(3)
    return np.array([point[0], point[1], point[2], 1.0])


def _format_yeadon_file_value(value: float) -> str:
    return f"{float(value):.15g}"


def _import_yeadon() -> Any:
    try:
        import yeadon
    except ImportError as error:
        raise ImportError(
            "YeadonTable requires the `yeadon` package. Install it with `pip install yeadon` "
            "or install BioBuddy with its declared dependencies."
        ) from error
    return yeadon
