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
from ..utils.enums import Rotations, Translations


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
        measurements: Mapping[str, float] | str | Path | None = None,
        configuration: Mapping[str, float] | str | Path | None = None,
        symmetric: bool = True,
        density_set: YeadonDensitySet | str = YeadonDensitySet.DEMPSTER,
        total_mass: float | None = None,
    ):
        """
        Compute subject-specific segment inertia parameters using the Yeadon model.

        Parameters
        ----------
        measurements
            Either a mapping with Yeadon's 95 measurement names in meters, or a path to a Yeadon measurement file.
        configuration
            Optional Yeadon configuration mapping or file. If omitted, the neutral configuration is used.
        symmetric
            If true, Yeadon averages left/right limb measurements before computing inertia parameters.
        density_set
            One of Chandler, Clauser, or Dempster density sets.
        total_mass
            Optional measured total mass in kilograms. When provided, Yeadon's density model is scaled to this mass.
        """
        self.measurements = None
        self.configuration = None
        self.symmetric = symmetric
        self.density_set = density_set.value if isinstance(density_set, YeadonDensitySet) else density_set
        self.total_mass = total_mass
        self.human = None
        self.inertial_table: dict[YeadonSegmentName, InertiaParametersReal] = {}
        if measurements is not None:
            self.from_measurement(
                measurements=measurements,
                configuration=configuration,
                symmetric=symmetric,
                density_set=density_set,
                total_mass=total_mass,
            )

    def from_measurement(
        self,
        measurements: Mapping[str, float] | str | Path,
        configuration: Mapping[str, float] | str | Path | None = None,
        symmetric: bool = True,
        density_set: YeadonDensitySet | str = YeadonDensitySet.DEMPSTER,
        total_mass: float | None = None,
    ) -> None:
        """
        Compute the Yeadon inertial table from anthropometric measurements.

        Parameters
        ----------
        measurements
            Either a mapping with Yeadon's 95 measurement names in meters, or a path to a Yeadon measurement file.
        configuration
            Optional Yeadon configuration mapping or file. If omitted, the neutral configuration is used.
        symmetric
            If true, Yeadon averages left/right limb measurements before computing inertia parameters.
        density_set
            One of Chandler, Clauser, or Dempster density sets.
        total_mass
            Optional measured total mass in kilograms. When provided, Yeadon's density model is scaled to this mass.
        """
        yeadon = _import_yeadon()
        density_set = density_set.value if isinstance(density_set, YeadonDensitySet) else density_set
        measurements = str(measurements) if isinstance(measurements, Path) else measurements
        configuration = str(configuration) if isinstance(configuration, Path) else configuration

        self.measurements = measurements
        self.configuration = configuration
        self.symmetric = symmetric
        self.density_set = density_set
        self.total_mass = total_mass
        self.human = yeadon.Human(
            measurements,
            CFG=configuration,
            symmetric=symmetric,
            density_set=density_set,
        )
        if total_mass is not None:
            self.human.scale_human_by_mass(total_mass)

        self.inertial_table = {
            segment_name: self._inertia_parameters_from_segment(self._yeadon_segment(segment_name))
            for segment_name in YeadonSegmentName
        }

    def from_measurements(
        self,
        measurements: Mapping[str, float] | str | Path,
        configuration: Mapping[str, float] | str | Path | None = None,
        symmetric: bool = True,
        density_set: YeadonDensitySet | str = YeadonDensitySet.DEMPSTER,
        total_mass: float | None = None,
    ) -> None:
        """
        Alias for :meth:`from_measurement`.
        """
        self.from_measurement(
            measurements=measurements,
            configuration=configuration,
            symmetric=symmetric,
            density_set=density_set,
            total_mass=total_mass,
        )

    def from_file(
        self,
        filepath: str | Path,
        configuration: Mapping[str, float] | str | Path | None = None,
        symmetric: bool = True,
        density_set: YeadonDensitySet | str = YeadonDensitySet.DEMPSTER,
        total_mass: float | None = None,
    ) -> None:
        """
        Compute the Yeadon inertial table from a Yeadon measurement text file.
        """
        self.from_measurement(
            measurements=filepath,
            configuration=configuration,
            symmetric=symmetric,
            density_set=density_set,
            total_mass=total_mass,
        )

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

    @staticmethod
    def measurement_specs() -> tuple[YeadonMeasurementSpec, ...]:
        return YEADON_MEASUREMENT_SPECS

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

    def _measurement_mapping_for_export(self) -> Mapping[str, float]:
        if isinstance(self.measurements, Mapping):
            measurements = self.measurements
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
