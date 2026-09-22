import numpy as np
import numpy.testing as npt
import pytest
from dataclasses import fields

from biobuddy import (
    BiomechanicalModelReal,
    SegmentEditorData,
    available_inertial_models,
    apply_segment_editor_data,
    build_inertial_parameters_from_model,
    get_segment_editor_data,
    inertial_model_segment_names,
    SegmentCoordinateSystemReal,
    RotoTransMatrix,
)
from biobuddy.gui.segment_editor import validate_parent_name
from biobuddy.gui.segment_editor import load_model
from biobuddy.components.generic.rigidbody.range_of_motion import RangeOfMotion, Ranges
from biobuddy.components.real.rigidbody.inertia_parameters_real import (
    InertiaParametersReal,
)
from biobuddy.components.real.rigidbody.segment_real import SegmentReal
from biobuddy.utils.enums import Rotations, Translations


def _build_segment() -> SegmentReal:
    return SegmentReal(
        name="Arm",
        parent_name="Shoulder",
        segment_coordinate_system=SegmentCoordinateSystemReal(scs=RotoTransMatrix(), is_scs_local=True),
        translations=Translations.NONE,
        rotations=Rotations.XYZ,
        dof_names=["Arm_rotX", "Arm_rotY", "Arm_rotZ"],
        q_ranges=RangeOfMotion(Ranges.Q, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
        qdot_ranges=RangeOfMotion(Ranges.Qdot, [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]),
        inertia_parameters=InertiaParametersReal(
            mass=5.0,
            center_of_mass=np.array([0.05, 0.0, 0.0]),
            inertia=np.array([[0.5, 0.1, 0.0], [0.1, 0.6, 0.2], [0.0, 0.2, 0.7]]),
        ),
        mesh=None,
        mesh_file=None,
    )


def test_get_segment_editor_data_extracts_editable_values():
    """
    Convert a populated segment into form-friendly values.
    """
    segment = SegmentReal(
        name="Thigh",
        parent_name="Pelvis",
        translations=Translations.X,
        rotations=Rotations.YZ,
        q_ranges=RangeOfMotion(Ranges.Q, [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]),
        inertia_parameters=InertiaParametersReal(
            mass=8.0,
            center_of_mass=np.array([0.1, 0.2, 0.3]),
            inertia=np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]),
        ),
    )

    data = get_segment_editor_data(segment)

    assert data.parent_name == "Pelvis"
    assert data.translations == "x"
    assert data.rotations == "yz"
    assert data.q_min == [-1.0, -2.0, -3.0]
    assert data.q_max == [1.0, 2.0, 3.0]
    assert data.mass == 8.0
    assert data.center_of_mass == [0.1, 0.2, 0.3]
    assert data.inertia_matrix == [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]
    assert data.inertia_diagonal == [1.0, 2.0, 3.0]


def test_apply_segment_editor_data_updates_segment():
    """
    Apply edited fields back to a segment.
    """
    segment = SegmentReal(name="Shank")
    data = SegmentEditorData(
        parent_name="Thigh",
        translations="",
        rotations="xz",
        q_min=[-1.0, -2.0],
        q_max=[1.0, 2.0],
        mass=4.5,
        center_of_mass=[0.0, -0.2, 0.0],
        inertia_matrix=[[0.1, 0.01, 0.02], [0.01, 0.2, 0.03], [0.02, 0.03, 0.3]],
    )

    apply_segment_editor_data(segment, data)

    assert segment.parent_name == "Thigh"
    assert segment.translations == Translations.NONE
    assert segment.rotations == Rotations.XZ
    assert segment.dof_names == ["Shank_rotX", "Shank_rotZ"]
    assert segment.q_ranges.min_bound == [-1.0, -2.0]
    assert segment.q_ranges.max_bound == [1.0, 2.0]
    assert segment.inertia_parameters.mass == 4.5
    npt.assert_array_equal(segment.inertia_parameters.center_of_mass[:3, 0], np.array([0.0, -0.2, 0.0]))
    npt.assert_array_equal(
        segment.inertia_parameters.inertia[:3, :3],
        np.array([[0.1, 0.01, 0.02], [0.01, 0.2, 0.03], [0.02, 0.03, 0.3]]),
    )


def test_apply_segment_editor_data_rejects_incompatible_ranges():
    """
    Reject edited ranges whose length no longer matches the segment DoFs.
    """
    segment = SegmentReal(name="Foot", rotations=Rotations.XY)
    data = SegmentEditorData(
        parent_name="Shank",
        translations="",
        rotations="xy",
        q_min=[-1.0],
        q_max=[1.0],
        mass=None,
        center_of_mass=[0.0, 0.0, 0.0],
        inertia_matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    with pytest.raises(ValueError, match="Expected either 0 or 2 range values"):
        apply_segment_editor_data(segment, data)


def test_validate_parent_name_rejects_unknown_and_self_parents():
    """
    Reject parent choices that would make the edited hierarchy invalid.
    """
    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="Pelvis"))
    model.add_segment(SegmentReal(name="Thigh", parent_name="Pelvis"))

    with pytest.raises(ValueError, match="Unknown parent segment"):
        validate_parent_name(model=model, segment_name="Thigh", parent_name="Missing")

    with pytest.raises(ValueError, match="cannot be its own parent"):
        validate_parent_name(model=model, segment_name="Thigh", parent_name="Thigh")


def test_segment_data_class():
    """
    Test that the SegmentEditorData class covers all the necessary fields of SegmentReal for future improvements.
    """
    editor_fields = {f"_{f.name}" for f in fields(SegmentEditorData)}
    segment = _build_segment()
    real_fields = set(vars(segment).keys()) - {
        "_name",  # TODO: to be added
        "_segment_coordinate_system",  # TODO: to be added
        "_dof_names",  # TODO: to be added
        "_q_ranges",  # -> q_min, q_max
        "_qdot_ranges",  # TODO: to be added
        "_markers",  # TODO: to be added
        "_contacts",  # TODO: to be added
        "_imus",  # TODO: to be added
        "_inertia_parameters",  # -> mass, center_of_mass, inertia_matrix
        "_mesh",  # -> TODO: to be added
        "_mesh_file",  # -> TODO: to be added
        "mesh_file",  # TODO: to be added
    }
    derived_editor_fields = {
        "_q_min",
        "_q_max",
        "_mass",
        "_center_of_mass",
        "_inertia_matrix",
    }
    direct_editor_fields = editor_fields - derived_editor_fields

    assert real_fields == direct_editor_fields, (
        f"Fields mismatch between SegmentReal and SegmentEditorData.\n"
        f"  In SegmentReal but not SegmentEditorData: {real_fields - direct_editor_fields}\n"
        f"  In SegmentEditorData but not SegmentReal: {direct_editor_fields - real_fields}"
    )
    assert derived_editor_fields <= editor_fields


def test_available_inertial_models_and_segments():
    """
    Expose only configurable inertial models and their source segments.
    """
    assert available_inertial_models() == ("de Leva", "Yeadon")
    assert "TRUNK" in inertial_model_segment_names("de Leva")
    assert "P" in inertial_model_segment_names("Yeadon")


def test_build_de_leva_inertial_parameters_from_model():
    """
    Build segment inertia from de Leva model-specific subject inputs.
    """
    inertia_parameters = build_inertial_parameters_from_model(
        "de Leva",
        "R_THIGH",
        {"total_mass": "70", "total_height": "1.75", "sex": "male"},
    )

    assert inertia_parameters.mass > 0
    assert inertia_parameters.center_of_mass.shape == (4, 1)
    assert inertia_parameters.inertia.shape == (4, 4)
    assert np.any(inertia_parameters.inertia[:3, :3])
