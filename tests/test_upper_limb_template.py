from pathlib import Path

import numpy as np
import pytest

from biobuddy.gui.full_body_bela_template import parse_s2m_model
from biobuddy.gui.model_builder import required_static_markers
from biobuddy.gui.upper_limb_template import (
    upper_limb_inertia_by_segment,
    upper_limb_marker_attachments,
    upper_limb_marker_names,
    upper_limb_segment_specs,
    upper_limb_template,
    upper_limb_unresolved_marker_references,
    upper_limb_virtual_axis_endpoint_names,
    upper_limb_virtual_feature_requirements,
    upper_limb_virtual_point_name,
)


def test_upper_limb_template_contains_expected_chain_and_markers():
    segments = upper_limb_segment_specs()
    marker_names = upper_limb_marker_names()
    attachments = upper_limb_marker_attachments()

    assert len(segments) == 8
    assert segments[0].name == "Pelvis"
    assert segments[-1].name == "Hand"
    assert segments[-1].parent_name == "LowerArm2"
    assert len(marker_names) == 43
    assert marker_names[:4] == ("ASISl", "ASISr", "PSISl", "PSISr")
    assert marker_names[-4:] == ("INDEX", "LASTC", "MEDH", "LATH")
    assert len(attachments) == 43


def test_upper_limb_template_reports_virtual_marker_references():
    unresolved = upper_limb_unresolved_marker_references()

    assert unresolved["Thorax"] == (7,)
    assert unresolved["Clavicule"] == (6, 7)
    assert unresolved["Scapula"] == (10,)
    assert unresolved["Arm"] == (8, 9, 10)
    assert unresolved["LowerArm1"] == (5, 6)
    assert unresolved["LowerArm2"] == (5, 6)
    assert unresolved["Hand"] == (5,)


def test_upper_limb_model_template_exposes_virtual_feature_placeholders():
    template = upper_limb_template()
    requirement_names = {requirement.name for requirement in upper_limb_virtual_feature_requirements()}

    assert template.root_segment_name == "Pelvis"
    assert [segment.name for segment in template.segments] == [
        "Pelvis",
        "Thorax",
        "Clavicule",
        "Scapula",
        "Arm",
        "LowerArm1",
        "LowerArm2",
        "Hand",
    ]
    assert upper_limb_virtual_point_name("Thorax", 7) in requirement_names
    assert upper_limb_virtual_point_name("Arm", 8) in requirement_names
    assert "Clavicule_u_axis" in requirement_names
    assert upper_limb_virtual_point_name("Hand", 5) in requirement_names

    required_markers = set(required_static_markers(template))
    clavicle_axis_start, clavicle_axis_end = upper_limb_virtual_axis_endpoint_names("Clavicule", "u_axis")
    assert upper_limb_virtual_point_name("Thorax", 7) in required_markers
    assert clavicle_axis_start in required_markers
    assert clavicle_axis_end in required_markers


def test_upper_limb_inertia_parameters_match_reference_values():
    inertia = upper_limb_inertia_by_segment()

    assert inertia["Pelvis"]["mass"] is None
    assert inertia["Thorax"]["mass"] == 48.71
    np.testing.assert_allclose(inertia["Thorax"]["center_of_mass"], np.array([-0.000676, 0.016992, 0.123972]))
    np.testing.assert_allclose(np.diag(inertia["Arm"]["inertia"]), np.array([0.001835, 0.001605, 0.000520]))
    np.testing.assert_allclose(inertia["LowerArm2"]["center_of_mass"], np.array([-0.018328, 0.010440, -0.002528]))
    np.testing.assert_allclose(inertia["Hand"]["center_of_mass"], np.array([-0.001984, -0.016656, -0.047286]))


def test_upper_limb_reference_model_matches_template_when_available():
    filepath = Path("/Volumes/10.89.24.15/BackUp/F/Data/Shoulder/Lib/IRSST_EmiDd/Model_2/Model.s2mMod")
    if not filepath.exists():
        pytest.skip("Upper-limb reference Model.s2mMod file is not available on this machine.")

    reference_segments = parse_s2m_model(filepath)
    template_segments = {segment.name: segment for segment in upper_limb_segment_specs()}

    assert len(reference_segments) == 8
    assert sum(len(segment.markers) for segment in reference_segments) == 43
    for reference_segment in reference_segments:
        template_segment = template_segments[reference_segment.name]
        assert reference_segment.parent_name == template_segment.parent_name
        assert tuple(marker.name for marker in reference_segment.markers) == template_segment.marker_names
        assert reference_segment.translations == template_segment.translations
        assert reference_segment.rotations == template_segment.rotations
