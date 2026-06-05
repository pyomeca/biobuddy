import numpy as np
import pytest

from biobuddy.gui.full_body_bela_template import (
    bela_inertia_by_segment,
    bela_marker_attachments,
    bela_marker_names,
    bela_segment_specs,
    bela_unresolved_marker_references,
    guse_inertia_by_segment,
    signed_marker_groups,
    subject_inertia_by_segment,
)


def test_bela_template_contains_expected_full_body_chain():
    segments = bela_segment_specs()

    assert len(segments) == 17
    assert segments[0].name == "Pelvis"
    assert segments[0].parent_name == "base"
    assert segments[1].name == "Thorax"
    assert segments[1].parent_name == "Pelvis"
    assert segments[-1].name == "PiedG"
    assert segments[-1].parent_name == "JambeG"


def test_bela_markers_follow_matlab_order_and_segment_ownership():
    marker_names = bela_marker_names()
    attachments = bela_marker_attachments()

    assert len(marker_names) == 94
    assert marker_names[:6] == ("EIASD", "CID", "EIPSD", "EIPSG", "CIG", "EIASG")
    assert marker_names[-6:] == ("CALCG", "MIDMETA4G", "MIDMETA1G", "SCAPHOIDEG", "METAT5G", "METAT1G")
    assert "CLAV3D" not in marker_names
    assert len(attachments) == 94
    assert attachments[0].name == "EIASD"
    assert attachments[0].segment_names == ("Pelvis",)


def test_bela_inertial_parameters_are_available_by_segment_name():
    inertia = bela_inertia_by_segment()

    assert inertia["Pelvis"]["mass"] == pytest.approx(11.5688)
    np.testing.assert_allclose(inertia["Pelvis"]["center_of_mass"], np.array([0.0, 0.0, 0.1147]))
    np.testing.assert_allclose(np.diag(inertia["PiedG"]["inertia"]), np.array([0.0068, 0.0066, 0.0012]))


def test_guse_inertial_parameters_use_the_same_segments_with_subject_values():
    bela_inertia = subject_inertia_by_segment("BeLa")
    guse_inertia = guse_inertia_by_segment()

    assert set(guse_inertia) == set(bela_inertia)
    assert guse_inertia["Pelvis"]["mass"] == pytest.approx(9.5842)
    np.testing.assert_allclose(guse_inertia["Pelvis"]["center_of_mass"], np.array([0.0, 0.0, 0.0918]))
    np.testing.assert_allclose(guse_inertia["EpauleD"]["center_of_mass"], np.array([0.0858, 0.0, 0.0]))
    np.testing.assert_allclose(guse_inertia["EpauleG"]["center_of_mass"], np.array([-0.0858, 0.0, 0.0]))
    assert bela_inertia["Pelvis"]["mass"] != guse_inertia["Pelvis"]["mass"]


def test_signed_marker_groups_convert_matlab_axis_indices_when_all_markers_are_raw():
    pelvis = bela_segment_specs()[0]

    start_markers, end_markers = signed_marker_groups(pelvis, pelvis.u_indices)

    assert start_markers == ("EIASG",)
    assert end_markers == ("EIASD",)


def test_unresolved_marker_references_report_virtual_or_functional_points():
    unresolved = bela_unresolved_marker_references()

    assert "Pelvis" not in unresolved
    assert unresolved["Thorax"] == (7,)
    assert unresolved["ABrasD"] == (9, 10)
    assert unresolved["JambeD"] == (7, 8, 9)
    with pytest.raises(ValueError, match="references marker index 7"):
        signed_marker_groups(bela_segment_specs()[1], bela_segment_specs()[1].origin_indices)
