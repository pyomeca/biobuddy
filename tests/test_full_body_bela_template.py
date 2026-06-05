from pathlib import Path

import numpy as np
import pytest

from biobuddy.gui.full_body_bela_template import (
    bela_inertia_by_segment,
    bela_marker_attachments,
    bela_marker_names,
    bela_segment_specs,
    bela_unresolved_marker_references,
    guse_inertia_by_segment,
    parse_s2m_model,
    rotations_from_matlab_dof,
    signed_marker_groups,
    subject_inertia_by_segment,
    translations_from_matlab_dof,
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
    np.testing.assert_allclose(inertia["Thorax"]["center_of_mass"], np.array([0.0, 0.0, 0.1130523729]))
    np.testing.assert_allclose(inertia["EpauleD"]["center_of_mass"], np.array([0.1123, 0.0, 0.0]))
    np.testing.assert_allclose(inertia["EpauleG"]["center_of_mass"], np.array([-0.1123, 0.0, 0.0]))
    np.testing.assert_allclose(
        inertia["MainD"]["center_of_mass"], np.array([0.0201989512, -0.0490185172, -0.027307392])
    )
    np.testing.assert_allclose(
        inertia["MainG"]["center_of_mass"], np.array([-0.0264342737, -0.0469823183, -0.0252076569])
    )
    np.testing.assert_allclose(np.diag(inertia["PiedG"]["inertia"]), np.array([0.0068, 0.0066, 0.0012]))


def test_matlab_dof_signs_do_not_change_model_axes():
    segments = {segment.name: segment for segment in bela_segment_specs()}

    assert translations_from_matlab_dof(segments["Pelvis"]) == "xyz"
    assert rotations_from_matlab_dof(segments["Pelvis"]) == "xyz"
    assert rotations_from_matlab_dof(segments["EpauleD"]) == "yz"
    assert rotations_from_matlab_dof(segments["ABrasD"]) == "xz"
    assert rotations_from_matlab_dof(segments["MainD"]) == "xy"
    assert rotations_from_matlab_dof(segments["JambeD"]) == "x"
    assert rotations_from_matlab_dof(segments["PiedD"]) == "xz"


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


def test_parse_s2m_model_reads_segments_markers_and_local_properties(tmp_path):
    filepath = tmp_path / "minimal.s2mMod"
    filepath.write_text(
        """
version\t1

segment\tPelvis
\tRT
\t\t1\t0\t0\t0
\t\t0\t1\t0\t0
\t\t0\t0\t1\t0
\t\t0\t0\t0\t1
\ttranslations\txyz
\trotations\txyz
\tmass\t11.57
\tinertia
\t\t0.0801\t0\t0
\t\t0\t0.1117\t0
\t\t0\t0\t0.0975
\tcom\t0\t0\t0.1147
endsegment

marker\tEIASD
\tparent\tPelvis
\tposition\t0.1\t0.2\t0.3
\ttechnical 1
endmarker
""",
        encoding="utf-8",
    )

    segments = parse_s2m_model(filepath)

    assert len(segments) == 1
    assert segments[0].name == "Pelvis"
    assert segments[0].translations == "xyz"
    assert segments[0].rotations == "xyz"
    assert segments[0].mass == pytest.approx(11.57)
    np.testing.assert_allclose(segments[0].inertia, np.diag([0.0801, 0.1117, 0.0975]))
    assert segments[0].markers[0].name == "EIASD"
    assert segments[0].markers[0].position == (0.1, 0.2, 0.3)


def test_bela_s2m_model_is_consistent_with_matlab_chain_when_available():
    filepath = Path("/Users/mickaelbegon/Downloads/BeLa_2.s2mMod")
    if not filepath.exists():
        pytest.skip("Historical BeLa .s2mMod file is not available on this machine.")

    s2m_segments = parse_s2m_model(filepath)
    matlab_segments = {segment.name: segment for segment in bela_segment_specs()}

    assert len(s2m_segments) == 17
    assert sum(len(segment.markers) for segment in s2m_segments) == 94
    for s2m_segment in s2m_segments:
        matlab_segment = matlab_segments[s2m_segment.name]
        assert s2m_segment.parent_name == matlab_segment.parent_name
        assert tuple(marker.name for marker in s2m_segment.markers) == matlab_segment.marker_names
        assert s2m_segment.translations == translations_from_matlab_dof(matlab_segment)
        assert s2m_segment.rotations == rotations_from_matlab_dof(matlab_segment)


def test_bela_biomod_reference_is_consistent_with_template_when_available():
    filepath = Path("/Users/mickaelbegon/Downloads/BeLa.bioMod")
    if not filepath.exists():
        pytest.skip("Reference BeLa .bioMod file is not available on this machine.")

    reference_segments = parse_s2m_model(filepath)
    matlab_segments = {segment.name: segment for segment in bela_segment_specs()}
    inertia = bela_inertia_by_segment()

    assert len(reference_segments) == 17
    assert sum(len(segment.markers) for segment in reference_segments) == 94
    for reference_segment in reference_segments:
        matlab_segment = matlab_segments[reference_segment.name]
        expected_inertia = inertia[reference_segment.name]
        assert reference_segment.parent_name == matlab_segment.parent_name
        assert tuple(marker.name for marker in reference_segment.markers) == matlab_segment.marker_names
        assert reference_segment.translations == translations_from_matlab_dof(matlab_segment)
        assert reference_segment.rotations == rotations_from_matlab_dof(matlab_segment)
        assert reference_segment.mass == pytest.approx(expected_inertia["mass"], abs=0.005)
        np.testing.assert_allclose(
            reference_segment.center_of_mass,
            expected_inertia["center_of_mass"],
            atol=1e-10,
        )
        np.testing.assert_allclose(reference_segment.inertia, expected_inertia["inertia"], atol=1e-10)
