import pytest
import numpy as np
import numpy.testing as npt
import biorbd
from deepdiff import DeepDiff

from biobuddy import (
    MergeSegmentsTool,
    SegmentMerge,
    BiomechanicalModelReal,
    Translations,
    Rotations,
    RotoTransMatrix,
)
from test_utils import create_simple_model


def test_segment_merge_init():
    """Test the initialization of SegmentMerge"""

    # Without merged_origin_name
    segment_merge = SegmentMerge(name="UPPER_ARMS", first_segment_name="L_UPPER_ARM", second_segment_name="R_UPPER_ARM")
    assert segment_merge.name == "UPPER_ARMS"
    assert segment_merge.first_segment_name == "L_UPPER_ARM"
    assert segment_merge.second_segment_name == "R_UPPER_ARM"
    assert segment_merge.merged_origin_name is None

    # With merged_origin_name
    segment_merge = SegmentMerge(
        name="UPPER_ARMS",
        first_segment_name="L_UPPER_ARM",
        second_segment_name="R_UPPER_ARM",
        merged_origin_name="L_UPPER_ARM",
    )
    assert segment_merge.name == "UPPER_ARMS"
    assert segment_merge.first_segment_name == "L_UPPER_ARM"
    assert segment_merge.second_segment_name == "R_UPPER_ARM"
    assert segment_merge.merged_origin_name == "L_UPPER_ARM"

    # Bad merged_origin_name
    with pytest.raises(
        RuntimeError,
        match="The merged origin name must be one of the two segments being merged or None if you want it to be the mean of both origins.",
    ):
        segment_merge = SegmentMerge(
            name="UPPER_ARMS",
            first_segment_name="L_UPPER_ARM",
            second_segment_name="R_UPPER_ARM",
            merged_origin_name="bad_name",
        )


def test_merge_segment_tool_init():
    """
    Test the initialization of the MergeSegmentsTool
    """
    original_model = create_simple_model()

    merge_tool = MergeSegmentsTool(original_model)
    assert DeepDiff(merge_tool.original_model, original_model, ignore_order=True) == {}
    assert isinstance(merge_tool.merged_model, BiomechanicalModelReal)
    assert merge_tool.segments_to_merge == []

    # Test the addition of a segment merge
    assert len(merge_tool.segments_to_merge) == 0
    merge_tool.add(SegmentMerge(name="UPPER_ARMS", first_segment_name="L_UPPER_ARM", second_segment_name="R_UPPER_ARM"))
    assert len(merge_tool.segments_to_merge) == 1
    assert merge_tool.segments_to_merge[0].name == "UPPER_ARMS"
    assert merge_tool.segments_to_merge["UPPER_ARMS"].first_segment_name == "L_UPPER_ARM"
    assert merge_tool.segments_to_merge["UPPER_ARMS"].second_segment_name == "R_UPPER_ARM"


def test_merge_segments_tool_merge():
    """Test the merge functionality of MergeSegmentsTool"""

    original_model = create_simple_model()
    new_rt = RotoTransMatrix()
    new_rt.translation = np.array([0.1, 0.2, 0.3])
    original_model.segments["root"].segment_coordinate_system.scs = new_rt

    # Create an equivalent biorbd model for comparison
    original_model.segments["child"].translations = Translations.NONE
    original_model.segments["child"].rotations = Rotations.NONE
    original_model.segments["child"].dof_names = []
    original_model.to_biomod("merged_model.bioMod")
    biorbd_merged = biorbd.Model("merged_model.bioMod")
    q_zeros = np.zeros((biorbd_merged.nbQ(),))

    # Merge in BioBuddy
    merge_tool = MergeSegmentsTool(original_model)
    merge_tool.add(
        SegmentMerge(name="both", first_segment_name="root", second_segment_name="child", merged_origin_name="root")
    )
    merged_model = merge_tool.merge()

    # Check the segment's name
    assert merged_model.segments[0].name == "both"
    assert merged_model.segments["both"].name == "both"

    # Check the segment's parent name
    assert merged_model.segments["both"].parent.name == "base"

    # Check the segment's dofs
    assert merged_model.segments["both"].nb_dof == 6
    assert merged_model.segments["both"].translations == Translations.XYZ
    assert merged_model.segments["both"].rotations == Rotations.XYZ
    assert merged_model.segments["both"].dof_names == []
    assert merged_model.segments["both"].q_ranges.min_bound == [-np.pi] * 6
    assert merged_model.segments["both"].qdot_ranges is None

    # Check the segment's scs
    npt.assert_almost_equal(merged_model.segments["both"].scs.rt_matrix, new_rt.rt_matrix)

    # Check the merged segment's mass
    npt.assert_almost_equal(merged_model.segments["both"].mass, biorbd_merged.mass(), decimal=5)

    # Check the merged segment's com
    biorbd_com = biorbd_merged.CoM(q_zeros).to_array()
    biobuddy_com = merged_model.segments["both"].inertia_parameters.center_of_mass.reshape(
        4,
    )[:3]
    npt.assert_almost_equal(biorbd_com, biobuddy_com, decimal=5)

    # Check the merged segment's inertia
    biorbd_inertia = biorbd_merged.bodyInertia(q_zeros).to_array()
    biobuddy_inertia = merged_model.segments["both"].inertia_parameters.inertia
    npt.assert_almost_equal(biorbd_inertia, biobuddy_inertia, decimal=5)

    # Check the merged segment's mesh
    npt.assert_almost_equal(merged_model.segments["both"].mesh.positions, np.array([]), decimal=5)

    # Check the merged segment's markers
    biorbd_marker = biorbd_merged.marker(q_zeros, 0).to_array()
    biobuddy_marker = merged_model.segments["both"].markers[0].position.reshape(
        3,
    ) + np.array([0.1, 0.2, 0.3])
    npt.assert_almost_equal()
