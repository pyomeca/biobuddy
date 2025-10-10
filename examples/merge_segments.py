"""
This example shows how to create a De Leva model, merge left and right segments, and replace the root segment at the foot.
"""

import os

import numpy as np
import itertools
from biobuddy import (
    Axis,
    BiomechanicalModel,
    C3dData,
    Marker,
    Mesh,
    Segment,
    SegmentCoordinateSystem,
    Translations,
    Rotations,
    DeLevaTable,
    Sex,
    SegmentName,
    ViewAs,
    SegmentCoordinateSystemUtils,
    RotoTransMatrix,
    MergeSegmentsTool,
    SegmentMerge,
    ModifyKinematicChainTool,
    ChangeFirstSegment,
)


def create_model():

    mass = 70  # Kg
    height = 1.75  # m
    # Create the inertial table for this model
    inertia_table = DeLevaTable(mass, sex=Sex.FEMALE)
    inertia_table.from_height(total_height=height)
    # Please note that as the weight are not provided in De Leva 1996, the hip and shoulder width are set to zero when
    # using `inertia_table.from_height`

    # Create the model
    real_model = inertia_table.to_simple_model()
    # real_model.animate()

    # Modify the model to merge both arms and both legs together
    merge_tool = MergeSegmentsTool(real_model)
    merge_tool.add(SegmentMerge(name="UPPER_ARMS", first_segment_name="L_UPPER_ARM", second_segment_name="R_UPPER_ARM"))
    merge_tool.add(SegmentMerge(name="LOWER_ARMS", first_segment_name="L_LOWER_ARM", second_segment_name="R_LOWER_ARM"))
    merge_tool.add(SegmentMerge(name="HANDS", first_segment_name="L_HAND", second_segment_name="R_HAND"))
    merge_tool.add(SegmentMerge(name="THIGHS", first_segment_name="L_THIGH", second_segment_name="R_THIGH"))
    merge_tool.add(SegmentMerge(name="SHANKS", first_segment_name="L_SHANK", second_segment_name="R_SHANK"))
    merge_tool.add(SegmentMerge(name="FEET", first_segment_name="L_FOOT", second_segment_name="R_FOOT"))
    merged_model = merge_tool.merge()
    # merged_model.animate()

    # Modify the model to place the root segment at the feet
    kinematic_chain_modifier = ModifyKinematicChainTool(merged_model)
    kinematic_chain_modifier.add(ChangeFirstSegment(first_segment_name="FEET", new_segment_name="PELVIS"))
    feet_root_model = kinematic_chain_modifier.modify()

    # Remove some unused degrees of freedom
    # Please note that the dof names were defined in the model before merge (the first segment dofs names were kept)
    feet_root_model.remove_dofs(
        [
            "FEET_transY",
            "FEET_rotX",
            "FEET_rotZ",
            "L_THIGH_rotX",
            "HEAD_rotX",
            "HEAD_rotY",
            "HEAD_rotZ",
            "L_UPPER_ARM_rotZ",
            "L_UPPER_ARM_rotX",
            "L_LOWER_ARM_rotY",
            "L_HAND_rotY",
        ]
    )
    # Add a shoulder dof in the plane
    feet_root_model.segments["UPPER_ARMS"].rotations = Rotations.Y
    feet_root_model.segments["UPPER_ARMS"].dof_names = ["UPPER_ARMS_rotY"]

    feet_root_model.animate()

    return feet_root_model


def main():

    feet_root_model = create_model()

    # Exporting the output model as a biomod file
    feet_root_model.to_biomod(f"feet_root_model.bioMod")


if __name__ == "__main__":
    main()
