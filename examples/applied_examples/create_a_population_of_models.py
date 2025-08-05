"""
This example shows how to create a batch of De Leva models within a range of anthropometry.
This could be useful if you want for example assess the impact of anthropometry on a movement.
Here, we want to create a population of female gymnasts performing acrobatics on the bars. Thus, the kinematic chain
will also be modified to place the root segment at the hands.
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


def main():

    # Set the range of anthropometry that you want to create
    # TODO: set these as -std, -1/2std, mean, +1/2std, +std
    total_mass = [50, 60, 70, 80]  # Kg
    total_height = [1.50, 1.70, 1.90]  # m
    ankle_height = [0.01]  # m
    knee_height_coeff = [0.23, 0.25, 0.27]
    pelvis_height_coeff = [0.48, 0.49, 0.51]
    shoulder_height_coeff = [0.78, 0.80, 0.82]
    shoulder_width_coeff = [0.30, 0.32, 0.34]
    elbow_span_coeff = [0.63, 0.65, 0.67]
    wrist_span_coeff = [0.80, 0.82, 0.84]
    finger_span_coeff = [1.0, 1.02, 1.04]
    foot_length_coeff = [0.3, 0.32, 0.34]
    hip_width_coeff = [0.30, 0.32, 0.34]

    # Create all combinations using itertools.product
    model_number = 0
    for combination in itertools.product(
        total_mass,
        total_height,
        ankle_height,
        knee_height_coeff,
        pelvis_height_coeff,
        shoulder_height_coeff,
        shoulder_width_coeff,
        elbow_span_coeff,
        wrist_span_coeff,
        finger_span_coeff,
        foot_length_coeff,
        hip_width_coeff,
    ):
        (
            this_mass,
            this_height,
            this_ankle_height_coeff,
            this_knee_height_coeff,
            this_pelvis_height_coeff,
            this_shoulder_height_coeff,
            this_shoulder_span_coeff,
            this_elbow_span_coeff,
            this_wrist_span_coeff,
            this_finger_span_coeff,
            this_foot_length_coeff,
            this_hip_width_coeff,
        ) = combination

        # Get the measurements for this model
        this_ankle_height = this_ankle_height_coeff * this_height
        this_knee_height = this_knee_height_coeff * this_height
        this_pelvis_height = this_pelvis_height_coeff * this_height
        this_shoulder_height = this_shoulder_height_coeff * this_height
        this_shoulder_span = this_shoulder_span_coeff * this_height
        this_elbow_span = this_elbow_span_coeff * this_height
        this_wrist_span = this_wrist_span_coeff * this_height
        this_finger_span = this_finger_span_coeff * this_height
        this_foot_length = this_foot_length_coeff * this_height
        this_hip_width = this_hip_width_coeff * this_height

        # Create the inertial table for this model
        inertia_table = DeLevaTable(this_mass, sex=Sex.FEMALE)
        inertia_table.from_measurements(
            total_height=this_height,
            ankle_height=this_ankle_height,
            knee_height=this_knee_height,
            pelvis_height=this_pelvis_height,
            shoulder_height=this_shoulder_height,
            finger_span=this_finger_span,
            wrist_span=this_wrist_span,
            elbow_span=this_elbow_span,
            shoulder_span=this_shoulder_span,
            hip_width=this_hip_width,
            foot_length=this_foot_length,
        )

        # Create the model
        real_model = inertia_table.to_simple_model()
        # real_model.animate()

        # Modify the model to merge both arms together
        merge_tool = MergeSegmentsTool(real_model)
        merge_tool.add(SegmentMerge(name="UPPER_ARMS", first_segment_name="L_UPPER_ARM", second_segment_name="R_UPPER_ARM"))
        merge_tool.add(SegmentMerge(name="LOWER_ARMS", first_segment_name="L_LOWER_ARM", second_segment_name="R_LOWER_ARM"))
        merge_tool.add(SegmentMerge(name="HANDS", first_segment_name="L_HAND", second_segment_name="R_HAND"))
        merged_model = merge_tool.merge()
        # merged_model.animate()

        # Modify the model to place the root segment at the hands
        kinematic_chain_modifier = ModifyKinematicChainTool(merged_model)
        kinematic_chain_modifier.add(ChangeFirstSegment(first_segment_name="HANDS", new_segment_name="PELVIS"))
        hand_root_model = kinematic_chain_modifier.modify()
        hand_root_model.animate()

        # Exporting the output model as a biomod file
        hand_root_model.to_biomod(f"population_model_{model_number}.bioMod")
        model_number += 1


if __name__ == "__main__":
    main()
