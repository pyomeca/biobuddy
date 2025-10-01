"""
This example shows how to use biobuddy and biorbd to check the muscle lever arms and their capabilities. This is useful
to validate the muscle parameters of a model to make sure they are adequate for the movement at hand.

It plots 3 graphs to check the capability of muscles within a musculoskeletal model (.bioMod model), plots the max
strength of each muscle, their moment arm over each joint, and the max torque that each muscle can apply on each joint.

Only the model path is required to perform the check. The muscles parameters will by default automatically be computed
over the entire range of motion of each joint. Optionally, a custom range for the joints can be provided in the
states_from_model_ranges function.

If you want to add additional passive joint forces in the joint torque computation that aren't directly defined in
the .bioMod file but are computed later on in your simulations, it is possible to do so in the indicated line in
the compute_torques function.
"""

from pathlib import Path

from biobuddy import BiomechanicalModelReal, MuscleValidator


def plot_muscle_validation(model_path: str):

    # Get the model
    model = BiomechanicalModelReal().from_biomod(model_path)

    # Create the MuscleValidator object
    muscle_validator = MuscleValidator(model)
    muscle_validator.plot_force_length()
    muscle_validator.plot_moment_arm()
    muscle_validator.plot_torques()


def main():

    # Path to the model to check
    current_path_file = Path(__file__).parent
    model_path = f"{current_path_file}/models/wholebody_reference.bioMod"

    # Plot the muscle validation graphs
    plot_muscle_validation(model_path)


if __name__ == "__main__":
    main()
