from copy import deepcopy
from ..components.real.biomechanical_model_real import BiomechanicalModelReal


def modify_muscle_parameters(original_model: BiomechanicalModelReal, new_model: BiomechanicalModelReal) -> None:
    """
    Modify the muscle parameters of the new_model based on the muscle length difference between the original and new models.
    """
    for muscle_group in original_model.muscle_groups:
        for muscle_name in muscle_group.muscle_names:
            if muscle_group.muscles[muscle_name].optimal_length is None:
                raise RuntimeError(
                    f"The muscle {muscle_name} does not have an optimal length. Please set the optimal length of the muscle in the original model."
                )
            elif muscle_group.muscles[muscle_name].tendon_slack_length is None:
                raise RuntimeError(
                    f"The muscle {muscle_name} does not have a tendon slack length. Please set the tendon slack length of the muscle in the original model."
                )

            original_muscle_tendon_length = original_model.muscle_tendon_length(muscle_name)
            new_muscle_tendon_length = new_model.muscle_tendon_length(muscle_name)
            length_ratio = new_muscle_tendon_length / original_muscle_tendon_length

            new_model.muscle_groups[muscle_group.name].muscles[muscle_name].optimal_length = (
                deepcopy(muscle_group.muscles[muscle_name].optimal_length) * length_ratio
            )
            new_model.muscle_groups[muscle_group.name].muscles[muscle_name].tendon_slack_length = (
                deepcopy(muscle_group.muscles[muscle_name].tendon_slack_length) * length_ratio
            )
