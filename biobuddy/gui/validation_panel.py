from dataclasses import dataclass

from ..components.real.biomechanical_model_real import BiomechanicalModelReal


@dataclass
class ValidationReport:
    """
    Result of validating one biomechanical model.
    """

    is_valid: bool
    messages: list[str]
    category: str


def validate_model_for_editor(model: BiomechanicalModelReal) -> ValidationReport:
    """
    Validate a model and return editor-friendly messages instead of raising.
    """
    try:
        model.validate_model()
    except Exception as error:
        return ValidationReport(
            is_valid=False,
            messages=[str(error)],
            category=_categorize_validation_error(str(error)),
        )
    return ValidationReport(is_valid=True, messages=["Model validation passed."], category="ok")


def _categorize_validation_error(message: str) -> str:
    """
    Categorize the most common model-validation failures for the UI.
    """
    lower_message = message.lower()
    if "dof" in lower_message or "q_ranges" in lower_message:
        return "degrees_of_freedom"
    if "origin" in lower_message or "insertion" in lower_message or "via point" in lower_message:
        return "muscles"
    if "kinematic chain" in lower_message or "segment" in lower_message:
        return "kinematic_chain"
    return "general"
