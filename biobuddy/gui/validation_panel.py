from dataclasses import dataclass

from ..components.real.biomechanical_model_real import BiomechanicalModelReal


@dataclass
class ValidationReport:
    """
    Result of validating one biomechanical model.
    """

    is_valid: bool
    messages: list[str]


def validate_model_for_editor(model: BiomechanicalModelReal) -> ValidationReport:
    """
    Validate a model and return editor-friendly messages instead of raising.
    """
    try:
        model.validate_model()
    except Exception as error:
        return ValidationReport(is_valid=False, messages=[str(error)])
    return ValidationReport(is_valid=True, messages=["Model validation passed."])
