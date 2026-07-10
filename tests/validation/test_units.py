import pandas as pd
import pytest

from climadc.errors import ContractError
from climadc.validation.units import validate_unit_consistency


def test_unit_validation_parses_every_unit() -> None:
    frame = pd.DataFrame(
        {
            "metric": ["power", "power"],
            "unit": ["kW", "definitely_not_a_unit"],
        }
    )

    with pytest.raises(ContractError, match=r"invalid unit.*1 offending row"):
        validate_unit_consistency(frame, "metric", "unit")


def test_unit_validation_rejects_incompatible_dimensions_within_group() -> None:
    frame = pd.DataFrame(
        {
            "metric": ["power", "power"],
            "unit": ["kW", "meter"],
        }
    )

    with pytest.raises(ContractError, match=r"incompatible units.*2 offending row"):
        validate_unit_consistency(frame, "metric", "unit")


def test_unit_validation_accepts_compatible_dimensions_without_rewriting_labels() -> None:
    frame = pd.DataFrame(
        {
            "metric": ["power", "power", "temperature", "temperature"],
            "unit": ["W", "kW", "degC", "kelvin"],
        }
    )
    original = frame.copy(deep=True)

    validate_unit_consistency(frame, "metric", "unit")

    pd.testing.assert_frame_equal(frame, original)
