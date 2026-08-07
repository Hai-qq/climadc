from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd
from pint import Unit, UnitRegistry
from pint.errors import PintError

from climadc.errors import ContractError

UNIT_REGISTRY: UnitRegistry[Any] = UnitRegistry()

# Pint intentionally does not ship domain-specific emissions labels or currency
# units. These definitions make contract units explicit while keeping different
# currencies incompatible unless a caller supplies an exchange rate.
for _definition in (
    "gCO2e = gram",
    "kgCO2e = kilogram",
    "tCO2e = metric_ton",
    "GBP = [currency_gbp]",
    "USD = [currency_usd]",
    "EUR = [currency_eur]",
    "CNY = [currency_cny]",
    "JPY = [currency_jpy]",
):
    UNIT_REGISTRY.define(_definition)


def _row_word(count: int) -> str:
    return "row" if count == 1 else "rows"


def validate_unit_consistency(
    frame: pd.DataFrame,
    name_column: str,
    unit_column: str,
) -> None:
    """Require parseable, dimensionally compatible units within each named group."""
    missing_columns = {name_column, unit_column}.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ContractError(f"Unit validation requires columns: {missing}")

    parsed_units: list[Unit | None] = []
    invalid_positions: list[int] = []
    for position, label in enumerate(frame[unit_column].tolist()):
        if not isinstance(label, str) or not label.strip():
            parsed_units.append(None)
            invalid_positions.append(position)
            continue
        try:
            parsed_units.append(UNIT_REGISTRY.parse_units(label))
        except (PintError, TypeError, ValueError):
            parsed_units.append(None)
            invalid_positions.append(position)

    if invalid_positions:
        count = len(invalid_positions)
        raise ContractError(f"invalid unit: {count} offending {_row_word(count)}")

    incompatible_positions: set[int] = set()
    groups = frame.groupby(name_column, dropna=False, sort=False, observed=True).indices
    for positions in groups.values():
        position_list = [int(position) for position in positions]
        reference = cast(Unit, parsed_units[position_list[0]])
        if any(
            not cast(Unit, parsed_units[position]).is_compatible_with(reference)
            for position in position_list[1:]
        ):
            incompatible_positions.update(position_list)

    if incompatible_positions:
        count = len(incompatible_positions)
        raise ContractError(f"incompatible units: {count} offending {_row_word(count)}")


def validate_expected_unit_dimension(
    frame: pd.DataFrame,
    unit_column: str,
    expected_units: Sequence[str],
    context: str,
) -> None:
    """Require every unit to match at least one declared physical dimension."""

    if unit_column not in frame.columns:
        raise ContractError(f"{context}: missing unit column {unit_column!r}")
    if not expected_units:
        raise ContractError(f"{context}: expected_units must not be empty")

    try:
        references = [UNIT_REGISTRY.parse_units(label) for label in expected_units]
    except (PintError, TypeError, ValueError) as exc:
        raise ContractError(f"{context}: invalid expected unit definition") from exc

    invalid = 0
    for label in frame[unit_column].tolist():
        try:
            parsed = UNIT_REGISTRY.parse_units(label)
        except (PintError, TypeError, ValueError):
            invalid += 1
            continue
        if not any(parsed.is_compatible_with(reference) for reference in references):
            invalid += 1

    if invalid:
        expected = ", ".join(expected_units)
        raise ContractError(
            f"{context}: {unit_column} unit must be compatible with one of [{expected}]: "
            f"{invalid} offending {_row_word(invalid)}"
        )
