from __future__ import annotations

from typing import Any, cast

import pandas as pd
from pint import Unit, UnitRegistry
from pint.errors import PintError

from climadc.errors import ContractError

UNIT_REGISTRY: UnitRegistry[Any] = UnitRegistry()


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
