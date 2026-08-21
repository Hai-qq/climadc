from __future__ import annotations

from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

from climadc.errors import ConfigurationError

TariffTimeBasis = Literal["UTC", "Europe/London"]


def declared_tariff_value(
    slot: pd.Timestamp,
    *,
    time_basis: TariffTimeBasis = "UTC",
) -> float:
    """Evaluate the illustrative tariff on an explicit civil-time basis.

    The packaged v1 fixture declares ``UTC``. ``Europe/London`` exists for
    versioned future fixtures and maps from an unambiguous UTC instant, so a
    fall-back repeated hour remains two distinct instants and a spring-forward
    missing hour is never invented.
    """
    if (
        not isinstance(slot, pd.Timestamp)
        or pd.isna(slot)
        or slot.tzinfo is None
        or slot.utcoffset() is None
    ):
        raise ConfigurationError("tariff slot must be a timezone-aware pandas Timestamp")
    if time_basis not in {"UTC", "Europe/London"}:
        raise ConfigurationError(f"unsupported tariff time basis: {time_basis}")
    instant = slot.tz_convert(ZoneInfo(time_basis))
    hour = instant.hour
    if hour < 6:
        return 0.12
    if hour < 16:
        return 0.25
    if hour < 20:
        return 0.45
    return 0.18
