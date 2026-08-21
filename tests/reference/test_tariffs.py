from __future__ import annotations

import pandas as pd
import pytest

from climadc.errors import ConfigurationError
from climadc.reference.tariffs import declared_tariff_value


def test_london_spring_forward_maps_from_real_utc_instants_without_missing_hour() -> None:
    instants = pd.DatetimeIndex(["2026-03-29T00:00:00Z", "2026-03-29T01:00:00Z"])
    local = instants.tz_convert("Europe/London")

    assert list(local.hour) == [0, 2]
    assert [declared_tariff_value(slot, time_basis="Europe/London") for slot in instants] == [
        0.12,
        0.12,
    ]


def test_london_fall_back_preserves_both_repeated_hour_instants() -> None:
    instants = pd.DatetimeIndex(["2026-10-25T00:00:00Z", "2026-10-25T01:00:00Z"])
    local = instants.tz_convert("Europe/London")

    assert list(local.hour) == [1, 1]
    assert local[0].utcoffset() != local[1].utcoffset()
    assert [declared_tariff_value(slot, time_basis="Europe/London") for slot in instants] == [
        0.12,
        0.12,
    ]


def test_explicit_utc_and_london_bases_can_differ_in_summer() -> None:
    slot = pd.Timestamp("2026-08-01T05:00:00Z")

    assert declared_tariff_value(slot, time_basis="UTC") == 0.12
    assert declared_tariff_value(slot, time_basis="Europe/London") == 0.25


def test_tariff_rejects_naive_time_and_unknown_basis() -> None:
    with pytest.raises(ConfigurationError, match="timezone-aware"):
        declared_tariff_value(pd.Timestamp("2026-01-01T00:00:00"))
    with pytest.raises(ConfigurationError, match="unsupported"):
        declared_tariff_value(  # type: ignore[arg-type]
            pd.Timestamp("2026-01-01T00:00:00Z"), time_basis="local"
        )
