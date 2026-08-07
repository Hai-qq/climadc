from __future__ import annotations

import pandas as pd
import pytest

from climadc.adapters.neso import NESOCarbonIntensityAdapter
from climadc.adapters.openmeteo_history import OpenMeteoHistoryAdapter


@pytest.mark.network
def test_reference_provider_endpoints_still_satisfy_contracts() -> None:
    decision = pd.Timestamp("2026-08-01T00:00:00Z")
    horizon = pd.Timedelta(hours=24)

    weather = OpenMeteoHistoryAdapter().fetch(
        latitude=51.5074,
        longitude=-0.1278,
        site_id="gb-london-reference",
        decision_time=decision,
        horizon=horizon,
    )
    carbon = NESOCarbonIntensityAdapter().fetch(
        site_id="gb-london-reference", decision_time=decision, horizon=horizon
    )

    assert len(weather.forecast.to_pandas()) == 24
    assert len(weather.actual.to_pandas()) == 24
    assert len(carbon.to_pandas()) == 48
