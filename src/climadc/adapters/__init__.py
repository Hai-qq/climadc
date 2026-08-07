"""Converters from supported climate and telemetry sources into canonical contracts."""

from climadc.adapters.carbon_aware import CarbonAwareResult, CarbonAwareSDKAdapter
from climadc.adapters.local import (
    read_climate,
    read_flexible_workload,
    read_grid_signals,
    read_telemetry,
    read_workload,
)
from climadc.adapters.neso import NESOCarbonIntensityAdapter
from climadc.adapters.openmeteo import OpenMeteoAdapter
from climadc.adapters.openmeteo_history import OpenMeteoHistoryAdapter, OpenMeteoHistoryResult
from climadc.adapters.prometheus import (
    KeplerPrometheusAdapter,
    PrometheusRangeAdapter,
    PrometheusRangeResult,
)
from climadc.adapters.sustaindc import SustainDCAdapter, SustainDCResult
from climadc.adapters.weatherdc import WeatherDCAdapter
from climadc.adapters.xarray import climate_from_xarray

__all__ = [
    "CarbonAwareResult",
    "CarbonAwareSDKAdapter",
    "NESOCarbonIntensityAdapter",
    "OpenMeteoAdapter",
    "OpenMeteoHistoryAdapter",
    "OpenMeteoHistoryResult",
    "KeplerPrometheusAdapter",
    "PrometheusRangeAdapter",
    "PrometheusRangeResult",
    "SustainDCAdapter",
    "SustainDCResult",
    "WeatherDCAdapter",
    "climate_from_xarray",
    "read_climate",
    "read_flexible_workload",
    "read_grid_signals",
    "read_telemetry",
    "read_workload",
]
