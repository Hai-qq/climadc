"""Converters from supported climate and telemetry sources into canonical contracts."""

from climadc.adapters.local import read_climate, read_telemetry, read_workload
from climadc.adapters.openmeteo import OpenMeteoAdapter
from climadc.adapters.weatherdc import WeatherDCAdapter
from climadc.adapters.xarray import climate_from_xarray

__all__ = [
    "OpenMeteoAdapter",
    "WeatherDCAdapter",
    "climate_from_xarray",
    "read_climate",
    "read_telemetry",
    "read_workload",
]
