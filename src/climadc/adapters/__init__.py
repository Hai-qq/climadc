"""Converters from supported climate and telemetry sources into canonical contracts."""

from climadc.adapters.local import read_climate, read_telemetry
from climadc.adapters.openmeteo import OpenMeteoAdapter
from climadc.adapters.xarray import climate_from_xarray

__all__ = ["OpenMeteoAdapter", "climate_from_xarray", "read_climate", "read_telemetry"]
