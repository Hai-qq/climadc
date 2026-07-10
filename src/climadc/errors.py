class ClimaDCError(Exception):
    """Base exception for stable user-facing ClimaDC failures."""


class ContractError(ClimaDCError):
    pass


class LeakageError(ClimaDCError):
    pass


class ConfigurationError(ClimaDCError):
    pass
