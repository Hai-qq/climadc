from typer.testing import CliRunner

import climadc
from climadc.adapters import (
    CarbonAwareSDKAdapter,
    KeplerPrometheusAdapter,
    PrometheusRangeAdapter,
    SustainDCAdapter,
    read_flexible_workload,
    read_grid_signals,
)
from climadc.cli.app import app
from climadc.contracts import FlexibleWorkloadFrame, GridSignalFrame
from climadc.reference import packaged_study_path, packaged_suite_path
from climadc.replay import (
    ALL_POLICY_NAMES,
    RISK_AWARE_POLICY,
    ReplayArtifactWriter,
    ReplayConfig,
    ReplayEngine,
    ReplayStudyConfig,
    ReplayStudyRunner,
    ReplaySuiteArtifactWriter,
    ReplaySuiteConfig,
    ReplaySuiteRunner,
    RollingReplayEngine,
    RollingReplayResult,
    RollingReplaySettings,
    SourceManifest,
    TemperatureSensitivePUEModel,
)


def test_package_version_and_cli() -> None:
    assert climadc.__version__ == "0.3.0a1"
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "climadc 0.3.0a1"


def test_v02_engineering_apis_are_exported_from_public_packages() -> None:
    assert GridSignalFrame.__name__ == "GridSignalFrame"
    assert FlexibleWorkloadFrame.__name__ == "FlexibleWorkloadFrame"
    assert callable(read_grid_signals)
    assert callable(read_flexible_workload)
    assert ReplayConfig.__name__ == "ReplayConfig"
    assert ReplayEngine.__name__ == "ReplayEngine"
    assert TemperatureSensitivePUEModel.__name__ == "TemperatureSensitivePUEModel"
    assert ReplayStudyConfig.__name__ == "ReplayStudyConfig"
    assert ReplayStudyRunner.__name__ == "ReplayStudyRunner"
    assert ReplayArtifactWriter.__name__ == "ReplayArtifactWriter"
    assert ReplaySuiteConfig.__name__ == "ReplaySuiteConfig"
    assert ReplaySuiteRunner.__name__ == "ReplaySuiteRunner"
    assert ReplaySuiteArtifactWriter.__name__ == "ReplaySuiteArtifactWriter"
    assert RollingReplayEngine.__name__ == "RollingReplayEngine"
    assert RollingReplayResult.__name__ == "RollingReplayResult"
    assert RollingReplaySettings.__name__ == "RollingReplaySettings"
    assert RISK_AWARE_POLICY == "risk_aware"
    assert RISK_AWARE_POLICY in ALL_POLICY_NAMES
    assert SourceManifest.__name__ == "SourceManifest"
    assert packaged_study_path().is_file()
    assert packaged_suite_path().is_file()
    assert PrometheusRangeAdapter.__name__ == "PrometheusRangeAdapter"
    assert KeplerPrometheusAdapter.__name__ == "KeplerPrometheusAdapter"
    assert CarbonAwareSDKAdapter.__name__ == "CarbonAwareSDKAdapter"
    assert SustainDCAdapter.__name__ == "SustainDCAdapter"
