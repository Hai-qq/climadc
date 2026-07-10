from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from climadc.config import StudyConfig
from climadc.errors import ConfigurationError


def _study_yaml(*, horizon: str = "4h", timezone: str = "UTC") -> str:
    return f"""study_id: demo
horizon: {horizon}
climate:
  path: climate.csv
  format: csv
  timezone: {timezone}
  card: climate-card.yaml
telemetry:
  path: telemetry.csv
  format: csv
  timezone: UTC
  card: telemetry-card.yaml
backtest:
  strategy: blocked
  min_train: 48
  calibration_size: 12
  test_size: 12
  step: 12
models:
  - kind: persistence
    model_id: persistence
decision:
  enabled: false
"""


def test_study_config_loads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    path.write_text(
        """study_id: demo
horizon: 4h
climate:
  path: climate.csv
  format: csv
  timezone: UTC
  card: climate-card.yaml
telemetry:
  path: telemetry.csv
  format: csv
  timezone: UTC
  card: telemetry-card.yaml
backtest:
  strategy: blocked
  min_train: 48
  calibration_size: 12
  test_size: 12
  step: 12
models:
  - kind: persistence
    model_id: persistence
decision:
  enabled: false
""",
        encoding="utf-8",
    )
    cfg = StudyConfig.from_yaml(path)
    assert cfg.study_id == "demo"
    assert cfg.horizon == "4h"


def test_study_config_resolves_relative_paths(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    path.write_text(
        _study_yaml()
        + """workload:
  path: workload.csv
  format: parquet
  timezone: UTC
  card: workload-card.yaml
output_dir: custom-runs
""",
        encoding="utf-8",
    )

    cfg = StudyConfig.from_yaml(path)

    assert cfg.climate.path == (tmp_path / "climate.csv").resolve()
    assert cfg.climate.card == (tmp_path / "climate-card.yaml").resolve()
    assert cfg.telemetry.path == (tmp_path / "telemetry.csv").resolve()
    assert cfg.telemetry.card == (tmp_path / "telemetry-card.yaml").resolve()
    assert cfg.workload is not None
    assert cfg.workload.path == (tmp_path / "workload.csv").resolve()
    assert cfg.workload.card == (tmp_path / "workload-card.yaml").resolve()
    assert cfg.output_dir == (tmp_path / "custom-runs").resolve()


def test_study_config_preserves_absolute_paths(tmp_path: Path) -> None:
    absolute_data = tmp_path / "absolute.csv"
    path = tmp_path / "study.yaml"
    path.write_text(
        _study_yaml().replace("path: climate.csv", f"path: {absolute_data}", 1),
        encoding="utf-8",
    )

    cfg = StudyConfig.from_yaml(path)

    assert cfg.climate.path == absolute_data


@pytest.mark.parametrize("horizon", ["0h", "-1h", "not-a-duration"])
def test_study_config_rejects_invalid_horizon(tmp_path: Path, horizon: str) -> None:
    path = tmp_path / "study.yaml"
    path.write_text(_study_yaml(horizon=horizon), encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        StudyConfig.from_yaml(path)

    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_study_config_rejects_invalid_timezone(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    path.write_text(_study_yaml(timezone="Mars/Olympus_Mons"), encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        StudyConfig.from_yaml(path)

    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_study_config_wraps_file_errors(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigurationError, match="Invalid study config") as exc_info:
        StudyConfig.from_yaml(path)

    assert str(path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, OSError)


def test_study_config_wraps_encoding_errors(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.yaml"
    path.write_bytes(b"\xff")

    with pytest.raises(ConfigurationError) as exc_info:
        StudyConfig.from_yaml(path)

    assert str(path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_study_config_wraps_yaml_errors(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("models: [", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid study config") as exc_info:
        StudyConfig.from_yaml(path)

    assert str(path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, yaml.YAMLError)


def test_study_config_wraps_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("study_id: incomplete", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid study config") as exc_info:
        StudyConfig.from_yaml(path)

    assert str(path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValidationError)
