from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import uuid4

import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from climadc import __version__
from climadc.adapters.local import read_flexible_workload
from climadc.contracts import FlexibleWorkloadFrame
from climadc.contracts.frames import FLEXIBLE_WORKLOAD_COLUMNS
from climadc.errors import ConfigurationError
from climadc.evidence.checksums import (
    CHECKSUM_FILE,
    safe_relative_path,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from climadc.evidence.environment import git_state
from climadc.evidence.manifest import Sha256

GOOGLE_V3_DATASET: Literal["google-clusterdata-2019-v3"] = "google-clusterdata-2019-v3"
GOOGLE_V3_OFFICIAL_URL: Literal[
    "https://github.com/google/cluster-data/blob/master/ClusterData2019.md"
] = "https://github.com/google/cluster-data/blob/master/ClusterData2019.md"
GOOGLE_V3_LICENSE: Literal["CC-BY-4.0"] = "CC-BY-4.0"
GOOGLE_V3_EXPORT_COLUMNS = (
    "collection_id",
    "instance_index",
    "submit_time_us",
    "finish_time_us",
    "requested_cpu",
    "priority",
    "scheduling_class",
    "missing_type",
    "collection_type",
    "alloc_collection_id",
    "submit_count",
    "finish_count",
)
GOOGLE_V3_CONFIG_FILE: Literal["conversion-config.yaml"] = "conversion-config.yaml"
GOOGLE_V3_MANIFEST_FILE: Literal["conversion-manifest.json"] = "conversion-manifest.json"
GOOGLE_V3_QUERY_FILE: Literal["export-query.sql"] = "export-query.sql"
GOOGLE_V3_WORKLOAD_FILE: Literal["workload.csv"] = "workload.csv"
GOOGLE_V3_ARTIFACTS = tuple(
    sorted(
        (
            CHECKSUM_FILE,
            GOOGLE_V3_CONFIG_FILE,
            GOOGLE_V3_MANIFEST_FILE,
            GOOGLE_V3_QUERY_FILE,
            GOOGLE_V3_WORKLOAD_FILE,
        )
    )
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_NONNEGATIVE_INTEGER = re.compile(r"^[0-9]+$")


def _utc_datetime(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


class GoogleV3TraceWindow(BaseModel):
    """Bounded relative-time window used by the recorded BigQuery export."""

    model_config = ConfigDict(extra="forbid")

    start_time_us: int = Field(ge=0)
    end_time_us: int = Field(gt=0)
    finish_cutoff_time_us: int = Field(gt=0)

    @model_validator(mode="after")
    def times_are_ordered(self) -> GoogleV3TraceWindow:
        if not self.start_time_us < self.end_time_us <= self.finish_cutoff_time_us:
            raise ValueError(
                "trace window requires start_time_us < end_time_us <= finish_cutoff_time_us"
            )
        return self


class GoogleV3PowerMapping(BaseModel):
    """Declared CPU-to-power assumption; it is not measured facility power."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["requested_cpu_linear"] = "requested_cpu_linear"
    kw_per_normalized_cpu: float = Field(gt=0.0)
    utilization_fraction: float = Field(gt=0.0, le=1.0)

    @field_validator("kw_per_normalized_cpu", "utilization_fraction")
    @classmethod
    def values_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("power mapping values must be finite")
        return value


class GoogleV3DeadlineMapping(BaseModel):
    """Declared deadline window derived from the trace-complete observed runtime."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["observed_runtime_multiplier"] = "observed_runtime_multiplier"
    multiplier: float = Field(ge=1.0)

    @field_validator("multiplier")
    @classmethod
    def multiplier_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("deadline multiplier must be finite")
        return value


class GoogleV3PreemptibilityMapping(BaseModel):
    """Explicit scenario assumption required by ClimaDC's current workload contract."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["assume_preemptible"] = "assume_preemptible"
    value: Literal[True] = True


class GoogleV3ConversionConfig(BaseModel):
    """Fail-closed mapping contract for one exported ClusterData2019 task slice."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    dataset: Literal["google-clusterdata-2019-v3"]
    cell: Literal["a", "b", "c", "d", "e", "f", "g", "h"]
    site_id: NonEmptyText
    exported_at: datetime
    input_sha256: Sha256
    scenario_epoch: datetime
    trace_window: GoogleV3TraceWindow
    allowed_scheduling_classes: list[Literal[0, 1]] = Field(min_length=1)
    power_mapping: GoogleV3PowerMapping
    deadline_mapping: GoogleV3DeadlineMapping
    preemptibility_mapping: GoogleV3PreemptibilityMapping

    @field_validator("exported_at")
    @classmethod
    def exported_at_is_utc(cls, value: datetime) -> datetime:
        return _utc_datetime(value, field="exported_at")

    @field_validator("scenario_epoch")
    @classmethod
    def scenario_epoch_is_utc(cls, value: datetime) -> datetime:
        return _utc_datetime(value, field="scenario_epoch")

    @field_validator("allowed_scheduling_classes")
    @classmethod
    def scheduling_classes_are_stable(cls, values: list[int]) -> list[int]:
        if values != sorted(set(values)):
            raise ValueError("allowed_scheduling_classes must be unique and sorted")
        return values

    @classmethod
    def from_yaml(cls, path: Path) -> GoogleV3ConversionConfig:
        try:
            payload = Path(path).read_bytes()
            raw = yaml.safe_load(payload.decode("utf-8"))
            return cls.model_validate(raw)
        except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
            raise ConfigurationError(f"Invalid Google v3 conversion config {path}: {exc}") from exc


class GoogleV3DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["Google"] = "Google"
    dataset: Literal["google-clusterdata-2019-v3"] = GOOGLE_V3_DATASET
    official_url: Literal[
        "https://github.com/google/cluster-data/blob/master/ClusterData2019.md"
    ] = GOOGLE_V3_OFFICIAL_URL
    license: Literal["CC-BY-4.0"] = GOOGLE_V3_LICENSE
    source_time_unit: Literal["microseconds since trace start"] = "microseconds since trace start"


class GoogleV3SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_name: NonEmptyText
    input_sha256: Sha256
    input_bytes: int = Field(gt=0)
    rows: int = Field(gt=0)
    query_path: Literal["export-query.sql"] = GOOGLE_V3_QUERY_FILE
    query_sha256: Sha256
    exported_at: datetime
    cell: Literal["a", "b", "c", "d", "e", "f", "g", "h"]
    trace_window: GoogleV3TraceWindow

    @field_validator("csv_name")
    @classmethod
    def csv_name_is_a_basename(cls, value: str) -> str:
        if Path(value).name != value or "/" in value or "\\" in value:
            raise ValueError("csv_name must be a basename without directories")
        return value

    @field_validator("exported_at")
    @classmethod
    def exported_at_is_utc(cls, value: datetime) -> datetime:
        return _utc_datetime(value, field="exported_at")


class GoogleV3TimeSemanticsRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_epoch: datetime
    scenario_epoch_basis: Literal["scenario_assumption"] = "scenario_assumption"
    release_time_basis: Literal["trace_submit_event"] = "trace_submit_event"
    available_at_basis: Literal["trace_submit_event"] = "trace_submit_event"
    runtime_basis: Literal["trace_finish_minus_submit"] = "trace_finish_minus_submit"
    deadline_basis: Literal["observed_runtime_scenario_mapping"] = (
        "observed_runtime_scenario_mapping"
    )
    observed_runtime_uses_future_trace_fact: Literal[True] = True

    @field_validator("scenario_epoch")
    @classmethod
    def scenario_epoch_is_utc(cls, value: datetime) -> datetime:
        return _utc_datetime(value, field="scenario_epoch")


class GoogleV3OutputRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Literal["workload.csv"] = GOOGLE_V3_WORKLOAD_FILE
    sha256: Sha256
    rows: int = Field(gt=0)
    contract: Literal["FlexibleWorkloadFrame"] = "FlexibleWorkloadFrame"


class GoogleV3ConversionManifest(BaseModel):
    """Portable provenance contract for a Google v3 workload conversion."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    conversion_type: Literal["google-v3-task-summary-to-flexible-workload"] = (
        "google-v3-task-summary-to-flexible-workload"
    )
    climadc_version: NonEmptyText
    git_commit: NonEmptyText
    git_dirty: bool | None
    highest_possible_evidence_level: Literal["E2"] = "E2"
    evidence_status: Literal["DATA_REQUIRED"] = "DATA_REQUIRED"
    claim_eligible: Literal[False] = False
    dataset: GoogleV3DatasetRecord
    source: GoogleV3SourceRecord
    config_path: Literal["conversion-config.yaml"] = GOOGLE_V3_CONFIG_FILE
    config_sha256: Sha256
    time_semantics: GoogleV3TimeSemanticsRecord
    allowed_scheduling_classes: list[Literal[0, 1]]
    power_mapping: GoogleV3PowerMapping
    deadline_mapping: GoogleV3DeadlineMapping
    preemptibility_mapping: GoogleV3PreemptibilityMapping
    output: GoogleV3OutputRecord
    artifacts: list[NonEmptyText]
    limitations: list[NonEmptyText] = Field(min_length=1)

    @field_validator("allowed_scheduling_classes")
    @classmethod
    def scheduling_classes_are_stable(cls, values: list[int]) -> list[int]:
        if values != sorted(set(values)):
            raise ValueError("allowed_scheduling_classes must be unique and sorted")
        return values

    @field_validator("artifacts")
    @classmethod
    def artifacts_are_safe_unique_sorted(cls, values: list[str]) -> list[str]:
        checked = [safe_relative_path(value) for value in values]
        if checked != sorted(set(checked)):
            raise ValueError("artifacts must be unique and use stable lexical ordering")
        return checked


@dataclass(frozen=True)
class GoogleV3ConversionResult:
    output_directory: Path
    rows: int
    input_sha256: str
    workload_sha256: str


@dataclass(frozen=True)
class GoogleV3ConversionVerification:
    directory: Path
    rows: int
    source_verified: bool


@dataclass(frozen=True)
class _GoogleV3Task:
    collection_id: int
    instance_index: int
    submit_time_us: int
    finish_time_us: int
    requested_cpu: Decimal
    priority: int
    scheduling_class: int


def _parse_nonnegative_integer(value: str, *, column: str, row_number: int) -> int:
    if _NONNEGATIVE_INTEGER.fullmatch(value) is None:
        raise ConfigurationError(
            f"Google v3 export row {row_number} column {column} must be a nonnegative integer"
        )
    return int(value)


def _parse_positive_decimal(value: str, *, column: str, row_number: int) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ConfigurationError(
            f"Google v3 export row {row_number} column {column} must be numeric"
        ) from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ConfigurationError(
            f"Google v3 export row {row_number} column {column} must be finite and positive"
        )
    return parsed


def _read_google_v3_export(
    source_csv: Path,
    config: GoogleV3ConversionConfig,
) -> tuple[_GoogleV3Task, ...]:
    try:
        raw = pd.read_csv(
            source_csv,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise ConfigurationError(f"Unable to read Google v3 export {source_csv}: {exc}") from exc

    actual_columns = list(raw.columns)
    expected_columns = list(GOOGLE_V3_EXPORT_COLUMNS)
    if len(actual_columns) != len(expected_columns) or set(actual_columns) != set(expected_columns):
        missing = sorted(set(expected_columns).difference(actual_columns))
        extra = sorted(set(actual_columns).difference(expected_columns))
        raise ConfigurationError(
            f"Google v3 export requires exact columns; missing={missing}, extra={extra}"
        )
    if raw.empty:
        raise ConfigurationError("Google v3 export must contain at least one task")

    tasks: list[_GoogleV3Task] = []
    keys: set[tuple[int, int]] = set()
    window = config.trace_window
    allowed = set(config.allowed_scheduling_classes)
    for row_number, (_, row) in enumerate(raw.loc[:, expected_columns].iterrows(), start=2):
        values = {column: str(row[column]) for column in expected_columns}
        collection_id = _parse_nonnegative_integer(
            values["collection_id"], column="collection_id", row_number=row_number
        )
        instance_index = _parse_nonnegative_integer(
            values["instance_index"], column="instance_index", row_number=row_number
        )
        submit_time_us = _parse_nonnegative_integer(
            values["submit_time_us"], column="submit_time_us", row_number=row_number
        )
        finish_time_us = _parse_nonnegative_integer(
            values["finish_time_us"], column="finish_time_us", row_number=row_number
        )
        requested_cpu = _parse_positive_decimal(
            values["requested_cpu"], column="requested_cpu", row_number=row_number
        )
        priority = _parse_nonnegative_integer(
            values["priority"], column="priority", row_number=row_number
        )
        scheduling_class = _parse_nonnegative_integer(
            values["scheduling_class"], column="scheduling_class", row_number=row_number
        )
        missing_type = _parse_nonnegative_integer(
            values["missing_type"], column="missing_type", row_number=row_number
        )
        collection_type = _parse_nonnegative_integer(
            values["collection_type"], column="collection_type", row_number=row_number
        )
        alloc_collection_id = _parse_nonnegative_integer(
            values["alloc_collection_id"],
            column="alloc_collection_id",
            row_number=row_number,
        )
        submit_count = _parse_nonnegative_integer(
            values["submit_count"], column="submit_count", row_number=row_number
        )
        finish_count = _parse_nonnegative_integer(
            values["finish_count"], column="finish_count", row_number=row_number
        )

        if not window.start_time_us <= submit_time_us < window.end_time_us:
            raise ConfigurationError(
                f"Google v3 export row {row_number} submit_time_us is outside trace_window"
            )
        if not submit_time_us < finish_time_us < window.finish_cutoff_time_us:
            raise ConfigurationError(
                f"Google v3 export row {row_number} requires submit_time_us < finish_time_us "
                "< finish_cutoff_time_us"
            )
        if scheduling_class not in allowed:
            raise ConfigurationError(
                f"Google v3 export row {row_number} scheduling_class {scheduling_class} "
                f"is outside declared classes {sorted(allowed)}"
            )
        if missing_type != 0:
            raise ConfigurationError(
                f"Google v3 export row {row_number} has synthesized/missing event data"
            )
        if collection_type != 0 or alloc_collection_id != 0:
            raise ConfigurationError(
                f"Google v3 export row {row_number} is not a top-level job task"
            )
        if submit_count != 1 or finish_count != 1:
            raise ConfigurationError(
                f"Google v3 export row {row_number} requires exactly one submit and one finish"
            )
        key = (collection_id, instance_index)
        if key in keys:
            raise ConfigurationError(
                f"Google v3 export contains duplicate task key {collection_id}/{instance_index}"
            )
        keys.add(key)
        tasks.append(
            _GoogleV3Task(
                collection_id=collection_id,
                instance_index=instance_index,
                submit_time_us=submit_time_us,
                finish_time_us=finish_time_us,
                requested_cpu=requested_cpu,
                priority=priority,
                scheduling_class=scheduling_class,
            )
        )
    return tuple(sorted(tasks, key=lambda task: (task.collection_id, task.instance_index)))


def _timestamp(epoch: datetime, offset_us: int, *, label: str) -> pd.Timestamp:
    try:
        value = pd.Timestamp(epoch) + pd.Timedelta(microseconds=offset_us)
    except (OverflowError, ValueError) as exc:
        raise ConfigurationError(f"Unable to map {label} into the scenario epoch") from exc
    if pd.isna(value):
        raise ConfigurationError(f"Unable to map {label} into the scenario epoch")
    return value


def _convert_tasks(
    tasks: tuple[_GoogleV3Task, ...],
    config: GoogleV3ConversionConfig,
) -> FlexibleWorkloadFrame:
    rows: list[dict[str, object]] = []
    kw_per_cpu = Decimal(str(config.power_mapping.kw_per_normalized_cpu))
    utilization = Decimal(str(config.power_mapping.utilization_fraction))
    deadline_multiplier = Decimal(str(config.deadline_mapping.multiplier))
    microseconds_per_hour = Decimal("3600000000")
    for task in tasks:
        runtime_us = task.finish_time_us - task.submit_time_us
        deadline_offset_us = int(
            (Decimal(runtime_us) * deadline_multiplier).to_integral_value(rounding=ROUND_CEILING)
        )
        max_power = task.requested_cpu * kw_per_cpu
        energy = max_power * Decimal(runtime_us) / microseconds_per_hour * utilization
        max_power_float = float(max_power)
        energy_float = float(energy)
        if (
            not math.isfinite(max_power_float)
            or not math.isfinite(energy_float)
            or max_power_float <= 0.0
            or energy_float <= 0.0
        ):
            raise ConfigurationError(
                f"Google v3 task {task.collection_id}/{task.instance_index} mapping is non-finite"
            )
        release = _timestamp(
            config.scenario_epoch,
            task.submit_time_us,
            label=f"task {task.collection_id}/{task.instance_index} release",
        )
        deadline = _timestamp(
            config.scenario_epoch,
            task.submit_time_us + deadline_offset_us,
            label=f"task {task.collection_id}/{task.instance_index} deadline",
        )
        rows.append(
            {
                "job_id": f"google-v3-{config.cell}-{task.collection_id}-{task.instance_index}",
                "site_id": config.site_id,
                "release_time": release,
                "available_at": release,
                "deadline": deadline,
                "energy": energy_float,
                "energy_unit": "kWh",
                "max_power": max_power_float,
                "power_unit": "kW",
                "preemptible": config.preemptibility_mapping.value,
                "priority": task.priority,
            }
        )
    return FlexibleWorkloadFrame.from_pandas(
        pd.DataFrame(rows, columns=list(FLEXIBLE_WORKLOAD_COLUMNS))
    )


def _timestamp_text(value: object) -> str:
    timestamp = pd.Timestamp(cast(str | int | float | datetime, value))
    return timestamp.isoformat().replace("+00:00", "Z")


def _number_text(value: object) -> str:
    return format(float(cast(float, value)), ".15g")


def _workload_csv_bytes(workload: FlexibleWorkloadFrame) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(FLEXIBLE_WORKLOAD_COLUMNS)
    frame = workload.to_pandas()
    for _, row in frame.iterrows():
        writer.writerow(
            (
                str(row["job_id"]),
                str(row["site_id"]),
                _timestamp_text(row["release_time"]),
                _timestamp_text(row["available_at"]),
                _timestamp_text(row["deadline"]),
                _number_text(row["energy"]),
                str(row["energy_unit"]),
                _number_text(row["max_power"]),
                str(row["power_unit"]),
                "True" if bool(row["preemptible"]) else "False",
                _number_text(row["priority"]),
            )
        )
    return stream.getvalue().encode("utf-8")


def _json_text(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _read_nonempty_bytes(path: Path, *, label: str) -> bytes:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise ConfigurationError(f"Unable to read {label} {path}: {exc}") from exc
    if not payload:
        raise ConfigurationError(f"{label} must not be empty: {path}")
    return payload


def _validate_query(query_bytes: bytes, config: GoogleV3ConversionConfig) -> None:
    try:
        query = query_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise ConfigurationError("Google v3 export query must be UTF-8 text") from exc
    expected_table = (
        f"`google.com:google-cluster-data`.clusterdata_2019_{config.cell}.instance_events"
    )
    if expected_table not in query:
        raise ConfigurationError(
            f"Google v3 export query must reference the declared cell table {expected_table}"
        )
    missing_parameters = sorted(
        parameter
        for parameter in ("@start_time_us", "@end_time_us", "@finish_cutoff_time_us")
        if parameter not in query
    )
    if missing_parameters:
        raise ConfigurationError(
            f"Google v3 export query is missing bounded-window parameters: {missing_parameters}"
        )


def _configuration_payload(path: Path) -> tuple[GoogleV3ConversionConfig, bytes]:
    payload = _read_nonempty_bytes(path, label="Google v3 conversion config")
    return GoogleV3ConversionConfig.from_yaml(path), payload


def _manifest(
    *,
    config: GoogleV3ConversionConfig,
    source_csv: Path,
    source_bytes: int,
    source_rows: int,
    query_sha256: str,
    config_sha256: str,
    workload_sha256: str,
) -> GoogleV3ConversionManifest:
    commit, dirty = git_state()
    return GoogleV3ConversionManifest(
        climadc_version=__version__,
        git_commit=commit,
        git_dirty=dirty,
        dataset=GoogleV3DatasetRecord(),
        source=GoogleV3SourceRecord(
            csv_name=source_csv.name,
            input_sha256=config.input_sha256,
            input_bytes=source_bytes,
            rows=source_rows,
            query_sha256=query_sha256,
            exported_at=config.exported_at,
            cell=config.cell,
            trace_window=config.trace_window,
        ),
        config_sha256=config_sha256,
        time_semantics=GoogleV3TimeSemanticsRecord(scenario_epoch=config.scenario_epoch),
        allowed_scheduling_classes=config.allowed_scheduling_classes,
        power_mapping=config.power_mapping,
        deadline_mapping=config.deadline_mapping,
        preemptibility_mapping=config.preemptibility_mapping,
        output=GoogleV3OutputRecord(sha256=workload_sha256, rows=source_rows),
        artifacts=list(GOOGLE_V3_ARTIFACTS),
        limitations=[
            "The Google trace uses relative time; scenario_epoch is an explicit analysis anchor, not an observed wall-clock timestamp.",
            "Energy, maximum power, deadlines, and preemptibility are declared scenario mappings, not fields measured or promised by Google.",
            "Observed finish time is a trace-complete future fact used to construct job energy and deadline assumptions; it must not be exposed as decision-time telemetry.",
            "This conversion alone is not an E2 result or a same-site operational validation; independent slices and external grid/weather vintages remain DATA_REQUIRED.",
        ],
    )


def convert_google_v3_export(
    source_csv: Path,
    config_path: Path,
    query_sql: Path,
    output_directory: Path,
) -> GoogleV3ConversionResult:
    """Convert one hash-bound Google v3 CSV export into a provenance-bearing workload bundle."""

    source_csv = Path(source_csv)
    config_path = Path(config_path)
    query_sql = Path(query_sql)
    config, config_bytes = _configuration_payload(config_path)
    source_payload = _read_nonempty_bytes(source_csv, label="Google v3 source CSV")
    actual_input_sha256 = hashlib.sha256(source_payload).hexdigest()
    if actual_input_sha256 != config.input_sha256:
        raise ConfigurationError(
            "Google v3 source SHA-256 mismatch: "
            f"expected {config.input_sha256}, found {actual_input_sha256}"
        )
    query_bytes = _read_nonempty_bytes(query_sql, label="Google v3 export query")
    _validate_query(query_bytes, config)
    tasks = _read_google_v3_export(source_csv, config)
    workload = _convert_tasks(tasks, config)
    workload_bytes = _workload_csv_bytes(workload)

    raw_output = Path(output_directory)
    if raw_output.name in {"", ".", ".."}:
        raise ConfigurationError("Google v3 output directory must name a child directory")
    try:
        parent = raw_output.parent.resolve()
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(f"Unable to prepare Google v3 output parent: {exc}") from exc
    destination = parent / raw_output.name
    if destination.exists():
        raise ConfigurationError(f"Google v3 output directory already exists: {destination}")
    temporary = parent / f".{raw_output.name}.tmp-{uuid4().hex}"
    try:
        temporary.mkdir()
        (temporary / GOOGLE_V3_CONFIG_FILE).write_bytes(config_bytes)
        (temporary / GOOGLE_V3_QUERY_FILE).write_bytes(query_bytes)
        (temporary / GOOGLE_V3_WORKLOAD_FILE).write_bytes(workload_bytes)
        query_sha256 = sha256_file(temporary / GOOGLE_V3_QUERY_FILE)
        config_sha256 = sha256_file(temporary / GOOGLE_V3_CONFIG_FILE)
        workload_sha256 = sha256_file(temporary / GOOGLE_V3_WORKLOAD_FILE)
        manifest = _manifest(
            config=config,
            source_csv=source_csv,
            source_bytes=len(source_payload),
            source_rows=len(tasks),
            query_sha256=query_sha256,
            config_sha256=config_sha256,
            workload_sha256=workload_sha256,
        )
        (temporary / GOOGLE_V3_MANIFEST_FILE).write_text(
            _json_text(manifest.model_dump(mode="json")), encoding="utf-8", newline="\n"
        )
        write_checksums(temporary)
        temporary.rename(destination)
    except ConfigurationError:
        raise
    except OSError as exc:
        raise ConfigurationError(f"Unable to publish Google v3 conversion: {exc}") from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)

    verify_google_v3_conversion(destination, source_csv=source_csv)
    return GoogleV3ConversionResult(
        output_directory=destination,
        rows=len(tasks),
        input_sha256=actual_input_sha256,
        workload_sha256=hashlib.sha256(workload_bytes).hexdigest(),
    )


def _load_manifest(directory: Path) -> GoogleV3ConversionManifest:
    path = directory / GOOGLE_V3_MANIFEST_FILE
    try:
        return GoogleV3ConversionManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise ConfigurationError(f"Invalid Google v3 conversion manifest {path}: {exc}") from exc


def verify_google_v3_conversion(
    directory: Path,
    *,
    source_csv: Path | None = None,
) -> GoogleV3ConversionVerification:
    """Verify a conversion bundle and optionally reproduce it from the hash-bound source CSV."""

    root = Path(directory)
    if not root.is_dir():
        raise ConfigurationError(f"Google v3 conversion directory does not exist: {root}")
    checksums = verify_checksums(root)
    expected_checksum_entries = set(GOOGLE_V3_ARTIFACTS).difference({CHECKSUM_FILE})
    if set(checksums) != expected_checksum_entries:
        raise ConfigurationError(
            "Google v3 conversion checksum membership mismatch: "
            f"expected={sorted(expected_checksum_entries)}, found={sorted(checksums)}"
        )
    manifest = _load_manifest(root)
    if tuple(manifest.artifacts) != GOOGLE_V3_ARTIFACTS:
        raise ConfigurationError("Google v3 conversion manifest artifact membership mismatch")

    config_path = root / GOOGLE_V3_CONFIG_FILE
    config = GoogleV3ConversionConfig.from_yaml(config_path)
    if sha256_file(config_path) != manifest.config_sha256:
        raise ConfigurationError("Google v3 conversion config hash does not match manifest")
    if config.input_sha256 != manifest.source.input_sha256:
        raise ConfigurationError("Google v3 conversion source hash does not match config")
    if config.cell != manifest.source.cell or config.trace_window != manifest.source.trace_window:
        raise ConfigurationError("Google v3 conversion source window does not match config")
    if config.exported_at != manifest.source.exported_at:
        raise ConfigurationError("Google v3 conversion exported_at does not match config")
    if config.scenario_epoch != manifest.time_semantics.scenario_epoch:
        raise ConfigurationError("Google v3 conversion scenario_epoch does not match config")
    if config.allowed_scheduling_classes != manifest.allowed_scheduling_classes:
        raise ConfigurationError("Google v3 conversion scheduling classes do not match config")
    if config.power_mapping != manifest.power_mapping:
        raise ConfigurationError("Google v3 conversion power mapping does not match config")
    if config.deadline_mapping != manifest.deadline_mapping:
        raise ConfigurationError("Google v3 conversion deadline mapping does not match config")
    if config.preemptibility_mapping != manifest.preemptibility_mapping:
        raise ConfigurationError(
            "Google v3 conversion preemptibility mapping does not match config"
        )

    query_path = root / GOOGLE_V3_QUERY_FILE
    query_bytes = _read_nonempty_bytes(query_path, label="Google v3 export query")
    _validate_query(query_bytes, config)
    if sha256_file(query_path) != manifest.source.query_sha256:
        raise ConfigurationError("Google v3 conversion query hash does not match manifest")

    workload_path = root / GOOGLE_V3_WORKLOAD_FILE
    if sha256_file(workload_path) != manifest.output.sha256:
        raise ConfigurationError("Google v3 workload hash does not match manifest")
    workload = read_flexible_workload(workload_path, "csv", {}, "UTC")
    rows = len(workload.to_pandas())
    if rows != manifest.output.rows or rows != manifest.source.rows:
        raise ConfigurationError("Google v3 workload row count does not match manifest")

    source_verified = False
    if source_csv is not None:
        source_path = Path(source_csv)
        if sha256_file(source_path) != manifest.source.input_sha256:
            raise ConfigurationError("Google v3 source CSV hash does not match manifest")
        tasks = _read_google_v3_export(source_path, config)
        reproduced = _workload_csv_bytes(_convert_tasks(tasks, config))
        if reproduced != workload_path.read_bytes():
            raise ConfigurationError("Google v3 workload does not reproduce from source and config")
        if len(tasks) != rows:
            raise ConfigurationError("Google v3 reproduced row count does not match workload")
        source_verified = True

    return GoogleV3ConversionVerification(
        directory=root.resolve(),
        rows=rows,
        source_verified=source_verified,
    )


__all__ = [
    "GOOGLE_V3_ARTIFACTS",
    "GOOGLE_V3_EXPORT_COLUMNS",
    "GoogleV3ConversionConfig",
    "GoogleV3ConversionResult",
    "GoogleV3ConversionVerification",
    "convert_google_v3_export",
    "verify_google_v3_conversion",
]
