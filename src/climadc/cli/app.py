from pathlib import Path
from typing import Annotated, Optional, cast

import pandas as pd
import typer

from climadc import __version__
from climadc.adapters.google_clusterdata import (
    convert_google_v3_export,
    verify_google_v3_conversion,
)
from climadc.benchmark import BenchmarkRunner
from climadc.cli.scaffold import scaffold_study
from climadc.config import StudyConfig
from climadc.errors import ClimaDCError, ConfigurationError
from climadc.evidence.verify import verify_run as verify_run_directory
from climadc.evidence.verify import verify_suite as verify_suite_directory
from climadc.reference import packaged_study_path, packaged_suite_path, refresh_carbon_shift
from climadc.replay import (
    ReplayArtifactWriter,
    ReplayStudyConfig,
    ReplayStudyRunner,
    ReplaySuiteArtifactWriter,
    ReplaySuiteConfig,
    ReplaySuiteRunner,
)
from climadc.reporting import ArtifactWriter, resolve_run_path

app = typer.Typer(no_args_is_help=True)
demo_app = typer.Typer(no_args_is_help=True)
trace_app = typer.Typer(no_args_is_help=True)
app.add_typer(demo_app, name="demo")
app.add_typer(trace_app, name="trace")


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"climadc {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(False, "--version", callback=version_callback, is_eager=True),
) -> None:
    """Climate-aware forecasting and benchmarking for data centers."""


def _user_error(exc: ClimaDCError) -> None:
    typer.echo(str(exc), err=True)
    raise typer.Exit(code=1)


@app.command()
def init(directory: Path) -> None:
    """Create a deterministic local example study."""

    try:
        config_path = scaffold_study(directory)
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(str(config_path.resolve()))


@app.command()
def validate(config_path: Path) -> None:
    """Load and validate a study without running models."""

    try:
        config = StudyConfig.from_yaml(config_path)
        data = BenchmarkRunner().validate(config)
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(f"Validated {config.study_id}: {len(data.cards)} inputs")


@app.command()
def benchmark(config_path: Path) -> None:
    """Execute a benchmark and atomically publish its artifacts."""

    try:
        config = StudyConfig.from_yaml(config_path)
        result = BenchmarkRunner().run(config)
        run_path = ArtifactWriter().write(result, config.output_dir)
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(str(run_path))


@app.command()
def replay(
    config_path: Path,
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir")] = None,
) -> None:
    """Run a verified local engineering replay and publish auditable artifacts."""

    try:
        config = ReplayStudyConfig.from_yaml(config_path)
        if output_dir is not None:
            config = config.with_output_dir(output_dir)
        result = ReplayStudyRunner().run(config)
        run_path = ReplayArtifactWriter().write(result, config.output_dir)
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(str(run_path))


@app.command("replay-suite")
def replay_suite(
    config_path: Path,
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir")] = None,
) -> None:
    """Run comparable replay studies and publish a sensitivity/robustness report."""

    try:
        config = ReplaySuiteConfig.from_yaml(config_path)
        if output_dir is not None:
            config = config.with_output_dir(output_dir)
        result = ReplaySuiteRunner().run(config)
        run_path = ReplaySuiteArtifactWriter().write(result, config.output_dir)
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(str(run_path))


@demo_app.command("carbon-shift")
def demo_carbon_shift(
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("climadc-replay-runs"),
) -> None:
    """Run the packaged 24-hour Great Britain carbon-shift fixture offline."""

    try:
        config = ReplayStudyConfig.from_yaml(packaged_study_path()).with_output_dir(output_dir)
        result = ReplayStudyRunner().run(config)
        run_path = ReplayArtifactWriter().write(result, config.output_dir)
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(str(run_path))


def _demo_sensitivity_suite(output_dir: Path) -> None:
    try:
        config = ReplaySuiteConfig.from_yaml(packaged_suite_path()).with_output_dir(output_dir)
        result = ReplaySuiteRunner().run(config)
        run_path = ReplaySuiteArtifactWriter().write(result, config.output_dir)
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(str(run_path))


@demo_app.command("sensitivity-suite")
def demo_sensitivity_suite(
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("climadc-replay-suite-runs"),
) -> None:
    """Run the packaged four-scenario sensitivity analysis offline."""

    _demo_sensitivity_suite(output_dir)


@demo_app.command("robustness-suite")
def demo_robustness_suite(
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("climadc-replay-suite-runs"),
) -> None:
    """Deprecated alias for demo sensitivity-suite."""

    typer.echo(
        "DEPRECATION: 'demo robustness-suite' is a compatibility alias; "
        "use 'demo sensitivity-suite'.",
        err=True,
    )
    _demo_sensitivity_suite(output_dir)


@demo_app.command("refresh-carbon-shift")
def demo_refresh_carbon_shift(
    destination: Path,
    decision_date: Annotated[str, typer.Option("--decision-date", help="UTC date: YYYY-MM-DD")],
) -> None:
    """Fetch a new historical GB snapshot into a new local directory."""

    try:
        try:
            decision_time = pd.Timestamp(decision_date)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError("decision-date must use YYYY-MM-DD") from exc
        if (
            decision_time.strftime("%Y-%m-%d") != decision_date
            or decision_time.time() != pd.Timestamp("00:00").time()
        ):
            raise ConfigurationError("decision-date must use YYYY-MM-DD")
        decision_time = decision_time.tz_localize("UTC")
        config_path = refresh_carbon_shift(destination, decision_time=decision_time)
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(str(config_path.resolve()))


@trace_app.command("convert-google-v3")
def trace_convert_google_v3(
    source_csv: Path,
    config_path: Path,
    output_directory: Path,
    query_sql: Annotated[Path, typer.Option("--query-sql")],
) -> None:
    """Convert a hash-bound Google ClusterData2019 task export offline."""

    try:
        result = convert_google_v3_export(
            source_csv=source_csv,
            config_path=config_path,
            query_sql=query_sql,
            output_directory=output_directory,
        )
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(str(result.output_directory))


@trace_app.command("verify-google-v3")
def trace_verify_google_v3(
    conversion_directory: Path,
    source_csv: Annotated[Optional[Path], typer.Option("--source-csv")] = None,
) -> None:
    """Verify a Google v3 conversion, optionally reproducing it from source."""

    try:
        result = verify_google_v3_conversion(
            conversion_directory,
            source_csv=source_csv,
        )
    except ClimaDCError as exc:
        _user_error(exc)
    source_state = "verified" if result.source_verified else "not supplied"
    typer.echo(f"VALID: google-v3 conversion rows={result.rows}; source={source_state}")


@app.command()
def report(run_or_pointer: Path) -> None:
    """Resolve and print the static report path without opening a browser."""

    try:
        run_path = resolve_run_path(run_or_pointer)
        report_path = run_path / "report.html"
        if not report_path.is_file():
            raise ConfigurationError(f"Report does not exist: {report_path}")
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(str(report_path))


def _emit_verification(report_value: object, *, json_output: bool) -> bool:
    from climadc.evidence.verify import VerificationReport

    report = cast(VerificationReport, report_value)
    if json_output:
        typer.echo(report.to_json(), nl=False)
    else:
        state = "VALID" if report.valid else "INVALID"
        typer.echo(f"{state}: {report.run_type} artifact schema {report.artifact_schema_version}")
        for check in report.checks:
            if check.status in {"fail", "warning", "skipped"}:
                typer.echo(f"{check.status.upper()} {check.check_id}: {check.message}")
        for limitation in report.limitations:
            typer.echo(f"LIMITATION: {limitation}")
    return report.valid


@app.command("verify-run")
def verify_run_command(
    run_directory: Path,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Verify one published run independently from its directory files."""

    verification = verify_run_directory(run_directory)
    if not _emit_verification(verification, json_output=json_output):
        raise typer.Exit(code=1)


@app.command("verify-suite")
def verify_suite_command(
    suite_directory: Path,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Recursively verify scenario subruns and reconstruct suite aggregates."""

    verification = verify_suite_directory(suite_directory)
    if not _emit_verification(verification, json_output=json_output):
        raise typer.Exit(code=1)


def main() -> None:
    app()
