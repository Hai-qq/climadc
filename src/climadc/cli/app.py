from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from climadc import __version__
from climadc.benchmark import BenchmarkRunner
from climadc.cli.scaffold import scaffold_study
from climadc.config import StudyConfig
from climadc.errors import ClimaDCError, ConfigurationError
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
app.add_typer(demo_app, name="demo")


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
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
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
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Run comparable replay studies and publish a robustness/Pareto report."""

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


@demo_app.command("robustness-suite")
def demo_robustness_suite(
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("climadc-replay-suite-runs"),
) -> None:
    """Run the packaged four-scenario policy-sensitivity suite offline."""

    try:
        config = ReplaySuiteConfig.from_yaml(packaged_suite_path()).with_output_dir(output_dir)
        result = ReplaySuiteRunner().run(config)
        run_path = ReplaySuiteArtifactWriter().write(result, config.output_dir)
    except ClimaDCError as exc:
        _user_error(exc)
    typer.echo(str(run_path))


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


def main() -> None:
    app()
