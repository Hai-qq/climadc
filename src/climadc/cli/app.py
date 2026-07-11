from pathlib import Path

import typer

from climadc import __version__
from climadc.benchmark import BenchmarkRunner
from climadc.cli.scaffold import scaffold_study
from climadc.config import StudyConfig
from climadc.errors import ClimaDCError, ConfigurationError
from climadc.reporting import ArtifactWriter, resolve_run_path

app = typer.Typer(no_args_is_help=True)


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
