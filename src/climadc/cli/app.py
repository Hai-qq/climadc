import typer

from climadc import __version__

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


def main() -> None:
    app()
