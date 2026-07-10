from typer.testing import CliRunner

import climadc
from climadc.cli.app import app


def test_package_version_and_cli() -> None:
    assert climadc.__version__ == "0.1.0a1"
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "climadc 0.1.0a1"
