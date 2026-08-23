# Contributing to ClimaDC

Thank you for helping improve reproducible climate-aware data-center research.

## Before opening a change

Use a GitHub issue or discussion to describe substantial API, contract, adapter, or benchmark changes before implementation. Keep contributions within ClimaDC's domain-connection scope: contracts, availability semantics, leakage-aware evaluation, adapters, and offline decision studies. General weather models, telemetry collectors, simulators, and online controllers belong in their upstream projects.

Never commit operational data without explicit redistribution rights. Fixtures must be small, synthetic, deterministic, and carry provenance and license metadata. Do not include credentials, API keys, cookies, or private endpoints.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs,lightgbm,xarray]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Required checks

```bash
python -m pytest -m "not network" --cov=climadc --cov-branch --cov-fail-under=85 -q
ruff check .
ruff format --check .
mypy src/climadc
mkdocs build --strict
python -m build
python -m twine check dist/*
```

Tests are expected to run offline by default. Mark a test `network` only when a real network boundary is essential; adapter unit tests should inject a local transport or downloader.

## Pull requests

- Explain the user-visible problem, scope, and non-goals.
- Add a failing regression/behavior test before implementation when code behavior changes.
- Update English documentation and its Chinese mirror when public behavior or claims change.
- Include exact verification commands and results.
- Keep commits focused and preserve unrelated work.

Changes to `main` are merged through a pull request. The repository ruleset requires the maintained
CI matrix, CodeQL analysis, dependency review, documentation build, and SBOM generation to pass
against the latest `main`, and requires review conversations to be resolved. Branch deletion and
force pushes are blocked. The sole-maintainer emergency bypass is limited to pull requests and
should be explained in the affected pull request if it is ever used.

By submitting a contribution, you agree that it is licensed under Apache-2.0.
