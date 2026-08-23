# Changelog

All notable changes to this project are documented in this file.

The project follows [Semantic Versioning](https://semver.org/). Alpha APIs may change before a stable release; changes will be recorded here.

## [Unreleased]

### Changed

- Citation, package, maintainer, and NOTICE metadata now use the maintainer-supplied name Gao Qing;
  the GitHub account remains `@Hai-qq`, while ORCID and affiliation remain unspecified.
- Validated dependency ranges now include Hatchling 1.32, PyArrow 24-25, Twine 7, and
  mkdocstrings 1.x; Dependabot excludes the minimum-compatibility constraint file so automated
  updates cannot silently turn the minimum-dependency job into a latest-dependency job. Twine 7
  is required for the Core Metadata 2.5 artifacts emitted by the expanded Hatchling range.

## [0.3.0-alpha.1] - 2026-08-21

### Added

- Versioned artifact schema v2 with run/environment manifests, recursive SHA-256 checksums, and
  independent `verify-run` / `verify-suite` commands.
- Raw Open-Meteo and NESO response capture with allowlisted retrieval metadata and explicit
  raw-to-canonical transformation hashes.
- Versioned monetized, epsilon-constraint, and complete fixed-point Pareto objective contracts.
- Evidence levels, machine-readable claims, and a reproducible compact London reference summary.

### Changed

- The built-in four-scenario suite is explicitly sensitivity analysis, with the prior robustness
  command retained only as a deprecated compatibility alias.
- Hatchling is constrained below 1.32 so release artifacts remain on Core Metadata 2.4 until the
  supported Twine line accepts Hatchling 1.32's Metadata 2.5 default.
- The dependency-audit job upgrades its virtual-environment bootstrap tooling before scanning, so
  stale bundled `setuptools` is not mistaken for a project runtime dependency failure.
- Jinja2, PyArrow, scikit-learn, and pytest minimums are raised to the first compatible versions
  that clear the known advisories detected by the release dependency audit.
- Minimum compatibility uses Pint 0.24.4 (the first 0.24 release compatible with FlexParser 0.4)
  and non-yanked SciPy 1.11.4 rather than broken or withdrawn initial releases.
- The compact golden check now applies its declared absolute tolerance semantically across JSON
  and CSV, while continuing to reject schema, row-order, and non-numeric changes.
- Equivalent HiGHS solutions now keep the primary aggregate slot power fixed and use the documented
  ASAP order to select one job allocation, making job-level schedules and `shifted_energy_kwh`
  reproducible across platforms without changing energy, cost, emissions, or peak results.
- Repository configurations, compact golden outputs, and source files referenced by byte-level
  claim hashes now use enforced LF checkout semantics on Windows, macOS, and Linux.
- The WeatherDC E0 claim now byte-binds a checked-in compact summary and verifies fresh OLS results
  under a declared `1e-12` absolute tolerance, instead of requiring architecture-specific raw JSON
  bytes to be identical.
- The Typer floor is raised to the vendored-Click generation, avoiding unsupported combinations
  between historical Typer releases and current Click while retaining Python 3.10 support.
- National-average carbon settlement is named estimated location-based emissions throughout the
  runtime artifact contract.
- The London tariff is explicitly a declared UTC scenario; the fixture remains a single-day
  mechanism demonstration rather than a savings or production-validation claim.

### Deprecated

- Dimensionally unscaled `cost_weight` / `carbon_weight` objectives remain readable but emit a
  migration warning and are marked as legacy in reports.

## [0.2.0-alpha.1] - 2026-08-07

### Added

- v0.2 engineering-replay design, delivery phases, and acceptance gates in English and Chinese.
- `GridSignalFrame` for causally distinct forecast and realized carbon-intensity or energy-price rows.
- `FlexibleWorkloadFrame` for unit-aware, deadline-constrained preemptible batch jobs.
- Local CSV/Parquet readers for the new grid-signal and flexible-workload contracts.
- A single-site `ReplayEngine` with ASAP, peak, price, carbon, joint, and realized-value Oracle
  policies backed by constrained linear optimization.
- A replaceable facility-energy protocol, bounded temperature-sensitive PUE reference model, and
  realized settlement for energy, emissions, cost, demand charge, peak, SLA, shifted energy, and
  Oracle regret.
- Historical Open-Meteo and national NESO Carbon Intensity adapters with explicit provenance and
  scenario-timing boundaries.
- A packaged, fully offline 24-hour Great Britain reference replay with deterministic tariff and
  workload inputs plus SHA-256-bound source manifests.
- `climadc replay`, `climadc demo carbon-shift`, and an overwrite-safe optional network refresh
  command.
- Atomic 12-artifact replay publication and a self-contained comparative HTML report.
- Offline reference integration tests and a scheduled provider-contract workflow.
- A generic Prometheus range adapter plus Kepler power-gauge queries with explicit retrieval-time
  availability and resource identity.
- A Carbon Aware SDK-compatible forecast/settlement adapter with online GET and offline-payload
  paths.
- A dependency-free SustainDC evaluation-export adapter for simulator telemetry and carbon signals.
- Optional upper-quantile joint replay using complete, causally available temperature, energy-price,
  and carbon-intensity quantile scenarios, with explicit failure instead of point-forecast fallback.
- Receding-horizon replay that re-solves at each origin, commits a configured step, preserves
  per-policy remaining job energy, and publishes decision-level audit records through the existing
  12-artifact contract and `climadc replay` CLI.
- Signed rolling Oracle deltas that preserve causal future-job arrivals instead of incorrectly
  enforcing a global perfect-foresight lower bound.
- Post-hoc upper-quantile diagnostics over committed slots, including inclusive marginal coverage,
  95% Wilson intervals, coverage gaps, exceedance magnitudes, and pinball loss in JSON and HTML
  replay reports.
- Equal-weight replay suites that run multiple independently verified studies, enforce comparable
  execution grids and currencies, and publish feasibility, mean/worst baseline deltas, and a
  three-objective Pareto frontier.
- Atomic eight-entry suite publication with machine-readable scenario rows and complete nested
  12-artifact evidence runs, plus `climadc replay-suite` and an offline four-scenario demo.

### Fixed

- Replay artifact validation now preserves nullable numeric semantics across Parquet round trips,
  so risk-enabled inputs with both point and quantile rows can publish the full artifact set.
- Hash-bound reference fixtures now force LF checkout semantics so manifest validation remains
  byte-identical on Windows, macOS, and Linux.

### Changed

- Contract nullability is now declared per frame instead of through one shared nullable-column list.
- Pint recognizes explicit CO2e labels and supported scenario currencies for dimensional validation.
- NumPy is constrained below 2.4 to keep the declared Python 3.10 mypy target and Pandas 2.x time
  operations on one tested compatibility boundary.
- Ruff's traditional base rule set is explicit so quality checks do not drift with tool defaults.

## [0.1.0-alpha.1] - 2026-07-11

### Added

- Canonical climate forecast, DC telemetry, workload, prediction, and dataset-card contracts.
- `LeakageGuard`, decision-time views, and blocked/rolling-origin temporal splits.
- Forecasting baselines, optional LightGBM, split conformal calibration, evaluation metrics, and weather slices.
- Energy-conserving shadow scheduling and decision comparison.
- Local CSV/Parquet, optional Xarray, Open-Meteo, and WeatherDC adapters.
- Auditable benchmark engine, CLI, deterministic eight-artifact publication, and static HTML reports.
- Fully offline synthetic Quickstart and WeatherDC reference benchmark.
- Verified conversion-only path for upstream WeatherDC/HII observations and meter streams.

[Unreleased]: https://github.com/Hai-qq/climadc/compare/v0.3.0-alpha.1...HEAD
[0.3.0-alpha.1]: https://github.com/Hai-qq/climadc/compare/v0.2.0-alpha.1...v0.3.0-alpha.1
[0.2.0-alpha.1]: https://github.com/Hai-qq/climadc/releases/tag/v0.2.0-alpha.1
[0.1.0-alpha.1]: https://github.com/Hai-qq/climadc/releases/tag/v0.1.0-alpha.1
