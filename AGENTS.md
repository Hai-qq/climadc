# ClimaDC agent guide

## Product boundary

ClimaDC is an offline evaluation and replay framework for climate-aware data-center decisions with
causal time semantics and auditable evidence chains. Do not expand it into an online controller,
Kubernetes scheduler, RL framework, digital twin, monitoring dashboard, or model repository.

## Sources of truth

- `pyproject.toml` defines supported Python and dependency bounds.
- `docs/adr/` records accepted causal, artifact, objective, and evidence decisions.
- `docs/evidence-model.md` defines artifact verification and claim boundaries.
- `evidence/claims.yaml` is the machine-readable registry for quantitative public claims.
- `ROADMAP.md` owns E2/E3 data gates; do not fabricate results for `DATA_REQUIRED` work.

## Development and verification

Install development dependencies with `python -m pip install -e ".[dev]"`. Keep offline tests
network-free and use injected transports for adapter unit tests. Before handoff, run the applicable
subset and, for release-facing changes, all of:

```bash
python -m pytest -m "not network" --cov=climadc --cov-branch --cov-fail-under=85 -q
ruff check .
ruff format --check .
mypy src/climadc
python scripts/check_docs_links.py
mkdocs build --strict
python -m build
python -m twine check dist/*
```

## Change rules

- Preserve `issue_time`, `available_at`, and `valid_time`; reject future information.
- Treat artifact membership as a versioned manifest contract, never a permanent file count.
- Keep forecast, estimated settlement, and measured observation distinct.
- Link every public quantitative claim to its registry entry and reproducible evidence.
- Preserve v1 read compatibility and document public API, CLI, schema, or field migrations.
- Update English documentation and its Chinese mirror with public behavior changes.
- Keep large run bundles out of Git; commit only compact generated references with provenance.
- Current repository evidence is E0/E1 only. E2/E3 claims require the gates in `ROADMAP.md`.
