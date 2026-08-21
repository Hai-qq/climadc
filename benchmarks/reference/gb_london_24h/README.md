# GB London 24-hour compact reference

This directory contains a code-generated compact summary of the packaged
`gb-london-carbon-shift-24h` replay. It is an **E1 single-day mechanism demonstration**:
external national grid and gridded-weather signals are combined with a project-generated
workload, UTC tariff scenario, and facility model. It is not an operational validation,
production savings claim, or estimate of marginal avoided emissions.

Regenerate from an installed checkout:

```bash
python benchmarks/reference/gb_london_24h/reproduce.py
python benchmarks/reference/gb_london_24h/reproduce.py --check
```

`summary.json` records the complete absolute and ASAP-relative policy metrics, units, objective
contract, configuration hash, input hashes, package version, and comparison tolerance.
`summary.csv` provides the same policy rows for simple review. The engine first fixes each policy's
primary aggregate slot power, then deterministically assigns that power to jobs using the documented
ASAP order; this makes job-level schedules and `shifted_energy_kwh` stable without changing cost,
emissions, energy, or peak results. The absolute `1e-8` comparison tolerance is smaller than the
replay's `1e-7 kWh` feasibility tolerance and only accommodates cross-platform floating-point solver
noise; it is not used to alter or select results.
