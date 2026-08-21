# ADR 0003: Make objective dimensions explicit

- Status: Accepted
- Date: 2026-08-21

## Decision

New configs use a versioned monetized objective, epsilon constraints, or a complete fixed-point
Pareto analysis. Carbon price converts kgCO2e to tonnes before adding it to currency cost. Legacy
numeric weights preserve their exact arithmetic, emit a warning, and are labelled unscaled.
After each primary solve, a second linear program holds aggregate power in every slot fixed and
assigns that power to jobs by the ASAP priority, deadline, release-time, and job-ID order. This
removes solver-dependent choices among job allocations without changing the primary objective or
aggregate schedule.

## Consequences

Existing files do not silently change primary objectives or aggregate slot power. Exact job-level
allocations, and therefore `shifted_energy_kwh`, can change when an older solver-selected allocation
was one of several equivalent choices. Reports may interpret monetized regret as currency but must
not make that interpretation for legacy scores. Pareto output includes every declared point.
