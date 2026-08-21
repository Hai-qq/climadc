# ADR 0003: Make objective dimensions explicit

- Status: Accepted
- Date: 2026-08-21

## Decision

New configs use a versioned monetized objective, epsilon constraints, or a complete fixed-point
Pareto analysis. Carbon price converts kgCO2e to tonnes before adding it to currency cost. Legacy
numeric weights preserve their exact arithmetic, emit a warning, and are labelled unscaled.

## Consequences

Existing files do not silently change schedules. Reports may interpret monetized regret as currency
but must not make that interpretation for legacy scores. Pareto output includes every declared point.
