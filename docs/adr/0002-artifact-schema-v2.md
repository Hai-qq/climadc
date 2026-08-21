# ADR 0002: Verify evidence independently with artifact schema v2

- Status: Accepted
- Date: 2026-08-21

## Decision

Published runs declare a complete relative artifact set, environment, Git state, solver contract,
source/config hashes, and recursive checksums. Verification reconstructs physical/economic metrics
from directory files and does not depend on the writer's in-memory objects or rerun optimization.

## Consequences

File count is no longer a public contract. V1 remains read-only with lower assurance. Re-solving,
if later implemented, must remain a separate optional check and may not replace reconstruction.
