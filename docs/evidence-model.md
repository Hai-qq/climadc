# Evidence model and claim boundary

ClimaDC is an offline evaluation and replay framework. Its evidence model links each published
statement through four independently inspectable layers:

1. **Source facts** — immutable bytes or project-owned fixture files, retrieval metadata,
   attribution, quality (`forecast`, `estimated`, or measured observation), and SHA-256 hashes.
2. **Causal computation** — configuration, `issue_time`, `available_at`, `valid_time`, canonical
   inputs, software revision, environment, solver contract, schedules, and interval profiles.
3. **Reconstructed result** — metrics recomputed from published files without rerunning the
   optimizer, plus a distinct optional future re-solve check if one is implemented.
4. **Bounded claim** — a registry entry whose wording, evidence level, hashes, output, command,
   status, and limitations are fixed together.

The first three layers are necessary but do not automatically justify the fourth. A hash proves
identity, not data quality; solver feasibility proves the declared mathematical constraints, not
real-site feasibility; and a counterfactual difference in a synthetic workload is not measured
savings.

## Artifact schema v2

Every v2 run contains `run-manifest.json`, `environment.json`, and `checksums.sha256` in addition to
its run-type files. `run-manifest.json` declares the complete recursive file set, source/config
hashes, Git state, software version, start time, and solver options. `checksums.sha256` uses sorted
POSIX relative paths and covers every file except itself. Backslashes, absolute paths, traversal,
symlinks/reparse points, undeclared files, and missing files are rejected.

`climadc verify-run RUN_DIRECTORY` and `climadc verify-suite SUITE_DIRECTORY` consume only the
published directory. They validate the directory contract, checksums, source and configuration
hashes, finite typed canonical data, exact UTC semantics, schedule/profile keys, job and facility
energy, release/deadline/power/capacity constraints, cost, estimated location-based emissions,
peak, shifted energy, solver records, report payloads, and suite aggregates. They do not rerun the
optimizer and therefore cannot replace corrupted evidence with a new answer.

Version 1 directories remain read-only inputs to the verifier. They are reported as `legacy` and
cannot provide v2 environment, complete checksum, exact artifact-set, or recursive-manifest
assurance.

## Raw-source chain

The network refresh path captures exact Open-Meteo and NESO response bytes before normalization.
Only public HTTPS request URLs and allowlisted response headers are persisted. Authorization,
cookies, credentials, and sensitive query fields are rejected. `raw/retrieval-metadata.json`
records the response status and hash, parser/schema version, transformation parameters, canonical
output hash, license, and attribution. Mapping-based injected transports remain available for
offline tests and are explicitly labelled `injected_mapping`; they must not be represented as
captured provider responses.

The checked-in London fixture predates this raw-response contract. Its canonical files and source
manifest remain hash-bound, but original provider bytes are not retroactively fabricated. A fresh
network refresh is required to create the raw chain.

## Claim registry

[`evidence/claims.yaml`](https://github.com/Hai-qq/climadc/blob/main/evidence/claims.yaml) is the
machine-readable claim registry. Each entry binds the exact wording to its configuration, inputs,
source-code state, named output artifact, and that output's SHA-256 digest. A dirty
working tree is recorded explicitly. Until a maintainer commits the release changes and regenerates
the summaries, its `code_commit` identifies the pinned HEAD and `code_dirty: true` prevents that
revision from being mistaken for a clean release build.

Claims must be removed or marked `deprecated` when their wording no longer matches their evidence.
Changing a metric, fixture, objective, source, or hash requires regenerating the output and updating
the registry in the same change.
