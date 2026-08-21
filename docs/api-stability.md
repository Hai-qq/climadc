# API and artifact stability

ClimaDC is Alpha software. Public Python imports documented in the API reference and CLI commands
without a deprecation notice are supported within the `0.3.x` line. Internal names beginning with
`_`, report CSS/markup outside the embedded machine payload, and test helpers are not public APIs.

Artifact compatibility is governed by `artifact_schema_version`, not a file count. Writers publish
v2. Verifiers retain read-only v1 recognition and report its missing assurance. A future writer
schema requires a migration guide, explicit verifier dispatch, and tests for both the new schema and
the supported read-only predecessor.

Legacy objective fields remain readable in v0.3 with a `DeprecationWarning`; their arithmetic is not
changed. The deprecated `demo robustness-suite` alias remains during the Alpha migration window.
Removal requires a later changelog entry and at least one prior release carrying the warning.
