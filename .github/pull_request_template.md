## Problem and scope

Describe the concrete problem, the smallest implemented change, and explicit non-goals.

## Causality and data provenance

- [ ] I considered `issue_time`, `available_at`, and `valid_time` semantics.
- [ ] New fixtures are small, deterministic, synthetic, and include provenance/license metadata.
- [ ] No credentials, private data, generated models, or non-redistributable upstream data are included.

## Verification

List exact commands and results. For behavior changes, include RED and GREEN evidence.

- [ ] Offline tests pass.
- [ ] Ruff, formatting, and mypy pass.
- [ ] Documentation and its Chinese mirror are updated when public behavior changes.
- [ ] Claims stay within offline research evidence; no production savings or control claims are introduced.
