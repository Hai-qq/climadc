# API reference

The documented modules below are the supported v0.3 Alpha extension and verification surface.

## Evidence verification

::: climadc.evidence.verify
    options:
      show_root_heading: true
      members:
        - VerificationCheck
        - VerificationReport
        - verify_run
        - verify_suite

## Evidence manifests and claims

::: climadc.evidence.manifest
    options:
      show_root_heading: true

::: climadc.evidence.claims
    options:
      show_root_heading: true

## Replay configuration

::: climadc.replay.config
    options:
      show_root_heading: true

## Replay engine

::: climadc.replay.engine
    options:
      show_root_heading: true
      members:
        - ReplayResult
        - ReplayEngine
        - replay_policy_names

## Google ClusterData2019 conversion

::: climadc.adapters.google_clusterdata
    options:
      show_root_heading: true
      members:
        - GoogleV3ConversionConfig
        - GoogleV3ConversionResult
        - GoogleV3ConversionVerification
        - convert_google_v3_export
        - verify_google_v3_conversion
