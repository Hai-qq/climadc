# WeatherDC small compact reference

This directory contains the compact E0 summary behind claim
`E0-WEATHERDC-SANITY-001`. The checked-in JSON is byte-bound by the claim registry; a fresh run is
compared semantically because the synthetic OLS path can differ at the last floating-point digits
across LAPACK implementations.

```bash
python benchmarks/reference/weatherdc_small/reproduce.py
python benchmarks/reference/weatherdc_small/reproduce.py --check
```

The declared absolute `1e-12` tolerance remains below half the least precise decimal place printed
in the public claim. It does not turn this project-generated fixture into operational accuracy
evidence.
