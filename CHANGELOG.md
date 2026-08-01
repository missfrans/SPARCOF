# Changelog

## 1.0.0 — 2026-08-01

- Reorganized the supplied research archive as an installable `src/` package.
- Added a configuration-driven CSV/Parquet/Excel loader and retained the
  UNSW-NB15 header adapter.
- Corrected transformed feature-name/matrix alignment and categorical missing
  value handling.
- Added stratified sampling, config/data resume guards, backend traceability,
  CV-based candidate scoring, confusion-matrix serialization, and environment
  metadata.
- Removed silent XGBoost/LightGBM-to-Random-Forest substitution.
- Added automated tests, CPU smoke data, CI, Docker packaging, provenance and
  release templates, and an Indonesian technical audit.
- Moved supplied result tables to `results/paper/` and removed local path
  metadata from one workbook.

Because feature alignment was corrected, this release requires a complete
article rerun before claiming numerical equivalence with the archived tables.
