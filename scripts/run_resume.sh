#!/usr/bin/env bash
set -euo pipefail
sparcof \
  --config configs/paper/full_revision_config.yaml \
  --mode full \
  --gpu-required \
  --resume
python scripts/make_reviewer_response_tables.py \
  --output-dir outputs/full_revision_baseline_family_matrix
