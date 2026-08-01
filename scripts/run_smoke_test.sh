#!/usr/bin/env bash
set -euo pipefail
sparcof \
  --config configs/paper/full_revision_config.yaml \
  --mode smoke \
  --max-rows-per-dataset 5000 \
  --force
