# Data directory

Raw research datasets are intentionally excluded from version control. Put data
under `data/raw/` (or another local path) and point each dataset's `path` field
to it. Do not commit files whose license forbids redistribution.

For a self-contained smoke test, run:

```bash
python scripts/make_demo_dataset.py
sparcof --config configs/example.yaml --validate-only
sparcof --config configs/example.yaml --mode smoke
```

Record the provider URL, version/release date, access date, license, and official
checksum for every real dataset in the article and in `docs/DATASETS.md`.
