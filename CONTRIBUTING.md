# Contributing

Bug reports should include the release/commit, sanitized config, Python and
dependency versions, operating system, requested and actual execution backend,
dataset shape/class counts, command, and complete traceback. Do not attach
restricted datasets, credentials, personal data, or proprietary paths.

Before submitting a change:

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
python scripts/make_demo_dataset.py
sparcof --config configs/example.yaml --mode smoke --force
```

Changes that alter preprocessing, splitting, feature-selection logic, metrics,
candidate scoring, or defaults must document whether paper results need a full
rerun. New optional algorithms must fail transparently when unavailable; they
must never silently substitute a different method under the requested name.
