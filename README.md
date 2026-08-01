# SPARCOF Reproducible Research Repository

This repository is a configuration-driven implementation of the SPARCOF
feature-selection and effectiveness–efficiency evaluation workflow. It retains
the three paper datasets as reproducibility configurations, while allowing a
researcher to add another supervised **tabular classification** dataset without
editing Python source code.

> Scope: “any dataset” here means a classification table stored as CSV,
> Parquet, or Excel. Images, raw packet captures, audio, and untransformed time
> series require a domain-specific feature-extraction stage before this pipeline.
> The current implementation is in-memory; the concatenated table and transformed
> arrays must fit the machine's RAM.

## What changed from the supplied archive

- Dataset paths, targets, mappings, dropped identifiers, and leakage columns are
  declared per dataset in YAML.
- A generic loader supports one file or multiple compatible files.
- The UNSW-NB15 special header format remains available through an explicit
  adapter (`loader: unsw_nb15`).
- The transformed feature-name order is guaranteed to match matrix columns.
- Missing categorical values are imputed before encoding.
- Sampling is stratified when feasible.
- Requested and actual execution backends are both recorded.
- Optional selector failures are reported instead of silently substituting a
  different algorithm.
- Resume is guarded by configuration and dataset fingerprints.
- Candidate scoring can use development cross-validation metrics rather than
  final-test metrics (`scoring.metric_source: cross_validation`).
- Packaging, tests, a demo dataset generator, provenance templates, and public
  release documentation are included.

See [the Indonesian audit report](docs/AUDIT_REPORT_ID.md) for consequences and
items that still require an author decision.

## Repository layout

```text
.
├── configs/
│   ├── example.yaml                 # dataset-agnostic example
│   └── paper/                       # archived three-dataset paper configs
├── data/                            # raw data is ignored by Git
├── docs/                            # audit, methods, provenance, release guide
├── results/paper/                   # supplied reference tables
├── scripts/                         # demo and helper commands
├── src/sparcof/                     # installable Python package
├── tests/                           # automated regression tests
├── pyproject.toml
└── run_sparcof_full_revision_pipeline.py  # legacy-compatible entry point
```

## 1. Install

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install only the optional algorithms actually declared in the selected config:

```bash
pip install -e ".[selectors]"  # XGBoost and LightGBM selectors
pip install -e ".[gpu]"        # PyTorch models; CUDA must match the machine
```

RAPIDS/cuML must be installed using the official compatibility instructions for
the machine's CUDA driver. The repository intentionally does not guess a cuML
version.

## 2. Verify the installation without research data

```bash
python scripts/make_demo_dataset.py
sparcof --config configs/example.yaml --validate-only
pytest
sparcof --config configs/example.yaml --mode smoke --force
```

The demo is synthetic and must never be reported as a research result.

## 3. Add a new dataset

Copy `configs/example.yaml`, then edit only the dataset block:

```yaml
datasets:
  - name: my_dataset
    display_name: My Dataset 2026
    loader: generic
    path: data/raw/my_dataset
    format: csv                  # csv, parquet, or excel
    file_glob: "*.csv"           # files are sorted and concatenated
    recursive: false
    target_col: class
    task: auto                   # auto, binary, or multiclass
    read_options:
      low_memory: false
    identifier_columns: [row_id, source_ip]
    leakage_columns: [post_event_status]
    drop_columns: []
    missing_value_tokens: ["?", "NA", "-"]
    benign_labels: [normal, benign]
    shuffle: true
```

Optional label collapsing is explicit and auditable:

```yaml
    target_mapping:
      case_insensitive: true
      mapping:
        benign: Normal
        "0": Normal
      default: Attack
```

If `default` is omitted, every observed label must appear in `mapping`; unknown
labels stop the run instead of being silently recoded.

Then validate before a long experiment:

```bash
sparcof --config configs/my_dataset.yaml --validate-only
sparcof --config configs/my_dataset.yaml --mode smoke --force
sparcof --config configs/my_dataset.yaml --mode focused --force
sparcof --config configs/my_dataset.yaml --mode full --force
```

Use `--resume` only with exactly the same config, mode, sampling cap, and raw
data. The program rejects incompatible reuse.

## 4. Reproduce the article experiment

Place each raw dataset according to the paths declared in
`configs/paper/full_revision_config.yaml`, install the optional GPU/selectors,
and run:

```bash
sparcof \
  --config configs/paper/full_revision_config.yaml \
  --mode full \
  --gpu-required \
  --force
```

The archived paper configs retain the original scoring behavior unless the
authors explicitly add `scoring.metric_source: cross_validation`. Changing that
field creates a new analysis protocol and should not be presented as an exact
reproduction of the prior tables.

## 5. Outputs and traceability

Each run saves dataset audits, target distributions, split indices,
preprocessing metadata, selector rankings, candidate feature sets, fold metrics,
holdout metrics, repeated timings, Pareto fronts, sensitivity analysis, core
coverage/purity, and manuscript-ready tables. It also saves:

- a normalized config snapshot;
- a run fingerprint;
- a dataset fingerprint based on file identity, size, and modification time;
- the actual execution backend (for example `torch_cuda`, `torch_cpu`,
  `cuml_gpu`, or `scikit_learn_cpu_fallback`).

For archival-grade provenance, also record provider-issued byte checksums in
`docs/DATASETS.md`; the fast runtime fingerprint is not a replacement for a
published SHA-256 checksum.

## 6. Publish to GitHub

Follow [the step-by-step Indonesian release guide](docs/GITHUB_RELEASE_GUIDE_ID.md).
Before making the repository public, the authors must replace the citation and
code-availability placeholders, choose a software license, verify dataset terms,
remove secrets/large raw data, rerun tests, and create a versioned release tied
to the manuscript.

## Citation and license

`CITATION.cff.template` is intentionally a template because author names, DOI,
repository URL, and manuscript title were not provided. Rename it to
`CITATION.cff` only after completing it. No software license has been selected on
the authors' behalf; without a license, public viewers do not automatically have
permission to reuse the code.
