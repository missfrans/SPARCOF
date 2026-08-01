# SPARCOF Baseline-Family Controlled Matrix Package

This package is adapted for the current no-RAPIDS environment. The controlled classifier matrix uses the original baseline classifier families consistently across datasets: CNN1D GPU, Linear SVM GPU, and Decision Tree CPU. It keeps incremental per-combination saving and combo-level resume. See `RUN_COMMANDS_BASELINE_FAMILY_MATRIX.md`.

# SPARCOF Full Revision Rerun Pipeline

This repository contains a reproducible, GPU-aware rerun pipeline for the SPARCOF major-revision response. It is designed to regenerate the numerical artifacts requested by the reviewers: dataset audits, split indices, feature-selection rankings, selected feature sets, model-evaluation tables, Pareto-front construction, core-feature coverage/purity, score sensitivity analysis, and repeated timing/memory reports.

## What this pipeline addresses

| Reviewer concern | Pipeline artifact |
|---|---|
| Dataset totals are inconsistent | `outputs/full_revision_rerun/00_dataset_audit/*` |
| Core-feature coverage is mathematically inconsistent | `06_coverage/core_coverage_and_purity.xlsx` and `selected_vs_core_intersection.xlsx` |
| Composite score formula is unclear | `05_pareto/effectiveness_efficiency_scores.xlsx` and `analysis_metadata.json` |
| Pareto-front/champion solution must be reproducible | `05_pareto/pareto_front_results.xlsx` and `champion_solutions.xlsx` |
| Code and split indices must be released | `01_splits/*`, config files, and this repository |
| Classifier is coupled to dataset | `09_tables_for_manuscript/revised_table_classifier_matrix.xlsx` |
| Deployability needs repeated timing and memory footprint | `08_timing_memory/*` |
| Score engineering needs sensitivity analysis | `07_sensitivity/*` |
| Figure/table regeneration | `09_tables_for_manuscript/*` and `figures/` |

## Expected raw-data layout

Place the raw datasets as follows:

```text
SPARCOF_full_rerun_github/
├── data/
│   └── raw/
│       ├── UNSW-NB15/
│       │   └── CSV_Files/
│       │       ├── UNSW-NB15_1.csv
│       │       ├── UNSW-NB15_2.csv
│       │       ├── UNSW-NB15_3.csv
│       │       ├── UNSW-NB15_4.csv
│       │       └── NUSW-NB15_features.csv
│       ├── HIKARI2021/
│       │   ├── *.csv
│       └── CICIoT2023/
│           └── dataset/
│               ├── *.csv
```

## Recommended environment

The pipeline is designed for RAPIDS/cuML and PyTorch on GPU. CPU fallbacks exist for some feature-selection methods, but use `--gpu-required` to make the run fail if no CUDA device is available.

```bash
conda env create -f environment.yml
conda activate sparcof-rerun
```

or install Python dependencies:

```bash
pip install -r requirements.txt
```

For RAPIDS/cuML, use the official RAPIDS conda installation matching your CUDA version.

## Quick smoke test

Use a small sample to validate paths and package installation:

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_revision_config.yaml \
  --max-rows-per-dataset 5000 \
  --mode smoke
```

## Full revision rerun

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_revision_config.yaml \
  --mode full \
  --gpu-required
```

## Focused rerun

This mode reruns a reduced set of selectors and classifiers while still producing coverage, timing, classifier-matrix, Pareto, and sensitivity artifacts.

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_revision_config.yaml \
  --mode focused \
  --gpu-required
```


## Resume interrupted runs

The pipeline supports resumable execution. If a long run stops in the middle, run the same command again with `--resume`:

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_revision_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

Or use the helper script:

```bash
bash scripts/run_resume.sh
```

Resume behavior:

- Completed dataset-level artifacts are reused instead of recomputed.
- Existing train/test split indices are reused, preserving reproducibility.
- Existing feature-selection rankings are reused when available.
- Existing candidate feature sets are reused when available.
- Existing model-evaluation/timing results are reused when available.
- Final consensus, Pareto, coverage, sensitivity, and manuscript tables are rebuilt from the collected artifacts.

If all dataset-level artifacts already exist and you only need to rebuild the final analysis/tables, use:

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_revision_config.yaml \
  --mode full \
  --analysis-only
```

To intentionally recompute everything, use `--force` instead of `--resume`. Do not combine `--resume` and `--force`.

Checkpoint markers are written to:

```text
outputs/full_revision_rerun/_checkpoints/
```

## Main outputs

```text
outputs/full_revision_rerun/
├── 00_dataset_audit/
├── 01_splits/
├── 02_feature_selection/
├── 03_feature_sets/
├── 04_model_evaluation/
├── 05_pareto/
├── 06_coverage/
├── 07_sensitivity/
├── 08_timing_memory/
└── 09_tables_for_manuscript/
```

## Notes for manuscript revision

1. Report both **Core Coverage** and **Core Purity**:
   - `Core Coverage = |Selected ∩ Core| / |Core|`
   - `Core Purity = |Selected ∩ Core| / |Selected|`
2. Do not claim 100% coverage when the selected feature set contains fewer features than the core set. In that case, the value may be 100% purity, not 100% coverage.
3. Use `revised_table_classifier_matrix.xlsx` to respond to classifier-coupling criticism.
4. Use `score_sensitivity_analysis.xlsx` and `champion_stability_across_weights.xlsx` to show whether recommendations are stable under alternative score weights.
5. Include the repository link in the manuscript in a Code Availability statement.

## Original-baseline alignment update

This package runs two complementary evaluation roles:

1. **Original dataset-specific baselines**, aligned with the initial scripts:
   - UNSW-NB15: 1D-CNN with Conv1D(32, kernel=25), Conv1D(64, kernel=25), Dense(1024), BCEWithLogitsLoss, SGD, learning rate 2e-3, 30 CV epochs and 50 final epochs.
   - HIKARI-2021: RBF SVM, C=1.0, gamma=scale, class-weight balanced when supported by the backend.
   - CICIoT-2023: Decision Tree, gini criterion, class-weight balanced.
2. **Controlled classifier matrix**, where the same classifiers (`svm_cpu`, `rf_gpu`, `mlp_gpu`) are evaluated across all datasets to address the reviewer comment about dataset-classifier coupling.

The output column `experiment_role` marks whether a row belongs to `original_baseline` or `controlled_classifier_matrix`.

## Incremental evaluation output and resume behavior

The evaluation stage writes results immediately after each completed combination of dataset + feature set + classifier. During long runs, the following files are updated incrementally:

- `outputs/full_revision_rerun/04_model_evaluation/<dataset>_evaluation_results.csv`
- `outputs/full_revision_rerun/04_model_evaluation/<dataset>_cv_fold_metrics_detailed.csv`
- `outputs/full_revision_rerun/04_model_evaluation/<dataset>_repeated_timing_runs.csv`
- `outputs/full_revision_rerun/04_model_evaluation/<dataset>_evaluation_progress.csv`

If a run stops during model evaluation, rerun with `--resume`. Completed combinations are skipped based on `<dataset>_evaluation_results.csv`, and unfinished combinations continue.


## SVM backend note (WSL/RAPIDS)
This package keeps `svm_cpu` enabled by default using scikit-learn SVC, while still saving results incrementally after each completed combination. If a specific machine encounters a SVM is set to `svm_cpu` by default because RAPIDS/cuML SVM may be unstable during long repeated-CV runs. GPU acceleration is still used for `cnn1d_gpu`, `rf_gpu`, and `mlp_gpu` where available. Completed combinations are preserved and can be resumed with `--resume`.

### Dataset-specific SVM-CPU / RF-CPU configs

For parallel execution across machines, ready-to-use configs are included:

- `configs/paper/unsw_only_svmcpu_config.yaml` → runs only UNSW-NB15 into `outputs/full_revision_unsw_svmcpu`
- `configs/paper/hikari_only_svmcpu_config.yaml` → runs only HIKARI-2021 into `outputs/full_revision_hikari_svmcpu`
- `configs/paper/ciciot_only_svmcpu_config.yaml` → runs only CICIoT-2023 into `outputs/full_revision_ciciot_svmcpu`

Example for Computer B running UNSW only:

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/unsw_only_svmcpu_config.yaml \
  --mode full \
  --gpu-required
```

Resume after interruption:

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/unsw_only_svmcpu_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```


## Current default: incremental resume with SVM CPU

This revision uses `svm_cpu` by default for all SVM evaluations to avoid repeated cuML SVM `KernelCache`/working-set runtime failures observed during long runs. The pipeline still uses GPU-backed classifiers where available for `cnn1d_gpu`, `rf_gpu`, and `mlp_gpu`.

Evaluation results are persisted immediately after each completed dataset-feature-set-classifier combination into `outputs/<run>/04_model_evaluation/<dataset>_evaluation_results.csv`. When rerun with `--resume`, completed combinations are detected from the CSV and skipped, so the run restarts from the last incomplete combination rather than from the beginning of the dataset.

Dataset-specific configs are provided:

- `configs/paper/unsw_only_svmcpu_config.yaml`
- `configs/paper/hikari_only_svmcpu_config.yaml`
- `configs/paper/ciciot_only_svmcpu_config.yaml`

Example:

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/hikari_only_svmcpu_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```


## No-RAPIDS stable configuration
This package defaults to `svm_cpu`, `rf_cpu`, and `mlp_gpu` for the controlled classifier matrix. It does not require a working cuML/RAPIDS installation. CNN1D and MLP still use PyTorch CUDA when available.
