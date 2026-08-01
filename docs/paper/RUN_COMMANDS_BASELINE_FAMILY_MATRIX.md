# SPARCOF Baseline-Family Controlled Matrix

This package is configured for the environment where PyTorch CUDA works but RAPIDS/cuML is unstable or unavailable.

## Controlled classifier matrix

The controlled matrix uses the classifier families already present in the original manuscript baseline and applies them uniformly to every dataset:

- `cnn1d_gpu` — CNN1D baseline family, implemented in PyTorch CUDA.
- `svm_linear_gpu` — SVM family implemented as a scalable GPU linear SVM with hinge-style loss in PyTorch. This is used for the controlled matrix because RBF-SVM is computationally impractical for repeated large-scale feature-set evaluation.
- `dt_cpu` — Decision Tree baseline family, implemented with scikit-learn CPU. This avoids RAPIDS/cuML dependency and is kept because Decision Tree is one of the original baseline families.

The config sets `include_original_baselines: false` so the controlled matrix does not duplicate or mix the original dataset-specific baselines. The original baseline reproduction can be reported separately.

## Run all datasets

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_baseline_family_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

## Run HIKARI only

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/hikari_only_baseline_family_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

## Run UNSW only

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/unsw_only_baseline_family_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

## Run CICIoT only

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/ciciot_only_baseline_family_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

## Incremental resume

Evaluation results are appended after every completed combination. When `--resume` is used, the script reads the existing `*_evaluation_results.csv` file and skips only completed combinations, not the whole dataset.

Example progress files:

```bash
tail -n 20 outputs/full_revision_hikari_baseline_family/04_model_evaluation/hikari_evaluation_progress.csv
tail -n 5 outputs/full_revision_hikari_baseline_family/04_model_evaluation/hikari_evaluation_results.csv
```

Do not use `--force` unless you want to recompute everything from scratch.
