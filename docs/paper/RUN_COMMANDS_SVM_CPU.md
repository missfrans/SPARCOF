# SPARCOF incremental rerun commands with SVM CPU

SVM is configured as `svm_cpu` for stability. GPU is still used by `cnn1d_gpu`, `rf_gpu`, and `mlp_gpu` when available.

## Full run

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_revision_config.yaml \
  --mode full \
  --gpu-required
```

## Resume

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_revision_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

## Dataset-only runs

UNSW:
```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/unsw_only_svmcpu_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

HIKARI:
```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/hikari_only_svmcpu_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

CICIoT:
```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/ciciot_only_svmcpu_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

## Progress check

```bash
tail -n 20 outputs/full_revision_rerun/04_model_evaluation/hikari_evaluation_progress.csv
tail -n 5 outputs/full_revision_rerun/04_model_evaluation/hikari_evaluation_results.csv
```
