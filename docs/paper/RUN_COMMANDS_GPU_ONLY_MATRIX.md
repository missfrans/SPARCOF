# SPARCOF GPU-only controlled classifier matrix

This package keeps the incremental/combo resume behavior, but the controlled classifier matrix is GPU-only and RAPIDS-free:

- `svm_linear_gpu` — PyTorch linear SVM with hinge-style loss
- `cnn1d_gpu` — PyTorch 1D-CNN
- `mlp_gpu` — PyTorch MLP

The config sets `include_original_baselines: false` so the run does not mix CPU original baselines with the controlled matrix. Original dataset-specific baselines can still be reported separately using the earlier baseline-aligned scripts/configs.

Run full matrix:

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_revision_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

Run HIKARI only:

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/hikari_only_svmcpu_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

The file names keep `svmcpu` for backward compatibility, but the controlled matrix inside the config is GPU-only.
