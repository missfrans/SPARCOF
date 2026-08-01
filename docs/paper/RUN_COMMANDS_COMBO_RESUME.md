# SPARCOF Combo-Level Resume Commands

This package uses SVM on CPU for stability and GPU for CNN1D/RF/MLP where available.

## Full resume from UNSW sequence

This starts from the dataset order in the config (UNSW -> HIKARI -> CICIoT), but it does **not** skip a dataset merely because a partial evaluation CSV exists. It reloads reusable artifacts and resumes evaluation from the last unfinished `feature_set_id + classifier` combination.

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_revision_config.yaml \
  --mode full \
  --gpu-required \
  --resume
```

## Fresh full run

```bash
python run_sparcof_full_revision_pipeline.py \
  --config configs/paper/full_revision_config.yaml \
  --mode full \
  --gpu-required
```

## Check progress

```bash
tail -n 20 outputs/full_revision_rerun/04_model_evaluation/unsw_evaluation_progress.csv
tail -n 5 outputs/full_revision_rerun/04_model_evaluation/unsw_evaluation_results.csv
```

## Important

Do not delete `04_model_evaluation/*_evaluation_results.csv` if you want combo-level resume.
Do not use `--force` unless you want to recompute everything.
