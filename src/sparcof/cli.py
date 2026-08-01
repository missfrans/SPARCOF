#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import find_project_root, normalize_config, run_fingerprint, validate_config
from .consensus import build_consensus_core, export_champion_feature_sets
from .coverage import compute_core_coverage
from .data import audit_dataset, load_dataset, write_combined_audit
from .evaluation import evaluate_candidate_feature_sets
from .feature_selection import expand_method_list, generate_feature_subsets, run_feature_selection_suite
from .preprocess import TabularPreprocessor
from .scoring import apply_scores, compute_pareto_and_champions, retention_and_cost_reduction
from .utils import (
    collect_environment_metadata,
    ensure_dir,
    load_yaml,
    read_json,
    require_gpu_or_warn,
    save_json,
    set_global_seed,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Reproducible SPARCOF feature-selection pipeline")
    p.add_argument("--config", default="configs/example.yaml")
    p.add_argument("--mode", choices=["smoke", "focused", "full"], default="focused")
    p.add_argument("--max-rows-per-dataset", type=int, default=None, help="Optional smoke/sampling cap for quick checks.")
    p.add_argument("--gpu-required", action="store_true", help="Fail if CUDA GPU is unavailable.")
    p.add_argument("--resume", action="store_true", help="Skip completed dataset-level stages and continue from available artifacts.")
    p.add_argument("--force", action="store_true", help="Recompute everything even if prior artifacts are available.")
    p.add_argument("--analysis-only", action="store_true", help="Rebuild consensus/Pareto/coverage/manuscript tables from existing artifacts only.")
    p.add_argument("--validate-only", action="store_true", help="Validate configuration and input datasets, then exit.")
    return p.parse_args(argv)


def select_classifiers_for_dataset(config: Dict, mode: str, dataset_name: str) -> List[str]:
    """Return classifiers for one dataset.

    The list intentionally includes two roles:
    1. the original dataset-specific baseline, aligned with the user's initial scripts; and
    2. the controlled classifier matrix, where the same classifiers are tested across datasets.

    Duplicates are removed while preserving order.
    """
    mev = config.get("model_evaluation", {})
    if mode == "smoke":
        return mev.get("smoke_classifiers", ["rf_gpu"])
    include_original = bool(mev.get("include_original_baselines", True))
    original = mev.get("original_baselines", {}).get(dataset_name) if include_original else None
    controlled = mev.get("controlled_classifiers", ["svm_linear_gpu", "cnn1d_gpu", "mlp_gpu"])
    ordered = []
    for clf in ([original] if original else []) + list(controlled):
        if clf and clf not in ordered:
            ordered.append(clf)
    return ordered

def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if _nonempty(path):
        return pd.read_csv(path)
    return None


def _write_checkpoint(out_root: Path, name: str, payload: Dict) -> None:
    ckpt = ensure_dir(out_root / "_checkpoints")
    payload = dict(payload)
    payload["checkpoint_name"] = name
    payload["timestamp_unix"] = time.time()
    save_json(payload, ckpt / f"{name}.done.json")


def save_split_indices(dataset_name: str, train_idx, test_idx, y_trainval, n_splits: int, seed: int, output_dir: Path):
    out = ensure_dir(output_dir)
    pd.DataFrame({"row_index": train_idx}).to_csv(out / f"{dataset_name}_train_cv_indices.csv", index=False)
    pd.DataFrame({"row_index": test_idx}).to_csv(out / f"{dataset_name}_final_test_indices.csv", index=False)
    folds = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_X = np.zeros(len(train_idx))
    for fold, (tr, va) in enumerate(skf.split(dummy_X, y_trainval), start=1):
        for i in tr:
            folds.append({"fold": fold, "split": "train", "position_in_train_cv": int(i), "original_row_index": int(train_idx[i])})
        for i in va:
            folds.append({"fold": fold, "split": "validation", "position_in_train_cv": int(i), "original_row_index": int(train_idx[i])})
    pd.DataFrame(folds).to_csv(out / f"{dataset_name}_cv_fold_indices.csv", index=False)


def load_or_create_split_indices(dataset_name: str, df: pd.DataFrame, target_col: str, test_size: float, seed: int, n_splits: int, output_dir: Path, resume: bool, force: bool) -> Tuple[np.ndarray, np.ndarray]:
    train_path = output_dir / f"{dataset_name}_train_cv_indices.csv"
    test_path = output_dir / f"{dataset_name}_final_test_indices.csv"
    if resume and not force and _nonempty(train_path) and _nonempty(test_path):
        print(f"[RESUME] Reusing saved split indices for {dataset_name}.")
        train_idx = pd.read_csv(train_path)["row_index"].to_numpy(dtype=int)
        test_idx = pd.read_csv(test_path)["row_index"].to_numpy(dtype=int)
        return train_idx, test_idx
    original_indices = np.arange(len(df))
    y_for_split = df[target_col].astype(str).values
    train_idx, test_idx = train_test_split(
        original_indices,
        test_size=test_size,
        random_state=seed,
        stratify=y_for_split,
    )
    # y_trainval is required for CV fold index export.
    y_trainval = df.iloc[train_idx][target_col].astype(str).values
    save_split_indices(dataset_name, train_idx, test_idx, y_trainval, n_splits, seed, output_dir)
    return train_idx, test_idx


def load_existing_dataset_artifacts(ds_name: str, dirs: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = _read_csv_if_exists(dirs["audit"] / f"{ds_name}_dataset_audit_summary.csv")
    labels = _read_csv_if_exists(dirs["audit"] / f"{ds_name}_label_counts.csv")
    rankings = _read_csv_if_exists(dirs["fs"] / f"{ds_name}_feature_selection_rankings.csv")
    candidates = _read_csv_if_exists(dirs["feature_sets"] / f"{ds_name}_candidate_feature_sets.csv")
    evals = _read_csv_if_exists(dirs["eval"] / f"{ds_name}_evaluation_results.csv")
    folds = _read_csv_if_exists(dirs["eval"] / f"{ds_name}_cv_fold_metrics_detailed.csv")
    timings = _read_csv_if_exists(dirs["eval"] / f"{ds_name}_repeated_timing_runs.csv")
    return audit, labels, rankings, candidates, evals, folds, timings


def _assert_resume_compatible(out_root: Path, fingerprint: str, resume: bool, force: bool) -> None:
    snapshot = out_root / "run_config_snapshot.json"
    if not resume or force or not snapshot.exists():
        return
    previous = read_json(snapshot)
    previous_fingerprint = previous.get("run_fingerprint")
    if previous_fingerprint and previous_fingerprint != fingerprint:
        raise ValueError(
            "--resume cannot reuse this output directory because the configuration, mode, or sampling cap changed. "
            "Use the original settings, choose a new project.output_dir, or use --force intentionally."
        )


def _assert_dataset_unchanged(bundle, metadata_path: Path, resume: bool, force: bool) -> None:
    if not resume or force or not metadata_path.exists():
        return
    previous = read_json(metadata_path)
    old = previous.get("dataset_fingerprint")
    new = bundle.loader_metadata.get("dataset_fingerprint")
    if old and new and old != new:
        raise ValueError(
            f"Dataset {bundle.name!r} changed since the saved artifacts were produced. "
            "Use a new output directory or rerun with --force."
        )


def _validate_bundle_for_splits(bundle, n_splits: int) -> None:
    class_counts = bundle.df[bundle.target_col].value_counts()
    if class_counts.empty:
        raise ValueError(f"Dataset {bundle.name!r} contains no target labels.")
    if class_counts.min() < n_splits:
        raise ValueError(
            f"Dataset {bundle.name!r} has a class with only {int(class_counts.min())} rows, "
            f"which is insufficient for {n_splits}-fold stratified CV. Reduce project.n_splits_cv."
        )


def _apply_row_cap(bundle, max_rows: int | None, seed: int) -> None:
    if not max_rows or len(bundle.df) <= max_rows:
        return
    _, sampled = train_test_split(
        bundle.df,
        test_size=max_rows,
        stratify=bundle.df[bundle.target_col].astype(str),
        random_state=seed,
    )
    bundle.df = sampled.reset_index(drop=True)
    bundle.loader_metadata["max_rows_per_dataset_applied"] = max_rows


def main(argv=None):
    args = parse_args(argv)
    if args.resume and args.force:
        raise ValueError("Use either --resume or --force, not both.")
    if args.analysis_only and args.force:
        raise ValueError("--analysis-only cannot be combined with --force.")

    config_path = Path(args.config).expanduser().resolve()
    config = normalize_config(load_yaml(config_path))
    validate_config(config)
    project_root = find_project_root(config_path)
    seed = int(config["project"].get("seed", 42))
    set_global_seed(seed)
    n_splits = int(config["project"].get("n_splits_cv", 10))
    if args.validate_only:
        for ds_cfg in config["datasets"]:
            bundle = load_dataset(ds_cfg, project_root, seed=seed)
            _apply_row_cap(bundle, args.max_rows_per_dataset, seed)
            _validate_bundle_for_splits(bundle, n_splits)
            print(
                f"[VALID] {bundle.name}: rows={len(bundle.df)}, features={bundle.df.shape[1] - 1}, "
                f"classes={bundle.df[bundle.target_col].nunique()}, "
                f"fingerprint={bundle.loader_metadata['dataset_fingerprint'][:12]}"
            )
        print("Configuration and dataset validation completed successfully.")
        return 0

    require_gpu_or_warn(args.gpu_required or bool(config["project"].get("gpu_required", False)))

    configured_output = Path(config["project"].get("output_dir", "outputs/run"))
    out_root = ensure_dir(configured_output if configured_output.is_absolute() else project_root / configured_output)
    fingerprint = run_fingerprint(config, mode=args.mode, max_rows=args.max_rows_per_dataset)
    if (out_root / "run_config_snapshot.json").exists() and not (args.resume or args.force or args.analysis_only):
        raise FileExistsError(
            f"Output directory already contains a run: {out_root}. "
            "Use --resume for an identical interrupted run, --force to recompute intentionally, "
            "or change project.output_dir."
        )
    _assert_resume_compatible(out_root, fingerprint, args.resume or args.analysis_only, args.force)
    dirs = {
        "audit": ensure_dir(out_root / "00_dataset_audit"),
        "splits": ensure_dir(out_root / "01_splits"),
        "fs": ensure_dir(out_root / "02_feature_selection"),
        "feature_sets": ensure_dir(out_root / "03_feature_sets"),
        "eval": ensure_dir(out_root / "04_model_evaluation"),
        "pareto": ensure_dir(out_root / "05_pareto"),
        "coverage": ensure_dir(out_root / "06_coverage"),
        "sensitivity": ensure_dir(out_root / "07_sensitivity"),
        "timing": ensure_dir(out_root / "08_timing_memory"),
        "tables": ensure_dir(out_root / "09_tables_for_manuscript"),
    }
    save_json({
        "mode": args.mode,
        "resume": args.resume,
        "force": args.force,
        "analysis_only": args.analysis_only,
        "validate_only": args.validate_only,
        "project_root": str(project_root),
        "run_fingerprint": fingerprint,
        "config": config,
    }, out_root / "run_config_snapshot.json")
    save_json(collect_environment_metadata(project_root), out_root / "environment_metadata.json")

    methods = expand_method_list(config, args.mode)
    if args.mode == "smoke":
        criteria = ["q25"]
    else:
        criteria = config["feature_selection"].get("criteria", ["elbow", "q25", "q50", "q75"])
    max_rows_for_fs = int(config["feature_selection"].get("max_rows_for_fs", 100000))
    min_features = int(config["feature_selection"].get("min_features", 3))
    test_size = float(config["project"].get("test_size", 0.2))

    all_audits, all_labels = [], []
    all_rankings, all_candidates = [], []
    all_evals, all_folds, all_timings = [], [], []

    for ds_cfg in config["datasets"]:
        ds_name = ds_cfg["name"]
        print(f"\n========== DATASET: {ds_name} ==========")

        audit_old, labels_old, rankings_old, candidates_old, evals_old, folds_old, timings_old = load_existing_dataset_artifacts(ds_name, dirs)
        # IMPORTANT COMBO-LEVEL RESUME POLICY
        # ------------------------------------
        # Older versions skipped a whole dataset when any evaluation CSV existed.
        # That was inconvenient for interrupted runs: partial evaluation files made
        # UNSW/HIKARI appear "complete" even when many candidate/classifier
        # combinations had not yet been evaluated.
        #
        # This version never skips a dataset-level evaluation just because
        # <dataset>_evaluation_results.csv exists. With --resume, it reloads the
        # reusable artifacts (audit/splits/FS/candidate sets), rebuilds the arrays,
        # then calls evaluate_candidate_feature_sets(..., resume=True). That
        # function reads the completed combinations from the CSV and skips only
        # those exact combinations. In practice this means a full command starts
        # from UNSW, then continues from the last unfinished combination.
        if args.analysis_only:
            dataset_complete = all(df is not None and not df.empty for df in [audit_old, labels_old, rankings_old, candidates_old, evals_old])
            if dataset_complete:
                print(f"[ANALYSIS-ONLY] Loading completed artifacts for {ds_name}; no model rerun.")
                all_audits.append(audit_old)
                all_labels.append(labels_old)
                all_rankings.append(rankings_old)
                all_candidates.append(candidates_old)
                all_evals.append(evals_old)
                if folds_old is not None:
                    all_folds.append(folds_old)
                if timings_old is not None:
                    all_timings.append(timings_old)
                continue
            raise FileNotFoundError(f"Missing completed artifacts for {ds_name}; cannot run --analysis-only.")

        if args.resume and not args.force and evals_old is not None and not evals_old.empty:
            print(f"[RESUME] {ds_name}: existing evaluation CSV found; dataset will NOT be skipped. Evaluation resumes per completed combination.")

        bundle = load_dataset(ds_cfg, project_root, seed=seed)
        loader_metadata_path = dirs["audit"] / f"{ds_name}_loader_metadata.json"
        _assert_dataset_unchanged(bundle, loader_metadata_path, args.resume, args.force)
        _apply_row_cap(bundle, args.max_rows_per_dataset, seed)
        _validate_bundle_for_splits(bundle, n_splits)

        if args.resume and not args.force and audit_old is not None and labels_old is not None:
            print(f"[RESUME] Reusing dataset audit for {ds_name}.")
            audit, labels = audit_old, labels_old
        else:
            audit, labels = audit_dataset(bundle, dirs["audit"])
            save_json(bundle.loader_metadata, loader_metadata_path)
            _write_checkpoint(out_root, f"{ds_name}__audit", {"dataset": ds_name})
        all_audits.append(audit)
        all_labels.append(labels)

        train_idx, test_idx = load_or_create_split_indices(
            ds_name, bundle.df, bundle.target_col, test_size, seed, n_splits, dirs["splits"], args.resume, args.force
        )
        df_trainval = bundle.df.iloc[train_idx].copy().reset_index(drop=True)
        df_test = bundle.df.iloc[test_idx].copy().reset_index(drop=True)

        pre = TabularPreprocessor(ds_name, bundle.target_col, dataset_config=ds_cfg)
        X_trainval, y_trainval, artifacts = pre.fit_transform(df_trainval)
        X_test, y_test = pre.transform(df_test)
        feature_names = artifacts.feature_names
        save_json({
            "dataset": ds_name,
            "feature_names": feature_names,
            "numeric_features": artifacts.numeric_features,
            "categorical_features": artifacts.categorical_features,
            "label_mapping": artifacts.label_mapping,
        }, dirs["audit"] / f"{ds_name}_preprocessing_artifacts.json")

        if args.resume and not args.force and rankings_old is not None and not rankings_old.empty:
            print(f"[RESUME] Reusing feature-selection rankings for {ds_name}.")
            rankings = rankings_old
        else:
            rankings = run_feature_selection_suite(
                dataset_name=ds_name,
                X=X_trainval,
                y=y_trainval,
                feature_names=feature_names,
                methods=methods,
                max_rows=max_rows_for_fs,
                seed=seed,
                output_dir=dirs["fs"],
            )
            _write_checkpoint(out_root, f"{ds_name}__feature_selection", {"dataset": ds_name, "rows": len(rankings)})
        all_rankings.append(rankings)

        if args.resume and not args.force and candidates_old is not None and not candidates_old.empty:
            print(f"[RESUME] Reusing candidate feature sets for {ds_name}.")
            candidates = candidates_old
        else:
            candidates = generate_feature_subsets(rankings, criteria=criteria, min_features=min_features, output_dir=dirs["feature_sets"])
            # Store a dataset-specific copy. The generic generator writes all_candidate_feature_sets.csv;
            # this file lets --resume skip per-dataset recomputation safely.
            candidates.to_csv(dirs["feature_sets"] / f"{ds_name}_candidate_feature_sets.csv", index=False)
            _write_checkpoint(out_root, f"{ds_name}__candidate_feature_sets", {"dataset": ds_name, "rows": len(candidates)})
        all_candidates.append(candidates)

        classifiers = select_classifiers_for_dataset(config, args.mode, ds_name)
        print(f"[CONFIG] Classifiers for {ds_name}: {classifiers}")

        # Always call evaluation. The evaluation function itself is incremental:
        # it loads completed feature_set_id/classifier pairs from CSV and skips
        # only those pairs. This avoids whole-dataset skipping and lets a full
        # command resume from the last completed combination.
        eval_results, fold_results, timing_results = evaluate_candidate_feature_sets(
            dataset_name=ds_name,
            X_trainval=X_trainval,
            y_trainval=y_trainval,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            candidates=candidates,
            classifiers=classifiers,
            config=config,
            seed=seed,
            output_dir=dirs["eval"],
            resume=args.resume,
            force=args.force,
        )
        _write_checkpoint(out_root, f"{ds_name}__model_evaluation", {"dataset": ds_name, "rows": len(eval_results)})
        all_evals.append(eval_results)
        if fold_results is not None and not fold_results.empty:
            all_folds.append(fold_results)
        if timing_results is not None and not timing_results.empty:
            all_timings.append(timing_results)

    print("\n========== COMBINE DATASET-LEVEL ARTIFACTS ==========")
    write_combined_audit(all_audits, all_labels, dirs["audit"])
    rankings_all = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    candidates_all = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    evals_all = pd.concat(all_evals, ignore_index=True) if all_evals else pd.DataFrame()
    folds_all = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    timings_all = pd.concat(all_timings, ignore_index=True) if all_timings else pd.DataFrame()

    rankings_all.to_excel(dirs["fs"] / "all_feature_selection_master_results.xlsx", index=False)
    candidates_all.to_excel(dirs["feature_sets"] / "all_candidate_feature_sets.xlsx", index=False)
    evals_all.to_excel(dirs["eval"] / "all_evaluation_results.xlsx", index=False)
    folds_all.to_excel(dirs["eval"] / "all_cv_fold_metrics_detailed.xlsx", index=False)
    timings_all.to_excel(dirs["timing"] / "repeated_timing_results.xlsx", index=False)
    _write_checkpoint(out_root, "combined_dataset_artifacts", {
        "rankings_rows": len(rankings_all), "candidate_rows": len(candidates_all), "evaluation_rows": len(evals_all)
    })

    print("\n========== CONSENSUS CORE FEATURES ==========")
    _, consensus_tiers = build_consensus_core(rankings_all, dirs["feature_sets"], random_state=seed)
    _write_checkpoint(out_root, "consensus_core_features", {"rows": len(consensus_tiers)})

    print("\n========== PARETO + SENSITIVITY ==========")
    scored = apply_scores(
        evals_all,
        config["scoring"]["scenarios"],
        metric_source=config["scoring"].get("metric_source", "final_test"),
    )
    pareto_df, champions, stability = compute_pareto_and_champions(
        scored,
        config["scoring"]["scenarios"],
        config["scoring"]["champion_zones"],
        dirs["pareto"],
    )
    retention = retention_and_cost_reduction(champions, dirs["pareto"])
    stability.to_excel(dirs["sensitivity"] / "champion_stability_across_weights.xlsx", index=False)
    scored.to_excel(dirs["sensitivity"] / "score_sensitivity_analysis.xlsx", index=False)
    _write_checkpoint(out_root, "pareto_sensitivity", {"scored_rows": len(scored), "champion_rows": len(champions)})

    print("\n========== EXPORT CHAMPIONS + COVERAGE ==========")
    export_champion_feature_sets(champions, dirs["feature_sets"])
    coverage, intersections = compute_core_coverage(champions, consensus_tiers, dirs["coverage"])
    _write_checkpoint(out_root, "coverage", {"coverage_rows": len(coverage), "intersection_rows": len(intersections)})

    print("\n========== MANUSCRIPT TABLES ==========")
    table_classifier = champions[champions["scenario"] == "equal"].copy() if not champions.empty else pd.DataFrame()
    if not table_classifier.empty:
        keep = [
            "dataset", "classifier", "experiment_role", "zone", "method", "criterion", "num_features", "accuracy", "precision", "recall", "f1",
            "macro_FAR", "final_training_plus_prediction_time_s", "final_inference_time_ms_per_sample_mean",
            "feature_set_id",
        ]
        table_classifier[[c for c in keep if c in table_classifier.columns]].to_excel(dirs["tables"] / "revised_table_classifier_matrix.xlsx", index=False)
    coverage.to_excel(dirs["tables"] / "revised_table_core_coverage.xlsx", index=False)
    retention.to_excel(dirs["tables"] / "revised_table_retention_cost_reduction.xlsx", index=False)
    pd.concat(all_audits, ignore_index=True).to_excel(dirs["tables"] / "revised_table_2_dataset_statistics.xlsx", index=False)
    scored.to_excel(dirs["tables"] / "revised_table_score_sensitivity_inputs.xlsx", index=False)
    if not timings_all.empty:
        timings_all.groupby(["dataset", "classifier", "feature_set_id"], dropna=False).agg(
            inference_time_ms_per_sample_mean=("inference_time_ms_per_sample", "mean"),
            inference_time_ms_per_sample_std=("inference_time_ms_per_sample", "std"),
            cpu_memory_mb_mean=("cpu_memory_mb", "mean"),
            gpu_peak_memory_mb_max=("gpu_peak_memory_mb", "max"),
        ).reset_index().to_excel(dirs["tables"] / "revised_table_timing_memory.xlsx", index=False)
    else:
        pd.DataFrame().to_excel(dirs["tables"] / "revised_table_timing_memory.xlsx", index=False)
    _write_checkpoint(out_root, "manuscript_tables", {"tables_dir": str(dirs["tables"])})

    print(f"\nDONE. Outputs saved to: {out_root}")
    print("Resume tip: rerun the same command with --resume. This version skips completed evaluation combinations, not whole datasets.")
    return 0


if __name__ == "__main__":
    main()
