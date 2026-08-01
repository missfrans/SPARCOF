from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir, save_json


def _minmax(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn, mx = s.min(), s.max()
    if abs(mx - mn) < 1e-12:
        return pd.Series(np.ones(len(s)), index=s.index)
    norm = (s - mn) / (mx - mn)
    if not higher_is_better:
        norm = 1 - norm
    return norm.fillna(0.0)


def apply_scores(results: pd.DataFrame, scenarios: Dict, metric_source: str = "final_test") -> pd.DataFrame:
    df = results.copy()
    if metric_source not in {"final_test", "cross_validation"}:
        raise ValueError("scoring.metric_source must be 'final_test' or 'cross_validation'.")
    metric_columns = {
        "accuracy": "accuracy" if metric_source == "final_test" else "cv_accuracy_mean",
        "precision": "precision" if metric_source == "final_test" else "cv_precision_mean",
        "recall": "recall" if metric_source == "final_test" else "cv_recall_mean",
        "f1": "f1" if metric_source == "final_test" else "cv_f1_mean",
        "far": "macro_FAR" if metric_source == "final_test" else "cv_macro_FAR_mean",
    }
    missing = [column for column in metric_columns.values() if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot score with metric_source={metric_source!r}; missing columns: {missing}")
    df["selection_metric_source"] = metric_source
    scored_frames = []
    # Normalize per dataset and classifier to avoid cross-dataset scale artifacts.
    group_cols = ["dataset", "classifier"]
    for _, g in df.groupby(group_cols, dropna=False):
        tmp = g.copy()
        tmp["n_accuracy"] = _minmax(tmp[metric_columns["accuracy"]], True)
        tmp["n_precision"] = _minmax(tmp[metric_columns["precision"]], True)
        tmp["n_recall"] = _minmax(tmp[metric_columns["recall"]], True)
        tmp["n_f1"] = _minmax(tmp[metric_columns["f1"]], True)
        tmp["n_inv_far"] = _minmax(tmp[metric_columns["far"]], False)
        tmp["n_inv_training_time"] = _minmax(tmp["final_training_plus_prediction_time_s"], False)
        tmp["n_inv_inference_time"] = _minmax(tmp["final_inference_time_ms_per_sample_mean"], False)
        tmp["n_inv_num_features"] = _minmax(tmp["num_features"], False)
        for scenario_name, weights in scenarios.items():
            eff = weights["effectiveness"]
            efs = weights["efficiency"]
            tmp[f"effectiveness_score__{scenario_name}"] = (
                eff.get("accuracy", 0) * tmp["n_accuracy"]
                + eff.get("precision", 0) * tmp["n_precision"]
                + eff.get("recall", 0) * tmp["n_recall"]
                + eff.get("f1", 0) * tmp["n_f1"]
                + eff.get("inv_far", 0) * tmp["n_inv_far"]
            )
            tmp[f"efficiency_score__{scenario_name}"] = (
                efs.get("inv_training_time", 0) * tmp["n_inv_training_time"]
                + efs.get("inv_inference_time", 0) * tmp["n_inv_inference_time"]
                + efs.get("inv_num_features", 0) * tmp["n_inv_num_features"]
            )
        scored_frames.append(tmp)
    return pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()


def pareto_mask(eff: np.ndarray, efs: np.ndarray) -> np.ndarray:
    n = len(eff)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = (eff >= eff[i]) & (efs >= efs[i]) & ((eff > eff[i]) | (efs > efs[i]))
        if dominates_i.any():
            keep[i] = False
    return keep


def compute_pareto_and_champions(scored: pd.DataFrame, scenarios: Dict, zone_cfg: Dict, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    pareto_rows = []
    champion_rows = []
    stability_rows = []
    for scenario_name in scenarios.keys():
        eff_col = f"effectiveness_score__{scenario_name}"
        efs_col = f"efficiency_score__{scenario_name}"
        for (dataset, classifier), g in scored.groupby(["dataset", "classifier"]):
            g = g.copy().reset_index(drop=True)
            mask = pareto_mask(g[eff_col].values, g[efs_col].values)
            front = g[mask].copy()
            front["scenario"] = scenario_name
            front["pareto_front"] = True
            pareto_rows.append(front)
            if front.empty:
                continue
            for zone, cfg in zone_cfg.items():
                eff_thr = front[eff_col].quantile(cfg.get("effectiveness_quantile_min", 0.0))
                efs_thr = front[efs_col].quantile(cfg.get("efficiency_quantile_min", 0.0))
                z = front[(front[eff_col] >= eff_thr) & (front[efs_col] >= efs_thr)].copy()
                if z.empty:
                    z = front.copy()
                local_eff_max = z[eff_col].max()
                local_efs_max = z[efs_col].max()
                z["local_utopia_distance"] = np.sqrt((local_eff_max - z[eff_col]) ** 2 + (local_efs_max - z[efs_col]) ** 2)
                champ = z.sort_values("local_utopia_distance").iloc[0].copy()
                champ["scenario"] = scenario_name
                champ["zone"] = zone
                champ["local_utopia_effectiveness"] = local_eff_max
                champ["local_utopia_efficiency"] = local_efs_max
                champion_rows.append(champ)
    pareto_df = pd.concat(pareto_rows, ignore_index=True) if pareto_rows else pd.DataFrame()
    champions = pd.DataFrame(champion_rows)

    if not champions.empty:
        for (dataset, classifier, zone), g in champions.groupby(["dataset", "classifier", "zone"]):
            feature_sets = g["feature_set_id"].astype(str).tolist()
            methods = g["method"].astype(str).tolist()
            stability_rows.append({
                "dataset": dataset,
                "classifier": classifier,
                "zone": zone,
                "num_scenarios": len(g),
                "num_unique_feature_sets": len(set(feature_sets)),
                "num_unique_methods": len(set(methods)),
                "stable_same_feature_set_across_scenarios": len(set(feature_sets)) == 1,
                "feature_sets_by_scenario": ";".join(feature_sets),
                "methods_by_scenario": ";".join(methods),
            })
    stability = pd.DataFrame(stability_rows)

    scored.to_excel(output_dir / "effectiveness_efficiency_scores.xlsx", index=False)
    pareto_df.to_excel(output_dir / "pareto_front_results.xlsx", index=False)
    champions.to_excel(output_dir / "champion_solutions.xlsx", index=False)
    stability.to_excel(output_dir / "champion_stability_across_weights.xlsx", index=False)
    save_json({"scenarios": scenarios, "zone_config": zone_cfg}, output_dir / "analysis_metadata.json")
    return pareto_df, champions, stability


def retention_and_cost_reduction(champions: pd.DataFrame, output_dir: str, reference_zone: str = "effectiveness") -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    rows = []
    if champions.empty:
        return pd.DataFrame()
    for (dataset, classifier, scenario), g in champions.groupby(["dataset", "classifier", "scenario"]):
        ref = g[g["zone"] == reference_zone]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for _, row in g.iterrows():
            rows.append({
                "dataset": dataset,
                "classifier": classifier,
                "scenario": scenario,
                "zone": row["zone"],
                "reference_zone": reference_zone,
                "accuracy_retention_vs_reference": row["accuracy"] / ref["accuracy"] if ref["accuracy"] else np.nan,
                "f1_retention_vs_reference": row["f1"] / ref["f1"] if ref["f1"] else np.nan,
                "training_time_reduction_factor": ref["final_training_plus_prediction_time_s"] / row["final_training_plus_prediction_time_s"] if row["final_training_plus_prediction_time_s"] else np.nan,
                "inference_time_reduction_factor": ref["final_inference_time_ms_per_sample_mean"] / row["final_inference_time_ms_per_sample_mean"] if row["final_inference_time_ms_per_sample_mean"] else np.nan,
                "feature_reduction_factor": ref["num_features"] / row["num_features"] if row["num_features"] else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_excel(output_dir / "retention_and_cost_reduction_report.xlsx", index=False)
    return df
