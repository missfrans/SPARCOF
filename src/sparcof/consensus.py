from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .utils import ensure_dir, save_feature_csv


def build_consensus_core(rankings: pd.DataFrame, output_dir: str, rrf_k: int = 60, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create Borda, frequency, RRF consensus, and 3-tier core features.

    This implements the reviewer-facing Stage-1 traceability artifact. The highest-mean RRF cluster is Tier 1/Core.
    """
    output_dir = ensure_dir(output_dir)
    ok = rankings[rankings["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No successful feature-selection rankings are available for consensus generation.")
    rows = []
    for dataset, gd in ok.groupby("dataset"):
        features = sorted(gd["feature"].dropna().astype(str).unique())
        max_rank = gd["rank"].max()
        for feat in features:
            gf = gd[gd["feature"] == feat]
            ranks = gf["rank"].astype(float).values
            frequency = len(gf)
            mean_rank = float(np.mean(ranks)) if len(ranks) else np.inf
            borda = float(np.sum(max_rank - ranks + 1)) if len(ranks) else 0.0
            rrf = float(np.sum(1.0 / (rrf_k + ranks))) if len(ranks) else 0.0
            rows.append({
                "dataset": dataset,
                "feature": feat,
                "selection_frequency": frequency,
                "mean_rank": mean_rank,
                "borda_score": borda,
                "rrf_score": rrf,
            })
    consensus = pd.DataFrame(rows)
    tier_rows = []
    for dataset, gd in consensus.groupby("dataset"):
        X = gd[["rrf_score"]].values
        n_clusters = min(3, len(gd))
        if n_clusters <= 1:
            labels = np.zeros(len(gd), dtype=int)
        else:
            km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = km.fit_predict(X)
        tmp = gd.copy()
        tmp["cluster"] = labels
        cluster_means = tmp.groupby("cluster")["rrf_score"].mean().sort_values(ascending=False)
        tier_map = {cluster: i + 1 for i, cluster in enumerate(cluster_means.index)}
        tmp["tier"] = tmp["cluster"].map(tier_map)
        tmp["tier_label"] = tmp["tier"].map({1: "Core", 2: "Important", 3: "Marginal"}).fillna("Marginal")
        tier_rows.append(tmp)
        core = tmp[tmp["tier"] == 1].sort_values("rrf_score", ascending=False)["feature"].tolist()
        save_feature_csv(core, output_dir / f"{dataset}_core.csv")
    consensus_tiers = pd.concat(tier_rows, ignore_index=True)
    consensus.to_excel(output_dir / "consensus_scores.xlsx", index=False)
    consensus_tiers.to_excel(output_dir / "consensus_tiers_and_core_features.xlsx", index=False)
    return consensus, consensus_tiers


def export_champion_feature_sets(champions: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    rows = []
    if champions.empty:
        return pd.DataFrame()
    for _, row in champions.iterrows():
        dataset = row["dataset"]
        zone = row["zone"]
        features = str(row.get("selected_features", "")).split(";") if row.get("selected_features", "") else []
        path = output_dir / f"{dataset}_{zone}.csv"
        save_feature_csv(features, path)
        rows.append({"dataset": dataset, "zone": zone, "num_features": len(features), "path": str(path)})
    out = pd.DataFrame(rows)
    out.to_excel(output_dir / "exported_champion_feature_sets.xlsx", index=False)
    return out
