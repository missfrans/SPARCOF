from __future__ import annotations

import pandas as pd

from .utils import ensure_dir


def compute_core_coverage(champions: pd.DataFrame, consensus_tiers: pd.DataFrame, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    rows = []
    inter_rows = []
    core_by_dataset = {
        d: set(g[g["tier"] == 1]["feature"].astype(str).tolist())
        for d, g in consensus_tiers.groupby("dataset")
    }
    for _, row in champions.iterrows():
        dataset = row["dataset"]
        zone = row.get("zone", "unknown")
        selected = set(str(row.get("selected_features", "")).split(";")) if row.get("selected_features", "") else set()
        selected = {x for x in selected if x}
        core = core_by_dataset.get(dataset, set())
        intersection = selected & core
        coverage = len(intersection) / len(core) if core else 0.0
        purity = len(intersection) / len(selected) if selected else 0.0
        rows.append({
            "dataset": dataset,
            "zone": zone,
            "method": row.get("method", ""),
            "criterion": row.get("criterion", ""),
            "classifier": row.get("classifier", ""),
            "num_core_features": len(core),
            "num_selected_features": len(selected),
            "num_intersection": len(intersection),
            "core_coverage_recall": coverage,
            "core_purity_alignment": purity,
        })
        for feat in sorted(selected | core):
            inter_rows.append({
                "dataset": dataset,
                "zone": zone,
                "feature": feat,
                "in_core": feat in core,
                "in_selected": feat in selected,
                "in_intersection": feat in intersection,
            })
    coverage = pd.DataFrame(rows)
    intersections = pd.DataFrame(inter_rows)
    coverage.to_excel(output_dir / "core_coverage_and_purity.xlsx", index=False)
    intersections.to_excel(output_dir / "selected_vs_core_intersection.xlsx", index=False)
    return coverage, intersections
