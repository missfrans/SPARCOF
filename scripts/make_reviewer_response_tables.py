#!/usr/bin/env python3
"""Collect key Excel outputs into a reviewer-response evidence workbook."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_excel(path)
    return pd.DataFrame([{"missing_file": str(path)}])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/full_revision_rerun")
    p.add_argument("--evidence-workbook", default="outputs/full_revision_rerun/reviewer_response_evidence_workbook.xlsx")
    args = p.parse_args()
    root = Path(args.output_dir)
    sheets = {
        "dataset_statistics": read_if_exists(root / "09_tables_for_manuscript" / "revised_table_2_dataset_statistics.xlsx"),
        "classifier_matrix": read_if_exists(root / "09_tables_for_manuscript" / "revised_table_classifier_matrix.xlsx"),
        "core_coverage": read_if_exists(root / "09_tables_for_manuscript" / "revised_table_core_coverage.xlsx"),
        "retention_cost": read_if_exists(root / "09_tables_for_manuscript" / "revised_table_retention_cost_reduction.xlsx"),
        "timing_memory": read_if_exists(root / "09_tables_for_manuscript" / "revised_table_timing_memory.xlsx"),
        "champion_stability": read_if_exists(root / "07_sensitivity" / "champion_stability_across_weights.xlsx"),
        "pareto_champions": read_if_exists(root / "05_pareto" / "champion_solutions.xlsx"),
    }
    out = Path(args.evidence_workbook)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
