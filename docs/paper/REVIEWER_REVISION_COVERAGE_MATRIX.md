# Reviewer Revision Coverage Matrix

| Reviewer/editor comment | Pipeline response | Output path |
|---|---|---|
| Figures are blurry/small | Regenerate figures/tables from CSV/XLSX outputs rather than screenshots | `outputs/full_revision_rerun/09_tables_for_manuscript/` |
| Dataset totals inconsistent | Dataset audit calculates loaded rows, target distribution, benign/malicious estimate, missing values, duplicates | `00_dataset_audit/dataset_audit_summary.xlsx` |
| Core-feature coverage inconsistent | Core Coverage and Core Purity are separately computed from selected vs. core features | `06_coverage/core_coverage_and_purity.xlsx` |
| Five data scales vs Jaccard matrix mismatch | The pipeline stores explicit split and configuration. If no 1M scale is used, manuscript should state the actual evaluated scales only | `run_config_snapshot.json` and `01_splits/` |
| 17-fold/99.5% claim unclear | Retention/cost-reduction report separates Accuracy retention, F1 retention, training-time reduction, inference-time reduction, and feature-reduction factor | `05_pareto/retention_and_cost_reduction_report.xlsx` |
| Efficiency score appears unusual | Score components and normalized metrics are exported with scenario weights | `05_pareto/effectiveness_efficiency_scores.xlsx`, `analysis_metadata.json` |
| GitHub repository requested | Repository contains code, configuration, split-index generation, and table-generation utilities | Entire repository |
| Lacks post-2024 comparison | The feature-selection/model-evaluation registry can add external comparator methods under identical datasets/splits/classifiers. Add new method names in config and `feature_selection.py` as needed | `02_feature_selection/`, `04_model_evaluation/` |
| Missing feature-selection parameters | Hyperparameters are encoded in method implementations and config snapshot; rankings export `params_json` | `02_feature_selection/all_feature_selection_master_results.xlsx` |
| Dataset-classifier coupling | The pipeline now runs original dataset-specific baselines for continuity and a separate controlled classifier matrix using the same classifiers across UNSW, HIKARI, and CICIoT. Rows are marked by `experiment_role`. | `04_model_evaluation/all_evaluation_results.xlsx`, `09_tables_for_manuscript/revised_table_classifier_matrix.xlsx` |
| Composite score not justified | Multiple scenarios are defined in YAML and exported for sensitivity analysis | `07_sensitivity/score_sensitivity_analysis.xlsx` |
| Deployability overstated | Repeated inference timing, CPU memory, and GPU peak-memory reports are generated | `08_timing_memory/repeated_timing_results.xlsx`, `09_tables_for_manuscript/revised_table_timing_memory.xlsx` |
| Response letter needs traceability | Evidence workbook collates key tables for response letter | `reviewer_response_evidence_workbook.xlsx` |
