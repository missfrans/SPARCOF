from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical config while preserving the original paper YAML files.

    Early SPARCOF configs stored dataset paths in a top-level ``paths`` mapping.
    New configs keep each path beside its loader settings. This compatibility
    conversion lets archived article configs remain executable.
    """

    normalized = deepcopy(dict(config))
    legacy_paths = normalized.get("paths", {}) or {}
    key_by_name = {"unsw": "unsw_dir", "hikari": "hikari_dir", "ciciot": "ciciot_dir"}
    for dataset in normalized.get("datasets", []) or []:
        name = str(dataset.get("name", "")).lower()
        if not dataset.get("path") and key_by_name.get(name) in legacy_paths:
            dataset["path"] = legacy_paths[key_by_name[name]]
        if name == "unsw" and "loader" not in dataset:
            dataset["loader"] = "unsw_nb15"
            dataset.setdefault("normalize_column_names", True)
            dataset.setdefault("benign_labels", [0, "0"])
        elif name == "hikari" and "loader" not in dataset:
            dataset["loader"] = "generic"
            dataset.setdefault("file_glob", "*.csv")
            dataset.setdefault(
                "drop_columns",
                ["no", "Unnamed: 0", "uid", "originh", "originp", "responh", "responp", "traffic_category"],
            )
            dataset.setdefault("benign_labels", ["Benign", "Background", "Normal", 0, "0"])
        elif name == "ciciot" and "loader" not in dataset:
            dataset["loader"] = "generic"
            dataset.setdefault("file_glob", "*.csv")
            dataset.setdefault("shuffle", True)
            dataset.setdefault(
                "target_mapping",
                {"case_insensitive": True, "mapping": {"benign": "Benign", "0": "Benign"}, "default": "Attack"},
            )
            dataset.setdefault("benign_labels", ["Benign"])
    return normalized


def find_project_root(config_path: str | Path) -> Path:
    """Locate the repository root so relative paths do not depend on the shell CWD."""

    config_path = Path(config_path).expanduser().resolve()
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return config_path.parent


def validate_config(config: Mapping[str, Any]) -> None:
    errors: list[str] = []
    project = config.get("project")
    datasets = config.get("datasets")
    feature_selection = config.get("feature_selection")
    model_evaluation = config.get("model_evaluation")
    scoring = config.get("scoring")

    if not isinstance(project, Mapping):
        errors.append("project must be a mapping.")
    if not isinstance(datasets, list) or not datasets:
        errors.append("datasets must be a non-empty list.")
    else:
        names: list[str] = []
        for index, dataset in enumerate(datasets):
            prefix = f"datasets[{index}]"
            if not isinstance(dataset, Mapping):
                errors.append(f"{prefix} must be a mapping.")
                continue
            for field in ("name", "path", "target_col"):
                if not str(dataset.get(field, "")).strip():
                    errors.append(f"{prefix}.{field} is required.")
            names.append(str(dataset.get("name", "")).strip())
        duplicates = sorted({name for name in names if name and names.count(name) > 1})
        if duplicates:
            errors.append(f"Dataset names must be unique; duplicates: {duplicates}")

    if not isinstance(feature_selection, Mapping):
        errors.append("feature_selection must be a mapping.")
    if not isinstance(model_evaluation, Mapping):
        errors.append("model_evaluation must be a mapping.")
    if not isinstance(scoring, Mapping) or not isinstance(scoring.get("scenarios"), Mapping):
        errors.append("scoring.scenarios must be a mapping.")
    else:
        if scoring.get("metric_source", "final_test") not in {"final_test", "cross_validation"}:
            errors.append("scoring.metric_source must be 'final_test' or 'cross_validation'.")
        for scenario_name, scenario in scoring["scenarios"].items():
            for group in ("effectiveness", "efficiency"):
                weights = scenario.get(group, {}) if isinstance(scenario, Mapping) else {}
                if not isinstance(weights, Mapping) or not weights:
                    errors.append(f"scoring.scenarios.{scenario_name}.{group} must contain weights.")
                    continue
                total = sum(float(value) for value in weights.values())
                if abs(total - 1.0) > 1e-9:
                    errors.append(
                        f"Weights in scoring.scenarios.{scenario_name}.{group} must sum to 1.0; got {total:.12g}."
                    )
                if any(float(value) < 0 for value in weights.values()):
                    errors.append(f"Weights in scoring.scenarios.{scenario_name}.{group} cannot be negative.")

    if isinstance(project, Mapping):
        n_splits = int(project.get("n_splits_cv", 10))
        test_size = float(project.get("test_size", 0.2))
        if n_splits < 2:
            errors.append("project.n_splits_cv must be at least 2.")
        if not 0 < test_size < 1:
            errors.append("project.test_size must be between 0 and 1.")

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))


def run_fingerprint(config: Mapping[str, Any], *, mode: str, max_rows: int | None) -> str:
    payload = {"config": config, "mode": mode, "max_rows_per_dataset": max_rows}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
