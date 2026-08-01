from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir


@dataclass
class DatasetBundle:
    """A loaded tabular dataset plus the metadata needed for traceability."""

    name: str
    display_name: str
    df: pd.DataFrame
    target_col: str
    task: str
    loader_metadata: Dict[str, Any]
    benign_labels: tuple[str, ...] = ()


def _clean_columns(columns: Iterable[Any]) -> list[str]:
    return (
        pd.Series(list(columns), dtype="string")
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.lower()
        .tolist()
    )


def _resolve_path(value: str | Path, project_root: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


def _discover_files(data_path: Path, file_glob: str, recursive: bool) -> list[Path]:
    if data_path.is_file():
        return [data_path]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
    iterator = data_path.rglob(file_glob) if recursive else data_path.glob(file_glob)
    files = sorted(path for path in iterator if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files matching {file_glob!r} were found in {data_path}")
    return files


def _read_tabular_file(path: Path, fmt: str, options: Mapping[str, Any]) -> pd.DataFrame:
    options = dict(options)
    if fmt == "csv":
        return pd.read_csv(path, **options)
    if fmt == "parquet":
        return pd.read_parquet(path, **options)
    if fmt in {"excel", "xlsx", "xls"}:
        return pd.read_excel(path, **options)
    raise ValueError(f"Unsupported dataset format {fmt!r}; use csv, parquet, or excel.")


def _apply_target_mapping(series: pd.Series, spec: Mapping[str, Any]) -> tuple[pd.Series, Dict[str, Any]]:
    if not spec:
        return series, {}
    mapping = spec.get("mapping", {})
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("target_mapping.mapping must be a non-empty key/value object.")
    case_insensitive = bool(spec.get("case_insensitive", False))
    default_is_set = "default" in spec
    default = spec.get("default")

    source = series.astype("string").str.strip()
    if case_insensitive:
        source = source.str.casefold()
        normalized_mapping = {str(key).strip().casefold(): value for key, value in mapping.items()}
    else:
        normalized_mapping = {str(key).strip(): value for key, value in mapping.items()}
    mapped = source.map(normalized_mapping)
    unknown = sorted(source[mapped.isna()].dropna().unique().tolist())
    if unknown and not default_is_set:
        preview = unknown[:10]
        raise ValueError(
            f"Target mapping does not cover {len(unknown)} label(s): {preview}. "
            "Add them to mapping or set target_mapping.default explicitly."
        )
    if default_is_set:
        mapped = mapped.fillna(default)
    metadata = {
        "target_mapping": dict(mapping),
        "target_mapping_case_insensitive": case_insensitive,
        "target_mapping_default": default if default_is_set else None,
        "unmapped_labels_routed_to_default": unknown,
    }
    return mapped, metadata


def _dataset_fingerprint(files: Sequence[Path]) -> str:
    """Hash file identities without rereading multi-gigabyte datasets.

    The fingerprint detects path, size, and modification-time changes. Researchers
    who require a byte-level provenance checksum can additionally publish the
    checksums supplied by the original dataset provider.
    """

    digest = hashlib.sha256()
    common_root = Path(os.path.commonpath([str(path.resolve().parent) for path in files]))
    for path in files:
        stat = path.stat()
        digest.update(str(path.resolve().relative_to(common_root)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _load_unsw_nb15(spec: Mapping[str, Any], project_root: Path) -> tuple[pd.DataFrame, list[Path], Dict[str, Any]]:
    data_path = _resolve_path(spec["path"], project_root)
    patterns = spec.get("files", [f"UNSW-NB15_{i}.csv" for i in range(1, 5)])
    files = [data_path / str(name) for name in patterns]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing UNSW-NB15 data file(s): {missing}")
    frames = [pd.read_csv(path, header=None, low_memory=False) for path in files]
    df = pd.concat(frames, ignore_index=True)

    configured_names_file = spec.get("column_names_file")
    if configured_names_file:
        name_candidates = [data_path / str(configured_names_file)]
    else:
        # The provider publishes ``UNSW-NB15_features.csv``. Some legacy copies
        # of the original notebook/archive contain the transposed spelling
        # ``NUSW-NB15_features.csv``, so accept it as a compatibility fallback.
        name_candidates = [
            data_path / "UNSW-NB15_features.csv",
            data_path / "NUSW-NB15_features.csv",
        ]
    names_path = next((path for path in name_candidates if path.exists()), name_candidates[0])
    if not names_path.exists():
        expected = ", ".join(str(path) for path in name_candidates)
        raise FileNotFoundError(f"Missing UNSW-NB15 column-name file; expected one of: {expected}")
    names_df = pd.read_csv(names_path, encoding=spec.get("column_names_encoding", "ISO-8859-1"))
    name_col = spec.get("column_names_field", "Name")
    if name_col not in names_df.columns:
        raise ValueError(f"Column-name field {name_col!r} is absent from {names_path}")
    names = _clean_columns(names_df[name_col].tolist())
    if len(names) != df.shape[1]:
        raise ValueError(f"UNSW column-name count ({len(names)}) does not match data width ({df.shape[1]}).")
    df.columns = names
    target_col = str(spec.get("target_col", "label"))
    if target_col in df.columns and target_col == "label":
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    files.append(names_path)
    return df, files, {"loader": "unsw_nb15", "column_names_file": str(names_path)}


def _load_generic(spec: Mapping[str, Any], project_root: Path) -> tuple[pd.DataFrame, list[Path], Dict[str, Any]]:
    if "path" not in spec:
        raise ValueError(f"Dataset {spec.get('name', '<unnamed>')!r} is missing required field 'path'.")
    data_path = _resolve_path(spec["path"], project_root)
    fmt = str(spec.get("format", "csv")).lower()
    default_glob = {"csv": "*.csv", "parquet": "*.parquet", "excel": "*.xlsx"}.get(fmt, "*")
    file_glob = str(spec.get("file_glob", default_glob))
    files = _discover_files(data_path, file_glob, bool(spec.get("recursive", False)))
    read_options = spec.get("read_options", {}) or {}
    frames = [_read_tabular_file(path, fmt, read_options) for path in files]
    widths = {frame.shape[1] for frame in frames}
    if len(widths) != 1:
        raise ValueError(f"Input files for {spec['name']!r} do not have a consistent number of columns: {sorted(widths)}")
    reference_columns = list(frames[0].columns)
    inconsistent = [str(files[index]) for index, frame in enumerate(frames[1:], start=1) if list(frame.columns) != reference_columns]
    if inconsistent:
        raise ValueError(
            f"Input files for {spec['name']!r} do not have identical column names/order. "
            f"First mismatches: {inconsistent[:5]}"
        )
    df = pd.concat(frames, ignore_index=True)
    if bool(spec.get("normalize_column_names", False)):
        df.columns = _clean_columns(df.columns)
    return df, files, {"loader": "generic", "format": fmt, "file_glob": file_glob}


def load_dataset(spec: Mapping[str, Any], project_root: str | Path, seed: int = 42) -> DatasetBundle:
    """Load any configured tabular dataset, with an optional legacy adapter."""

    name = str(spec.get("name", "")).strip()
    if not name:
        raise ValueError("Every dataset requires a non-empty 'name'.")
    display_name = str(spec.get("display_name", name))
    target_col = str(spec.get("target_col", "")).strip()
    if not target_col:
        raise ValueError(f"Dataset {name!r} requires target_col.")

    project_root = Path(project_root)
    loader = str(spec.get("loader", "generic")).lower()
    if loader == "unsw_nb15":
        df, files, metadata = _load_unsw_nb15(spec, project_root)
    elif loader == "generic":
        df, files, metadata = _load_generic(spec, project_root)
    else:
        raise ValueError(f"Unsupported loader {loader!r} for dataset {name!r}.")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column {target_col!r} was not found for {name!r}. "
            f"Available columns begin with: {list(df.columns)[:15]}"
        )
    duplicate_columns = pd.Index(df.columns)[pd.Index(df.columns).duplicated()].astype(str).tolist()
    if duplicate_columns:
        raise ValueError(f"Dataset {name!r} contains duplicate column names: {duplicate_columns[:15]}")
    semicolon_columns = [str(column) for column in df.columns if ";" in str(column)]
    if semicolon_columns:
        raise ValueError(
            f"Dataset {name!r} contains semicolons in column names: {semicolon_columns[:15]}. "
            "Rename these columns because semicolons serialize feature sets."
        )

    df[target_col], mapping_metadata = _apply_target_mapping(df[target_col], spec.get("target_mapping", {}) or {})
    if df[target_col].isna().any():
        raise ValueError(f"Target column {target_col!r} contains missing values after target mapping.")
    if df[target_col].nunique(dropna=True) < 2:
        raise ValueError(f"Dataset {name!r} must contain at least two target classes.")

    if bool(spec.get("shuffle", False)):
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    task = str(spec.get("task", "auto")).lower()
    if task == "auto":
        task = "binary" if df[target_col].nunique(dropna=True) == 2 else "multiclass"
    if task not in {"binary", "multiclass"}:
        raise ValueError(f"Dataset {name!r} has unsupported task {task!r}.")
    if task == "binary" and df[target_col].nunique(dropna=True) != 2:
        raise ValueError(f"Dataset {name!r} is configured as binary but has {df[target_col].nunique()} target classes.")

    metadata.update(mapping_metadata)
    metadata.update(
        {
            "raw_files": [str(path) for path in files],
            "dataset_fingerprint": _dataset_fingerprint(files),
            "loaded_rows": int(len(df)),
            "loaded_columns": int(df.shape[1]),
        }
    )
    benign_labels = tuple(str(value).casefold() for value in spec.get("benign_labels", []))
    return DatasetBundle(name, display_name, df, target_col, task, metadata, benign_labels)


def audit_dataset(bundle: DatasetBundle, output_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    df = bundle.df
    target = bundle.target_col
    label_counts = df[target].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    label_counts.insert(0, "dataset", bundle.display_name)

    normalized_target = df[target].astype("string").str.strip().str.casefold()
    if bundle.benign_labels:
        benign_mask = normalized_target.isin(bundle.benign_labels)
        benign_count = int(benign_mask.sum())
        malicious_count = int((~benign_mask).sum())
    else:
        benign_count = np.nan
        malicious_count = np.nan

    summary = pd.DataFrame(
        [
            {
                "dataset": bundle.display_name,
                "raw_total_rows_after_loading": int(len(df)),
                "num_columns_after_loading": int(df.shape[1]),
                "num_features_before_preprocessing": int(df.shape[1] - 1),
                "target_column": target,
                "task": bundle.task,
                "num_target_classes": int(df[target].nunique(dropna=True)),
                "benign_or_normal_count": benign_count,
                "malicious_or_attack_count": malicious_count,
                "missing_values_total": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "dataset_fingerprint": bundle.loader_metadata.get("dataset_fingerprint", ""),
            }
        ]
    )
    summary.to_csv(output_dir / f"{bundle.name}_dataset_audit_summary.csv", index=False)
    label_counts.to_csv(output_dir / f"{bundle.name}_label_counts.csv", index=False)
    return summary, label_counts


def write_combined_audit(
    summaries: list[pd.DataFrame], label_counts: list[pd.DataFrame], output_dir: str | Path
) -> None:
    output_dir = ensure_dir(output_dir)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_excel(output_dir / "dataset_audit_summary.xlsx", index=False)
    if label_counts:
        pd.concat(label_counts, ignore_index=True).to_excel(
            output_dir / "label_distribution_all_datasets.xlsx", index=False
        )
