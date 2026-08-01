from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def require_gpu_or_warn(required: bool = False) -> None:
    available = gpu_available()
    if required and not available:
        raise RuntimeError(
            "GPU was required, but no CUDA device was detected by PyTorch. "
            "Install CUDA/PyTorch/RAPIDS or run without --gpu-required for CPU fallback."
        )
    if not available:
        print("[WARN] No CUDA device detected by PyTorch. GPU-first components will use CPU fallback where possible.")


def try_import_cuml() -> bool:
    try:
        import cuml  # noqa: F401
        return True
    except Exception:
        return False


def write_excel_sheets(path: str | Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = name[:31].replace("/", "_")
            df.to_excel(writer, index=False, sheet_name=safe_name)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        if ";" in value:
            return [x.strip() for x in value.split(";") if x.strip()]
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        return [value.strip()] if value.strip() else []
    return [str(value)]


def save_feature_csv(features: Iterable[str], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame({"feature": list(features)}).to_csv(path, index=False)


def collect_environment_metadata(project_root: str | Path) -> Dict[str, Any]:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "PyYAML",
        "openpyxl",
        "psutil",
        "torch",
        "xgboost",
        "lightgbm",
        "cupy",
        "cuml",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        git_commit = result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    cuda = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception:
        pass
    return {
        "python": sys.version,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "package_versions": versions,
        "git_commit": git_commit,
        "cuda": cuda,
    }


def load_feature_csv(path: str | Path) -> List[str]:
    df = pd.read_csv(path)
    if "feature" in df.columns:
        return df["feature"].dropna().astype(str).tolist()
    return df.iloc[:, 0].dropna().astype(str).tolist()
