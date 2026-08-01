from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .metrics import classification_metrics
from .utils import ensure_dir


def _gpu_memory_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    except Exception:
        pass
    return float("nan")


def _cpu_memory_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
    except ImportError:
        # ru_maxrss is KiB on Linux and bytes on macOS. It is a peak value,
        # whereas psutil reports current RSS; the backend column and environment
        # metadata should therefore accompany timing/memory comparisons.
        try:
            import resource

            value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value / (1024 ** 2) if sys.platform == "darwin" else value / 1024
        except ImportError:
            return float("nan")


class TorchLinearSVM:
    """GPU-accelerated linear SVM implemented in PyTorch.

    This avoids RAPIDS/cuML and uses hinge-style objectives:
    - binary: standard hinge loss with labels {-1, +1};
    - multiclass: MultiMarginLoss, a multiclass hinge loss.
    It is intended for the controlled GPU-only classifier matrix, not as a
    drop-in replacement for the original RBF-SVM baseline.
    """
    def __init__(self, input_dim: int, num_classes: int, cfg: dict, seed: int):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.execution_backend = f"torch_{self.device.type}"
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.num_classes = int(num_classes)
        self.binary = self.num_classes == 2
        output_dim = 1 if self.binary else self.num_classes
        self.model = nn.Linear(input_dim, output_dim).to(self.device)
        self.epochs = int(cfg.get("epochs", 15))
        self.batch_size = int(cfg.get("batch_size", 4096))
        self.lr = float(cfg.get("learning_rate", 0.001))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))
        self.margin = float(cfg.get("margin", 1.0))

    def fit(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X_t = torch.tensor(X, dtype=torch.float32)
        y_arr = np.asarray(y).astype(int)
        if self.binary:
            y_signed = np.where(y_arr > 0, 1.0, -1.0).astype("float32")
            y_t = torch.tensor(y_signed, dtype=torch.float32).view(-1, 1)
        else:
            y_t = torch.tensor(y_arr, dtype=torch.long)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.binary:
            def loss_fn(scores, targets):
                return torch.clamp(self.margin - targets * scores, min=0.0).mean()
        else:
            loss_fn = torch.nn.MultiMarginLoss(margin=self.margin)
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                scores = self.model(xb)
                loss = loss_fn(scores, yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.tensor(X[start:start+self.batch_size], dtype=torch.float32).to(self.device)
                scores = self.model(xb)
                if self.binary:
                    pred = (scores.view(-1) >= 0).long()
                else:
                    pred = torch.argmax(scores, dim=1)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds) if preds else np.array([])


class TorchMLP:
    def __init__(self, input_dim: int, num_classes: int, cfg: dict, seed: int):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.execution_backend = f"torch_{self.device.type}"
        torch.manual_seed(seed)
        hidden = cfg.get("hidden_layers", [256, 128])
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.model = nn.Sequential(*layers).to(self.device)
        self.epochs = int(cfg.get("epochs", 20))
        self.batch_size = int(cfg.get("batch_size", 1024))
        self.lr = float(cfg.get("learning_rate", 0.001))
        self.num_classes = num_classes

    def fit(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.tensor(X[start:start+self.batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([])


class TorchCNN1D:
    """1D-CNN baseline aligned with the original UNSW script.

    Original-baseline mode for UNSW uses:
    Conv1D(32, kernel=25, padding=12) -> MaxPool1D(3, stride=3, padding=1) ->
    Conv1D(64, kernel=25, padding=12) -> MaxPool1D(3, stride=3, padding=1) ->
    Dense(1024) -> one binary logit, BCEWithLogitsLoss, SGD, lr=2e-3.

    For non-binary use, the same backbone is used with CrossEntropyLoss and
    n-class output, but the manuscript's original baseline uses this class for
    UNSW-NB15 binary detection.
    """
    def __init__(self, input_dim: int, num_classes: int, cfg: dict, seed: int, phase: str = "cv"):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.execution_backend = f"torch_{self.device.type}"
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.num_classes = int(num_classes)
        self.binary_one_logit = bool(cfg.get("binary_one_logit", True)) and self.num_classes == 2
        output_dim = 1 if self.binary_one_logit else self.num_classes

        class IDS1DCNN(nn.Module):
            def __init__(self, input_len, output_dim):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=25, padding=12)
                self.pool1 = nn.MaxPool1d(3, 3, padding=1)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=25, padding=12)
                self.pool2 = nn.MaxPool1d(3, 3, padding=1)
                l = input_len
                l = (l + 2 * 1 - 3) // 3 + 1
                l = (l + 2 * 1 - 3) // 3 + 1
                self.fc1 = nn.Linear(l * 64, 1024)
                self.fc2 = nn.Linear(1024, output_dim)

            def forward(self, x):
                import torch.nn.functional as F
                x = x.unsqueeze(1)
                x = F.relu(self.conv1(x))
                x = self.pool1(x)
                x = F.relu(self.conv2(x))
                x = self.pool2(x)
                x = x.reshape(x.size(0), -1)
                x = F.relu(self.fc1(x))
                return self.fc2(x)

        self.model = IDS1DCNN(input_dim, output_dim).to(self.device)
        if phase == "final":
            self.epochs = int(cfg.get("epochs_final", cfg.get("epochs", 50)))
        else:
            self.epochs = int(cfg.get("epochs_cv", cfg.get("epochs", 30)))
        self.batch_size = int(cfg.get("batch_size", 1024))
        self.lr = float(cfg.get("learning_rate", 2e-3))
        self.optimizer_name = str(cfg.get("optimizer", "sgd")).lower()

    def fit(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X_t = torch.tensor(X, dtype=torch.float32)
        if self.binary_one_logit:
            y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            y_t = torch.tensor(y, dtype=torch.long)
            loss_fn = torch.nn.CrossEntropyLoss()
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        if self.optimizer_name == "sgd":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.tensor(X[start:start+self.batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(xb)
                if self.binary_one_logit:
                    pred = (torch.sigmoid(logits) >= 0.5).long().view(-1)
                else:
                    pred = torch.argmax(logits, dim=1)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds) if preds else np.array([])


def build_classifier(name: str, input_dim: int, num_classes: int, config: dict, seed: int, phase: str = "cv"):
    name = name.lower()
    use_cuml = False
    if name in {"svm_gpu", "rf_gpu"}:
        try:
            import cupy as cp  # noqa: F401
            use_cuml = True
        except Exception:
            use_cuml = False
    if name in {"svm_gpu", "svm_cpu"}:
        svm_cfg = config.get("model_evaluation", {}).get("svm", {})
        kernel = svm_cfg.get("kernel", "rbf")
        C = float(svm_cfg.get("C", 1.0))
        gamma = svm_cfg.get("gamma", "scale")
        class_weight = svm_cfg.get("class_weight", "balanced")

        # svm_gpu uses RAPIDS/cuML SVC when available. If cuML is unavailable,
        # the implementation falls back to sklearn SVC. For very long WSL/RAPIDS
        # runs, users may switch the config from svm_gpu to svm_cpu.
        if name == "svm_cpu":
            model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight)
            model.execution_backend = "scikit_learn_cpu"
            return model

        if use_cuml:
            try:
                from cuml.svm import SVC as cuSVC
                try:
                    model = cuSVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight)
                except TypeError:
                    model = cuSVC(kernel=kernel, C=C, gamma=gamma)
                model.execution_backend = "cuml_gpu"
                return model
            except Exception as exc:
                print(f"[WARN] cuML SVC unavailable/failed during build ({exc}); falling back to sklearn SVC.")
        model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight)
        model.execution_backend = "scikit_learn_cpu_fallback"
        return model
    if name in {"svm_linear_gpu", "linear_svm_gpu"}:
        return TorchLinearSVM(input_dim, num_classes, config.get("model_evaluation", {}).get("svm_linear_gpu", {}), seed)
    if name == "linear_svm":
        model = LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=3000)
        model.execution_backend = "scikit_learn_cpu"
        return model
    if name in {"rf_gpu", "rf_cpu"}:
        if name == "rf_cpu":
            model = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)
            model.execution_backend = "scikit_learn_cpu"
            return model
        if use_cuml:
            try:
                from cuml.ensemble import RandomForestClassifier as cuRF
                model = cuRF(n_estimators=300, max_depth=16, random_state=seed)
                model.execution_backend = "cuml_gpu"
                return model
            except Exception as exc:
                print(f"[WARN] cuML RF unavailable/failed during build ({exc}); falling back to sklearn RandomForest.")
        model = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)
        model.execution_backend = "scikit_learn_cpu_fallback"
        return model
    if name == "dt_cpu":
        dt_cfg = config.get("model_evaluation", {}).get("decision_tree", {})
        model = DecisionTreeClassifier(
            criterion=dt_cfg.get("criterion", "gini"),
            max_depth=dt_cfg.get("max_depth", None),
            random_state=seed,
            class_weight=dt_cfg.get("class_weight", "balanced"),
        )
        model.execution_backend = "scikit_learn_cpu"
        return model
    if name == "mlp_gpu":
        return TorchMLP(input_dim, num_classes, config.get("model_evaluation", {}).get("mlp", {}), seed)
    if name == "cnn1d_gpu":
        return TorchCNN1D(input_dim, num_classes, config.get("model_evaluation", {}).get("cnn1d", {}), seed, phase=phase)
    raise ValueError(f"Unsupported classifier: {name}")


def _fit_predict_model(model, X_train, y_train, X_test, classifier_name: str):
    use_cuml = classifier_name in {"svm_gpu", "rf_gpu"} and "cuml" in str(type(model)).lower()
    if use_cuml:
        import cupy as cp
        X_train_g = cp.asarray(X_train)
        y_train_g = cp.asarray(y_train)
        X_test_g = cp.asarray(X_test)
        model.fit(X_train_g, y_train_g)
        pred = model.predict(X_test_g)
        pred = cp.asnumpy(pred).astype(int)
        return pred
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return np.asarray(pred).astype(int)


def repeated_inference_timing(model, X_test, classifier_name: str, repeats: int, warmup: int) -> pd.DataFrame:
    rows = []
    # Warmup
    for _ in range(max(0, warmup)):
        try:
            if "cuml" in str(type(model)).lower():
                import cupy as cp
                _ = model.predict(cp.asarray(X_test[: min(len(X_test), 2048)]))
                cp.cuda.Stream.null.synchronize()
            else:
                _ = model.predict(X_test[: min(len(X_test), 2048)])
        except Exception:
            pass
    for run in range(repeats):
        start = time.perf_counter()
        if "cuml" in str(type(model)).lower():
            import cupy as cp
            pred = model.predict(cp.asarray(X_test))
            cp.cuda.Stream.null.synchronize()
            _ = cp.asnumpy(pred)
        else:
            _ = model.predict(X_test)
        elapsed = time.perf_counter() - start
        rows.append({
            "timing_run": run + 1,
            "inference_time_total_s": elapsed,
            "inference_time_ms_per_sample": (elapsed / len(X_test)) * 1000 if len(X_test) else np.nan,
            "cpu_memory_mb": _cpu_memory_mb(),
            "gpu_peak_memory_mb": _gpu_memory_mb(),
        })
    return pd.DataFrame(rows)


def _append_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    """Append rows to CSV immediately after one evaluation combination finishes.

    This makes long reruns safer: if the process is interrupted, completed
    combinations remain on disk and can be skipped by --resume.
    """
    if df is None or df.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def _combo_key(feature_set_id, classifier: str) -> str:
    return f"{feature_set_id}||{classifier}"


def _read_completed_combo_keys(path: Path) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, usecols=["feature_set_id", "classifier"])
    except Exception:
        return set()
    return {_combo_key(r["feature_set_id"], r["classifier"]) for _, r in df.dropna(subset=["feature_set_id", "classifier"]).iterrows()}


def _deduplicate_csv(path: Path, subset: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    available = [column for column in subset if column in frame.columns]
    if available:
        frame = frame.drop_duplicates(subset=available, keep="last").reset_index(drop=True)
        frame.to_csv(path, index=False)
    return frame


def evaluate_candidate_feature_sets(
    dataset_name: str,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    candidates: pd.DataFrame,
    classifiers: List[str],
    config: dict,
    seed: int,
    output_dir: str,
    resume: bool = False,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    labels = np.unique(np.concatenate([y_trainval, y_test]))
    idx_map = {f: i for i, f in enumerate(feature_names)}
    n_splits = int(config["project"].get("n_splits_cv", 10))
    timing_repeats = int(config["project"].get("timing_repeats", 30))
    timing_warmup = int(config["project"].get("timing_warmup", 5))

    # These files are intentionally written incrementally. They are both the
    # final dataset-level artifact and the resume source of truth.
    results_path = output_dir / f"{dataset_name}_evaluation_results.csv"
    folds_path = output_dir / f"{dataset_name}_cv_fold_metrics_detailed.csv"
    timings_path = output_dir / f"{dataset_name}_repeated_timing_runs.csv"
    progress_path = output_dir / f"{dataset_name}_evaluation_progress.csv"

    if force:
        for pth in [results_path, folds_path, timings_path, progress_path]:
            if pth.exists():
                pth.unlink()

    completed_keys = _read_completed_combo_keys(results_path) if resume and not force else set()
    total_combos = int(len(candidates) * len(classifiers))
    done_initial = len(completed_keys)
    if completed_keys:
        print(f"[RESUME] {dataset_name}: found {done_initial} completed evaluation combinations; these will be skipped.")

    for _, cand in candidates.iterrows():
        selected = [f for f in str(cand["selected_features"]).split(";") if f]
        missing_features = [feature for feature in selected if feature not in idx_map]
        if missing_features:
            raise ValueError(
                f"Candidate {cand['feature_set_id']!r} contains features absent from the transformed matrix: "
                f"{missing_features[:20]}"
            )
        if len(selected) != len(set(selected)):
            raise ValueError(f"Candidate {cand['feature_set_id']!r} contains duplicate feature names.")
        feat_idx = [idx_map[f] for f in selected]
        if not feat_idx:
            continue
        X_tv = X_trainval[:, feat_idx]
        X_te = X_test[:, feat_idx]
        for clf_name in classifiers:
            key = _combo_key(cand["feature_set_id"], clf_name)
            if key in completed_keys:
                print(f"[RESUME][SKIP] {dataset_name}: {cand['method']}/{cand['criterion']} k={len(feat_idx)} clf={clf_name}")
                continue

            print(f"[EVAL] {dataset_name}: {cand['method']}/{cand['criterion']} k={len(feat_idx)} clf={clf_name}")
            combo_start = time.perf_counter()
            fold_rows_combo = []
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tv, y_trainval), start=1):
                model = build_classifier(clf_name, X_tv.shape[1], len(labels), config, seed + fold, phase="cv")
                t0 = time.perf_counter()
                pred = _fit_predict_model(model, X_tv[tr_idx], y_trainval[tr_idx], X_tv[va_idx], clf_name)
                train_eval_time = time.perf_counter() - t0
                met = classification_metrics(y_trainval[va_idx], pred, labels=labels)
                row = {k: v for k, v in met.items() if k != "confusion_matrix"}
                row["confusion_matrix_json"] = json.dumps(met["confusion_matrix"].tolist())
                row["encoded_labels_json"] = json.dumps(labels.tolist())
                row.update({
                    "dataset": dataset_name,
                    "classifier": clf_name,
                    "feature_set_id": cand["feature_set_id"],
                    "method": cand["method"],
                    "family": cand["family"],
                    "criterion": cand["criterion"],
                    "num_features": len(feat_idx),
                    "fold": fold,
                    "training_time_s": train_eval_time,
                    "execution_backend": getattr(model, "execution_backend", type(model).__module__),
                })
                fold_rows_combo.append(row)

            final_model = build_classifier(clf_name, X_tv.shape[1], len(labels), config, seed, phase="final")
            if clf_name == "svm_gpu" and len(X_tv) > int(config["model_evaluation"].get("max_rows_for_svm", 150000)):
                rng = np.random.default_rng(seed)
                sampled_idx = rng.choice(len(X_tv), size=int(config["model_evaluation"].get("max_rows_for_svm", 150000)), replace=False)
                fit_X, fit_y = X_tv[sampled_idx], y_trainval[sampled_idx]
            else:
                fit_X, fit_y = X_tv, y_trainval
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

            train_start = time.perf_counter()
            final_pred = _fit_predict_model(final_model, fit_X, fit_y, X_te, clf_name)
            final_train_plus_pred_s = time.perf_counter() - train_start
            final_metrics = classification_metrics(y_test, final_pred, labels=labels)
            fold_frame = pd.DataFrame(fold_rows_combo)
            cv_metric_columns = ["accuracy", "precision", "recall", "f1", "macro_DR", "macro_FAR", "macro_BCI"]
            cv_summary = {}
            for metric_name in cv_metric_columns:
                if metric_name in fold_frame:
                    cv_summary[f"cv_{metric_name}_mean"] = float(fold_frame[metric_name].mean())
                    cv_summary[f"cv_{metric_name}_std"] = float(fold_frame[metric_name].std(ddof=1))
            timing_df = repeated_inference_timing(final_model, X_te, clf_name, timing_repeats, timing_warmup)
            timing_df.insert(0, "dataset", dataset_name)
            timing_df.insert(1, "classifier", clf_name)
            timing_df.insert(2, "feature_set_id", cand["feature_set_id"])
            timing_df.insert(3, "method", cand["method"])
            timing_df.insert(4, "criterion", cand["criterion"])
            timing_summary = timing_df["inference_time_ms_per_sample"].agg(["mean", "std", "min", "max"]).to_dict()
            final_row = {k: v for k, v in final_metrics.items() if k != "confusion_matrix"}
            final_row["confusion_matrix_json"] = json.dumps(final_metrics["confusion_matrix"].tolist())
            final_row["encoded_labels_json"] = json.dumps(labels.tolist())
            final_row.update({
                "dataset": dataset_name,
                "classifier": clf_name,
                "feature_set_id": cand["feature_set_id"],
                "method": cand["method"],
                "family": cand["family"],
                "criterion": cand["criterion"],
                "num_features": len(feat_idx),
                "experiment_role": "original_baseline" if clf_name == config.get("model_evaluation", {}).get("original_baselines", {}).get(dataset_name) else "controlled_classifier_matrix",
                "execution_backend": getattr(final_model, "execution_backend", type(final_model).__module__),
                "selected_features": ";".join(selected),
                "final_training_plus_prediction_time_s": final_train_plus_pred_s,
                "final_inference_time_ms_per_sample_mean": timing_summary.get("mean"),
                "final_inference_time_ms_per_sample_std": timing_summary.get("std"),
                "final_inference_time_ms_per_sample_min": timing_summary.get("min"),
                "final_inference_time_ms_per_sample_max": timing_summary.get("max"),
                "cpu_memory_mb": _cpu_memory_mb(),
                "gpu_peak_memory_mb": _gpu_memory_mb(),
                "combo_elapsed_s": time.perf_counter() - combo_start,
            })
            final_row.update(cv_summary)

            # IMPORTANT: persist completed combination immediately. This means
            # 04_model_evaluation/ is no longer empty during long evaluation,
            # and --resume can skip already completed combinations.
            _append_dataframe_csv(pd.DataFrame(fold_rows_combo), folds_path)
            _append_dataframe_csv(timing_df, timings_path)
            _append_dataframe_csv(pd.DataFrame([final_row]), results_path)
            progress_row = {
                "dataset": dataset_name,
                "feature_set_id": cand["feature_set_id"],
                "method": cand["method"],
                "criterion": cand["criterion"],
                "classifier": clf_name,
                "num_features": len(feat_idx),
                "status": "completed",
                "combo_elapsed_s": final_row["combo_elapsed_s"],
                "completed_unix": time.time(),
            }
            _append_dataframe_csv(pd.DataFrame([progress_row]), progress_path)
            completed_keys.add(key)
            print(f"[SAVE] {dataset_name}: appended completed combo to {results_path}")

    results = _deduplicate_csv(results_path, ["dataset", "feature_set_id", "classifier"])
    folds = _deduplicate_csv(folds_path, ["dataset", "feature_set_id", "classifier", "fold"])
    timings = _deduplicate_csv(
        timings_path,
        ["dataset", "feature_set_id", "classifier", "timing_run"],
    )
    _deduplicate_csv(progress_path, ["dataset", "feature_set_id", "classifier", "status"])
    print(f"[EVAL DONE] {dataset_name}: completed {len(_read_completed_combo_keys(results_path))}/{total_combos} combinations. Results: {results_path}")
    return results, folds, timings
