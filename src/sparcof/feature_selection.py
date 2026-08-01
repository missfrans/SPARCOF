from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from .utils import ensure_dir


@dataclass
class RankingResult:
    method: str
    family: str
    ranked_features: List[str]
    ranked_scores: List[float]
    runtime_seconds: float
    params: Dict


def _as_sample(X: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_rows and len(X) > max_rows:
        indices = np.arange(len(X))
        try:
            _, idx = train_test_split(
                indices,
                test_size=max_rows,
                stratify=y,
                random_state=seed,
            )
        except ValueError:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=max_rows, replace=False)
        return X[idx], y[idx]
    return X, y


def _rank_from_scores(feature_names: List[str], scores: np.ndarray, descending: bool = True) -> Tuple[List[str], List[float]]:
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    return [feature_names[i] for i in order], [float(scores[i]) for i in order]


def _safe_abs_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    scores = []
    for j in range(X.shape[1]):
        x = X[:, j].astype(float)
        if np.std(x) == 0 or np.std(y) == 0:
            scores.append(0.0)
        else:
            scores.append(abs(np.corrcoef(x, y)[0, 1]))
    return np.asarray(scores)


def _fisher_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        mean_c = Xc.mean(axis=0)
        var_c = Xc.var(axis=0)
        num += len(Xc) * (mean_c - overall_mean) ** 2
        den += len(Xc) * var_c
    return num / (den + 1e-12)


def _reliefF_score(X: np.ndarray, y: np.ndarray, n_neighbors: int = 10, max_rows: int = 5000, seed: int = 42) -> np.ndarray:
    Xs, ys = _as_sample(X, y, max_rows, seed)
    n = len(Xs)
    if n < 3:
        return np.zeros(X.shape[1])
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n), metric="euclidean")
    nn.fit(Xs)
    distances, indices = nn.kneighbors(Xs)
    scores = np.zeros(Xs.shape[1])
    for i in range(n):
        neigh = indices[i, 1:]
        hit = neigh[ys[neigh] == ys[i]]
        miss = neigh[ys[neigh] != ys[i]]
        if len(hit) > 0:
            scores -= np.mean(np.abs(Xs[i] - Xs[hit]), axis=0) / n
        if len(miss) > 0:
            scores += np.mean(np.abs(Xs[i] - Xs[miss]), axis=0) / n
    return scores


def _mrmr_ranking(X: np.ndarray, y: np.ndarray, feature_names: List[str], seed: int, max_features: Optional[int] = None) -> Tuple[List[str], List[float]]:
    relevance = mutual_info_classif(X, y, random_state=seed, discrete_features=False)
    relevance = np.nan_to_num(relevance)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    n_features = len(feature_names)
    max_features = max_features or n_features
    selected = []
    remaining = list(range(n_features))
    scores_trace = []
    while remaining and len(selected) < max_features:
        best_i, best_score = None, -np.inf
        for i in remaining:
            redundancy = np.mean([corr[i, j] for j in selected]) if selected else 0.0
            score = relevance[i] - redundancy
            if score > best_score:
                best_i, best_score = i, score
        selected.append(best_i)
        remaining.remove(best_i)
        scores_trace.append(float(best_score))
    # Append any remaining features by relevance to preserve a full ranking.
    if remaining:
        rem_order = sorted(remaining, key=lambda i: relevance[i], reverse=True)
        selected.extend(rem_order)
        scores_trace.extend([float(relevance[i]) for i in rem_order])
    return [feature_names[i] for i in selected], scores_trace


def _logreg_scores(X: np.ndarray, y: np.ndarray, penalty: str, C: float, l1_ratio: Optional[float], seed: int) -> np.ndarray:
    kwargs = dict(max_iter=2000, random_state=seed, class_weight="balanced", n_jobs=-1)
    if penalty == "elasticnet":
        model = LogisticRegression(penalty="elasticnet", C=C, solver="saga", l1_ratio=l1_ratio, **kwargs)
    else:
        model = LogisticRegression(penalty=penalty, C=C, solver="saga" if penalty == "l1" else "lbfgs", **kwargs)
    model.fit(X, y)
    coef = getattr(model, "coef_", np.zeros((1, X.shape[1])))
    return np.mean(np.abs(coef), axis=0)


def _tree_importance(X: np.ndarray, y: np.ndarray, seed: int, model_name: str = "rf") -> np.ndarray:
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1,
            )
            model.fit(X, y)
            return np.asarray(model.feature_importances_)
        except Exception as exc:
            raise RuntimeError(
                "The xgboost selector was requested but XGBoost could not run. "
                "Install the optional dependency and verify its runtime."
            ) from exc
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1, verbose=-1)
            model.fit(X, y)
            return np.asarray(model.feature_importances_, dtype=float)
        except Exception as exc:
            raise RuntimeError(
                "The lightgbm selector was requested but LightGBM could not run. "
                "Install the optional dependency and verify its runtime."
            ) from exc
    model = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)
    model.fit(X, y)
    return np.asarray(model.feature_importances_, dtype=float)


def _rfe_ranking(X: np.ndarray, y: np.ndarray, seed: int, rfecv: bool = False) -> np.ndarray:
    base = LogisticRegression(max_iter=1500, solver="liblinear", class_weight="balanced", random_state=seed)
    if rfecv:
        selector = RFECV(base, step=1, cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="f1_weighted", n_jobs=-1)
        selector.fit(X, y)
        return -np.asarray(selector.ranking_, dtype=float)
    selector = RFE(base, n_features_to_select=max(1, X.shape[1] // 2), step=1)
    selector.fit(X, y)
    return -np.asarray(selector.ranking_, dtype=float)


def _boruta_like_score(X: np.ndarray, y: np.ndarray, seed: int, n_iter: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hit_counts = np.zeros(X.shape[1])
    mean_importances = np.zeros(X.shape[1])
    for i in range(n_iter):
        X_shadow = X.copy()
        for j in range(X_shadow.shape[1]):
            rng.shuffle(X_shadow[:, j])
        X_aug = np.hstack([X, X_shadow])
        model = RandomForestClassifier(n_estimators=150, random_state=seed + i, class_weight="balanced", n_jobs=-1)
        model.fit(X_aug, y)
        imp = model.feature_importances_
        real_imp = imp[: X.shape[1]]
        shadow_threshold = np.max(imp[X.shape[1] :])
        hit_counts += real_imp > shadow_threshold
        mean_importances += real_imp
    return hit_counts + (mean_importances / max(n_iter, 1))


def _fitness_subset(X: np.ndarray, y: np.ndarray, mask: np.ndarray, seed: int) -> float:
    if mask.sum() == 0:
        return 0.0
    Xs = X[:, mask]
    try:
        X_tr, X_va, y_tr, y_va = train_test_split(Xs, y, test_size=0.25, stratify=y, random_state=seed)
        clf = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced", random_state=seed)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_va)
        f1 = f1_score(y_va, pred, average="weighted", zero_division=0)
        penalty = 0.01 * (mask.sum() / len(mask))
        return float(f1 - penalty)
    except Exception:
        return 0.0


def _ga_score(X: np.ndarray, y: np.ndarray, seed: int, pool_size: int = 30, population: int = 20, generations: int = 10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    relevance = mutual_info_classif(X, y, random_state=seed)
    pool = np.argsort(relevance)[::-1][: min(pool_size, X.shape[1])]
    pop = rng.random((population, len(pool))) < 0.5
    best_mask_pool = pop[0].copy()
    best_score = -np.inf
    selection_counts = np.zeros(len(pool))
    for gen in range(generations):
        scores = np.array([_fitness_subset(X[:, pool], y, ind, seed + gen) for ind in pop])
        order = np.argsort(scores)[::-1]
        if scores[order[0]] > best_score:
            best_score = scores[order[0]]
            best_mask_pool = pop[order[0]].copy()
        elites = pop[order[: max(2, population // 4)]]
        selection_counts += pop[order[: max(2, population // 2)]].sum(axis=0)
        children = []
        while len(children) < population - len(elites):
            a, b = elites[rng.integers(len(elites))], elites[rng.integers(len(elites))]
            cut = rng.integers(1, len(pool)) if len(pool) > 1 else 1
            child = np.concatenate([a[:cut], b[cut:]])
            mutation = rng.random(len(pool)) < 0.05
            child[mutation] = ~child[mutation]
            children.append(child)
        pop = np.vstack([elites, np.asarray(children)])
    scores_all = np.zeros(X.shape[1])
    scores_all[pool] = selection_counts + best_mask_pool.astype(float) * 100.0 + relevance[pool]
    scores_all += relevance * 0.001
    return scores_all


def _pso_score(X: np.ndarray, y: np.ndarray, seed: int, pool_size: int = 30, particles: int = 20, iterations: int = 10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    relevance = mutual_info_classif(X, y, random_state=seed)
    pool = np.argsort(relevance)[::-1][: min(pool_size, X.shape[1])]
    dim = len(pool)
    pos = rng.random((particles, dim))
    vel = rng.normal(0, 0.1, (particles, dim))
    pbest = pos.copy()
    pbest_scores = np.array([_fitness_subset(X[:, pool], y, p > 0.5, seed) for p in pos])
    gbest = pbest[np.argmax(pbest_scores)].copy()
    trace = np.zeros(dim)
    for it in range(iterations):
        r1, r2 = rng.random((particles, dim)), rng.random((particles, dim))
        vel = 0.7 * vel + 1.4 * r1 * (pbest - pos) + 1.4 * r2 * (gbest - pos)
        pos = 1 / (1 + np.exp(-(pos + vel)))
        scores = np.array([_fitness_subset(X[:, pool], y, p > 0.5, seed + it) for p in pos])
        improve = scores > pbest_scores
        pbest[improve] = pos[improve]
        pbest_scores[improve] = scores[improve]
        gbest = pbest[np.argmax(pbest_scores)].copy()
        trace += (pos > 0.5).sum(axis=0)
    scores_all = np.zeros(X.shape[1])
    scores_all[pool] = trace + (gbest > 0.5).astype(float) * 100.0 + relevance[pool]
    scores_all += relevance * 0.001
    return scores_all


def run_single_selector(method: str, X: np.ndarray, y: np.ndarray, feature_names: List[str], seed: int) -> RankingResult:
    t0 = time.perf_counter()
    params = {"seed": seed}
    family = "unknown"
    method_lower = method.lower()

    if method_lower == "pearson":
        family = "filter"
        scores = _safe_abs_corr(X, y)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "chi2":
        family = "filter"
        X_pos = MinMaxScaler().fit_transform(X)
        scores, _ = chi2(X_pos, y)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "mutual_info":
        family = "filter"
        scores = mutual_info_classif(X, y, random_state=seed)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "fisher":
        family = "filter"
        scores = _fisher_score(X, y)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "relieff":
        family = "filter"
        params.update({"n_neighbors": 10, "max_rows": 5000})
        scores = _reliefF_score(X, y, n_neighbors=10, max_rows=5000, seed=seed)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "mrmr":
        family = "filter"
        ranked, rscores = _mrmr_ranking(X, y, feature_names, seed)
    elif method_lower.startswith("lasso"):
        family = "embedded"
        C = float(method_lower.split("c")[-1]) if "c" in method_lower else 1.0
        params.update({"penalty": "l1", "C": C})
        scores = _logreg_scores(X, y, penalty="l1", C=C, l1_ratio=None, seed=seed)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower.startswith("elasticnet"):
        family = "embedded"
        C = float(method_lower.split("c")[-1]) if "c" in method_lower else 1.0
        params.update({"penalty": "elasticnet", "C": C, "l1_ratio": 0.5})
        scores = _logreg_scores(X, y, penalty="elasticnet", C=C, l1_ratio=0.5, seed=seed)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "rf":
        family = "embedded"
        scores = _tree_importance(X, y, seed, "rf")
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "xgboost":
        family = "embedded"
        scores = _tree_importance(X, y, seed, "xgboost")
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "lightgbm":
        family = "embedded"
        scores = _tree_importance(X, y, seed, "lightgbm")
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "rfe":
        family = "wrapper"
        scores = _rfe_ranking(X, y, seed, rfecv=False)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "rfecv":
        family = "wrapper"
        scores = _rfe_ranking(X, y, seed, rfecv=True)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "boruta":
        family = "wrapper"
        scores = _boruta_like_score(X, y, seed, n_iter=20)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "ga":
        family = "metaheuristic"
        scores = _ga_score(X, y, seed)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower == "pso":
        family = "metaheuristic"
        scores = _pso_score(X, y, seed)
        ranked, rscores = _rank_from_scores(feature_names, scores)
    elif method_lower in ["mi_rfe", "mi_ga", "chi2_boruta", "chi2_xgboost", "fisher_lightgbm"]:
        family = "hybrid"
        if method_lower.startswith("mi"):
            base_scores = mutual_info_classif(X, y, random_state=seed)
        elif method_lower.startswith("chi2"):
            base_scores, _ = chi2(MinMaxScaler().fit_transform(X), y)
        else:
            base_scores = _fisher_score(X, y)
        pool_idx = np.argsort(base_scores)[::-1][: max(3, min(X.shape[1], int(math.ceil(0.5 * X.shape[1]))))]
        X_pool = X[:, pool_idx]
        names_pool = [feature_names[i] for i in pool_idx]
        if method_lower.endswith("rfe"):
            scores_pool = _rfe_ranking(X_pool, y, seed, rfecv=False)
        elif method_lower.endswith("ga"):
            scores_pool = _ga_score(X_pool, y, seed)
        elif method_lower.endswith("boruta"):
            scores_pool = _boruta_like_score(X_pool, y, seed, n_iter=15)
        elif method_lower.endswith("xgboost"):
            scores_pool = _tree_importance(X_pool, y, seed, "xgboost")
        else:
            scores_pool = _tree_importance(X_pool, y, seed, "lightgbm")
        ranked_pool, rscores_pool = _rank_from_scores(names_pool, scores_pool)
        remainder = [f for f in feature_names if f not in set(ranked_pool)]
        ranked = ranked_pool + remainder
        rscores = rscores_pool + [0.0] * len(remainder)
        params.update({"prefilter": method_lower.split("_")[0], "pool_ratio": 0.5})
    else:
        raise ValueError(f"Unsupported feature-selection method: {method}")

    runtime = time.perf_counter() - t0
    return RankingResult(method=method, family=family, ranked_features=ranked, ranked_scores=rscores, runtime_seconds=runtime, params=params)


def run_feature_selection_suite(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    methods: List[str],
    max_rows: int,
    seed: int,
    output_dir: str,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    X_fs, y_fs = _as_sample(X, y, max_rows, seed)
    rows = []
    for method in methods:
        print(f"[FS] {dataset_name}: running {method} on shape={X_fs.shape}")
        try:
            result = run_single_selector(method, X_fs, y_fs, feature_names, seed)
            for rank, (feat, score) in enumerate(zip(result.ranked_features, result.ranked_scores), start=1):
                rows.append({
                    "dataset": dataset_name,
                    "method": result.method,
                    "family": result.family,
                    "rank": rank,
                    "feature": feat,
                    "score": score,
                    "runtime_seconds": result.runtime_seconds,
                    "params_json": json.dumps(result.params),
                    "status": "ok",
                    "error": "",
                })
        except Exception as exc:
            print(f"[WARN] Feature selector failed: {dataset_name}/{method}: {exc}")
            rows.append({
                "dataset": dataset_name,
                "method": method,
                "family": "unknown",
                "rank": np.nan,
                "feature": "",
                "score": np.nan,
                "runtime_seconds": np.nan,
                "params_json": "{}",
                "status": "failed",
                "error": str(exc),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{dataset_name}_feature_selection_rankings.csv", index=False)
    return df


def expand_method_list(config: dict, mode: str) -> List[str]:
    fs_cfg = config["feature_selection"]
    if mode == "smoke":
        return fs_cfg.get("smoke_methods", ["pearson", "rf"])
    if mode == "focused":
        return fs_cfg.get("focused_methods", ["pearson", "mutual_info", "rf"])
    methods = []
    for family_methods in fs_cfg["methods"].values():
        methods.extend(family_methods)
    return methods


def determine_subset_size(scores: List[float], criterion: str, min_features: int = 3) -> int:
    n = len(scores)
    if n == 0:
        return 0
    if criterion == "q25":
        return min(n, max(min_features, int(math.ceil(0.25 * n))))
    if criterion == "q50":
        return min(n, max(min_features, int(math.ceil(0.50 * n))))
    if criterion == "q75":
        return min(n, max(min_features, int(math.ceil(0.75 * n))))
    if criterion == "elbow":
        arr = np.asarray(scores, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        if len(arr) <= min_features:
            return len(arr)
        # Normalize rank and score, then pick the maximum distance from the straight line.
        x = np.linspace(0, 1, len(arr))
        y = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
        line = np.linspace(y[0], y[-1], len(arr))
        dist = np.abs(y - line)
        k = int(np.argmax(dist) + 1)
        return min(n, max(min_features, k))
    raise ValueError(f"Unsupported subset criterion: {criterion}")


def generate_feature_subsets(rankings: pd.DataFrame, criteria: List[str], min_features: int, output_dir: str) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    rows = []
    grouped = rankings[rankings["status"] == "ok"].sort_values(["dataset", "method", "rank"]).groupby(["dataset", "method", "family"])
    for (dataset, method, family), g in grouped:
        features = g["feature"].astype(str).tolist()
        scores = g["score"].astype(float).tolist()
        for criterion in criteria:
            k = determine_subset_size(scores, criterion, min_features=min_features)
            selected = features[:k]
            rows.append({
                "dataset": dataset,
                "method": method,
                "family": family,
                "criterion": criterion,
                "feature_set_id": f"{dataset}__{method}__{criterion}__k{k}",
                "num_features": k,
                "selected_features": ";".join(selected),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "all_candidate_feature_sets.csv", index=False)
    return df
