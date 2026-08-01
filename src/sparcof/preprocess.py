from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


UNSW_DROP_COLS = ["srcip", "sport", "dstip", "dsport"]


@dataclass
class PreprocessArtifacts:
    feature_names: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    label_mapping: Dict[str, int]


class TabularPreprocessor:
    """Leakage-safe train-fitted preprocessor.

    The feature list is applied to raw columns after dataset-specific leakage/identifier columns are removed.
    Encoders, imputers, and scalers are fitted only on the training/CV partition and reused for test data.
    """

    def __init__(
        self,
        dataset_name: str,
        target_col: str,
        selected_features: Optional[List[str]] = None,
        dataset_config: Optional[Mapping[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        self.target_col = target_col
        self.selected_features = selected_features
        self.dataset_config = dict(dataset_config or {})
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names_: List[str] = []
        self.numeric_features_: List[str] = []
        self.categorical_features_: List[str] = []

    def _clean_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        configured_drop = []
        for key in ("drop_columns", "identifier_columns", "leakage_columns"):
            configured_drop.extend(str(value) for value in self.dataset_config.get(key, []))
        configured_drop = [column for column in dict.fromkeys(configured_drop) if column in df.columns]
        if self.target_col in configured_drop:
            raise ValueError(f"Target column {self.target_col!r} cannot also be configured as a dropped column.")
        if configured_drop:
            df = df.drop(columns=configured_drop)

        missing_tokens = self.dataset_config.get("missing_value_tokens", [])
        if missing_tokens:
            df = df.replace(list(missing_tokens), np.nan)

        # Backward-compatible handling for the original UNSW-NB15 experiment.
        if self.dataset_name == "unsw":
            if self.target_col == "label":
                if "attack_cat" in df.columns:
                    df = df.drop(columns=["attack_cat"])
            else:
                if "label" in df.columns:
                    df = df.drop(columns=["label"])
            drop = [c for c in UNSW_DROP_COLS if c in df.columns]
            df = df.drop(columns=drop)
            if "ct_flw_http_mthd" in df.columns:
                df["ct_flw_http_mthd"] = df["ct_flw_http_mthd"].fillna(0)
            if "is_ftp_login" in df.columns:
                df["is_ftp_login"] = df["is_ftp_login"].fillna(0).astype(int)
            if "service" in df.columns:
                df["service"] = df["service"].replace("-", "service_unknown")
            if "attack_cat" in df.columns:
                df["attack_cat"] = df["attack_cat"].fillna("normal").astype(str).str.strip().str.lower().str.replace("backdoors", "backdoor", regex=False)
        return df

    def _split_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = self._clean_frame(df)
        if self.target_col not in df.columns:
            raise ValueError(f"Target column {self.target_col!r} not found. Available: {list(df.columns)[:10]}...")
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])
        if self.selected_features:
            missing = [f for f in self.selected_features if f not in X.columns]
            if missing:
                raise ValueError(
                    f"Selected features not found in {self.dataset_name}: {missing[:20]} "
                    f"(showing up to 20). Check spelling/capitalization."
                )
            X = X[self.selected_features]
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, PreprocessArtifacts]:
        X, y = self._split_X_y(df)
        X = X.replace([np.inf, -np.inf], np.nan)
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = [c for c in X.columns if c not in self.numeric_features_]
        # The output matrix is concatenated in this exact order. Keeping the
        # feature-name vector aligned prevents selected columns from silently
        # pointing at the wrong transformed feature when raw types are interleaved.
        self.feature_names_ = self.numeric_features_ + self.categorical_features_

        blocks = []
        if self.numeric_features_:
            X_num = X[self.numeric_features_].apply(pd.to_numeric, errors="coerce")
            X_num_imp = self.numeric_imputer.fit_transform(X_num)
            X_num_scaled = self.scaler.fit_transform(X_num_imp)
            blocks.append(X_num_scaled)
        if self.categorical_features_:
            X_cat = X[self.categorical_features_].apply(
                lambda column: column.map(lambda value: str(value) if pd.notna(value) else np.nan)
            )
            X_cat_imp = self.categorical_imputer.fit_transform(X_cat)
            X_cat_enc = self.categorical_encoder.fit_transform(X_cat_imp)
            blocks.append(X_cat_enc)

        X_out = np.hstack(blocks).astype(np.float32) if blocks else np.empty((len(df), 0), dtype=np.float32)
        y_out = self.label_encoder.fit_transform(y.astype(str))
        label_mapping = {str(label): int(i) for i, label in enumerate(self.label_encoder.classes_)}
        artifacts = PreprocessArtifacts(
            self.feature_names_, self.numeric_features_, self.categorical_features_, label_mapping
        )
        return X_out, y_out, artifacts

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self._split_X_y(df)
        X = X[self.feature_names_]
        X = X.replace([np.inf, -np.inf], np.nan)
        blocks = []
        if self.numeric_features_:
            X_num = X[self.numeric_features_].apply(pd.to_numeric, errors="coerce")
            X_num_imp = self.numeric_imputer.transform(X_num)
            X_num_scaled = self.scaler.transform(X_num_imp)
            blocks.append(X_num_scaled)
        if self.categorical_features_:
            X_cat = X[self.categorical_features_].apply(
                lambda column: column.map(lambda value: str(value) if pd.notna(value) else np.nan)
            )
            X_cat_imp = self.categorical_imputer.transform(X_cat)
            X_cat_enc = self.categorical_encoder.transform(X_cat_imp)
            blocks.append(X_cat_enc)
        X_out = np.hstack(blocks).astype(np.float32) if blocks else np.empty((len(df), 0), dtype=np.float32)
        y_out = self.label_encoder.transform(y.astype(str))
        return X_out, y_out
