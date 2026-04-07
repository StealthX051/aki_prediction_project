"""Fold-local preprocessing utilities for Catch22 model training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


RAW_CATEGORICAL_PREOP_COLUMNS: Sequence[str] = (
    "sex",
    "emop",
    "department",
    "approach",
    "asa",
    "optype",
    "ane_type",
    "preop_htn",
    "preop_dm",
    "preop_ecg",
    "preop_pft",
)

CONTINUOUS_PREOP_COLUMNS: Sequence[str] = (
    "age",
    "height",
    "weight",
    "bmi",
    "preop_hb",
    "preop_plt",
    "preop_pt",
    "preop_aptt",
    "preop_na",
    "preop_k",
    "preop_gluc",
    "preop_alb",
    "preop_ast",
    "preop_alt",
    "preop_bun",
    "preop_cr",
    "preop_hco3",
    "preop_ph",
    "preop_be",
    "preop_pao2",
    "preop_paco2",
    "preop_sao2",
    "preop_wbc",
    "preop_crp",
    "preop_lac",
    "preop_egfr_ckdepi_2021",
)

MISSING_CATEGORY = "__missing__"
OTHER_CATEGORY = "__other__"


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=float)


@dataclass
class FoldPreprocessor:
    """Train-partition-only preprocessing for preoperative + waveform features."""

    impute_missing: bool = False
    rare_category_columns: Sequence[str] = ("department",)
    rare_category_min_count: int = 30
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    categorical_columns_: List[str] = field(default_factory=list, init=False)
    numeric_columns_: List[str] = field(default_factory=list, init=False)
    feature_names_: List[str] = field(default_factory=list, init=False)
    rare_mappings_: Dict[str, set] = field(default_factory=dict, init=False)
    clip_bounds_: Dict[str, tuple[float, float]] = field(default_factory=dict, init=False)
    numeric_impute_values_: Dict[str, float] = field(default_factory=dict, init=False)
    encoder_: Optional[OneHotEncoder] = field(default=None, init=False)

    def fit(self, X: pd.DataFrame) -> "FoldPreprocessor":
        X_df = X.copy()
        candidate_categoricals = [
            col for col in RAW_CATEGORICAL_PREOP_COLUMNS if col in X_df.columns
        ]
        inferred_categoricals = [
            col
            for col in X_df.columns
            if pd.api.types.is_object_dtype(X_df[col])
            or isinstance(X_df[col].dtype, pd.CategoricalDtype)
        ]
        self.categorical_columns_ = list(dict.fromkeys(candidate_categoricals + inferred_categoricals))
        self.numeric_columns_ = [col for col in X_df.columns if col not in self.categorical_columns_]

        self.rare_mappings_.clear()
        for col in self.categorical_columns_:
            values = self._prepare_categorical_series(X_df[col])
            if col in self.rare_category_columns:
                counts = values.value_counts()
                rare_values = set(counts[counts < self.rare_category_min_count].index.tolist())
                if rare_values:
                    self.rare_mappings_[col] = rare_values

        self.clip_bounds_.clear()
        self.numeric_impute_values_.clear()
        for col in self.numeric_columns_:
            numeric = pd.to_numeric(X_df[col], errors="coerce")
            if col in CONTINUOUS_PREOP_COLUMNS:
                clean = numeric.dropna()
                if not clean.empty:
                    lower = float(clean.quantile(self.lower_quantile))
                    upper = float(clean.quantile(self.upper_quantile))
                    if lower > upper:
                        lower, upper = upper, lower
                    self.clip_bounds_[col] = (lower, upper)
            if self.impute_missing:
                clean = numeric.dropna()
                self.numeric_impute_values_[col] = float(clean.median()) if not clean.empty else 0.0

        if self.categorical_columns_:
            self.encoder_ = _make_one_hot_encoder()
            cat_train = self._transform_categorical_frame(X_df[self.categorical_columns_])
            self.encoder_.fit(cat_train)
            encoded_names = self.encoder_.get_feature_names_out(self.categorical_columns_).tolist()
        else:
            self.encoder_ = None
            encoded_names = []

        self.feature_names_ = list(self.numeric_columns_) + encoded_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        if self.numeric_columns_:
            numeric_data: Dict[str, pd.Series] = {}
            for col in self.numeric_columns_:
                numeric = pd.to_numeric(X_df[col], errors="coerce")
                if col in self.clip_bounds_:
                    lower, upper = self.clip_bounds_[col]
                    numeric = numeric.clip(lower=lower, upper=upper)
                if self.impute_missing:
                    numeric = numeric.fillna(self.numeric_impute_values_.get(col, 0.0))
                numeric_data[col] = numeric.astype(float)
            numeric_df = pd.DataFrame(numeric_data, index=X_df.index)
        else:
            numeric_df = pd.DataFrame(index=X_df.index)

        if self.categorical_columns_:
            if self.encoder_ is None:
                raise ValueError("Categorical encoder has not been fitted.")
            cat_df = self._transform_categorical_frame(X_df[self.categorical_columns_])
            encoded = self.encoder_.transform(cat_df)
            encoded_df = pd.DataFrame(
                encoded,
                index=X_df.index,
                columns=self.encoder_.get_feature_names_out(self.categorical_columns_),
            )
        else:
            encoded_df = pd.DataFrame(index=X_df.index)

        transformed = pd.concat([numeric_df, encoded_df], axis=1)
        transformed = transformed.reindex(columns=self.feature_names_, fill_value=0.0)
        return transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def _prepare_categorical_series(self, series: pd.Series) -> pd.Series:
        values = series.copy()
        values = values.astype("string")
        values = values.fillna(MISSING_CATEGORY)
        return values.astype(str)

    def _transform_categorical_frame(self, X_cat: pd.DataFrame) -> pd.DataFrame:
        transformed: Dict[str, pd.Series] = {}
        for col in self.categorical_columns_:
            values = self._prepare_categorical_series(X_cat[col])
            rare_values = self.rare_mappings_.get(col)
            if rare_values:
                values = values.where(~values.isin(rare_values), OTHER_CATEGORY)
            transformed[col] = values
        return pd.DataFrame(transformed, index=X_cat.index)


def get_raw_categorical_columns(columns: Iterable[str]) -> List[str]:
    return [col for col in RAW_CATEGORICAL_PREOP_COLUMNS if col in columns]
