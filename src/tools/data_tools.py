"""
src/tools/data_tools.py

Reusable, Scikit-Learn-compatible cleaning transformers.

Each transformer follows the standard sklearn Transformer interface
(fit / transform), which means:
  - They can be dropped into a `sklearn.pipeline.Pipeline` directly
  - They are individually unit-testable in isolation
  - `fit` learns any statistics needed (e.g. median values) ONLY from the
    data passed to it -- callers are responsible for fitting only on
    training data and calling `transform` (not `fit_transform`) on
    validation/test data, which is what prevents data leakage across
    folds later in the ML Agent stage.

These transformers only DEPEND on pandas/numpy/sklearn -- no LLM calls
happen here. That split is intentional: deterministic, testable data
manipulation stays in plain code; the LLM is reserved for the agent
"brain" (see src/agents/cleaner.py) that decides *when* and *how* to
apply these tools and interprets the results.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class DuplicateRemover(BaseEstimator, TransformerMixin):
    """Drops exact duplicate rows.

    Parameters
    ----------
    subset : list[str] | None
        Columns to consider when identifying duplicates. If None, all
        columns are used.
    """

    def __init__(self, subset: Optional[list[str]] = None):
        self.subset = subset
        self.n_duplicates_removed_: int = 0

    def fit(self, X: pd.DataFrame, y=None) -> "DuplicateRemover":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        before = len(X)
        X_clean = X.drop_duplicates(subset=self.subset).copy()
        self.n_duplicates_removed_ = before - len(X_clean)
        return X_clean


DEFAULT_PLACEHOLDER_VALUES = ["unknown", "error", "n/a", "na", "none", "null", "-", "--", ""]


class PlaceholderNullNormalizer(BaseEstimator, TransformerMixin):
    """Converts placeholder "fake null" strings (e.g. "UNKNOWN", "ERROR",
    "N/A") into real NaN so that MissingValueImputer -- which only
    recognizes true NaN/null -- picks them up correctly instead of
    treating them as legitimate categories.

    Only object/string-dtype columns are touched; numeric columns are
    left alone. Matching is case-insensitive and whitespace-stripped, but
    requires an EXACT match against `placeholder_values` -- a value is
    never treated as a placeholder just because it contains one as a
    substring, so genuine short values (e.g. a real category that happens
    to render as "-") are only swapped when they equal a listed
    placeholder exactly.
    """

    def __init__(self, placeholder_values: Optional[list[str]] = None):
        self.placeholder_values = placeholder_values
        self.placeholder_counts_: dict = {}

    def fit(self, X: pd.DataFrame, y=None) -> "PlaceholderNullNormalizer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = self.placeholder_values if self.placeholder_values is not None else DEFAULT_PLACEHOLDER_VALUES
        normalized_placeholders = {str(v).strip().lower() for v in values}

        X_clean = X.copy()
        self.placeholder_counts_ = {}

        for col in X_clean.select_dtypes(exclude=[np.number]).columns:
            is_placeholder = X_clean[col].apply(
                lambda v: isinstance(v, str) and v.strip().lower() in normalized_placeholders
            )
            count = int(is_placeholder.sum())
            if count > 0:
                X_clean.loc[is_placeholder, col] = np.nan
                self.placeholder_counts_[col] = count

        return X_clean


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values: median for numeric columns, mode for
    categorical columns.

    Statistics are learned in `fit` and stored, so the SAME values learned
    on training data get applied via `transform` on validation/test data
    -- this is the mechanism that prevents imputation-based data leakage.

    Columns whose missing-value fraction exceeds `high_missing_threshold`
    are flagged (not silently dropped) via `high_missing_columns_`, so the
    calling agent can make an explicit decision about them.
    """

    def __init__(self, high_missing_threshold: float = 0.5):
        self.high_missing_threshold = high_missing_threshold
        self.numeric_fill_values_: dict = {}
        self.categorical_fill_values_: dict = {}
        self.high_missing_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "MissingValueImputer":
        missing_frac = X.isnull().mean()
        self.high_missing_columns_ = missing_frac[
            missing_frac > self.high_missing_threshold
        ].index.tolist()

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns

        for col in numeric_cols:
            self.numeric_fill_values_[col] = X[col].median()

        for col in categorical_cols:
            mode = X[col].mode()
            self.categorical_fill_values_[col] = mode.iloc[0] if not mode.empty else "unknown"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()
        for col, fill_value in self.numeric_fill_values_.items():
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(fill_value)
        for col, fill_value in self.categorical_fill_values_.items():
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(fill_value)
        return X_clean


class DataTypeCorrector(BaseEstimator, TransformerMixin):
    """Attempts to coerce object-dtype columns that look numeric (e.g. a
    money column stored as text with occasional blanks) into proper
    numeric dtype. Columns that fail to convert are left untouched.
    """

    def __init__(self, columns_to_check: Optional[list[str]] = None):
        self.columns_to_check = columns_to_check
        self.converted_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "DataTypeCorrector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()
        # Use exclude=[number] rather than include="object" -- pandas 3.x
        # introduced a distinct "str" dtype separate from "object", so
        # relying on include="object" alone silently misses string columns
        # on newer pandas versions.
        cols = self.columns_to_check or X_clean.select_dtypes(exclude=[np.number]).columns

        self.converted_columns_ = []
        for col in cols:
            if col not in X_clean.columns:
                continue
            # Treat blank/whitespace-only strings as already-missing, not
            # as "real" values we'd be destroying by converting -- a blank
            # cell in a numeric-looking column is a missing value, not a
            # genuine non-numeric entry.
            stripped = X_clean[col].astype(str).str.strip()
            non_blank_mask = ~stripped.isin(["", "nan", "None", "NaN"])
            non_null_before = non_blank_mask.sum()

            coerced = pd.to_numeric(X_clean[col], errors="coerce")
            non_null_after = coerced.notna().sum()

            if non_null_before > 0 and (non_null_after / non_null_before) > 0.9:
                X_clean[col] = coerced
                self.converted_columns_.append(col)
        return X_clean


class LowVarianceColumnDropper(BaseEstimator, TransformerMixin):
    """Flags and drops columns where a single value accounts for more
    than `threshold` fraction of all rows (near-constant columns carry
    almost no predictive signal and add noise to downstream models).
    """

    def __init__(self, threshold: float = 0.99):
        self.threshold = threshold
        self.dropped_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "LowVarianceColumnDropper":
        self.dropped_columns_ = []
        for col in X.columns:
            top_value_frac = X[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_value_frac >= self.threshold:
                self.dropped_columns_.append(col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_present = [c for c in self.dropped_columns_ if c in X.columns]
        return X.drop(columns=cols_present)
