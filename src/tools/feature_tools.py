"""
src/tools/feature_tools.py

Reusable, Scikit-Learn-compatible feature engineering transformers.

Each transformer follows the standard sklearn interface (fit / transform):
  - fit learns statistics ONLY from the data passed to it.
  - transform applies those statistics to new data without recomputing them.
  - They compose cleanly inside a sklearn Pipeline.

No LLM calls happen here.  The FeatureEngineeringAgent (src/agents/
feature_engineer.py) decides which transformers to chain and interprets
the results; these classes are pure, deterministic pandas/numpy computation.

PROTECTED_COLS are columns that must never be modified by any transformer:
  - "is_late_delivery" — the prediction target
  - "order_id"         — a row identifier with no predictive meaning
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

PROTECTED_COLS: frozenset[str] = frozenset(["is_late_delivery", "order_id"])


# ---------------------------------------------------------------------------
# RedundantFeatureDropper
# ---------------------------------------------------------------------------

class RedundantFeatureDropper(BaseEstimator, TransformerMixin):
    """Drops one column from any pair of numeric columns whose absolute
    Pearson correlation exceeds `threshold`.

    For each correlated pair, the first-encountered column is kept and the
    second is dropped.  Protected columns are never candidates for dropping,
    even if they are highly correlated with another column.

    Parameters
    ----------
    threshold : float
        Absolute correlation above which a column is considered redundant.
    protected_cols : list[str] | None
        Columns exempt from dropping. Defaults to PROTECTED_COLS.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        protected_cols: Optional[list[str]] = None,
    ):
        self.threshold = threshold
        self.protected_cols = protected_cols
        self.dropped_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "RedundantFeatureDropper":
        protected = set(self.protected_cols if self.protected_cols is not None else PROTECTED_COLS)
        numeric_cols = [
            c for c in X.select_dtypes(include=[np.number]).columns
            if c not in protected
        ]

        self.dropped_columns_ = []
        if len(numeric_cols) < 2:
            return self

        corr = X[numeric_cols].corr(method="pearson").abs()
        to_drop: set[str] = set()

        for i, col_a in enumerate(numeric_cols):
            if col_a in to_drop:
                continue
            for col_b in numeric_cols[i + 1:]:
                if col_b in to_drop:
                    continue
                if corr.at[col_a, col_b] > self.threshold:
                    to_drop.add(col_b)
                    logger.info(
                        "RedundantFeatureDropper: dropping '%s' (corr with '%s' = %.4f)",
                        col_b, col_a, corr.at[col_a, col_b],
                    )

        self.dropped_columns_ = list(to_drop)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_present = [c for c in self.dropped_columns_ if c in X.columns]
        return X.drop(columns=cols_present)


# ---------------------------------------------------------------------------
# SkewnessReducer
# ---------------------------------------------------------------------------

class SkewnessReducer(BaseEstimator, TransformerMixin):
    """Applies np.log1p to numeric columns whose absolute skewness exceeds
    `skew_threshold`, provided the column minimum is >= 0.

    Columns with negative values are skipped: log1p is undefined for
    x < -1 and semantically misleading for mixed-sign distributions.
    Protected columns are never transformed.

    Parameters
    ----------
    skew_threshold : float
        Minimum |skewness| that triggers log1p transformation.
    protected_cols : list[str] | None
        Columns exempt from transformation. Defaults to PROTECTED_COLS.
    """

    def __init__(
        self,
        skew_threshold: float = 1.0,
        protected_cols: Optional[list[str]] = None,
    ):
        self.skew_threshold = skew_threshold
        self.protected_cols = protected_cols
        self.log_transformed_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "SkewnessReducer":
        protected = set(self.protected_cols if self.protected_cols is not None else PROTECTED_COLS)
        numeric_cols = [
            c for c in X.select_dtypes(include=[np.number]).columns
            if c not in protected
        ]

        self.log_transformed_columns_ = []
        for col in numeric_cols:
            skew = X[col].skew()
            col_min = X[col].min()
            if abs(skew) > self.skew_threshold and col_min >= 0:
                self.log_transformed_columns_.append(col)
                logger.info(
                    "SkewnessReducer: will log1p-transform '%s' (skew=%.3f, min=%.3f)",
                    col, skew, col_min,
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.log_transformed_columns_:
            if col in X_out.columns:
                X_out[col] = np.log1p(X_out[col])
        return X_out


# ---------------------------------------------------------------------------
# NumericScaler
# ---------------------------------------------------------------------------

class NumericScaler(BaseEstimator, TransformerMixin):
    """Applies StandardScaler-style (zero-mean, unit-variance) normalization.

    Mean and std are learned exclusively on fit data and stored, so the
    same statistics are applied during transform — preventing leakage when
    used in a train/test split.  Columns with zero variance are skipped
    (to avoid division-by-zero) and recorded in `zero_variance_columns_`.

    Protected columns are never scaled.

    Parameters
    ----------
    protected_cols : list[str] | None
        Columns exempt from scaling. Defaults to PROTECTED_COLS.
    """

    def __init__(self, protected_cols: Optional[list[str]] = None):
        self.protected_cols = protected_cols
        self.scale_stats_: dict[str, dict[str, float]] = {}
        self.scaled_columns_: list[str] = []
        self.zero_variance_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "NumericScaler":
        protected = set(self.protected_cols if self.protected_cols is not None else PROTECTED_COLS)
        numeric_cols = [
            c for c in X.select_dtypes(include=[np.number]).columns
            if c not in protected
        ]

        self.scale_stats_ = {}
        self.scaled_columns_ = []
        self.zero_variance_columns_ = []

        for col in numeric_cols:
            mean = float(X[col].mean())
            std = float(X[col].std())
            if std == 0.0:
                self.zero_variance_columns_.append(col)
                logger.info("NumericScaler: skipping '%s' (zero variance)", col)
            else:
                self.scale_stats_[col] = {"mean": mean, "std": std}
                self.scaled_columns_.append(col)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, stats in self.scale_stats_.items():
            if col in X_out.columns:
                X_out[col] = (X_out[col] - stats["mean"]) / stats["std"]
        return X_out


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns using one of two strategies:

    - **One-hot encoding**: for columns with cardinality < `ohe_threshold`.
      Categories unseen during fit are silently ignored in transform (no
      extra dummy columns are created; new-category rows get all zeros for
      that column's dummies).
    - **Frequency encoding**: for columns with cardinality >= `ohe_threshold`.
      Each category value is replaced with the proportion (count / n_rows)
      of its appearances in the fit data, landing in [0, 1] rather than raw
      counts — otherwise high-cardinality columns end up on a wildly larger
      scale than every other feature, which blows up models sensitive to
      feature scale (e.g. LogisticRegression). Values unseen during fit are
      mapped to 0.

    The original categorical column is dropped and replaced by its encoded
    form.  Protected columns are never encoded.

    Parameters
    ----------
    ohe_threshold : int
        Cardinality below which one-hot encoding is used.
    protected_cols : list[str] | None
        Columns exempt from encoding. Defaults to PROTECTED_COLS.
    """

    def __init__(
        self,
        ohe_threshold: int = 20,
        protected_cols: Optional[list[str]] = None,
    ):
        self.ohe_threshold = ohe_threshold
        self.protected_cols = protected_cols
        self.ohe_columns_: dict[str, list[str]] = {}   # col -> list of dummy col names
        self.freq_columns_: dict[str, dict] = {}        # col -> {value: count}
        self.encoding_map_: dict[str, str] = {}         # col -> "one_hot" | "frequency"

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalEncoder":
        protected = set(self.protected_cols if self.protected_cols is not None else PROTECTED_COLS)
        cat_cols = [
            c for c in X.select_dtypes(exclude=[np.number]).columns
            if c not in protected
        ]

        self.ohe_columns_ = {}
        self.freq_columns_ = {}
        self.encoding_map_ = {}

        for col in cat_cols:
            n_unique = X[col].nunique(dropna=True)
            if n_unique < self.ohe_threshold:
                filled = X[col].fillna("__unknown__")
                dummies = pd.get_dummies(filled, prefix=col, dtype=int)
                self.ohe_columns_[col] = list(dummies.columns)
                self.encoding_map_[col] = "one_hot"
                logger.info(
                    "CategoricalEncoder: '%s' → one_hot (%d unique values)", col, n_unique
                )
            else:
                self.freq_columns_[col] = (X[col].value_counts() / len(X)).to_dict()
                self.encoding_map_[col] = "frequency"
                logger.info(
                    "CategoricalEncoder: '%s' → frequency (%d unique values)", col, n_unique
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        for col, dummy_cols in self.ohe_columns_.items():
            if col not in X_out.columns:
                continue
            filled = X_out[col].fillna("__unknown__")
            dummies = pd.get_dummies(filled, prefix=col, dtype=int)
            # Align to the exact columns seen at fit time; fill any gaps with 0
            dummies = dummies.reindex(columns=dummy_cols, fill_value=0)
            X_out = X_out.drop(columns=[col])
            X_out = pd.concat([X_out, dummies], axis=1)

        for col, freq_map in self.freq_columns_.items():
            if col not in X_out.columns:
                continue
            X_out[col] = X_out[col].map(freq_map).fillna(0.0).astype(float)

        return X_out
