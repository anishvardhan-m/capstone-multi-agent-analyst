"""
src/tools/ml_tools.py

Utility functions shared across the ML Agent pipeline.

Kept deliberately small: one pure function for task-type detection and
one for computing adjusted R², both with no side-effects so they are
trivially unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_task_type(
    y: pd.Series,
    multiclass_unique_threshold: int = 20,
) -> str:
    """Infer whether a target column requires binary classification,
    multiclass classification, or regression.

    Decision rules (applied in order):
    1. Exactly 2 unique non-null values → "binary_classification"
    2. Integer or object dtype AND unique count <= multiclass_unique_threshold
       → "multiclass_classification"
    3. Anything else (continuous numeric) → "regression"

    Parameters
    ----------
    y : pd.Series
        The target column.
    multiclass_unique_threshold : int
        Maximum number of distinct values before a column is treated as
        continuous regression rather than multiclass classification.

    Returns
    -------
    "binary_classification" | "multiclass_classification" | "regression"
    """
    n_unique = y.nunique(dropna=True)

    if n_unique == 2:
        return "binary_classification"

    is_integer_like = pd.api.types.is_integer_dtype(y) or pd.api.types.is_object_dtype(y)
    if is_integer_like and n_unique <= multiclass_unique_threshold:
        return "multiclass_classification"

    return "regression"


def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """Compute adjusted R² from plain R², sample count, and feature count.

    Returns NaN when the denominator would be zero (n_samples <= n_features + 1).
    """
    denom = n_samples - n_features - 1
    if denom <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n_samples - 1) / denom
