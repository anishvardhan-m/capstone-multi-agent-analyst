"""
src/agents/ml_agent.py

Machine Learning Agent.

Responsibilities (per the capstone handbook, Section 10.1): detect the
modelling task type automatically from the target column, compare a suite
of candidate models via cross-validation, refit the winner on the full
training set, evaluate once on a held-out test set, serialize the best
model to disk, and produce a structured, auditable report.

Design note: like the preceding agents, no LLM calls happen here.
All logic is deterministic sklearn code. Task-type detection and metric
helpers live in src/tools/ml_tools.py to keep this file focused on
orchestration.

Class imbalance strategy
------------------------
All classification models use class_weight="balanced" to prevent the
majority-class collapse seen with default weights on skewed targets like
is_late_delivery (8.1% positive rate).

For LogisticRegression and RandomForestClassifier, class_weight is a
native constructor parameter. For HistGradientBoostingClassifier, the
parameter was added in sklearn 1.2; in older environments the agent falls
back to computing sample weights via compute_sample_weight("balanced",
y_train) and passing them to .fit() directly — both at CV time and at
the final refit.

Threshold analysis
------------------
For binary classification, after fitting the best model the agent sweeps
three decision thresholds (0.3, 0.4, 0.5) on predict_proba output and
records precision / recall / F1 / confusion matrix at each one, exposing
the precision-recall tradeoff explicitly in the report.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    check_scoring,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.tools.audit_db import audit_logged, log_ml_experiment
from src.tools.logging_config import get_agent_logger
from src.tools.ml_tools import adjusted_r2, detect_task_type, expected_calibration_error

logger = get_agent_logger("MLAgent")

_RANDOM_STATE = 42
_N_CV_SPLITS = 5
_THRESHOLD_SWEEP = (0.3, 0.4, 0.5)
_PERMUTATION_N_REPEATS = 10
_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
)
_CALIBRATION_N_BINS = 10
# ECE above this is treated as "meaningfully miscalibrated" -- common
# rule-of-thumb cutoff in the calibration literature (e.g. Guo et al. 2017
# report well-calibrated modern classifiers well under this).
_ECE_MISCALIBRATION_THRESHOLD = 0.05
_CLASS_WEIGHT_CALIBRATION_CAVEAT = (
    "This model uses class_weight='balanced' to handle class imbalance, "
    "which systematically shifts predict_proba outputs away from true "
    "probabilities -- exactly what a calibration curve measures. Treat a "
    "decision threshold (e.g. threshold=0.3 in the threshold sweep) as a "
    "precision/recall cutoff chosen for a business tradeoff, not a literal "
    "percentage chance of the outcome."
)

# Error analysis by segment (handbook F6). A column qualifies as a segment
# axis when its distinct-value count falls in this range -- the generic
# post-encoding signature of a categorical column (one-hot flags, or a
# frequency/label-encoded column with ~one value per original category),
# regardless of what the dataset or its raw column names are. The upper
# bound is set to 50 (not lower) deliberately: common real-world
# categoricals like a US state or a product category can easily carry
# 20-50 distinct values, and excluding those would silently drop exactly
# the segments a business reader most wants to see, in favor of
# low-cardinality count-like columns that happen to be more "categorical"
# by the letter of this heuristic but far less actionable in practice.
_ERROR_ANALYSIS_MIN_CATEGORIES = 2
_ERROR_ANALYSIS_MAX_CATEGORIES = 50
_ERROR_ANALYSIS_MAX_SEGMENT_COLS = 5
# A segment needs at least this many rows (or, for a rate's denominator
# class, this many members of that class) before its rate/MAE is trusted.
_ERROR_ANALYSIS_MIN_SEGMENT_SIZE = 20
# A segment's error metric is flagged "elevated" only once it clears BOTH
# a proportional margin (so a high overall rate needs a proportionally
# bigger gap to count) and, for the bounded [0, 1] rate metrics, an
# absolute floor (so a near-zero overall rate can't be "elevated" by a
# proportionally huge but practically meaningless jump).
_ERROR_ANALYSIS_REL_MARGIN = 0.5
_ERROR_ANALYSIS_ABS_MARGIN = 0.10

# Robustness check across random seeds (handbook F7). Re-runs the entire
# pipeline -- split, model selection, refit, permutation importance --
# under each seed below (42 first, since it's this project's standing
# default everywhere else) to check whether the headline story (which
# model wins, how good it is, which features matter) holds up, or was an
# artifact of one particular seed.
_ROBUSTNESS_DEFAULT_SEEDS: tuple[int, ...] = (42, 7, 123, 2024, 99)
_ROBUSTNESS_TOP_K_FEATURES = 5

# Hyperparameter grids swept via GridSearchCV (handbook Sections 6.5, 10.2).
# LinearRegression has no entry — it has no hyperparameters worth tuning,
# so it's cross-validated directly instead of wrapped in GridSearchCV.
_PARAM_GRIDS: dict[str, dict] = {
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
    "RandomForestClassifier": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    "RandomForestRegressor": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    "HistGradientBoostingClassifier": {"max_iter": [100, 200], "learning_rate": [0.05, 0.1]},
    "HistGradientBoostingRegressor": {"max_iter": [100, 200], "learning_rate": [0.05, 0.1]},
}

# Runtime detection: class_weight was added to HGBT in sklearn 1.2.
# Falls back to sample_weight path on older environments.
_HGBT_SUPPORTS_CLASS_WEIGHT = (
    "class_weight" in HistGradientBoostingClassifier._get_param_names()
)
logger.info(
    "HistGradientBoostingClassifier class_weight support: %s",
    _HGBT_SUPPORTS_CLASS_WEIGHT,
)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class MLReport:
    """Structured, JSON-serializable summary of the ML training run.

    cv_scores : dict
        Maps model name → best CV score from GridSearchCV (higher-is-
        better: F1-macro for classification, negative-RMSE for
        regression). LinearRegression has no grid, so its score is a
        plain (untuned) cross-validation mean.
    best_hyperparameters : dict
        The winning model's best_params_ from GridSearchCV. Empty for
        models with no hyperparameter grid (LinearRegression).
    test_metrics : dict
        Held-out test-set metrics at the default 0.5 threshold.
    threshold_metrics : list[dict] | None
        Binary classification only. Precision / recall / F1 / confusion
        matrix at each threshold in _THRESHOLD_SWEEP, so the
        precision-recall tradeoff is explicit in the report.
    confusion_matrix : list[list[int]] | None
        Classification only. Corresponds to the default (0.5) threshold.
    feature_importances : dict
        Maps feature name -> {"importance_mean": float, "importance_std":
        float, "distinguishable_from_zero": bool}. Computed via
        permutation importance (_PERMUTATION_N_REPEATS reshuffles per
        feature) rather than model attributes (feature_importances_/
        coef_), so the same method applies no matter which model wins and
        every feature gets a genuine mean/std across repeats instead of
        one deterministic number. distinguishable_from_zero is False when
        the feature's [mean - std, mean + std] range spans zero -- i.e.
        its measured effect isn't distinguishable from no effect at that
        repeat count (handbook F5).
    error_analysis : dict
        Held-out errors broken down by segment (handbook F6): {"task_type",
        "segment_columns", "overall", "segments", "detection_note", "note"}.
        segment_columns is either caller-configured (MLAgent(segment_cols=
        [...])) or auto-detected purely from feature-matrix cardinality --
        see _detect_segment_columns -- so nothing here is hardcoded to any
        dataset's column names. "segments" maps each segment column to a
        list of per-value rows with false_negative_rate/false_positive_rate
        (binary classification), misclassification_rate (multiclass), or
        mae/mean_error (regression), each with an elevated_* flag marking
        segments whose error is concentrated well above the overall value.
    test_predictions : dict | None
        Regression only. {"actual": [...], "predicted": [...]} pairs from
        the held-out test set -- the exact predictions used to compute
        rmse/mae/r2 above. Downsampled to at most _MAX_TEST_PREDICTIONS
        points when the test set is larger (affects only how many points
        the Visualization Agent's actual-vs-predicted/residual scatter
        plots draw, not any reported metric).
    test_predictions_table : dict | None
        Per-record held-out predictions for the dashboard (generic across
        task types, populated for BOTH classification and regression --
        unlike test_predictions above, which is regression-only and feeds
        the Visualization Agent's scatter plots instead). Shape:
        {"task_type", "id_col", "columns", "rows", "total_test_rows",
        "sampled", "sample_size", "note"}. Each row in "rows" has a
        "row_id" (the id_col value for that test row when id_col was
        given and present in the data, otherwise that row's positional
        index in the input CSV) plus, for classification:
        "actual_label"/"predicted_label"/"confidence" (predict_proba of
        the predicted class, None if the model has no predict_proba) /
        "correct"; for regression: "actual_value"/"predicted_value"/
        "error"/"abs_error". Capped at _MAX_TEST_PREDICTIONS_TABLE rows --
        a fixed-seed random sample when the test set is larger, with
        "sampled"=True and a note recording it.
    split_strategy : str
        "grouped" when the train/test split was built with GroupShuffleSplit
        so no single group_col value (e.g. a repeat customer) appears in
        both splits; "row_random" for the plain stratified/random split
        used when no group_col is given.
    group_col : str | None
        The column grouped on, when split_strategy == "grouped".
    cv_fold_scores : dict
        Maps model name -> raw per-fold CV scores (at its winning
        hyperparameters), so the point estimate in cv_scores can be judged
        against its actual spread rather than taken at face value.
    cv_std : dict
        Maps model name -> standard deviation of cv_fold_scores.
    model_selection_note : str | None
        Populated when the top two candidates' CV scores are within their
        combined std of each other -- i.e. not distinguishable from CV
        noise -- so a reader doesn't mistake a close call for a clear win.
    nested_cv_score, nested_cv_std : float | None
        A de-biased CV estimate for the WINNING model only, from genuine
        nested cross-validation (hyperparameters re-tuned per outer fold).
        cv_scores/cv_std above come from the same folds used to pick the
        winning hyperparameters, which optimistically biases that number;
        nested_cv_score corrects for it. None when nested CV was skipped
        (see nested_cv_note) -- e.g. no hyperparameters were tuned for the
        winner, so there's no selection bias to correct.
    nested_cv_note : str
        Always populated: either what nested_cv_score means, or why it
        was skipped.
    calibration : dict | None
        Binary classification only. Reliability-diagram data for the raw
        model on the held-out test set: {"prob_true": [...], "prob_pred":
        [...], "bin_counts": [...], "brier_score": float, "ece": float}.
        None for regression/multiclass or if predict_proba is unavailable.
    calibrated_comparison : dict | None
        Present only when calibration["ece"] exceeded
        _ECE_MISCALIBRATION_THRESHOLD and a CalibratedClassifierCV
        correction was fit for comparison: same shape as `calibration`
        plus a "method" key ("sigmoid" or "isotonic"). Diagnostic only --
        never replaces the serialized production model.
    calibration_note : str
        Always populated for binary classification: explains what
        calibration/ece mean, the class_weight='balanced' probability
        caveat, and whether/why a calibrated comparison was fit.
    target_col : str
        Name of the column this run predicted. Recorded here (not just
        passed as a run() argument) so a report loaded later -- a
        different session, a fresh dashboard page load -- can still show
        what was predicted without guessing or falling back to a
        hardcoded default (handbook F4).
    """

    task_type: str
    best_model_name: str
    cv_scores: dict = field(default_factory=dict)
    best_hyperparameters: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    confusion_matrix: Optional[list] = None
    threshold_metrics: Optional[list] = None
    feature_importances: dict = field(default_factory=dict)
    error_analysis: dict = field(default_factory=dict)
    test_predictions: Optional[dict] = None
    test_predictions_table: Optional[dict] = None
    split_strategy: str = "row_random"
    group_col: Optional[str] = None
    cv_fold_scores: dict = field(default_factory=dict)
    cv_std: dict = field(default_factory=dict)
    model_selection_note: Optional[str] = None
    nested_cv_score: Optional[float] = None
    nested_cv_std: Optional[float] = None
    nested_cv_note: str = ""
    calibration: Optional[dict] = None
    calibrated_comparison: Optional[dict] = None
    calibration_note: str = ""
    target_col: str = ""

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "best_model_name": self.best_model_name,
            "target_col": self.target_col,
            "cv_scores": self.cv_scores,
            "best_hyperparameters": self.best_hyperparameters,
            "test_metrics": self.test_metrics,
            "confusion_matrix": self.confusion_matrix,
            "threshold_metrics": self.threshold_metrics,
            "feature_importances": self.feature_importances,
            "error_analysis": self.error_analysis,
            "test_predictions": self.test_predictions,
            "test_predictions_table": self.test_predictions_table,
            "split_strategy": self.split_strategy,
            "group_col": self.group_col,
            "cv_fold_scores": self.cv_fold_scores,
            "cv_std": self.cv_std,
            "model_selection_note": self.model_selection_note,
            "calibration": self.calibration,
            "calibrated_comparison": self.calibrated_comparison,
            "calibration_note": self.calibration_note,
            "nested_cv_score": self.nested_cv_score,
            "nested_cv_std": self.nested_cv_std,
            "nested_cv_note": self.nested_cv_note,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cv_with_sample_weight(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    scoring: str,
) -> np.ndarray:
    """Cross-validate a model using per-fold balanced sample weights.

    Used as a fallback when the model doesn't accept class_weight as a
    constructor parameter (older sklearn versions of HGBT).
    """
    scorer = check_scoring(model, scoring=scoring)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_tr = X_train.iloc[train_idx]
        y_fold_tr = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        sw = compute_sample_weight("balanced", y_fold_tr)
        fitted = clone(model)
        fitted.fit(X_fold_tr, y_fold_tr, sample_weight=sw)
        scores.append(scorer(fitted, X_fold_val, y_fold_val))

    return np.array(scores)


def _run_grid_search(
    models: list[tuple[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    scoring: str,
    sample_weight_models: Optional[set] = None,
) -> tuple[dict[str, float], dict[str, dict], dict[str, Any], dict[str, list], dict[str, float]]:
    """Tune each candidate model and return per-model best CV scores.

    Models with an entry in _PARAM_GRIDS are wrapped in GridSearchCV
    (refit=True), so .best_score_ drives model comparison and
    .best_estimator_ is already fit on the full training set. Models
    without a grid (LinearRegression) fall back to a plain CV score with
    no tuning, and are left unfit for the caller to refit if selected.

    Parameters
    ----------
    sample_weight_models : set[str] | None
        Model names that require the sample_weight CV path rather than
        the native class_weight path. The balanced sample weight is
        computed once on the full y_train and passed as a GridSearchCV
        fit_param, which sklearn slices per-fold automatically.

    Returns
    -------
    (cv_scores, best_params, fitted_estimators, cv_fold_scores, cv_std)
        fitted_estimators[name] is None for un-tuned models that still
        need to be refit by the caller. cv_fold_scores[name] is the raw
        per-fold score list at the winning hyperparameters (or the only
        configuration, for un-tuned models); cv_std[name] is its standard
        deviation -- both needed so the caller can judge whether two
        candidates' mean CV scores are actually distinguishable from CV
        noise, not just compare point estimates.
    """
    cv_scores: dict[str, float] = {}
    best_params: dict[str, dict] = {}
    fitted_estimators: dict[str, Any] = {}
    cv_fold_scores: dict[str, list] = {}
    cv_std: dict[str, float] = {}
    sw_set = sample_weight_models or set()

    for name, model in models:
        fit_kwargs = {}
        if name in sw_set:
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)

        grid = _PARAM_GRIDS.get(name)
        if grid:
            search = GridSearchCV(
                model, grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True
            )
            search.fit(X_train, y_train, **fit_kwargs)
            mean_score = round(float(search.best_score_), 6)
            cv_scores[name] = mean_score
            best_params[name] = search.best_params_
            fitted_estimators[name] = search.best_estimator_

            n_splits = cv.get_n_splits(X_train, y_train)
            fold_scores = [
                round(float(search.cv_results_[f"split{i}_test_score"][search.best_index_]), 6)
                for i in range(n_splits)
            ]
            cv_fold_scores[name] = fold_scores
            cv_std[name] = round(
                float(search.cv_results_["std_test_score"][search.best_index_]), 6
            )
            logger.info(
                "GridSearchCV %-40s  best %s = %.4f (std=%.4f)  folds=%s  best_params=%s",
                name, scoring, mean_score, cv_std[name], fold_scores, search.best_params_,
            )
        else:
            if name in sw_set:
                scores = _cv_with_sample_weight(model, X_train, y_train, cv, scoring)
            else:
                scores = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
                )
            mean_score = round(float(scores.mean()), 6)
            cv_scores[name] = mean_score
            best_params[name] = {}
            fitted_estimators[name] = None
            cv_fold_scores[name] = [round(float(s), 6) for s in scores]
            cv_std[name] = round(float(scores.std()), 6)
            logger.info(
                "CV  %-40s  %s = %.4f (±%.4f)  [no hyperparameters to tune]",
                name, scoring, mean_score, cv_std[name],
            )

    return cv_scores, best_params, fitted_estimators, cv_fold_scores, cv_std


def _model_selection_note(
    cv_scores: dict, cv_std: dict, best_name: str
) -> Optional[str]:
    """Flag when the top two candidates' CV scores are not distinguishable
    from CV noise: their mean +/- std "error bars" overlap. Returns None
    when there's a clear winner (or fewer than 2 candidates)."""
    if len(cv_scores) < 2:
        return None

    ranked = sorted(cv_scores.items(), key=lambda kv: kv[1], reverse=True)
    (name1, score1), (name2, score2) = ranked[0], ranked[1]
    std1 = cv_std.get(name1, 0.0)
    std2 = cv_std.get(name2, 0.0)
    diff = abs(score1 - score2)
    combined_std = std1 + std2

    if combined_std > 0 and diff <= combined_std:
        return (
            f"'{name1}' (CV={score1:.4f} ± {std1:.4f}) and '{name2}' "
            f"(CV={score2:.4f} ± {std2:.4f}) differ by only {diff:.4f}, "
            f"within their combined CV standard deviation ({combined_std:.4f}) "
            "-- this difference is not distinguishable from CV noise. Treat "
            f"'{name1}' as a close call against '{name2}', not a clear winner."
        )
    return None


def _nested_cv_for_winner(
    name: str,
    model_template: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    scoring: str,
    sample_weight_models: set,
    n_outer_splits: int = 3,
    random_state: int = _RANDOM_STATE,
) -> tuple[Optional[float], Optional[float], str]:
    """De-biased CV estimate for the single WINNING model via genuine
    nested cross-validation: an outer CV loop wraps a fresh inner
    GridSearchCV that re-tunes hyperparameters independently on each outer
    training fold.

    Why this exists: cv_scores/cv_std (from _run_grid_search) come from
    the same folds GridSearchCV used to pick the winning hyperparameters --
    reporting that same score as "the model's CV performance" is
    optimistically biased, since the folds have effectively been peeked at
    twice (once to tune, once to report). Nested CV decouples the two.

    Deliberately scoped to the winning model only, not every candidate --
    full nested CV for all candidates would multiply the already-expensive
    grid search by n_outer_splits per candidate, for a check the review
    only asked to "consider" rather than requiring outright.

    Returns (score, std, note); (None, None, reason) when skipped:
      * model needs the manual sample_weight fit path (this project's rare
        HGBT-without-class_weight fallback) -- composing that with
        cross_val_score's black-box outer loop would need sample weights
        re-derived by hand per outer fold; not implemented.
      * model has no hyperparameter grid (LinearRegression) -- nothing to
        re-tune, so cv_scores is already unbiased; nested CV would just be
        plain CV on fewer folds.
    """
    if name in sample_weight_models:
        return None, None, (
            f"Nested CV skipped for '{name}': requires the manual sample_weight "
            "fit path, which the nested-CV helper does not support."
        )

    grid = _PARAM_GRIDS.get(name)
    if not grid:
        return None, None, (
            f"Nested CV skipped for '{name}': no hyperparameters were tuned for "
            "this model, so its cv_scores value already has no selection-bias "
            "to correct for."
        )

    is_classification = task_type in ("binary_classification", "multiclass_classification")
    if is_classification:
        outer_cv = StratifiedKFold(
            n_splits=n_outer_splits, shuffle=True, random_state=random_state + 1
        )
        inner_cv = StratifiedKFold(
            n_splits=_N_CV_SPLITS, shuffle=True, random_state=random_state
        )
    else:
        outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state + 1)
        inner_cv = KFold(n_splits=_N_CV_SPLITS, shuffle=True, random_state=random_state)

    nested_estimator = GridSearchCV(
        clone(model_template), grid, cv=inner_cv, scoring=scoring, n_jobs=-1, refit=True,
    )
    scores = cross_val_score(
        nested_estimator, X_train, y_train, cv=outer_cv, scoring=scoring, n_jobs=-1
    )
    mean_score = round(float(scores.mean()), 6)
    std_score = round(float(scores.std()), 6)
    logger.info(
        "Nested CV for winner '%s': %s = %.4f (std=%.4f) across %d outer folds "
        "(hyperparameters re-tuned per outer fold)",
        name, scoring, mean_score, std_score, n_outer_splits,
    )
    note = (
        f"De-biased nested-CV estimate for '{name}' across {n_outer_splits} outer "
        f"folds (hyperparameters re-tuned independently per outer fold): "
        f"{scoring}={mean_score:.4f} ± {std_score:.4f}. A large gap against the "
        "model-selection CV score above means that score was optimistic."
    )
    return mean_score, std_score, note


def _compute_calibration(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = _CALIBRATION_N_BINS,
) -> Optional[dict]:
    """Reliability-diagram data + calibration-quality metrics for a fitted
    binary classifier's predict_proba output on the held-out test set.

    Bins are quantile-based over the model's own predicted probabilities
    (not fixed-width) -- this project's classifiers are trained with
    class_weight='balanced' on an ~8% positive rate, so their probability
    outputs cluster in a narrow range; fixed-width bins would leave several
    high-probability bins nearly empty. Returns None when the model has no
    predict_proba, or when predicted probabilities have too little spread
    to form at least 2 distinct bins.
    """
    if not hasattr(model, "predict_proba"):
        return None

    y_prob = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
    y_true = np.asarray(y_test, dtype=float)

    n_bins_eff = min(n_bins, len(np.unique(y_prob)))
    if n_bins_eff < 2:
        return None

    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins_eff + 1)))
    if len(edges) < 3:
        return None
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, len(edges) - 2)

    prob_true: list[float] = []
    prob_pred: list[float] = []
    bin_counts: list[int] = []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        n = int(mask.sum())
        if n == 0:
            continue
        prob_true.append(round(float(y_true[mask].mean()), 6))
        prob_pred.append(round(float(y_prob[mask].mean()), 6))
        bin_counts.append(n)

    brier = round(float(brier_score_loss(y_true, y_prob)), 6)
    ece = round(expected_calibration_error(prob_true, prob_pred, bin_counts), 6)

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "bin_counts": bin_counts,
        "brier_score": brier,
        "ece": ece,
    }


def _maybe_calibrate(
    name: str,
    model_template: Any,
    best_hyperparameters: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_calibration: dict,
    sample_weight_models: set,
) -> tuple[Optional[dict], str]:
    """If the raw model's ECE exceeds _ECE_MISCALIBRATION_THRESHOLD, fit a
    CalibratedClassifierCV (Platt/sigmoid or isotonic, cv=3 internal folds
    so the calibrator is never evaluated on rows it was fit on) using the
    winning model's own hyperparameters, and compute the same calibration
    diagnostics on its output for a side-by-side comparison.

    Diagnostic only: this does NOT replace the serialized production model
    (CalibratedClassifierCV wraps the estimator and no longer exposes
    feature_importances_/coef_ the way the raw model does, which the rest
    of this report depends on).

    Returns (calibrated_metrics_or_None, note) -- note is always populated
    and always leads with the class_weight probability caveat.
    """
    if raw_calibration["ece"] <= _ECE_MISCALIBRATION_THRESHOLD:
        return None, (
            f"{_CLASS_WEIGHT_CALIBRATION_CAVEAT} Measured calibration error "
            f"(ECE) = {raw_calibration['ece']:.4f}, at or below the "
            f"{_ECE_MISCALIBRATION_THRESHOLD:.2f} threshold this project treats "
            "as acceptable, so no calibration correction was applied."
        )

    if name in sample_weight_models:
        return None, (
            f"{_CLASS_WEIGHT_CALIBRATION_CAVEAT} Measured ECE = "
            f"{raw_calibration['ece']:.4f}, above the "
            f"{_ECE_MISCALIBRATION_THRESHOLD:.2f} threshold, but a "
            f"CalibratedClassifierCV correction was skipped for '{name}': it "
            "requires the manual sample_weight fit path, not supported here."
        )

    method = "isotonic" if len(y_train) >= 1000 else "sigmoid"
    try:
        base = clone(model_template)
        if best_hyperparameters:
            base.set_params(**best_hyperparameters)
        calibrated = CalibratedClassifierCV(base, method=method, cv=3)
        calibrated.fit(X_train, y_train)
    except Exception as exc:
        logger.warning("CalibratedClassifierCV failed for '%s': %s", name, exc)
        return None, (
            f"{_CLASS_WEIGHT_CALIBRATION_CAVEAT} Measured ECE = "
            f"{raw_calibration['ece']:.4f}, above the "
            f"{_ECE_MISCALIBRATION_THRESHOLD:.2f} threshold, but fitting a "
            f"CalibratedClassifierCV comparison failed: {exc}"
        )

    calibrated_metrics = _compute_calibration(calibrated, X_test, y_test)
    if calibrated_metrics is None:
        return None, (
            f"{_CLASS_WEIGHT_CALIBRATION_CAVEAT} Measured ECE = "
            f"{raw_calibration['ece']:.4f}, above the "
            f"{_ECE_MISCALIBRATION_THRESHOLD:.2f} threshold; a "
            f"CalibratedClassifierCV comparison was fit for '{name}' but "
            "produced no usable predict_proba output on the test set."
        )

    logger.info(
        "Calibration comparison for '%s': raw ECE=%.4f -> %s-calibrated ECE=%.4f",
        name, raw_calibration["ece"], method, calibrated_metrics["ece"],
    )
    note = (
        f"{_CLASS_WEIGHT_CALIBRATION_CAVEAT} Measured ECE = "
        f"{raw_calibration['ece']:.4f} exceeded the "
        f"{_ECE_MISCALIBRATION_THRESHOLD:.2f} threshold, so a {method}-calibrated "
        f"comparison model was fit ('{name}' + CalibratedClassifierCV, cv=3). Its "
        f"ECE = {calibrated_metrics['ece']:.4f} vs. the raw model's "
        f"{raw_calibration['ece']:.4f} (lower is better). This comparison is "
        "diagnostic only -- the serialized production model is still the raw, "
        "uncalibrated estimator."
    )
    return {"method": method, **calibrated_metrics}, note


def _extract_feature_importances(
    model: Any,
    feature_names: list[str],
    X_eval: pd.DataFrame,
    y_eval,
    top_n: int = 15,
    random_state: int = _RANDOM_STATE,
) -> dict:
    """Rank features by permutation importance on held-out data.

    Permutation importance is used unconditionally -- not just as a
    fallback for models without feature_importances_/coef_ -- because it
    is the only method here that is both model-agnostic (identical
    treatment no matter which candidate wins model selection) and
    naturally produces a distribution across repeats rather than a single
    deterministic number. Each feature is independently shuffled
    _PERMUTATION_N_REPEATS times; the mean and std of the resulting score
    drop are both kept, and a feature is flagged as not distinguishable
    from having no effect whenever its [mean - std, mean + std] range
    spans zero.

    Returns {feature_name: {"importance_mean": float, "importance_std":
    float, "distinguishable_from_zero": bool}} for the top_n features by
    mean importance. Generic over feature count/names -- nothing here
    assumes a specific dataset's columns.
    """
    n = len(X_eval)
    if n == 0:
        return {}

    sample_size = min(2000, n)
    idx = np.random.default_rng(random_state).choice(n, sample_size, replace=False)
    X_sample = X_eval.iloc[idx]
    y_sample = y_eval.iloc[idx] if hasattr(y_eval, "iloc") else y_eval[idx]

    logger.info(
        "Computing permutation importance on %d-row sample (n_repeats=%d)...",
        sample_size, _PERMUTATION_N_REPEATS,
    )
    perm = permutation_importance(
        model, X_sample, y_sample,
        n_repeats=_PERMUTATION_N_REPEATS, random_state=random_state, n_jobs=-1,
    )

    indices = np.argsort(perm.importances_mean)[::-1][:top_n]
    result = {}
    for i in indices:
        mean = float(perm.importances_mean[i])
        std = float(perm.importances_std[i])
        result[feature_names[i]] = {
            "importance_mean": round(mean, 6),
            "importance_std": round(std, 6),
            "distinguishable_from_zero": not (mean - std <= 0.0 <= mean + std),
        }
    return result


def _detect_segment_columns(
    X: pd.DataFrame,
    min_categories: int = _ERROR_ANALYSIS_MIN_CATEGORIES,
    max_categories: int = _ERROR_ANALYSIS_MAX_CATEGORIES,
    max_cols: int = _ERROR_ANALYSIS_MAX_SEGMENT_COLS,
) -> list[str]:
    """Auto-detect which feature-matrix columns look like categorical
    segments to break error rates down by.

    Generic by construction -- no column name is ever referenced. By the
    time a dataset reaches this agent, feature_engineer.py has already
    turned every raw categorical column into a numeric one (frequency,
    one-hot, or label encoding), so "categorical" can no longer be read
    off dtype. What survives encoding is cardinality: a one-hot flag has
    2 distinct values, a frequency- or label-encoded column has
    (approximately) one distinct value per original category. A truly
    continuous numeric feature, by contrast, has close to as many
    distinct values as there are rows. Selecting on distinct-value count
    alone means this works identically on any dataset's feature matrix,
    whatever its columns happen to be called.
    """
    n_rows = len(X)
    candidates = []
    for col in X.columns:
        nunique = int(X[col].nunique(dropna=True))
        if min_categories <= nunique <= max_categories and nunique < n_rows:
            candidates.append((col, nunique))
    # Higher cardinality (within the allowed range) first: a more granular
    # breakdown is more informative. Pure display-order heuristic.
    candidates.sort(key=lambda c: c[1], reverse=True)
    return [col for col, _ in candidates[:max_cols]]


def _is_elevated(
    segment_metric: float, overall_metric: float, abs_margin: Optional[float] = None,
) -> bool:
    """A segment's error metric counts as "concentrated" relative to the
    overall value once it clears a proportional margin -- scaled to the
    metric's own value so the same rule works whether the overall rate is
    2% or 40% -- and, when abs_margin is given (bounded [0, 1] rate
    metrics only), an absolute floor as well, so a near-zero overall rate
    can't be flagged "elevated" by a proportionally huge but practically
    meaningless jump.
    """
    margin = overall_metric * _ERROR_ANALYSIS_REL_MARGIN
    if abs_margin is not None:
        margin = max(margin, abs_margin)
    return (segment_metric - overall_metric) >= margin


def _json_safe_scalar(val: Any) -> Any:
    """Coerce a single segment value to a JSON-serialisable native type.

    Segment columns are numeric in this project's real pipeline (feature
    engineering encodes every categorical column to a float), but the
    functions here never assume that -- a caller-supplied segment_cols
    could point at a raw string/bool column just as easily, so this
    normalizes numpy scalar types without forcing a float() cast that
    would break on non-numeric values.
    """
    if isinstance(val, np.floating):
        return round(float(val), 6)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.bool_):
        return bool(val)
    return val


def _segment_detection_note(segment_cols: list[str], auto_detected: bool) -> str:
    if not segment_cols:
        return (
            "No segment columns were configured or auto-detected (no feature "
            f"had between {_ERROR_ANALYSIS_MIN_CATEGORIES} and "
            f"{_ERROR_ANALYSIS_MAX_CATEGORIES} distinct values) -- error "
            "analysis by segment was skipped."
        )
    how = (
        f"auto-detected as feature-matrix columns with between "
        f"{_ERROR_ANALYSIS_MIN_CATEGORIES} and {_ERROR_ANALYSIS_MAX_CATEGORIES} "
        "distinct values (the generic post-encoding signature of a "
        "categorical column)"
        if auto_detected else "explicitly configured via segment_cols"
    )
    return f"Segment columns {how}: {segment_cols}."


def _error_analysis_classification(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    segment_cols: list[str],
    task_type: str,
) -> dict:
    """Break false-negative/false-positive rates (binary) or the
    misclassification rate (multiclass) down by segment.

    Binary classification gets the standard FN/FP-rate lens because those
    are the two error modes a business decision actually cares about
    separately (missed positives vs. false alarms). Multiclass gets a
    single misclassification rate per segment instead -- a full per-class
    FP/FN grid per segment would explode combinatorially without adding
    anything actionable here.

    Segments with fewer rows than _ERROR_ANALYSIS_MIN_SEGMENT_SIZE (or,
    for a rate's denominator class specifically, fewer members of that
    class) are still reported but never flagged "elevated" -- too little
    data for the rate to mean anything.
    """
    y_pred = np.asarray(model.predict(X_test))
    y_true = np.asarray(y_test)

    result: dict = {"task_type": task_type, "segment_columns": segment_cols, "segments": {}}

    if task_type == "binary_classification":
        pos_label = sorted(pd.unique(y_true))[-1]
        is_pos = y_true == pos_label
        is_neg = ~is_pos
        pred_pos = y_pred == pos_label

        def rates(mask_pos: np.ndarray, mask_neg: np.ndarray) -> dict:
            n_pos, n_neg = int(mask_pos.sum()), int(mask_neg.sum())
            fn = float((mask_pos & ~pred_pos).sum()) / n_pos if n_pos else None
            fp = float((mask_neg & pred_pos).sum()) / n_neg if n_neg else None
            return {"n_positive": n_pos, "n_negative": n_neg, "fn": fn, "fp": fp}

        overall = rates(is_pos, is_neg)
        result["overall"] = {
            "false_negative_rate": round(overall["fn"], 6) if overall["fn"] is not None else None,
            "false_positive_rate": round(overall["fp"], 6) if overall["fp"] is not None else None,
        }

        for col in segment_cols:
            values = X_test[col].to_numpy()
            rows = []
            for val in sorted(pd.unique(values)):
                mask = values == val
                n = int(mask.sum())
                if n < _ERROR_ANALYSIS_MIN_SEGMENT_SIZE:
                    continue
                seg = rates(mask & is_pos, mask & is_neg)
                elevated_fn = (
                    seg["fn"] is not None and overall["fn"] is not None
                    and seg["n_positive"] >= _ERROR_ANALYSIS_MIN_SEGMENT_SIZE
                    and _is_elevated(seg["fn"], overall["fn"], _ERROR_ANALYSIS_ABS_MARGIN)
                )
                elevated_fp = (
                    seg["fp"] is not None and overall["fp"] is not None
                    and seg["n_negative"] >= _ERROR_ANALYSIS_MIN_SEGMENT_SIZE
                    and _is_elevated(seg["fp"], overall["fp"], _ERROR_ANALYSIS_ABS_MARGIN)
                )
                rows.append({
                    "segment_value": _json_safe_scalar(val),
                    "n": n,
                    "n_positive": seg["n_positive"],
                    "n_negative": seg["n_negative"],
                    "false_negative_rate": round(seg["fn"], 6) if seg["fn"] is not None else None,
                    "false_positive_rate": round(seg["fp"], 6) if seg["fp"] is not None else None,
                    "elevated_false_negative_rate": elevated_fn,
                    "elevated_false_positive_rate": elevated_fp,
                })
            result["segments"][col] = rows
    else:
        correct = y_true == y_pred
        overall_rate = float((~correct).mean())
        result["overall"] = {"misclassification_rate": round(overall_rate, 6)}

        for col in segment_cols:
            values = X_test[col].to_numpy()
            rows = []
            for val in sorted(pd.unique(values)):
                mask = values == val
                n = int(mask.sum())
                if n < _ERROR_ANALYSIS_MIN_SEGMENT_SIZE:
                    continue
                seg_rate = float((~correct[mask]).mean())
                rows.append({
                    "segment_value": _json_safe_scalar(val),
                    "n": n,
                    "misclassification_rate": round(seg_rate, 6),
                    "elevated_misclassification_rate": _is_elevated(
                        seg_rate, overall_rate, _ERROR_ANALYSIS_ABS_MARGIN
                    ),
                })
            result["segments"][col] = rows

    result["note"] = (
        f"Segments with fewer than {_ERROR_ANALYSIS_MIN_SEGMENT_SIZE} relevant "
        "rows are omitted -- too little data for a reliable rate. A segment's "
        "rate is flagged elevated_* when it exceeds the overall rate by at "
        f"least {_ERROR_ANALYSIS_ABS_MARGIN:.2f} absolute or "
        f"{int(_ERROR_ANALYSIS_REL_MARGIN * 100)}% relative, whichever is larger."
    )
    return result


def _error_analysis_regression(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    segment_cols: list[str],
) -> dict:
    """Break MAE and signed mean error (bias) down by segment.

    mean_error uses (actual - predicted), so a positive value means the
    model systematically under-predicts in that segment and a negative
    value means it systematically over-predicts -- not just noisier, but
    biased in a specific direction.
    """
    y_true_arr = np.asarray(y_test, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr
    abs_err = np.abs(residuals)

    overall_mae = float(abs_err.mean())
    overall_bias = float(residuals.mean())
    result: dict = {
        "task_type": "regression",
        "segment_columns": segment_cols,
        "overall": {"mae": round(overall_mae, 6), "mean_error": round(overall_bias, 6)},
        "segments": {},
    }

    for col in segment_cols:
        values = X_test[col].to_numpy()
        rows = []
        for val in sorted(pd.unique(values)):
            mask = values == val
            n = int(mask.sum())
            if n < _ERROR_ANALYSIS_MIN_SEGMENT_SIZE:
                continue
            seg_mae = float(abs_err[mask].mean())
            seg_bias = float(residuals[mask].mean())
            rows.append({
                "segment_value": _json_safe_scalar(val),
                "n": n,
                "mae": round(seg_mae, 6),
                "mean_error": round(seg_bias, 6),
                "elevated_mae": _is_elevated(seg_mae, overall_mae),
                "elevated_bias": abs(seg_bias) >= overall_mae * _ERROR_ANALYSIS_REL_MARGIN,
            })
        result["segments"][col] = rows

    result["note"] = (
        f"Segments with fewer than {_ERROR_ANALYSIS_MIN_SEGMENT_SIZE} test rows "
        "are omitted -- too little data for a reliable estimate. elevated_mae "
        f"flags a segment whose MAE exceeds the overall MAE by at least "
        f"{int(_ERROR_ANALYSIS_REL_MARGIN * 100)}%; elevated_bias flags a "
        "segment whose systematic over/under-prediction (mean_error) is at "
        f"least {int(_ERROR_ANALYSIS_REL_MARGIN * 100)}% of the overall MAE in "
        "one consistent direction."
    )
    return result


def _eval_classification(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> tuple[dict, list]:
    """Compute held-out classification metrics and confusion matrix at threshold 0.5."""
    y_pred = model.predict(X_test)

    metrics: dict = {
        "f1_macro": round(
            float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 6
        ),
        "precision_macro": round(
            float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 6
        ),
        "recall_macro": round(
            float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 6
        ),
    }

    if task_type == "binary_classification" and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 6)
        except Exception:
            pass

    cm = confusion_matrix(y_test, y_pred).tolist()
    return metrics, cm


def _eval_thresholds(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: tuple = _THRESHOLD_SWEEP,
) -> Optional[list]:
    """Sweep decision thresholds for binary classification, reporting per-class
    precision/recall and the confusion matrix at each cut-point."""
    if not hasattr(model, "predict_proba"):
        return None

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        return None

    classes = sorted(y_test.unique())
    pos_label = classes[-1]  # higher label = positive (works for 0/1 and similar)

    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
        results.append({
            "threshold": t,
            "f1_macro": round(
                float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 6
            ),
            "precision_minority": round(
                float(precision_score(
                    y_test, y_pred, pos_label=pos_label, zero_division=0
                )), 6
            ),
            "recall_minority": round(
                float(recall_score(
                    y_test, y_pred, pos_label=pos_label, zero_division=0
                )), 6
            ),
            "confusion_matrix": cm,
        })
        logger.info(
            "Threshold %.2f → F1-macro=%.4f  precision_minority=%.4f  recall_minority=%.4f",
            t, results[-1]["f1_macro"],
            results[-1]["precision_minority"],
            results[-1]["recall_minority"],
        )

    return results


def _eval_regression(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, np.ndarray]:
    """Compute held-out regression metrics; also return the raw predictions.

    The raw predictions are returned (not just aggregate metrics) so the
    caller can persist actual-vs-predicted pairs in the report -- the
    Visualization Agent needs genuine held-out predictions to draw an
    actual-vs-predicted scatter and a residual plot, and reconstructing
    them independently (re-loading the model, re-deriving the split) would
    risk silently drifting from the exact test set used here.
    """
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    adj_r2 = adjusted_r2(r2, n_samples=len(y_test), n_features=X_test.shape[1])
    metrics = {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
        "adjusted_r2": round(adj_r2, 6),
    }
    return metrics, y_pred


_MAX_TEST_PREDICTIONS = 2000


def _build_test_predictions(
    y_test: pd.Series,
    y_pred: np.ndarray,
    max_points: int = _MAX_TEST_PREDICTIONS,
    random_state: int = _RANDOM_STATE,
) -> dict:
    """Package held-out actual/predicted pairs for regression diagnostics.

    Downsampled deterministically to at most max_points when the test set
    is larger -- this only affects how many points the Visualization
    Agent's scatter plots draw, not any reported metric (those are always
    computed on the full test set in _eval_regression).
    """
    y_test_arr = np.asarray(y_test, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = len(y_test_arr)
    if n > max_points:
        idx = np.random.default_rng(random_state).choice(n, max_points, replace=False)
        y_test_arr = y_test_arr[idx]
        y_pred_arr = y_pred_arr[idx]
    return {
        "actual": [round(float(v), 6) for v in y_test_arr],
        "predicted": [round(float(v), 6) for v in y_pred_arr],
    }


_MAX_TEST_PREDICTIONS_TABLE = 5000


def _build_test_predictions_table(
    row_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    confidence: Optional[np.ndarray] = None,
    id_col: Optional[str] = None,
    max_rows: int = _MAX_TEST_PREDICTIONS_TABLE,
    random_state: int = _RANDOM_STATE,
) -> dict:
    """Package per-record held-out predictions for the dashboard.

    Generic across task types by construction: the row identity (row_ids)
    and the mechanism (build a list of row dicts, cap/sample if too many)
    are identical for classification and regression -- only the metric
    columns differ (actual_label/predicted_label/confidence/correct vs.
    actual_value/predicted_value/error/abs_error). row_ids is whatever the
    caller resolved id_col to: the id_col column's values when id_col was
    given and present in the data, otherwise each row's positional index
    in the input CSV -- either way this function treats it as an opaque
    identifier and never assumes a particular dtype or column name.

    Capped at max_rows via a fixed-seed random sample (not a head/tail
    slice) so the sample isn't skewed toward whatever order the test rows
    happened to fall in after the train/test split.
    """
    n = len(y_true)
    is_classification = task_type in ("binary_classification", "multiclass_classification")
    row_ids_arr = np.asarray(row_ids)

    sampled = n > max_rows
    if sampled:
        order = np.sort(np.random.default_rng(random_state).choice(n, max_rows, replace=False))
    else:
        order = np.arange(n)

    rows: list[dict] = []
    for i in order:
        row: dict = {"row_id": _json_safe_scalar(row_ids_arr[i])}
        if is_classification:
            correct = bool(y_true[i] == y_pred[i])
            row["actual_label"] = _json_safe_scalar(y_true[i])
            row["predicted_label"] = _json_safe_scalar(y_pred[i])
            row["confidence"] = (
                round(float(confidence[i]), 6) if confidence is not None else None
            )
            row["correct"] = correct
        else:
            actual = float(y_true[i])
            predicted = float(y_pred[i])
            row["actual_value"] = round(actual, 6)
            row["predicted_value"] = round(predicted, 6)
            row["error"] = round(predicted - actual, 6)
            row["abs_error"] = round(abs(predicted - actual), 6)
        rows.append(row)

    columns = (
        ["row_id", "actual_label", "predicted_label", "confidence", "correct"]
        if is_classification
        else ["row_id", "actual_value", "predicted_value", "error", "abs_error"]
    )

    note = (
        f"Sampled {max_rows} of {n} held-out test rows (fixed seed={random_state}) "
        "to keep the report size manageable -- the CSV download and dashboard "
        "table below reflect only this sample, not the full test set."
        if sampled else
        f"All {n} held-out test rows are included."
    )

    return {
        "task_type": task_type,
        "id_col": id_col,
        "columns": columns,
        "rows": rows,
        "total_test_rows": n,
        "sampled": sampled,
        "sample_size": len(rows),
        "note": note,
    }


def _split_indices(
    n_rows: int,
    y: pd.Series,
    task_type: str,
    test_size: float,
    random_state: int,
    groups: Optional[pd.Series] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) as positional row indices.

    Two mutually exclusive strategies:
      * groups given -- GroupShuffleSplit, so every row sharing a group_col
        value (e.g. one repeat customer's several orders) lands entirely in
        train or entirely in test. Prevents a model from partly memorizing
        group-specific patterns via rows of the same group split across
        both sides, which would otherwise inflate the held-out test score.
        GroupShuffleSplit has no native class-stratification support, so a
        grouped split trades stratification for that stronger guarantee --
        acceptable here since with many more groups than the minority
        class needs, the class-rate drift from an unstratified shuffle is
        small in practice, and it's reported for the user to see either way
        (test_metrics/threshold_metrics still reflect whatever split rate
        actually resulted).
      * groups absent -- the original plain/stratified train_test_split.
    """
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(np.zeros(n_rows), y, groups=groups))
        return train_idx, test_idx

    stratify = y if task_type in ("binary_classification", "multiclass_classification") else None
    indices = np.arange(n_rows)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train_idx, test_idx


def _aggregate_robustness(
    seed_reports: list[tuple[int, "MLReport"]],
    top_k_features: int = _ROBUSTNESS_TOP_K_FEATURES,
) -> dict:
    """Summarize a robustness check (handbook F7) across seed_reports --
    one (seed, MLReport) pair per successfully completed seed.

    Entirely generic over task type, feature count/names, and metric set:
    it reads whatever keys MLReport.test_metrics/feature_importances
    happen to have for this run rather than assuming specific ones, so a
    dataset with wholly different columns/metrics summarizes the same way.
    """
    task_type = seed_reports[0][1].task_type
    n_seeds = len(seed_reports)

    model_counts = Counter(report.best_model_name for _, report in seed_reports)
    winning_model, win_count = model_counts.most_common(1)[0]

    metric_keys = sorted({
        k for _, report in seed_reports for k in report.test_metrics
    })
    metrics_by_seed = {
        str(seed): report.test_metrics for seed, report in seed_reports
    }
    metrics_summary = {}
    for key in metric_keys:
        vals = [
            report.test_metrics[key] for _, report in seed_reports
            if key in report.test_metrics
        ]
        if vals:
            metrics_summary[key] = {
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals)), 6),
                "min": round(float(min(vals)), 6),
                "max": round(float(max(vals)), 6),
            }

    top_features_by_seed = {}
    for seed, report in seed_reports:
        ranked = sorted(
            report.feature_importances.items(),
            key=lambda kv: kv[1]["importance_mean"], reverse=True,
        )
        top_features_by_seed[str(seed)] = [f for f, _ in ranked[:top_k_features]]

    feature_appearance_counts = Counter(
        f for feats in top_features_by_seed.values() for f in feats
    )
    always_in_top_k = sorted(
        f for f, c in feature_appearance_counts.items() if c == n_seeds
    )
    seed_dependent = sorted(
        f for f, c in feature_appearance_counts.items() if c < n_seeds
    )

    return {
        "task_type": task_type,
        "n_seeds": n_seeds,
        "seeds": [seed for seed, _ in seed_reports],
        "best_model_by_seed": {
            str(seed): report.best_model_name for seed, report in seed_reports
        },
        "model_agreement": dict(model_counts),
        "winning_model": winning_model,
        "winning_model_seed_agreement": win_count,
        "winning_model_is_robust": win_count == n_seeds,
        "test_metrics_by_seed": metrics_by_seed,
        "test_metrics_summary": metrics_summary,
        "top_k_features": top_k_features,
        "top_features_by_seed": top_features_by_seed,
        "feature_appearance_counts": dict(feature_appearance_counts),
        "always_in_top_k_features": always_in_top_k,
        "seed_dependent_features": seed_dependent,
        "note": (
            f"Full pipeline (split, model selection, refit, permutation "
            f"importance) re-run under {n_seeds} different random seeds "
            f"{[s for s, _ in seed_reports]}. winning_model_is_robust=False "
            f"means the best model changed across seeds -- the single-seed "
            f"report's 'best model' claim should be treated as seed-"
            f"sensitive, not a settled fact. seed_dependent_features cleared "
            f"the top-{top_k_features} in some seeds but not all -- their "
            f"apparent importance may partly reflect this run's particular "
            f"train/test split and CV folds rather than a stable signal."
        ),
    }


# ---------------------------------------------------------------------------
# MLAgent
# ---------------------------------------------------------------------------

class MLAgent:
    """Detects task type, compares candidate models via CV, refits the best
    model on training data, evaluates on held-out test data, and writes
    an auditable JSON report.

    Parameters
    ----------
    test_size : float
        Fraction of rows reserved for the final held-out test set.
    multiclass_unique_threshold : int
        Passed to detect_task_type — maximum distinct values before the
        target is treated as continuous regression.
    top_n_features : int
        Number of top features to include in the report.
    segment_cols : list[str] | None
        Feature-matrix columns to break the held-out error analysis down
        by (handbook F6). None (the default) auto-detects them from
        feature cardinality alone -- see _detect_segment_columns -- so
        this works unmodified on a dataset with entirely different
        categorical columns. Pass an explicit list to override detection
        for a given run.
    random_state : int
        Seeds the train/test split, every candidate model, all CV
        splitters, nested CV, and permutation importance for run(). Not
        used directly by run_robustness_check (handbook F7), which varies
        this same seed across its own `seeds` argument instead, one
        _train_and_evaluate call at a time.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        multiclass_unique_threshold: int = 20,
        top_n_features: int = 15,
        segment_cols: Optional[list[str]] = None,
        random_state: int = _RANDOM_STATE,
    ):
        self.test_size = test_size
        self.multiclass_unique_threshold = multiclass_unique_threshold
        self.top_n_features = top_n_features
        self.segment_cols = segment_cols
        self.random_state = random_state
        self.report_: Optional[MLReport] = None
        self.best_model_: Any = None
        self.robustness_report_: Optional[dict] = None
        self.n_features_: Optional[int] = None

    # ------------------------------------------------------------------
    # Candidate model suites
    # ------------------------------------------------------------------

    @staticmethod
    def _classification_candidates(
        random_state: int = _RANDOM_STATE,
    ) -> tuple[list[tuple[str, Any]], set[str]]:
        """Return (candidates, sample_weight_model_names).

        sample_weight_model_names is non-empty only when HGBT does not
        support class_weight as a constructor parameter; in that case CV
        must use the manual sample_weight path for those models.

        random_state is a parameter (not the module-level _RANDOM_STATE
        constant read directly) so a robustness check across seeds
        (handbook F7) can build a fresh, differently-seeded candidate
        suite per seed without mutating agent state.
        """
        if _HGBT_SUPPORTS_CLASS_WEIGHT:
            hgbt = HistGradientBoostingClassifier(
                random_state=random_state, class_weight="balanced"
            )
            sw_names: set[str] = set()
        else:
            hgbt = HistGradientBoostingClassifier(random_state=random_state)
            sw_names = {"HistGradientBoostingClassifier"}
            logger.info(
                "HGBT does not support class_weight; using sample_weight path for CV"
            )

        candidates = [
            # lbfgs's internal line search transiently overflows on rejected
            # trial steps (RuntimeWarning: overflow/invalid in matmul) — expected
            # and harmless; verified no NaN/Inf ever lands in coef_ or
            # predict_proba, and iteration counts stay well under max_iter.
            ("LogisticRegression", LogisticRegression(
                max_iter=1000, random_state=random_state, class_weight="balanced"
            )),
            ("RandomForestClassifier", RandomForestClassifier(
                n_estimators=200, random_state=random_state,
                n_jobs=-1, class_weight="balanced"
            )),
            ("HistGradientBoostingClassifier", hgbt),
        ]
        return candidates, sw_names

    @staticmethod
    def _regression_candidates(random_state: int = _RANDOM_STATE) -> list[tuple[str, Any]]:
        return [
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor(
                n_estimators=200, random_state=random_state, n_jobs=-1
            )),
            ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(
                random_state=random_state
            )),
        ]

    # ------------------------------------------------------------------
    # Common orchestration shell
    # ------------------------------------------------------------------

    def _train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        task_type: str,
        feature_names: list[str],
        model_output_path: str,
        target_col: str,
        split_strategy: str = "row_random",
        group_col: Optional[str] = None,
        random_state: Optional[int] = None,
        save_model: bool = True,
        id_col: Optional[str] = None,
        row_ids_test: Optional[np.ndarray] = None,
    ) -> MLReport:
        """random_state defaults to self.random_state when None -- a
        robustness check across seeds (handbook F7) passes an explicit
        per-seed value here instead, without mutating agent state.
        save_model=False skips serialising to model_output_path: a
        robustness-check run is exploratory and must never overwrite the
        one canonical production model on disk.

        row_ids_test : np.ndarray | None
            Per-test-row identifiers for test_predictions_table, aligned
            positionally with X_test/y_test. The caller (_prepare_and_train)
            resolves this to id_col's values when given and present in the
            data, otherwise each row's positional index -- this method
            never makes that decision itself, it just labels rows with
            whatever it's given. id_col is carried through only to record
            in the report which column (if any) row_id came from.
        """
        rs = self.random_state if random_state is None else random_state
        is_classification = task_type in ("binary_classification", "multiclass_classification")

        if is_classification:
            candidates, sw_names = self._classification_candidates(rs)
            cv = StratifiedKFold(
                n_splits=_N_CV_SPLITS, shuffle=True, random_state=rs
            )
            scoring = "f1_macro"
        else:
            candidates = self._regression_candidates(rs)
            sw_names = set()
            cv = KFold(n_splits=_N_CV_SPLITS, shuffle=True, random_state=rs)
            scoring = "neg_root_mean_squared_error"

        logger.info(
            "Running %d-fold GridSearchCV on %d candidate models "
            "(scoring=%s, class_weight=balanced for classifiers)",
            _N_CV_SPLITS, len(candidates), scoring,
        )
        cv_scores, best_params, fitted_estimators, cv_fold_scores, cv_std = _run_grid_search(
            candidates, X_train, y_train, cv, scoring,
            sample_weight_models=sw_names,
        )

        best_name = max(cv_scores, key=cv_scores.__getitem__)
        best_hyperparameters = best_params[best_name]
        logger.info(
            "Best model: %s (CV score=%.4f ± %.4f)  best_hyperparameters=%s",
            best_name, cv_scores[best_name], cv_std.get(best_name, 0.0), best_hyperparameters,
        )

        model_selection_note = _model_selection_note(cv_scores, cv_std, best_name)
        if model_selection_note:
            logger.warning("Model selection note: %s", model_selection_note)

        nested_cv_score, nested_cv_std, nested_cv_note = _nested_cv_for_winner(
            best_name, dict(candidates)[best_name], X_train, y_train,
            task_type, scoring, sw_names, random_state=rs,
        )
        if nested_cv_score is not None:
            nested_cv_note += (
                f" Model-selection score for comparison: {scoring}="
                f"{cv_scores[best_name]:.4f}."
            )
        logger.info("Nested CV note: %s", nested_cv_note)

        best_model = fitted_estimators[best_name]
        if best_model is None:
            # No hyperparameter grid for this model (LinearRegression) —
            # GridSearchCV never ran, so it still needs an explicit fit.
            best_model = dict(candidates)[best_name]
            logger.info("Refitting %s on full training set (no grid)...", best_name)
            if best_name in sw_names:
                sw = compute_sample_weight("balanced", y_train)
                best_model.fit(X_train, y_train, sample_weight=sw)
                logger.info("Fitted with sample_weight (class_weight fallback)")
            else:
                best_model.fit(X_train, y_train)
        else:
            logger.info(
                "%s already refit on full training set via GridSearchCV(refit=True)",
                best_name,
            )
        self.best_model_ = best_model

        if save_model:
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
            joblib.dump(best_model, model_output_path)
            logger.info("Best model serialised to %s", model_output_path)
        else:
            logger.info(
                "save_model=False (exploratory run, e.g. robustness check) -- "
                "not overwriting %s", model_output_path,
            )

        feature_importances = _extract_feature_importances(
            best_model, feature_names, X_test, y_test,
            top_n=self.top_n_features, random_state=rs,
        )

        segment_cols = self.segment_cols or _detect_segment_columns(X_train)
        segment_cols = [c for c in segment_cols if c in X_test.columns]
        detection_note = _segment_detection_note(segment_cols, auto_detected=not self.segment_cols)
        logger.info("Error analysis: %s", detection_note)

        row_ids = row_ids_test if row_ids_test is not None else np.arange(len(X_test))

        if is_classification:
            test_metrics, cm = _eval_classification(best_model, X_test, y_test, task_type)
            y_pred_clf = np.asarray(best_model.predict(X_test))
            confidence = None
            if hasattr(best_model, "predict_proba"):
                try:
                    proba = best_model.predict_proba(X_test)
                    class_to_idx = {c: i for i, c in enumerate(best_model.classes_)}
                    pred_idx = np.array([class_to_idx[p] for p in y_pred_clf])
                    confidence = proba[np.arange(len(y_pred_clf)), pred_idx]
                except Exception:
                    confidence = None
            test_predictions_table = _build_test_predictions_table(
                row_ids, np.asarray(y_test), y_pred_clf, task_type,
                confidence=confidence, id_col=id_col, random_state=rs,
            )
            error_analysis = _error_analysis_classification(
                best_model, X_test, y_test, segment_cols, task_type,
            )
            error_analysis["detection_note"] = detection_note
            threshold_metrics = None
            calibration = None
            calibrated_comparison = None
            calibration_note = "Calibration not applicable (not a binary classification task)."
            if task_type == "binary_classification":
                threshold_metrics = _eval_thresholds(best_model, X_test, y_test)
                calibration = _compute_calibration(best_model, X_test, y_test)
                if calibration is None:
                    calibration_note = (
                        "Calibration not computed: the model has no predict_proba, "
                        "or its predicted probabilities had too little spread to "
                        "form calibration bins."
                    )
                else:
                    calibrated_comparison, calibration_note = _maybe_calibrate(
                        best_name, dict(candidates)[best_name], best_hyperparameters,
                        X_train, y_train, X_test, y_test, calibration, sw_names,
                    )
            logger.info("Test metrics (threshold=0.5): %s", test_metrics)
            return MLReport(
                task_type=task_type,
                best_model_name=best_name,
                target_col=target_col,
                cv_scores=cv_scores,
                best_hyperparameters=best_hyperparameters,
                test_metrics=test_metrics,
                confusion_matrix=cm,
                threshold_metrics=threshold_metrics,
                feature_importances=feature_importances,
                error_analysis=error_analysis,
                test_predictions_table=test_predictions_table,
                split_strategy=split_strategy,
                group_col=group_col,
                cv_fold_scores=cv_fold_scores,
                cv_std=cv_std,
                model_selection_note=model_selection_note,
                nested_cv_score=nested_cv_score,
                nested_cv_std=nested_cv_std,
                nested_cv_note=nested_cv_note,
                calibration=calibration,
                calibrated_comparison=calibrated_comparison,
                calibration_note=calibration_note,
            )
        else:
            test_metrics, y_pred = _eval_regression(best_model, X_test, y_test)
            logger.info("Test metrics: %s", test_metrics)
            test_predictions = _build_test_predictions(y_test, y_pred, random_state=rs)
            test_predictions_table = _build_test_predictions_table(
                row_ids, np.asarray(y_test, dtype=float), y_pred, task_type,
                id_col=id_col, random_state=rs,
            )
            error_analysis = _error_analysis_regression(X_test, y_test, y_pred, segment_cols)
            error_analysis["detection_note"] = detection_note
            return MLReport(
                task_type=task_type,
                best_model_name=best_name,
                target_col=target_col,
                cv_scores=cv_scores,
                best_hyperparameters=best_hyperparameters,
                test_metrics=test_metrics,
                confusion_matrix=None,
                threshold_metrics=None,
                feature_importances=feature_importances,
                error_analysis=error_analysis,
                test_predictions=test_predictions,
                test_predictions_table=test_predictions_table,
                split_strategy=split_strategy,
                group_col=group_col,
                cv_fold_scores=cv_fold_scores,
                cv_std=cv_std,
                model_selection_note=model_selection_note,
                nested_cv_score=nested_cv_score,
                nested_cv_std=nested_cv_std,
                nested_cv_note=nested_cv_note,
                calibration_note="Calibration not applicable (regression task).",
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _prepare_and_train(
        self,
        data_path: str,
        target_col: str,
        id_col: Optional[str] = None,
        group_col: Optional[str] = None,
        random_state: Optional[int] = None,
        save_model: bool = True,
        model_output_path: Optional[str] = None,
    ) -> tuple[bool, Optional["MLReport"], Optional[str]]:
        """Shared guts of run() and run_robustness_check() (handbook F7):
        load the CSV, build the feature matrix, split, and train/evaluate.

        Factored out so a robustness check can call this once per seed
        with random_state overridden and save_model=False, without
        duplicating the data-prep logic or writing N throwaway reports/
        models to disk. Returns (success, report_or_None, error_or_None).

        model_output_path : str | None
            Passed straight through to where the winning model gets
            serialized. None resolves to the fixed default path (see
            run()'s docstring); a caller (e.g. the Orchestrator, when
            given a run_id) can override it to keep two datasets' models
            from colliding.
        """
        logger.info(
            "Starting ML run on %s  target=%s  id_col=%s  group_col=%s  "
            "random_state=%s",
            data_path, target_col, id_col, group_col,
            self.random_state if random_state is None else random_state,
        )

        try:
            df = pd.read_csv(data_path)
        except Exception as exc:
            logger.error("Failed to read input file: %s", exc)
            return False, None, f"Failed to read input file: {exc}"

        if target_col not in df.columns:
            msg = f"Target column '{target_col}' not found in {list(df.columns)}"
            logger.error(msg)
            return False, None, msg

        group_values = None
        effective_group_col = None
        if group_col:
            if group_col in df.columns:
                group_values = df[group_col]
                effective_group_col = group_col
            else:
                logger.warning(
                    "group_col '%s' not found in data; falling back to a plain "
                    "stratified/random split (no grouping).", group_col,
                )

        drop_cols = [target_col]
        effective_id_col: Optional[str] = None
        if id_col and id_col in df.columns:
            drop_cols.append(id_col)
            effective_id_col = id_col
        else:
            if id_col:
                logger.warning(
                    "id_col '%s' not found in data; falling back to each row's "
                    "positional index as its identifier in test_predictions_table.",
                    id_col,
                )
        if effective_group_col and effective_group_col not in drop_cols:
            drop_cols.append(effective_group_col)

        # Row identifiers for test_predictions_table -- the id_col's own
        # values when usable, otherwise each row's positional index in this
        # CSV (works identically whether or not the caller supplied id_col).
        row_id_values = (
            df[effective_id_col].to_numpy() if effective_id_col else np.arange(len(df))
        )

        X = df.drop(columns=drop_cols)
        y = df[target_col]

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.warning("Dropping non-numeric columns from features: %s", non_numeric)
            X = X.drop(columns=non_numeric)

        feature_names = list(X.columns)
        self.n_features_ = len(feature_names)
        logger.info("Feature matrix: %d rows × %d cols", *X.shape)

        task_type = detect_task_type(
            y, multiclass_unique_threshold=self.multiclass_unique_threshold
        )
        logger.info(
            "Detected task type: %s  (target unique values: %d)",
            task_type, y.nunique(),
        )

        rs = self.random_state if random_state is None else random_state
        train_idx, test_idx = _split_indices(
            len(df), y, task_type, self.test_size, rs, groups=group_values,
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        row_ids_test = row_id_values[test_idx]
        split_strategy = "grouped" if effective_group_col else "row_random"
        logger.info(
            "Train size: %d  Test size: %d  split_strategy=%s",
            len(X_train), len(X_test), split_strategy,
        )
        if effective_group_col:
            train_groups = set(group_values.iloc[train_idx])
            test_groups = set(group_values.iloc[test_idx])
            overlap = train_groups & test_groups
            if overlap:
                # Should be structurally impossible with GroupShuffleSplit;
                # a non-empty overlap here means the grouping guarantee has
                # been silently broken and must not pass unnoticed.
                logger.error(
                    "%d group(s) of '%s' appear in BOTH train and test despite "
                    "GroupShuffleSplit -- grouping guarantee violated.",
                    len(overlap), effective_group_col,
                )

        if model_output_path is None:
            model_output_path = os.path.join(_MODEL_DIR, "best_production_model.pkl")

        try:
            report = self._train_and_evaluate(
                X_train, X_test, y_train, y_test,
                task_type, feature_names, model_output_path, target_col=target_col,
                split_strategy=split_strategy, group_col=effective_group_col,
                random_state=rs, save_model=save_model,
                id_col=effective_id_col, row_ids_test=row_ids_test,
            )
        except Exception as exc:
            logger.error("ML pipeline failed: %s", exc, exc_info=True)
            return False, None, f"ML pipeline failed: {exc}"

        return True, report, None

    @audit_logged("MLAgent")
    def run(
        self,
        data_path: str,
        target_col: str,
        id_col: Optional[str] = None,
        group_col: Optional[str] = None,
        model_output_path: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Run the full ML pipeline on a CSV file.

        Parameters
        ----------
        data_path : str
            Path to the feature-engineered CSV.
        target_col : str
            Name of the column to predict.
        id_col : str | None
            Row-identifier column to exclude from features. Also used to
            label rows in test_predictions_table (the per-record
            predictions the dashboard shows); when omitted, or when given
            but not present in the data, each test row is labeled with its
            positional index in the input CSV instead.
        group_col : str | None
            A column identifying which rows belong to the same real-world
            entity (e.g. a repeat-customer ID where one customer can place
            several orders/rows). When given and present in the data, the
            train/test split uses GroupShuffleSplit on this column instead
            of a plain row-wise split, so no entity's rows are split across
            both train and test -- preventing the model from partly
            memorizing entity-specific patterns and reporting an
            optimistically inflated held-out score. Also excluded from the
            feature matrix (an identifier is not itself a business
            feature). Falls back to the plain stratified split with a
            logged warning if the column isn't present in the data.
        model_output_path : str | None
            Where to serialize the winning model. Defaults to None, which
            resolves to the fixed `models/best_production_model.pkl` path
            (unchanged default behavior -- the dashboard and the Olist demo
            depend on this exact path). Pass an explicit path (e.g. a
            run-id-namespaced one) so a second dataset's model doesn't
            silently overwrite the first's.

        Returns
        -------
        (success, report_path_or_error_message)
        """
        success, report, error = self._prepare_and_train(
            data_path, target_col, id_col=id_col, group_col=group_col,
            model_output_path=model_output_path,
        )
        if not success:
            return False, error

        self.report_ = report
        base, _ = os.path.splitext(data_path)
        report_path = f"{base}_ml_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("ML report written to %s", report_path)

        try:
            log_ml_experiment(
                data_path=data_path,
                target_col=target_col,
                task_type=report.task_type,
                best_model_name=report.best_model_name,
                best_hyperparameters=report.best_hyperparameters,
                cv_scores=report.cv_scores,
                cv_std=report.cv_std,
                test_metrics=report.test_metrics,
                split_strategy=report.split_strategy,
                group_col=report.group_col,
                random_state=self.random_state,
                n_features=self.n_features_,
                model_selection_note=report.model_selection_note,
                nested_cv_score=report.nested_cv_score,
                nested_cv_std=report.nested_cv_std,
                report_path=report_path,
            )
        except Exception as exc:
            # Best-effort telemetry, same spirit as audit_logged: a
            # broken experiment-tracking write must never fail a
            # otherwise-successful ML run.
            logger.warning("Failed to log ml_experiment row: %s", exc)

        return True, report_path

    @audit_logged("MLAgent")
    def run_robustness_check(
        self,
        data_path: str,
        target_col: str,
        id_col: Optional[str] = None,
        group_col: Optional[str] = None,
        seeds: tuple[int, ...] = _ROBUSTNESS_DEFAULT_SEEDS,
    ) -> tuple[bool, str]:
        """Re-run the full pipeline once per seed in `seeds` and check
        whether the single-seed story (which model wins, how good the
        held-out metrics are, which features matter) holds up across
        seeds (handbook F7).

        Each seed gets a completely independent train/test split, model
        selection, refit, and permutation importance run via
        _prepare_and_train(..., save_model=False) -- save_model=False so
        this exploratory sweep never overwrites the one canonical
        production model that a prior run()/orchestrator call produced.
        Nothing here writes/reads model_output_path for keeps.

        Generic over task type and dataset: seeds only ever parameterize
        randomness (split, CV folds, model internals), never anything
        about which columns or how many the dataset has -- the same call
        works unmodified on a dataset with an entirely different schema.

        Returns
        -------
        (success, robustness_report_path_or_error_message)
        """
        logger.info(
            "Starting robustness check on %s  target=%s  seeds=%s",
            data_path, target_col, seeds,
        )
        if not seeds:
            return False, "run_robustness_check requires at least one seed"

        seed_reports: list[tuple[int, MLReport]] = []
        failures: list[str] = []
        for seed in seeds:
            success, report, error = self._prepare_and_train(
                data_path, target_col, id_col=id_col, group_col=group_col,
                random_state=seed, save_model=False,
            )
            if success:
                seed_reports.append((seed, report))
            else:
                logger.warning("Robustness check: seed %d failed: %s", seed, error)
                failures.append(f"seed {seed}: {error}")

        if not seed_reports:
            return False, f"All {len(seeds)} seeds failed: {'; '.join(failures)}"

        robustness_report = _aggregate_robustness(seed_reports)
        robustness_report["failed_seeds"] = failures
        robustness_report["n_seeds_attempted"] = len(seeds)
        if failures:
            logger.warning(
                "Robustness check: %d/%d seeds failed and were excluded from "
                "aggregation: %s", len(failures), len(seeds), failures,
            )
        self.robustness_report_ = robustness_report

        base, _ = os.path.splitext(data_path)
        report_path = f"{base}_robustness_report.json"
        with open(report_path, "w") as f:
            json.dump(robustness_report, f, indent=2)
        logger.info(
            "Robustness report written to %s (winning_model_is_robust=%s)",
            report_path, robustness_report["winning_model_is_robust"],
        )

        return True, report_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.agents.ml_agent <path_to_csv> <target_col> [id_col]")
        sys.exit(1)

    id_col_arg = sys.argv[3] if len(sys.argv) > 3 else None
    agent = MLAgent()
    success, result = agent.run(sys.argv[1], target_col=sys.argv[2], id_col=id_col_arg)
    if success:
        print(f"Success. Report at: {result}")
        print(json.dumps(agent.report_.to_dict(), indent=2))
    else:
        print(f"Failed: {result}")
        sys.exit(1)
