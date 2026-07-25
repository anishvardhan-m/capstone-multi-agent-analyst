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
from dataclasses import dataclass, field
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
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
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.tools.audit_db import audit_logged
from src.tools.logging_config import get_agent_logger
from src.tools.ml_tools import adjusted_r2, detect_task_type

logger = get_agent_logger("MLAgent")

_RANDOM_STATE = 42
_N_CV_SPLITS = 5
_THRESHOLD_SWEEP = (0.3, 0.4, 0.5)
_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
)

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
        Top-N importances from the best model.
    test_predictions : dict | None
        Regression only. {"actual": [...], "predicted": [...]} pairs from
        the held-out test set -- the exact predictions used to compute
        rmse/mae/r2 above. Downsampled to at most _MAX_TEST_PREDICTIONS
        points when the test set is larger (affects only how many points
        the Visualization Agent's actual-vs-predicted/residual scatter
        plots draw, not any reported metric).
    """

    task_type: str
    best_model_name: str
    cv_scores: dict = field(default_factory=dict)
    best_hyperparameters: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    confusion_matrix: Optional[list] = None
    threshold_metrics: Optional[list] = None
    feature_importances: dict = field(default_factory=dict)
    test_predictions: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "best_model_name": self.best_model_name,
            "cv_scores": self.cv_scores,
            "best_hyperparameters": self.best_hyperparameters,
            "test_metrics": self.test_metrics,
            "confusion_matrix": self.confusion_matrix,
            "threshold_metrics": self.threshold_metrics,
            "feature_importances": self.feature_importances,
            "test_predictions": self.test_predictions,
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
) -> tuple[dict[str, float], dict[str, dict], dict[str, Any]]:
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
    (cv_scores, best_params, fitted_estimators)
        fitted_estimators[name] is None for un-tuned models that still
        need to be refit by the caller.
    """
    cv_scores: dict[str, float] = {}
    best_params: dict[str, dict] = {}
    fitted_estimators: dict[str, Any] = {}
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
            logger.info(
                "GridSearchCV %-40s  best %s = %.4f  best_params=%s",
                name, scoring, mean_score, search.best_params_,
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
            logger.info(
                "CV  %-40s  %s = %.4f (±%.4f)  [no hyperparameters to tune]",
                name, scoring, mean_score, scores.std(),
            )

    return cv_scores, best_params, fitted_estimators


def _extract_feature_importances(
    model: Any,
    feature_names: list[str],
    top_n: int = 15,
    X_fallback: Optional[pd.DataFrame] = None,
    y_fallback=None,
) -> dict:
    """Extract per-feature importance from a fitted model.

    Tries model attributes first (feature_importances_, coef_). Falls back
    to permutation importance on a sample when the model exposes neither.
    """
    importances: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is not None and len(importances) == len(feature_names):
        indices = np.argsort(importances)[::-1][:top_n]
        return {feature_names[i]: round(float(importances[i]), 6) for i in indices}

    if X_fallback is not None and y_fallback is not None:
        sample_size = min(2000, len(X_fallback))
        idx = np.random.default_rng(_RANDOM_STATE).choice(
            len(X_fallback), sample_size, replace=False
        )
        X_sample = X_fallback.iloc[idx]
        y_sample = (
            y_fallback.iloc[idx] if hasattr(y_fallback, "iloc") else y_fallback[idx]
        )
        logger.info("Computing permutation importance on %d-row sample...", sample_size)
        perm = permutation_importance(
            model, X_sample, y_sample,
            n_repeats=5, random_state=_RANDOM_STATE, n_jobs=-1,
        )
        importances = perm.importances_mean
        indices = np.argsort(importances)[::-1][:top_n]
        return {feature_names[i]: round(float(importances[i]), 6) for i in indices}

    return {}


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
        idx = np.random.default_rng(_RANDOM_STATE).choice(n, max_points, replace=False)
        y_test_arr = y_test_arr[idx]
        y_pred_arr = y_pred_arr[idx]
    return {
        "actual": [round(float(v), 6) for v in y_test_arr],
        "predicted": [round(float(v), 6) for v in y_pred_arr],
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
    """

    def __init__(
        self,
        test_size: float = 0.2,
        multiclass_unique_threshold: int = 20,
        top_n_features: int = 15,
    ):
        self.test_size = test_size
        self.multiclass_unique_threshold = multiclass_unique_threshold
        self.top_n_features = top_n_features
        self.report_: Optional[MLReport] = None
        self.best_model_: Any = None

    # ------------------------------------------------------------------
    # Candidate model suites
    # ------------------------------------------------------------------

    @staticmethod
    def _classification_candidates() -> tuple[list[tuple[str, Any]], set[str]]:
        """Return (candidates, sample_weight_model_names).

        sample_weight_model_names is non-empty only when HGBT does not
        support class_weight as a constructor parameter; in that case CV
        must use the manual sample_weight path for those models.
        """
        if _HGBT_SUPPORTS_CLASS_WEIGHT:
            hgbt = HistGradientBoostingClassifier(
                random_state=_RANDOM_STATE, class_weight="balanced"
            )
            sw_names: set[str] = set()
        else:
            hgbt = HistGradientBoostingClassifier(random_state=_RANDOM_STATE)
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
                max_iter=1000, random_state=_RANDOM_STATE, class_weight="balanced"
            )),
            ("RandomForestClassifier", RandomForestClassifier(
                n_estimators=200, random_state=_RANDOM_STATE,
                n_jobs=-1, class_weight="balanced"
            )),
            ("HistGradientBoostingClassifier", hgbt),
        ]
        return candidates, sw_names

    @staticmethod
    def _regression_candidates() -> list[tuple[str, Any]]:
        return [
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor(
                n_estimators=200, random_state=_RANDOM_STATE, n_jobs=-1
            )),
            ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(
                random_state=_RANDOM_STATE
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
    ) -> MLReport:
        is_classification = task_type in ("binary_classification", "multiclass_classification")

        if is_classification:
            candidates, sw_names = self._classification_candidates()
            cv = StratifiedKFold(
                n_splits=_N_CV_SPLITS, shuffle=True, random_state=_RANDOM_STATE
            )
            scoring = "f1_macro"
        else:
            candidates = self._regression_candidates()
            sw_names = set()
            cv = KFold(n_splits=_N_CV_SPLITS, shuffle=True, random_state=_RANDOM_STATE)
            scoring = "neg_root_mean_squared_error"

        logger.info(
            "Running %d-fold GridSearchCV on %d candidate models "
            "(scoring=%s, class_weight=balanced for classifiers)",
            _N_CV_SPLITS, len(candidates), scoring,
        )
        cv_scores, best_params, fitted_estimators = _run_grid_search(
            candidates, X_train, y_train, cv, scoring,
            sample_weight_models=sw_names,
        )

        best_name = max(cv_scores, key=cv_scores.__getitem__)
        best_hyperparameters = best_params[best_name]
        logger.info(
            "Best model: %s (CV score=%.4f)  best_hyperparameters=%s",
            best_name, cv_scores[best_name], best_hyperparameters,
        )

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

        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(best_model, model_output_path)
        logger.info("Best model serialised to %s", model_output_path)

        feature_importances = _extract_feature_importances(
            best_model, feature_names,
            top_n=self.top_n_features,
            X_fallback=X_test,
            y_fallback=y_test,
        )

        if is_classification:
            test_metrics, cm = _eval_classification(best_model, X_test, y_test, task_type)
            threshold_metrics = None
            if task_type == "binary_classification":
                threshold_metrics = _eval_thresholds(best_model, X_test, y_test)
            logger.info("Test metrics (threshold=0.5): %s", test_metrics)
            return MLReport(
                task_type=task_type,
                best_model_name=best_name,
                cv_scores=cv_scores,
                best_hyperparameters=best_hyperparameters,
                test_metrics=test_metrics,
                confusion_matrix=cm,
                threshold_metrics=threshold_metrics,
                feature_importances=feature_importances,
            )
        else:
            test_metrics, y_pred = _eval_regression(best_model, X_test, y_test)
            logger.info("Test metrics: %s", test_metrics)
            test_predictions = _build_test_predictions(y_test, y_pred)
            return MLReport(
                task_type=task_type,
                best_model_name=best_name,
                cv_scores=cv_scores,
                best_hyperparameters=best_hyperparameters,
                test_metrics=test_metrics,
                confusion_matrix=None,
                threshold_metrics=None,
                feature_importances=feature_importances,
                test_predictions=test_predictions,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @audit_logged("MLAgent")
    def run(
        self,
        data_path: str,
        target_col: str,
        id_col: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Run the full ML pipeline on a CSV file.

        Parameters
        ----------
        data_path : str
            Path to the feature-engineered CSV.
        target_col : str
            Name of the column to predict.
        id_col : str | None
            Row-identifier column to exclude from features.

        Returns
        -------
        (success, report_path_or_error_message)
        """
        logger.info(
            "Starting ML run on %s  target=%s  id_col=%s",
            data_path, target_col, id_col,
        )

        try:
            df = pd.read_csv(data_path)
        except Exception as exc:
            logger.error("Failed to read input file: %s", exc)
            return False, f"Failed to read input file: {exc}"

        if target_col not in df.columns:
            msg = f"Target column '{target_col}' not found in {list(df.columns)}"
            logger.error(msg)
            return False, msg

        drop_cols = [target_col]
        if id_col and id_col in df.columns:
            drop_cols.append(id_col)

        X = df.drop(columns=drop_cols)
        y = df[target_col]

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.warning("Dropping non-numeric columns from features: %s", non_numeric)
            X = X.drop(columns=non_numeric)

        feature_names = list(X.columns)
        logger.info("Feature matrix: %d rows × %d cols", *X.shape)

        task_type = detect_task_type(
            y, multiclass_unique_threshold=self.multiclass_unique_threshold
        )
        logger.info(
            "Detected task type: %s  (target unique values: %d)",
            task_type, y.nunique(),
        )

        stratify = (
            y if task_type in ("binary_classification", "multiclass_classification")
            else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=_RANDOM_STATE,
            stratify=stratify,
        )
        logger.info("Train size: %d  Test size: %d", len(X_train), len(X_test))

        model_output_path = os.path.join(_MODEL_DIR, "best_production_model.pkl")

        try:
            self.report_ = self._train_and_evaluate(
                X_train, X_test, y_train, y_test,
                task_type, feature_names, model_output_path,
            )
        except Exception as exc:
            logger.error("ML pipeline failed: %s", exc, exc_info=True)
            return False, f"ML pipeline failed: {exc}"

        base, _ = os.path.splitext(data_path)
        report_path = f"{base}_ml_report.json"
        with open(report_path, "w") as f:
            json.dump(self.report_.to_dict(), f, indent=2)
        logger.info("ML report written to %s", report_path)

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
