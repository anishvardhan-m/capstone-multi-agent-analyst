"""
tests/test_ml_agent.py

Unit tests for the ML tools (src/tools/ml_tools.py) and the ML Agent
(src/agents/ml_agent.py).

Run with:
    pytest tests/test_ml_agent.py -v
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.agents.ml_agent as ml_agent
from src.agents.ml_agent import (
    MLAgent,
    MLReport,
    _RANDOM_STATE,
    _aggregate_robustness,
    _compute_calibration,
    _detect_segment_columns,
    _error_analysis_classification,
    _error_analysis_regression,
    _eval_classification,
    _eval_regression,
    _extract_feature_importances,
    _maybe_calibrate,
    _model_selection_note,
    _nested_cv_for_winner,
    _run_grid_search,
    _split_indices,
)
from src.tools.audit_db import get_recent_experiments
from src.tools.ml_tools import adjusted_r2, detect_task_type, expected_calibration_error


# ---------------------------------------------------------------------------
# detect_task_type
# ---------------------------------------------------------------------------

def test_detect_binary_classification():
    y = pd.Series([0, 1, 0, 1, 1])
    assert detect_task_type(y) == "binary_classification"


def test_detect_multiclass_classification_integer():
    y = pd.Series([0, 1, 2, 3, 2, 1, 0])
    assert detect_task_type(y) == "multiclass_classification"


def test_detect_multiclass_classification_object():
    y = pd.Series(["cat", "dog", "fish", "cat", "dog"])
    assert detect_task_type(y) == "multiclass_classification"


def test_detect_regression_continuous():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.uniform(0, 100, 500))
    assert detect_task_type(y) == "regression"


def test_detect_regression_when_many_integers():
    # Many unique integer values → treated as regression, not multiclass
    y = pd.Series(range(100))  # 100 unique integers
    assert detect_task_type(y, multiclass_unique_threshold=20) == "regression"


def test_detect_uses_threshold_parameter():
    # 5 unique integer values; threshold=3 → regression; threshold=10 → multiclass
    y = pd.Series([1, 2, 3, 4, 5])
    assert detect_task_type(y, multiclass_unique_threshold=3) == "regression"
    assert detect_task_type(y, multiclass_unique_threshold=10) == "multiclass_classification"


# ---------------------------------------------------------------------------
# adjusted_r2
# ---------------------------------------------------------------------------

def test_adjusted_r2_perfect_fit():
    result = adjusted_r2(r2=1.0, n_samples=100, n_features=5)
    assert result == pytest.approx(1.0)


def test_adjusted_r2_worse_than_r2_with_many_features():
    # Adding useless features should lower adjusted R²
    base_r2 = 0.8
    result = adjusted_r2(r2=base_r2, n_samples=100, n_features=50)
    assert result < base_r2


def test_adjusted_r2_returns_nan_when_degenerate():
    result = adjusted_r2(r2=0.9, n_samples=5, n_features=10)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def test_eval_classification_metrics_on_known_example():
    """Manually verify F1 and confusion matrix on a trivially known example."""
    from sklearn.dummy import DummyClassifier

    X = pd.DataFrame({"a": range(10)})
    y_test = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Predict all-zeros: confusion matrix should be [[5,0],[5,0]]
    model = DummyClassifier(strategy="constant", constant=0)
    model.fit(X, y_test)

    metrics, cm = _eval_classification(model, X, y_test, "binary_classification")

    assert cm == [[5, 0], [5, 0]]
    # class 0 recall = 1.0, class 1 recall = 0.0 → macro = 0.5
    assert metrics["recall_macro"] == pytest.approx(0.5, abs=0.01)


def test_eval_regression_metrics_on_known_example():
    """Verify RMSE and MAE on a simple known case."""
    from sklearn.dummy import DummyRegressor

    X = pd.DataFrame({"a": range(4)})
    y_test = pd.Series([1.0, 2.0, 3.0, 4.0])
    # DummyRegressor predicts the mean (2.5) for every row
    model = DummyRegressor(strategy="mean")
    model.fit(X, y_test)

    metrics, y_pred = _eval_regression(model, X, y_test)

    # Errors: [-1.5, -0.5, 0.5, 1.5] → RMSE = sqrt(mean([2.25,0.25,0.25,2.25]))
    expected_rmse = np.sqrt(np.mean([2.25, 0.25, 0.25, 2.25]))
    expected_mae = np.mean([1.5, 0.5, 0.5, 1.5])

    assert metrics["rmse"] == pytest.approx(expected_rmse, rel=1e-4)
    assert metrics["mae"] == pytest.approx(expected_mae, rel=1e-4)
    assert list(y_pred) == pytest.approx([2.5, 2.5, 2.5, 2.5])


# ---------------------------------------------------------------------------
# test_predictions (regression diagnostics, consumed by VisualizationAgent)
# ---------------------------------------------------------------------------

def test_build_test_predictions_keeps_all_points_under_the_cap():
    from src.agents.ml_agent import _build_test_predictions

    y_test = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    result = _build_test_predictions(y_test, y_pred, max_points=10)

    assert result["actual"] == pytest.approx([1.0, 2.0, 3.0])
    assert result["predicted"] == pytest.approx([1.1, 1.9, 3.2])


def test_build_test_predictions_downsamples_above_the_cap():
    from src.agents.ml_agent import _build_test_predictions

    y_test = pd.Series(np.arange(100, dtype=float))
    y_pred = np.arange(100, dtype=float) + 0.5

    result = _build_test_predictions(y_test, y_pred, max_points=10)

    assert len(result["actual"]) == 10
    assert len(result["predicted"]) == 10
    # Every sampled (actual, predicted) pair must still be a real, matched
    # observation from the original arrays -- not independently resampled.
    for a, p in zip(result["actual"], result["predicted"]):
        assert p == pytest.approx(a + 0.5)


# ---------------------------------------------------------------------------
# test_predictions_table (per-record predictions, dashboard-facing,
# generic across task types)
# ---------------------------------------------------------------------------

def test_build_test_predictions_table_classification_shape_and_values():
    from src.agents.ml_agent import _build_test_predictions_table

    row_ids = np.array(["r0", "r1", "r2", "r3"])
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    confidence = np.array([0.9, 0.8, 0.55, 0.99])

    table = _build_test_predictions_table(
        row_ids, y_true, y_pred, "binary_classification", confidence=confidence,
        id_col="order_id",
    )

    assert table["task_type"] == "binary_classification"
    assert table["id_col"] == "order_id"
    assert table["columns"] == [
        "row_id", "actual_label", "predicted_label", "confidence", "correct",
    ]
    assert table["total_test_rows"] == 4
    assert table["sampled"] is False
    assert table["sample_size"] == 4
    assert len(table["rows"]) == 4

    row2 = next(r for r in table["rows"] if r["row_id"] == "r2")
    assert row2["actual_label"] == 1
    assert row2["predicted_label"] == 0
    assert row2["confidence"] == pytest.approx(0.55)
    assert row2["correct"] is False

    row0 = next(r for r in table["rows"] if r["row_id"] == "r0")
    assert row0["correct"] is True


def test_build_test_predictions_table_regression_shape_and_values():
    from src.agents.ml_agent import _build_test_predictions_table

    row_ids = np.arange(3)
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 19.0, 25.0])

    table = _build_test_predictions_table(row_ids, y_true, y_pred, "regression", id_col=None)

    assert table["id_col"] is None
    assert table["columns"] == [
        "row_id", "actual_value", "predicted_value", "error", "abs_error",
    ]
    row0 = table["rows"][0]
    assert row0["row_id"] == 0
    assert row0["actual_value"] == pytest.approx(10.0)
    assert row0["predicted_value"] == pytest.approx(12.0)
    assert row0["error"] == pytest.approx(2.0)
    assert row0["abs_error"] == pytest.approx(2.0)


def test_build_test_predictions_table_no_confidence_when_unavailable():
    from src.agents.ml_agent import _build_test_predictions_table

    row_ids = np.arange(2)
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    table = _build_test_predictions_table(row_ids, y_true, y_pred, "binary_classification")

    assert all(r["confidence"] is None for r in table["rows"])


def test_build_test_predictions_table_under_cap_includes_every_row():
    from src.agents.ml_agent import _build_test_predictions_table

    n = 50
    row_ids = np.arange(n)
    y_true = np.zeros(n, dtype=float)
    y_pred = np.ones(n, dtype=float)

    table = _build_test_predictions_table(
        row_ids, y_true, y_pred, "regression", max_rows=5000,
    )

    assert table["sampled"] is False
    assert table["total_test_rows"] == n
    assert table["sample_size"] == n
    assert len(table["rows"]) == n
    assert "All 50 held-out test rows" in table["note"]


def test_build_test_predictions_table_caps_and_samples_with_fixed_seed():
    from src.agents.ml_agent import _build_test_predictions_table

    n = 200
    row_ids = np.arange(n)
    y_true = np.arange(n, dtype=float)
    y_pred = np.arange(n, dtype=float) + 1.0

    table_a = _build_test_predictions_table(
        row_ids, y_true, y_pred, "regression", max_rows=20, random_state=7,
    )
    table_b = _build_test_predictions_table(
        row_ids, y_true, y_pred, "regression", max_rows=20, random_state=7,
    )

    assert table_a["sampled"] is True
    assert table_a["total_test_rows"] == n
    assert table_a["sample_size"] == 20
    assert len(table_a["rows"]) == 20
    assert "Sampled 20 of 200" in table_a["note"]

    # Same seed -> identical sample (deterministic, reproducible report).
    ids_a = [r["row_id"] for r in table_a["rows"]]
    ids_b = [r["row_id"] for r in table_b["rows"]]
    assert ids_a == ids_b

    # Every sampled row is still a genuine, correctly-matched observation.
    for row in table_a["rows"]:
        assert row["predicted_value"] == pytest.approx(row["actual_value"] + 1.0)


def test_agent_e2e_classification_populates_test_predictions_table(classification_csv):
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="row_id")

    table = agent.report_.test_predictions_table
    assert table is not None
    assert table["task_type"] == "binary_classification"
    assert table["id_col"] == "row_id"
    n_test = int(round(400 * agent.test_size))
    assert table["total_test_rows"] == n_test
    assert len(table["rows"]) == n_test  # well under the 5000 cap

    row = table["rows"][0]
    assert row["row_id"].startswith("id_")
    assert row["actual_label"] in (0, 1)
    assert row["predicted_label"] in (0, 1)
    assert 0.0 <= row["confidence"] <= 1.0
    assert isinstance(row["correct"], bool)
    assert row["correct"] == (row["actual_label"] == row["predicted_label"])


def test_agent_e2e_regression_populates_test_predictions_table(regression_csv):
    agent = MLAgent()
    agent.run(regression_csv, target_col="price")  # no id_col supplied

    table = agent.report_.test_predictions_table
    assert table is not None
    assert table["task_type"] == "regression"
    assert table["id_col"] is None  # falls back to positional index

    row_ids = [r["row_id"] for r in table["rows"]]
    assert all(isinstance(rid, int) for rid in row_ids)
    assert len(set(row_ids)) == len(row_ids)  # positional indices are unique

    row = table["rows"][0]
    assert "actual_value" in row and "predicted_value" in row
    assert row["abs_error"] == pytest.approx(abs(row["error"]))


def test_agent_e2e_test_predictions_table_id_col_missing_falls_back_to_index(classification_csv):
    """id_col requested but not actually in the data -- must still populate
    the table via positional index, not crash or silently omit the field."""
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="not_a_real_column")

    table = agent.report_.test_predictions_table
    assert table is not None
    assert table["id_col"] is None
    row_ids = [r["row_id"] for r in table["rows"]]
    assert all(isinstance(rid, int) for rid in row_ids)


def test_agent_e2e_test_predictions_table_generic_synthetic_dataset_no_id_col(tmp_path):
    """Genericity check: a synthetic dataset with entirely different column
    names, a multiclass target, and no id_col at all -- must populate
    test_predictions_table with the same mechanism, proving this isn't
    Olist-specific in any way."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_classes=3,
        n_clusters_per_class=1, random_state=11,
    )
    df = pd.DataFrame(X, columns=[f"sensor_reading_{i}" for i in range(6)])
    df["equipment_failure_mode"] = y
    path = tmp_path / "synthetic_equipment.csv"
    df.to_csv(path, index=False)

    agent = MLAgent()
    success, _ = agent.run(str(path), target_col="equipment_failure_mode")
    assert success is True

    report = agent.report_
    assert report.task_type == "multiclass_classification"
    table = report.test_predictions_table
    assert table is not None
    assert table["id_col"] is None
    assert table["columns"] == [
        "row_id", "actual_label", "predicted_label", "confidence", "correct",
    ]
    row_ids = [r["row_id"] for r in table["rows"]]
    assert all(isinstance(rid, int) for rid in row_ids)
    assert len(set(row_ids)) == len(row_ids)
    for row in table["rows"]:
        assert row["actual_label"] in (0, 1, 2)
        assert row["predicted_label"] in (0, 1, 2)


# ---------------------------------------------------------------------------
# Stratified split preserves class balance (classification)
# ---------------------------------------------------------------------------

def test_stratified_split_preserves_class_balance():
    rng = np.random.default_rng(1)
    # 90% class 0, 10% class 1
    y = pd.Series(np.where(rng.uniform(size=1000) < 0.1, 1, 0))
    X = pd.DataFrame({"f": rng.standard_normal(1000)})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    # Both splits should be within 2 pp of the original 10% rate
    assert abs(train_rate - 0.10) < 0.02
    assert abs(test_rate - 0.10) < 0.02


# ---------------------------------------------------------------------------
# Leakage-prevention: preprocessing fitted on train only
# ---------------------------------------------------------------------------

def test_scaler_stats_differ_train_only_vs_full_data():
    """Fitting a NumericScaler on training data alone must produce different
    statistics from fitting on the full dataset, confirming that the agent
    does not leak test information into the preprocessing step."""
    from src.tools.feature_tools import NumericScaler

    rng = np.random.default_rng(99)
    n = 200
    # Introduce a large outlier block that only appears in the 'test' portion
    values = np.concatenate([rng.standard_normal(160), rng.standard_normal(40) * 10 + 50])
    df_full = pd.DataFrame({"x": values})
    df_train = df_full.iloc[:160]

    scaler_train_only = NumericScaler()
    scaler_train_only.fit(df_train)

    scaler_full = NumericScaler()
    scaler_full.fit(df_full)

    assert scaler_train_only.scale_stats_["x"]["mean"] != scaler_full.scale_stats_["x"]["mean"]


# ---------------------------------------------------------------------------
# Grouped split (F1): no group appears in both train and test
# ---------------------------------------------------------------------------

def test_split_indices_grouped_has_no_overlap():
    rng = np.random.default_rng(11)
    n = 1000
    # 150 groups, each repeated a handful of times -- like repeat customers.
    groups = pd.Series(rng.integers(0, 150, size=n))
    y = pd.Series(rng.integers(0, 2, size=n))

    train_idx, test_idx = _split_indices(
        n, y, "binary_classification", test_size=0.2, random_state=42, groups=groups
    )

    assert set(train_idx).isdisjoint(set(test_idx))
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    assert train_groups.isdisjoint(test_groups), (
        "A group_col value appeared in both train and test despite GroupShuffleSplit"
    )
    # Sanity: grouping actually did something (not literally every row its own group)
    assert len(train_groups) + len(test_groups) <= 150


def test_split_indices_without_groups_falls_back_to_row_random():
    rng = np.random.default_rng(12)
    n = 500
    y = pd.Series(rng.integers(0, 2, size=n))

    train_idx, test_idx = _split_indices(
        n, y, "binary_classification", test_size=0.2, random_state=42, groups=None
    )
    assert set(train_idx).isdisjoint(set(test_idx))
    assert len(train_idx) + len(test_idx) == n
    assert len(test_idx) == pytest.approx(n * 0.2, abs=1)


def test_split_indices_grouped_preserves_row_count():
    rng = np.random.default_rng(13)
    n = 800
    groups = pd.Series(rng.integers(0, 200, size=n))
    y = pd.Series(rng.integers(0, 2, size=n))

    train_idx, test_idx = _split_indices(
        n, y, "binary_classification", test_size=0.25, random_state=42, groups=groups
    )
    assert len(set(train_idx) | set(test_idx)) == n


@pytest.fixture
def grouped_classification_csv(tmp_path):
    """Synthetic data with a repeat-customer-style group_col: 60 groups,
    each contributing 1-6 rows (rows = orders, groups = customers)."""
    rng = np.random.default_rng(21)
    n_groups = 60
    rows = []
    for gid in range(n_groups):
        n_rows_for_group = rng.integers(1, 7)
        for _ in range(n_rows_for_group):
            rows.append(gid)
    customer_ids = np.array(rows)
    n = len(customer_ids)
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["label"] = y
    df["customer_unique_id"] = customer_ids
    path = tmp_path / "grouped_clf.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_agent_e2e_group_col_no_group_split_across_train_and_test(grouped_classification_csv):
    """Regression test for F1: when group_col is given, re-derive the split
    from scratch (mirroring MLAgent.run's own logic) and assert no group_col
    value's rows are divided between train and test."""
    df = pd.read_csv(grouped_classification_csv)
    groups = df["customer_unique_id"]
    y = df["label"]

    train_idx, test_idx = _split_indices(
        len(df), y, "binary_classification", test_size=0.2,
        random_state=42, groups=groups,
    )
    train_customers = set(groups.iloc[train_idx])
    test_customers = set(groups.iloc[test_idx])
    assert train_customers.isdisjoint(test_customers)

    # The agent itself must run successfully with group_col wired through,
    # exclude it from features, and record the strategy used.
    agent = MLAgent()
    success, report_path = agent.run(
        grouped_classification_csv, target_col="label", group_col="customer_unique_id",
    )
    assert success is True
    assert agent.report_.split_strategy == "grouped"
    assert agent.report_.group_col == "customer_unique_id"
    assert "customer_unique_id" not in agent.report_.feature_importances

    with open(report_path) as f:
        data = json.load(f)
    assert data["split_strategy"] == "grouped"
    assert data["group_col"] == "customer_unique_id"


def test_agent_group_col_missing_from_data_falls_back_gracefully(classification_csv):
    """A group_col that doesn't exist in the data must not crash the run --
    it falls back to the plain split with a logged warning."""
    agent = MLAgent()
    success, _ = agent.run(
        classification_csv, target_col="label", id_col="row_id",
        group_col="no_such_column",
    )
    assert success is True
    assert agent.report_.split_strategy == "row_random"
    assert agent.report_.group_col is None


def test_agent_default_split_strategy_is_row_random(classification_csv):
    """Backward compatibility: omitting group_col entirely must behave
    exactly as before (plain stratified split), reported as row_random."""
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="row_id")
    assert agent.report_.split_strategy == "row_random"
    assert agent.report_.group_col is None


# ---------------------------------------------------------------------------
# F2: model-selection note (CV-noise flag) and nested CV
# ---------------------------------------------------------------------------

def test_model_selection_note_flags_indistinguishable_top_two():
    cv_scores = {"A": 0.700, "B": 0.695, "C": 0.500}
    cv_std = {"A": 0.010, "B": 0.010, "C": 0.005}  # diff(A,B)=0.005 <= std sum 0.020
    note = _model_selection_note(cv_scores, cv_std, best_name="A")
    assert note is not None
    assert "not distinguishable from CV noise" in note
    assert "'A'" in note and "'B'" in note


def test_model_selection_note_none_when_clear_winner():
    cv_scores = {"A": 0.900, "B": 0.500, "C": 0.400}
    cv_std = {"A": 0.010, "B": 0.010, "C": 0.010}  # diff(A,B)=0.4 >> std sum 0.02
    note = _model_selection_note(cv_scores, cv_std, best_name="A")
    assert note is None


def test_model_selection_note_none_with_single_candidate():
    assert _model_selection_note({"A": 0.9}, {"A": 0.01}, best_name="A") is None


def test_nested_cv_skipped_for_sample_weight_model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((100, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.integers(0, 2, 100))

    score, std, note = _nested_cv_for_winner(
        "HistGradientBoostingClassifier",
        HistGradientBoostingClassifier(random_state=42),
        X, y, "binary_classification", "f1_macro",
        sample_weight_models={"HistGradientBoostingClassifier"},
    )
    assert score is None and std is None
    assert "sample_weight" in note


def test_nested_cv_skipped_for_untuned_model():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.standard_normal((100, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.standard_normal(100))

    score, std, note = _nested_cv_for_winner(
        "LinearRegression", LinearRegression(),
        X, y, "regression", "neg_root_mean_squared_error",
        sample_weight_models=set(),
    )
    assert score is None and std is None
    assert "no hyperparameters were tuned" in note


def test_nested_cv_runs_for_tuned_model():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int))

    score, std, note = _nested_cv_for_winner(
        "LogisticRegression", LogisticRegression(max_iter=1000, random_state=42),
        X, y, "binary_classification", "f1_macro",
        sample_weight_models=set(), n_outer_splits=3,
    )
    assert score is not None
    assert 0.0 <= score <= 1.0
    assert std is not None and std >= 0.0
    assert "De-biased nested-CV estimate" in note


def test_agent_e2e_report_carries_cv_rigor_fields(classification_csv):
    """End-to-end: cv_fold_scores/cv_std/model_selection_note/nested_cv_*
    must all reach the persisted report, not just live on the dataclass."""
    agent = MLAgent()
    _, report_path = agent.run(classification_csv, target_col="label", id_col="row_id")
    report = agent.report_

    assert set(report.cv_fold_scores.keys()) == set(report.cv_scores.keys())
    for name, folds in report.cv_fold_scores.items():
        assert len(folds) == 5  # MLAgent's _N_CV_SPLITS
        assert report.cv_std[name] == pytest.approx(float(np.std(folds)), abs=1e-4)

    # Nested CV ran for the winner (every classification candidate has a grid).
    assert report.nested_cv_score is not None
    assert report.nested_cv_std is not None
    assert report.nested_cv_note

    with open(report_path) as f:
        parsed = json.load(f)
    assert "cv_fold_scores" in parsed
    assert "nested_cv_score" in parsed
    assert "model_selection_note" in parsed


# ---------------------------------------------------------------------------
# F3: calibration
# ---------------------------------------------------------------------------

def test_expected_calibration_error_zero_when_perfectly_calibrated():
    ece = expected_calibration_error([0.1, 0.5, 0.9], [0.1, 0.5, 0.9], [10, 10, 10])
    assert ece == pytest.approx(0.0)


def test_expected_calibration_error_weighted_by_bin_count():
    # Bin 1 (huge gap, tiny weight) should barely move the result vs bin 2
    # (small gap, huge weight).
    ece = expected_calibration_error([0.0, 0.51], [1.0, 0.50], [1, 999])
    assert ece == pytest.approx((1.0 * 1 + 0.01 * 999) / 1000, abs=1e-6)


def test_expected_calibration_error_empty_bins_returns_zero():
    assert expected_calibration_error([], [], []) == 0.0


def test_compute_calibration_none_without_predict_proba():
    from sklearn.svm import LinearSVC  # has no predict_proba by default

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.integers(0, 2, 50))
    model = LinearSVC()
    model.fit(X, y)
    assert _compute_calibration(model, X, y) is None


def test_compute_calibration_returns_expected_shape():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(2)
    n = 400
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((X["f0"] + X["f1"] + rng.standard_normal(n) * 0.5 > 0).astype(int))
    model = LogisticRegression(max_iter=1000).fit(X, y)

    result = _compute_calibration(model, X, y, n_bins=5)
    assert result is not None
    assert len(result["prob_true"]) == len(result["prob_pred"]) == len(result["bin_counts"])
    assert sum(result["bin_counts"]) == n
    assert 0.0 <= result["ece"]
    assert 0.0 <= result["brier_score"] <= 1.0


def test_compute_calibration_well_separated_model_has_low_ece():
    """A model on cleanly separable, balanced data (no class_weight
    distortion) should calibrate well -- ECE close to 0."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    n = 2000
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["a", "b"])
    y = pd.Series((X["a"] + X["b"] + rng.standard_normal(n) * 0.3 > 0).astype(int))
    model = LogisticRegression(max_iter=1000).fit(X, y)

    result = _compute_calibration(model, X, y, n_bins=10)
    assert result["ece"] < 0.1


def test_maybe_calibrate_skips_when_ece_acceptable():
    from sklearn.linear_model import LogisticRegression

    calibration = {"ece": 0.01, "prob_true": [], "prob_pred": [], "bin_counts": [], "brier_score": 0.0}
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.integers(0, 2, 50))

    result, note = _maybe_calibrate(
        "LogisticRegression", LogisticRegression(max_iter=1000), {},
        X, y, X, y, calibration, sample_weight_models=set(),
    )
    assert result is None
    assert "no calibration correction was applied" in note
    assert "class_weight='balanced'" in note


def test_maybe_calibrate_skips_for_sample_weight_model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    calibration = {"ece": 0.5, "prob_true": [], "prob_pred": [], "bin_counts": [], "brier_score": 0.0}
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.integers(0, 2, 50))

    result, note = _maybe_calibrate(
        "HistGradientBoostingClassifier", HistGradientBoostingClassifier(), {},
        X, y, X, y, calibration, sample_weight_models={"HistGradientBoostingClassifier"},
    )
    assert result is None
    assert "sample_weight" in note


def test_maybe_calibrate_corrects_miscalibrated_model():
    """A deliberately miscalibrated model (trained on a heavily imbalanced
    target with class_weight='balanced') should have its ECE substantially
    reduced by the CalibratedClassifierCV comparison."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(6)
    n = 1200
    X = rng.standard_normal((n, 6))
    logits = 0.3 * X[:, 0] + 0.2 * X[:, 1] + rng.standard_normal(n) * 1.5
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < np.clip(p * 0.15, 0, 1)).astype(int)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y_s = pd.Series(y)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
    ).fit(X_df, y_s)
    raw_calibration = _compute_calibration(model, X_df, y_s)
    assert raw_calibration is not None

    result, note = _maybe_calibrate(
        "RandomForestClassifier",
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"n_estimators": 100, "max_depth": 5},
        X_df, y_s, X_df, y_s, raw_calibration, sample_weight_models=set(),
    )

    if raw_calibration["ece"] <= 0.05:
        pytest.skip("This particular random draw happened to be well-calibrated already")
    assert result is not None
    assert result["method"] in ("sigmoid", "isotonic")
    assert result["ece"] < raw_calibration["ece"]
    assert "CalibratedClassifierCV" in note
    assert "diagnostic only" in note


def test_agent_e2e_binary_classification_has_calibration_fields(classification_csv):
    agent = MLAgent()
    _, report_path = agent.run(classification_csv, target_col="label", id_col="row_id")
    report = agent.report_
    assert report.calibration is not None
    assert "ece" in report.calibration
    assert "class_weight='balanced'" in report.calibration_note

    with open(report_path) as f:
        parsed = json.load(f)
    assert "calibration" in parsed
    assert "calibrated_comparison" in parsed
    assert "calibration_note" in parsed


def test_agent_e2e_regression_calibration_not_applicable(regression_csv):
    agent = MLAgent()
    agent.run(regression_csv, target_col="price")
    report = agent.report_
    assert report.calibration is None
    assert report.calibrated_comparison is None
    assert "not applicable" in report.calibration_note.lower()


# ---------------------------------------------------------------------------
# GridSearchCV hyperparameter tuning
# ---------------------------------------------------------------------------

def test_grid_search_tries_multiple_configurations():
    """GridSearchCV must actually try every hyperparameter combination in the
    configured grid, not just the model's default configuration — and
    _run_grid_search must surface that same result."""
    from src.agents.ml_agent import _PARAM_GRIDS
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int))

    grid = _PARAM_GRIDS["LogisticRegression"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Reference GridSearchCV run: confirm every configured C value is tried.
    reference = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        grid, cv=cv, scoring="f1_macro", n_jobs=1,
    )
    reference.fit(X, y)
    assert len(reference.cv_results_["params"]) == len(grid["C"])
    assert reference.best_params_["C"] in grid["C"]

    # _run_grid_search must wire the same grid and surface the same winner.
    cv_scores, best_params, fitted, cv_fold_scores, cv_std = _run_grid_search(
        [("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42))],
        X, y, cv, "f1_macro",
    )
    assert best_params["LogisticRegression"] == reference.best_params_
    assert fitted["LogisticRegression"] is not None  # refit=True already fit it

    # New in F2: per-fold scores and their std must also be surfaced.
    assert len(cv_fold_scores["LogisticRegression"]) == 5  # 5-fold cv
    assert cv_std["LogisticRegression"] >= 0.0
    expected_std = float(np.std(cv_fold_scores["LogisticRegression"]))
    assert cv_std["LogisticRegression"] == pytest.approx(expected_std, abs=1e-4)


def test_linear_regression_skips_hyperparameter_grid():
    """LinearRegression has no grid — it must be plain-CV'd, not GridSearchCV'd,
    and left unfit for the caller to refit."""
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(8)
    n = 100
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] * 2 + X["b"] + rng.standard_normal(n) * 0.1)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores, best_params, fitted, cv_fold_scores, cv_std = _run_grid_search(
        [("LinearRegression", LinearRegression())],
        X, y, cv, "neg_root_mean_squared_error",
    )
    assert best_params["LinearRegression"] == {}
    assert fitted["LinearRegression"] is None
    assert len(cv_fold_scores["LinearRegression"]) == 5
    assert cv_std["LinearRegression"] >= 0.0


def test_agent_e2e_classification_best_hyperparameters_populated(classification_csv):
    """Every classification candidate has a grid, so whichever model wins
    must report a non-empty best_hyperparameters dict."""
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="row_id")
    report = agent.report_
    assert isinstance(report.best_hyperparameters, dict)
    assert len(report.best_hyperparameters) > 0


def test_agent_e2e_regression_best_hyperparameters(regression_csv):
    """best_hyperparameters is non-empty when a tuned model wins, and
    exactly empty when LinearRegression (the only untuned candidate) wins."""
    agent = MLAgent()
    agent.run(regression_csv, target_col="price")
    report = agent.report_
    if report.best_model_name == "LinearRegression":
        assert report.best_hyperparameters == {}
    else:
        assert len(report.best_hyperparameters) > 0


# ---------------------------------------------------------------------------
# Model selection picks genuinely best CV score
# ---------------------------------------------------------------------------

def test_model_selection_picks_best_cv_score_classification(tmp_path):
    rng = np.random.default_rng(5)
    n = 300
    X = rng.standard_normal((n, 5))
    # Linearly separable target — LogisticRegression should shine
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    path = str(tmp_path / "clf.csv")
    df.to_csv(path, index=False)

    agent = MLAgent()
    success, _ = agent.run(path, target_col="target")
    assert success is True
    assert agent.report_ is not None
    # Best model must have the highest CV score among all candidates
    best_name = agent.report_.best_model_name
    best_score = agent.report_.cv_scores[best_name]
    for name, score in agent.report_.cv_scores.items():
        assert best_score >= score, f"{best_name} is not better than {name}"


def test_model_selection_picks_best_cv_score_regression(tmp_path):
    rng = np.random.default_rng(6)
    n = 300
    X = rng.standard_normal((n, 5))
    y = X[:, 0] * 3 + X[:, 1] * 2 + rng.standard_normal(n) * 0.1  # very linear

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    path = str(tmp_path / "reg.csv")
    df.to_csv(path, index=False)

    agent = MLAgent()
    success, _ = agent.run(path, target_col="target")
    assert success is True
    best_name = agent.report_.best_model_name
    best_score = agent.report_.cv_scores[best_name]
    for name, score in agent.report_.cv_scores.items():
        assert best_score >= score, f"{best_name} is not better than {name}"


# ---------------------------------------------------------------------------
# End-to-end: synthetic classification dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_csv(tmp_path):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=400, n_features=8, n_informative=4,
        n_classes=2, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])
    df["label"] = y
    df["row_id"] = [f"id_{i}" for i in range(400)]
    path = tmp_path / "clf_e2e.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_agent_e2e_classification(classification_csv, tmp_path):
    agent = MLAgent()
    success, report_path = agent.run(
        classification_csv, target_col="label", id_col="row_id"
    )
    assert success is True
    assert os.path.exists(report_path)

    report = agent.report_
    assert report.task_type == "binary_classification"
    assert report.confusion_matrix is not None
    assert len(report.confusion_matrix) == 2
    assert report.test_metrics.get("f1_macro", 0) > 0
    assert "roc_auc" in report.test_metrics

    with open(report_path) as f:
        data = json.load(f)
    assert "cv_scores" in data
    assert "best_model_name" in data
    assert "threshold_metrics" in data
    assert "best_hyperparameters" in data


def test_agent_e2e_report_records_target_col(classification_csv, regression_csv):
    """F4: the report must record which column it predicted, so a page
    that loads the report later (a different session, a fresh reload)
    can show it without any Olist-specific hardcoded fallback."""
    clf_agent = MLAgent()
    _, clf_report_path = clf_agent.run(classification_csv, target_col="label", id_col="row_id")
    assert clf_agent.report_.target_col == "label"
    with open(clf_report_path) as f:
        assert json.load(f)["target_col"] == "label"

    reg_agent = MLAgent()
    _, reg_report_path = reg_agent.run(regression_csv, target_col="price")
    assert reg_agent.report_.target_col == "price"
    with open(reg_report_path) as f:
        assert json.load(f)["target_col"] == "price"


def test_agent_e2e_classification_threshold_metrics_structure(classification_csv):
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="row_id")
    tm = agent.report_.threshold_metrics
    assert tm is not None
    assert len(tm) == 3  # 0.3, 0.4, 0.5
    thresholds_seen = [entry["threshold"] for entry in tm]
    assert thresholds_seen == [0.3, 0.4, 0.5]
    for entry in tm:
        assert "f1_macro" in entry
        assert "precision_minority" in entry
        assert "recall_minority" in entry
        assert "confusion_matrix" in entry
        assert len(entry["confusion_matrix"]) == 2


def test_lower_threshold_increases_minority_recall(classification_csv):
    """Reducing the decision threshold should increase minority-class recall
    at the cost of lower precision — this verifies the tradeoff is captured."""
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="row_id")
    tm = {entry["threshold"]: entry for entry in agent.report_.threshold_metrics}
    # recall at 0.3 should be >= recall at 0.5
    assert tm[0.3]["recall_minority"] >= tm[0.5]["recall_minority"]


def test_agent_e2e_classification_id_col_excluded_from_features(classification_csv):
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="row_id")
    # row_id should not appear as a feature (it's an object dtype identifier)
    assert "row_id" not in agent.report_.feature_importances


# ---------------------------------------------------------------------------
# _extract_feature_importances: permutation importance, mean +/- std,
# zero-crossing flag (handbook F5)
# ---------------------------------------------------------------------------

class _StubModel:
    """Fit/predict/score no-op -- permutation_importance itself is
    monkeypatched in the tests that use this, so the model is never
    actually evaluated."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def score(self, X, y):
        return 0.0


def test_extract_feature_importances_uses_configured_repeat_count(monkeypatch):
    """The whole point of F5 is more repeats than the old default of 5 --
    assert the constant is actually >= 10 and is what gets passed through
    to sklearn's permutation_importance."""
    assert ml_agent._PERMUTATION_N_REPEATS >= 10

    captured = {}

    def fake_permutation_importance(model, X, y, n_repeats, random_state, n_jobs):
        captured["n_repeats"] = n_repeats
        return SimpleNamespace(
            importances_mean=np.zeros(X.shape[1]),
            importances_std=np.zeros(X.shape[1]),
        )

    monkeypatch.setattr(ml_agent, "permutation_importance", fake_permutation_importance)

    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    _extract_feature_importances(_StubModel(), ["a", "b"], X, pd.Series([0, 1, 0]))

    assert captured["n_repeats"] == ml_agent._PERMUTATION_N_REPEATS


def test_extract_feature_importances_reports_mean_and_std_not_point_estimate(monkeypatch):
    """Every feature must carry mean AND std -- not a single scalar."""
    def fake_permutation_importance(model, X, y, n_repeats, random_state, n_jobs):
        return SimpleNamespace(
            importances_mean=np.array([0.05, 0.02]),
            importances_std=np.array([0.01, 0.03]),
        )

    monkeypatch.setattr(ml_agent, "permutation_importance", fake_permutation_importance)

    X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
    result = _extract_feature_importances(_StubModel(), ["x1", "x2"], X, pd.Series([0, 1, 0]))

    for feat in ("x1", "x2"):
        assert set(result[feat]) == {
            "importance_mean", "importance_std", "distinguishable_from_zero",
        }
    assert result["x1"]["importance_mean"] == 0.05
    assert result["x1"]["importance_std"] == 0.01


def test_extract_feature_importances_flags_ranges_that_cross_zero(monkeypatch):
    """Deterministic check of the flagging rule itself, independent of any
    real model's noise: a feature's [mean-std, mean+std] range spanning
    zero must be flagged distinguishable_from_zero=False; a range that
    stays strictly on one side of zero must be True."""
    def fake_permutation_importance(model, X, y, n_repeats, random_state, n_jobs):
        return SimpleNamespace(
            # alpha: clearly positive and away from zero
            # bravo: positive mean, but std pulls the range below zero
            # charlie: negative mean, but std pulls the range above zero
            # delta: ~zero mean regardless of std
            importances_mean=np.array([0.05, 0.01, -0.02, 0.0]),
            importances_std=np.array([0.01, 0.02, 0.03, 0.001]),
        )

    monkeypatch.setattr(ml_agent, "permutation_importance", fake_permutation_importance)

    feature_names = ["alpha", "bravo", "charlie", "delta"]
    X = pd.DataFrame({name: [0, 1, 2, 3] for name in feature_names})
    result = _extract_feature_importances(
        _StubModel(), feature_names, X, pd.Series([0, 1, 0, 1]), top_n=10,
    )

    assert result["alpha"]["distinguishable_from_zero"] is True
    assert result["bravo"]["distinguishable_from_zero"] is False
    assert result["charlie"]["distinguishable_from_zero"] is False
    assert result["delta"]["distinguishable_from_zero"] is False


def test_extract_feature_importances_empty_eval_set_returns_empty_dict():
    empty_X = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=float)})
    empty_y = pd.Series(dtype=int)
    assert _extract_feature_importances(_StubModel(), ["a", "b"], empty_X, empty_y) == {}


def test_extract_feature_importances_generic_over_arbitrary_columns_and_count():
    """No hardcoded feature names/count: a dataset with 9 arbitrarily-named
    columns (none resembling this project's Olist schema) must work
    exactly the same way, respecting top_n and returning the same shape
    for every feature regardless of what it's called."""
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({f"custom_col_{i}": rng.standard_normal(n) for i in range(9)})
    y = pd.Series((df["custom_col_0"] + rng.standard_normal(n) * 0.5 > 0).astype(int))

    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=_RANDOM_STATE)
    model.fit(df, y)

    result = _extract_feature_importances(model, list(df.columns), df, y, top_n=5)

    assert 0 < len(result) <= 5
    assert set(result).issubset(set(df.columns))
    for feat, v in result.items():
        assert set(v) == {
            "importance_mean", "importance_std", "distinguishable_from_zero",
        }
        assert isinstance(v["importance_mean"], float)
        assert isinstance(v["importance_std"], float)
        assert isinstance(v["distinguishable_from_zero"], bool)
        assert v["importance_std"] >= 0


# ---------------------------------------------------------------------------
# Error analysis by segment (handbook F6)
# ---------------------------------------------------------------------------

class _FixedPredictModel:
    """predict() returns a pre-baked array -- lets a test dictate the exact
    confusion outcome per row instead of depending on a real model's fit."""

    def __init__(self, preds):
        self._preds = np.asarray(preds)

    def predict(self, X):
        return self._preds


def test_detect_segment_columns_generic_selects_only_low_cardinality_columns():
    """No hardcoded column names: selection is driven purely by how many
    distinct values a column has, so a binary flag and a 5-category column
    qualify while a near-unique continuous column does not -- regardless
    of what any of them are called."""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "binary_flag_xyz": rng.integers(0, 2, n),
        "five_category_abc": rng.integers(0, 5, n),
        "continuous_qqq": rng.standard_normal(n),
        "near_unique_id_like": np.arange(n),
    })

    detected = _detect_segment_columns(df)

    assert set(detected) == {"binary_flag_xyz", "five_category_abc"}


def test_detect_segment_columns_respects_max_cols_and_prefers_higher_cardinality():
    n = 500
    df = pd.DataFrame({
        f"cat_{k}": [i % k for i in range(n)] for k in (2, 3, 4, 5, 6)
    })
    detected = _detect_segment_columns(df, max_cols=2)
    assert detected == ["cat_6", "cat_5"]


def test_detect_segment_columns_empty_when_nothing_qualifies():
    df = pd.DataFrame({"continuous_a": np.random.default_rng(0).standard_normal(200)})
    assert _detect_segment_columns(df) == []


def test_error_analysis_classification_flags_concentrated_false_negative_segment():
    """Three segments, two clean (10% FN/FP) and one concentrated (90% FN)
    -- only the concentrated one should be flagged, and the overall rate
    must reflect all three pooled together."""
    def build(n_pos, n_neg, fn_count, fp_count):
        y_true = [1] * n_pos + [0] * n_neg
        pred_pos = [1] * (n_pos - fn_count) + [0] * fn_count
        pred_neg = [0] * (n_neg - fp_count) + [1] * fp_count
        return y_true, pred_pos + pred_neg

    a_true, a_pred = build(30, 30, 3, 3)
    b_true, b_pred = build(30, 30, 3, 3)
    c_true, c_pred = build(30, 30, 27, 3)

    y_test = pd.Series(a_true + b_true + c_true)
    y_pred = a_pred + b_pred + c_pred
    X_test = pd.DataFrame({"region": ["A"] * 60 + ["B"] * 60 + ["C"] * 60})

    result = _error_analysis_classification(
        _FixedPredictModel(y_pred), X_test, y_test, ["region"], "binary_classification",
    )

    rows = {r["segment_value"]: r for r in result["segments"]["region"]}
    assert rows["A"]["elevated_false_negative_rate"] is False
    assert rows["B"]["elevated_false_negative_rate"] is False
    assert rows["C"]["elevated_false_negative_rate"] is True
    assert all(not r["elevated_false_positive_rate"] for r in rows.values())
    assert result["overall"]["false_negative_rate"] == pytest.approx(33 / 90, rel=1e-6)


def test_error_analysis_classification_skips_segments_below_min_size():
    y_test = pd.Series([1, 0] * 40 + [1, 0])  # 82 rows: 80 in "big", 2 in "tiny"
    X_test = pd.DataFrame({"grp": ["big"] * 80 + ["tiny"] * 2})
    model = _FixedPredictModel(y_test.to_numpy())  # perfect predictions

    result = _error_analysis_classification(
        model, X_test, y_test, ["grp"], "binary_classification",
    )

    segment_values = {r["segment_value"] for r in result["segments"]["grp"]}
    assert segment_values == {"big"}  # "tiny" (n=2) dropped, below the min-size floor


def test_error_analysis_classification_multiclass_reports_misclassification_rate():
    y_test = pd.Series([0, 1, 2] * 20)
    y_pred = np.array(y_test.tolist())
    # Force every "2" in segment "x" to be misclassified.
    X_test = pd.DataFrame({"grp": (["x"] * 30) + (["y"] * 30)})
    y_pred[(y_test.to_numpy() == 2) & (X_test["grp"].to_numpy() == "x")] = 0

    result = _error_analysis_classification(
        _FixedPredictModel(y_pred), X_test, y_test, ["grp"], "multiclass_classification",
    )

    rows = {r["segment_value"]: r for r in result["segments"]["grp"]}
    assert rows["x"]["elevated_misclassification_rate"] is True
    assert rows["y"]["misclassification_rate"] == 0.0


def test_error_analysis_regression_flags_elevated_mae_and_bias():
    rng = np.random.default_rng(0)
    n_per_segment = 50
    y_true_a = rng.uniform(100, 200, n_per_segment)
    y_pred_a = y_true_a + rng.normal(0, 2, n_per_segment)          # tiny, unbiased error
    y_true_b = rng.uniform(100, 200, n_per_segment)
    y_pred_b = y_true_b - 80                                       # large, systematic under-prediction

    y_test = pd.Series(np.concatenate([y_true_a, y_true_b]))
    y_pred = np.concatenate([y_pred_a, y_pred_b])
    X_test = pd.DataFrame({"zone": ["low_error"] * n_per_segment + ["high_error"] * n_per_segment})

    result = _error_analysis_regression(X_test, y_test, y_pred, ["zone"])

    rows = {r["segment_value"]: r for r in result["segments"]["zone"]}
    assert rows["low_error"]["elevated_mae"] is False
    assert rows["high_error"]["elevated_mae"] is True
    assert rows["high_error"]["elevated_bias"] is True
    assert rows["high_error"]["mean_error"] > 0   # actual - predicted > 0 => under-prediction


def test_error_analysis_generic_over_arbitrary_columns_and_values():
    """Same genericity bar as feature importance: an arbitrary, non-Olist
    column name holding non-numeric segment values must work identically
    -- nothing in _error_analysis_classification assumes numeric-encoded
    segments, even though this project's real pipeline always produces
    them."""
    rng = np.random.default_rng(3)
    n = 400
    y = pd.Series(rng.integers(0, 2, n))
    y_pred = y.to_numpy().copy()
    # Deliberately corrupt predictions only within the "west" segment so it
    # has a visibly different error profile than the rest.
    df = pd.DataFrame({
        "widget_score": rng.standard_normal(n),
        "region_code": rng.choice(["north", "south", "east", "west"], n),
    })
    west_mask = (df["region_code"] == "west").to_numpy()
    y_pred[west_mask] = 1 - y_pred[west_mask]

    result = _error_analysis_classification(
        _FixedPredictModel(y_pred), df, y, ["region_code"], "binary_classification",
    )

    assert result["segment_columns"] == ["region_code"]
    seen_values = {r["segment_value"] for r in result["segments"]["region_code"]}
    assert seen_values <= {"north", "south", "east", "west"}
    for row in result["segments"]["region_code"]:
        assert {"n", "n_positive", "n_negative", "false_negative_rate",
                "false_positive_rate", "elevated_false_negative_rate",
                "elevated_false_positive_rate", "segment_value"} == set(row)
    west_row = next(r for r in result["segments"]["region_code"] if r["segment_value"] == "west")
    assert west_row["false_negative_rate"] == 1.0
    assert west_row["false_positive_rate"] == 1.0


def test_agent_e2e_error_analysis_present_with_no_qualifying_segments(classification_csv):
    """classification_csv's features are all continuous (make_classification) --
    no column should qualify as a segment axis, and the report must say so
    rather than silently omitting the field."""
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="row_id")

    ea = agent.report_.error_analysis
    assert ea["segment_columns"] == []
    assert ea["segments"] == {}
    assert "overall" in ea
    assert "No segment columns were configured or auto-detected" in ea["detection_note"]


@pytest.fixture
def classification_csv_with_segment(tmp_path):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=4, n_classes=2, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(6)])
    df["label"] = y
    df["row_id"] = [f"id_{i}" for i in range(600)]
    rng = np.random.default_rng(5)
    df["region_segment"] = rng.integers(0, 4, 600)  # low-cardinality -> auto-detectable
    path = tmp_path / "clf_segment_e2e.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_agent_e2e_error_analysis_auto_detects_segment_column(classification_csv_with_segment):
    agent = MLAgent()
    agent.run(classification_csv_with_segment, target_col="label", id_col="row_id")

    ea = agent.report_.error_analysis
    assert ea["segment_columns"] == ["region_segment"]
    assert "auto-detected" in ea["detection_note"]
    assert set(ea["segments"]) == {"region_segment"}
    assert len(ea["segments"]["region_segment"]) > 0
    for row in ea["segments"]["region_segment"]:
        assert row["n"] >= ml_agent._ERROR_ANALYSIS_MIN_SEGMENT_SIZE


def test_agent_e2e_error_analysis_respects_explicit_segment_cols_override(classification_csv_with_segment):
    """Explicit configuration must win over auto-detection."""
    agent = MLAgent(segment_cols=["feat_0"])  # continuous -- would never be auto-detected
    agent.run(classification_csv_with_segment, target_col="label", id_col="row_id")

    ea = agent.report_.error_analysis
    assert ea["segment_columns"] == ["feat_0"]
    assert "explicitly configured" in ea["detection_note"]


def test_agent_e2e_regression_error_analysis_uses_mae_and_bias(regression_csv):
    agent = MLAgent()
    agent.run(regression_csv, target_col="price")

    ea = agent.report_.error_analysis
    assert ea["task_type"] == "regression"
    assert "mae" in ea["overall"]
    assert "mean_error" in ea["overall"]


# ---------------------------------------------------------------------------
# Robustness check across random seeds (handbook F7)
# ---------------------------------------------------------------------------

def _make_report(
    best_model_name: str,
    test_metrics: dict,
    feature_importances: dict,
    task_type: str = "binary_classification",
) -> MLReport:
    return MLReport(
        task_type=task_type,
        best_model_name=best_model_name,
        test_metrics=test_metrics,
        feature_importances=feature_importances,
    )


def test_aggregate_robustness_flags_robust_when_model_agrees_every_seed():
    seed_reports = [
        (1, _make_report("HGB", {"f1_macro": 0.55}, {"a": {"importance_mean": 0.1}})),
        (2, _make_report("HGB", {"f1_macro": 0.57}, {"a": {"importance_mean": 0.09}})),
        (3, _make_report("HGB", {"f1_macro": 0.56}, {"a": {"importance_mean": 0.11}})),
    ]
    result = _aggregate_robustness(seed_reports, top_k_features=5)

    assert result["winning_model"] == "HGB"
    assert result["winning_model_seed_agreement"] == 3
    assert result["winning_model_is_robust"] is True
    assert result["model_agreement"] == {"HGB": 3}


def test_aggregate_robustness_flags_non_robust_when_model_disagrees():
    seed_reports = [
        (1, _make_report("HGB", {"f1_macro": 0.55}, {})),
        (2, _make_report("HGB", {"f1_macro": 0.57}, {})),
        (3, _make_report("RandomForest", {"f1_macro": 0.56}, {})),
    ]
    result = _aggregate_robustness(seed_reports)

    assert result["winning_model"] == "HGB"
    assert result["winning_model_seed_agreement"] == 2
    assert result["winning_model_is_robust"] is False
    assert result["model_agreement"] == {"HGB": 2, "RandomForest": 1}


def test_aggregate_robustness_computes_metric_summary_stats():
    seed_reports = [
        (1, _make_report("HGB", {"f1_macro": 0.50, "roc_auc": 0.70}, {})),
        (2, _make_report("HGB", {"f1_macro": 0.60, "roc_auc": 0.80}, {})),
    ]
    result = _aggregate_robustness(seed_reports)

    assert result["test_metrics_summary"]["f1_macro"]["mean"] == pytest.approx(0.55)
    assert result["test_metrics_summary"]["f1_macro"]["min"] == 0.50
    assert result["test_metrics_summary"]["f1_macro"]["max"] == 0.60
    assert result["test_metrics_summary"]["roc_auc"]["mean"] == pytest.approx(0.75)


def test_aggregate_robustness_identifies_always_vs_seed_dependent_features():
    seed_reports = [
        (1, _make_report("HGB", {}, {
            "stable_feat": {"importance_mean": 0.5},
            "flaky_feat": {"importance_mean": 0.09},
            "only_seed1": {"importance_mean": 0.08},
        })),
        (2, _make_report("HGB", {}, {
            "stable_feat": {"importance_mean": 0.4},
            "flaky_feat": {"importance_mean": 0.07},
            "only_seed2": {"importance_mean": 0.06},
        })),
    ]
    result = _aggregate_robustness(seed_reports, top_k_features=3)

    assert "stable_feat" in result["always_in_top_k_features"]
    assert "flaky_feat" in result["always_in_top_k_features"]
    assert "only_seed1" in result["seed_dependent_features"]
    assert "only_seed2" in result["seed_dependent_features"]
    assert "stable_feat" not in result["seed_dependent_features"]


def test_aggregate_robustness_generic_over_arbitrary_task_type_and_metrics():
    """No hardcoded metric names or task type -- a regression report with
    entirely different metric keys must summarize the same way."""
    seed_reports = [
        (10, _make_report(
            "LinearRegression", {"rmse": 100.0, "mae": 80.0},
            {"square_footage": {"importance_mean": 0.9}}, task_type="regression",
        )),
        (20, _make_report(
            "LinearRegression", {"rmse": 110.0, "mae": 85.0},
            {"square_footage": {"importance_mean": 0.85}}, task_type="regression",
        )),
    ]
    result = _aggregate_robustness(seed_reports)

    assert result["task_type"] == "regression"
    assert set(result["test_metrics_summary"]) == {"rmse", "mae"}
    assert "square_footage" in result["always_in_top_k_features"]


def test_random_state_override_changes_train_test_split(classification_csv):
    """Different random_state values passed through _prepare_and_train
    must actually produce different splits/results -- otherwise a
    robustness check across seeds would be theater, not a real check."""
    agent = MLAgent()
    ok1, report1, _ = agent._prepare_and_train(
        classification_csv, "label", id_col="row_id", random_state=1, save_model=False,
    )
    ok2, report2, _ = agent._prepare_and_train(
        classification_csv, "label", id_col="row_id", random_state=2, save_model=False,
    )
    assert ok1 and ok2
    # Different seeds should not produce byte-identical CV fold scores for
    # every candidate -- if they did, the seed wasn't actually threaded through.
    assert report1.cv_fold_scores != report2.cv_fold_scores


def test_prepare_and_train_save_model_false_does_not_touch_disk(classification_csv, tmp_path, monkeypatch):
    """save_model=False (used by every robustness-check seed) must never
    call joblib.dump -- it must not overwrite the one canonical production
    model a prior real run() produced."""
    import src.agents.ml_agent as mlmod

    calls = []
    monkeypatch.setattr(mlmod.joblib, "dump", lambda *a, **k: calls.append(a))

    agent = MLAgent()
    ok, report, err = agent._prepare_and_train(
        classification_csv, "label", id_col="row_id", random_state=1, save_model=False,
    )
    assert ok, err
    assert calls == []


def test_robustness_check_writes_report_with_expected_shape(classification_csv):
    agent = MLAgent()
    success, report_path = agent.run_robustness_check(
        classification_csv, target_col="label", id_col="row_id", seeds=(1, 2, 3),
    )
    assert success is True
    assert report_path.endswith("_robustness_report.json")
    assert os.path.exists(report_path)

    with open(report_path) as f:
        data = json.load(f)
    assert data["n_seeds"] == 3
    assert data["seeds"] == [1, 2, 3]
    assert "winning_model_is_robust" in data
    assert "test_metrics_summary" in data
    assert agent.robustness_report_ == data


def test_robustness_check_does_not_overwrite_production_model(classification_csv):
    """The whole point of save_model=False: a robustness sweep must not
    clobber the canonical model a real run() already serialized."""
    agent = MLAgent()
    success, report_path = agent.run(classification_csv, target_col="label", id_col="row_id")
    assert success is True
    model_path = os.path.join(ml_agent._MODEL_DIR, "best_production_model.pkl")
    mtime_before = os.path.getmtime(model_path)

    agent.run_robustness_check(classification_csv, target_col="label", id_col="row_id", seeds=(1, 2))

    assert os.path.getmtime(model_path) == mtime_before


def test_robustness_check_fails_with_missing_target_col(classification_csv):
    agent = MLAgent()
    success, message = agent.run_robustness_check(
        classification_csv, target_col="no_such_col", seeds=(1, 2),
    )
    assert success is False
    assert "no_such_col" in message


def test_robustness_check_requires_at_least_one_seed(classification_csv):
    agent = MLAgent()
    success, message = agent.run_robustness_check(
        classification_csv, target_col="label", seeds=(),
    )
    assert success is False


def test_robustness_check_generic_for_regression(regression_csv):
    """Same call, regression dataset -- proves this isn't classification-only."""
    agent = MLAgent()
    success, report_path = agent.run_robustness_check(
        regression_csv, target_col="price", seeds=(1, 2),
    )
    assert success is True
    with open(report_path) as f:
        data = json.load(f)
    assert data["task_type"] == "regression"
    assert "rmse" in data["test_metrics_summary"] or "mae" in data["test_metrics_summary"]


# ---------------------------------------------------------------------------
# Experiment tracking: ml_experiments table (handbook F9)
# ---------------------------------------------------------------------------

def test_run_logs_ml_experiment(classification_csv, isolate_audit_db):
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="row_id")

    rows = get_recent_experiments(db_path=isolate_audit_db)
    assert len(rows) == 1
    row = rows[0]
    assert row["data_path"] == classification_csv
    assert row["target_col"] == "label"
    assert row["task_type"] == "binary_classification"
    assert row["best_model_name"] == agent.report_.best_model_name
    assert row["random_state"] == 42
    assert row["n_features"] == agent.n_features_
    assert row["test_metrics"] == agent.report_.test_metrics
    assert row["cv_scores"] == agent.report_.cv_scores
    assert row["report_path"].endswith("_ml_report.json")


def test_run_logs_ml_experiment_with_split_and_group_col(grouped_classification_csv, isolate_audit_db):
    agent = MLAgent()
    agent.run(grouped_classification_csv, target_col="label", group_col="customer_unique_id")

    row = get_recent_experiments(db_path=isolate_audit_db)[0]
    assert row["split_strategy"] == "grouped"
    assert row["group_col"] == "customer_unique_id"


def test_run_robustness_check_does_not_log_ml_experiment(classification_csv, isolate_audit_db):
    """Robustness-check seeds are exploratory (save_model=False) -- they
    must not pollute the experiment-tracking history either."""
    agent = MLAgent()
    agent.run_robustness_check(
        classification_csv, target_col="label", id_col="row_id", seeds=(1, 2),
    )

    assert get_recent_experiments(db_path=isolate_audit_db) == []


def test_run_twice_appends_two_ml_experiment_rows(classification_csv, isolate_audit_db):
    """Unlike the JSON report (overwritten every call), history must
    accumulate -- that's the entire point of experiment tracking."""
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="row_id")
    agent.run(classification_csv, target_col="label", id_col="row_id")

    rows = get_recent_experiments(db_path=isolate_audit_db)
    assert len(rows) == 2


def test_run_logs_ml_experiment_generic_for_regression(regression_csv, isolate_audit_db):
    """No hardcoded metric names or task type in the logging call --
    a regression run's rmse/mae must round-trip the same way."""
    agent = MLAgent()
    agent.run(regression_csv, target_col="price")

    row = get_recent_experiments(db_path=isolate_audit_db)[0]
    assert row["task_type"] == "regression"
    assert "rmse" in row["test_metrics"]


# ---------------------------------------------------------------------------
# End-to-end: synthetic regression dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_csv(tmp_path):
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=400, n_features=8, n_informative=5, noise=10.0, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])
    df["price"] = y
    path = tmp_path / "reg_e2e.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_agent_e2e_regression(regression_csv):
    agent = MLAgent()
    success, report_path = agent.run(regression_csv, target_col="price")
    assert success is True

    report = agent.report_
    assert report.task_type == "regression"
    assert report.confusion_matrix is None
    assert report.threshold_metrics is None  # regression has no threshold sweep
    assert "rmse" in report.test_metrics
    assert "mae" in report.test_metrics
    assert "adjusted_r2" in report.test_metrics

    # test_predictions must be populated (VisualizationAgent's actual-vs-
    # predicted/residual charts read this) and match the held-out test size.
    assert report.test_predictions is not None
    n_test = round(400 * 0.2)  # MLAgent's default test_size
    assert len(report.test_predictions["actual"]) == n_test
    assert len(report.test_predictions["predicted"]) == n_test

    with open(report_path) as f:
        data = json.load(f)
    assert "test_predictions" in data
    assert data["test_predictions"]["actual"] is not None


def test_agent_e2e_classification_has_no_test_predictions(classification_csv):
    """test_predictions is a regression-only field -- classification reports
    already have confusion_matrix/threshold_metrics for diagnostics and
    should not carry a redundant (and here, meaningless) predictions list."""
    agent = MLAgent()
    agent.run(classification_csv, target_col="label", id_col="row_id")
    assert agent.report_.test_predictions is None


# ---------------------------------------------------------------------------
# Class weighting: minority class recall
# ---------------------------------------------------------------------------

def test_class_weighting_minority_recall_nonzero(tmp_path):
    """With class_weight='balanced', the model must detect at least some
    minority-class samples across the threshold sweep (recall_minority > 0
    at threshold 0.3). The data has real signal so the model can learn.

    Without balancing, models predict the majority class for every row on
    imbalanced data, giving recall = 0 at every threshold. class_weight
    forces the model to trade precision for recall on the minority class."""
    rng = np.random.default_rng(42)
    n = 800
    X = rng.standard_normal((n, 6))
    # Minority class (10%) is strongly predictable from the first two features
    y = np.where((X[:, 0] + X[:, 1] > 2.5), 1, 0)   # ~5-10% positives, separable
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    path = str(tmp_path / "imbalanced_signal.csv")
    df.to_csv(path, index=False)

    agent = MLAgent()
    success, _ = agent.run(path, target_col="target")
    assert success is True

    # Check recall at the lowest threshold (0.3) from the threshold sweep —
    # class weighting must push at least some probability mass onto positives.
    tm = {entry["threshold"]: entry for entry in agent.report_.threshold_metrics}
    minority_recall_at_03 = tm[0.3]["recall_minority"]
    assert minority_recall_at_03 > 0.0, (
        f"Expected minority recall > 0 at threshold=0.3 with class_weight='balanced', "
        f"got {minority_recall_at_03:.1%}."
    )


def test_agent_e2e_regression_model_file_exists(regression_csv):
    agent = MLAgent()
    agent.run(regression_csv, target_col="price")
    model_path = os.path.join(ml_agent._MODEL_DIR, "best_production_model.pkl")
    assert os.path.exists(model_path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_agent_handles_nonexistent_file():
    agent = MLAgent()
    success, message = agent.run("no_such_file.csv", target_col="y")
    assert success is False
    assert "Failed to read" in message


def test_agent_handles_missing_target_column(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = str(tmp_path / "data.csv")
    df.to_csv(path, index=False)
    agent = MLAgent()
    success, message = agent.run(path, target_col="nonexistent")
    assert success is False
    assert "nonexistent" in message
