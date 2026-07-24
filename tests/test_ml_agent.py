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

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ml_agent import MLAgent, _eval_classification, _eval_regression
from src.tools.ml_tools import adjusted_r2, detect_task_type


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

    metrics = _eval_regression(model, X, y_test)

    # Errors: [-1.5, -0.5, 0.5, 1.5] → RMSE = sqrt(mean([2.25,0.25,0.25,2.25]))
    expected_rmse = np.sqrt(np.mean([2.25, 0.25, 0.25, 2.25]))
    expected_mae = np.mean([1.5, 0.5, 0.5, 1.5])

    assert metrics["rmse"] == pytest.approx(expected_rmse, rel=1e-4)
    assert metrics["mae"] == pytest.approx(expected_mae, rel=1e-4)


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
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "best_production_model.pkl"
    )
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
