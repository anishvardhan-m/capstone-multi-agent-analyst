"""
tests/test_visualizer.py

Unit tests for the Visualization Agent (src/agents/visualizer.py).

Each chart-generating method is tested in isolation: we call it directly
with synthetic inputs, then assert the output PNG exists and is non-trivial
in size (> 5 KB, ruling out blank/corrupt files).

Run with:
    pytest tests/test_visualizer.py -v
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.visualizer import VisualizationAgent, VisualizationReport

_MIN_FILE_BYTES = 5_000   # any real PNG must be larger than this


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "charts"
    # Deliberately do NOT pre-create it — tests verify agent creates it
    return str(d)


@pytest.fixture
def agent():
    return VisualizationAgent(top_dist_features=3, top_importance_features=5)


@pytest.fixture
def simple_df():
    """Small numeric DataFrame with a binary target."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feature_a": rng.exponential(scale=50, size=200),
        "feature_b": rng.standard_normal(200) * 10 + 30,
        "feature_c": rng.uniform(0, 100, 200),
        "category": rng.choice(["x", "y", "z"], 200),
        "is_late_delivery": rng.choice([0, 1], 200, p=[0.9, 0.1]),
    })


@pytest.fixture
def sample_feature_importances():
    return {
        "feature_a": 0.045,
        "feature_b": 0.030,
        "feature_c": 0.010,
        "category": -0.002,
    }


@pytest.fixture
def sample_corr_matrix():
    rng = np.random.default_rng(1)
    cols = ["feature_a", "feature_b", "feature_c"]
    data = pd.DataFrame(rng.standard_normal((100, 3)), columns=cols)
    corr = data.corr().to_dict()
    return corr


@pytest.fixture
def sample_confusion_matrix():
    return [[170, 20], [8, 2]]


@pytest.fixture
def sample_threshold_metrics():
    return [
        {"threshold": 0.3, "f1_macro": 0.38, "precision_minority": 0.12, "recall_minority": 0.90, "confusion_matrix": [[100, 70], [5, 45]]},
        {"threshold": 0.4, "f1_macro": 0.50, "precision_minority": 0.16, "recall_minority": 0.80, "confusion_matrix": [[130, 40], [9, 41]]},
        {"threshold": 0.5, "f1_macro": 0.57, "precision_minority": 0.20, "recall_minority": 0.65, "confusion_matrix": [[155, 15], [18, 32]]},
    ]


# ---------------------------------------------------------------------------
# Chart 1 — Distributions
# ---------------------------------------------------------------------------

def test_distributions_creates_nonempty_png(agent, simple_df, sample_feature_importances, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_distributions(simple_df, sample_feature_importances, output_dir, report)
    assert len(report.charts) == 1
    path = report.charts[0]["path"]
    assert os.path.exists(path)
    assert os.path.getsize(path) > _MIN_FILE_BYTES


def test_distributions_falls_back_to_variance_when_no_importances(agent, simple_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_distributions(simple_df, {}, output_dir, report)
    # Should still produce a chart (variance-based fallback)
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES


def test_distributions_respects_top_n(agent, simple_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent.top_dist_features = 2
    agent._chart_distributions(simple_df, {}, output_dir, report)
    assert len(report.charts) == 1   # one multi-panel PNG


# ---------------------------------------------------------------------------
# Chart 2 — Correlation heatmap
# ---------------------------------------------------------------------------

def test_correlation_heatmap_creates_nonempty_png(agent, sample_corr_matrix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_correlation_heatmap(sample_corr_matrix, output_dir, report)
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES


def test_correlation_heatmap_skips_gracefully_on_empty_input(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_correlation_heatmap({}, output_dir, report)
    assert len(report.charts) == 0
    assert len(report.skipped) == 1
    assert report.skipped[0]["name"] == "correlation_heatmap"


# ---------------------------------------------------------------------------
# Chart 3 — Feature importance
# ---------------------------------------------------------------------------

def test_feature_importance_creates_nonempty_png(agent, sample_feature_importances, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_feature_importance(sample_feature_importances, output_dir, report)
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES


def test_feature_importance_skips_on_empty_dict(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_feature_importance({}, output_dir, report)
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "feature_importance"


def test_feature_importance_respects_top_n(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    importances = {f"feat_{i}": float(10 - i) for i in range(20)}
    report = VisualizationReport()
    agent.top_importance_features = 5
    agent._chart_feature_importance(importances, output_dir, report)
    assert len(report.charts) == 1


# ---------------------------------------------------------------------------
# Chart 4 — Confusion matrix
# ---------------------------------------------------------------------------

def test_confusion_matrix_creates_nonempty_png(agent, sample_confusion_matrix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_confusion_matrix(sample_confusion_matrix, output_dir, report)
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES


def test_confusion_matrix_skips_on_none(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_confusion_matrix(None, output_dir, report)
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "confusion_matrix"


def test_confusion_matrix_skips_on_empty_list(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_confusion_matrix([], output_dir, report)
    assert len(report.charts) == 0


# ---------------------------------------------------------------------------
# Chart 5 — Threshold tradeoff
# ---------------------------------------------------------------------------

def test_threshold_tradeoff_creates_nonempty_png(agent, sample_threshold_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_threshold_tradeoff(sample_threshold_metrics, output_dir, report)
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES


def test_threshold_tradeoff_skips_on_none(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_threshold_tradeoff(None, output_dir, report)
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "threshold_tradeoff"


# ---------------------------------------------------------------------------
# Chart 6 — Top feature vs. target box plot
# ---------------------------------------------------------------------------

def test_top_feature_vs_target_creates_nonempty_png(
    agent, simple_df, sample_feature_importances, output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_top_feature_vs_target(
        simple_df, "is_late_delivery", sample_feature_importances, output_dir, report
    )
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES


def test_top_feature_vs_target_skips_on_empty_importances(agent, simple_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_top_feature_vs_target(simple_df, "is_late_delivery", {}, output_dir, report)
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "top_feature_vs_target"


def test_top_feature_vs_target_skips_when_target_missing(
    agent, simple_df, sample_feature_importances, output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_top_feature_vs_target(
        simple_df, "nonexistent_col", sample_feature_importances, output_dir, report
    )
    assert len(report.charts) == 0


# ---------------------------------------------------------------------------
# VisualizationAgent.run — end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def e2e_csv(tmp_path):
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        "feat_a": rng.exponential(10, n),
        "feat_b": rng.standard_normal(n),
        "feat_c": rng.uniform(0, 50, n),
        "city": rng.choice(["SP", "RJ", "BH"], n),
        "is_late_delivery": rng.choice([0, 1], n, p=[0.9, 0.1]),
    })
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def e2e_eda_report(tmp_path, e2e_csv):
    df = pd.read_csv(e2e_csv)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric].corr().to_dict()
    report = {
        "input_shape": list(df.shape),
        "numeric_columns": numeric,
        "categorical_columns": ["city"],
        "descriptive_stats": {},
        "correlation_matrix": corr,
        "skewness": {},
        "outlier_summary": {},
    }
    p = tmp_path / "eda_report.json"
    p.write_text(json.dumps(report))
    return str(p)


@pytest.fixture
def e2e_ml_report(tmp_path):
    report = {
        "task_type": "binary_classification",
        "best_model_name": "RandomForest",
        "cv_scores": {"RandomForest": 0.55},
        "test_metrics": {"f1_macro": 0.55, "roc_auc": 0.72},
        "confusion_matrix": [[130, 20], [10, 5]],
        "threshold_metrics": [
            {"threshold": 0.3, "f1_macro": 0.4, "precision_minority": 0.10, "recall_minority": 0.80, "confusion_matrix": [[100, 50], [3, 12]]},
            {"threshold": 0.4, "f1_macro": 0.5, "precision_minority": 0.14, "recall_minority": 0.70, "confusion_matrix": [[115, 35], [4, 11]]},
            {"threshold": 0.5, "f1_macro": 0.55, "precision_minority": 0.20, "recall_minority": 0.50, "confusion_matrix": [[140, 10], [7, 8]]},
        ],
        "feature_importances": {"feat_a": 0.04, "feat_b": 0.02, "feat_c": 0.01},
    }
    p = tmp_path / "ml_report.json"
    p.write_text(json.dumps(report))
    return str(p)


def test_agent_run_creates_output_dir(agent, e2e_eda_report, e2e_ml_report, e2e_csv, tmp_path):
    out = str(tmp_path / "new_viz_dir")
    assert not os.path.exists(out)
    success, _ = agent.run(e2e_eda_report, e2e_ml_report, e2e_csv, output_dir=out)
    assert success is True
    assert os.path.isdir(out)


def test_agent_run_generates_all_six_charts(agent, e2e_eda_report, e2e_ml_report, e2e_csv, tmp_path):
    out = str(tmp_path / "viz")
    agent.run(e2e_eda_report, e2e_ml_report, e2e_csv, output_dir=out)
    assert len(agent.report_.charts) == 6


def test_agent_run_all_pngs_nonempty(agent, e2e_eda_report, e2e_ml_report, e2e_csv, tmp_path):
    out = str(tmp_path / "viz")
    agent.run(e2e_eda_report, e2e_ml_report, e2e_csv, output_dir=out)
    for chart in agent.report_.charts:
        assert os.path.getsize(chart["path"]) > _MIN_FILE_BYTES, (
            f"{chart['name']} PNG is suspiciously small"
        )


def test_agent_run_writes_valid_report_json(agent, e2e_eda_report, e2e_ml_report, e2e_csv, tmp_path):
    out = str(tmp_path / "viz")
    _, report_path = agent.run(e2e_eda_report, e2e_ml_report, e2e_csv, output_dir=out)
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert "charts" in data
    assert "skipped" in data
    assert len(data["charts"]) == 6


def test_agent_run_no_crash_with_empty_feature_importances(
    agent, e2e_eda_report, e2e_csv, tmp_path
):
    """Empty feature_importances must not crash the agent — affected charts skipped."""
    out = str(tmp_path / "viz_no_fi")
    report_data = {
        "task_type": "binary_classification",
        "best_model_name": "X",
        "cv_scores": {},
        "test_metrics": {},
        "confusion_matrix": [[90, 10], [5, 5]],
        "threshold_metrics": None,
        "feature_importances": {},
    }
    ml_path = str(tmp_path / "ml_empty.json")
    with open(ml_path, "w") as f:
        json.dump(report_data, f)

    success, _ = agent.run(e2e_eda_report, ml_path, e2e_csv, output_dir=out)
    assert success is True
    # feature_importance and top_feature_vs_target should be skipped
    skipped_names = {s["name"] for s in agent.report_.skipped}
    assert "feature_importance" in skipped_names
    assert "top_feature_vs_target" in skipped_names


def test_agent_run_no_crash_with_missing_ml_report(agent, e2e_eda_report, e2e_csv, tmp_path):
    out = str(tmp_path / "viz_no_ml")
    success, _ = agent.run(
        e2e_eda_report, "nonexistent_ml_report.json", e2e_csv, output_dir=out
    )
    assert success is True  # graceful degradation, not a crash
