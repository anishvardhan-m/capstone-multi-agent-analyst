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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.visualizer import VisualizationAgent, VisualizationReport

_MIN_FILE_BYTES = 5_000   # any real PNG must be larger than this


def _perm_imp(means: dict) -> dict:
    """Build a permutation-importance dict (mean/std/flag), matching
    MLReport.feature_importances' shape, from a flat name->mean mapping."""
    result = {}
    for name, mean in means.items():
        std = abs(mean) * 0.1 if mean else 0.001
        result[name] = {
            "importance_mean": mean,
            "importance_std": std,
            "distinguishable_from_zero": not (mean - std <= 0 <= mean + std),
        }
    return result


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


def test_constructor_defaults_are_generic_not_olist_specific():
    agent = VisualizationAgent()
    assert agent.positive_label == "Positive"
    assert agent.negative_label == "Negative"
    assert agent.unit_label == "record"


def test_constructor_accepts_custom_labels():
    agent = VisualizationAgent(positive_label="Late", negative_label="On-time", unit_label="order")
    assert agent.positive_label == "Late"
    assert agent.negative_label == "On-time"
    assert agent.unit_label == "order"


@pytest.fixture
def keep_figure_open(monkeypatch):
    """Prevent a chart method's internal plt.close(fig) from discarding
    the figure, so the test can inspect tick labels/titles/legend text via
    plt.gcf() immediately afterward. Real cleanup happens in the fixture's
    teardown instead."""
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    yield
    plt.close("all")


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
    return _perm_imp({
        "feature_a": 0.045,
        "feature_b": 0.030,
        "feature_c": 0.010,
        "category": -0.002,
    })


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


@pytest.fixture
def sample_calibration():
    return {
        "prob_true": [0.05, 0.20, 0.45, 0.70, 0.90],
        "prob_pred": [0.10, 0.25, 0.40, 0.65, 0.85],
        "bin_counts": [40, 40, 40, 40, 40],
        "brier_score": 0.08,
        "ece": 0.06,
    }


@pytest.fixture
def sample_calibrated_comparison():
    return {
        "method": "isotonic",
        "prob_true": [0.06, 0.21, 0.44, 0.68, 0.89],
        "prob_pred": [0.07, 0.22, 0.43, 0.67, 0.88],
        "bin_counts": [40, 40, 40, 40, 40],
        "brier_score": 0.05,
        "ece": 0.02,
    }


@pytest.fixture
def sample_test_predictions():
    """Synthetic regression actual/predicted pairs (handbook Section 8.2 genericity)."""
    rng = np.random.default_rng(2)
    actual = rng.uniform(50, 500, 150).tolist()
    predicted = [a + n for a, n in zip(actual, rng.standard_normal(150) * 20)]
    return {"actual": actual, "predicted": predicted}


@pytest.fixture
def regression_df():
    """Small numeric DataFrame with a continuous target."""
    rng = np.random.default_rng(3)
    square_footage = rng.uniform(800, 4000, 200)
    price = square_footage * 150 + rng.standard_normal(200) * 20000 + 50000
    return pd.DataFrame({
        "square_footage": square_footage,
        "num_bedrooms": rng.integers(1, 6, 200),
        "price": price,
    })


@pytest.fixture
def regression_feature_importances():
    return _perm_imp({"square_footage": 0.42, "num_bedrooms": 0.05})


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
    importances = _perm_imp({f"feat_{i}": float(10 - i) for i in range(20)})
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


def test_confusion_matrix_uses_generic_defaults_when_no_labels_given(
    agent, sample_confusion_matrix, output_dir, keep_figure_open
):
    """VisualizationAgent() with no labels must still produce a coherent
    chart -- generic 'Positive'/'Negative' ticks, not a crash or a leaked
    Olist-specific default."""
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_confusion_matrix(sample_confusion_matrix, output_dir, report)

    ax = plt.gcf().axes[0]
    tick_texts = [t.get_text() for t in ax.get_xticklabels()]
    assert "Positive (1)" in tick_texts
    assert "Negative (0)" in tick_texts
    assert "Late (1)" not in tick_texts


def test_confusion_matrix_uses_custom_labels_when_provided(
    sample_confusion_matrix, output_dir, keep_figure_open
):
    """Regression test: a genericity fix previously replaced this
    project's approved 'On-time'/'Late' confusion-matrix labels with
    generic 'Negative'/'Positive' unconditionally. Custom labels passed to
    the constructor must now appear in the actual chart ticks."""
    os.makedirs(output_dir, exist_ok=True)
    custom_agent = VisualizationAgent(positive_label="Late", negative_label="On-time")
    report = VisualizationReport()
    custom_agent._chart_confusion_matrix(sample_confusion_matrix, output_dir, report)

    ax = plt.gcf().axes[0]
    tick_texts = [t.get_text() for t in ax.get_xticklabels()]
    assert "Late (1)" in tick_texts
    assert "On-time (0)" in tick_texts
    assert "Positive (1)" not in tick_texts


# ---------------------------------------------------------------------------
# Chart 5 — Threshold tradeoff
# ---------------------------------------------------------------------------

def test_threshold_tradeoff_creates_nonempty_png(agent, sample_threshold_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_threshold_tradeoff(sample_threshold_metrics, output_dir, report)
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES


def test_threshold_tradeoff_uses_generic_default_positive_label(
    agent, sample_threshold_metrics, output_dir, keep_figure_open
):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_threshold_tradeoff(sample_threshold_metrics, output_dir, report)

    ax = plt.gcf().axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Positive Recall" in legend_texts
    assert "Positive Precision" in legend_texts
    assert "Positive" in ax.get_title()


def test_threshold_tradeoff_uses_custom_positive_label_when_provided(
    sample_threshold_metrics, output_dir, keep_figure_open
):
    os.makedirs(output_dir, exist_ok=True)
    custom_agent = VisualizationAgent(positive_label="Late")
    report = VisualizationReport()
    custom_agent._chart_threshold_tradeoff(sample_threshold_metrics, output_dir, report)

    ax = plt.gcf().axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Late Recall" in legend_texts
    assert "Late Precision" in legend_texts
    assert "Late" in ax.get_title()


def test_threshold_tradeoff_skips_on_none(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_threshold_tradeoff(None, output_dir, report)
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "threshold_tradeoff"


# ---------------------------------------------------------------------------
# Chart 7 — Calibration curve (F3)
# ---------------------------------------------------------------------------

def test_calibration_curve_creates_nonempty_png(agent, sample_calibration, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_calibration_curve(sample_calibration, None, output_dir, report)
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES
    assert report.charts[0]["name"] == "calibration_curve"


def test_calibration_curve_overlays_calibrated_comparison(
    agent, sample_calibration, sample_calibrated_comparison, output_dir, keep_figure_open
):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_calibration_curve(
        sample_calibration, sample_calibrated_comparison, output_dir, report
    )

    ax = plt.gcf().axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Raw model" in t for t in legend_texts)
    assert any("Isotonic-calibrated" in t for t in legend_texts)
    assert any("Perfectly calibrated" in t for t in legend_texts)


def test_calibration_curve_no_comparison_overlay_when_absent(
    agent, sample_calibration, output_dir, keep_figure_open
):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_calibration_curve(sample_calibration, None, output_dir, report)

    ax = plt.gcf().axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    # Only the reference diagonal + raw-model curve -- no comparison line.
    assert legend_texts == ["Perfectly calibrated", "Raw model (ECE=0.060)"]
    assert len(ax.get_lines()) == 2


def test_calibration_curve_skips_on_none(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_calibration_curve(None, None, output_dir, report)
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "calibration_curve"


def test_calibration_curve_skips_on_empty_dict(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_calibration_curve({}, None, output_dir, report)
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "calibration_curve"


# ---------------------------------------------------------------------------
# Chart 4 (regression variant) — Actual vs. predicted scatter
# (handbook Section 8.2 genericity: classification/regression parity)
# ---------------------------------------------------------------------------

def test_actual_vs_predicted_creates_nonempty_png(agent, sample_test_predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_actual_vs_predicted(sample_test_predictions, output_dir, report)
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES


def test_actual_vs_predicted_skips_on_none(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_actual_vs_predicted(None, output_dir, report)
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "actual_vs_predicted"


def test_actual_vs_predicted_skips_on_empty_lists(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_actual_vs_predicted({"actual": [], "predicted": []}, output_dir, report)
    assert len(report.charts) == 0


# ---------------------------------------------------------------------------
# Chart 5 (regression variant) — Residual plot
# ---------------------------------------------------------------------------

def test_residuals_creates_nonempty_png(agent, sample_test_predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_residuals(sample_test_predictions, output_dir, report)
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES


def test_residuals_skips_on_none(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_residuals(None, output_dir, report)
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "residuals"


def test_residuals_skips_on_empty_lists(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_residuals({"actual": [], "predicted": []}, output_dir, report)
    assert len(report.charts) == 0


# ---------------------------------------------------------------------------
# Chart 6 — Top feature vs. target: box plot (classification) vs.
# scatter (regression), selected by task_type
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


def test_top_feature_vs_target_uses_boxplot_for_classification(
    agent, simple_df, sample_feature_importances, output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_top_feature_vs_target(
        simple_df, "is_late_delivery", sample_feature_importances, output_dir, report,
        task_type="binary_classification",
    )
    assert len(report.charts) == 1
    assert "Box plot" in report.charts[0]["description"]


def test_top_feature_vs_target_uses_scatter_for_regression(
    agent, regression_df, regression_feature_importances, output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_top_feature_vs_target(
        regression_df, "price", regression_feature_importances, output_dir, report,
        task_type="regression",
    )
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES
    assert "Scatter plot" in report.charts[0]["description"]


def test_top_feature_vs_target_boxplot_uses_generic_default_labels(
    agent, simple_df, sample_feature_importances, output_dir, keep_figure_open
):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_top_feature_vs_target(
        simple_df, "is_late_delivery", sample_feature_importances, output_dir, report,
        task_type="binary_classification",
    )
    ax = plt.gcf().axes[0]
    tick_texts = [t.get_text() for t in ax.get_xticklabels()]
    assert "Positive (1)" in tick_texts
    assert "Negative (0)" in tick_texts
    assert "Record" in ax.get_xlabel()


def test_top_feature_vs_target_boxplot_uses_custom_labels_when_provided(
    simple_df, sample_feature_importances, output_dir, keep_figure_open
):
    """Regression test: custom labels must reach the box plot's tick
    labels and axis text too, not just the confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)
    custom_agent = VisualizationAgent(
        positive_label="Late", negative_label="On-time", unit_label="order",
    )
    report = VisualizationReport()
    custom_agent._chart_top_feature_vs_target(
        simple_df, "is_late_delivery", sample_feature_importances, output_dir, report,
        task_type="binary_classification",
    )
    ax = plt.gcf().axes[0]
    tick_texts = [t.get_text() for t in ax.get_xticklabels()]
    assert "Late (1)" in tick_texts
    assert "On-time (0)" in tick_texts
    assert "Order" in ax.get_xlabel()


# ---------------------------------------------------------------------------
# Chart 8 — Error rate by segment (handbook F6)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_error_analysis_binary():
    return {
        "task_type": "binary_classification",
        "segment_columns": ["region_code", "channel"],
        "overall": {"false_negative_rate": 0.30, "false_positive_rate": 0.20},
        "segments": {
            "region_code": [
                {"segment_value": 0.01, "n": 100, "n_positive": 40, "n_negative": 60,
                 "false_negative_rate": 0.30, "false_positive_rate": 0.20,
                 "elevated_false_negative_rate": False, "elevated_false_positive_rate": False},
                {"segment_value": 0.42, "n": 200, "n_positive": 80, "n_negative": 120,
                 "false_negative_rate": 0.70, "false_positive_rate": 0.15,
                 "elevated_false_negative_rate": True, "elevated_false_positive_rate": False},
            ],
            "channel": [
                {"segment_value": 1.0, "n": 150, "n_positive": 60, "n_negative": 90,
                 "false_negative_rate": 0.28, "false_positive_rate": 0.18,
                 "elevated_false_negative_rate": False, "elevated_false_positive_rate": False},
            ],
        },
        "detection_note": "Segment columns auto-detected...",
        "note": "Segments with fewer than 20 relevant rows are omitted...",
    }


@pytest.fixture
def sample_error_analysis_regression():
    return {
        "task_type": "regression",
        "segment_columns": ["zone"],
        "overall": {"mae": 10.0, "mean_error": 1.0},
        "segments": {
            "zone": [
                {"segment_value": 0.1, "n": 100, "mae": 9.0, "mean_error": 0.5,
                 "elevated_mae": False, "elevated_bias": False},
                {"segment_value": 0.9, "n": 100, "mae": 25.0, "mean_error": 15.0,
                 "elevated_mae": True, "elevated_bias": True},
            ],
        },
        "detection_note": "Segment columns auto-detected...",
        "note": "Segments with fewer than 20 test rows are omitted...",
    }


def test_error_by_segment_creates_nonempty_png(agent, sample_error_analysis_binary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_error_by_segment(sample_error_analysis_binary, output_dir, report)
    assert len(report.charts) == 1
    assert os.path.getsize(report.charts[0]["path"]) > _MIN_FILE_BYTES


def test_error_by_segment_plots_at_most_two_columns(agent, sample_error_analysis_binary, output_dir, keep_figure_open):
    """error_analysis has 2 segment_columns -- both (already <= 2) should
    get their own panel; a 3rd would be dropped."""
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_error_by_segment(sample_error_analysis_binary, output_dir, report)
    fig = plt.gcf()
    assert len(fig.axes) == 2
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["region_code", "channel"]


def test_error_by_segment_marks_elevated_segments_in_tick_labels(
    agent, sample_error_analysis_binary, output_dir, keep_figure_open
):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_error_by_segment(sample_error_analysis_binary, output_dir, report)
    ax = plt.gcf().axes[0]  # region_code panel
    tick_texts = [t.get_text() for t in ax.get_xticklabels()]
    assert "0.42*" in tick_texts   # elevated_false_negative_rate=True
    assert "0.01" in tick_texts    # not elevated -- no asterisk


def test_error_by_segment_regression_uses_mae(agent, sample_error_analysis_regression, output_dir, keep_figure_open):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_error_by_segment(sample_error_analysis_regression, output_dir, report)
    ax = plt.gcf().axes[0]
    assert ax.get_ylabel() == "MAE"
    tick_texts = [t.get_text() for t in ax.get_xticklabels()]
    assert "0.9*" in tick_texts
    assert "0.1" in tick_texts


def test_error_by_segment_skips_on_empty_dict(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_error_by_segment({}, output_dir, report)
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "error_by_segment"


def test_error_by_segment_skips_when_segments_empty(agent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = VisualizationReport()
    agent._chart_error_by_segment(
        {"task_type": "binary_classification", "segment_columns": ["x"], "segments": {}, "overall": {}},
        output_dir, report,
    )
    assert len(report.charts) == 0
    assert report.skipped[0]["name"] == "error_by_segment"


def test_error_by_segment_generic_over_arbitrary_column_and_values(agent, output_dir, keep_figure_open):
    """Non-Olist column name, string segment values, multiclass task --
    nothing here is hardcoded to this project's schema."""
    os.makedirs(output_dir, exist_ok=True)
    error_analysis = {
        "task_type": "multiclass_classification",
        "segment_columns": ["shipping_carrier"],
        "overall": {"misclassification_rate": 0.15},
        "segments": {
            "shipping_carrier": [
                {"segment_value": "carrier_x", "n": 50, "misclassification_rate": 0.14,
                 "elevated_misclassification_rate": False},
                {"segment_value": "carrier_y", "n": 50, "misclassification_rate": 0.40,
                 "elevated_misclassification_rate": True},
            ],
        },
    }
    report = VisualizationReport()
    agent._chart_error_by_segment(error_analysis, output_dir, report)
    assert len(report.charts) == 1
    ax = plt.gcf().axes[0]
    assert ax.get_ylabel() == "Misclassification rate"
    tick_texts = [t.get_text() for t in ax.get_xticklabels()]
    assert "carrier_y*" in tick_texts
    assert "carrier_x" in tick_texts


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
        "feature_importances": _perm_imp({"feat_a": 0.04, "feat_b": 0.02, "feat_c": 0.01}),
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
    """e2e_ml_report has no 'calibration' key -- the 7th chart is skipped,
    not generated, so this stays at 6 (see the calibration-present variant
    below for the 7-chart case)."""
    out = str(tmp_path / "viz")
    agent.run(e2e_eda_report, e2e_ml_report, e2e_csv, output_dir=out)
    assert len(agent.report_.charts) == 6
    chart_names = [c["name"] for c in agent.report_.charts]
    assert "calibration_curve" not in chart_names
    assert any(s["name"] == "calibration_curve" for s in agent.report_.skipped)


@pytest.fixture
def e2e_ml_report_with_calibration(tmp_path):
    report = {
        "task_type": "binary_classification",
        "best_model_name": "RandomForest",
        "cv_scores": {"RandomForest": 0.55},
        "test_metrics": {"f1_macro": 0.55, "roc_auc": 0.72},
        "confusion_matrix": [[130, 20], [10, 5]],
        "threshold_metrics": [
            {"threshold": 0.3, "f1_macro": 0.4, "precision_minority": 0.10, "recall_minority": 0.80, "confusion_matrix": [[100, 50], [3, 12]]},
            {"threshold": 0.5, "f1_macro": 0.55, "precision_minority": 0.20, "recall_minority": 0.50, "confusion_matrix": [[140, 10], [7, 8]]},
        ],
        "feature_importances": _perm_imp({"feat_a": 0.04, "feat_b": 0.02, "feat_c": 0.01}),
        "calibration": {
            "prob_true": [0.05, 0.45, 0.90], "prob_pred": [0.10, 0.40, 0.85],
            "bin_counts": [40, 40, 40], "brier_score": 0.08, "ece": 0.06,
        },
        "calibrated_comparison": {
            "method": "isotonic",
            "prob_true": [0.06, 0.44, 0.89], "prob_pred": [0.07, 0.43, 0.88],
            "bin_counts": [40, 40, 40], "brier_score": 0.05, "ece": 0.02,
        },
    }
    p = tmp_path / "ml_report_with_calibration.json"
    p.write_text(json.dumps(report))
    return str(p)


def test_agent_run_generates_seventh_chart_when_calibration_present(
    agent, e2e_eda_report, e2e_ml_report_with_calibration, e2e_csv, tmp_path
):
    out = str(tmp_path / "viz")
    agent.run(e2e_eda_report, e2e_ml_report_with_calibration, e2e_csv, output_dir=out)
    chart_names = [c["name"] for c in agent.report_.charts]
    assert len(agent.report_.charts) == 7
    assert "calibration_curve" in chart_names


@pytest.fixture
def e2e_ml_report_with_error_analysis(tmp_path):
    report = {
        "task_type": "binary_classification",
        "best_model_name": "RandomForest",
        "cv_scores": {"RandomForest": 0.55},
        "test_metrics": {"f1_macro": 0.55, "roc_auc": 0.72},
        "confusion_matrix": [[130, 20], [10, 5]],
        "threshold_metrics": [
            {"threshold": 0.3, "f1_macro": 0.4, "precision_minority": 0.10, "recall_minority": 0.80, "confusion_matrix": [[100, 50], [3, 12]]},
            {"threshold": 0.5, "f1_macro": 0.55, "precision_minority": 0.20, "recall_minority": 0.50, "confusion_matrix": [[140, 10], [7, 8]]},
        ],
        "feature_importances": _perm_imp({"feat_a": 0.04, "feat_b": 0.02, "feat_c": 0.01}),
        "error_analysis": {
            "task_type": "binary_classification",
            "segment_columns": ["feat_a"],
            "overall": {"false_negative_rate": 0.3, "false_positive_rate": 0.2},
            "segments": {
                "feat_a": [
                    {"segment_value": 0.1, "n": 100, "n_positive": 40, "n_negative": 60,
                     "false_negative_rate": 0.7, "false_positive_rate": 0.2,
                     "elevated_false_negative_rate": True, "elevated_false_positive_rate": False},
                ],
            },
            "detection_note": "auto-detected",
            "note": "note",
        },
    }
    p = tmp_path / "ml_report_with_error_analysis.json"
    p.write_text(json.dumps(report))
    return str(p)


def test_agent_run_generates_eighth_chart_when_error_analysis_present(
    agent, e2e_eda_report, e2e_ml_report_with_error_analysis, e2e_csv, tmp_path
):
    out = str(tmp_path / "viz")
    agent.run(e2e_eda_report, e2e_ml_report_with_error_analysis, e2e_csv, output_dir=out)
    chart_names = [c["name"] for c in agent.report_.charts]
    assert "error_by_segment" in chart_names


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


# ---------------------------------------------------------------------------
# VisualizationAgent.run — end-to-end, REGRESSION branch
# (handbook Section 8.2 genericity: classification/regression parity)
# ---------------------------------------------------------------------------

@pytest.fixture
def e2e_regression_csv(tmp_path):
    rng = np.random.default_rng(11)
    n = 300
    square_footage = rng.uniform(800, 4000, n)
    price = square_footage * 150 + rng.standard_normal(n) * 20000 + 50000
    df = pd.DataFrame({
        "square_footage": square_footage,
        "num_bedrooms": rng.integers(1, 6, n),
        "lot_size": rng.uniform(0.1, 2.0, n),
        "neighborhood": rng.choice(["A", "B", "C"], n),
        "price": price,
    })
    p = tmp_path / "reg_data.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def e2e_regression_eda_report(tmp_path, e2e_regression_csv):
    df = pd.read_csv(e2e_regression_csv)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric].corr().to_dict()
    report = {
        "input_shape": list(df.shape),
        "numeric_columns": numeric,
        "categorical_columns": ["neighborhood"],
        "descriptive_stats": {},
        "correlation_matrix": corr,
        "skewness": {},
        "outlier_summary": {},
    }
    p = tmp_path / "reg_eda_report.json"
    p.write_text(json.dumps(report))
    return str(p)


@pytest.fixture
def e2e_regression_ml_report(tmp_path, sample_test_predictions):
    report = {
        "task_type": "regression",
        "best_model_name": "RandomForestRegressor",
        "cv_scores": {"RandomForestRegressor": -18000.0},
        "best_hyperparameters": {"n_estimators": 200},
        "test_metrics": {"rmse": 18500.2, "mae": 14200.5, "r2": 0.81, "adjusted_r2": 0.80},
        "confusion_matrix": None,
        "threshold_metrics": None,
        "feature_importances": _perm_imp({"square_footage": 0.42, "num_bedrooms": 0.05, "lot_size": 0.02}),
        "test_predictions": sample_test_predictions,
    }
    p = tmp_path / "reg_ml_report.json"
    p.write_text(json.dumps(report))
    return str(p)


def test_agent_run_regression_generates_regression_charts_not_classification(
    agent, e2e_regression_eda_report, e2e_regression_ml_report, e2e_regression_csv, tmp_path
):
    """The core genericity proof: a regression ML report must steer
    VisualizationAgent.run() into the actual-vs-predicted/residual/scatter
    charts and away from confusion-matrix/threshold-tradeoff/box-plot --
    the classification-only charts."""
    out = str(tmp_path / "viz_reg")
    success, _ = agent.run(
        e2e_regression_eda_report, e2e_regression_ml_report, e2e_regression_csv,
        target_col="price", output_dir=out,
    )
    assert success is True

    chart_names = {c["name"] for c in agent.report_.charts}
    assert chart_names == {
        "distributions", "correlation_heatmap", "feature_importance",
        "actual_vs_predicted", "residuals", "top_feature_vs_target",
    }
    assert "confusion_matrix" not in chart_names
    assert "threshold_tradeoff" not in chart_names

    top_feature_chart = next(
        c for c in agent.report_.charts if c["name"] == "top_feature_vs_target"
    )
    assert "Scatter plot" in top_feature_chart["description"]

    for chart in agent.report_.charts:
        assert os.path.getsize(chart["path"]) > _MIN_FILE_BYTES, (
            f"{chart['name']} PNG is suspiciously small"
        )


def test_agent_run_regression_writes_valid_report_json(
    agent, e2e_regression_eda_report, e2e_regression_ml_report, e2e_regression_csv, tmp_path
):
    out = str(tmp_path / "viz_reg_json")
    _, report_path = agent.run(
        e2e_regression_eda_report, e2e_regression_ml_report, e2e_regression_csv,
        target_col="price", output_dir=out,
    )
    with open(report_path) as f:
        data = json.load(f)
    assert len(data["charts"]) == 6
    names = {c["name"] for c in data["charts"]}
    assert "actual_vs_predicted" in names
    assert "residuals" in names
