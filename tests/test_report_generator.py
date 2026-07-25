"""
tests/test_report_generator.py

Unit tests for the Report Generation Agent (src/agents/report_generator.py).

No LLM calls happen in this agent (pure HTML templating -> PDF via
WeasyPrint), so nothing here needs mocking for non-determinism -- these
tests exercise the real markdown parser, the real HTML builders, and (for
the end-to-end tests) a real WeasyPrint PDF render.

Run with:
    pytest tests/test_report_generator.py -v
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.report_generator import (
    ReportGenerationAgent,
    ReportGenerationReport,
    _build_charts_html,
    _build_data_diagnostics_html,
    _build_html_document,
    _build_model_performance_html,
    _map_insights_sections,
    _markdown_to_html,
    _parse_insights_markdown,
)

_MIN_PDF_BYTES = 10_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_insights_md() -> str:
    return (
        "## What We Found\n"
        "The model achieves an OVERALL ACCURACY of 74.3%. The ROC-AUC is 0.769.\n\n"
        "## What Matters Most\n"
        "The customer's state matters most for predicting late delivery.\n\n"
        "## Recommendations\n"
        "1. **Use threshold 0.4** as the default operating point.\n"
        "2. **Use threshold 0.3** to maximize recall.\n"
    )


@pytest.fixture
def sample_cleaning_report() -> dict:
    return {
        "input_shape": [96476, 23],
        "output_shape": [96476, 22],
        "n_duplicates_removed": 0,
        "high_missing_columns_flagged": [],
        "low_variance_columns_dropped": ["order_status"],
        "columns_type_corrected": [],
        "numeric_columns_imputed": ["total_price", "total_freight"],
        "categorical_columns_imputed": ["customer_state", "customer_city"],
    }


@pytest.fixture
def sample_classification_ml_report() -> dict:
    return {
        "task_type": "binary_classification",
        "best_model_name": "HistGradientBoostingClassifier",
        "cv_scores": {
            "LogisticRegression": 0.4749,
            "RandomForestClassifier": 0.5660,
            "HistGradientBoostingClassifier": 0.5721,
        },
        "best_hyperparameters": {"learning_rate": 0.1, "max_iter": 100},
        "test_metrics": {
            "f1_macro": 0.56731, "precision_macro": 0.574267,
            "recall_macro": 0.70167, "roc_auc": 0.769292,
        },
        "confusion_matrix": [[13315, 4416], [544, 1021]],
        "threshold_metrics": [
            {"threshold": 0.3, "f1_macro": 0.378521, "precision_minority": 0.11436, "recall_minority": 0.897125},
            {"threshold": 0.5, "f1_macro": 0.56731, "precision_minority": 0.187787, "recall_minority": 0.652396},
        ],
        "feature_importances": {"customer_state": 0.0476, "purchase_to_estimated_days": 0.0212},
        "test_predictions": None,
    }


@pytest.fixture
def sample_regression_ml_report() -> dict:
    return {
        "task_type": "regression",
        "best_model_name": "LinearRegression",
        "cv_scores": {"LinearRegression": -24646.88, "RandomForestRegressor": -28155.35},
        "best_hyperparameters": {},
        "test_metrics": {"rmse": 24503.264, "mae": 19157.447, "r2": 0.985984, "adjusted_r2": 0.985561},
        "confusion_matrix": None,
        "threshold_metrics": None,
        "feature_importances": {"square_footage": 180.1, "num_bathrooms": 6409.7},
        "test_predictions": {"actual": [100.0, 200.0], "predicted": [110.0, 190.0]},
    }


@pytest.fixture
def tiny_chart_png(tmp_path):
    """A real, valid, tiny PNG -- generated with matplotlib (already a
    project dependency) rather than hand-rolled bytes."""
    path = tmp_path / "01_chart.png"
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    fig.savefig(str(path))
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

def test_parse_insights_markdown_splits_three_sections(sample_insights_md):
    sections = _parse_insights_markdown(sample_insights_md)

    assert len(sections) == 3
    headers = [h for h, _ in sections]
    assert headers == ["What We Found", "What Matters Most", "Recommendations"]

    bodies = dict(sections)
    assert "74.3%" in bodies["What We Found"]
    assert "customer's state" in bodies["What Matters Most"]
    assert "threshold 0.4" in bodies["Recommendations"]


def test_map_insights_sections_assigns_by_keyword(sample_insights_md):
    sections = _parse_insights_markdown(sample_insights_md)
    mapped = _map_insights_sections(sections)

    assert "74.3%" in mapped["executive_summary"]
    assert "customer's state" in mapped["business_insights"]
    assert "threshold 0.4" in mapped["recommendations"]


def test_map_insights_sections_falls_back_positionally_for_unusual_headers():
    """Even with headers that don't contain the expected keywords, the
    3 sections must still map to the 3 slots in order rather than being
    dropped."""
    md = (
        "## Summary Stuff\n"
        "first body\n\n"
        "## Key Drivers\n"
        "second body\n\n"
        "## Next Steps\n"
        "third body\n"
    )
    sections = _parse_insights_markdown(md)
    mapped = _map_insights_sections(sections)

    assert mapped["executive_summary"] == "first body"
    assert mapped["business_insights"] == "second body"
    assert mapped["recommendations"] == "third body"


def test_parse_insights_markdown_handles_malformed_input_gracefully():
    """No '## ' headers at all -> empty section list, not a crash."""
    sections = _parse_insights_markdown("just some plain text with no headers")
    assert sections == []
    mapped = _map_insights_sections(sections)
    assert mapped == {"executive_summary": None, "business_insights": None, "recommendations": None}


# ---------------------------------------------------------------------------
# Markdown -> HTML rendering
# ---------------------------------------------------------------------------

def test_markdown_to_html_renders_bold_and_numbered_list():
    html = _markdown_to_html("1. **Bold label**: rest of the sentence.\n2. Second item.")
    assert "<ol>" in html
    assert "<strong>Bold label</strong>" in html
    assert "<li>" in html


def test_markdown_to_html_empty_input_returns_empty_string():
    assert _markdown_to_html(None) == ""
    assert _markdown_to_html("   ") == ""


# ---------------------------------------------------------------------------
# HTML template — all 5 section headers present
# ---------------------------------------------------------------------------

def test_html_template_contains_all_five_section_headers():
    html = _build_html_document(
        project_identifier="TEST-PROJECT",
        generated_at="2026-01-01 00:00",
        executive_summary_html="<p>summary</p>",
        data_diagnostics_html="<table></table>",
        model_performance_html="<table></table>",
        business_insights_html="<p>insights</p>",
        charts_html="<div></div>",
        recommendations_html="<p>recs</p>",
    )
    assert "Executive Summary" in html
    assert "Data Diagnostics Profile" in html
    assert "Model Performance Leaderboard" in html
    assert "Automated Business Insights" in html
    assert "Strategic Recommendations" in html
    assert "Project Identifier: TEST-PROJECT" in html


# ---------------------------------------------------------------------------
# Data diagnostics section
# ---------------------------------------------------------------------------

def test_data_diagnostics_html_contains_key_figures(sample_cleaning_report):
    html = _build_data_diagnostics_html(sample_cleaning_report)
    assert "96,476" in html
    assert "order_status" in html
    assert "total_price" in html


def test_data_diagnostics_html_placeholder_on_missing_report():
    html = _build_data_diagnostics_html({})
    assert "unavailable" in html


# ---------------------------------------------------------------------------
# Model performance leaderboard — classification vs regression branch
# ---------------------------------------------------------------------------

def test_model_performance_classification_shows_confusion_matrix_not_rmse(
    sample_classification_ml_report,
):
    html = _build_model_performance_html(sample_classification_ml_report)
    assert "confusion-table" in html
    assert "HistGradientBoostingClassifier" in html
    assert "ROC-AUC" in html
    assert "RMSE" not in html
    assert "MAE" not in html


def test_model_performance_regression_shows_rmse_not_confusion_matrix(
    sample_regression_ml_report,
):
    html = _build_model_performance_html(sample_regression_ml_report)
    assert "RMSE" in html
    assert "MAE" in html
    assert "confusion-table" not in html
    assert "ROC-AUC" not in html


def test_model_performance_placeholder_on_missing_report():
    html = _build_model_performance_html({})
    assert "unavailable" in html


# ---------------------------------------------------------------------------
# Chart embedding — missing files skipped gracefully
# ---------------------------------------------------------------------------

def test_charts_html_embeds_valid_chart(tiny_chart_png):
    report = ReportGenerationReport(output_path="x.pdf")
    html = _build_charts_html([tiny_chart_png], report)

    assert "<img src='data:image/png;base64," in html
    assert report.charts_embedded == [tiny_chart_png]
    assert report.charts_skipped == []


def test_charts_html_skips_missing_chart_gracefully(tiny_chart_png, tmp_path):
    missing_path = str(tmp_path / "does_not_exist.png")
    report = ReportGenerationReport(output_path="x.pdf")
    html = _build_charts_html([tiny_chart_png, missing_path], report)

    # The valid chart must still render.
    assert html.count("<img src=") == 1
    assert report.charts_embedded == [tiny_chart_png]
    assert len(report.charts_skipped) == 1
    assert report.charts_skipped[0]["path"] == missing_path


def test_charts_html_all_missing_returns_placeholder(tmp_path):
    report = ReportGenerationReport(output_path="x.pdf")
    html = _build_charts_html([str(tmp_path / "gone.png")], report)
    assert "No charts available" in html


# ---------------------------------------------------------------------------
# End-to-end: run() on valid synthetic inputs produces a real PDF
# ---------------------------------------------------------------------------

@pytest.fixture
def report_input_paths(tmp_path, sample_cleaning_report, sample_classification_ml_report, sample_insights_md):
    cleaning_path = tmp_path / "cleaning_report.json"
    ml_path = tmp_path / "ml_report.json"
    md_path = tmp_path / "business_insights.md"

    cleaning_path.write_text(json.dumps(sample_cleaning_report))
    ml_path.write_text(json.dumps(sample_classification_ml_report))
    md_path.write_text(sample_insights_md)

    return str(cleaning_path), str(ml_path), str(md_path)


def test_run_produces_real_nontrivial_pdf(report_input_paths, tiny_chart_png, tmp_path):
    cleaning_path, ml_path, md_path = report_input_paths
    output_path = str(tmp_path / "report.pdf")

    agent = ReportGenerationAgent()
    success, result = agent.run(
        cleaning_report_path=cleaning_path,
        ml_report_path=ml_path,
        insights_md_path=md_path,
        chart_paths=[tiny_chart_png],
        output_path=output_path,
    )

    assert success is True
    assert result == output_path
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > _MIN_PDF_BYTES

    assert agent.report_ is not None
    assert agent.report_.sections_included == [
        "executive_summary", "data_diagnostics", "model_performance",
        "business_insights", "recommendations",
    ]
    assert agent.report_.charts_embedded == [tiny_chart_png]


def test_run_regression_report_produces_real_pdf(
    tmp_path, sample_cleaning_report, sample_regression_ml_report, sample_insights_md, tiny_chart_png
):
    cleaning_path = tmp_path / "cleaning_report.json"
    ml_path = tmp_path / "ml_report.json"
    md_path = tmp_path / "business_insights.md"
    cleaning_path.write_text(json.dumps(sample_cleaning_report))
    ml_path.write_text(json.dumps(sample_regression_ml_report))
    md_path.write_text(sample_insights_md)
    output_path = str(tmp_path / "reg_report.pdf")

    agent = ReportGenerationAgent()
    success, result = agent.run(
        cleaning_report_path=str(cleaning_path),
        ml_report_path=str(ml_path),
        insights_md_path=str(md_path),
        chart_paths=[tiny_chart_png],
        output_path=output_path,
    )

    assert success is True
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > _MIN_PDF_BYTES


def test_run_handles_missing_inputs_gracefully(tmp_path):
    """None of the 4 inputs exist -- run() must still succeed and produce
    a placeholder-filled PDF rather than crashing."""
    output_path = str(tmp_path / "degraded_report.pdf")

    agent = ReportGenerationAgent()
    success, result = agent.run(
        cleaning_report_path=str(tmp_path / "no_cleaning.json"),
        ml_report_path=str(tmp_path / "no_ml.json"),
        insights_md_path=str(tmp_path / "no_insights.md"),
        chart_paths=[str(tmp_path / "no_chart.png")],
        output_path=output_path,
    )

    assert success is True
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > _MIN_PDF_BYTES

    report = agent.report_
    assert len(report.warnings) >= 3  # cleaning, ML, insights all missing
    assert len(report.charts_skipped) == 1
    assert report.charts_embedded == []


def test_run_uses_auto_generated_project_identifier_when_not_provided(
    report_input_paths, tiny_chart_png, tmp_path
):
    cleaning_path, ml_path, md_path = report_input_paths
    output_path = str(tmp_path / "auto_id_report.pdf")

    agent = ReportGenerationAgent()
    success, _ = agent.run(
        cleaning_report_path=cleaning_path,
        ml_report_path=ml_path,
        insights_md_path=md_path,
        chart_paths=[tiny_chart_png],
        output_path=output_path,
    )
    assert success is True
    assert os.path.exists(output_path)
