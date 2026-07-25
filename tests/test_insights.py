"""
tests/test_insights.py

Unit tests for the Business Insights Agent (src/agents/insights.py).

The LLM client is always mocked here -- no real network call is made by
this test suite. See src/agents/insights.py's __main__ block for the one
genuine OpenRouter call.

Run with:
    pytest tests/test_insights.py -v
"""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.insights import (
    _PRIMARY_MODEL,
    _FALLBACK_MODEL,
    BusinessInsightsAgent,
    _build_prompt,
)


# ---------------------------------------------------------------------------
# Shared fixtures: a synthetic EDA + ML report pair
# ---------------------------------------------------------------------------

@pytest.fixture
def eda_report() -> dict:
    return {
        "input_shape": [20000, 13],
        "numeric_columns": ["total_price", "total_freight", "n_items"],
        "categorical_columns": ["customer_state", "product_category"],
        "descriptive_stats": {},
        "correlation_matrix": {
            "total_price": {"total_price": 1.0, "total_freight": 0.83, "n_items": 0.12},
            "total_freight": {"total_price": 0.83, "total_freight": 1.0, "n_items": 0.05},
            "n_items": {"total_price": 0.12, "total_freight": 0.05, "n_items": 1.0},
        },
        "skewness": {"total_price": 9.89, "total_freight": 12.28, "n_items": 7.56},
        "outlier_summary": {
            "total_price": {"n_outliers": 7658, "pct_outliers": 7.94},
            "total_freight": {"n_outliers": 9694, "pct_outliers": 10.05},
            "n_items": {"n_outliers": 9636, "pct_outliers": 9.99},
        },
    }


@pytest.fixture
def ml_report() -> dict:
    return {
        "task_type": "binary_classification",
        "best_model_name": "HistGradientBoostingClassifier",
        "cv_scores": {"HistGradientBoostingClassifier": 0.5721},
        "best_hyperparameters": {"learning_rate": 0.1, "max_iter": 100},
        "test_metrics": {
            "f1_macro": 0.56731,
            "precision_macro": 0.574267,
            "recall_macro": 0.70167,
            "roc_auc": 0.769292,
        },
        "confusion_matrix": [[13315, 4416], [544, 1021]],
        "threshold_metrics": [
            {"threshold": 0.3, "f1_macro": 0.378521, "precision_minority": 0.11436, "recall_minority": 0.897125},
            {"threshold": 0.4, "f1_macro": 0.494104, "precision_minority": 0.148573, "recall_minority": 0.801917},
            {"threshold": 0.5, "f1_macro": 0.56731, "precision_minority": 0.187787, "recall_minority": 0.652396},
        ],
        "feature_importances": {
            "customer_state": 0.0476,
            "order_estimated_delivery_date": 0.0292,
            "purchase_to_estimated_days": 0.0212,
            # Near-zero/noise features -- below the 0.01 materiality
            # threshold -- must never surface in the prompt's feature list.
            "customer_unique_id": 0.0004,
            "n_distinct_products": 0.0,
            "primary_seller_state": -0.0002,
        },
    }


@pytest.fixture
def report_paths(tmp_path, eda_report, ml_report):
    eda_path = tmp_path / "eda_report.json"
    ml_path = tmp_path / "ml_report.json"
    eda_path.write_text(json.dumps(eda_report))
    ml_path.write_text(json.dumps(ml_report))
    return str(eda_path), str(ml_path)


def _mock_client_with_response(text: str) -> MagicMock:
    client = MagicMock()
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    client.chat.completions.create.return_value = completion
    return client


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_prompt_contains_key_eda_and_ml_figures(eda_report, ml_report):
    prompt = _build_prompt(eda_report, ml_report)

    # EDA figures
    assert "20,000 rows" in prompt
    assert "total_price <-> total_freight: r = 0.83" in prompt
    assert "total_freight: skew = 12.28" in prompt
    assert "10.0% of rows flagged as outliers" in prompt or "10.1%" in prompt

    # ML figures
    assert "HistGradientBoostingClassifier" in prompt
    assert "binary_classification" in prompt
    assert "customer_state: 0.0476" in prompt

    # Accuracy (from confusion matrix) and ROC-AUC must appear as distinct,
    # clearly labeled figures -- this is what prevents the LLM from
    # conflating them (see test_prompt_instructs_llm_not_to_conflate_metrics).
    # cm = [[13315, 4416], [544, 1021]] -> accuracy = (13315+1021)/19296
    expected_accuracy = (13315 + 1021) / (13315 + 4416 + 544 + 1021)
    assert "OVERALL ACCURACY" in prompt
    assert f"{expected_accuracy:.1%}" in prompt
    assert "ROC-AUC" in prompt
    assert "0.769" in prompt

    # Threshold tradeoff numbers, verbatim
    assert "threshold=0.3" in prompt
    assert "precision=0.114" in prompt
    assert "recall=0.897" in prompt
    assert "threshold=0.5" in prompt

    # It must not just be a JSON dump
    assert '"task_type"' not in prompt


def test_prompt_asks_for_three_named_sections(eda_report, ml_report):
    prompt = _build_prompt(eda_report, ml_report)
    assert "## What We Found" in prompt
    assert "## What Matters Most" in prompt
    assert "## Recommendations" in prompt


def test_prompt_uses_concrete_domain_language_not_generic_ml_terms(eda_report, ml_report):
    """Regression test: an earlier version of this prompt never named the
    business concept being predicted, so the LLM wrote a generic-sounding
    narrative ('business outcome', 'positive event') instead of an actual
    late-delivery narrative. The prompt must name 'late delivery'/'order'
    explicitly and forbid the generic phrasing."""
    prompt = _build_prompt(eda_report, ml_report)

    assert "late delivery" in prompt
    assert "order" in prompt

    # Explicit ban on the generic phrasing the LLM previously drifted into.
    assert "business outcome" in prompt  # named only inside the prohibition
    assert "positive event" in prompt
    assert "do not use generic" in prompt

    # The confusion matrix must be spelled out in plain, domain-specific
    # terms (this is what grounded the original narrative's concrete
    # counts like "1,021 late deliveries caught").
    assert "1,021 orders correctly caught as late deliveries" in prompt
    assert "544 actual late deliveries the model missed" in prompt
    assert "4,416 on-time orders incorrectly flagged as late deliveries" in prompt


def test_prompt_excludes_near_zero_importance_features(eda_report, ml_report):
    """Regression test: the LLM previously invented a 'repeat offenders'
    business story for customer_unique_id (importance 0.0004 -- noise, not
    signal). Near-zero-importance features must not appear in the feature
    list sent to the LLM, and the prompt must explicitly forbid drawing
    conclusions from any feature below the materiality threshold."""
    prompt = _build_prompt(eda_report, ml_report)

    # Material features are still present.
    assert "customer_state: 0.0476" in prompt
    assert "purchase_to_estimated_days: 0.0212" in prompt

    # Noise features must be filtered out of the feature list entirely.
    assert "customer_unique_id" not in prompt
    assert "n_distinct_products" not in prompt
    assert "primary_seller_state" not in prompt

    # And the prompt must explicitly instruct against inventing conclusions
    # from near-zero-importance features.
    assert "0.01" in prompt
    assert "near-zero importance" in prompt
    assert "not supported by the" in prompt


def test_prompt_instructs_llm_not_to_conflate_accuracy_and_roc_auc(eda_report, ml_report):
    """Regression test: the LLM previously reported ROC-AUC (0.769) as if it
    were overall accuracy. The prompt must supply both figures separately,
    clearly labeled, plus an explicit instruction not to conflate them."""
    prompt = _build_prompt(eda_report, ml_report)

    assert "do not conflate them" in prompt
    assert "OVERALL ACCURACY" in prompt
    assert "ROC-AUC" in prompt

    # The two figures must actually be numerically distinct in this fixture,
    # so a test that only checked "some accuracy-shaped number is present"
    # couldn't pass by accident using the ROC-AUC value.
    accuracy = (13315 + 1021) / (13315 + 4416 + 544 + 1021)
    roc_auc = ml_report["test_metrics"]["roc_auc"]
    assert accuracy != pytest.approx(roc_auc, abs=0.01)
    assert f"{accuracy:.1%}" in prompt


def test_run_sends_expected_prompt_to_client(report_paths, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    eda_path, ml_path = report_paths
    client = _mock_client_with_response("## What We Found\ntext\n")

    agent = BusinessInsightsAgent(client=client)
    agent.run(eda_path, ml_path, output_dir=str(tmp_path / "out"))

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == _PRIMARY_MODEL
    sent_messages = call_kwargs["messages"]
    user_content = sent_messages[-1]["content"]
    assert "HistGradientBoostingClassifier" in user_content
    assert "threshold=0.3" in user_content


# ---------------------------------------------------------------------------
# Successful run: narrative written to markdown + report
# ---------------------------------------------------------------------------

def test_successful_response_written_to_markdown_and_report(report_paths, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    eda_path, ml_path = report_paths
    narrative = "## What We Found\nThe model predicts late deliveries well.\n"
    client = _mock_client_with_response(narrative)

    agent = BusinessInsightsAgent(client=client)
    out_dir = str(tmp_path / "out")
    success, output_path = agent.run(eda_path, ml_path, output_dir=out_dir)

    assert success is True
    assert output_path == os.path.join(out_dir, "business_insights.md")
    assert os.path.exists(output_path)

    with open(output_path) as f:
        written = f.read()
    assert written == narrative.strip()

    assert agent.report_ is not None
    assert agent.report_.narrative == narrative.strip()
    assert agent.report_.model_used == _PRIMARY_MODEL
    assert agent.report_.output_path == output_path


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

def test_primary_failure_triggers_fallback_model(report_paths, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    eda_path, ml_path = report_paths

    client = MagicMock()

    fallback_message = MagicMock()
    fallback_message.content = "## What We Found\nFallback narrative.\n"
    fallback_choice = MagicMock()
    fallback_choice.message = fallback_message
    fallback_completion = MagicMock()
    fallback_completion.choices = [fallback_choice]

    client.chat.completions.create.side_effect = [
        Exception("rate limited"),
        fallback_completion,
    ]

    agent = BusinessInsightsAgent(client=client)
    success, output_path = agent.run(eda_path, ml_path, output_dir=str(tmp_path / "out"))

    assert success is True
    assert client.chat.completions.create.call_count == 2
    first_model = client.chat.completions.create.call_args_list[0].kwargs["model"]
    second_model = client.chat.completions.create.call_args_list[1].kwargs["model"]
    assert first_model == _PRIMARY_MODEL
    assert second_model == _FALLBACK_MODEL
    assert agent.report_.model_used == _FALLBACK_MODEL


def test_both_models_failing_returns_false_with_error_message(report_paths, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    eda_path, ml_path = report_paths

    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("service unavailable")

    agent = BusinessInsightsAgent(client=client)
    success, message = agent.run(eda_path, ml_path, output_dir=str(tmp_path / "out"))

    assert success is False
    assert "service unavailable" in message
    assert _PRIMARY_MODEL in message
    assert _FALLBACK_MODEL in message
    assert client.chat.completions.create.call_count == 2
    assert agent.report_ is None
    assert not os.path.exists(os.path.join(str(tmp_path / "out"), "business_insights.md"))


# ---------------------------------------------------------------------------
# Missing API key: caught early, no call attempted
# ---------------------------------------------------------------------------

def test_missing_api_key_caught_before_any_call(report_paths, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    eda_path, ml_path = report_paths
    client = MagicMock()

    agent = BusinessInsightsAgent(client=client)
    success, message = agent.run(eda_path, ml_path, output_dir=str(tmp_path / "out"))

    assert success is False
    assert "OPENAI_API_KEY" in message
    client.chat.completions.create.assert_not_called()


def test_empty_api_key_caught_before_any_call(report_paths, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    eda_path, ml_path = report_paths
    client = MagicMock()

    agent = BusinessInsightsAgent(client=client)
    success, message = agent.run(eda_path, ml_path, output_dir=str(tmp_path / "out"))

    assert success is False
    assert "OPENAI_API_KEY" in message
    client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Edge cases: missing report files
# ---------------------------------------------------------------------------

def test_missing_eda_report_returns_error(monkeypatch, tmp_path, ml_report):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ml_path = tmp_path / "ml_report.json"
    ml_path.write_text(json.dumps(ml_report))

    agent = BusinessInsightsAgent(client=MagicMock())
    success, message = agent.run("no_such_eda.json", str(ml_path), output_dir=str(tmp_path / "out"))

    assert success is False
    assert "EDA report" in message


def test_missing_ml_report_returns_error(monkeypatch, tmp_path, eda_report):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    eda_path = tmp_path / "eda_report.json"
    eda_path.write_text(json.dumps(eda_report))

    agent = BusinessInsightsAgent(client=MagicMock())
    success, message = agent.run(str(eda_path), "no_such_ml.json", output_dir=str(tmp_path / "out"))

    assert success is False
    assert "ML report" in message
