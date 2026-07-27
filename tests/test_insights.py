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
    _CAUSAL_LANGUAGE_GUARDRAIL,
    _ERROR_CONCENTRATION_HEDGE,
    BusinessInsightsAgent,
    _accuracy_from_confusion_matrix,
    _build_prompt,
    _format_error_analysis_lines,
    _majority_baseline_accuracy,
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
            "customer_state": {
                "importance_mean": 0.0476, "importance_std": 0.0060,
                "distinguishable_from_zero": True,
            },
            "order_estimated_delivery_date": {
                "importance_mean": 0.0292, "importance_std": 0.0040,
                "distinguishable_from_zero": True,
            },
            "purchase_to_estimated_days": {
                "importance_mean": 0.0212, "importance_std": 0.0030,
                "distinguishable_from_zero": True,
            },
            # Above materiality but its mean +/- std range spans zero --
            # must surface with the "NOT DISTINGUISHABLE FROM ZERO" flag.
            "delivery_zone_mismatch": {
                "importance_mean": 0.0150, "importance_std": 0.0200,
                "distinguishable_from_zero": False,
            },
            # Near-zero/noise features -- below the 0.01 materiality
            # threshold -- must never surface in the prompt's feature list.
            "customer_unique_id": {
                "importance_mean": 0.0004, "importance_std": 0.0010,
                "distinguishable_from_zero": False,
            },
            "n_distinct_products": {
                "importance_mean": 0.0, "importance_std": 0.0005,
                "distinguishable_from_zero": False,
            },
            "primary_seller_state": {
                "importance_mean": -0.0002, "importance_std": 0.0008,
                "distinguishable_from_zero": False,
            },
        },
    }


@pytest.fixture
def report_paths(tmp_path, eda_report, ml_report):
    eda_path = tmp_path / "eda_report.json"
    ml_path = tmp_path / "ml_report.json"
    eda_path.write_text(json.dumps(eda_report))
    ml_path.write_text(json.dumps(ml_report))
    return str(eda_path), str(ml_path)


# ---------------------------------------------------------------------------
# Shared fixtures: a synthetic EDA + REGRESSION ML report pair (genericity)
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_eda_report() -> dict:
    return {
        "input_shape": [500, 6],
        "numeric_columns": ["square_footage", "num_bedrooms", "lot_size", "year_built", "price"],
        "categorical_columns": ["neighborhood"],
        "descriptive_stats": {},
        "correlation_matrix": {
            "square_footage": {"square_footage": 1.0, "price": 0.78},
            "price": {"square_footage": 0.78, "price": 1.0},
        },
        "skewness": {"lot_size": 2.3},
        "outlier_summary": {"lot_size": {"n_outliers": 20, "pct_outliers": 4.0}},
    }


@pytest.fixture
def regression_ml_report() -> dict:
    return {
        "task_type": "regression",
        "best_model_name": "RandomForestRegressor",
        "cv_scores": {"RandomForestRegressor": -12.5},
        "best_hyperparameters": {"n_estimators": 200, "max_depth": 10},
        "test_metrics": {
            "rmse": 18500.234,
            "mae": 14200.567,
            "r2": 0.812,
            "adjusted_r2": 0.803,
        },
        "confusion_matrix": None,
        "threshold_metrics": None,
        "feature_importances": {
            "square_footage": {
                "importance_mean": 0.42, "importance_std": 0.05,
                "distinguishable_from_zero": True,
            },
            "num_bedrooms": {
                "importance_mean": 0.18, "importance_std": 0.03,
                "distinguishable_from_zero": True,
            },
            "lot_size": {
                "importance_mean": 0.015, "importance_std": 0.004,
                "distinguishable_from_zero": True,
            },
            # Near-zero/noise features -- must be filtered out here too.
            "year_built": {
                "importance_mean": 0.0006, "importance_std": 0.001,
                "distinguishable_from_zero": False,
            },
            "has_garage": {
                "importance_mean": 0.0, "importance_std": 0.0005,
                "distinguishable_from_zero": False,
            },
        },
    }


@pytest.fixture
def regression_report_paths(tmp_path, regression_eda_report, regression_ml_report):
    eda_path = tmp_path / "reg_eda_report.json"
    ml_path = tmp_path / "reg_ml_report.json"
    eda_path.write_text(json.dumps(regression_eda_report))
    ml_path.write_text(json.dumps(regression_ml_report))
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

def test_prompt_never_naively_pluralizes_positive_label(eda_report, ml_report):
    """Regression test: a real run against a label ending in 'y'
    ('late delivery') produced 'late deliverys' in the LLM's output,
    because the prompt itself naively wrote '{positive_label}s'. The
    prompt must phrase plurals as 'cases of X' / 'instances of X' instead
    of ever appending a bare 's' to a caller-supplied label."""
    prompt = _build_prompt(
        eda_report, ml_report,
        positive_label="late delivery",
        negative_label="on-time delivery",
        unit_label="order",
    )

    assert "late deliverys" not in prompt
    assert "cases of late delivery" in prompt


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
    late-delivery narrative. When the caller supplies domain labels
    ("late delivery"/"on-time delivery"/"order"), the prompt must name them
    explicitly and forbid the generic phrasing."""
    prompt = _build_prompt(
        eda_report, ml_report,
        positive_label="late delivery",
        negative_label="on-time delivery",
        unit_label="order",
    )

    assert "late delivery" in prompt
    assert "order" in prompt

    # Explicit ban on the generic phrasing the LLM previously drifted into.
    assert "business outcome" in prompt  # named only inside the prohibition
    assert "positive event" in prompt
    assert "do not use generic" in prompt

    # The confusion matrix must be spelled out in plain, domain-specific
    # terms (this is what grounded the original narrative's concrete
    # counts like "1,021 late deliveries caught").
    assert "1,021 orders correctly identified as late delivery" in prompt
    assert "544 orders that were actually late delivery but the model missed" in prompt
    assert (
        "4,416 orders that were actually on-time delivery but were "
        "incorrectly flagged as late delivery"
    ) in prompt


def test_prompt_uses_generic_defaults_when_labels_not_provided(eda_report, ml_report):
    """When a caller doesn't supply a domain vocabulary (e.g. a brand-new
    dataset this agent has never seen), the prompt must still be coherent
    -- using generic but sensible defaults rather than crashing or leaking
    a hardcoded Olist-specific label."""
    prompt = _build_prompt(eda_report, ml_report)

    assert "positive case" in prompt
    assert "negative case" in prompt
    assert "record" in prompt
    assert "late delivery" not in prompt


def test_prompt_uses_custom_labels_when_provided_instead_of_defaults(eda_report, ml_report):
    prompt = _build_prompt(
        eda_report, ml_report,
        positive_label="late delivery",
        negative_label="on-time delivery",
        unit_label="order",
    )

    assert "late delivery" in prompt
    assert "on-time delivery" in prompt
    assert "order" in prompt
    assert "positive case" not in prompt
    assert "negative case" not in prompt
    assert "record" not in prompt


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


def test_prompt_flags_material_but_statistically_indistinguishable_feature(eda_report, ml_report):
    """delivery_zone_mismatch clears the 0.01 materiality threshold (mean
    0.015) but its mean +/- std range spans zero -- it must still surface
    in the feature list (materiality only filters on the mean), tagged so
    the LLM knows not to build a business claim around it, and the prompt
    must forbid doing so."""
    prompt = _build_prompt(eda_report, ml_report)

    assert "delivery_zone_mismatch: 0.0150" in prompt
    assert "NOT DISTINGUISHABLE FROM ZERO" in prompt

    # A genuinely material, distinguishable feature must NOT be flagged.
    customer_state_line = next(
        line for line in prompt.splitlines() if line.strip().startswith("customer_state:")
    )
    assert "NOT DISTINGUISHABLE FROM ZERO" not in customer_state_line

    assert "never build a claim around a feature marked NOT DISTINGUISHABLE FROM ZERO" in prompt


# ---------------------------------------------------------------------------
# Error analysis by segment (handbook F6/F8): hedged association language
# ---------------------------------------------------------------------------

def test_format_error_analysis_lines_empty_dict_returns_no_lines():
    assert _format_error_analysis_lines({}) == []


def test_format_error_analysis_lines_states_evenly_spread_when_nothing_elevated():
    error_analysis = {
        "task_type": "binary_classification",
        "segments": {
            "region": [
                {"segment_value": 0.1, "n": 50, "false_negative_rate": 0.3,
                 "false_positive_rate": 0.2, "elevated_false_negative_rate": False,
                 "elevated_false_positive_rate": False},
            ],
        },
    }
    lines = _format_error_analysis_lines(error_analysis)
    assert len(lines) == 1
    assert "evenly spread" in lines[0]


def test_format_error_analysis_lines_classification_includes_hedge_and_rates():
    error_analysis = {
        "task_type": "binary_classification",
        "segments": {
            "region": [
                {"segment_value": 0.42, "n": 200, "false_negative_rate": 0.55,
                 "false_positive_rate": 0.1, "elevated_false_negative_rate": True,
                 "elevated_false_positive_rate": False},
            ],
        },
    }
    lines = _format_error_analysis_lines(error_analysis)
    joined = "\n".join(lines)
    assert _ERROR_CONCENTRATION_HEDGE in joined
    assert "region=0.42" in joined
    assert "n=200" in joined
    assert "false_negative_rate=0.550" in joined
    assert "elevated false negative rate" in joined


def test_format_error_analysis_lines_regression_uses_mae_and_mean_error():
    error_analysis = {
        "task_type": "regression",
        "segments": {
            "zone": [
                {"segment_value": 0.9, "n": 100, "mae": 25.0, "mean_error": 15.0,
                 "elevated_mae": True, "elevated_bias": True},
            ],
        },
    }
    lines = _format_error_analysis_lines(error_analysis)
    joined = "\n".join(lines)
    assert "mae=25.000" in joined
    assert "mean_error=15.000" in joined
    assert "elevated mae, bias" in joined


def test_format_error_analysis_lines_generic_over_arbitrary_column_and_values():
    """Non-Olist column name, string segment value -- must render the same way."""
    error_analysis = {
        "task_type": "binary_classification",
        "segments": {
            "shipping_carrier": [
                {"segment_value": "carrier_y", "n": 80, "false_negative_rate": 0.6,
                 "false_positive_rate": 0.1, "elevated_false_negative_rate": True,
                 "elevated_false_positive_rate": False},
            ],
        },
    }
    lines = _format_error_analysis_lines(error_analysis)
    assert "shipping_carrier=carrier_y" in "\n".join(lines)


@pytest.fixture
def ml_report_with_error_analysis(ml_report) -> dict:
    report = dict(ml_report)
    report["error_analysis"] = {
        "task_type": "binary_classification",
        "segment_columns": ["customer_state"],
        "overall": {"false_negative_rate": 0.35, "false_positive_rate": 0.25},
        "segments": {
            "customer_state": [
                {"segment_value": 0.42, "n": 8091, "n_positive": 488, "n_negative": 7603,
                 "false_negative_rate": 0.547, "false_positive_rate": 0.137,
                 "elevated_false_negative_rate": True, "elevated_false_positive_rate": False},
            ],
        },
        "detection_note": "auto-detected",
        "note": "note",
    }
    return report


def test_prompt_includes_error_analysis_hedge_and_segment_detail(eda_report, ml_report_with_error_analysis):
    prompt = _build_prompt(eda_report, ml_report_with_error_analysis)

    assert _ERROR_CONCENTRATION_HEDGE in prompt
    assert "customer_state=0.42" in prompt
    assert "false_negative_rate=0.547" in prompt
    # The instruction telling the LLM how to talk about it must also be present.
    assert "never as a reason or explanation" in prompt
    assert "do not speculate about what causes the segment to differ" in prompt


def test_prompt_omits_error_analysis_data_line_when_not_present(eda_report, ml_report):
    """ml_report (the base fixture) has no error_analysis key at all -- the
    prompt must not crash and must simply omit the segment-findings data
    line (the static instructions still mention 'Error analysis by segment'
    as a phrase, but no actual "- Error analysis by segment ..." data bullet
    should be generated when there's nothing to report)."""
    prompt = _build_prompt(eda_report, ml_report)
    assert "- Error analysis by segment" not in prompt


def test_regression_prompt_includes_error_analysis_hedge(regression_eda_report, regression_ml_report):
    report = dict(regression_ml_report)
    report["error_analysis"] = {
        "task_type": "regression",
        "segment_columns": ["neighborhood"],
        "overall": {"mae": 10.0, "mean_error": 1.0},
        "segments": {
            "neighborhood": [
                {"segment_value": 0.7, "n": 60, "mae": 30.0, "mean_error": 20.0,
                 "elevated_mae": True, "elevated_bias": True},
            ],
        },
        "detection_note": "auto-detected",
        "note": "note",
    }
    prompt = _build_prompt(regression_eda_report, report, unit_label="house")

    assert _ERROR_CONCENTRATION_HEDGE in prompt
    assert "neighborhood=0.7" in prompt
    assert "mae=30.000" in prompt
    assert "systematic over/under-prediction" in prompt


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


# ---------------------------------------------------------------------------
# Accuracy framing + causal language (handbook F8)
# ---------------------------------------------------------------------------

def test_majority_baseline_accuracy_computes_from_confusion_matrix():
    # 90 actual negatives, 10 actual positives -- trivial "always predict
    # negative" baseline is 90/100 = 90%.
    cm = [[80, 10], [5, 5]]
    assert _majority_baseline_accuracy(cm) == pytest.approx(90 / 100)


def test_majority_baseline_accuracy_none_without_confusion_matrix():
    assert _majority_baseline_accuracy(None) is None
    assert _majority_baseline_accuracy([]) is None


def test_prompt_flags_accuracy_below_majority_baseline(eda_report, ml_report):
    """Regression test for the actual failure mode F8 exists to catch: on
    this project's ~8% positive rate, class_weight='balanced' trades raw
    accuracy for minority recall, so OVERALL ACCURACY (74.3%) is actually
    WORSE than the trivial 'always predict on-time' baseline (91.9%) --
    presenting 74.3% alone would mislead a reader into thinking it's a
    solid, unqualified number."""
    prompt = _build_prompt(eda_report, ml_report)

    cm = ml_report["confusion_matrix"]
    accuracy = _accuracy_from_confusion_matrix(cm)
    baseline = _majority_baseline_accuracy(cm)
    assert accuracy < baseline  # sanity: this fixture must exercise the below-baseline path

    assert "MAJORITY-CLASS BASELINE ACCURACY" in prompt
    assert f"{baseline:.1%}" in prompt
    assert "BELOW this trivial baseline" in prompt
    assert f"{accuracy:.1%} vs. {baseline:.1%}" in prompt

    # And the instruction requiring the LLM to actually use it.
    assert "you must also state the MAJORITY-CLASS BASELINE" in prompt
    assert "expected tradeoff of correcting for class imbalance" in prompt


def test_prompt_flags_accuracy_above_majority_baseline_when_model_beats_it(eda_report, ml_report):
    """The opposite case: a well-separated model whose accuracy beats the
    trivial baseline must be labeled 'above', not 'BELOW'."""
    report = dict(ml_report)
    # 95 correct out of 100, actual class split 60/40 -> baseline = 60%.
    report["confusion_matrix"] = [[55, 5], [0, 40]]
    prompt = _build_prompt(eda_report, report)

    cm = report["confusion_matrix"]
    accuracy = _accuracy_from_confusion_matrix(cm)
    baseline = _majority_baseline_accuracy(cm)
    assert accuracy > baseline

    assert "beats this trivial baseline" in prompt
    assert "BELOW this trivial baseline" not in prompt


def test_causal_language_guardrail_present_in_classification_prompt(eda_report, ml_report):
    prompt = _build_prompt(eda_report, ml_report)
    assert _CAUSAL_LANGUAGE_GUARDRAIL in prompt
    assert "causes" in prompt  # named as a banned word
    assert "is associated with" in prompt  # named as the approved alternative


def test_causal_language_guardrail_present_in_regression_prompt(
    regression_eda_report, regression_ml_report
):
    prompt = _build_prompt(regression_eda_report, regression_ml_report, unit_label="house")
    assert _CAUSAL_LANGUAGE_GUARDRAIL in prompt


# ---------------------------------------------------------------------------
# Genericity: regression branch (handbook Section 8.2)
# ---------------------------------------------------------------------------

def test_regression_prompt_uses_rmse_mae_r2_not_classification_language(
    regression_eda_report, regression_ml_report
):
    """The prompt must branch on task_type == 'regression': RMSE/MAE/
    Adjusted R^2 reported explicitly, and none of the classification-only
    vocabulary (accuracy, ROC-AUC, confusion matrix, threshold sweep,
    precision/recall) should appear as data."""
    prompt = _build_prompt(regression_eda_report, regression_ml_report, unit_label="house")

    rmse = regression_ml_report["test_metrics"]["rmse"]
    mae = regression_ml_report["test_metrics"]["mae"]
    adj_r2 = regression_ml_report["test_metrics"]["adjusted_r2"]
    assert "RMSE" in prompt
    assert f"{rmse:.4f}" in prompt
    assert "MAE" in prompt
    assert f"{mae:.4f}" in prompt
    assert "Adjusted R^2" in prompt
    assert f"{adj_r2:.4f}" in prompt

    # Classification-only formatted lines/tokens must never appear.
    assert "- OVERALL ACCURACY" not in prompt
    assert "- ROC-AUC" not in prompt
    assert "- Confusion matrix at threshold" not in prompt
    assert "Decision threshold tradeoff" not in prompt
    assert "precision=" not in prompt
    assert "recall=" not in prompt

    # Materiality filter still applies to regression feature importances.
    assert "square_footage: 0.4200" in prompt
    assert "year_built" not in prompt
    assert "has_garage" not in prompt

    # Recommendations must be tied to error magnitude, not thresholds.
    assert "predictions are typically off by" in prompt
    assert "Do NOT phrase" in prompt
    assert "REGRESSION model" in prompt
    assert "house" in prompt


def test_regression_run_writes_narrative_and_sends_regression_prompt(
    regression_report_paths, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    eda_path, ml_path = regression_report_paths
    narrative = "## What We Found\nPredictions are typically off by $14,200.\n"
    client = _mock_client_with_response(narrative)

    agent = BusinessInsightsAgent(client=client, unit_label="house")
    success, output_path = agent.run(eda_path, ml_path, output_dir=str(tmp_path / "out"))

    assert success is True
    with open(output_path) as f:
        assert f.read() == narrative.strip()
    assert agent.report_.narrative == narrative.strip()

    sent_prompt = client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert "RMSE" in sent_prompt
    assert "house" in sent_prompt
    assert "precision=" not in sent_prompt
    assert "- OVERALL ACCURACY" not in sent_prompt


# ---------------------------------------------------------------------------
# Genericity: constructor-supplied labels thread into the prompt
# ---------------------------------------------------------------------------

def test_agent_constructor_labels_thread_into_sent_prompt(report_paths, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    eda_path, ml_path = report_paths
    client = _mock_client_with_response("## What We Found\ntext\n")

    agent = BusinessInsightsAgent(
        client=client,
        positive_label="late delivery",
        negative_label="on-time delivery",
        unit_label="order",
    )
    agent.run(eda_path, ml_path, output_dir=str(tmp_path / "out"))

    sent_prompt = client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert "late delivery" in sent_prompt
    assert "on-time delivery" in sent_prompt
    assert "order" in sent_prompt
    assert "positive case" not in sent_prompt


def test_agent_uses_generic_defaults_when_constructed_without_labels(report_paths, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    eda_path, ml_path = report_paths
    client = _mock_client_with_response("## What We Found\ntext\n")

    agent = BusinessInsightsAgent(client=client)
    agent.run(eda_path, ml_path, output_dir=str(tmp_path / "out"))

    sent_prompt = client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert "positive case" in sent_prompt
    assert "record" in sent_prompt


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
