"""
src/agents/insights.py

Business Insights Agent.

Responsibilities (per the capstone handbook, Section 6.7): read the EDA
and ML reports, summarize their key figures into a compact prompt, and
call an LLM to produce a plain-English narrative that connects the
model's findings to concrete business recommendations -- grounded
directly in the model's actual performance numbers.

Design note: this is the first agent in the pipeline that makes a genuine
LLM call; every prior agent is fully deterministic. Two things keep that
non-determinism contained:
  1. A two-model fallback -- the primary free model is tried first; on
     any failure (including rate-limiting) the agent retries once against
     OpenRouter's free-model auto-router before giving up and returning
     (False, error_message).
  2. The LLM client is injected via the constructor, so tests exercise
     the prompt-construction and fallback logic against a mock client --
     no real network calls happen in the test suite.

Genericity (handbook Section 8.2): this agent must work for classification
AND regression, on any dataset -- not just the Olist late-delivery model.
Two things keep it generic:
  1. The prompt branches on ml_report["task_type"]. Classification gets
     accuracy/ROC-AUC/confusion-matrix/threshold-tradeoff language;
     regression gets RMSE/MAE/Adjusted R^2 and error-magnitude language.
     Neither branch ever borrows the other's vocabulary.
  2. The business-domain labels (what a "positive case" is called, what a
     row is called) are constructor parameters with generic defaults, not
     hardcoded strings. Callers with a concrete domain (e.g. this
     project's Olist model) pass "late delivery"/"on-time delivery"/
     "order" explicitly -- see the __main__ block below.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.tools.audit_db import audit_logged
from src.tools.logging_config import get_agent_logger

logger = get_agent_logger("BusinessInsightsAgent")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

_PRIMARY_MODEL = "inclusionai/ling-3.0-flash:free"
_FALLBACK_MODEL = "openrouter/free"

_SYSTEM_PROMPT = (
    "You are a business analyst translating a data science team's model "
    "results into a report for non-technical stakeholders. Be concrete, "
    "avoid statistical jargon, and ground every claim in the numbers "
    "you're given rather than generic advice. Never fall back on abstract, "
    "templated ML-report language such as 'business outcome', 'positive "
    "event', or 'positive outcome' -- always name the actual thing being "
    "predicted."
)

# Generic fallback labels used only when a caller doesn't supply a domain
# vocabulary. Callers with a concrete domain (e.g. this project's Olist
# late-delivery model) should pass their own labels into
# BusinessInsightsAgent's constructor instead of relying on these.
_DEFAULT_UNIT_LABEL = "record"
_DEFAULT_POSITIVE_LABEL = "positive case"
_DEFAULT_NEGATIVE_LABEL = "negative case"

# Feature importances below this magnitude are statistical noise, not real
# signal (e.g. a near-zero id-column importance vs. a genuine driver an
# order of magnitude larger). Filtering them out of the prompt itself is
# what stops the LLM from inventing a business story for a feature that
# isn't actually predictive.
_IMPORTANCE_MATERIALITY_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class InsightsReport:
    """Structured summary of the business-insights LLM run."""

    narrative: str
    model_used: str
    output_path: str

    def to_dict(self) -> dict:
        return {
            "narrative": self.narrative,
            "model_used": self.model_used,
            "output_path": self.output_path,
        }


# ---------------------------------------------------------------------------
# EDA summary (shared by classification and regression)
# ---------------------------------------------------------------------------

def _top_correlations(corr_matrix: dict, top_n: int = 5) -> list[tuple[str, str, float]]:
    """Return the top-N strongest absolute pairwise correlations, deduplicated."""
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, float]] = []
    for col_a, row in corr_matrix.items():
        for col_b, r in row.items():
            if col_a == col_b or r is None:
                continue
            key = tuple(sorted((col_a, col_b)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((col_a, col_b, r))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pairs[:top_n]


def _skewed_columns(skewness: dict, threshold: float = 1.0, top_n: int = 5) -> list[tuple[str, float]]:
    """Return the most skewed numeric columns beyond a |skew| threshold."""
    items = [(c, v) for c, v in skewness.items() if v is not None and abs(v) > threshold]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return items[:top_n]


def _outlier_highlights(outlier_summary: dict, min_pct: float = 5.0, top_n: int = 5) -> list[tuple[str, float]]:
    """Return columns whose outlier prevalence exceeds min_pct, worst first."""
    items = [
        (c, v.get("pct_outliers", 0.0))
        for c, v in outlier_summary.items()
        if v.get("pct_outliers", 0.0) >= min_pct
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n]


def _format_eda_summary(eda: dict) -> str:
    """Render the EDA report's key figures as plain text (not raw JSON)."""
    lines: list[str] = []

    shape = eda.get("input_shape") or []
    if len(shape) == 2:
        lines.append(
            f"- Dataset: {shape[0]:,} rows x {shape[1]} columns "
            f"({len(eda.get('numeric_columns', []))} numeric, "
            f"{len(eda.get('categorical_columns', []))} categorical)"
        )

    top_corr = _top_correlations(eda.get("correlation_matrix") or {})
    if top_corr:
        lines.append("- Strongest pairwise correlations:")
        for a, b, r in top_corr:
            lines.append(f"    {a} <-> {b}: r = {r:.2f}")

    skewed = _skewed_columns(eda.get("skewness") or {})
    if skewed:
        lines.append("- Most skewed numeric columns:")
        for col, val in skewed:
            lines.append(f"    {col}: skew = {val:.2f}")

    outliers = _outlier_highlights(eda.get("outlier_summary") or {})
    if outliers:
        lines.append("- Columns with the most outliers (IQR method):")
        for col, pct in outliers:
            lines.append(f"    {col}: {pct:.1f}% of rows flagged as outliers")

    return "\n".join(lines) if lines else "(no EDA summary available)"


# ---------------------------------------------------------------------------
# ML summary helpers shared by both task types
# ---------------------------------------------------------------------------

def _format_feature_importance_lines(importances: dict) -> list[str]:
    """Render feature importances filtered to those above materiality.

    Shared by the classification and regression summaries so both apply
    the exact same noise filter.
    """
    lines: list[str] = []
    material = [
        (f, v) for f, v in importances.items()
        if abs(v) > _IMPORTANCE_MATERIALITY_THRESHOLD
    ]
    if material:
        material.sort(key=lambda x: x[1], reverse=True)
        material = material[:8]
        lines.append(
            f"- Top features by importance (only features with importance > "
            f"{_IMPORTANCE_MATERIALITY_THRESHOLD} are shown -- every other "
            f"feature the model saw was statistical noise, not a real driver):"
        )
        for feat, val in material:
            lines.append(f"    {feat}: {val:.4f}")
    elif importances:
        lines.append(
            f"- No feature cleared the {_IMPORTANCE_MATERIALITY_THRESHOLD} "
            f"importance materiality threshold -- treat this model as not "
            f"having an identifiable dominant driver."
        )
    return lines


# ---------------------------------------------------------------------------
# Classification: ML summary + prompt
# ---------------------------------------------------------------------------

def _accuracy_from_confusion_matrix(cm: Optional[list]) -> Optional[float]:
    """Compute overall accuracy (trace / total) directly from a confusion matrix.

    This is deliberately computed here rather than trusted to the LLM: the
    ML report's test_metrics only contains ROC-AUC (a ranking/separability
    score) alongside macro precision/recall/F1 -- it has no plain accuracy
    figure, and the two are easy to conflate since both are single numbers
    roughly in the same 0-1 range.
    """
    if not cm:
        return None
    total = sum(sum(row) for row in cm)
    if total == 0:
        return None
    correct = sum(cm[i][i] for i in range(len(cm)))
    return correct / total


def _confusion_breakdown(
    cm: Optional[list],
    unit_label: str,
    positive_label: str,
    negative_label: str,
) -> Optional[str]:
    """Describe a binary confusion matrix in the caller's domain terms.

    Assumes sklearn's confusion_matrix(y_true, y_pred) convention with
    classes sorted [0, 1] -- matches MLAgent's binary-classification output
    (0 = negative, 1 = positive), so cm[0][0]=TN, cm[0][1]=FP, cm[1][0]=FN,
    cm[1][1]=TP. Spelling this out in words (rather than leaving the LLM to
    infer it from the raw matrix) is what lets the narrative say concrete
    things like "1,021 late deliveries caught" instead of a generic
    "positive event."
    """
    if not cm or len(cm) != 2 or len(cm[0]) != 2:
        return None
    tn, fp = cm[0]
    fn, tp = cm[1]
    return (
        f"    {tp:,} {unit_label}s correctly identified as {positive_label}\n"
        f"    {fn:,} {unit_label}s that were actually {positive_label} but "
        f"the model missed (predicted {negative_label})\n"
        f"    {fp:,} {unit_label}s that were actually {negative_label} but "
        f"were incorrectly flagged as {positive_label}\n"
        f"    {tn:,} {unit_label}s correctly identified as {negative_label}"
    )


def _format_ml_summary_classification(
    ml: dict, positive_label: str, negative_label: str, unit_label: str
) -> str:
    """Render a classification ML report's key figures as plain text."""
    lines: list[str] = []

    lines.append(f"- Task type: {ml.get('task_type', 'unknown')}")
    lines.append(f"- Best model: {ml.get('best_model_name', 'unknown')}")

    test_metrics = ml.get("test_metrics") or {}
    cm = ml.get("confusion_matrix")

    accuracy = _accuracy_from_confusion_matrix(cm)
    if accuracy is not None:
        lines.append(
            f"- OVERALL ACCURACY (fraction of all predictions that were "
            f"correct, computed from the confusion matrix): {accuracy:.1%}"
        )
    if "roc_auc" in test_metrics:
        lines.append(
            f"- ROC-AUC (a separate ranking/separability metric -- this is "
            f"NOT accuracy and will not equal the accuracy figure above): "
            f"{test_metrics['roc_auc']:.3f}"
        )
    other_metrics = {k: v for k, v in test_metrics.items() if k != "roc_auc"}
    if other_metrics:
        metrics_str = ", ".join(f"{k} = {v:.3f}" for k, v in other_metrics.items())
        lines.append(f"- Other held-out test metrics (default 0.5 threshold): {metrics_str}")

    if cm:
        lines.append(f"- Confusion matrix at threshold 0.5 (rows=actual, cols=predicted): {cm}")
        breakdown = _confusion_breakdown(cm, unit_label, positive_label, negative_label)
        if breakdown:
            lines.append(f"- That confusion matrix, spelled out in plain terms:\n{breakdown}")

    lines.extend(_format_feature_importance_lines(ml.get("feature_importances") or {}))

    thresholds = ml.get("threshold_metrics")
    if thresholds:
        lines.append(f"- Decision threshold tradeoff for catching cases of {positive_label}:")
        for t in thresholds:
            lines.append(
                f"    threshold={t['threshold']}: "
                f"precision={t['precision_minority']:.3f}, "
                f"recall={t['recall_minority']:.3f}, "
                f"f1_macro={t['f1_macro']:.3f}"
            )

    return "\n".join(lines) if lines else "(no ML summary available)"


def _build_classification_prompt(
    eda_summary: str,
    ml_summary: str,
    positive_label: str,
    negative_label: str,
    unit_label: str,
) -> str:
    return (
        f"Here is a summary of an exploratory data analysis and a trained "
        f"predictive model. The model predicts whether a {unit_label} will "
        f"be a {positive_label} -- every metric below (accuracy, ROC-AUC, "
        f"precision, recall, confusion matrix, feature importances) is "
        f"scored against that same target: {positive_label} (positive/"
        f"minority class) vs. {negative_label} (negative/majority class). "
        f"Write a business insights narrative in markdown with exactly "
        f"three sections:\n\n"
        "## What We Found\n"
        f"A plain-English summary of what the model found and how well it "
        f"performs at predicting cases of {positive_label}. Every sentence "
        f"in this section must refer to '{positive_label}' and "
        f"'{unit_label}s' by name -- do not use generic, templated "
        f"ML-report language like 'business outcome', 'positive outcome', "
        f"'positive event', or 'the target'. Do NOT pluralize "
        f"'{positive_label}' by simply appending 's' if that would produce "
        f"an awkward or incorrect plural -- write 'cases of {positive_label}' "
        f"or 'instances of {positive_label}' instead.\n\n"
        "## What Matters Most\n"
        f"Which factors matter most for predicting a {positive_label} and "
        f"why, explained in business terms (not statistical jargon), "
        f"continuing to refer to '{positive_label}' cases and {unit_label}s "
        f"concretely rather than generically. Only discuss features listed "
        f"under 'Top features by importance' below -- every one of those "
        f"already cleared the {_IMPORTANCE_MATERIALITY_THRESHOLD} "
        f"materiality threshold. Do not draw a business conclusion from any "
        f"feature with a near-zero importance value, even if you see it "
        f"listed elsewhere (e.g. in the confusion matrix section or your "
        f"own general knowledge of the dataset) -- a near-zero importance "
        f"means the model found no real signal there, so inventing a "
        f"reason for it is not supported by the data and must not appear "
        f"in the narrative.\n\n"
        "## Recommendations\n"
        f"2-3 concrete, actionable recommendations for handling cases of "
        f"{positive_label}. Each must be tied directly to the specific "
        f"decision-threshold tradeoff numbers below -- name the actual "
        f"precision/recall values and which threshold to use for which "
        f"business goal.\n\n"
        "IMPORTANT: OVERALL ACCURACY and ROC-AUC are two different metrics "
        "reported separately below -- do not conflate them, do not use one "
        "number to describe the other, and do not average or blend them into "
        "a single 'accuracy' figure. When you say 'accuracy', use only the "
        "number explicitly labeled OVERALL ACCURACY. When you describe the "
        "model's ability to rank/separate the two classes, use only the "
        "number explicitly labeled ROC-AUC, and name it as ROC-AUC, not "
        "accuracy.\n\n"
        "=== EXPLORATORY DATA ANALYSIS ===\n"
        f"{eda_summary}\n\n"
        "=== MODEL RESULTS ===\n"
        f"{ml_summary}\n"
    )


# ---------------------------------------------------------------------------
# Regression: ML summary + prompt
# ---------------------------------------------------------------------------

def _format_ml_summary_regression(ml: dict, unit_label: str) -> str:
    """Render a regression ML report's key figures as plain text.

    Deliberately has no accuracy/ROC-AUC/confusion-matrix/threshold
    vocabulary -- those concepts don't exist for a continuous target.
    """
    lines: list[str] = []

    lines.append(f"- Task type: {ml.get('task_type', 'unknown')}")
    lines.append(f"- Best model: {ml.get('best_model_name', 'unknown')}")

    test_metrics = ml.get("test_metrics") or {}
    if "rmse" in test_metrics:
        lines.append(
            f"- RMSE (Root Mean Squared Error -- typical prediction error "
            f"in the target's original units, penalizes large misses more "
            f"heavily than small ones): {test_metrics['rmse']:.4f}"
        )
    if "mae" in test_metrics:
        lines.append(
            f"- MAE (Mean Absolute Error -- the typical size of a "
            f"prediction error in the target's original units, treating "
            f"every miss equally): {test_metrics['mae']:.4f}"
        )
    if "adjusted_r2" in test_metrics:
        lines.append(
            f"- Adjusted R^2 (fraction of the target's variance the model "
            f"explains, penalized for the number of features used; 1.0 = "
            f"perfect, 0.0 = no better than always guessing the average "
            f"{unit_label}'s value): {test_metrics['adjusted_r2']:.4f}"
        )
    elif "r2" in test_metrics:
        lines.append(
            f"- R^2 (fraction of the target's variance the model explains; "
            f"1.0 = perfect, 0.0 = no better than always guessing the "
            f"average {unit_label}'s value): {test_metrics['r2']:.4f}"
        )

    lines.extend(_format_feature_importance_lines(ml.get("feature_importances") or {}))

    return "\n".join(lines) if lines else "(no ML summary available)"


def _build_regression_prompt(eda_summary: str, ml_summary: str, unit_label: str) -> str:
    return (
        f"Here is a summary of an exploratory data analysis and a trained "
        f"predictive model. The model predicts a continuous numeric value "
        f"for each {unit_label} -- every metric below (RMSE, MAE, R^2, "
        f"feature importances) describes how close those predictions come "
        f"to the true value. This is a REGRESSION model: there is no "
        f"positive/negative class, no accuracy, no ROC-AUC, and no "
        f"confusion matrix for this model -- do not invent any of those "
        f"concepts anywhere in the narrative. Write a business insights "
        f"narrative in markdown with exactly three sections:\n\n"
        "## What We Found\n"
        f"A plain-English summary of what the model found and how well it "
        f"predicts the target value for each {unit_label}, stated in terms "
        f"of typical prediction error using the MAE/RMSE figures below "
        f"(e.g. 'predictions are typically off by X') rather than accuracy "
        f"or classification language.\n\n"
        "## What Matters Most\n"
        f"Which factors matter most for predicting the target value and "
        f"why, explained in business terms (not statistical jargon), "
        f"referring to {unit_label}s concretely. Only discuss features "
        f"listed under 'Top features by importance' below -- every one of "
        f"those already cleared the {_IMPORTANCE_MATERIALITY_THRESHOLD} "
        f"materiality threshold. Do not draw a business conclusion from any "
        f"feature with a near-zero importance value, even if you see it "
        f"elsewhere or from your own general knowledge of the dataset -- a "
        f"near-zero importance means the model found no real signal "
        f"there.\n\n"
        "## Recommendations\n"
        f"2-3 concrete, actionable recommendations. Each must be tied "
        f"directly to the practical size of the prediction error -- use "
        f"the actual MAE/RMSE figures below to state something like "
        f"'predictions are typically off by X, which means...' and explain "
        f"what that error magnitude means for a real business decision "
        f"(e.g. how much safety margin to build in, or when to trust the "
        f"model's number vs. seek a human estimate). Do NOT phrase any "
        f"recommendation in terms of decision thresholds, precision, or "
        f"recall -- those concepts don't apply to a regression model.\n\n"
        "=== EXPLORATORY DATA ANALYSIS ===\n"
        f"{eda_summary}\n\n"
        "=== MODEL RESULTS ===\n"
        f"{ml_summary}\n"
    )


# ---------------------------------------------------------------------------
# Top-level prompt dispatcher
# ---------------------------------------------------------------------------

def _build_prompt(
    eda: dict,
    ml: dict,
    positive_label: Optional[str] = None,
    negative_label: Optional[str] = None,
    unit_label: Optional[str] = None,
) -> str:
    """Build the full user prompt sent to the LLM.

    Deliberately summarizes both reports into readable bullet points
    rather than dumping raw JSON, so the model reasons over the numbers
    that matter instead of parsing structure. Branches on
    ml["task_type"]: regression gets RMSE/MAE/R^2 and error-magnitude
    language; everything else (binary/multiclass classification) gets
    accuracy/ROC-AUC/confusion-matrix/threshold-tradeoff language. The two
    vocabularies never mix.
    """
    unit_label = unit_label or _DEFAULT_UNIT_LABEL
    eda_summary = _format_eda_summary(eda)

    if ml.get("task_type") == "regression":
        ml_summary = _format_ml_summary_regression(ml, unit_label)
        return _build_regression_prompt(eda_summary, ml_summary, unit_label)

    positive_label = positive_label or _DEFAULT_POSITIVE_LABEL
    negative_label = negative_label or _DEFAULT_NEGATIVE_LABEL
    ml_summary = _format_ml_summary_classification(
        ml, positive_label, negative_label, unit_label
    )
    return _build_classification_prompt(
        eda_summary, ml_summary, positive_label, negative_label, unit_label
    )


# ---------------------------------------------------------------------------
# BusinessInsightsAgent
# ---------------------------------------------------------------------------

class BusinessInsightsAgent:
    """Summarizes upstream reports and calls an LLM for a narrative writeup.

    Parameters
    ----------
    client : openai.OpenAI | None
        Pre-built client to use instead of constructing one from the
        OPENAI_API_KEY / OPENAI_API_BASE_URL environment variables. Tests
        inject a mock here so no real network call is made.
    positive_label, negative_label : str | None
        What the model's positive/negative class is called in this
        business domain (e.g. "late delivery" / "on-time delivery").
        Ignored for regression reports. Defaults to generic "positive
        case" / "negative case" when not supplied -- callers with a real
        domain should always pass their own labels.
    unit_label : str | None
        What one row/prediction is called in this business domain (e.g.
        "order", "house", "customer"). Defaults to generic "record".
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        positive_label: Optional[str] = None,
        negative_label: Optional[str] = None,
        unit_label: Optional[str] = None,
    ):
        self._client = client
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.unit_label = unit_label
        self.report_: Optional[InsightsReport] = None

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL"),
        )
        return self._client

    @staticmethod
    def _call_model(client: OpenAI, model: str, prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError(f"Model '{model}' returned an empty response")
        return content.strip()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @audit_logged("BusinessInsightsAgent", input_arg=("eda_report_path", "ml_report_path"))
    def run(
        self,
        eda_report_path: str,
        ml_report_path: str,
        output_dir: str = "workspace",
    ) -> tuple[bool, str]:
        """Generate a business-insights narrative from the EDA and ML reports.

        Parameters
        ----------
        eda_report_path : str
            Path to the EDA agent's JSON report.
        ml_report_path : str
            Path to the ML agent's JSON report.
        output_dir : str
            Directory to write ``business_insights.md`` into.

        Returns
        -------
        (success, output_path_or_error_message)
        """
        logger.info(
            "Starting business insights run  eda=%s  ml=%s",
            eda_report_path, ml_report_path,
        )

        if not os.getenv("OPENAI_API_KEY", "").strip():
            msg = "OPENAI_API_KEY is not set; cannot call the LLM"
            logger.error(msg)
            return False, msg

        try:
            with open(eda_report_path) as f:
                eda = json.load(f)
        except Exception as exc:
            logger.error("Failed to read EDA report: %s", exc)
            return False, f"Failed to read EDA report: {exc}"

        try:
            with open(ml_report_path) as f:
                ml = json.load(f)
        except Exception as exc:
            logger.error("Failed to read ML report: %s", exc)
            return False, f"Failed to read ML report: {exc}"

        prompt = _build_prompt(
            eda, ml,
            positive_label=self.positive_label,
            negative_label=self.negative_label,
            unit_label=self.unit_label,
        )
        client = self._get_client()

        narrative: Optional[str] = None
        model_used: Optional[str] = None
        errors: list[str] = []
        for model in (_PRIMARY_MODEL, _FALLBACK_MODEL):
            try:
                logger.info("Calling LLM model=%s", model)
                narrative = self._call_model(client, model, prompt)
                model_used = model
                break
            except Exception as exc:
                logger.warning("Model '%s' failed: %s", model, exc)
                errors.append(f"{model}: {exc}")

        if narrative is None:
            msg = "All LLM models failed -- " + "; ".join(errors)
            logger.error(msg)
            return False, msg

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "business_insights.md")
        with open(output_path, "w") as f:
            f.write(narrative)
        logger.info(
            "Business insights written to %s (model=%s)", output_path, model_used
        )

        self.report_ = InsightsReport(
            narrative=narrative,
            model_used=model_used,
            output_path=output_path,
        )

        return True, output_path


if __name__ == "__main__":
    import sys

    # This project's model always predicts one concrete thing: whether an
    # ORDER will be a LATE DELIVERY. Passed explicitly here rather than
    # relying on a hardcoded default, since BusinessInsightsAgent itself
    # is domain-agnostic (see class docstring).
    agent = BusinessInsightsAgent(
        positive_label="late delivery",
        negative_label="on-time delivery",
        unit_label="order",
    )
    success, result = agent.run(
        eda_report_path="data/processed/olist_flattened_cleaned_eda_report.json",
        ml_report_path="data/processed/olist_flattened_cleaned_features_ml_report.json",
        output_dir="workspace",
    )
    if success:
        print(f"Success. Insights written to: {result}")
        print(f"Model used: {agent.report_.model_used}")
        print()
        print(agent.report_.narrative)
    else:
        print(f"Failed: {result}")
        sys.exit(1)
