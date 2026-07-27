"""
app/pages/4_ML_Studio.py

Machine Learning Studio (capstone handbook, Section 13): task type, model
comparison, best hyperparameters, held-out test metrics, and confusion
matrix / actual-vs-predicted, branching on task_type.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "app"))

import streamlit as st

import dashboard_helpers as dh

st.set_page_config(page_title="ML Studio | AI Data Analyst", page_icon="🤖", layout="wide")
dh.ensure_session_state(st.session_state)

st.title("Machine Learning Studio")

ml_report_path, ml_source = dh.resolve_report_path(
    st.session_state.get("ml_report_path"), dh.DEFAULT_ML_REPORT_REL
)

if not ml_report_path:
    st.info("Run the pipeline first from the **Dataset Ingestion** page.")
    st.stop()

if ml_source == "fallback":
    st.caption(
        "Showing output from a previous run (no pipeline run found in this "
        "session) — run the pipeline yourself from **Dataset Ingestion** to "
        "see your own dataset's results here."
    )

report = dh.load_json_report(ml_report_path)

if report is None:
    st.warning("ML report unavailable (file missing or the step was skipped/failed).")
    st.stop()

banner = dh.prediction_banner_text(report)
if banner:
    st.markdown(f"**{banner}**")

summary = dh.ml_summary(report)
col1, col2 = st.columns(2)
col1.metric("Task type", summary["task_type"].replace("_", " ").title())
col2.metric("Best model", summary["best_model_name"])

split_strategy = report.get("split_strategy", "row_random")
split_group_col = report.get("group_col")
if split_strategy == "grouped" and split_group_col:
    st.caption(
        f"Train/test split: grouped by `{split_group_col}` — no single "
        f"`{split_group_col}` value's rows appear in both train and test."
    )
else:
    st.caption(
        "Train/test split: plain row-wise (stratified for classification). "
        "No group column was supplied, so if the same real-world entity "
        "contributes multiple rows, some of its rows could land in both "
        "train and test."
    )

st.subheader("Model Comparison (Cross-Validation)")
st.dataframe(dh.ml_cv_scores_df(report), use_container_width=True, hide_index=True)
st.caption("cv_std is the standard deviation across CV folds at each model's winning hyperparameters.")

model_selection_note = report.get("model_selection_note")
if model_selection_note:
    st.warning(model_selection_note)

nested_cv_score = report.get("nested_cv_score")
if nested_cv_score is not None:
    ncol1, ncol2 = st.columns(2)
    ncol1.metric("Nested CV score (de-biased)", f"{nested_cv_score:.4f}")
    ncol2.metric("Nested CV std", f"{report.get('nested_cv_std', 0):.4f}")
    st.caption(report.get("nested_cv_note", ""))
elif report.get("nested_cv_note"):
    st.caption(report["nested_cv_note"])

st.subheader("Best Hyperparameters")
best_params = report.get("best_hyperparameters") or {}
if best_params:
    st.json(best_params)
else:
    st.caption("No tunable hyperparameters for this model.")

st.subheader("Held-Out Test Metrics")
st.dataframe(dh.ml_test_metrics_df(report), use_container_width=True, hide_index=True)

task_type = summary["task_type"]
if task_type == "regression":
    st.subheader("Actual vs. Predicted (held-out test set sample)")
    preds_df = dh.ml_test_predictions_df(report)
    if preds_df is None:
        st.caption("No test predictions recorded.")
    else:
        st.dataframe(preds_df, use_container_width=True)
else:
    st.subheader("Confusion Matrix (threshold = 0.5)")
    cm_df = dh.ml_confusion_matrix_df(report)
    if cm_df is None:
        st.caption("No confusion matrix recorded.")
    else:
        st.dataframe(cm_df, use_container_width=True)

    threshold_metrics = report.get("threshold_metrics")
    if threshold_metrics:
        st.subheader("Decision Threshold Sweep")
        import pandas as pd
        st.dataframe(
            pd.DataFrame(threshold_metrics).drop(columns=["confusion_matrix"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )

    caveat = dh.calibration_caveat(report)
    if caveat:
        st.info(caveat)

    st.subheader("Calibration")
    calibration = report.get("calibration")
    if calibration is None:
        st.caption(report.get("calibration_note") or "Calibration not computed for this run.")
    else:
        ccol1, ccol2 = st.columns(2)
        ccol1.metric("Expected Calibration Error (ECE)", f"{calibration.get('ece', 0):.4f}")
        ccol2.metric("Brier score", f"{calibration.get('brier_score', 0):.4f}")

        calibrated_comparison = report.get("calibrated_comparison")
        if calibrated_comparison:
            ccol3, ccol4 = st.columns(2)
            ccol3.metric(
                f"{calibrated_comparison.get('method', 'calibrated').title()}-calibrated ECE",
                f"{calibrated_comparison.get('ece', 0):.4f}",
            )
            ccol4.metric(
                f"{calibrated_comparison.get('method', 'calibrated').title()}-calibrated Brier",
                f"{calibrated_comparison.get('brier_score', 0):.4f}",
            )
        st.caption(report.get("calibration_note", ""))
        st.caption(
            "See the Visualization Gallery for the reliability diagram "
            "(predicted probability vs. observed frequency)."
        )

st.subheader("Feature Importances")
importances_df = dh.ml_feature_importances_df(report)
if importances_df.empty:
    st.caption("No feature importances recorded.")
else:
    st.caption(
        "Permutation importance, mean ± std across repeated shuffles. Rows "
        "marked False under 'distinguishable_from_zero' have a mean ± std "
        "range that spans zero -- not distinguishable from no effect."
    )
    st.dataframe(importances_df, use_container_width=True, hide_index=True)
    st.bar_chart(importances_df.set_index("feature")["importance_mean"])

st.subheader("Error Analysis by Segment")
error_analysis = report.get("error_analysis") or {}
error_df = dh.ml_error_analysis_df(report)
if error_df.empty:
    st.caption(
        error_analysis.get("detection_note")
        or "No segment columns were configured or auto-detected for this run."
    )
else:
    st.caption(
        "Held-out error rate broken down by segment -- an association "
        "observed in this held-out test set, not a causal claim about why "
        "a segment differs. Rows with any `elevated_*` column set to True "
        "have an error rate/MAE meaningfully above the overall value "
        "shown below."
    )
    overall = error_analysis.get("overall") or {}
    if overall:
        st.write("Overall:", overall)
    st.dataframe(error_df, use_container_width=True, hide_index=True)
    st.caption(error_analysis.get("detection_note", ""))
    st.caption(error_analysis.get("note", ""))
