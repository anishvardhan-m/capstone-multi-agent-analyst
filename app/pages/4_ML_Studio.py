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

st.subheader("Per-Record Predictions")
preds_table = report.get("test_predictions_table")
preds_df = dh.ml_predictions_table_df(report)
if preds_df is None or preds_df.empty:
    st.caption("No per-record predictions recorded for this run.")
else:
    st.caption(preds_table.get("note", ""))

    fcol1, fcol2 = st.columns([2, 1])
    search_query = fcol1.text_input(
        "Search row ID", key="ml_studio_pred_search",
        placeholder="e.g. a specific order/row ID",
    )
    filtered_df = dh.search_predictions_df(preds_df, search_query)

    is_classification_table = "correct" in filtered_df.columns
    if is_classification_table:
        show_incorrect_only = fcol2.checkbox(
            "Show only incorrect predictions", key="ml_studio_pred_incorrect_only",
        )
        if show_incorrect_only:
            filtered_df = dh.filter_incorrect_predictions(filtered_df)

        confidence_caveat = dh.calibration_caveat(report)
        if confidence_caveat:
            st.caption(
                f"{confidence_caveat} See **Calibration** below for this run's "
                "measured calibration error."
            )
        else:
            st.caption(
                "This model uses class-weighted training, so the 'confidence' "
                "column above is not a literal probability -- treat it as a "
                "relative ranking signal, not a calibrated percentage chance."
            )
    elif "abs_error" in filtered_df.columns:
        max_possible_error = float(filtered_df["abs_error"].max())
        min_abs_error = fcol2.number_input(
            "Min. absolute error", min_value=0.0, max_value=max_possible_error,
            value=0.0, key="ml_studio_pred_min_error",
        )
        if min_abs_error > 0:
            filtered_df = dh.filter_large_error_predictions(filtered_df, min_abs_error)

    st.download_button(
        "Download predictions table (CSV)",
        data=preds_df.to_csv(index=False).encode("utf-8"),
        file_name="test_predictions_table.csv",
        mime="text/csv",
        help="Downloads the full captured predictions table (see note above "
             "if it was sampled), not just the current filtered/paginated view.",
    )

    page_size = 50
    total_rows = len(filtered_df)
    if total_rows == 0:
        st.caption("No rows match the current search/filter.")
    else:
        n_pages = -(-total_rows // page_size)
        page = st.number_input(
            "Page", min_value=1, max_value=max(1, n_pages), value=1, step=1,
            key="ml_studio_pred_page",
        )
        page_df = dh.paginate_df(filtered_df, int(page), page_size)
        st.dataframe(page_df, use_container_width=True, hide_index=True)
        start = (int(page) - 1) * page_size
        st.caption(
            f"Showing rows {start + 1}-{min(start + page_size, total_rows)} of "
            f"{total_rows} (page {page} of {n_pages})."
        )

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
