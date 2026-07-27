"""
app/pages/9_Run_History.py

Run History / experiment tracking (handbook F9): a queryable view of the
SQLite `ml_experiments` table (src/tools/audit_db.py) -- what each past
MLAgent.run() produced (best model, held-out metrics, hyperparameters,
split strategy), so a metric's history survives past the next run
overwriting `<data>_ml_report.json`.

Deliberately separate from System Log Explorer (P8): that page answers
"did this agent run succeed and how long did it take" for every agent;
this page answers "what did each ML experiment actually produce, and how
did it compare to the others" -- ML-specific, metric-first. Only real
MLAgent.run() calls appear here; run_robustness_check's seeds are
exploratory and are never logged as experiments (see MLAgent.run for why).
"""

import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "app"))

import pandas as pd
import streamlit as st

import dashboard_helpers as dh
from src.tools.audit_db import get_recent_experiments

st.set_page_config(page_title="Run History | AI Data Analyst", page_icon="📜", layout="wide")
dh.ensure_session_state(st.session_state)

st.title("Run History")
st.caption(
    "Every successful ML run, logged permanently -- unlike the ML Studio "
    "page's report, which only ever shows the most recent run for a "
    "dataset, this page keeps every prior run for comparison."
)

limit = st.slider("Rows to show", min_value=10, max_value=500, value=50, step=10)
rows = get_recent_experiments(limit=limit)

if not rows:
    st.info(
        "No ML experiments recorded yet -- run the pipeline first from the "
        "**Dataset Ingestion** page."
    )
    st.stop()

df = dh.experiments_to_dataframe(rows)

datasets = sorted(df["data_path"].dropna().unique().tolist()) if "data_path" in df.columns else []
models = sorted(df["best_model_name"].dropna().unique().tolist()) if "best_model_name" in df.columns else []

col1, col2 = st.columns(2)
dataset_filter = col1.multiselect("Filter by dataset", datasets, default=datasets)
model_filter = col2.multiselect("Filter by winning model", models, default=models)

filtered = df[df["data_path"].isin(dataset_filter) & df["best_model_name"].isin(model_filter)]

st.subheader("Experiments")
st.dataframe(
    filtered.drop(columns=["test_metrics", "best_hyperparameters"], errors="ignore"),
    use_container_width=True, hide_index=True,
)
st.caption(f"Showing {len(filtered)} of {len(df)} fetched rows (most recent first).")

if not filtered.empty:
    st.subheader("Primary Metric Over Time")
    st.caption(
        "primary_metric_name is chosen per run (ROC-AUC/F1 for "
        "classification, R^2/RMSE for regression, or whatever this "
        "report's first metric is named) -- comparing across a mix of "
        "task types on one axis is not meaningful, so this chart only "
        "makes sense when the filtered rows share one metric."
    )
    metric_names = sorted(filtered["primary_metric_name"].dropna().unique().tolist())
    if len(metric_names) > 1:
        st.warning(
            f"Filtered rows use {len(metric_names)} different primary metrics "
            f"({', '.join(metric_names)}) -- narrow the filters above to a "
            "single dataset/task type for a directly comparable chart."
        )
    else:
        chart_df = (
            filtered[["logged_at", "primary_metric_value"]]
            .dropna()
            .sort_values("logged_at")
            .set_index("logged_at")
        )
        if not chart_df.empty:
            st.line_chart(chart_df["primary_metric_value"])

    st.subheader("Inspect a Run")
    selected_id = st.selectbox("Row id", filtered["id"].tolist())
    selected_row = next(r for r in rows if r["id"] == selected_id)
    icol1, icol2 = st.columns(2)
    with icol1:
        st.markdown("**test_metrics**")
        st.json(selected_row.get("test_metrics") or {})
        if selected_row.get("cv_scores"):
            st.markdown("**cv_scores**")
            st.json(selected_row["cv_scores"])
    with icol2:
        st.markdown("**best_hyperparameters**")
        st.json(selected_row.get("best_hyperparameters") or {})
        if selected_row.get("model_selection_note"):
            st.caption(selected_row["model_selection_note"])
