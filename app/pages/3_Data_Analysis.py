"""
app/pages/3_Data_Analysis.py

Data Analysis Console (capstone handbook, Section 13): the Cleaning
Agent's and EDA Agent's reports, rendered as readable tables rather than
raw JSON.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "app"))

import pandas as pd
import streamlit as st

import dashboard_helpers as dh

st.set_page_config(page_title="Data Analysis | AI Data Analyst", page_icon="🔍", layout="wide")
dh.ensure_session_state(st.session_state)

st.title("Data Analysis Console")

cleaning_report_path, cleaning_source = dh.resolve_report_path(
    st.session_state.get("cleaning_report_path"), dh.DEFAULT_CLEANING_REPORT_REL
)
eda_report_path, eda_source = dh.resolve_report_path(
    st.session_state.get("eda_report_path"), dh.DEFAULT_EDA_REPORT_REL
)

if not cleaning_report_path and not eda_report_path:
    st.info("Run the pipeline first from the **Dataset Ingestion** page.")
    st.stop()

ml_report_path, _ = dh.resolve_report_path(
    st.session_state.get("ml_report_path"), dh.DEFAULT_ML_REPORT_REL
)
ml_report = dh.load_json_report(ml_report_path) if ml_report_path else None
banner = dh.prediction_banner_text(ml_report) if ml_report else None
if banner:
    st.markdown(f"**{banner}**")

if "fallback" in (cleaning_source, eda_source):
    st.caption(
        "Showing output from a previous run (no pipeline run found in this "
        "session) — run the pipeline yourself from **Dataset Ingestion** to "
        "see your own dataset's results here."
    )

cleaning_report = dh.load_json_report(cleaning_report_path)
eda_report = dh.load_json_report(eda_report_path)

st.subheader("Data Cleaning Report")
if cleaning_report is None:
    st.warning("Cleaning report unavailable (file missing or the step was skipped/failed).")
else:
    rows = dh.cleaning_report_rows(cleaning_report)
    st.table(pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric"))

st.divider()
st.subheader("Exploratory Data Analysis Report")
if eda_report is None:
    st.warning("EDA report unavailable (file missing or the step was skipped/failed).")
else:
    st.markdown("**Descriptive statistics**")
    stats_df = dh.eda_descriptive_stats_df(eda_report)
    if stats_df.empty:
        st.caption("No numeric columns to describe.")
    else:
        st.dataframe(stats_df, use_container_width=True)

    st.markdown("**Correlation matrix**")
    corr_df = dh.eda_correlation_df(eda_report)
    if corr_df.empty:
        st.caption("Fewer than 2 numeric columns — no correlation matrix.")
    else:
        st.dataframe(
            corr_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"),
            use_container_width=True,
        )

    st.markdown("**Skewness**")
    skew_df = dh.eda_skewness_df(eda_report)
    if skew_df.empty:
        st.caption("No numeric columns.")
    else:
        st.dataframe(skew_df, use_container_width=True, hide_index=True)

    st.markdown("**Outliers (IQR method)**")
    outliers_df = dh.eda_outliers_df(eda_report)
    if outliers_df.empty:
        st.caption("No numeric columns.")
    else:
        st.dataframe(outliers_df, use_container_width=True)
