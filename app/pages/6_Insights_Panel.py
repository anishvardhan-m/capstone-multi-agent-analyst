"""
app/pages/6_Insights_Panel.py

Insights Panel (capstone handbook, Section 13): renders
workspace/business_insights.md (the Business Insights Agent's LLM-written
narrative) as formatted markdown. Reads the fixed workspace path directly,
same rationale as the Visualization Gallery.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "app"))

import streamlit as st

import dashboard_helpers as dh

st.set_page_config(page_title="Insights Panel | AI Data Analyst", page_icon="💡", layout="wide")
dh.ensure_session_state(st.session_state)

st.title("Business Insights Panel")

ml_report_path, _ = dh.resolve_report_path(
    st.session_state.get("ml_report_path"), dh.DEFAULT_ML_REPORT_REL
)
ml_report = dh.load_json_report(ml_report_path) if ml_report_path else None
banner = dh.prediction_banner_text(ml_report) if ml_report else None
if banner:
    st.markdown(f"**{banner}**")

insights_path = os.path.join(_PROJECT_ROOT, "workspace", "business_insights.md")

if not os.path.isfile(insights_path):
    st.info("No business insights yet — run the pipeline first from the **Dataset Ingestion** page.")
    st.stop()

try:
    with open(insights_path) as f:
        md_text = f.read()
except OSError as exc:
    st.error(f"Could not read business insights file: {exc}")
    st.stop()

if not md_text.strip():
    st.warning("Business insights file is empty.")
else:
    caveat = dh.calibration_caveat(ml_report) if ml_report else None
    if caveat:
        st.info(caveat)

    st.markdown(md_text)
