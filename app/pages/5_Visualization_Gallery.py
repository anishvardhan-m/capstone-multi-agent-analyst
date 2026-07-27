"""
app/pages/5_Visualization_Gallery.py

Visualization Gallery (capstone handbook, Section 13): every chart PNG in
workspace/visualizations/, laid out in a grid with captions. Reads directly
from the fixed workspace path rather than session_state, so charts from any
previous pipeline run (even in an earlier session) still show up.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "app"))

import streamlit as st

import dashboard_helpers as dh

st.set_page_config(page_title="Visualization Gallery | AI Data Analyst", page_icon="📈", layout="wide")
dh.ensure_session_state(st.session_state)

st.title("Visualization Gallery")

viz_dir = os.path.join(_PROJECT_ROOT, "workspace", "visualizations")
charts = dh.list_visualization_charts(viz_dir)
available = [c for c in charts if c["exists"]]

if not available:
    st.info("No visualizations yet — run the pipeline first from the **Dataset Ingestion** page.")
    st.stop()

n_cols = 2
columns = st.columns(n_cols)
for i, chart in enumerate(available):
    with columns[i % n_cols]:
        st.image(chart["path"], caption=chart["caption"], use_container_width=True)

missing = [c for c in charts if not c["exists"]]
if missing:
    st.caption(
        "Some charts from the last run were skipped or are missing: "
        + ", ".join(c["name"] for c in missing)
    )
