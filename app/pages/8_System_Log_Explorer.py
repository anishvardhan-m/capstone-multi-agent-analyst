"""
app/pages/8_System_Log_Explorer.py

System Log Explorer (capstone handbook, Section 13): a queryable view of
the SQLite audit trail (src/tools/audit_db.py's agent_runs table) — which
agent ran, when, with what status, and how long it took.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "app"))

import streamlit as st

import dashboard_helpers as dh
from src.tools.audit_db import get_recent_runs

st.set_page_config(page_title="System Log Explorer | AI Data Analyst", page_icon="🗒️", layout="wide")
dh.ensure_session_state(st.session_state)

st.title("System Log Explorer")
st.caption("Live query of the audit trail every agent writes to on every run.")

limit = st.slider("Rows to show", min_value=10, max_value=500, value=50, step=10)
rows = get_recent_runs(limit=limit)

if not rows:
    st.info("No agent runs recorded yet — run the pipeline first from the **Dataset Ingestion** page.")
    st.stop()

df = dh.runs_to_dataframe(rows)

agents = sorted(df["agent_name"].dropna().unique().tolist()) if "agent_name" in df.columns else []
statuses = sorted(df["status"].dropna().unique().tolist()) if "status" in df.columns else []

col1, col2 = st.columns(2)
agent_filter = col1.multiselect("Filter by agent", agents, default=agents)
status_filter = col2.multiselect("Filter by status", statuses, default=statuses)

filtered = df[df["agent_name"].isin(agent_filter) & df["status"].isin(status_filter)]

st.dataframe(filtered, use_container_width=True, hide_index=True)
st.caption(f"Showing {len(filtered)} of {len(df)} fetched rows (most recent first).")
