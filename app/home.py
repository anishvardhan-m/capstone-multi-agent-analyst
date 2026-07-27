"""
app/home.py

Home page / entry point for the AI Data Analyst dashboard (capstone
handbook, Section 13). Run with:

    streamlit run app/home.py

Pure UI: project intro, usage instructions, and a system status check.
No agent logic lives here -- see app/dashboard_helpers.py for the checks
this page renders.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "app"))

import streamlit as st

import dashboard_helpers as dh

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
dh.ensure_session_state(st.session_state)

st.title("Multi-Agent AI Data Analyst")

st.markdown(
    """
A multi-agent system that takes any tabular dataset (CSV) and autonomously
cleans it, explores it, engineers features, trains and evaluates ML models,
generates visualizations, writes business insights, and compiles an
executive PDF report — with a self-correcting orchestration layer that
recovers from agent errors automatically.

Capstone project — 6 Month Internship in Data Science, AI & ML
(Techible x IIT Jammu).
"""
)

st.subheader("How to use this dashboard")
st.markdown(
    """
1. **Dataset Ingestion** — upload a CSV, preview it, pick a target column
   (and optionally an ID/group column), then run the full 7-agent pipeline.
   You pick the target — the system can't guess your intent — but it
   auto-detects the task type (classification vs. regression) from that
   column's values.
2. **Data Analysis Console** — review what the Cleaning and EDA agents
   found in your data.
3. **ML Studio** — see which model won, its hyperparameters, and its
   held-out test performance.
4. **Visualization Gallery** — browse the generated charts.
5. **Insights Panel** — read the LLM-generated business narrative.
6. **Reports Hub** — download the compiled executive PDF report.
7. **System Log Explorer** — inspect the audit trail of every agent run.

Pages 2 onward are numbered in the sidebar in the order above; run the
pipeline from **Dataset Ingestion** first, since every later page displays
that run's output.
"""
)

st.divider()
st.subheader("System status")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Configuration**")
    env_status = dh.get_env_status()
    if env_status["api_key_configured"]:
        st.success("OpenRouter/OpenAI-compatible API key configured (.env)")
    else:
        st.error(
            "No API key configured — copy `.env.example` to `.env` and fill in "
            "`OPENAI_API_KEY`. The Business Insights and Orchestrator recovery "
            "steps need this to call an LLM."
        )
    st.caption(f"API base URL: {env_status['api_base_url'] or 'not set'}")

with col2:
    st.markdown("**Directories**")
    dir_status = dh.get_directory_status()
    for label, exists in dir_status.items():
        st.write(("✅ " if exists else "❌ ") + f"`{label}/`")

if st.session_state.get("pipeline_ran"):
    st.divider()
    st.subheader("Last pipeline run (this session)")
    if st.session_state.get("pipeline_success"):
        st.success(f"Succeeded: {st.session_state.get('orchestrator_message')}")
    else:
        st.error(f"Did not complete: {st.session_state.get('orchestrator_message')}")
