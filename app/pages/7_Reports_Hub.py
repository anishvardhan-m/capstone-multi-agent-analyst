"""
app/pages/7_Reports_Hub.py

Reports Hub (capstone handbook, Section 13): download the compiled
executive PDF report, with basic file metadata. Reads this session's own
workspace/{run_id}/executive_report.pdf, falling back to the fixed
workspace/executive_report.pdf (the Olist demo's committed output) when
this session hasn't run its own pipeline yet, same pattern as the
Gallery/Insights pages.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "app"))

import streamlit as st

import dashboard_helpers as dh

st.set_page_config(page_title="Reports Hub | AI Data Analyst", page_icon="📄", layout="wide")
dh.ensure_session_state(st.session_state)

st.title("Reports Hub")

report_path, report_source = dh.resolve_report_path(
    st.session_state.get("report_pdf_path"), dh.DEFAULT_REPORT_PDF_REL
)
meta = dh.get_file_metadata(report_path)

if not meta["exists"]:
    st.info("No executive report yet — run the pipeline first from the **Dataset Ingestion** page.")
    st.stop()

if report_source == "fallback":
    st.caption(
        "Showing output from a previous run (no pipeline run found in this "
        "session) — run the pipeline yourself from **Dataset Ingestion** to "
        "see your own dataset's report here."
    )

col1, col2 = st.columns(2)
col1.metric("File size", meta["size_human"])
col2.metric("Generated at", meta["modified_at"])

with open(report_path, "rb") as f:
    pdf_bytes = f.read()

st.download_button(
    "Download Executive Report (PDF)",
    data=pdf_bytes,
    file_name="executive_report.pdf",
    mime="application/pdf",
    type="primary",
)

st.divider()
st.subheader("Known Limitations")
st.caption(
    "Specific, verified limitations this project's own rigor testing "
    "found -- calibration, accuracy framing, seed-sensitivity of "
    "feature importance, and error concentration by segment -- not a "
    "generic disclaimer."
)
limitations_path = os.path.join(_PROJECT_ROOT, "KNOWN_LIMITATIONS.md")
if os.path.exists(limitations_path):
    with open(limitations_path) as f:
        limitations_md = f.read()
    with st.expander("Read Known Limitations", expanded=False):
        st.markdown(limitations_md)
    st.download_button(
        "Download Known Limitations (Markdown)",
        data=limitations_md,
        file_name="KNOWN_LIMITATIONS.md",
        mime="text/markdown",
    )
else:
    st.caption("KNOWN_LIMITATIONS.md not found.")
