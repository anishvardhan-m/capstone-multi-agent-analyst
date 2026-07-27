"""
app/pages/2_Dataset_Ingestion.py

Dataset Ingestion Portal (capstone handbook, Section 13). Human-in-the-loop
step: the system can clean/explore/model any CSV, but it cannot guess which
column is the prediction target, so the user picks it here before the
pipeline runs.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "app"))

import pandas as pd
import streamlit as st

import dashboard_helpers as dh
from src.agents.orchestrator import OrchestratorAgent

st.set_page_config(page_title="Dataset Ingestion | AI Data Analyst", page_icon="📥", layout="wide")
dh.ensure_session_state(st.session_state)

st.title("Dataset Ingestion Portal")
st.caption(
    "Upload a CSV, tell the system what to predict, then run the full "
    "Cleaning → EDA → Feature Engineering → ML → Visualization → Insights → "
    "Report pipeline."
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to get started.")
else:
    file_bytes = uploaded_file.getvalue()
    try:
        preview_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this file as CSV: {exc}")
        preview_df = None

    if preview_df is not None:
        st.success(f"Loaded `{uploaded_file.name}` — {preview_df.shape[0]:,} rows × {preview_df.shape[1]} columns")

        n_rows = st.slider("Rows to preview", min_value=5, max_value=min(100, max(5, len(preview_df))), value=10)
        st.dataframe(preview_df.head(n_rows), use_container_width=True)

        st.subheader("Modeling configuration")
        st.caption(
            "You choose the target column — the system can't guess your intent. "
            "Once you pick it, MLAgent auto-detects the task type from its values: "
            "2 unique values → binary classification, a handful of distinct "
            "values → multiclass classification, otherwise → regression."
        )
        columns = preview_df.columns.tolist()
        target_col = st.selectbox("Target column (what should the model predict?)", columns)
        id_col_choice = st.selectbox(
            "ID / row-identifier column (optional — excluded from ML features)",
            ["(none)"] + columns,
        )
        group_col_choice = st.selectbox(
            "Group column (optional — e.g. a repeat-customer ID)",
            ["(none)"] + columns,
            help=(
                "If one real-world entity (a customer, a household, a store) "
                "can contribute more than one row, pick that entity's ID here. "
                "The train/test split will then keep every row for the same "
                "entity entirely on one side, so the model can't be evaluated "
                "on a customer it partly trained on — otherwise the reported "
                "test accuracy would be optimistically inflated."
            ),
        )

        with st.expander("Optional: business vocabulary for charts and the insights report"):
            st.caption(
                "Leave blank to use generic labels. Only positive/negative class "
                "labels matter for classification tasks; unit label applies to both."
            )
            positive_label = st.text_input("Positive class label (e.g. 'late delivery')", "")
            negative_label = st.text_input("Negative class label (e.g. 'on-time delivery')", "")
            unit_label = st.text_input("Row/record label (e.g. 'order', 'house')", "")

        run_clicked = st.button("Save & Run Full Pipeline", type="primary")

        if run_clicked:
            dest_dir = os.path.join(_PROJECT_ROOT, "data", "raw")
            saved_path = dh.save_uploaded_file(file_bytes, uploaded_file.name, dest_dir)

            id_col = None if id_col_choice == "(none)" else id_col_choice
            group_col = None if group_col_choice == "(none)" else group_col_choice

            st.session_state["raw_data_path"] = saved_path
            st.session_state["target_col"] = target_col
            st.session_state["id_col"] = id_col
            st.session_state["group_col"] = group_col
            st.session_state.update(dh.derive_pipeline_paths(saved_path))

            with st.spinner(
                "Running the full pipeline — this trains multiple models via "
                "cross-validation and can take a few minutes..."
            ):
                agent = OrchestratorAgent()
                success, result = agent.run(
                    data_path=saved_path,
                    target_col=target_col,
                    id_col=id_col,
                    group_col=group_col,
                    positive_label=positive_label or None,
                    negative_label=negative_label or None,
                    unit_label=unit_label or None,
                )

            st.session_state["pipeline_ran"] = True
            st.session_state["pipeline_success"] = success
            st.session_state["orchestrator_message"] = result
            st.session_state["orchestrator_report"] = agent.report_.to_dict() if agent.report_ else None

            if success:
                st.success(f"Pipeline complete. Final report: {result}")
            else:
                st.error(f"Pipeline aborted: {result}")

if st.session_state.get("pipeline_ran"):
    st.divider()
    st.subheader("Last run summary")
    report = st.session_state.get("orchestrator_report")
    if report:
        steps_df = pd.DataFrame(report["steps"])
        st.dataframe(steps_df, use_container_width=True)
        st.caption(f"Total duration: {report['total_duration_seconds']:.1f}s")
        if report.get("aborted"):
            st.error(f"Aborted: {report.get('abort_reason')}")
    else:
        status = "succeeded" if st.session_state.get("pipeline_success") else "failed"
        st.write(f"Last run {status}: {st.session_state.get('orchestrator_message')}")
