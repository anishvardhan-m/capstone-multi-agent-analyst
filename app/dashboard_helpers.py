"""
app/dashboard_helpers.py

Pure(ish) helper functions extracted out of the Streamlit pages so they can
be unit tested without a running Streamlit session -- report loading,
report-to-DataFrame formatting, file-existence/metadata checks, pipeline
output-path derivation, and session-state defaults.

Nothing in this module makes an LLM call or trains a model; it only reads
already-written JSON reports / files and reshapes them for display. All
functions degrade gracefully on missing/malformed input (return None or an
empty DataFrame) rather than raising, since the dashboard's whole point is
to handle a fresh install with no pipeline run yet.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_EXPECTED_DIRS = {
    "data/raw": os.path.join("data", "raw"),
    "data/processed": os.path.join("data", "processed"),
    "models": "models",
    "workspace": "workspace",
    "workspace/visualizations": os.path.join("workspace", "visualizations"),
    "workspace/metadata": os.path.join("workspace", "metadata"),
}

_PLACEHOLDER_KEY_MARKERS = ("xxxxxxxx", "your-api-key", "sk-proj-xxxx")


# ---------------------------------------------------------------------------
# Home page: system status
# ---------------------------------------------------------------------------

def get_env_status(project_root: str = PROJECT_ROOT) -> dict:
    """Check whether an OpenRouter/OpenAI-compatible API key is configured.

    Only checks presence/shape of the key from .env -- never returns the
    raw key value (it's a secret; the dashboard must never render it).
    Does not make a network call, so this is instant and free to run on
    every Home page load.
    """
    # override=True: a status check must reflect the target .env file
    # exactly, not whatever OPENAI_API_KEY some earlier load_dotenv() call
    # (e.g. from importing insights.py/orchestrator.py) already left in
    # os.environ -- python-dotenv's default (override=False) would let
    # that stale value silently win.
    load_dotenv(os.path.join(project_root, ".env"), override=True)
    api_key = os.getenv("OPENAI_API_KEY", "") or ""
    api_base_url = os.getenv("OPENAI_API_BASE_URL", "") or ""
    stripped = api_key.strip()
    looks_like_placeholder = any(marker in stripped.lower() for marker in _PLACEHOLDER_KEY_MARKERS)
    key_configured = bool(stripped) and not looks_like_placeholder
    return {
        "api_key_configured": key_configured,
        "api_base_url": api_base_url or None,
    }


def get_directory_status(project_root: str = PROJECT_ROOT) -> dict:
    """Return {label: exists} for each directory the pipeline depends on."""
    return {
        label: os.path.isdir(os.path.join(project_root, rel))
        for label, rel in _EXPECTED_DIRS.items()
    }


# ---------------------------------------------------------------------------
# Dataset Ingestion (P2)
# ---------------------------------------------------------------------------

def save_uploaded_file(file_bytes: bytes, filename: str, dest_dir: str) -> str:
    """Write uploaded CSV bytes to dest_dir, returning the saved path.

    Uses only the basename of `filename` so a maliciously crafted upload
    name (e.g. containing '../') can never write outside dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = os.path.basename(filename)
    dest_path = os.path.join(dest_dir, safe_name)
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
    return dest_path


def make_run_id(filename: str) -> str:
    """Derive a filesystem-safe run_id from an uploaded filename's stem.

    Passed to OrchestratorAgent.run(run_id=...) so a dataset's model file
    and workspace outputs (charts, insights, PDF) land under
    models/{run_id}_... and workspace/{run_id}/ instead of the fixed
    default paths the Olist demo ships pre-run output at -- otherwise
    uploading any new dataset would silently overwrite that demo output.
    Falls back to a timestamp if the filename sanitizes to nothing (e.g.
    a name made entirely of punctuation).
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    sanitized = re.sub(r"[^a-z0-9_-]+", "_", stem.lower()).strip("_")
    return sanitized or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def derive_workspace_paths(run_id: str) -> dict:
    """Reconstruct this run's namespaced workspace output paths (charts
    dir, insights markdown, final PDF) from its run_id, mirroring the
    exact `workspace/{run_id}/...` layout OrchestratorAgent.run() uses
    when given a run_id. Needed so the dashboard's session_state can
    point the Visualization Gallery / Insights Panel / Reports Hub pages
    at this run's own outputs instead of the fixed workspace/ defaults.
    """
    workspace_dir = os.path.join(PROJECT_ROOT, "workspace", run_id)
    return {
        "viz_dir": os.path.join(workspace_dir, "visualizations"),
        "insights_md_path": os.path.join(workspace_dir, "business_insights.md"),
        "report_pdf_path": os.path.join(workspace_dir, "executive_report.pdf"),
    }


def derive_pipeline_paths(raw_data_path: str) -> dict:
    """Reconstruct every intermediate artifact path the pipeline will
    produce from a raw input CSV path, following the exact naming
    convention each agent's own run() uses (cleaner.py, eda.py,
    feature_engineer.py, ml_agent.py). Needed because the orchestrator's
    return value is only the final PDF path -- the dashboard needs the
    intermediate report paths too, to show P3/P4 without re-deriving logic
    that already lives in those agents.
    """
    base, ext = os.path.splitext(raw_data_path)
    cleaned_csv_path = f"{base}_cleaned{ext}"
    features_csv_path = f"{base}_cleaned_features{ext}"
    return {
        "cleaned_csv_path": cleaned_csv_path,
        "cleaning_report_path": f"{base}_cleaned_report.json",
        "eda_report_path": f"{base}_cleaned_eda_report.json",
        "features_csv_path": features_csv_path,
        "features_report_path": f"{base}_cleaned_features_features_report.json",
        "ml_report_path": f"{base}_cleaned_features_ml_report.json",
    }


# Standard output paths for this project's own Olist demo dataset (see
# README / data/flatten_olist.py). Used as a fallback on P3/P4 so a fresh
# session (or a page reload) still shows real prior pipeline output instead
# of an empty state, exactly like P5-P8 already do by reading fixed
# workspace/ paths -- these are P3/P4's equivalent "fixed path" for the one
# dataset this repo ships pre-run output for.
DEFAULT_CLEANING_REPORT_REL = os.path.join("data", "processed", "olist_flattened_cleaned_report.json")
DEFAULT_EDA_REPORT_REL = os.path.join("data", "processed", "olist_flattened_cleaned_eda_report.json")
DEFAULT_ML_REPORT_REL = os.path.join("data", "processed", "olist_flattened_cleaned_features_ml_report.json")

# Same rationale, for the run_id-namespaced workspace outputs (P5-P7):
# these are the fixed paths OrchestratorAgent.run() writes to when no
# run_id is given -- i.e. the Olist demo's own committed workspace output.
DEFAULT_VIZ_DIR_REL = os.path.join("workspace", "visualizations")
DEFAULT_INSIGHTS_MD_REL = os.path.join("workspace", "business_insights.md")
DEFAULT_REPORT_PDF_REL = os.path.join("workspace", "executive_report.pdf")


def resolve_report_path(
    session_path: Optional[str],
    default_relative_path: str,
    project_root: str = PROJECT_ROOT,
) -> tuple:
    """Pick which report path to display: this session's own pipeline run
    if it produced one and the file still exists, otherwise the standard
    demo-dataset path if that exists, otherwise nothing.

    Returns (path_or_None, source) where source is "session", "fallback",
    or None -- callers use `source` to tell the user when they're looking
    at prior/demo output rather than a run from this session.
    """
    if session_path and os.path.isfile(session_path):
        return session_path, "session"
    fallback = os.path.join(project_root, default_relative_path)
    if os.path.isfile(fallback):
        return fallback, "fallback"
    return None, None


def resolve_dir_path(
    session_path: Optional[str],
    default_relative_path: str,
    project_root: str = PROJECT_ROOT,
) -> tuple:
    """Directory counterpart to resolve_report_path (for the Visualization
    Gallery's chart directory): this session's own run_id-namespaced
    workspace dir if it exists, otherwise the fixed Olist demo directory,
    otherwise nothing.
    """
    if session_path and os.path.isdir(session_path):
        return session_path, "session"
    fallback = os.path.join(project_root, default_relative_path)
    if os.path.isdir(fallback):
        return fallback, "fallback"
    return None, None


# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------

def load_json_report(path: Optional[str]) -> Optional[dict]:
    """Load a JSON report, returning None on any missing/unreadable/
    malformed file instead of raising -- the standard graceful-degradation
    contract every agent in this project already follows."""
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Data Analysis Console (P3)
# ---------------------------------------------------------------------------

def _fmt_shape(shape) -> str:
    if not shape or len(shape) != 2:
        return "unknown"
    try:
        return f"{int(shape[0]):,} rows × {int(shape[1])} columns"
    except (TypeError, ValueError):
        return f"{shape[0]} rows × {shape[1]} columns"


def _fmt_list(items: Optional[list]) -> str:
    return ", ".join(str(i) for i in items) if items else "None"


def cleaning_report_rows(report: dict) -> list:
    """Flatten a CleaningReport dict into (label, value) display rows."""
    return [
        ("Input shape", _fmt_shape(report.get("input_shape"))),
        ("Output shape", _fmt_shape(report.get("output_shape"))),
        ("Duplicate rows removed", f"{report.get('n_duplicates_removed', 0):,}"),
        ("High-missing columns flagged", _fmt_list(report.get("high_missing_columns_flagged"))),
        ("Low-variance columns dropped", _fmt_list(report.get("low_variance_columns_dropped"))),
        ("Columns type-corrected", _fmt_list(report.get("columns_type_corrected"))),
        ("Numeric columns imputed", _fmt_list(report.get("numeric_columns_imputed"))),
        ("Categorical columns imputed", _fmt_list(report.get("categorical_columns_imputed"))),
    ]


def eda_descriptive_stats_df(report: dict) -> pd.DataFrame:
    stats = report.get("descriptive_stats") or {}
    if not stats:
        return pd.DataFrame()
    df = pd.DataFrame(stats).T
    df.index.name = "column"
    return df


def eda_correlation_df(report: dict) -> pd.DataFrame:
    corr = report.get("correlation_matrix") or {}
    if not corr:
        return pd.DataFrame()
    return pd.DataFrame(corr)


def eda_skewness_df(report: dict) -> pd.DataFrame:
    skew = report.get("skewness") or {}
    if not skew:
        return pd.DataFrame(columns=["column", "skewness"])
    rows = sorted(skew.items(), key=lambda kv: abs(kv[1]) if kv[1] is not None else -1, reverse=True)
    return pd.DataFrame(rows, columns=["column", "skewness"])


def eda_outliers_df(report: dict) -> pd.DataFrame:
    outliers = report.get("outlier_summary") or {}
    if not outliers:
        return pd.DataFrame()
    df = pd.DataFrame(outliers).T
    df.index.name = "column"
    if "pct_outliers" in df.columns:
        df = df.sort_values("pct_outliers", ascending=False)
    return df


# ---------------------------------------------------------------------------
# ML Studio (P4)
# ---------------------------------------------------------------------------

def ml_summary(report: dict) -> dict:
    return {
        "task_type": report.get("task_type", "unknown"),
        "best_model_name": report.get("best_model_name", "unknown"),
    }


_TASK_TYPE_DISPLAY_LABELS = {
    "binary_classification": "Classification",
    "multiclass_classification": "Classification",
    "regression": "Regression",
}


def format_task_type_label(task_type: Optional[str]) -> str:
    """Map an MLReport task_type onto the coarse Classification/Regression
    label the dashboard's prediction banner uses (handbook F4)."""
    return _TASK_TYPE_DISPLAY_LABELS.get(task_type, "Unknown")


def prediction_banner_text(report: Optional[dict]) -> Optional[str]:
    """"Predicting: {target} (task type: Classification/Regression,
    auto-detected)" banner text for P3/P4/P6 (handbook F4). Reads
    target_col straight from the ML report (set by MLAgent itself), never
    a hardcoded per-dataset default -- so it's correct for any dataset,
    not just the Olist demo. Returns None when the report or its
    target_col isn't available yet (fresh install, pipeline not run)."""
    if not report:
        return None
    target_col = report.get("target_col")
    if not target_col:
        return None
    task_label = format_task_type_label(report.get("task_type"))
    return f"Predicting: {target_col} (task type: {task_label}, auto-detected)"


def ml_cv_scores_df(report: dict) -> pd.DataFrame:
    """Model comparison table with per-model CV std alongside the mean
    (F2 rigor: a point estimate alone hides whether a "win" is real)."""
    scores = report.get("cv_scores") or {}
    if not scores:
        return pd.DataFrame(columns=["model", "cv_score", "cv_std", "is_best"])
    best = report.get("best_model_name")
    std = report.get("cv_std") or {}
    rows = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame(
        [
            {"model": name, "cv_score": score, "cv_std": std.get(name), "is_best": name == best}
            for name, score in rows
        ]
    )


def ml_test_metrics_df(report: dict) -> pd.DataFrame:
    metrics = report.get("test_metrics") or {}
    if not metrics:
        return pd.DataFrame(columns=["metric", "value"])
    return pd.DataFrame(sorted(metrics.items()), columns=["metric", "value"])


def ml_feature_importances_df(report: dict) -> pd.DataFrame:
    """Permutation importance per feature: mean, std, and whether the
    mean +/- std range is distinguishable from zero (see MLReport docs)."""
    importances = report.get("feature_importances") or {}
    if not importances:
        return pd.DataFrame(
            columns=["feature", "importance_mean", "importance_std", "distinguishable_from_zero"]
        )
    rows = sorted(
        importances.items(), key=lambda kv: kv[1]["importance_mean"], reverse=True
    )
    return pd.DataFrame(
        [
            (feat, v["importance_mean"], v["importance_std"], v["distinguishable_from_zero"])
            for feat, v in rows
        ],
        columns=["feature", "importance_mean", "importance_std", "distinguishable_from_zero"],
    )


def ml_error_analysis_df(report: dict) -> pd.DataFrame:
    """Long-format segment breakdown from error_analysis (handbook F6):
    one row per (segment column, segment value). Columns adapt to
    whatever metrics this task type reported -- false_negative_rate/
    false_positive_rate for binary classification, misclassification_rate
    for multiclass, mae/mean_error for regression -- plus their
    elevated_* flags, so nothing here is hardcoded to one task type or
    column set.
    """
    error_analysis = report.get("error_analysis") or {}
    segments = error_analysis.get("segments") or {}
    if not segments:
        return pd.DataFrame(columns=["segment_column", "segment_value", "n"])

    records = [
        {"segment_column": col, **row}
        for col, rows in segments.items()
        for row in rows
    ]
    df = pd.DataFrame.from_records(records)
    leading = ["segment_column", "segment_value", "n"]
    ordered = leading + [c for c in df.columns if c not in leading]
    return df[ordered]


def ml_confusion_matrix_df(report: dict) -> Optional[pd.DataFrame]:
    cm = report.get("confusion_matrix")
    if not cm:
        return None
    n = len(cm)
    index = [f"Actual {i}" for i in range(n)]
    columns = [f"Predicted {i}" for i in range(n)]
    return pd.DataFrame(cm, index=index, columns=columns)


def calibration_caveat(report: dict) -> Optional[str]:
    """Class-weight/threshold caveat shown on P4 (ML Studio) and P6
    (Insights Panel) -- same message as the PDF report's caveat (F3):
    class_weight='balanced' shifts raw probabilities, so a threshold is a
    decision cutoff, not a literal percentage chance. None when the report
    isn't binary classification (nothing to caveat)."""
    if not report or report.get("task_type") != "binary_classification":
        return None
    text = (
        "This model uses class-weighted training, so its predicted "
        "probabilities are not literal percentages. Thresholds mentioned "
        'here (e.g. "threshold=0.3") are decision cutoffs chosen for a '
        "precision/recall tradeoff, not the actual chance of the outcome."
    )
    calibration = report.get("calibration")
    if calibration and calibration.get("ece") is not None:
        text += f" Measured calibration error (ECE): {calibration['ece']:.3f}."
    return text


def ml_test_predictions_df(report: dict, max_rows: int = 25) -> Optional[pd.DataFrame]:
    """Regression counterpart to the confusion matrix: a sample of held-out
    actual/predicted pairs. Returns None for classification tasks (no
    test_predictions field) or when the field is empty."""
    preds = report.get("test_predictions")
    if not preds or not preds.get("actual") or not preds.get("predicted"):
        return None
    df = pd.DataFrame({"actual": preds["actual"], "predicted": preds["predicted"]})
    df["residual"] = df["actual"] - df["predicted"]
    return df.head(max_rows)


# ---------------------------------------------------------------------------
# Visualization Gallery (P5)
# ---------------------------------------------------------------------------

def list_visualization_charts(viz_dir: str) -> list:
    """Return [{name, path, caption, exists}, ...] for the gallery.

    Prefers VisualizationAgent's own visualization_report.json (accurate
    descriptions, correct generation order); falls back to a plain PNG
    glob of viz_dir if that report is missing/unreadable, so charts still
    show up even if only the report went missing.
    """
    report = load_json_report(os.path.join(viz_dir, "visualization_report.json"))
    charts = []
    if report and report.get("charts"):
        for c in report["charts"]:
            path = c.get("path", "")
            abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
            charts.append({
                "name": c.get("name", ""),
                "path": abs_path,
                "caption": c.get("description") or c.get("name", ""),
                "exists": os.path.isfile(abs_path),
            })
        return charts

    if os.path.isdir(viz_dir):
        for fname in sorted(os.listdir(viz_dir)):
            if fname.lower().endswith(".png"):
                abs_path = os.path.join(viz_dir, fname)
                caption = re.sub(r"^\d+_", "", os.path.splitext(fname)[0]).replace("_", " ").title()
                charts.append({"name": fname, "path": abs_path, "caption": caption, "exists": True})
    return charts


# ---------------------------------------------------------------------------
# Reports Hub (P7)
# ---------------------------------------------------------------------------

def format_file_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def get_file_metadata(path: Optional[str]) -> dict:
    """File existence + size + last-modified time, or an 'absent' record."""
    if not path or not os.path.isfile(path):
        return {"exists": False, "size_bytes": None, "size_human": None, "modified_at": None}
    stat = os.stat(path)
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "size_human": format_file_size(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# System Log Explorer (P8)
# ---------------------------------------------------------------------------

def runs_to_dataframe(rows: list) -> pd.DataFrame:
    """Format get_recent_runs() rows for display: rounded durations, a
    stable column order. Returns an empty-but-columned DataFrame for an
    empty input so the page can render a table shell either way."""
    columns = [
        "id", "agent_name", "status", "duration_seconds",
        "started_at", "finished_at", "input_path", "output_path", "error_message",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    if "duration_seconds" in df.columns:
        df["duration_seconds"] = df["duration_seconds"].round(3)
    present_cols = [c for c in columns if c in df.columns]
    return df[present_cols]


# ---------------------------------------------------------------------------
# Run History / experiment tracking (P9, handbook F9)
# ---------------------------------------------------------------------------

# Preference order for which held-out metric represents a run at a glance.
# Not exhaustive by design: whichever of these keys is present first wins,
# and a report using entirely different metric names still gets a primary
# metric via the fallback below -- nothing here assumes one task type.
_PRIMARY_METRIC_PRIORITY = ("roc_auc", "f1_macro", "adjusted_r2", "r2", "rmse", "mae")


def _pick_primary_metric(test_metrics: dict) -> tuple:
    """Return (metric_name, value) for the metric that best represents a
    run's headline performance, or (None, None) if test_metrics is empty.
    Falls back to whatever the first available metric is named, so a
    dataset whose report uses metric keys not in the priority list above
    still gets a primary metric instead of a blank column.
    """
    for key in _PRIMARY_METRIC_PRIORITY:
        if key in test_metrics:
            return key, test_metrics[key]
    if test_metrics:
        key = next(iter(test_metrics))
        return key, test_metrics[key]
    return None, None


def experiments_to_dataframe(rows: list) -> pd.DataFrame:
    """Format get_recent_experiments() rows for the Run History table: a
    derived primary-metric column up front (see _pick_primary_metric),
    dict-valued fields flattened to compact JSON strings for a clean
    table cell, stable column order. Returns an empty-but-columned
    DataFrame for an empty input so the page can render a table shell
    either way.
    """
    columns = [
        "id", "logged_at", "data_path", "target_col", "task_type",
        "best_model_name", "primary_metric_name", "primary_metric_value",
        "split_strategy", "group_col", "random_state", "n_features",
        "test_metrics", "best_hyperparameters", "model_selection_note",
        "nested_cv_score", "report_path",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    records = []
    for row in rows:
        metric_name, metric_value = _pick_primary_metric(row.get("test_metrics") or {})
        record = dict(row)
        record["primary_metric_name"] = metric_name
        record["primary_metric_value"] = (
            round(metric_value, 4) if metric_value is not None else None
        )
        record["test_metrics"] = json.dumps(row.get("test_metrics") or {})
        record["best_hyperparameters"] = json.dumps(row.get("best_hyperparameters") or {})
        records.append(record)

    df = pd.DataFrame(records)
    present_cols = [c for c in columns if c in df.columns]
    return df[present_cols]


# ---------------------------------------------------------------------------
# Cross-page: output availability + session-state defaults
# ---------------------------------------------------------------------------

def outputs_available(paths: dict) -> dict:
    """{key: bool} for whether each named path exists on disk right now."""
    return {key: bool(p) and os.path.isfile(p) for key, p in paths.items()}


DEFAULT_SESSION_STATE = {
    "raw_data_path": None,
    "target_col": None,
    "id_col": None,
    "group_col": None,
    "pipeline_ran": False,
    "pipeline_success": None,
    "orchestrator_message": None,
    "orchestrator_report": None,
    "run_id": None,
    "cleaned_csv_path": None,
    "cleaning_report_path": None,
    "eda_report_path": None,
    "features_csv_path": None,
    "features_report_path": None,
    "ml_report_path": None,
    "viz_dir": None,
    "insights_md_path": None,
    "report_pdf_path": None,
}


def ensure_session_state(state) -> None:
    """Populate any missing DEFAULT_SESSION_STATE keys on `state`.

    Works against st.session_state (dict-like: supports `in` / __setitem__)
    or a plain dict, so it's testable without a running Streamlit session.
    """
    for key, default in DEFAULT_SESSION_STATE.items():
        if key not in state:
            state[key] = default
