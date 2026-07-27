"""
tests/test_dashboard_helpers.py

Unit tests for app/dashboard_helpers.py -- the pure/file-system-only
helper functions the Streamlit dashboard pages use to load and format
agent reports for display. UI rendering itself isn't tested here (that's
what clicking through the app is for); this covers the parsing/formatting/
file-existence logic that was worth extracting out of the page scripts.
"""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import dashboard_helpers as dh


# ---------------------------------------------------------------------------
# get_env_status
# ---------------------------------------------------------------------------

def test_get_env_status_detects_configured_key(tmp_path):
    (tmp_path / ".env").write_text(
        'OPENAI_API_KEY="sk-real-looking-key-12345"\n'
        'OPENAI_API_BASE_URL="https://openrouter.ai/api/v1"\n'
    )
    status = dh.get_env_status(str(tmp_path))
    assert status["api_key_configured"] is True
    assert status["api_base_url"] == "https://openrouter.ai/api/v1"


def test_get_env_status_flags_placeholder_key(tmp_path):
    (tmp_path / ".env").write_text('OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxx"\n')
    status = dh.get_env_status(str(tmp_path))
    assert status["api_key_configured"] is False


def test_get_env_status_missing_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    status = dh.get_env_status(str(tmp_path / "does_not_exist"))
    assert status["api_key_configured"] is False


def test_get_env_status_never_leaks_raw_key(tmp_path):
    (tmp_path / ".env").write_text('OPENAI_API_KEY="sk-super-secret-value"\n')
    status = dh.get_env_status(str(tmp_path))
    assert "sk-super-secret-value" not in json.dumps(status)


# ---------------------------------------------------------------------------
# get_directory_status
# ---------------------------------------------------------------------------

def test_get_directory_status_all_missing(tmp_path):
    status = dh.get_directory_status(str(tmp_path))
    assert all(exists is False for exists in status.values())


def test_get_directory_status_detects_present_dirs(tmp_path):
    os.makedirs(tmp_path / "workspace" / "visualizations")
    status = dh.get_directory_status(str(tmp_path))
    assert status["workspace"] is True
    assert status["workspace/visualizations"] is True
    assert status["models"] is False


# ---------------------------------------------------------------------------
# save_uploaded_file
# ---------------------------------------------------------------------------

def test_save_uploaded_file_writes_bytes(tmp_path):
    dest_dir = str(tmp_path / "data" / "raw")
    path = dh.save_uploaded_file(b"a,b\n1,2\n", "input.csv", dest_dir)
    assert os.path.isfile(path)
    with open(path, "rb") as f:
        assert f.read() == b"a,b\n1,2\n"


def test_save_uploaded_file_strips_path_traversal(tmp_path):
    dest_dir = str(tmp_path / "data" / "raw")
    path = dh.save_uploaded_file(b"x", "../../etc/evil.csv", dest_dir)
    assert os.path.dirname(path) == dest_dir
    assert os.path.basename(path) == "evil.csv"


# ---------------------------------------------------------------------------
# derive_pipeline_paths
# ---------------------------------------------------------------------------

def test_derive_pipeline_paths_matches_agent_naming_convention():
    paths = dh.derive_pipeline_paths("data/raw/my_data.csv")
    assert paths["cleaned_csv_path"] == "data/raw/my_data_cleaned.csv"
    assert paths["cleaning_report_path"] == "data/raw/my_data_cleaned_report.json"
    assert paths["eda_report_path"] == "data/raw/my_data_cleaned_eda_report.json"
    assert paths["features_csv_path"] == "data/raw/my_data_cleaned_features.csv"
    assert paths["features_report_path"] == "data/raw/my_data_cleaned_features_features_report.json"
    assert paths["ml_report_path"] == "data/raw/my_data_cleaned_features_ml_report.json"


# ---------------------------------------------------------------------------
# resolve_report_path
# ---------------------------------------------------------------------------

def test_resolve_report_path_prefers_session_path(tmp_path):
    session_file = tmp_path / "session_report.json"
    session_file.write_text("{}")
    fallback_dir = tmp_path / "data" / "processed"
    fallback_dir.mkdir(parents=True)
    (fallback_dir / "olist_flattened_cleaned_report.json").write_text("{}")

    path, source = dh.resolve_report_path(
        str(session_file), dh.DEFAULT_CLEANING_REPORT_REL, project_root=str(tmp_path)
    )
    assert path == str(session_file)
    assert source == "session"


def test_resolve_report_path_falls_back_when_session_path_missing_file(tmp_path):
    fallback_dir = tmp_path / "data" / "processed"
    fallback_dir.mkdir(parents=True)
    fallback_file = fallback_dir / "olist_flattened_cleaned_report.json"
    fallback_file.write_text("{}")

    path, source = dh.resolve_report_path(
        str(tmp_path / "does_not_exist.json"), dh.DEFAULT_CLEANING_REPORT_REL, project_root=str(tmp_path)
    )
    assert path == str(fallback_file)
    assert source == "fallback"


def test_resolve_report_path_falls_back_when_session_path_none(tmp_path):
    fallback_dir = tmp_path / "data" / "processed"
    fallback_dir.mkdir(parents=True)
    fallback_file = fallback_dir / "olist_flattened_cleaned_eda_report.json"
    fallback_file.write_text("{}")

    path, source = dh.resolve_report_path(
        None, dh.DEFAULT_EDA_REPORT_REL, project_root=str(tmp_path)
    )
    assert path == str(fallback_file)
    assert source == "fallback"


def test_resolve_report_path_none_when_neither_exists(tmp_path):
    path, source = dh.resolve_report_path(
        None, dh.DEFAULT_ML_REPORT_REL, project_root=str(tmp_path)
    )
    assert path is None
    assert source is None


# ---------------------------------------------------------------------------
# load_json_report
# ---------------------------------------------------------------------------

def test_load_json_report_none_path():
    assert dh.load_json_report(None) is None
    assert dh.load_json_report("") is None


def test_load_json_report_missing_file(tmp_path):
    assert dh.load_json_report(str(tmp_path / "nope.json")) is None


def test_load_json_report_malformed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    assert dh.load_json_report(str(p)) is None


def test_load_json_report_valid(tmp_path):
    p = tmp_path / "good.json"
    p.write_text(json.dumps({"a": 1}))
    assert dh.load_json_report(str(p)) == {"a": 1}


# ---------------------------------------------------------------------------
# cleaning_report_rows
# ---------------------------------------------------------------------------

def test_cleaning_report_rows_full():
    report = {
        "input_shape": [100, 10],
        "output_shape": [95, 9],
        "n_duplicates_removed": 5,
        "high_missing_columns_flagged": ["col_a"],
        "low_variance_columns_dropped": [],
        "columns_type_corrected": ["col_b"],
        "numeric_columns_imputed": ["col_c"],
        "categorical_columns_imputed": [],
    }
    rows = dh.cleaning_report_rows(report)
    rows_dict = dict(rows)
    assert rows_dict["Input shape"] == "100 rows × 10 columns"
    assert rows_dict["Output shape"] == "95 rows × 9 columns"
    assert rows_dict["Duplicate rows removed"] == "5"
    assert rows_dict["High-missing columns flagged"] == "col_a"
    assert rows_dict["Low-variance columns dropped"] == "None"


def test_cleaning_report_rows_empty_dict():
    rows = dh.cleaning_report_rows({})
    rows_dict = dict(rows)
    assert rows_dict["Input shape"] == "unknown"
    assert rows_dict["Duplicate rows removed"] == "0"


# ---------------------------------------------------------------------------
# EDA formatters
# ---------------------------------------------------------------------------

def test_eda_descriptive_stats_df():
    report = {"descriptive_stats": {"col_a": {"mean": 1.0, "std": 0.5}}}
    df = dh.eda_descriptive_stats_df(report)
    assert list(df.index) == ["col_a"]
    assert df.loc["col_a", "mean"] == 1.0


def test_eda_descriptive_stats_df_empty():
    assert dh.eda_descriptive_stats_df({}).empty


def test_eda_correlation_df():
    report = {"correlation_matrix": {"a": {"a": 1.0, "b": 0.5}, "b": {"a": 0.5, "b": 1.0}}}
    df = dh.eda_correlation_df(report)
    assert df.loc["a", "b"] == 0.5


def test_eda_skewness_df_sorted_by_magnitude():
    report = {"skewness": {"low": 0.1, "high": -3.5, "mid": 1.2}}
    df = dh.eda_skewness_df(report)
    assert list(df["column"]) == ["high", "mid", "low"]


def test_eda_outliers_df_sorted_by_pct():
    report = {
        "outlier_summary": {
            "col_a": {"n_outliers": 2, "pct_outliers": 1.0, "lower_fence": 0, "upper_fence": 10},
            "col_b": {"n_outliers": 20, "pct_outliers": 9.5, "lower_fence": 0, "upper_fence": 10},
        }
    }
    df = dh.eda_outliers_df(report)
    assert list(df.index) == ["col_b", "col_a"]


# ---------------------------------------------------------------------------
# ML formatters
# ---------------------------------------------------------------------------

def test_ml_summary():
    report = {"task_type": "binary_classification", "best_model_name": "RandomForestClassifier"}
    assert dh.ml_summary(report) == {
        "task_type": "binary_classification",
        "best_model_name": "RandomForestClassifier",
    }


def test_ml_summary_missing_keys():
    assert dh.ml_summary({}) == {"task_type": "unknown", "best_model_name": "unknown"}


@pytest.mark.parametrize("task_type,expected", [
    ("binary_classification", "Classification"),
    ("multiclass_classification", "Classification"),
    ("regression", "Regression"),
    ("something_unrecognized", "Unknown"),
    (None, "Unknown"),
])
def test_format_task_type_label(task_type, expected):
    assert dh.format_task_type_label(task_type) == expected


def test_prediction_banner_text_binary_classification():
    report = {"target_col": "is_late_delivery", "task_type": "binary_classification"}
    text = dh.prediction_banner_text(report)
    assert text == "Predicting: is_late_delivery (task type: Classification, auto-detected)"


def test_prediction_banner_text_regression():
    report = {"target_col": "house_price", "task_type": "regression"}
    text = dh.prediction_banner_text(report)
    assert text == "Predicting: house_price (task type: Regression, auto-detected)"


def test_prediction_banner_text_none_without_target_col():
    assert dh.prediction_banner_text({"task_type": "regression"}) is None
    assert dh.prediction_banner_text({}) is None
    assert dh.prediction_banner_text(None) is None


def test_ml_cv_scores_df_marks_best():
    report = {
        "best_model_name": "ModelB",
        "cv_scores": {"ModelA": 0.7, "ModelB": 0.9},
    }
    df = dh.ml_cv_scores_df(report)
    assert df.iloc[0]["model"] == "ModelB"
    assert bool(df.iloc[0]["is_best"]) is True
    assert bool(df.iloc[1]["is_best"]) is False


def test_ml_cv_scores_df_includes_std():
    report = {
        "best_model_name": "ModelB",
        "cv_scores": {"ModelA": 0.7, "ModelB": 0.9},
        "cv_std": {"ModelA": 0.05, "ModelB": 0.02},
    }
    df = dh.ml_cv_scores_df(report)
    row_b = df[df["model"] == "ModelB"].iloc[0]
    assert row_b["cv_std"] == pytest.approx(0.02)


def test_ml_cv_scores_df_std_missing_is_none():
    report = {"best_model_name": "ModelA", "cv_scores": {"ModelA": 0.7}}
    df = dh.ml_cv_scores_df(report)
    assert df.iloc[0]["cv_std"] is None


def test_ml_feature_importances_df_sorted_desc():
    report = {
        "feature_importances": {
            "low": {"importance_mean": 0.1, "importance_std": 0.02, "distinguishable_from_zero": True},
            "high": {"importance_mean": 0.9, "importance_std": 0.05, "distinguishable_from_zero": True},
        }
    }
    df = dh.ml_feature_importances_df(report)
    assert list(df["feature"]) == ["high", "low"]
    assert list(df["importance_mean"]) == [0.9, 0.1]


def test_ml_error_analysis_df_empty_when_no_error_analysis():
    df = dh.ml_error_analysis_df({})
    assert df.empty
    assert list(df.columns) == ["segment_column", "segment_value", "n"]


def test_ml_error_analysis_df_flattens_segments_classification():
    report = {
        "error_analysis": {
            "task_type": "binary_classification",
            "segment_columns": ["region"],
            "overall": {"false_negative_rate": 0.3, "false_positive_rate": 0.2},
            "segments": {
                "region": [
                    {"segment_value": 0.1, "n": 50, "n_positive": 20, "n_negative": 30,
                     "false_negative_rate": 0.25, "false_positive_rate": 0.2,
                     "elevated_false_negative_rate": False, "elevated_false_positive_rate": False},
                    {"segment_value": 0.9, "n": 60, "n_positive": 25, "n_negative": 35,
                     "false_negative_rate": 0.7, "false_positive_rate": 0.15,
                     "elevated_false_negative_rate": True, "elevated_false_positive_rate": False},
                ],
            },
        }
    }
    df = dh.ml_error_analysis_df(report)
    assert list(df.columns[:3]) == ["segment_column", "segment_value", "n"]
    assert len(df) == 2
    assert bool(df.loc[df["segment_value"] == 0.9, "elevated_false_negative_rate"].iloc[0]) is True


def test_ml_error_analysis_df_generic_over_arbitrary_segment_column_and_regression_metrics():
    """Non-Olist column name, regression metric shape (mae/mean_error) --
    the helper must not assume any specific task type or column."""
    report = {
        "error_analysis": {
            "task_type": "regression",
            "segment_columns": ["widget_zone"],
            "overall": {"mae": 10.0, "mean_error": 1.0},
            "segments": {
                "widget_zone": [
                    {"segment_value": "zone_a", "n": 40, "mae": 9.5, "mean_error": 0.5,
                     "elevated_mae": False, "elevated_bias": False},
                ],
            },
        }
    }
    df = dh.ml_error_analysis_df(report)
    assert list(df["segment_column"]) == ["widget_zone"]
    assert list(df["segment_value"]) == ["zone_a"]
    assert "mae" in df.columns and "mean_error" in df.columns


def test_ml_confusion_matrix_df():
    report = {"confusion_matrix": [[10, 2], [3, 20]]}
    df = dh.ml_confusion_matrix_df(report)
    assert df.loc["Actual 0", "Predicted 1"] == 2
    assert df.loc["Actual 1", "Predicted 1"] == 20


def test_ml_confusion_matrix_df_none_for_regression():
    assert dh.ml_confusion_matrix_df({"confusion_matrix": None}) is None
    assert dh.ml_confusion_matrix_df({}) is None


def test_ml_test_predictions_df_regression():
    report = {"test_predictions": {"actual": [1.0, 2.0], "predicted": [1.1, 1.9]}}
    df = dh.ml_test_predictions_df(report)
    assert list(df["residual"].round(2)) == [-0.1, 0.1]


def test_ml_test_predictions_df_none_for_classification():
    assert dh.ml_test_predictions_df({"test_predictions": None}) is None
    assert dh.ml_test_predictions_df({}) is None


def test_calibration_caveat_shown_for_binary_classification():
    report = {"task_type": "binary_classification"}
    caveat = dh.calibration_caveat(report)
    assert caveat is not None
    assert "class-weighted training" in caveat


def test_calibration_caveat_none_for_regression():
    assert dh.calibration_caveat({"task_type": "regression"}) is None


def test_calibration_caveat_none_for_empty_report():
    assert dh.calibration_caveat({}) is None
    assert dh.calibration_caveat(None) is None


def test_calibration_caveat_includes_ece_when_present():
    report = {
        "task_type": "binary_classification",
        "calibration": {"ece": 0.1234},
    }
    caveat = dh.calibration_caveat(report)
    assert "0.123" in caveat


# ---------------------------------------------------------------------------
# list_visualization_charts
# ---------------------------------------------------------------------------

def test_list_visualization_charts_uses_report(tmp_path):
    viz_dir = tmp_path / "viz"
    viz_dir.mkdir()
    chart_path = viz_dir / "01_distributions.png"
    chart_path.write_bytes(b"fake-png")
    report = {"charts": [{"name": "distributions", "path": str(chart_path), "description": "Histograms"}]}
    (viz_dir / "visualization_report.json").write_text(json.dumps(report))

    charts = dh.list_visualization_charts(str(viz_dir))
    assert len(charts) == 1
    assert charts[0]["exists"] is True
    assert charts[0]["caption"] == "Histograms"


def test_list_visualization_charts_flags_missing_file(tmp_path):
    viz_dir = tmp_path / "viz"
    viz_dir.mkdir()
    report = {"charts": [{"name": "gone", "path": str(viz_dir / "gone.png"), "description": "d"}]}
    (viz_dir / "visualization_report.json").write_text(json.dumps(report))

    charts = dh.list_visualization_charts(str(viz_dir))
    assert charts[0]["exists"] is False


def test_list_visualization_charts_falls_back_to_glob(tmp_path):
    viz_dir = tmp_path / "viz"
    viz_dir.mkdir()
    (viz_dir / "02_correlation_heatmap.png").write_bytes(b"fake-png")

    charts = dh.list_visualization_charts(str(viz_dir))
    assert len(charts) == 1
    assert charts[0]["caption"] == "Correlation Heatmap"


def test_list_visualization_charts_empty_dir(tmp_path):
    viz_dir = tmp_path / "viz"
    viz_dir.mkdir()
    assert dh.list_visualization_charts(str(viz_dir)) == []


def test_list_visualization_charts_missing_dir(tmp_path):
    assert dh.list_visualization_charts(str(tmp_path / "nope")) == []


# ---------------------------------------------------------------------------
# format_file_size / get_file_metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_bytes,expected", [
    (0, "0 B"),
    (512, "512 B"),
    (2048, "2.0 KB"),
    (5 * 1024 * 1024, "5.0 MB"),
])
def test_format_file_size(num_bytes, expected):
    assert dh.format_file_size(num_bytes) == expected


def test_get_file_metadata_missing():
    meta = dh.get_file_metadata("/does/not/exist.pdf")
    assert meta == {"exists": False, "size_bytes": None, "size_human": None, "modified_at": None}


def test_get_file_metadata_none_path():
    meta = dh.get_file_metadata(None)
    assert meta["exists"] is False


def test_get_file_metadata_existing_file(tmp_path):
    p = tmp_path / "report.pdf"
    p.write_bytes(b"x" * 2048)
    meta = dh.get_file_metadata(str(p))
    assert meta["exists"] is True
    assert meta["size_bytes"] == 2048
    assert meta["size_human"] == "2.0 KB"
    assert meta["modified_at"] is not None


# ---------------------------------------------------------------------------
# runs_to_dataframe
# ---------------------------------------------------------------------------

def test_runs_to_dataframe_empty():
    df = dh.runs_to_dataframe([])
    assert df.empty
    assert "agent_name" in df.columns


def test_runs_to_dataframe_rounds_duration():
    rows = [{
        "id": 1, "agent_name": "EDAAgent", "status": "success",
        "duration_seconds": 1.23456789, "started_at": "2026-01-01T00:00:00",
        "finished_at": "2026-01-01T00:00:01", "input_path": "x.csv",
        "output_path": "y.json", "error_message": None,
    }]
    df = dh.runs_to_dataframe(rows)
    assert df.iloc[0]["duration_seconds"] == 1.235


# ---------------------------------------------------------------------------
# experiments_to_dataframe / _pick_primary_metric (handbook F9)
# ---------------------------------------------------------------------------

def test_pick_primary_metric_prefers_roc_auc_over_f1():
    name, value = dh._pick_primary_metric({"f1_macro": 0.5, "roc_auc": 0.8})
    assert name == "roc_auc"
    assert value == 0.8


def test_pick_primary_metric_falls_back_to_first_key_for_unknown_metrics():
    """A dataset whose report uses metric names not in the priority list
    must still get a primary metric, not a blank column."""
    name, value = dh._pick_primary_metric({"custom_score": 42.0})
    assert name == "custom_score"
    assert value == 42.0


def test_pick_primary_metric_none_for_empty_dict():
    assert dh._pick_primary_metric({}) == (None, None)


def test_experiments_to_dataframe_empty():
    df = dh.experiments_to_dataframe([])
    assert df.empty
    assert "best_model_name" in df.columns


def _sample_experiment_row(**overrides) -> dict:
    row = {
        "id": 1,
        "logged_at": "2026-07-26T22:00:00+00:00",
        "data_path": "data/features.csv",
        "target_col": "is_late_delivery",
        "task_type": "binary_classification",
        "best_model_name": "HistGradientBoostingClassifier",
        "split_strategy": "grouped",
        "group_col": "customer_unique_id",
        "random_state": 42,
        "n_features": 21,
        "best_hyperparameters": {"learning_rate": 0.1, "max_iter": 100},
        "cv_scores": {"HistGradientBoostingClassifier": 0.57},
        "cv_std": {"HistGradientBoostingClassifier": 0.005},
        "test_metrics": {"f1_macro": 0.574, "roc_auc": 0.781},
        "model_selection_note": None,
        "nested_cv_score": 0.565,
        "nested_cv_std": 0.003,
        "report_path": "data/features_ml_report.json",
    }
    row.update(overrides)
    return row


def test_experiments_to_dataframe_derives_primary_metric_column():
    df = dh.experiments_to_dataframe([_sample_experiment_row()])
    assert df.iloc[0]["primary_metric_name"] == "roc_auc"
    assert df.iloc[0]["primary_metric_value"] == pytest.approx(0.781)


def test_experiments_to_dataframe_flattens_dict_fields_to_json_strings():
    """The main table must render dict-valued fields as compact JSON
    strings, not raw Python dict reprs, for a clean table cell."""
    df = dh.experiments_to_dataframe([_sample_experiment_row()])
    assert isinstance(df.iloc[0]["test_metrics"], str)
    assert json.loads(df.iloc[0]["test_metrics"]) == {"f1_macro": 0.574, "roc_auc": 0.781}
    assert isinstance(df.iloc[0]["best_hyperparameters"], str)


def test_experiments_to_dataframe_generic_for_regression_metrics():
    """No hardcoded task type: a regression row's rmse must be picked as
    the primary metric when roc_auc/f1_macro aren't present."""
    row = _sample_experiment_row(
        task_type="regression",
        best_model_name="RandomForestRegressor",
        test_metrics={"rmse": 18500.2, "mae": 14200.5, "adjusted_r2": 0.81},
    )
    df = dh.experiments_to_dataframe([row])
    assert df.iloc[0]["primary_metric_name"] == "adjusted_r2"
    assert df.iloc[0]["primary_metric_value"] == pytest.approx(0.81)


# ---------------------------------------------------------------------------
# outputs_available / ensure_session_state
# ---------------------------------------------------------------------------

def test_outputs_available(tmp_path):
    present = tmp_path / "present.json"
    present.write_text("{}")
    result = dh.outputs_available({
        "present": str(present),
        "missing": str(tmp_path / "missing.json"),
        "none": None,
    })
    assert result == {"present": True, "missing": False, "none": False}


def test_ensure_session_state_fills_missing_keys():
    state = {}
    dh.ensure_session_state(state)
    assert state["pipeline_ran"] is False
    assert state["target_col"] is None
    assert set(dh.DEFAULT_SESSION_STATE) <= set(state)


def test_ensure_session_state_preserves_existing_values():
    state = {"pipeline_ran": True, "target_col": "is_late_delivery"}
    dh.ensure_session_state(state)
    assert state["pipeline_ran"] is True
    assert state["target_col"] == "is_late_delivery"
