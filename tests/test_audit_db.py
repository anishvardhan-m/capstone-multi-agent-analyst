"""
tests/test_audit_db.py

Unit tests for the SQLite audit trail (src/tools/audit_db.py).

Run with:
    pytest tests/test_audit_db.py -v
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.audit_db import (
    audit_logged,
    get_recent_experiments,
    get_recent_runs,
    init_db,
    log_agent_run,
    log_ml_experiment,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "audit_telemetry.db")


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

def test_init_db_creates_table_with_expected_columns(db_path):
    init_db(db_path)
    assert os.path.exists(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")}

    expected = {
        "id", "agent_name", "started_at", "finished_at", "status",
        "input_path", "output_path", "error_message", "duration_seconds",
    }
    assert cols == expected


def test_init_db_is_idempotent(db_path):
    # Calling repeatedly must not error or wipe existing data.
    init_db(db_path)
    init_db(db_path)
    init_db(db_path)

    started = datetime.now(timezone.utc)
    finished = started + timedelta(seconds=1)
    log_agent_run(
        agent_name="TestAgent", started_at=started, finished_at=finished,
        status="success", input_path="in.csv", output_path="out.csv",
        db_path=db_path,
    )

    init_db(db_path)  # must not clear the row just written
    rows = get_recent_runs(db_path=db_path)
    assert len(rows) == 1


def test_init_db_creates_parent_directory(tmp_path):
    nested_path = str(tmp_path / "nested" / "dir" / "audit.db")
    init_db(nested_path)
    assert os.path.exists(nested_path)


# ---------------------------------------------------------------------------
# log_agent_run / get_recent_runs
# ---------------------------------------------------------------------------

def test_log_agent_run_inserts_and_get_recent_runs_retrieves(db_path):
    started = datetime(2026, 7, 25, 9, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 7, 25, 9, 0, 5, tzinfo=timezone.utc)
    log_agent_run(
        agent_name="DataCleaningAgent",
        started_at=started,
        finished_at=finished,
        status="success",
        input_path="data/raw.csv",
        output_path="data/raw_cleaned.csv",
        db_path=db_path,
    )

    rows = get_recent_runs(db_path=db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["agent_name"] == "DataCleaningAgent"
    assert row["status"] == "success"
    assert row["input_path"] == "data/raw.csv"
    assert row["output_path"] == "data/raw_cleaned.csv"
    assert row["error_message"] is None
    assert row["duration_seconds"] == pytest.approx(5.0)


def test_log_agent_run_records_failure_with_error_message(db_path):
    started = datetime.now(timezone.utc)
    finished = started + timedelta(seconds=0.2)
    log_agent_run(
        agent_name="EDAAgent",
        started_at=started,
        finished_at=finished,
        status="failure",
        input_path="missing.csv",
        output_path=None,
        error_message="Failed to read input file: [Errno 2] No such file",
        db_path=db_path,
    )

    rows = get_recent_runs(db_path=db_path)
    assert rows[0]["status"] == "failure"
    assert rows[0]["output_path"] is None
    assert "No such file" in rows[0]["error_message"]


def test_get_recent_runs_orders_most_recent_first(db_path):
    base = datetime(2026, 7, 25, 9, 0, 0, tzinfo=timezone.utc)
    for i, name in enumerate(["First", "Second", "Third"]):
        started = base + timedelta(seconds=i * 10)
        finished = started + timedelta(seconds=1)
        log_agent_run(
            agent_name=name, started_at=started, finished_at=finished,
            status="success", input_path="x.csv", output_path="y.csv",
            db_path=db_path,
        )

    rows = get_recent_runs(db_path=db_path)
    assert [r["agent_name"] for r in rows] == ["Third", "Second", "First"]


def test_get_recent_runs_respects_limit(db_path):
    base = datetime(2026, 7, 25, 9, 0, 0, tzinfo=timezone.utc)
    for i in range(10):
        started = base + timedelta(seconds=i)
        finished = started + timedelta(seconds=1)
        log_agent_run(
            agent_name=f"Agent{i}", started_at=started, finished_at=finished,
            status="success", input_path="x.csv", db_path=db_path,
        )

    rows = get_recent_runs(limit=3, db_path=db_path)
    assert len(rows) == 3
    assert [r["agent_name"] for r in rows] == ["Agent9", "Agent8", "Agent7"]


# ---------------------------------------------------------------------------
# audit_logged decorator
# ---------------------------------------------------------------------------

class _FakeAgent:
    def run(self, data_path: str, output_path=None) -> tuple:
        return True, output_path or f"{data_path}.out"

    def run_failing(self, data_path: str) -> tuple:
        return False, f"could not process {data_path}"

    def run_raising(self, data_path: str) -> tuple:
        raise RuntimeError("boom")

    def run_multi_input(self, path_a: str, path_b: str) -> tuple:
        return True, "combined_output.json"


def test_audit_logged_logs_success(db_path):
    agent = _FakeAgent()
    agent.run = audit_logged("FakeAgent", db_path=db_path)(_FakeAgent.run).__get__(agent)
    success, result = agent.run("input.csv")
    assert success is True

    rows = get_recent_runs(db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["agent_name"] == "FakeAgent"
    assert rows[0]["status"] == "success"
    assert rows[0]["input_path"] == "input.csv"
    assert rows[0]["output_path"] == result


def test_audit_logged_logs_failure_without_raising(db_path):
    agent = _FakeAgent()
    agent.run_failing = audit_logged("FakeAgent", db_path=db_path)(
        _FakeAgent.run_failing
    ).__get__(agent)
    success, message = agent.run_failing("bad.csv")
    assert success is False

    rows = get_recent_runs(db_path=db_path)
    assert rows[0]["status"] == "failure"
    assert rows[0]["output_path"] is None
    assert rows[0]["error_message"] == message


def test_audit_logged_logs_and_reraises_unexpected_exception(db_path):
    agent = _FakeAgent()
    agent.run_raising = audit_logged("FakeAgent", db_path=db_path)(
        _FakeAgent.run_raising
    ).__get__(agent)

    with pytest.raises(RuntimeError, match="boom"):
        agent.run_raising("input.csv")

    rows = get_recent_runs(db_path=db_path)
    assert rows[0]["status"] == "failure"
    assert rows[0]["error_message"] == "boom"


def test_audit_logged_supports_multiple_input_args(db_path):
    agent = _FakeAgent()
    agent.run_multi_input = audit_logged(
        "FakeAgent", input_arg=("path_a", "path_b"), db_path=db_path
    )(_FakeAgent.run_multi_input).__get__(agent)
    agent.run_multi_input("a.json", "b.json")

    rows = get_recent_runs(db_path=db_path)
    assert rows[0]["input_path"] == "a.json; b.json"


def test_audit_logged_does_not_alter_return_value(db_path):
    agent = _FakeAgent()
    agent.run = audit_logged("FakeAgent", db_path=db_path)(_FakeAgent.run).__get__(agent)
    result = agent.run("input.csv", output_path="explicit_out.csv")
    assert result == (True, "explicit_out.csv")


# ---------------------------------------------------------------------------
# ml_experiments (handbook F9)
# ---------------------------------------------------------------------------

def test_init_db_creates_ml_experiments_table_with_expected_columns(db_path):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(ml_experiments)")}

    expected = {
        "id", "logged_at", "data_path", "target_col", "task_type", "best_model_name",
        "split_strategy", "group_col", "random_state", "n_features",
        "best_hyperparameters", "cv_scores", "cv_std", "test_metrics",
        "model_selection_note", "nested_cv_score", "nested_cv_std", "report_path",
    }
    assert cols == expected


def _log_sample_experiment(db_path, **overrides):
    kwargs = dict(
        data_path="data/features.csv",
        target_col="is_late_delivery",
        task_type="binary_classification",
        best_model_name="HistGradientBoostingClassifier",
        best_hyperparameters={"learning_rate": 0.1, "max_iter": 100},
        cv_scores={"HistGradientBoostingClassifier": 0.57},
        cv_std={"HistGradientBoostingClassifier": 0.005},
        test_metrics={"f1_macro": 0.57, "roc_auc": 0.78},
        split_strategy="grouped",
        group_col="customer_unique_id",
        random_state=42,
        n_features=21,
        model_selection_note=None,
        nested_cv_score=0.565,
        nested_cv_std=0.003,
        report_path="data/features_ml_report.json",
        db_path=db_path,
    )
    kwargs.update(overrides)
    log_ml_experiment(**kwargs)


def test_log_ml_experiment_inserts_and_get_recent_experiments_retrieves(db_path):
    _log_sample_experiment(db_path)

    rows = get_recent_experiments(db_path=db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["data_path"] == "data/features.csv"
    assert row["target_col"] == "is_late_delivery"
    assert row["task_type"] == "binary_classification"
    assert row["best_model_name"] == "HistGradientBoostingClassifier"
    assert row["split_strategy"] == "grouped"
    assert row["group_col"] == "customer_unique_id"
    assert row["random_state"] == 42
    assert row["n_features"] == 21
    assert row["nested_cv_score"] == pytest.approx(0.565)
    assert row["report_path"] == "data/features_ml_report.json"


def test_log_ml_experiment_round_trips_dict_fields_as_json(db_path):
    """best_hyperparameters/cv_scores/cv_std/test_metrics must decode back
    into real dicts, not stay as JSON strings, and must survive arbitrary
    keys -- nothing here is hardcoded to a specific metric name or model."""
    _log_sample_experiment(
        db_path,
        best_hyperparameters={"max_depth": 10, "n_estimators": 200},
        cv_scores={"ModelA": 0.5, "ModelB": 0.6},
        cv_std={"ModelA": 0.01, "ModelB": 0.02},
        test_metrics={"rmse": 123.4, "mae": 98.7, "adjusted_r2": 0.81},
    )

    row = get_recent_experiments(db_path=db_path)[0]
    assert row["best_hyperparameters"] == {"max_depth": 10, "n_estimators": 200}
    assert row["cv_scores"] == {"ModelA": 0.5, "ModelB": 0.6}
    assert row["cv_std"] == {"ModelA": 0.01, "ModelB": 0.02}
    assert row["test_metrics"] == {"rmse": 123.4, "mae": 98.7, "adjusted_r2": 0.81}


def test_get_recent_experiments_orders_most_recent_first(db_path):
    # logged_at is set internally to datetime.now() at insert time, so
    # ordering relies on the id-descending tiebreak (same as
    # get_recent_runs) rather than on forcing distinct timestamps here.
    for name in ["First", "Second", "Third"]:
        _log_sample_experiment(db_path, best_model_name=name)

    rows = get_recent_experiments(db_path=db_path)
    assert [r["best_model_name"] for r in rows] == ["Third", "Second", "First"]


def test_get_recent_experiments_respects_limit(db_path):
    for i in range(10):
        _log_sample_experiment(db_path, best_model_name=f"Model{i}")

    rows = get_recent_experiments(limit=3, db_path=db_path)
    assert len(rows) == 3


def test_get_recent_experiments_empty_when_nothing_logged(db_path):
    assert get_recent_experiments(db_path=db_path) == []


def test_log_ml_experiment_handles_missing_optional_fields(db_path):
    """group_col/nested_cv/model_selection_note are all legitimately None
    for many runs (row_random split, untuned winner, no close call)."""
    _log_sample_experiment(
        db_path,
        split_strategy="row_random",
        group_col=None,
        model_selection_note=None,
        nested_cv_score=None,
        nested_cv_std=None,
    )

    row = get_recent_experiments(db_path=db_path)[0]
    assert row["group_col"] is None
    assert row["nested_cv_score"] is None
