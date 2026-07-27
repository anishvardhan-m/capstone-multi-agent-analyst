"""
tests/conftest.py

Shared pytest fixtures for the whole test suite.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.agents import ml_agent
from src.tools import audit_db


@pytest.fixture(autouse=True)
def isolate_audit_db(tmp_path, monkeypatch):
    """Redirect every agent's audit logging to a per-test scratch database.

    Agents call @audit_logged(...) with no explicit db_path, so they fall
    back to audit_db.DEFAULT_DB_PATH -- the real workspace/metadata/
    audit_telemetry.db. Without this fixture, every test that runs an
    agent's .run() (even against tmp_path fixture data) would write a row
    into that production database. audit_db resolves DEFAULT_DB_PATH fresh
    on every call (see audit_db.py), so patching the module attribute here
    reliably redirects all of it for the duration of each test.

    tests/test_audit_db.py's own tests pass an explicit db_path already
    and are unaffected either way.
    """
    test_db_path = str(tmp_path / "test_audit_telemetry.db")
    monkeypatch.setattr(audit_db, "DEFAULT_DB_PATH", test_db_path)
    yield test_db_path


@pytest.fixture(autouse=True)
def isolate_model_output_dir(tmp_path, monkeypatch):
    """Redirect MLAgent's serialized model to a per-test scratch directory.

    MLAgent._prepare_and_train always writes to
    os.path.join(_MODEL_DIR, "best_production_model.pkl") -- a fixed,
    absolute path under the real repo's models/ directory -- with no way
    for a caller to override it. Without this fixture, every test that
    calls agent.run() (even against tmp_path fixture data, and even
    MLAgent(...).run_robustness_check's seeds that pass save_model=False
    only for their own loop -- a plain run() call still writes for real)
    would overwrite the real production model, exactly the isolation gap
    already solved for the audit DB above. ml_agent.py reads _MODEL_DIR as
    a plain module-level name inside the function body, so it's resolved
    fresh on every call and this monkeypatch reliably redirects it for the
    duration of each test.
    """
    test_model_dir = str(tmp_path / "test_models")
    monkeypatch.setattr(ml_agent, "_MODEL_DIR", test_model_dir)
    yield test_model_dir
