"""
tests/conftest.py

Shared pytest fixtures for the whole test suite.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

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
