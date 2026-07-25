"""
src/tools/audit_db.py

Persistent SQLite audit trail for agent runs (capstone handbook, Sections
4.1, 14.2). Every agent's .run() call gets one row in `agent_runs` --
success or failure -- so the dashboard's System Log Explorer page can show
a queryable history of what ran, when, and with what result, complementing
the human-readable text log from src/tools/logging_config.py.

Design note: no LLM calls happen here. This is a small, deterministic
persistence layer built on Python's built-in sqlite3 module (no new
dependency). Agents wire it in via the `audit_logged` decorator below
rather than duplicating try/except/timing boilerplate in every run()
method.
"""

from __future__ import annotations

import functools
import inspect
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Union

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "workspace", "metadata", "audit_telemetry.db")


def init_db(db_path: Optional[str] = None) -> None:
    """Create the SQLite audit database and `agent_runs` table if missing.

    Idempotent -- safe to call on every process start or before every
    write; CREATE TABLE IF NOT EXISTS means repeated calls never error or
    duplicate the schema.

    db_path defaults to the module-level DEFAULT_DB_PATH, resolved fresh on
    every call (rather than baked into the signature) so tests can
    monkeypatch `src.tools.audit_db.DEFAULT_DB_PATH` and have every caller
    that omits db_path -- including agents decorated with @audit_logged at
    import time -- pick up the override.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                status TEXT NOT NULL,
                input_path TEXT,
                output_path TEXT,
                error_message TEXT,
                duration_seconds REAL NOT NULL
            )
            """
        )
        conn.commit()


def log_agent_run(
    agent_name: str,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    input_path: Optional[str],
    output_path: Optional[str] = None,
    error_message: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Insert one row into `agent_runs`.

    Calls init_db() first, so this is safe to use standalone even if the
    database/table doesn't exist yet. See init_db for why db_path defaults
    to None rather than DEFAULT_DB_PATH directly.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    init_db(db_path)
    duration_seconds = (finished_at - started_at).total_seconds()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO agent_runs
                (agent_name, started_at, finished_at, status,
                 input_path, output_path, error_message, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_name,
                started_at.isoformat(),
                finished_at.isoformat(),
                status,
                input_path,
                output_path,
                error_message,
                duration_seconds,
            ),
        )
        conn.commit()


def get_recent_runs(limit: int = 50, db_path: Optional[str] = None) -> list[dict]:
    """Return the most recent `limit` rows from `agent_runs`, newest first.

    Ties in started_at (sub-second-identical runs) break on id descending,
    so ordering is always well-defined for the System Log Explorer page.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, agent_name, started_at, finished_at, status,
                   input_path, output_path, error_message, duration_seconds
            FROM agent_runs
            ORDER BY started_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# audit_logged decorator
# ---------------------------------------------------------------------------

def audit_logged(
    agent_name: str,
    input_arg: Union[str, tuple[str, ...]] = "data_path",
    db_path: Optional[str] = None,
) -> Callable:
    """Decorate an agent's run() method to log one audit_runs row per call.

    Every existing agent's run() already follows the (success, str) return
    contract, catching its own expected failure modes internally. This
    decorator adds audit logging around that without changing behavior:
    the return value (or a raised exception, for the unexpected case) passes
    through unchanged -- it only observes timing and outcome.

    Parameters
    ----------
    agent_name : str
        Human-readable name recorded in the agent_runs table.
    input_arg : str | tuple[str, ...]
        Name(s) of the decorated method's parameter(s) that identify its
        input path(s). A tuple is joined with "; " for agents that read
        multiple input files (e.g. VisualizationAgent reads three report
        paths, not one data_path).
    db_path : str | None
        Passed straight through to log_agent_run. Left as None (rather
        than defaulting to DEFAULT_DB_PATH here) so that agent modules --
        which apply this decorator once, at import time -- don't freeze in
        a stale path; every actual run() call resolves the live
        DEFAULT_DB_PATH at call time via log_agent_run, which is what lets
        tests monkeypatch it per-test.
    """
    names = (input_arg,) if isinstance(input_arg, str) else tuple(input_arg)

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            resolved = [
                str(bound.arguments[name])
                for name in names
                if bound.arguments.get(name) is not None
            ]
            input_path = "; ".join(resolved) if resolved else None

            started_at = datetime.now(timezone.utc)
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                finished_at = datetime.now(timezone.utc)
                log_agent_run(
                    agent_name=agent_name,
                    started_at=started_at,
                    finished_at=finished_at,
                    status="failure",
                    input_path=input_path,
                    output_path=None,
                    error_message=str(exc),
                    db_path=db_path,
                )
                raise

            finished_at = datetime.now(timezone.utc)
            success, message = result
            log_agent_run(
                agent_name=agent_name,
                started_at=started_at,
                finished_at=finished_at,
                status="success" if success else "failure",
                input_path=input_path,
                output_path=message if success else None,
                error_message=None if success else message,
                db_path=db_path,
            )
            return result

        return wrapper

    return decorator
