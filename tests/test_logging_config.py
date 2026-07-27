"""
tests/test_logging_config.py

Unit tests for src/tools/logging_config.py.

get_agent_logger caches by name via Python's global logging registry
(logging.getLogger(name) returns the same object process-wide), so every
test here uses a unique logger name never used by any real agent --
reusing a real agent's name would pick up handlers a prior test/import
already attached, pointed at the real workspace/metadata/agent_activity.log.
Handlers created in each test are removed afterward so they don't leak a
FileHandler pointed at a tmp_path pytest has since cleaned up.
"""

import logging

from src.tools import logging_config


def _cleanup(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_get_agent_logger_attaches_console_and_file_handlers(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_config, "LOG_FILE", str(tmp_path / "agent_activity.log"))
    logger = logging_config.get_agent_logger("_TestLogger_Handlers")
    try:
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == str(tmp_path / "agent_activity.log")
    finally:
        _cleanup(logger)


def test_get_agent_logger_writes_to_file_with_expected_format(tmp_path, monkeypatch):
    log_path = tmp_path / "agent_activity.log"
    monkeypatch.setattr(logging_config, "LOG_FILE", str(log_path))
    logger = logging_config.get_agent_logger("_TestLogger_Format")
    try:
        logger.info("hello world")
        for handler in logger.handlers:
            handler.flush()
        content = log_path.read_text()
        assert "_TestLogger_Format" in content
        assert "INFO" in content
        assert "hello world" in content
    finally:
        _cleanup(logger)


def test_get_agent_logger_does_not_duplicate_handlers_on_repeat_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_config, "LOG_FILE", str(tmp_path / "agent_activity.log"))
    logger_first = logging_config.get_agent_logger("_TestLogger_NoDupes")
    try:
        n_handlers_first_call = len(logger_first.handlers)
        logger_second = logging_config.get_agent_logger("_TestLogger_NoDupes")
        assert logger_second is logger_first
        assert len(logger_second.handlers) == n_handlers_first_call
    finally:
        _cleanup(logger_first)
