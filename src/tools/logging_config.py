"""
src/tools/logging_config.py

Shared structured logging setup for the multi-agent system. Every agent
imports `get_agent_logger` rather than configuring its own logger, so log
format and destination stay consistent across the whole pipeline -- this
is what makes the "System Log Explorer" dashboard page and audit trail
possible later.
"""

import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "workspace", "metadata")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "agent_activity.log")

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_agent_logger(agent_name: str) -> logging.Logger:
    """Return a configured logger for a given agent.

    Each agent should call this once, e.g.:
        logger = get_agent_logger("DataCleaningAgent")

    Logs go to both console (for live dev feedback) and a shared log file
    (for the audit trail / System Log Explorer dashboard page).
    """
    logger = logging.getLogger(agent_name)
    if logger.handlers:
        # Already configured (avoids duplicate handlers on re-import)
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
