"""
tests/test_orchestrator.py

Unit tests for the Orchestrator Agent (src/agents/orchestrator.py).

Every one of the 7 pipeline agents is mocked at the class level (patched
in src.agents.orchestrator's namespace, where OrchestratorAgent imports
them), and the recovery LLM client is injected via the constructor -- no
real agent logic or network call runs in this test suite.

Run with:
    pytest tests/test_orchestrator.py -v
"""

import contextlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.orchestrator import STEP_NAMES, OrchestratorAgent


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _mock_llm_response(text: str) -> MagicMock:
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _mock_recovery_client(action: str, reason: str = "test reason") -> MagicMock:
    """A client whose recovery LLM call always returns the given action."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_llm_response(
        f"ACTION: {action}\nREASON: {reason}\n"
    )
    return client


@pytest.fixture
def patched_agents():
    """Patch all 7 pipeline agent classes to succeed deterministically.

    Yields a dict of the mocked *instances* (agent_cls.return_value) keyed
    by short name, so individual tests can override .run.side_effect /
    .report_ to simulate a failure.
    """
    with contextlib.ExitStack() as stack:
        cleaning_cls = stack.enter_context(patch("src.agents.orchestrator.DataCleaningAgent"))
        eda_cls = stack.enter_context(patch("src.agents.orchestrator.EDAAgent"))
        fe_cls = stack.enter_context(patch("src.agents.orchestrator.FeatureEngineeringAgent"))
        ml_cls = stack.enter_context(patch("src.agents.orchestrator.MLAgent"))
        viz_cls = stack.enter_context(patch("src.agents.orchestrator.VisualizationAgent"))
        insights_cls = stack.enter_context(patch("src.agents.orchestrator.BusinessInsightsAgent"))
        report_cls = stack.enter_context(patch("src.agents.orchestrator.ReportGenerationAgent"))

        cleaning = cleaning_cls.return_value
        cleaning.run.return_value = (True, "data/processed/x_cleaned.csv")

        eda = eda_cls.return_value
        eda.run.return_value = (True, "data/processed/x_eda_report.json")

        fe = fe_cls.return_value
        fe.run.return_value = (True, "data/processed/x_features.csv")

        ml = ml_cls.return_value
        ml.run.return_value = (True, "data/processed/x_ml_report.json")

        viz = viz_cls.return_value
        viz.run.return_value = (True, "workspace/visualizations/visualization_report.json")
        viz.report_ = MagicMock()
        viz.report_.charts = [{"path": "chart1.png"}, {"path": "chart2.png"}]

        insights = insights_cls.return_value
        insights.run.return_value = (True, "workspace/business_insights.md")

        report = report_cls.return_value
        report.run.return_value = (True, "workspace/executive_report.pdf")

        yield {
            "cleaning": cleaning, "eda": eda, "fe": fe, "ml": ml,
            "viz": viz, "insights": insights, "report": report,
        }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_all_seven_steps_succeed(patched_agents):
    client = MagicMock()  # recovery LLM must never be called
    agent = OrchestratorAgent(client=client)

    success, result = agent.run(data_path="raw.csv", target_col="target")

    assert success is True
    assert result == "workspace/executive_report.pdf"
    client.chat.completions.create.assert_not_called()

    report = agent.report_
    assert report.aborted is False
    assert [s.name for s in report.steps] == STEP_NAMES
    assert all(s.status == "success" and s.attempts == 1 for s in report.steps)
    assert report.model_path == "models/best_production_model.pkl"
    assert report.final_report_path == "workspace/executive_report.pdf"
    assert report.total_duration_seconds >= 0

    # Each agent's run() must have been called exactly once, in order.
    for inst in patched_agents.values():
        assert inst.run.call_count == 1

    # ReportGenerationAgent must receive the chart paths VisualizationAgent produced.
    _, report_kwargs = patched_agents["report"].run.call_args
    assert report_kwargs["chart_paths"] == ["chart1.png", "chart2.png"]


def test_happy_path_threads_labels_into_viz_and_insights(patched_agents):
    with patch("src.agents.orchestrator.VisualizationAgent") as viz_cls, \
         patch("src.agents.orchestrator.BusinessInsightsAgent") as insights_cls:
        viz_cls.return_value.run.return_value = (True, "viz_report.json")
        viz_cls.return_value.report_ = MagicMock(charts=[])
        insights_cls.return_value.run.return_value = (True, "insights.md")

        agent = OrchestratorAgent(client=MagicMock())
        agent.run(
            data_path="raw.csv", target_col="target",
            positive_label="late delivery", negative_label="on-time delivery", unit_label="order",
        )

        assert viz_cls.call_args.kwargs["positive_label"] == "late delivery"
        assert viz_cls.call_args.kwargs["negative_label"] == "on-time delivery"
        assert viz_cls.call_args.kwargs["unit_label"] == "order"
        assert insights_cls.call_args.kwargs["positive_label"] == "late delivery"
        assert insights_cls.call_args.kwargs["unit_label"] == "order"


# ---------------------------------------------------------------------------
# Failure + successful retry
# ---------------------------------------------------------------------------

def test_failure_then_successful_retry(patched_agents):
    patched_agents["ml"].run.side_effect = [
        (False, "transient GridSearchCV worker crash"),
        (True, "data/processed/x_ml_report.json"),
    ]
    client = _mock_recovery_client("retry", "Looks like a transient worker crash, worth retrying.")

    agent = OrchestratorAgent(client=client)
    success, result = agent.run(data_path="raw.csv", target_col="target")

    assert success is True
    assert patched_agents["ml"].run.call_count == 2
    client.chat.completions.create.assert_called_once()

    ml_step = next(s for s in agent.report_.steps if s.name == "MLAgent")
    assert ml_step.status == "success_after_retry"
    assert ml_step.attempts == 2
    assert ml_step.llm_action == "retry"

    # Pipeline must have continued through to the end.
    assert [s.name for s in agent.report_.steps] == STEP_NAMES
    assert agent.report_.aborted is False


# ---------------------------------------------------------------------------
# Failure + graceful skip
# ---------------------------------------------------------------------------

def test_failure_then_graceful_skip(patched_agents):
    patched_agents["viz"].run.return_value = (False, "matplotlib backend crashed")
    patched_agents["viz"].report_ = None  # real VisualizationAgent leaves this None on failure
    client = _mock_recovery_client("skip", "Charts are supplementary; continue without them.")

    agent = OrchestratorAgent(client=client)
    success, result = agent.run(data_path="raw.csv", target_col="target")

    assert success is True
    assert result == "workspace/executive_report.pdf"

    viz_step = next(s for s in agent.report_.steps if s.name == "VisualizationAgent")
    assert viz_step.status == "skipped"
    assert viz_step.llm_action == "skip"

    # Pipeline must still run the remaining 2 steps.
    assert [s.name for s in agent.report_.steps] == STEP_NAMES
    assert patched_agents["insights"].run.call_count == 1
    assert patched_agents["report"].run.call_count == 1

    # No charts were produced, so ReportGenerationAgent must get an empty list,
    # not crash on a missing/None report_.
    _, report_kwargs = patched_agents["report"].run.call_args
    assert report_kwargs["chart_paths"] == []
    assert agent.report_.aborted is False


# ---------------------------------------------------------------------------
# Failure + abort
# ---------------------------------------------------------------------------

def test_failure_then_abort(patched_agents):
    patched_agents["eda"].run.return_value = (False, "disk full while writing report")
    client = _mock_recovery_client("abort", "Disk-full is a fundamental environment failure.")

    agent = OrchestratorAgent(client=client)
    success, result = agent.run(data_path="raw.csv", target_col="target")

    assert success is False
    assert "EDAAgent" in result
    assert "disk full" in result

    assert agent.report_.aborted is True
    assert "EDAAgent" in agent.report_.abort_reason

    # Steps after the failure must never have run.
    assert [s.name for s in agent.report_.steps] == ["DataCleaningAgent", "EDAAgent"]
    assert patched_agents["fe"].run.call_count == 0
    assert patched_agents["ml"].run.call_count == 0
    assert patched_agents["viz"].run.call_count == 0
    assert patched_agents["insights"].run.call_count == 0
    assert patched_agents["report"].run.call_count == 0


# ---------------------------------------------------------------------------
# Retry cap is respected
# ---------------------------------------------------------------------------

def test_retry_cap_respected_even_if_llm_keeps_recommending_retry(patched_agents):
    patched_agents["fe"].run.return_value = (False, "persistent corruption error")
    client = _mock_recovery_client("retry", "Worth another shot.")  # always says retry

    agent = OrchestratorAgent(client=client, max_retries=1)
    success, result = agent.run(data_path="raw.csv", target_col="target")

    # Exactly 1 initial attempt + 1 retry = 2 calls, never a 3rd, no matter
    # how many times the LLM would say "retry".
    assert patched_agents["fe"].run.call_count == 2
    # The LLM is asked once (after the 1st failure) -- after the retry also
    # fails, the cap is enforced without asking a second time.
    assert client.chat.completions.create.call_count == 1

    assert success is False
    fe_step = next(s for s in agent.report_.steps if s.name == "FeatureEngineeringAgent")
    assert fe_step.status == "failed"
    assert fe_step.attempts == 2
    assert agent.report_.aborted is True

    # Downstream steps never ran.
    assert patched_agents["ml"].run.call_count == 0


def test_custom_max_retries_allows_more_than_one_retry(patched_agents):
    patched_agents["ml"].run.side_effect = [
        (False, "fail 1"), (False, "fail 2"), (True, "ml_report.json"),
    ]
    client = _mock_recovery_client("retry")

    agent = OrchestratorAgent(client=client, max_retries=2)
    success, result = agent.run(data_path="raw.csv", target_col="target")

    assert success is True
    assert patched_agents["ml"].run.call_count == 3
    ml_step = next(s for s in agent.report_.steps if s.name == "MLAgent")
    assert ml_step.status == "success_after_retry"
    assert ml_step.attempts == 3
