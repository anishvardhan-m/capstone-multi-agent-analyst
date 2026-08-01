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
import glob
import hashlib
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
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

    # Default (no run_id) must reproduce today's exact fixed paths -- the
    # deployed dashboard and the Olist demo depend on these unchanged.
    _, ml_kwargs = patched_agents["ml"].run.call_args
    assert ml_kwargs["model_output_path"] == "models/best_production_model.pkl"
    _, viz_kwargs = patched_agents["viz"].run.call_args
    assert viz_kwargs["output_dir"] == "workspace/visualizations"
    _, insights_kwargs = patched_agents["insights"].run.call_args
    assert insights_kwargs["output_dir"] == "workspace"
    assert report_kwargs["output_path"] == "workspace/executive_report.pdf"


# ---------------------------------------------------------------------------
# run_id namespacing (genericity fix: two datasets must not clobber each other)
# ---------------------------------------------------------------------------

def test_run_id_namespaces_all_output_paths(patched_agents):
    """A run_id must redirect the model file and every workspace output
    into a namespaced location, leaving the fixed defaults (used when
    run_id is omitted) completely untouched by this code path."""
    agent = OrchestratorAgent(client=MagicMock())
    success, result = agent.run(
        data_path="raw.csv", target_col="target", run_id="my_dataset",
    )

    assert success is True

    _, ml_kwargs = patched_agents["ml"].run.call_args
    assert ml_kwargs["model_output_path"] == "models/my_dataset_best_production_model.pkl"

    _, viz_kwargs = patched_agents["viz"].run.call_args
    assert viz_kwargs["output_dir"] == "workspace/my_dataset/visualizations"

    _, insights_kwargs = patched_agents["insights"].run.call_args
    assert insights_kwargs["output_dir"] == "workspace/my_dataset"

    _, report_kwargs = patched_agents["report"].run.call_args
    assert report_kwargs["output_path"] == "workspace/my_dataset/executive_report.pdf"

    assert agent.report_.model_path == "models/my_dataset_best_production_model.pkl"


def test_run_id_none_is_indistinguishable_from_omitted(patched_agents):
    """Passing run_id=None explicitly must behave exactly like omitting it."""
    agent = OrchestratorAgent(client=MagicMock())
    agent.run(data_path="raw.csv", target_col="target", run_id=None)

    _, ml_kwargs = patched_agents["ml"].run.call_args
    assert ml_kwargs["model_output_path"] == "models/best_production_model.pkl"
    _, viz_kwargs = patched_agents["viz"].run.call_args
    assert viz_kwargs["output_dir"] == "workspace/visualizations"


def test_happy_path_threads_group_col_into_fe_and_ml(patched_agents):
    """F1: group_col must reach FeatureEngineeringAgent (as an extra
    protected column, so it survives feature engineering untouched) and
    MLAgent (to drive the grouped train/test split). target_col must
    ALWAYS be protected too, group_col or not (see the genericity fix
    below) -- feature_tools.PROTECTED_COLS' own defaults are Olist-
    specific ("is_late_delivery"), so without this, any other dataset's
    target column would be silently transformed like any other feature."""
    with patch("src.agents.orchestrator.FeatureEngineeringAgent") as fe_cls, \
         patch("src.agents.orchestrator.MLAgent") as ml_cls:
        fe_cls.return_value.run.return_value = (True, "data/processed/x_features.csv")
        ml_cls.return_value.run.return_value = (True, "data/processed/x_ml_report.json")

        agent = OrchestratorAgent(client=MagicMock())
        agent.run(
            data_path="raw.csv", target_col="target", group_col="customer_unique_id",
        )

        assert set(fe_cls.call_args.kwargs["extra_protected_cols"]) == {"target", "customer_unique_id"}
        _, ml_kwargs = ml_cls.return_value.run.call_args
        assert ml_kwargs["group_col"] == "customer_unique_id"


def test_happy_path_no_group_col_still_protects_target_col(patched_agents):
    with patch("src.agents.orchestrator.FeatureEngineeringAgent") as fe_cls, \
         patch("src.agents.orchestrator.MLAgent") as ml_cls:
        fe_cls.return_value.run.return_value = (True, "data/processed/x_features.csv")
        ml_cls.return_value.run.return_value = (True, "data/processed/x_ml_report.json")

        agent = OrchestratorAgent(client=MagicMock())
        agent.run(data_path="raw.csv", target_col="target")

        assert fe_cls.call_args.kwargs["extra_protected_cols"] == ["target"]
        _, ml_kwargs = ml_cls.return_value.run.call_args
        assert ml_kwargs["group_col"] is None


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
# MLAgent skipped (e.g. a genuine data problem like too-rare target classes)
# must cascade into a clean, clearly-worded skip of BusinessInsightsAgent --
# never the bare "[Errno 2] No such file or directory: ''" that used to
# surface when an empty ml_report_path got forwarded straight into
# BusinessInsightsAgent.run().
# ---------------------------------------------------------------------------

def test_ml_agent_skipped_cascades_to_clean_insights_skip(patched_agents):
    patched_agents["ml"].run.return_value = (
        False, "Target column 'target' has a class with only 2 samples -- too few for stratified CV"
    )
    client = _mock_recovery_client("skip", "Not enough data per class to train reliably; continue without a model.")

    agent = OrchestratorAgent(client=client)
    success, result = agent.run(data_path="raw.csv", target_col="target")

    # The pipeline must still complete in degraded form, not abort.
    assert success is True
    assert agent.report_.aborted is False
    assert [s.name for s in agent.report_.steps] == STEP_NAMES

    ml_step = next(s for s in agent.report_.steps if s.name == "MLAgent")
    assert ml_step.status == "skipped"

    insights_step = next(s for s in agent.report_.steps if s.name == "BusinessInsightsAgent")
    assert insights_step.status == "skipped"
    # BusinessInsightsAgent must never actually be invoked with a missing
    # path -- confirms the skip happens BEFORE any call, not via a crash
    # inside agent.run() that then gets classified as a "failure".
    assert patched_agents["insights"].run.call_count == 0
    assert insights_step.attempts == 0
    # The message must clearly name the upstream cause -- never a bare
    # file-path/errno error.
    assert "MLAgent" in insights_step.message
    assert "ML report" in insights_step.message
    assert "Errno" not in insights_step.message
    assert "No such file or directory" not in insights_step.message

    # VisualizationAgent (already tolerant of a missing ML report) and
    # ReportGenerationAgent (renders placeholder sections for what's
    # missing) must both still run rather than being skipped.
    viz_step = next(s for s in agent.report_.steps if s.name == "VisualizationAgent")
    assert viz_step.status == "success"
    report_step = next(s for s in agent.report_.steps if s.name == "ReportGenerationAgent")
    assert report_step.status == "success"
    assert patched_agents["report"].run.call_count == 1

    # ReportGenerationAgent must receive the real eda path but an empty
    # ml_report_path/insights_md_path -- it already degrades these to
    # placeholders internally rather than needing the orchestrator to skip
    # it too.
    _, report_kwargs = patched_agents["report"].run.call_args
    assert report_kwargs["ml_report_path"] == ""
    assert report_kwargs["insights_md_path"] == ""


def test_eda_agent_skipped_cascades_to_clean_insights_skip(patched_agents):
    """Same cascade, triggered from the other required upstream input."""
    patched_agents["eda"].run.return_value = (False, "EDA computation blew up on a malformed column")
    client = _mock_recovery_client("skip", "EDA is supplementary here; continue without it.")

    agent = OrchestratorAgent(client=client)
    success, result = agent.run(data_path="raw.csv", target_col="target")

    assert success is True
    assert agent.report_.aborted is False

    insights_step = next(s for s in agent.report_.steps if s.name == "BusinessInsightsAgent")
    assert insights_step.status == "skipped"
    assert patched_agents["insights"].run.call_count == 0
    assert "EDAAgent" in insights_step.message
    assert "EDA report" in insights_step.message


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


# ---------------------------------------------------------------------------
# Real end-to-end integration test: two datasets, two run_ids, no clobbering
# ---------------------------------------------------------------------------
# Unlike every test above, nothing here is mocked except the LLM network
# calls (BusinessInsightsAgent's OpenAI client, and the Orchestrator's own
# recovery client, which a happy path never even calls) -- the real
# DataCleaningAgent, EDAAgent, FeatureEngineeringAgent, MLAgent,
# VisualizationAgent, and ReportGenerationAgent all run for real, against
# two small but genuinely different synthetic regression datasets, to
# prove run_id namespacing actually holds up end to end, not just at the
# call-kwargs level the tests above check.

def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _make_small_regression_csv(path: str, seed: int, n: int = 160) -> None:
    """A tiny, genuinely learnable regression dataset -- distinct column
    names/count per seed so the two datasets used below aren't just the
    same schema with different random values."""
    rng = np.random.default_rng(seed)
    if seed == 1:
        size_sqft = rng.normal(1800, 400, n).clip(400, 4000)
        rooms = rng.integers(1, 6, n)
        age_years = rng.integers(0, 80, n)
        target = size_sqft * 120 + rooms * 3000 - age_years * 200 + rng.normal(0, 5000, n)
        df = pd.DataFrame({
            "size_sqft": size_sqft, "rooms": rooms, "age_years": age_years,
            "sale_price": target,
        })
    else:
        engine_size = rng.normal(2.5, 0.8, n).clip(1.0, 6.0)
        mileage = rng.normal(60000, 30000, n).clip(0, 200000)
        horsepower = rng.normal(200, 60, n).clip(90, 500)
        target = horsepower * 150 - mileage * 0.05 + engine_size * 800 + rng.normal(0, 3000, n)
        df = pd.DataFrame({
            "engine_size": engine_size, "mileage": mileage, "horsepower": horsepower,
            "car_value": target,
        })
    df.to_csv(path, index=False)


@pytest.fixture
def mocked_llm_client():
    """Patch the OpenAI class BusinessInsightsAgent constructs internally
    (src.agents.insights._get_client), so its real prompt-construction and
    markdown-writing logic runs, but no network call happens."""
    with patch("src.agents.insights.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            "## What We Found\nSome findings.\n\n"
            "## What Matters Most\nSome drivers.\n\n"
            "## Recommendations\nSome recommendations.\n"
        )
        mock_openai_cls.return_value = mock_client
        yield mock_client


def test_two_datasets_with_different_run_ids_do_not_clobber_each_other(
    tmp_path, monkeypatch, mocked_llm_client,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")

    os.makedirs("data/raw", exist_ok=True)
    _make_small_regression_csv("data/raw/houses.csv", seed=1)
    _make_small_regression_csv("data/raw/cars.csv", seed=2)

    agent = OrchestratorAgent(client=MagicMock())

    success_a, result_a = agent.run(
        data_path="data/raw/houses.csv", target_col="sale_price", run_id="houses",
    )
    assert success_a is True, result_a
    assert agent.report_.aborted is False

    model_a_path = "models/houses_best_production_model.pkl"
    assert os.path.isfile(model_a_path)
    model_a_hash_after_run_a = _md5(model_a_path)

    success_b, result_b = agent.run(
        data_path="data/raw/cars.csv", target_col="car_value", run_id="cars",
    )
    assert success_b is True, result_b
    assert agent.report_.aborted is False

    # Both models exist, are namespaced, and neither run touched the other's.
    model_b_path = "models/cars_best_production_model.pkl"
    assert os.path.isfile(model_a_path)
    assert os.path.isfile(model_b_path)
    model_a_hash_after_run_b = _md5(model_a_path)
    assert model_a_hash_after_run_b == model_a_hash_after_run_a

    # Both workspaces exist independently, each with its own charts, insights, PDF.
    assert os.path.isdir("workspace/houses")
    assert os.path.isdir("workspace/cars")
    assert os.path.isfile("workspace/houses/business_insights.md")
    assert os.path.isfile("workspace/cars/business_insights.md")
    assert os.path.isfile("workspace/houses/executive_report.pdf")
    assert os.path.isfile("workspace/cars/executive_report.pdf")

    houses_charts = sorted(glob.glob("workspace/houses/visualizations/*.png"))
    cars_charts = sorted(glob.glob("workspace/cars/visualizations/*.png"))
    assert len(houses_charts) > 0
    assert len(cars_charts) > 0

    # The two ml_report.json files reflect genuinely different data (proof
    # this isn't one run's output masquerading as two, e.g. via a bug that
    # accidentally pointed both run_ids at the same underlying path).
    import json
    with open("data/raw/houses_cleaned_features_ml_report.json") as f:
        houses_ml_report = json.load(f)
    with open("data/raw/cars_cleaned_features_ml_report.json") as f:
        cars_ml_report = json.load(f)
    assert set(houses_ml_report["feature_importances"].keys()) == {"size_sqft", "rooms", "age_years"}
    assert set(cars_ml_report["feature_importances"].keys()) == {"engine_size", "mileage", "horsepower"}

    # The fixed default paths (what run_id=None uses) were never touched --
    # confirming this run_id-based code path is fully additive.
    assert not os.path.exists("models/best_production_model.pkl")
    assert not os.path.exists("workspace/business_insights.md")
    assert not os.path.exists("workspace/executive_report.pdf")
    assert not os.path.exists("workspace/visualizations")
