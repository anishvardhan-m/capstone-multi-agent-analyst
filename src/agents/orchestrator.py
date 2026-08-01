"""
src/agents/orchestrator.py

Orchestrator Agent.

Responsibilities (per the capstone handbook, Section 6.1, with the
error-recovery design informed by Section 17.1): run the full seven-agent
pipeline in sequence --

    DataCleaningAgent -> EDAAgent -> FeatureEngineeringAgent -> MLAgent ->
    VisualizationAgent -> BusinessInsightsAgent -> ReportGenerationAgent

-- passing each agent's output path forward as the next agent's input, and
produce a single auditable OrchestratorReport describing what happened.

Design note: this is the second (and last) LLM-calling agent in the
pipeline, but for a different purpose than BusinessInsightsAgent -- it
never touches the data itself. When a step's (success, result) comes back
False, the orchestrator does not just abort. It calls an LLM (the same
OpenRouter client pattern as insights.py: a primary free model, with one
fallback model on failure) with the error message and pipeline context,
and asks it to classify the failure into exactly one recovery action:

  * retry  -- worth trying again once (e.g. a transient/environmental
              failure). Capped at exactly one retry per step, regardless
              of what the LLM recommends on the retry's own failure, so a
              misbehaving LLM can never cause an infinite retry loop.
  * skip   -- this step's output isn't strictly required for the
              remaining steps to produce a degraded-but-useful result, so
              continue without it. Every downstream agent in this project
              already tolerates a missing/unreadable upstream file
              gracefully (placeholder text, skipped chart, etc.) -- that
              existing graceful-degradation contract is exactly what
              makes "skip" a safe, well-defined action here rather than a
              hand-wave.
  * abort  -- the failure is fundamental (bad input data, a missing
              required file, invalid configuration) and continuing would
              only produce garbage or repeat the same failure.

If the recovery LLM call itself fails (both models), the orchestrator
defaults to 'abort' -- a broken LLM connection must never leave the
pipeline silently continuing in an undefined state.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.agents.cleaner import DataCleaningAgent
from src.agents.eda import EDAAgent
from src.agents.feature_engineer import FeatureEngineeringAgent
from src.agents.insights import BusinessInsightsAgent
from src.agents.ml_agent import MLAgent
from src.agents.report_generator import ReportGenerationAgent
from src.agents.visualizer import VisualizationAgent
from src.tools.audit_db import audit_logged, log_agent_run
from src.tools.logging_config import get_agent_logger

logger = get_agent_logger("OrchestratorAgent")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

_PRIMARY_MODEL = "inclusionai/ling-3.0-flash:free"
_FALLBACK_MODEL = "openrouter/free"

_MODEL_OUTPUT_PATH = "models/best_production_model.pkl"

STEP_NAMES = [
    "DataCleaningAgent",
    "EDAAgent",
    "FeatureEngineeringAgent",
    "MLAgent",
    "VisualizationAgent",
    "BusinessInsightsAgent",
    "ReportGenerationAgent",
]

_RECOVERY_SYSTEM_PROMPT = (
    "You are an SRE-style pipeline supervisor for a multi-agent data-analysis "
    "system. A pipeline step just failed. Decide exactly one recovery action:\n"
    "- 'retry': worth trying again once -- e.g. a transient/environmental failure.\n"
    "- 'skip': this step's output isn't strictly required for the remaining "
    "steps to still produce a degraded-but-useful result, so continue the "
    "pipeline without it.\n"
    "- 'abort': the failure is fundamental -- e.g. bad input data, a missing "
    "required file, invalid configuration -- and continuing would only "
    "produce garbage or repeat the same failure.\n\n"
    "Respond with EXACTLY two lines, no markdown, no extra commentary:\n"
    "ACTION: retry|skip|abort\n"
    "REASON: <one sentence>"
)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Outcome of a single pipeline step, including any LLM recovery decision."""

    name: str
    status: str  # "success" | "success_after_retry" | "skipped" | "failed"
    attempts: int
    message: str
    llm_action: Optional[str] = None
    llm_reasoning: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "attempts": self.attempts,
            "message": self.message,
            "llm_action": self.llm_action,
            "llm_reasoning": self.llm_reasoning,
        }


@dataclass
class OrchestratorReport:
    """Structured, JSON-serializable summary of one full pipeline run."""

    steps: list = field(default_factory=list)  # list[StepResult]
    total_duration_seconds: float = 0.0
    model_path: Optional[str] = None
    final_report_path: Optional[str] = None
    aborted: bool = False
    abort_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_duration_seconds": self.total_duration_seconds,
            "model_path": self.model_path,
            "final_report_path": self.final_report_path,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }


# ---------------------------------------------------------------------------
# Recovery-response parsing
# ---------------------------------------------------------------------------

def _parse_recovery_response(content: str) -> tuple[Optional[str], str]:
    """Parse the LLM's 'ACTION: ...\\nREASON: ...' response.

    Returns (action, reason). action is None if the response didn't contain
    a recognized action, in which case the caller treats it as a failed
    classification and falls back to the next model / the safe default.
    """
    action = None
    reason = ""
    for line in content.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("ACTION:"):
            candidate = stripped.split(":", 1)[1].strip().lower()
            if candidate in ("retry", "skip", "abort"):
                action = candidate
        elif upper.startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()
    if not reason:
        reason = content.strip()[:300]
    return action, reason


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

class OrchestratorAgent:
    """Runs the full 7-agent pipeline end to end with LLM-driven recovery.

    Parameters
    ----------
    client : openai.OpenAI | None
        Pre-built client for the recovery-classification LLM calls.
        Tests inject a mock here so no real network call is made.
    max_retries : int
        Maximum retries per failing step, regardless of what the recovery
        LLM recommends on a retry's own failure (default 1).
    """

    def __init__(self, client: Optional[OpenAI] = None, max_retries: int = 1):
        self._client = client
        self.max_retries = max_retries
        self.report_: Optional[OrchestratorReport] = None

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL"),
        )
        return self._client

    def _get_recovery_action(
        self,
        step_name: str,
        error_message: str,
        completed_steps: list,
        remaining_steps: list,
    ) -> tuple[str, str]:
        """Classify a step failure via LLM and choose retry/skip/abort.

        Tries the primary model, then the fallback model, same pattern as
        BusinessInsightsAgent. Defaults to 'abort' if both fail or return
        an unparseable response -- a broken LLM connection must never
        leave the pipeline's next action undefined.
        """
        prompt = (
            f"Pipeline step '{step_name}' just failed.\n"
            f"Error message: {error_message}\n"
            f"Steps completed so far: {', '.join(completed_steps) or 'none'}\n"
            f"Steps remaining after this one: {', '.join(remaining_steps) or 'none'}\n\n"
            "Choose exactly one recovery action for this step."
        )

        client = self._get_client()
        for model in (_PRIMARY_MODEL, _FALLBACK_MODEL):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _RECOVERY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.choices[0].message.content or ""
                action, reason = _parse_recovery_response(content)
                if action is not None:
                    return action, reason
                logger.warning(
                    "Recovery LLM response from '%s' was unparseable: %r", model, content
                )
            except Exception as exc:
                logger.warning("Recovery LLM call to '%s' failed: %s", model, exc)

        logger.error(
            "All recovery LLM attempts failed for step '%s'; defaulting to 'abort'.",
            step_name,
        )
        return (
            "abort",
            "LLM recovery classification was unavailable (both models failed); "
            "defaulting to abort for safety.",
        )

    def _execute_step(
        self,
        step_name: str,
        fn,
        completed_steps: list,
        remaining_steps: list,
    ) -> StepResult:
        """Run one pipeline step with the retry/skip/abort recovery loop."""
        attempts = 0
        last_action: Optional[str] = None
        last_reasoning: Optional[str] = None
        while True:
            attempts += 1
            try:
                success, result = fn()
            except Exception as exc:
                success, result = False, str(exc)

            if success:
                status = "success" if attempts == 1 else "success_after_retry"
                logger.info("%s succeeded (attempt %d)", step_name, attempts)
                return StepResult(
                    name=step_name, status=status, attempts=attempts, message=result,
                    llm_action=last_action, llm_reasoning=last_reasoning,
                )

            logger.warning("%s failed (attempt %d): %s", step_name, attempts, result)

            if attempts > self.max_retries:
                reason = "Retry limit reached after an LLM-recommended retry; escalating to abort."
                logger.error("%s exhausted its retry budget; aborting pipeline.", step_name)
                return StepResult(
                    name=step_name, status="failed", attempts=attempts, message=result,
                    llm_action="abort", llm_reasoning=reason,
                )

            action, reasoning = self._get_recovery_action(
                step_name, result, completed_steps, remaining_steps
            )
            logger.info(
                "Recovery decision for '%s' -> action=%s reason=%s", step_name, action, reasoning
            )
            last_action, last_reasoning = action, reasoning

            if action == "retry":
                continue

            return StepResult(
                name=step_name,
                status="skipped" if action == "skip" else "failed",
                attempts=attempts, message=result, llm_action=action, llm_reasoning=reasoning,
            )

    def _skip_step(self, step_name: str, reason: str) -> StepResult:
        """Build a StepResult for a step skipped BEFORE it ever runs, because
        a required upstream input is missing (that upstream step was itself
        skipped or failed).

        This is deliberately NOT routed through `_execute_step` /
        `_get_recovery_action`: whether to continue here is already fully
        determined by upstream state, not a genuine failure that needs an
        LLM's judgment call, and calling the step's real run() with an
        empty/placeholder path would just reproduce the confusing bare
        file-path error this method exists to avoid (see BusinessInsightsAgent
        below -- it cannot produce a grounded narrative without the ML
        report it summarizes, so skipping it outright, with a clear reason,
        is the right call rather than degrading in place).

        Also writes a synthetic 'skipped' row to the audit_runs table:
        since the step's own run() never executes, the @audit_logged
        decorator on its run() method never fires either, so without this
        the System Log Explorer would show no trace at all of this step --
        indistinguishable from the step never having been reached.
        """
        logger.info("%s skipped: %s", step_name, reason)
        now = datetime.now(timezone.utc)
        log_agent_run(
            agent_name=step_name, started_at=now, finished_at=now,
            status="skipped", input_path=None, output_path=None,
            error_message=reason,
        )
        return StepResult(
            name=step_name, status="skipped", attempts=0, message=reason,
            llm_action=None, llm_reasoning=None,
        )

    def _finalize_abort(
        self, report: OrchestratorReport, start_time: float, step_name: str, error_message: str
    ) -> tuple[bool, str]:
        report.aborted = True
        report.abort_reason = f"Pipeline aborted at '{step_name}': {error_message}"
        report.total_duration_seconds = round(time.monotonic() - start_time, 3)
        self.report_ = report
        logger.error(report.abort_reason)
        return False, report.abort_reason

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @audit_logged("OrchestratorAgent", input_arg="data_path")
    def run(
        self,
        data_path: str,
        target_col: str,
        id_col: Optional[str] = None,
        group_col: Optional[str] = None,
        positive_label: Optional[str] = None,
        negative_label: Optional[str] = None,
        unit_label: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Run the full 7-agent pipeline end to end.

        Parameters
        ----------
        data_path : str
            Path to the raw input CSV.
        target_col : str
            Name of the column MLAgent should predict.
        id_col : str | None
            Row-identifier column, excluded from ML features.
        group_col : str | None
            A column identifying which rows belong to the same real-world
            entity (e.g. a repeat-customer ID). When given, threaded into
            FeatureEngineeringAgent as an extra protected column (so it
            survives feature engineering as a raw identifier rather than
            being frequency-encoded into a leaky count/proportion) and into
            MLAgent so the train/test split uses GroupShuffleSplit on it --
            no entity's rows end up split across both train and test. See
            MLAgent.run's docstring for the full rationale.
        positive_label, negative_label, unit_label : str | None
            Business-domain vocabulary threaded into VisualizationAgent and
            BusinessInsightsAgent (e.g. "late delivery" / "on-time
            delivery" / "order"). Defaults to each agent's own generic
            fallback when not provided.
        run_id : str | None
            Namespaces this run's model file and workspace outputs so a
            second, different dataset never silently overwrites the
            first's results (found during genericity testing: every output
            path used to be fixed regardless of dataset). Defaults to
            None, which reproduces today's exact fixed paths --
            `models/best_production_model.pkl`,
            `workspace/visualizations/`, `workspace/business_insights.md`,
            `workspace/executive_report.pdf` -- unchanged, since the
            deployed dashboard and the Olist demo depend on those exact
            paths. When given, outputs instead go to
            `models/{run_id}_best_production_model.pkl` and
            `workspace/{run_id}/` (charts, insights markdown, and the PDF
            all under that subdirectory). Intermediate `data/processed/`
            (or wherever the input CSV lives) files already namespace
            themselves by the input filename's stem via each upstream
            agent's own `<stem>_...` naming, so they need no run_id.

        Returns
        -------
        (success, final_pdf_path_or_abort_reason)
        """
        start_time = time.monotonic()
        report = OrchestratorReport()
        completed: list = []
        logger.info(
            "Starting full pipeline run on %s (target=%s, id_col=%s, group_col=%s, run_id=%s)",
            data_path, target_col, id_col, group_col, run_id,
        )

        if run_id:
            model_output_path = f"models/{run_id}_best_production_model.pkl"
            workspace_dir = f"workspace/{run_id}"
        else:
            model_output_path = _MODEL_OUTPUT_PATH
            workspace_dir = "workspace"
        viz_output_dir = os.path.join(workspace_dir, "visualizations")
        report_output_path = os.path.join(workspace_dir, "executive_report.pdf")

        # ---- 1. DataCleaningAgent ----
        cleaning_agent = DataCleaningAgent()
        result = self._execute_step(
            "DataCleaningAgent", lambda: cleaning_agent.run(data_path),
            completed, STEP_NAMES[1:],
        )
        report.steps.append(result)
        completed.append(result.name)
        if result.status == "failed":
            return self._finalize_abort(report, start_time, result.name, result.message)
        cleaned_csv_path = result.message if result.status != "skipped" else None
        cleaning_report_path = (
            os.path.splitext(cleaned_csv_path)[0] + "_report.json" if cleaned_csv_path else None
        )

        # ---- 2. EDAAgent ----
        eda_agent = EDAAgent()
        result = self._execute_step(
            "EDAAgent",
            lambda: eda_agent.run(cleaned_csv_path) if cleaned_csv_path
            else (False, "Upstream input (cleaned CSV) unavailable"),
            completed, STEP_NAMES[2:],
        )
        report.steps.append(result)
        completed.append(result.name)
        if result.status == "failed":
            return self._finalize_abort(report, start_time, result.name, result.message)
        eda_report_path = result.message if result.status != "skipped" else None

        # ---- 3. FeatureEngineeringAgent ----
        # target_col is ALWAYS protected here, not just when a group_col is
        # given: feature_tools.PROTECTED_COLS' own defaults ("is_late_delivery",
        # "order_id") are Olist-specific, so without this, any other dataset's
        # target column would be silently scaled/log-transformed/encoded like
        # any other feature -- corrupting what MLAgent trains against and
        # making every downstream metric meaningless.
        fe_agent = FeatureEngineeringAgent(
            extra_protected_cols=[c for c in (target_col, group_col) if c]
        )
        result = self._execute_step(
            "FeatureEngineeringAgent",
            lambda: fe_agent.run(cleaned_csv_path) if cleaned_csv_path
            else (False, "Upstream input (cleaned CSV) unavailable"),
            completed, STEP_NAMES[3:],
        )
        report.steps.append(result)
        completed.append(result.name)
        if result.status == "failed":
            return self._finalize_abort(report, start_time, result.name, result.message)
        features_csv_path = result.message if result.status != "skipped" else None

        # ---- 4. MLAgent ----
        ml_agent = MLAgent()
        result = self._execute_step(
            "MLAgent",
            lambda: ml_agent.run(
                features_csv_path, target_col=target_col, id_col=id_col, group_col=group_col,
                model_output_path=model_output_path,
            )
            if features_csv_path else (False, "Upstream input (feature-engineered CSV) unavailable"),
            completed, STEP_NAMES[4:],
        )
        report.steps.append(result)
        completed.append(result.name)
        if result.status == "failed":
            return self._finalize_abort(report, start_time, result.name, result.message)
        ml_report_path = result.message if result.status != "skipped" else None
        if ml_report_path:
            report.model_path = model_output_path

        # ---- 5. VisualizationAgent ----
        viz_agent = VisualizationAgent(
            positive_label=positive_label, negative_label=negative_label, unit_label=unit_label,
        ) if positive_label or negative_label or unit_label else VisualizationAgent()
        result = self._execute_step(
            "VisualizationAgent",
            lambda: viz_agent.run(
                eda_report_path=eda_report_path or "",
                ml_report_path=ml_report_path or "",
                cleaned_data_path=cleaned_csv_path or "",
                target_col=target_col,
                output_dir=viz_output_dir,
            ),
            completed, STEP_NAMES[5:],
        )
        report.steps.append(result)
        completed.append(result.name)
        if result.status == "failed":
            return self._finalize_abort(report, start_time, result.name, result.message)
        chart_paths = (
            [c["path"] for c in viz_agent.report_.charts] if viz_agent.report_ else []
        )

        # ---- 6. BusinessInsightsAgent ----
        # Unlike VisualizationAgent (which can still produce partial charts
        # from whichever of eda/ml/cleaned-data it does have) and
        # ReportGenerationAgent (which renders placeholder sections for
        # whatever's missing), BusinessInsightsAgent's whole prompt branches
        # on the ML report's task_type -- it has no meaningful "EDA-only"
        # narrative to fall back to. So when either upstream report is
        # missing, skip it outright with a clear reason instead of calling
        # it with an empty path (which used to surface as a bare
        # "[Errno 2] No such file or directory: ''").
        missing_upstream = [
            name for name, path in (("EDAAgent", eda_report_path), ("MLAgent", ml_report_path))
            if not path
        ]
        if missing_upstream:
            report_labels = "/".join(n.replace("Agent", "") for n in missing_upstream)
            if len(missing_upstream) == 1:
                reason = (
                    f"Skipped: upstream {missing_upstream[0]} was skipped, "
                    f"no {report_labels} report available"
                )
            else:
                reason = (
                    f"Skipped: upstream {' and '.join(missing_upstream)} were "
                    f"skipped, no {report_labels} reports available"
                )
            result = self._skip_step("BusinessInsightsAgent", reason)
        else:
            insights_agent = BusinessInsightsAgent(
                positive_label=positive_label, negative_label=negative_label, unit_label=unit_label,
            )
            result = self._execute_step(
                "BusinessInsightsAgent",
                lambda: insights_agent.run(
                    eda_report_path=eda_report_path,
                    ml_report_path=ml_report_path,
                    output_dir=workspace_dir,
                ),
                completed, STEP_NAMES[6:],
            )
        report.steps.append(result)
        completed.append(result.name)
        if result.status == "failed":
            return self._finalize_abort(report, start_time, result.name, result.message)
        insights_md_path = result.message if result.status != "skipped" else None

        # ---- 7. ReportGenerationAgent ----
        report_agent = ReportGenerationAgent()
        result = self._execute_step(
            "ReportGenerationAgent",
            lambda: report_agent.run(
                cleaning_report_path=cleaning_report_path or "",
                ml_report_path=ml_report_path or "",
                insights_md_path=insights_md_path or "",
                chart_paths=chart_paths,
                output_path=report_output_path,
            ),
            completed, [],
        )
        report.steps.append(result)
        completed.append(result.name)
        if result.status == "failed":
            return self._finalize_abort(report, start_time, result.name, result.message)
        final_pdf_path = result.message if result.status != "skipped" else None

        report.final_report_path = final_pdf_path
        report.total_duration_seconds = round(time.monotonic() - start_time, 3)
        self.report_ = report

        logger.info(
            "Pipeline complete in %.2fs. Final report: %s",
            report.total_duration_seconds, final_pdf_path,
        )
        return True, final_pdf_path or "Pipeline completed with degraded output (final PDF unavailable)"


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Run the full 7-agent pipeline end to end on any "
        "classification/regression CSV.",
    )
    parser.add_argument("data_path", help="Path to the raw input CSV.")
    parser.add_argument("target_col", help="Column MLAgent should predict.")
    parser.add_argument(
        "--id-col", default=None,
        help="Row-identifier column, excluded from ML features.",
    )
    parser.add_argument(
        "--group-col", default=None,
        help="Column identifying rows belonging to the same real-world "
        "entity, for leakage-safe grouped train/test splitting.",
    )
    parser.add_argument("--positive-label", default=None)
    parser.add_argument("--negative-label", default=None)
    parser.add_argument("--unit-label", default=None)
    parser.add_argument(
        "--run-id", default=None,
        help="Namespace this run's model file and workspace outputs "
        "(models/{run-id}_best_production_model.pkl, workspace/{run-id}/) "
        "so a second dataset doesn't overwrite a prior run's results. "
        "Omit to use the fixed default paths (unchanged behavior).",
    )
    args = parser.parse_args()

    agent = OrchestratorAgent()
    success, result = agent.run(
        data_path=args.data_path,
        target_col=args.target_col,
        id_col=args.id_col,
        group_col=args.group_col,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        unit_label=args.unit_label,
        run_id=args.run_id,
    )
    print(json.dumps(agent.report_.to_dict(), indent=2))
    if success:
        print(f"\nSuccess. Final report: {result}")
    else:
        print(f"\nFailed: {result}")
        sys.exit(1)
