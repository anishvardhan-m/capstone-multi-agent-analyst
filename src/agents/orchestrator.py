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
from src.tools.audit_db import audit_logged
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
        positive_label: Optional[str] = None,
        negative_label: Optional[str] = None,
        unit_label: Optional[str] = None,
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
        positive_label, negative_label, unit_label : str | None
            Business-domain vocabulary threaded into VisualizationAgent and
            BusinessInsightsAgent (e.g. "late delivery" / "on-time
            delivery" / "order"). Defaults to each agent's own generic
            fallback when not provided.

        Returns
        -------
        (success, final_pdf_path_or_abort_reason)
        """
        start_time = time.monotonic()
        report = OrchestratorReport()
        completed: list = []
        logger.info(
            "Starting full pipeline run on %s (target=%s, id_col=%s)",
            data_path, target_col, id_col,
        )

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
        fe_agent = FeatureEngineeringAgent()
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
            lambda: ml_agent.run(features_csv_path, target_col=target_col, id_col=id_col)
            if features_csv_path else (False, "Upstream input (feature-engineered CSV) unavailable"),
            completed, STEP_NAMES[4:],
        )
        report.steps.append(result)
        completed.append(result.name)
        if result.status == "failed":
            return self._finalize_abort(report, start_time, result.name, result.message)
        ml_report_path = result.message if result.status != "skipped" else None
        if ml_report_path:
            report.model_path = _MODEL_OUTPUT_PATH

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
                output_dir="workspace/visualizations",
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
        insights_agent = BusinessInsightsAgent(
            positive_label=positive_label, negative_label=negative_label, unit_label=unit_label,
        )
        result = self._execute_step(
            "BusinessInsightsAgent",
            lambda: insights_agent.run(
                eda_report_path=eda_report_path or "",
                ml_report_path=ml_report_path or "",
                output_dir="workspace",
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
                output_path="workspace/executive_report.pdf",
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
    import json
    import sys

    agent = OrchestratorAgent()
    success, result = agent.run(
        data_path="data/processed/olist_flattened.csv",
        target_col="is_late_delivery",
        positive_label="late delivery",
        negative_label="on-time delivery",
        unit_label="order",
    )
    print(json.dumps(agent.report_.to_dict(), indent=2))
    if success:
        print(f"\nSuccess. Final report: {result}")
    else:
        print(f"\nFailed: {result}")
        sys.exit(1)
