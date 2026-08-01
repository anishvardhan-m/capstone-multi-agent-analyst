"""
src/agents/cleaner.py

Data Cleaning Agent.

Responsibilities (per the capstone handbook, Section 6.2): inspect data
layouts, detect corruption points, execute missing-data imputation based
on statistical properties, remove redundant duplicates, and enforce
uniform types across feature vectors.

Design note: this agent's cleaning LOGIC is fully deterministic (a
Scikit-Learn Pipeline of the transformers in src/tools/data_tools.py). No
LLM call happens inside this agent. This is intentional: cleaning rules
should be reproducible and auditable, not subject to LLM non-determinism.
The "agent" framing still holds -- it inspects the data, makes decisions
(e.g. which columns to flag as high-missing or low-variance), and
produces a structured report of what it did, exactly like a human data
cleaning specialist would.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from src.tools.audit_db import audit_logged
from src.tools.data_tools import (
    DataTypeCorrector,
    DuplicateRemover,
    LowVarianceColumnDropper,
    MissingValueImputer,
    PlaceholderNullNormalizer,
)
from src.tools.logging_config import get_agent_logger

logger = get_agent_logger("DataCleaningAgent")


@dataclass
class CleaningReport:
    """Structured, JSON-serializable summary of what the cleaning agent did.

    This is the 'Structural Logging Delta Report' the handbook calls for
    (Section 6.2) -- it's what the dashboard's Data Diagnostics panel and
    the final PDF report's section 2 will be built from.
    """

    input_shape: tuple
    output_shape: tuple
    n_duplicates_removed: int
    high_missing_columns_flagged: list = field(default_factory=list)
    low_variance_columns_dropped: list = field(default_factory=list)
    columns_type_corrected: list = field(default_factory=list)
    numeric_columns_imputed: list = field(default_factory=list)
    categorical_columns_imputed: list = field(default_factory=list)
    placeholder_values_normalized: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "n_duplicates_removed": self.n_duplicates_removed,
            "high_missing_columns_flagged": self.high_missing_columns_flagged,
            "low_variance_columns_dropped": self.low_variance_columns_dropped,
            "columns_type_corrected": self.columns_type_corrected,
            "numeric_columns_imputed": self.numeric_columns_imputed,
            "categorical_columns_imputed": self.categorical_columns_imputed,
            "placeholder_values_normalized": self.placeholder_values_normalized,
        }


class DataCleaningAgent:
    """Orchestrates the cleaning pipeline and produces an auditable report.

    Parameters
    ----------
    high_missing_threshold : float
        Columns with a missing fraction above this are flagged (not
        auto-dropped -- see `drop_high_missing_columns`).
    low_variance_threshold : float
        Columns where one value accounts for at least this fraction of
        rows are dropped as near-constant / low information.
    drop_high_missing_columns : bool
        If True, columns flagged as high-missing are dropped after being
        reported. If False, they are only flagged/logged and still
        imputed like any other column -- useful when a "missing" flag is
        itself informative and you'd rather keep + impute than lose the
        column entirely.
    """

    def __init__(
        self,
        high_missing_threshold: float = 0.5,
        low_variance_threshold: float = 0.99,
        drop_high_missing_columns: bool = False,
        placeholder_values: Optional[list[str]] = None,
    ):
        self.high_missing_threshold = high_missing_threshold
        self.low_variance_threshold = low_variance_threshold
        self.drop_high_missing_columns = drop_high_missing_columns
        self.placeholder_values = placeholder_values
        self.report_: Optional[CleaningReport] = None
        self._pipeline: Optional[Pipeline] = None

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("duplicate_remover", DuplicateRemover()),
                ("placeholder_normalizer", PlaceholderNullNormalizer(
                    placeholder_values=self.placeholder_values
                )),
                ("type_corrector", DataTypeCorrector()),
                ("missing_imputer", MissingValueImputer(
                    high_missing_threshold=self.high_missing_threshold
                )),
                ("low_variance_dropper", LowVarianceColumnDropper(
                    threshold=self.low_variance_threshold
                )),
            ]
        )

    @audit_logged("DataCleaningAgent")
    def run(self, data_path: str, output_path: Optional[str] = None) -> tuple[bool, str]:
        """Run the full cleaning pipeline on a CSV file.

        Parameters
        ----------
        data_path : str
            Path to the raw input CSV.
        output_path : str | None
            Where to write the cleaned CSV. Defaults to
            `<data_path>_cleaned.csv` alongside the input.

        Returns
        -------
        (success, output_path_or_error_message)
        """
        logger.info("Starting cleaning run on %s", data_path)
        try:
            df = pd.read_csv(data_path)
        except Exception as exc:
            logger.error("Failed to read input file: %s", exc)
            return False, f"Failed to read input file: {exc}"

        input_shape = df.shape

        self._pipeline = self._build_pipeline()
        try:
            cleaned_df = self._pipeline.fit_transform(df)
        except Exception as exc:
            logger.error("Cleaning pipeline failed: %s", exc)
            return False, f"Cleaning pipeline failed: {exc}"

        dup_remover = self._pipeline.named_steps["duplicate_remover"]
        placeholder_normalizer = self._pipeline.named_steps["placeholder_normalizer"]
        type_corrector = self._pipeline.named_steps["type_corrector"]
        imputer = self._pipeline.named_steps["missing_imputer"]
        variance_dropper = self._pipeline.named_steps["low_variance_dropper"]

        self.report_ = CleaningReport(
            input_shape=input_shape,
            output_shape=cleaned_df.shape,
            n_duplicates_removed=dup_remover.n_duplicates_removed_,
            high_missing_columns_flagged=imputer.high_missing_columns_,
            low_variance_columns_dropped=variance_dropper.dropped_columns_,
            columns_type_corrected=type_corrector.converted_columns_,
            numeric_columns_imputed=list(imputer.numeric_fill_values_.keys()),
            categorical_columns_imputed=list(imputer.categorical_fill_values_.keys()),
            placeholder_values_normalized=placeholder_normalizer.placeholder_counts_,
        )

        logger.info("Cleaning complete: %s", self.report_.to_dict())

        if output_path is None:
            base, ext = os.path.splitext(data_path)
            output_path = f"{base}_cleaned{ext}"

        cleaned_df.to_csv(output_path, index=False)
        logger.info("Cleaned data written to %s", output_path)

        # Persist the report alongside for auditability
        report_path = os.path.splitext(output_path)[0] + "_report.json"
        with open(report_path, "w") as f:
            json.dump(self.report_.to_dict(), f, indent=2)
        logger.info("Cleaning report written to %s", report_path)

        return True, output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.cleaner <path_to_csv>")
        sys.exit(1)

    agent = DataCleaningAgent()
    success, result = agent.run(sys.argv[1])
    if success:
        print(f"Success. Cleaned data at: {result}")
        print(json.dumps(agent.report_.to_dict(), indent=2))
    else:
        print(f"Failed: {result}")
        sys.exit(1)
