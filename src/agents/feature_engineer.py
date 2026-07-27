"""
src/agents/feature_engineer.py

Feature Engineering Agent.

Responsibilities (per the capstone handbook, Section 6.4): reduce
redundancy between correlated features, normalise skewed distributions,
encode categorical variables, and scale numeric features to zero-mean
unit-variance — producing a model-ready feature matrix alongside an
auditable JSON report of every transformation applied.

Design note: like the DataCleaningAgent and EDAAgent, this agent is fully
deterministic — a Scikit-Learn Pipeline of the transformers in
src/tools/feature_tools.py. No LLM calls happen here. The pipeline order
is:

  1. RedundantFeatureDropper  — prune highly correlated columns first so
                                 subsequent steps don't waste effort on
                                 features that will never be used.
  2. SkewnessReducer          — log1p-transform before scaling so the
                                 scaler learns from the already-normalised
                                 distribution.
  3. NumericScaler            — scale original numeric columns before OHE
                                 so binary dummy columns (which are already
                                 0/1) are never accidentally standardised.
  4. CategoricalEncoder       — OHE / frequency-encode last; new dummy
                                 columns inherit the 0/1 range without
                                 needing further scaling.

The two protected columns — "is_late_delivery" (target) and "order_id"
(identifier) — pass through every step completely unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from src.tools.audit_db import audit_logged
from src.tools.feature_tools import (
    PROTECTED_COLS,
    CategoricalEncoder,
    NumericScaler,
    RedundantFeatureDropper,
    SkewnessReducer,
)
from src.tools.logging_config import get_agent_logger

logger = get_agent_logger("FeatureEngineeringAgent")


@dataclass
class FeatureEngineeringReport:
    """Structured, JSON-serializable summary of all feature engineering steps.

    This is what the dashboard's Feature Engineering panel and the final
    PDF report's section 4 will be built from.
    """

    input_shape: tuple
    output_shape: tuple
    protected_columns: list = field(default_factory=list)
    redundant_columns_dropped: list = field(default_factory=list)
    log_transformed_columns: list = field(default_factory=list)
    scaled_columns: list = field(default_factory=list)
    zero_variance_columns_skipped: list = field(default_factory=list)
    encoding_map: dict = field(default_factory=dict)
    scaler_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "protected_columns": self.protected_columns,
            "redundant_columns_dropped": self.redundant_columns_dropped,
            "log_transformed_columns": self.log_transformed_columns,
            "scaled_columns": self.scaled_columns,
            "zero_variance_columns_skipped": self.zero_variance_columns_skipped,
            "encoding_map": self.encoding_map,
            "scaler_stats": self.scaler_stats,
        }


class FeatureEngineeringAgent:
    """Orchestrates the feature engineering pipeline and produces an auditable report.

    Parameters
    ----------
    corr_threshold : float
        Absolute Pearson correlation above which the second column of a
        pair is dropped as redundant (default 0.95).
    skew_threshold : float
        |skewness| above which a non-negative numeric column is log1p-
        transformed (default 1.0).
    ohe_threshold : int
        Categorical columns with fewer unique values than this receive
        one-hot encoding; higher-cardinality columns get frequency encoding
        (default 20).
    extra_protected_cols : list[str] | None
        Additional columns (beyond feature_tools.PROTECTED_COLS' generic
        "is_late_delivery"/"order_id" defaults) to pass through every
        transformer completely unmodified. Used for a caller-supplied
        grouping column (e.g. a repeat-customer identifier the ML Agent
        needs later to group-split train/test) -- without this, a
        high-cardinality ID column would get frequency-encoded into a
        count/proportion, destroying the true identity values grouping
        needs and quietly turning "how many total orders will this
        customer place" (a future-information leak) into a model feature.
    """

    def __init__(
        self,
        corr_threshold: float = 0.95,
        skew_threshold: float = 1.0,
        ohe_threshold: int = 20,
        extra_protected_cols: Optional[list[str]] = None,
    ):
        self.corr_threshold = corr_threshold
        self.skew_threshold = skew_threshold
        self.ohe_threshold = ohe_threshold
        self.extra_protected_cols = extra_protected_cols
        self.report_: Optional[FeatureEngineeringReport] = None
        self._pipeline: Optional[Pipeline] = None

    def _protected_cols(self) -> list[str]:
        return sorted(PROTECTED_COLS | set(self.extra_protected_cols or []))

    def _build_pipeline(self) -> Pipeline:
        protected = self._protected_cols()
        return Pipeline(
            steps=[
                ("redundant_dropper", RedundantFeatureDropper(
                    threshold=self.corr_threshold, protected_cols=protected,
                )),
                ("skewness_reducer", SkewnessReducer(
                    skew_threshold=self.skew_threshold, protected_cols=protected,
                )),
                ("numeric_scaler", NumericScaler(protected_cols=protected)),
                ("categorical_encoder", CategoricalEncoder(
                    ohe_threshold=self.ohe_threshold, protected_cols=protected,
                )),
            ]
        )

    @audit_logged("FeatureEngineeringAgent")
    def run(self, data_path: str, output_path: Optional[str] = None) -> tuple[bool, str]:
        """Run the full feature engineering pipeline on a CSV file.

        Parameters
        ----------
        data_path : str
            Path to the input CSV (typically the cleaned output from
            DataCleaningAgent).
        output_path : str | None
            Where to write the feature-engineered CSV. Defaults to
            ``<data_path stem>_features.csv`` alongside the input.

        Returns
        -------
        (success, output_path_or_error_message)
        """
        logger.info("Starting feature engineering run on %s", data_path)

        try:
            df = pd.read_csv(data_path)
        except Exception as exc:
            logger.error("Failed to read input file: %s", exc)
            return False, f"Failed to read input file: {exc}"

        input_shape = df.shape
        logger.info("Input shape: %s", input_shape)

        self._pipeline = self._build_pipeline()
        try:
            engineered_df = self._pipeline.fit_transform(df)
        except Exception as exc:
            logger.error("Feature engineering pipeline failed: %s", exc)
            return False, f"Feature engineering pipeline failed: {exc}"

        dropper = self._pipeline.named_steps["redundant_dropper"]
        reducer = self._pipeline.named_steps["skewness_reducer"]
        scaler = self._pipeline.named_steps["numeric_scaler"]
        encoder = self._pipeline.named_steps["categorical_encoder"]

        self.report_ = FeatureEngineeringReport(
            input_shape=input_shape,
            output_shape=engineered_df.shape,
            protected_columns=self._protected_cols(),
            redundant_columns_dropped=dropper.dropped_columns_,
            log_transformed_columns=reducer.log_transformed_columns_,
            scaled_columns=scaler.scaled_columns_,
            zero_variance_columns_skipped=scaler.zero_variance_columns_,
            encoding_map=encoder.encoding_map_,
            scaler_stats=scaler.scale_stats_,
        )

        logger.info(
            "Feature engineering complete: %d columns dropped, %d log-transformed, "
            "%d scaled, %d categorical cols encoded",
            len(dropper.dropped_columns_),
            len(reducer.log_transformed_columns_),
            len(scaler.scaled_columns_),
            len(encoder.encoding_map_),
        )

        if output_path is None:
            base, ext = os.path.splitext(data_path)
            output_path = f"{base}_features{ext}"

        engineered_df.to_csv(output_path, index=False)
        logger.info("Feature-engineered data written to %s", output_path)

        report_path = os.path.splitext(output_path)[0] + "_features_report.json"
        with open(report_path, "w") as f:
            json.dump(self.report_.to_dict(), f, indent=2)
        logger.info("Feature engineering report written to %s", report_path)

        return True, output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.feature_engineer <path_to_csv>")
        sys.exit(1)

    agent = FeatureEngineeringAgent()
    success, result = agent.run(sys.argv[1])
    if success:
        print(f"Success. Engineered data at: {result}")
        print(json.dumps(agent.report_.to_dict(), indent=2))
    else:
        print(f"Failed: {result}")
        sys.exit(1)
