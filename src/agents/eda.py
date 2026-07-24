"""
src/agents/eda.py

Exploratory Data Analysis (EDA) Agent.

Responsibilities (per the capstone handbook, Section 6.3): compute
descriptive statistics, correlation structure, distribution skewness, and
per-column outlier prevalence using the IQR fence method.

Design note: like the DataCleaningAgent, this agent is fully deterministic
-- pure pandas/numpy, no LLM calls. Reproducibility and auditability are
the priority; the structured EDAReport it emits feeds the dashboard's EDA
panel and the final PDF report's section 3.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.tools.logging_config import get_agent_logger

logger = get_agent_logger("EDAAgent")


@dataclass
class EDAReport:
    """Structured, JSON-serializable summary of exploratory analysis.

    All nested dicts use plain Python types (float, int, list) so the
    report can be serialised with json.dump without a custom encoder.
    """

    input_shape: tuple
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)

    # .describe() per numeric column: {"col": {"mean": x, "std": x, ...}}
    descriptive_stats: dict = field(default_factory=dict)

    # Pearson correlation matrix as a nested dict: {"col_a": {"col_b": r, ...}}
    correlation_matrix: dict = field(default_factory=dict)

    # Skewness per numeric column: {"col": skew_value}
    skewness: dict = field(default_factory=dict)

    # Outlier summary per numeric column:
    # {"col": {"n_outliers": int, "pct_outliers": float,
    #           "lower_fence": float, "upper_fence": float}}
    outlier_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "input_shape": list(self.input_shape),
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "descriptive_stats": self.descriptive_stats,
            "correlation_matrix": self.correlation_matrix,
            "skewness": self.skewness,
            "outlier_summary": self.outlier_summary,
        }


def _safe_float(value) -> Optional[float]:
    """Convert numpy scalar to Python float; return None for NaN/inf."""
    try:
        v = float(value)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return None


class EDAAgent:
    """Runs a suite of deterministic EDA computations and writes a report.

    Parameters
    ----------
    iqr_multiplier : float
        Multiplier applied to the IQR to define the outlier fences.
        Standard Tukey fence is 1.5; 3.0 is used for 'extreme' outliers.
    """

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self.report_: Optional[EDAReport] = None

    # ------------------------------------------------------------------
    # Internal computation helpers
    # ------------------------------------------------------------------

    def _descriptive_stats(self, df: pd.DataFrame, numeric_cols: list) -> dict:
        """Return .describe() for each numeric column as a plain dict."""
        if not numeric_cols:
            return {}
        stats = df[numeric_cols].describe()
        result: dict = {}
        for col in numeric_cols:
            col_stats: dict = {}
            for stat_name, value in stats[col].items():
                col_stats[stat_name] = _safe_float(value)
            result[col] = col_stats
        return result

    def _correlation_matrix(self, df: pd.DataFrame, numeric_cols: list) -> dict:
        """Return Pearson correlation as a nested plain-Python dict."""
        if len(numeric_cols) < 2:
            return {}
        corr = df[numeric_cols].corr(method="pearson")
        result: dict = {}
        for col in corr.columns:
            result[col] = {
                other: _safe_float(corr.at[col, other])
                for other in corr.columns
            }
        return result

    def _skewness(self, df: pd.DataFrame, numeric_cols: list) -> dict:
        """Return Fisher skewness per numeric column."""
        if not numeric_cols:
            return {}
        return {
            col: _safe_float(df[col].skew())
            for col in numeric_cols
        }

    def _outlier_summary(self, df: pd.DataFrame, numeric_cols: list) -> dict:
        """IQR-based outlier detection per numeric column.

        A value is an outlier when it falls below Q1 - k*IQR or above
        Q3 + k*IQR, where k is self.iqr_multiplier.  NaNs are excluded
        from both the fence computation and the outlier count.
        """
        if not numeric_cols:
            return {}
        result: dict = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                result[col] = {
                    "n_outliers": 0,
                    "pct_outliers": 0.0,
                    "lower_fence": None,
                    "upper_fence": None,
                }
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            n_outliers = int(((series < lower) | (series > upper)).sum())
            result[col] = {
                "n_outliers": n_outliers,
                "pct_outliers": round(n_outliers / len(series) * 100, 4),
                "lower_fence": _safe_float(lower),
                "upper_fence": _safe_float(upper),
            }
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, data_path: str) -> tuple[bool, str]:
        """Run the full EDA suite on a CSV file.

        Parameters
        ----------
        data_path : str
            Path to the input CSV (typically the cleaned output from
            DataCleaningAgent).

        Returns
        -------
        (success, report_path_or_error_message)
            On success, the returned string is the path of the written
            JSON report: ``<data_path stem>_eda_report.json``.
        """
        logger.info("Starting EDA run on %s", data_path)

        try:
            df = pd.read_csv(data_path)
        except Exception as exc:
            logger.error("Failed to read input file: %s", exc)
            return False, f"Failed to read input file: {exc}"

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        logger.info(
            "Shape %s | %d numeric cols | %d categorical cols",
            df.shape, len(numeric_cols), len(categorical_cols),
        )

        desc_stats = self._descriptive_stats(df, numeric_cols)
        logger.info("Descriptive stats computed for %d columns", len(desc_stats))

        corr = self._correlation_matrix(df, numeric_cols)
        logger.info("Correlation matrix computed (%dx%d)", len(corr), len(corr))

        skew = self._skewness(df, numeric_cols)
        logger.info("Skewness computed for %d columns", len(skew))

        outliers = self._outlier_summary(df, numeric_cols)
        total_outlier_cols = sum(
            1 for v in outliers.values() if v["n_outliers"] > 0
        )
        logger.info(
            "Outlier summary computed; %d columns have at least one outlier",
            total_outlier_cols,
        )

        self.report_ = EDAReport(
            input_shape=df.shape,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            descriptive_stats=desc_stats,
            correlation_matrix=corr,
            skewness=skew,
            outlier_summary=outliers,
        )

        base, _ = os.path.splitext(data_path)
        report_path = f"{base}_eda_report.json"
        with open(report_path, "w") as f:
            json.dump(self.report_.to_dict(), f, indent=2)
        logger.info("EDA report written to %s", report_path)

        return True, report_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.eda <path_to_csv>")
        sys.exit(1)

    agent = EDAAgent()
    success, result = agent.run(sys.argv[1])
    if success:
        print(f"Success. EDA report at: {result}")
        print(json.dumps(agent.report_.to_dict(), indent=2))
    else:
        print(f"Failed: {result}")
        sys.exit(1)
