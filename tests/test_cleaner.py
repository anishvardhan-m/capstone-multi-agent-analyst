"""
tests/test_cleaner.py

Unit tests for the cleaning transformers (src/tools/data_tools.py) and
the Data Cleaning Agent (src/agents/cleaner.py).

Run with:
    pytest tests/test_cleaner.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.cleaner import DataCleaningAgent
from src.tools.data_tools import (
    DataTypeCorrector,
    DuplicateRemover,
    LowVarianceColumnDropper,
    MissingValueImputer,
)


# ---------------------------------------------------------------------------
# DuplicateRemover
# ---------------------------------------------------------------------------

def test_duplicate_remover_removes_exact_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    remover = DuplicateRemover()
    result = remover.fit_transform(df)
    assert len(result) == 2
    assert remover.n_duplicates_removed_ == 1


def test_duplicate_remover_handles_no_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3]})
    remover = DuplicateRemover()
    result = remover.fit_transform(df)
    assert len(result) == 3
    assert remover.n_duplicates_removed_ == 0


def test_duplicate_remover_handles_empty_dataframe():
    df = pd.DataFrame({"a": [], "b": []})
    remover = DuplicateRemover()
    result = remover.fit_transform(df)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# MissingValueImputer
# ---------------------------------------------------------------------------

def test_imputer_fills_numeric_with_median():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
    imputer = MissingValueImputer()
    result = imputer.fit_transform(df)
    assert result["a"].isnull().sum() == 0
    assert result["a"].iloc[2] == 2.0  # median of [1,2,4]


def test_imputer_fills_categorical_with_mode():
    df = pd.DataFrame({"cat": ["x", "x", None, "y"]})
    imputer = MissingValueImputer()
    result = imputer.fit_transform(df)
    assert result["cat"].isnull().sum() == 0
    assert result["cat"].iloc[2] == "x"


def test_imputer_flags_high_missing_columns():
    df = pd.DataFrame({"mostly_null": [np.nan, np.nan, np.nan, 1.0]})
    imputer = MissingValueImputer(high_missing_threshold=0.5)
    imputer.fit(df)
    assert "mostly_null" in imputer.high_missing_columns_


def test_imputer_learns_stats_only_on_fit_data_not_transform_data():
    """Critical leakage-prevention test: fitting on train data and
    transforming test data must use TRAIN statistics, not recompute on
    the test set."""
    train = pd.DataFrame({"a": [10.0, 20.0, 30.0]})  # median = 20
    test = pd.DataFrame({"a": [np.nan, 1000.0]})

    imputer = MissingValueImputer()
    imputer.fit(train)
    result = imputer.transform(test)

    # The NaN in test should be filled with the TRAIN median (20), not
    # anything derived from the test set itself.
    assert result["a"].iloc[0] == 20.0


def test_imputer_handles_all_null_column():
    df = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan]})
    imputer = MissingValueImputer(high_missing_threshold=0.9)
    result = imputer.fit_transform(df)
    # Should not crash; column gets flagged as high-missing
    assert "all_null" in imputer.high_missing_columns_


# ---------------------------------------------------------------------------
# DataTypeCorrector
# ---------------------------------------------------------------------------

def test_type_corrector_converts_numeric_like_text_column():
    df = pd.DataFrame({"amount": ["10.5", "20.3", "", "15.0"]})
    corrector = DataTypeCorrector()
    result = corrector.fit_transform(df)
    assert pd.api.types.is_numeric_dtype(result["amount"])
    assert "amount" in corrector.converted_columns_


def test_type_corrector_leaves_genuine_text_column_untouched():
    df = pd.DataFrame({"city": ["sao paulo", "rio", "brasilia"]})
    corrector = DataTypeCorrector()
    result = corrector.fit_transform(df)
    assert not pd.api.types.is_numeric_dtype(result["city"])
    assert "city" not in corrector.converted_columns_


# ---------------------------------------------------------------------------
# LowVarianceColumnDropper
# ---------------------------------------------------------------------------

def test_low_variance_dropper_drops_near_constant_column():
    df = pd.DataFrame({
        "constant": ["delivered"] * 99 + ["canceled"],
        "varied": range(100),
    })
    dropper = LowVarianceColumnDropper(threshold=0.98)
    result = dropper.fit_transform(df)
    assert "constant" not in result.columns
    assert "varied" in result.columns


def test_low_variance_dropper_keeps_normal_variance_column():
    df = pd.DataFrame({"varied": [1, 2, 3, 4, 5] * 20})
    dropper = LowVarianceColumnDropper(threshold=0.99)
    result = dropper.fit_transform(df)
    assert "varied" in result.columns


# ---------------------------------------------------------------------------
# DataCleaningAgent (end-to-end)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "order_id": [1, 2, 2, 3, 4],  # row 2 is a duplicate
        "price": ["10.0", "20.0", "20.0", np.nan, "40.0"],
        "category": ["books", "toys", "toys", "books", None],
        "status": ["delivered"] * 5,  # zero variance
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_agent_runs_end_to_end_and_produces_report(sample_csv, tmp_path):
    agent = DataCleaningAgent(low_variance_threshold=0.9)
    output_path = str(tmp_path / "sample_cleaned.csv")
    success, result_path = agent.run(sample_csv, output_path=output_path)

    assert success is True
    assert os.path.exists(result_path)
    assert agent.report_ is not None
    assert agent.report_.n_duplicates_removed == 1
    assert "status" in agent.report_.low_variance_columns_dropped

    cleaned = pd.read_csv(result_path)
    assert cleaned.isnull().sum().sum() == 0  # no missing values remain
    assert "status" not in cleaned.columns


def test_agent_handles_nonexistent_file_gracefully():
    agent = DataCleaningAgent()
    success, message = agent.run("this_file_does_not_exist.csv")
    assert success is False
    assert "Failed to read" in message


def test_agent_writes_report_json(sample_csv, tmp_path):
    agent = DataCleaningAgent(low_variance_threshold=0.9)
    output_path = str(tmp_path / "sample_cleaned.csv")
    agent.run(sample_csv, output_path=output_path)

    report_path = str(tmp_path / "sample_cleaned_report.json")
    assert os.path.exists(report_path)
