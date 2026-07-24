"""
tests/test_eda.py

Unit tests for the EDA Agent (src/agents/eda.py).

Run with:
    pytest tests/test_eda.py -v
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.eda import EDAAgent, EDAReport


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_csv(tmp_path):
    df = pd.DataFrame({
        "price": [10.0, 20.0, 30.0, 40.0, 1000.0],  # 1000 is an outlier
        "weight": [1.5, 2.0, 2.5, 3.0, 3.5],
        "quantity": [1, 2, 3, 4, 5],
    })
    path = tmp_path / "numeric.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def categorical_csv(tmp_path):
    df = pd.DataFrame({
        "city": ["sao paulo", "rio", "brasilia", "curitiba"],
        "state": ["SP", "RJ", "DF", "PR"],
    })
    path = tmp_path / "categorical.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Normal numeric data
# ---------------------------------------------------------------------------

def test_run_on_numeric_data_succeeds(numeric_csv):
    agent = EDAAgent()
    success, result = agent.run(numeric_csv)
    assert success is True
    assert os.path.exists(result)
    assert agent.report_ is not None


def test_report_shape_matches_input(numeric_csv):
    agent = EDAAgent()
    agent.run(numeric_csv)
    assert agent.report_.input_shape == (5, 3)


def test_descriptive_stats_keys(numeric_csv):
    agent = EDAAgent()
    agent.run(numeric_csv)
    stats = agent.report_.descriptive_stats
    assert set(stats.keys()) == {"price", "weight", "quantity"}
    for col_stats in stats.values():
        assert "mean" in col_stats
        assert "std" in col_stats
        assert "min" in col_stats
        assert "max" in col_stats


def test_correlation_matrix_is_symmetric(numeric_csv):
    agent = EDAAgent()
    agent.run(numeric_csv)
    corr = agent.report_.correlation_matrix
    cols = list(corr.keys())
    for i, a in enumerate(cols):
        for b in cols[i:]:
            assert abs(corr[a][b] - corr[b][a]) < 1e-10


def test_skewness_values_are_floats_or_none(numeric_csv):
    agent = EDAAgent()
    agent.run(numeric_csv)
    for col, val in agent.report_.skewness.items():
        assert val is None or isinstance(val, float), (
            f"skewness[{col!r}] should be float or None, got {type(val)}"
        )


def test_report_is_json_serialisable(numeric_csv):
    agent = EDAAgent()
    agent.run(numeric_csv)
    # Must not raise
    serialised = json.dumps(agent.report_.to_dict())
    parsed = json.loads(serialised)
    assert "descriptive_stats" in parsed


# ---------------------------------------------------------------------------
# Column with no variance (constant column)
# ---------------------------------------------------------------------------

def test_zero_variance_column_does_not_crash(tmp_path):
    df = pd.DataFrame({
        "constant": [5.0, 5.0, 5.0, 5.0],
        "varied": [1.0, 2.0, 3.0, 4.0],
    })
    path = tmp_path / "zero_var.csv"
    df.to_csv(path, index=False)

    agent = EDAAgent()
    success, _ = agent.run(str(path))
    assert success is True


def test_zero_variance_column_outlier_counts_zero(tmp_path):
    df = pd.DataFrame({"constant": [7.0] * 10})
    path = tmp_path / "const.csv"
    df.to_csv(path, index=False)

    agent = EDAAgent()
    agent.run(str(path))
    summary = agent.report_.outlier_summary["constant"]
    # IQR is 0 → fences equal the constant → no outliers
    assert summary["n_outliers"] == 0


# ---------------------------------------------------------------------------
# DataFrame with only categorical columns
# ---------------------------------------------------------------------------

def test_only_categorical_columns_does_not_crash(categorical_csv):
    agent = EDAAgent()
    success, _ = agent.run(categorical_csv)
    assert success is True


def test_only_categorical_columns_produces_empty_numeric_sections(categorical_csv):
    agent = EDAAgent()
    agent.run(categorical_csv)
    report = agent.report_
    assert report.numeric_columns == []
    assert report.descriptive_stats == {}
    assert report.correlation_matrix == {}
    assert report.skewness == {}
    assert report.outlier_summary == {}


def test_only_categorical_columns_has_correct_cat_columns(categorical_csv):
    agent = EDAAgent()
    agent.run(categorical_csv)
    assert set(agent.report_.categorical_columns) == {"city", "state"}


# ---------------------------------------------------------------------------
# Outlier detection on a known synthetic example
# ---------------------------------------------------------------------------

def test_outlier_detection_known_example(tmp_path):
    # Values 1-10 are within fences; 100 is a clear outlier.
    # Q1=3.25, Q3=7.75, IQR=4.5, fences: [-3.5, 14.5]
    values = list(range(1, 11)) + [100]
    df = pd.DataFrame({"x": values})
    path = tmp_path / "outlier.csv"
    df.to_csv(path, index=False)

    agent = EDAAgent(iqr_multiplier=1.5)
    agent.run(str(path))
    summary = agent.report_.outlier_summary["x"]
    assert summary["n_outliers"] == 1
    assert summary["upper_fence"] < 100


def test_outlier_detection_no_outliers(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    path = tmp_path / "no_outlier.csv"
    df.to_csv(path, index=False)

    agent = EDAAgent(iqr_multiplier=1.5)
    agent.run(str(path))
    assert agent.report_.outlier_summary["x"]["n_outliers"] == 0


def test_outlier_pct_is_correct(tmp_path):
    # 2 outliers out of 10 non-null values → 20 %
    values = list(range(1, 9)) + [1000, -1000]
    df = pd.DataFrame({"x": values})
    path = tmp_path / "pct_outlier.csv"
    df.to_csv(path, index=False)

    agent = EDAAgent(iqr_multiplier=1.5)
    agent.run(str(path))
    summary = agent.report_.outlier_summary["x"]
    assert summary["n_outliers"] == 2
    assert abs(summary["pct_outliers"] - 20.0) < 0.01


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_nonexistent_file_returns_failure():
    agent = EDAAgent()
    success, message = agent.run("this_file_does_not_exist.csv")
    assert success is False
    assert "Failed to read" in message


def test_report_json_written_to_disk(numeric_csv, tmp_path):
    agent = EDAAgent()
    success, report_path = agent.run(numeric_csv)
    assert success is True
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert "outlier_summary" in data


def test_single_numeric_column_no_correlation(tmp_path):
    """Correlation matrix requires ≥2 numeric columns; one column → empty dict."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    path = tmp_path / "single.csv"
    df.to_csv(path, index=False)

    agent = EDAAgent()
    agent.run(str(path))
    assert agent.report_.correlation_matrix == {}
