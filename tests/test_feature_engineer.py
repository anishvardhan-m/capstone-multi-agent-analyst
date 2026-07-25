"""
tests/test_feature_engineer.py

Unit tests for the feature engineering transformers
(src/tools/feature_tools.py) and the FeatureEngineeringAgent
(src/agents/feature_engineer.py).

Run with:
    pytest tests/test_feature_engineer.py -v
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.feature_engineer import FeatureEngineeringAgent
from src.tools.feature_tools import (
    CategoricalEncoder,
    NumericScaler,
    RedundantFeatureDropper,
    SkewnessReducer,
)


# ---------------------------------------------------------------------------
# RedundantFeatureDropper
# ---------------------------------------------------------------------------

def test_redundant_dropper_drops_correlated_column():
    # b is a perfect linear function of a → corr = 1.0
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    dropper = RedundantFeatureDropper(threshold=0.95)
    result = dropper.fit_transform(df)
    assert "a" in result.columns
    assert "b" not in result.columns
    assert "b" in dropper.dropped_columns_


def test_redundant_dropper_keeps_uncorrelated_columns():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "x": rng.standard_normal(100),
        "y": rng.standard_normal(100),
    })
    dropper = RedundantFeatureDropper(threshold=0.95)
    result = dropper.fit_transform(df)
    assert "x" in result.columns
    assert "y" in result.columns
    assert dropper.dropped_columns_ == []


def test_redundant_dropper_never_drops_protected_columns():
    # Even if is_late_delivery is perfectly correlated with another column,
    # it must never be dropped.
    df = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0, 4.0],
        "is_late_delivery": [1.0, 2.0, 3.0, 4.0],  # perfectly correlated
    })
    dropper = RedundantFeatureDropper(threshold=0.95)
    result = dropper.fit_transform(df)
    assert "is_late_delivery" in result.columns


def test_redundant_dropper_keeps_first_of_correlated_pair():
    # a and b are correlated; a comes first — b should be dropped, not a.
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.1, 2.1, 3.1], "c": [10.0, 1.0, 5.0]})
    dropper = RedundantFeatureDropper(threshold=0.95)
    result = dropper.fit_transform(df)
    assert "a" in result.columns
    assert "b" not in result.columns


def test_redundant_dropper_single_column_does_not_crash():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    dropper = RedundantFeatureDropper(threshold=0.95)
    result = dropper.fit_transform(df)
    assert list(result.columns) == ["a"]


# ---------------------------------------------------------------------------
# SkewnessReducer
# ---------------------------------------------------------------------------

def test_skewness_reducer_lowers_skewness():
    # Exponential distribution is heavily right-skewed
    rng = np.random.default_rng(0)
    values = rng.exponential(scale=2.0, size=500)
    df = pd.DataFrame({"x": values})
    reducer = SkewnessReducer(skew_threshold=0.5)
    reducer.fit(df)
    assert "x" in reducer.log_transformed_columns_
    result = reducer.transform(df)
    assert result["x"].skew() < df["x"].skew()


def test_skewness_reducer_skips_column_with_negative_values():
    df = pd.DataFrame({"x": [-5.0, 1.0, 2.0, 100.0, 500.0]})
    reducer = SkewnessReducer(skew_threshold=0.5)
    reducer.fit(df)
    assert "x" not in reducer.log_transformed_columns_


def test_skewness_reducer_skips_low_skew_column():
    # Uniform-ish data has near-zero skewness
    df = pd.DataFrame({"x": np.linspace(1, 10, 100)})
    reducer = SkewnessReducer(skew_threshold=1.0)
    reducer.fit(df)
    assert "x" not in reducer.log_transformed_columns_


def test_skewness_reducer_never_transforms_protected_columns():
    # is_late_delivery has skew ~3 in the real data; must never be transformed
    df = pd.DataFrame({
        "is_late_delivery": [0] * 92 + [1] * 8,  # skewed binary
        "feature": list(range(100)),
    })
    reducer = SkewnessReducer(skew_threshold=0.5)
    reducer.fit(df)
    assert "is_late_delivery" not in reducer.log_transformed_columns_


def test_skewness_reducer_transform_uses_fit_columns_only():
    # Fit on skewed data, transform on different data — only fit-time columns
    # should be transformed.
    train = pd.DataFrame({"x": np.exp(np.arange(1, 11, dtype=float)), "y": np.ones(10)})
    test = pd.DataFrame({"x": np.exp(np.arange(1, 11, dtype=float)), "y": np.ones(10)})
    reducer = SkewnessReducer(skew_threshold=0.5)
    reducer.fit(train)
    result = reducer.transform(test)
    # x should be log-transformed, y should be untouched (it's constant)
    assert "x" in reducer.log_transformed_columns_
    assert "y" not in reducer.log_transformed_columns_
    pd.testing.assert_series_equal(result["y"], test["y"])


# ---------------------------------------------------------------------------
# NumericScaler
# ---------------------------------------------------------------------------

def test_scaler_fit_transform_produces_zero_mean_unit_std():
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 50.0]})
    scaler = NumericScaler()
    result = scaler.fit_transform(df)
    assert abs(result["a"].mean()) < 1e-10
    assert abs(result["a"].std(ddof=1) - 1.0) < 1e-10


def test_scaler_leakage_prevention():
    """Transform must use FIT-time mean/std, not recompute from transform data."""
    train = pd.DataFrame({"a": [10.0, 20.0, 30.0]})  # mean=20, std=10
    test = pd.DataFrame({"a": [1000.0, 2000.0, 3000.0]})

    scaler = NumericScaler()
    scaler.fit(train)

    result = scaler.transform(test)
    # If using train mean=20, std=10: (1000-20)/10 = 98.0
    assert abs(result["a"].iloc[0] - 98.0) < 1e-6


def test_scaler_skips_zero_variance_column():
    df = pd.DataFrame({"constant": [5.0, 5.0, 5.0], "varied": [1.0, 2.0, 3.0]})
    scaler = NumericScaler()
    result = scaler.fit_transform(df)
    assert "constant" in scaler.zero_variance_columns_
    # constant column values should be unchanged
    assert list(result["constant"]) == [5.0, 5.0, 5.0]


def test_scaler_never_scales_protected_columns():
    df = pd.DataFrame({
        "is_late_delivery": [0, 1, 0, 1],
        "order_id_num": [100.0, 200.0, 300.0, 400.0],  # not protected
        "feature": [1.0, 2.0, 3.0, 4.0],
    })
    scaler = NumericScaler()
    result = scaler.fit_transform(df)
    # is_late_delivery values must be exactly [0, 1, 0, 1]
    assert list(result["is_late_delivery"]) == [0, 1, 0, 1]
    # The non-protected columns should have been scaled
    assert "feature" in scaler.scaled_columns_


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------

def test_ohe_encoder_chosen_for_low_cardinality():
    df = pd.DataFrame({"payment": ["credit", "boleto", "credit", "voucher"]})
    encoder = CategoricalEncoder(ohe_threshold=20)
    result = encoder.fit_transform(df)
    assert encoder.encoding_map_["payment"] == "one_hot"
    assert "payment" not in result.columns
    assert any(c.startswith("payment_") for c in result.columns)


def test_frequency_encoder_chosen_for_high_cardinality():
    # 25 unique values → exceeds ohe_threshold of 20
    cats = [f"city_{i}" for i in range(25)]
    df = pd.DataFrame({"city": cats})
    encoder = CategoricalEncoder(ohe_threshold=20)
    result = encoder.fit_transform(df)
    assert encoder.encoding_map_["city"] == "frequency"
    assert "city" in result.columns
    assert pd.api.types.is_float_dtype(result["city"])
    # Proportions must land in [0, 1], not raw counts
    assert result["city"].between(0, 1).all()


def test_frequency_encoder_correct_proportions():
    # 3 unique values, threshold=2 → 3 is NOT < 2 → frequency encoding
    df = pd.DataFrame({"state": ["SP"] * 50 + ["RJ"] * 30 + ["MG"] * 20})
    encoder = CategoricalEncoder(ohe_threshold=2)
    encoder.fit(df)
    result = encoder.transform(df)
    assert encoder.encoding_map_["state"] == "frequency"
    # SP appears 50/100 times → should be encoded as 0.5
    sp_rows = result.loc[df["state"] == "SP", "state"]
    assert sp_rows.to_numpy() == pytest.approx(0.5)
    rj_rows = result.loc[df["state"] == "RJ", "state"]
    assert rj_rows.to_numpy() == pytest.approx(0.3)


def test_ohe_encoder_consistent_columns_on_transform():
    """Unseen categories in transform must not add new dummy columns."""
    train = pd.DataFrame({"color": ["red", "blue", "green"]})
    test = pd.DataFrame({"color": ["red", "purple"]})  # purple unseen
    encoder = CategoricalEncoder(ohe_threshold=20)
    encoder.fit(train)
    result = encoder.transform(test)
    fit_dummy_cols = set(encoder.ohe_columns_["color"])
    result_dummy_cols = {c for c in result.columns if c.startswith("color_")}
    assert result_dummy_cols == fit_dummy_cols
    # purple → all zeros for its row
    purple_row = result.iloc[1]
    for col in fit_dummy_cols:
        assert purple_row[col] == 0


def test_ohe_encoder_never_encodes_protected_columns():
    df = pd.DataFrame({
        "order_id": ["abc", "def", "ghi"],
        "payment": ["credit", "boleto", "credit"],
    })
    encoder = CategoricalEncoder(ohe_threshold=20)
    result = encoder.fit_transform(df)
    assert "order_id" in result.columns
    assert "payment" not in encoder.encoding_map_ or encoder.encoding_map_.get("order_id") is None


def test_frequency_encoder_maps_unseen_to_zero():
    # 2 unique values, threshold=1 → 2 is NOT < 1 → frequency encoding
    train = pd.DataFrame({"city": ["SP", "RJ", "SP"]})
    test = pd.DataFrame({"city": ["SP", "unknown_city"]})
    encoder = CategoricalEncoder(ohe_threshold=1)
    encoder.fit(train)
    result = encoder.transform(test)
    assert encoder.encoding_map_["city"] == "frequency"
    assert result["city"].iloc[0] == pytest.approx(2 / 3)  # SP is 2 of 3 train rows
    assert result["city"].iloc[1] == 0   # unseen → 0


# ---------------------------------------------------------------------------
# FeatureEngineeringAgent (end-to-end)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    rng = np.random.default_rng(7)
    n = 200
    price = rng.exponential(scale=50, size=n)  # skewed, positive
    df = pd.DataFrame({
        "order_id": [f"oid_{i}" for i in range(n)],
        "price": price,
        "total_payment": price * 1.05 + rng.normal(0, 0.1, n),  # corr ~0.9999 with price
        "weight": rng.uniform(100, 5000, n),
        "city": [f"city_{i % 30}" for i in range(n)],           # 30 unique → frequency
        "payment_type": rng.choice(["credit", "boleto", "voucher"], n),  # 3 unique → OHE
        "is_late_delivery": rng.choice([0, 1], n, p=[0.9, 0.1]),
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_agent_runs_end_to_end_and_produces_report(sample_csv):
    agent = FeatureEngineeringAgent(corr_threshold=0.95)
    success, result_path = agent.run(sample_csv)
    assert success is True
    assert os.path.exists(result_path)
    assert agent.report_ is not None


def test_agent_preserves_is_late_delivery_exactly(sample_csv):
    original = pd.read_csv(sample_csv)
    agent = FeatureEngineeringAgent(corr_threshold=0.95)
    success, result_path = agent.run(sample_csv)
    result = pd.read_csv(result_path)
    assert "is_late_delivery" in result.columns
    pd.testing.assert_series_equal(
        result["is_late_delivery"].reset_index(drop=True),
        original["is_late_delivery"].reset_index(drop=True),
        check_names=True,
    )


def test_agent_preserves_order_id_exactly(sample_csv):
    original = pd.read_csv(sample_csv)
    agent = FeatureEngineeringAgent(corr_threshold=0.95)
    success, result_path = agent.run(sample_csv)
    result = pd.read_csv(result_path)
    assert "order_id" in result.columns
    pd.testing.assert_series_equal(
        result["order_id"].reset_index(drop=True),
        original["order_id"].reset_index(drop=True),
        check_names=True,
    )


def test_agent_drops_redundant_column(sample_csv):
    agent = FeatureEngineeringAgent(corr_threshold=0.95)
    agent.run(sample_csv)
    # total_payment is nearly perfectly correlated with price → should be dropped
    assert len(agent.report_.redundant_columns_dropped) >= 1


def test_agent_log_transforms_skewed_columns(sample_csv):
    agent = FeatureEngineeringAgent(skew_threshold=1.0)
    agent.run(sample_csv)
    # price is exponentially distributed → heavily skewed → should be log-transformed
    assert "price" in agent.report_.log_transformed_columns


def test_agent_ohe_low_cardinality_categorical(sample_csv):
    agent = FeatureEngineeringAgent(ohe_threshold=20)
    success, result_path = agent.run(sample_csv)
    result = pd.read_csv(result_path)
    # payment_type (3 unique) → OHE → dummy columns present, original gone
    assert agent.report_.encoding_map.get("payment_type") == "one_hot"
    assert "payment_type" not in result.columns
    assert any(c.startswith("payment_type_") for c in result.columns)


def test_agent_freq_encodes_high_cardinality_categorical(sample_csv):
    agent = FeatureEngineeringAgent(ohe_threshold=20)
    success, result_path = agent.run(sample_csv)
    # city (30 unique) → frequency encoding
    assert agent.report_.encoding_map.get("city") == "frequency"


def test_agent_report_json_written_to_disk(sample_csv, tmp_path):
    agent = FeatureEngineeringAgent()
    output_path = str(tmp_path / "out_features.csv")
    agent.run(sample_csv, output_path=output_path)
    report_path = str(tmp_path / "out_features_features_report.json")
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert "redundant_columns_dropped" in data
    assert "log_transformed_columns" in data
    assert "encoding_map" in data
    assert "scaler_stats" in data


def test_agent_handles_nonexistent_file_gracefully():
    agent = FeatureEngineeringAgent()
    success, message = agent.run("this_file_does_not_exist.csv")
    assert success is False
    assert "Failed to read" in message


def test_agent_report_is_json_serialisable(sample_csv):
    agent = FeatureEngineeringAgent()
    agent.run(sample_csv)
    serialised = json.dumps(agent.report_.to_dict())
    parsed = json.loads(serialised)
    assert "protected_columns" in parsed
    assert "is_late_delivery" in parsed["protected_columns"]
