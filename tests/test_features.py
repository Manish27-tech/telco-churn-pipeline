"""Tests for engineered features and the feature matrix contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn import config
from churn.features import add_features, build_feature_matrix, feature_columns


def test_contract_risk_is_ordered(clean_frame: pd.DataFrame) -> None:
    engineered = add_features(clean_frame).drop_duplicates("Contract")
    mapping = dict(zip(engineered["Contract"], engineered["contract_risk"], strict=True))
    assert mapping["Month-to-month"] > mapping["One year"] > mapping["Two year"]


def test_new_customer_flag_matches_tenure(clean_frame: pd.DataFrame) -> None:
    engineered = add_features(clean_frame)
    expected = (engineered["tenure"] <= 6).astype(int)
    assert (engineered["is_new_customer"] == expected).all()


def test_num_services_counts_only_yes(clean_frame: pd.DataFrame) -> None:
    engineered = add_features(clean_frame)
    manual = engineered[config.SERVICE_COLUMNS].eq("Yes").sum(axis=1)
    assert (engineered["num_services"] == manual).all()
    assert engineered["num_services"].between(0, len(config.SERVICE_COLUMNS)).all()


def test_avg_monthly_spend_is_zero_safe(clean_frame: pd.DataFrame) -> None:
    engineered = add_features(clean_frame)
    assert engineered["avg_monthly_spend"].notna().all()
    assert np.isfinite(engineered["avg_monthly_spend"]).all()
    zero_tenure = engineered["tenure"] == 0
    if zero_tenure.any():
        expected = engineered.loc[zero_tenure, "TotalCharges"]
        assert np.allclose(engineered.loc[zero_tenure, "avg_monthly_spend"], expected)


def test_feature_matrix_has_no_non_finite_values(clean_frame: pd.DataFrame) -> None:
    X, y = build_feature_matrix(clean_frame)
    numeric, categorical = feature_columns()
    assert list(X.columns) == numeric + categorical
    assert X.select_dtypes(include=[np.number]).isna().sum().sum() == 0
    assert len(X) == len(y) == len(clean_frame)
    assert set(y.unique()) <= {0, 1}
