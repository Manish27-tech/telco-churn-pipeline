"""Tests for type coercion, imputation and cleaning bookkeeping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn import config
from churn.clean import clean, coerce_types, handle_missing


def test_coerce_types_maps_target_and_parses_total_charges(raw_frame: pd.DataFrame) -> None:
    typed = coerce_types(raw_frame)
    assert typed[config.TARGET].dtype == "Int64"
    assert set(typed[config.TARGET].dropna().unique()) <= {0, 1}
    assert pd.api.types.is_float_dtype(typed["TotalCharges"])
    assert typed.loc[typed["tenure"] == 0, "TotalCharges"].isna().all()


def test_handle_missing_sets_zero_for_new_customers(raw_frame: pd.DataFrame) -> None:
    typed = coerce_types(raw_frame)
    filled, log = handle_missing(typed)
    new_customers = filled["tenure"] == 0
    assert (filled.loc[new_customers, "TotalCharges"] == 0).all()
    assert log["total_charges_zero_for_new_customers"] == int(new_customers.sum())
    assert filled.isna().sum().sum() == 0


def test_handle_missing_uses_unknown_level_for_categories(raw_frame: pd.DataFrame) -> None:
    typed = coerce_types(raw_frame)
    typed.loc[typed.index[0], "Contract"] = np.nan
    filled, log = handle_missing(typed)
    assert filled.loc[0, "Contract"] == "Unknown"
    assert log["categorical_unknown_imputed"]["Contract"] == 1


def test_clean_removes_duplicates_and_records_log(raw_frame: pd.DataFrame) -> None:
    frame = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    cleaned, log = clean(frame)
    assert log["duplicate_customer_ids_removed"] == 1
    assert cleaned[config.ID_COLUMN].is_unique
    assert log["rows_out"] == len(cleaned) == len(raw_frame)
    assert 0 < log["churn_rate"] < 1


def test_clean_adds_outlier_flag_without_nulls(clean_frame: pd.DataFrame) -> None:
    assert "flag_high_spend" in clean_frame.columns
    assert set(clean_frame["flag_high_spend"].unique()) <= {0, 1}
    assert clean_frame[config.TARGET].dtype == "int64"


def test_clean_is_deterministic(raw_frame: pd.DataFrame) -> None:
    first, _ = clean(raw_frame)
    second, _ = clean(raw_frame)
    pd.testing.assert_frame_equal(first, second)
