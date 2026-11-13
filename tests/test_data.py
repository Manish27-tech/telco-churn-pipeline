"""Tests for schema validation and loading behaviour."""

from __future__ import annotations

import pandas as pd
import pytest

from churn import config, data


def test_validate_returns_all_checks(raw_frame: pd.DataFrame) -> None:
    report = data.validate(raw_frame)
    assert report["rows"] == len(raw_frame)
    assert len(report["checks"]) >= 8
    assert all("check" in check and "passed" in check for check in report["checks"])


def test_validate_raises_on_missing_column(raw_frame: pd.DataFrame) -> None:
    broken = raw_frame.drop(columns=["Contract"])
    with pytest.raises(data.DataValidationError, match="expected_columns"):
        data.validate(broken)


def test_validate_raises_on_duplicate_customer_ids(raw_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(data.DataValidationError, match="unique_customer_id"):
        data.validate(duplicated)


def test_validate_flags_null_target_but_does_not_raise(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.copy()
    frame.loc[frame.index[0], config.TARGET] = None
    report = data.validate(frame)
    target_check = next(c for c in report["checks"] if c["check"] == "target_not_null")
    assert not target_check["passed"]
    assert not target_check["critical"]


def test_load_raw_reads_csv(tmp_path, raw_frame: pd.DataFrame) -> None:
    path = tmp_path / "sample.csv"
    raw_frame.to_csv(path, index=False)
    loaded = data.load_raw(path)
    assert loaded.shape == raw_frame.shape
    assert list(loaded.columns) == config.SOURCE_COLUMNS
