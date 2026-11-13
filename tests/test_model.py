"""Tests for the modelling pipeline: preprocessing, fitting and evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn import config
from churn.features import build_feature_matrix, feature_columns
from churn.model import (
    build_preprocessor,
    candidate_models,
    feature_importance,
    select_threshold,
)


@pytest.fixture()
def matrices(clean_frame: pd.DataFrame):
    return build_feature_matrix(clean_frame)


def test_preprocessor_outputs_dense_numeric_matrix(matrices) -> None:
    X, y = matrices
    numeric, categorical = feature_columns()
    preprocessor = build_preprocessor(numeric, categorical)
    transformed = preprocessor.fit_transform(X, y)

    assert transformed.ndim == 2
    assert transformed.shape[0] == len(X)
    assert np.isfinite(transformed).all()
    names = preprocessor.get_feature_names_out()
    assert len(names) == transformed.shape[1]
    assert any(name.startswith("num__") for name in names)
    assert any(name.startswith("cat__") for name in names)


def test_pipeline_predicts_valid_probabilities(matrices) -> None:
    X, y = matrices
    numeric, categorical = feature_columns()
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X)[:, 1]

    assert probabilities.shape == (len(X),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_select_threshold_returns_in_range(matrices) -> None:
    X, y = matrices
    numeric, categorical = feature_columns()
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    threshold, curve = select_threshold(pipeline, X, y)

    assert 0.1 <= threshold <= 0.9
    assert len(curve) > 50
    best = max(curve, key=lambda row: row["f1"])
    assert best["threshold"] == pytest.approx(threshold)


def test_feature_importance_is_sorted(matrices) -> None:
    X, y = matrices
    numeric, categorical = feature_columns()
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
    ]).fit(X, y)

    importances = feature_importance(pipeline, top_n=5)
    values = [item["importance"] for item in importances]
    assert len(importances) == 5
    assert values == sorted(values, reverse=True)
    assert not any("__" in item["feature"] for item in importances)


def test_candidate_models_are_distinct_families() -> None:
    models = candidate_models()
    assert set(models) == {"logistic_regression", "random_forest", "gradient_boosting"}
    assert len({type(m).__name__ for m in models.values()}) == 3


def test_expected_target_balance(clean_frame: pd.DataFrame) -> None:
    rate = clean_frame[config.TARGET].mean()
    assert 0 < rate < 1
