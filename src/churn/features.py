"""Feature engineering for the churn model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive behavioural and contract features from the raw columns."""
    out = df.copy()

    tenure_safe = out["tenure"].clip(lower=1)

    # Revenue behaviour --------------------------------------------------- #
    out["tenure_years"] = out["tenure"] / 12.0
    out["avg_monthly_spend"] = out["TotalCharges"] / tenure_safe
    # Positive => currently paying more than the historical average (upsold).
    out["spend_delta"] = out["MonthlyCharges"] - out["avg_monthly_spend"]

    # Service footprint --------------------------------------------------- #
    service_flags = out[config.SERVICE_COLUMNS].eq("Yes")
    out["num_services"] = service_flags.sum(axis=1).astype(int)
    out["service_density"] = out["num_services"] / len(config.SERVICE_COLUMNS)
    out["has_internet"] = (out["InternetService"] != "No").astype(int)

    # Contract and billing -------------------------------------------------#
    out["contract_risk"] = (
        out["Contract"]
        .map({"Month-to-month": 2, "One year": 1, "Two year": 0})
        .fillna(1)
        .astype(int)
    )
    out["auto_payment"] = (
        out["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
    )
    out["is_new_customer"] = (out["tenure"] <= 6).astype(int)
    out["family_plan"] = (
        (out["Partner"] == "Yes") | (out["Dependents"] == "Yes")
    ).astype(int)

    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns() -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature lists used by the model."""
    numeric = [
        *config.NUMERIC_FEATURES,
        *config.ENGINEERED_NUMERIC,
        "flag_high_spend",
    ]
    categorical = [*config.CATEGORICAL_FEATURES, *config.ENGINEERED_CATEGORICAL]
    return numeric, categorical


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into model inputs and the binary target."""
    engineered = add_features(df)
    numeric, categorical = feature_columns()
    features = engineered[numeric + categorical].copy()
    target = engineered[config.TARGET].astype(int)
    return features, target
