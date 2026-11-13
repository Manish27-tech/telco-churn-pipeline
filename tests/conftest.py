"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn import config

SERVICES = ["Yes", "No", "No internet service"]


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    """A small but schema-complete version of the raw Telco extract."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    size = 60
    frame = pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(size)],
        "gender": rng.choice(["Female", "Male"], size),
        "SeniorCitizen": rng.integers(0, 2, size),
        "Partner": rng.choice(["Yes", "No"], size),
        "Dependents": rng.choice(["Yes", "No"], size),
        "tenure": rng.integers(0, 73, size),
        "PhoneService": rng.choice(["Yes", "No"], size),
        "MultipleLines": rng.choice(SERVICES, size),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size),
        "OnlineSecurity": rng.choice(SERVICES, size),
        "OnlineBackup": rng.choice(SERVICES, size),
        "DeviceProtection": rng.choice(SERVICES, size),
        "TechSupport": rng.choice(SERVICES, size),
        "StreamingTV": rng.choice(SERVICES, size),
        "StreamingMovies": rng.choice(SERVICES, size),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size),
        "PaperlessBilling": rng.choice(["Yes", "No"], size),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], size
        ),
        "MonthlyCharges": np.round(rng.uniform(20, 120, size), 2),
        "TotalCharges": np.round(rng.uniform(0, 8000, size), 2).astype(object),
        "Churn": rng.choice(["Yes", "No"], size, p=[0.3, 0.7]),
    })
    # Reproduce the real dataset's quirk: blank TotalCharges for tenure == 0.
    frame.loc[frame["tenure"] == 0, "TotalCharges"] = " "
    return frame


@pytest.fixture()
def clean_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    from churn.clean import clean

    cleaned, _ = clean(raw_frame)
    return cleaned
