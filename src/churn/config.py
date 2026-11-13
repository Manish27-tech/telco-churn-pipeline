"""Central configuration: paths, dataset locations and modelling constants."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
ARTIFACT_DIR = ROOT / "artifacts"
NOTEBOOK_DIR = ROOT / "notebooks"

for _directory in (RAW_DIR, PROCESSED_DIR, REPORT_DIR, FIGURE_DIR, ARTIFACT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# IBM's public sample extract of the Telco customer churn dataset.
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_FALLBACK_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/main/"
    "data/Telco-Customer-Churn.csv",
]
RAW_FILENAME = "Telco-Customer-Churn.csv"
PROCESSED_FILENAME = "telco_churn_clean.csv"

ID_COLUMN = "customerID"
TARGET = "Churn"

# Columns exactly as delivered by the source file.
SOURCE_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

ENGINEERED_NUMERIC = [
    "avg_monthly_spend",
    "spend_delta",
    "num_services",
    "service_density",
    "contract_risk",
    "tenure_years",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ENGINEERED_CATEGORICAL = [
    "auto_payment",
    "has_internet",
    "is_new_customer",
    "family_plan",
]

# Keep raw-ish column names around for reports and validation.
SERVICE_COLUMNS = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITER = 12
SEARCH_FOLDS = 3

FIG_DPI = 140
PALETTE = {0: "#4C72B0", 1: "#C44E52"}
