"""Cleaning: type coercion, duplicate removal and missing-value handling."""

from __future__ import annotations

import pandas as pd

from . import config


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes and strip stray whitespace from text columns."""
    out = df.copy()

    text_columns = out.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        out[column] = out[column].astype("string").str.strip()

    # TotalCharges arrives as text and contains blank entries for brand-new
    # customers (tenure == 0), so a plain numeric cast is required first.
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["SeniorCitizen"] = out["SeniorCitizen"].astype("int64")
    out[config.TARGET] = out[config.TARGET].map({"Yes": 1, "No": 0}).astype("Int64")
    return out


def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Impute missing values and return the frame plus a cleaning log."""
    out = df.copy()
    log: dict = {"rows_in": int(len(out))}

    # 1. New customers have no billing history -> TotalCharges is genuinely 0.
    new_customer_mask = out["TotalCharges"].isna() & (out["tenure"] == 0)
    log["total_charges_zero_for_new_customers"] = int(new_customer_mask.sum())
    out.loc[new_customer_mask, "TotalCharges"] = 0.0

    # 2. Any remaining numeric gaps are filled with the column median.
    numeric_columns = out.select_dtypes(include=["number"]).columns
    numeric_gaps = {c: int(out[c].isna().sum()) for c in numeric_columns if out[c].isna().any()}
    for column in numeric_gaps:
        out[column] = out[column].fillna(out[column].median())
    log["numeric_median_imputed"] = numeric_gaps

    # 3. Categorical gaps become an explicit "Unknown" level.
    categorical_columns = out.select_dtypes(include=["object", "string"]).columns
    categorical_columns = [c for c in categorical_columns if c not in {config.ID_COLUMN}]
    categorical_gaps = {
        c: int(out[c].isna().sum()) for c in categorical_columns if out[c].isna().any()
    }
    for column in categorical_gaps:
        out[column] = out[column].fillna("Unknown")
    log["categorical_unknown_imputed"] = categorical_gaps

    log["remaining_nulls"] = int(out.isna().sum().sum())
    return out, log


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full cleaning pass used by both the CLI and the notebooks."""
    typed = coerce_types(df)

    log: dict = {"rows_in": int(len(df))}

    before = int(typed.duplicated(subset=[config.ID_COLUMN]).sum())
    typed = typed.drop_duplicates(subset=[config.ID_COLUMN], keep="first")
    log["duplicate_customer_ids_removed"] = before

    target_missing = int(typed[config.TARGET].isna().sum())
    typed = typed[typed[config.TARGET].notna()].copy()
    typed[config.TARGET] = typed[config.TARGET].astype("int64")
    log["rows_missing_target_dropped"] = target_missing

    cleaned, imputation_log = handle_missing(typed)
    log.update(imputation_log)

    # Outlier flags are informative for EDA but never used as model features.
    q99 = float(cleaned["MonthlyCharges"].quantile(0.99))
    cleaned["flag_high_spend"] = (cleaned["MonthlyCharges"] > q99).astype(int)
    log["high_spend_threshold"] = round(q99, 2)
    log["high_spend_rows"] = int(cleaned["flag_high_spend"].sum())

    cleaned = cleaned.reset_index(drop=True)
    log["rows_out"] = int(len(cleaned))
    log["churn_rate"] = round(float(cleaned[config.TARGET].mean()), 4)
    cleaned.attrs["cleaning_log"] = log
    return cleaned, log


def save_processed(df: pd.DataFrame) -> str:
    """Persist the cleaned frame so downstream steps can reuse it."""
    destination = config.PROCESSED_DIR / config.PROCESSED_FILENAME
    df.to_csv(destination, index=False)
    return str(destination)
