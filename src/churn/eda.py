"""Exploratory data analysis: summary statistics, plots and insight tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config  # noqa: E402
from .features import add_features  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.autolayout"] = True

TENURE_BINS = [0, 6, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]

BINARY_MAP = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
ORDINAL_MAPS = {
    "Contract": {"Month-to-month": 2, "One year": 1, "Two year": 0},
    "InternetService": {"Fiber optic": 2, "DSL": 1, "No": 0},
    "PaymentMethod": {
        "Electronic check": 3,
        "Mailed check": 2,
        "Bank transfer (automatic)": 1,
        "Credit card (automatic)": 0,
    },
}


def _save(fig: plt.Figure, filename: str) -> Path:
    path = config.FIGURE_DIR / filename
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] figure -> {path.relative_to(config.ROOT)}")
    return path


def _numeric_view(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categories numerically so a correlation matrix is meaningful."""
    frame = add_features(df)
    columns: dict[str, pd.Series] = {}
    for column in config.CATEGORICAL_FEATURES:
        series = frame[column].astype(str)
        if column in ORDINAL_MAPS:
            columns[column] = series.map(ORDINAL_MAPS[column])
        else:
            columns[column] = series.map(BINARY_MAP)
    numeric = frame[
        ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
         "avg_monthly_spend", "num_services", "contract_risk", "service_density"]
    ].copy()
    for name, series in columns.items():
        numeric[name] = pd.to_numeric(series, errors="coerce")
    numeric[config.TARGET] = frame[config.TARGET]
    return numeric


# --------------------------------------------------------------------------- #
# Individual figures
# --------------------------------------------------------------------------- #
def plot_target_balance(df: pd.DataFrame) -> None:
    counts = df[config.TARGET].value_counts().sort_index()
    share = counts / counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(x=["Retained", "Churned"], y=counts.values, ax=axes[0],
                hue=["Retained", "Churned"], legend=False,
                palette=[config.PALETTE[0], config.PALETTE[1]])
    axes[0].set_title("Customer counts")
    axes[0].set_ylabel("Customers")
    for index, value in enumerate(counts.values):
        axes[0].text(index, value + 40, f"{value:,}", ha="center", fontsize=10)

    axes[1].pie(share.values, labels=["Retained", "Churned"], autopct="%1.1f%%",
                colors=[config.PALETTE[0], config.PALETTE[1]], startangle=90,
                wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Class share")
    fig.suptitle("Target distribution: churn is imbalanced (~1 in 4)", fontsize=13)
    _save(fig, "01_target_balance.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]
    frame = add_features(df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, column in zip(axes.ravel(), numeric, strict=True):
        for label, color in config.PALETTE.items():
            subset = frame.loc[frame[config.TARGET] == label, column].dropna()
            sns.kdeplot(subset, ax=ax, fill=True, alpha=0.35, color=color,
                        label="Churned" if label else "Retained", cut=0)
        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel("")
        ax.legend(title="")
    fig.suptitle("Distributions split by churn", fontsize=13)
    _save(fig, "02_numeric_distributions.png")


def plot_categorical_churn_rates(df: pd.DataFrame, columns: list[str],
                                 title: str, filename: str) -> None:
    rows = int(np.ceil(len(columns) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(14, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, columns, strict=False):
        rates = (
            df.groupby(column, observed=True)[config.TARGET]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )
        sns.barplot(x=rates["mean"], y=rates.index.astype(str), ax=ax,
                    hue=rates.index.astype(str), legend=False,
                    palette="rocket", orient="h")
        ax.axvline(df[config.TARGET].mean(), color="black", linestyle="--",
                   linewidth=1, label="overall")
        ax.set_title(column, fontsize=10)
        ax.set_xlabel("churn rate")
        ax.set_ylabel("")
        for index, (mean, count) in enumerate(zip(rates["mean"], rates["count"], strict=True)):
            ax.text(mean + 0.01, index, f"{mean:.0%} (n={count:,})", va="center", fontsize=7)

    for ax in axes[len(columns):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    _save(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    correlation = _numeric_view(df).corr(numeric_only=True)
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap="vlag",
                center=0, square=True, cbar_kws={"shrink": 0.75},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation matrix (ordinal-encoded categories)")
    _save(fig, "05_correlation_heatmap.png")


def plot_tenure_cohort(df: pd.DataFrame) -> None:
    frame = df.copy()
    frame["tenure_bucket"] = pd.cut(frame["tenure"], bins=TENURE_BINS,
                                    labels=TENURE_LABELS, include_lowest=True)
    grouped = frame.groupby("tenure_bucket", observed=True)[config.TARGET].agg(["mean", "count"])

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()
    ax2.bar(grouped.index.astype(str), grouped["count"], color="#d9d9d9",
            alpha=0.7, label="customers")
    ax1.plot(grouped.index.astype(str), grouped["mean"], color=config.PALETTE[1],
             marker="o", linewidth=2, label="churn rate")
    for index, value in enumerate(grouped["mean"]):
        ax1.annotate(f"{value:.0%}", (index, value), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color=config.PALETTE[1])
    ax1.set_ylim(0, max(0.6, grouped["mean"].max() * 1.2))
    ax1.set_ylabel("Churn rate", color=config.PALETTE[1])
    ax2.set_ylabel("Customers")
    ax1.set_xlabel("Tenure bucket (months)")
    ax1.set_title("Churn is front-loaded: new customers leave the fastest")
    ax1.legend(loc="upper right")
    _save(fig, "06_tenure_cohort.png")


def plot_spend_landscape(df: pd.DataFrame) -> None:
    frame = df.copy()
    frame["tenure"] = frame["tenure"].clip(lower=0.5)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, color in config.PALETTE.items():
        subset = frame[frame[config.TARGET] == label]
        ax.scatter(subset["tenure"], subset["MonthlyCharges"], s=9, alpha=0.35,
                   color=color, label="Churned" if label else "Retained",
                   edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Tenure (months, log scale)")
    ax.set_ylabel("Monthly charges")
    ax.set_title("Price / tenure landscape by churn status")
    ax.legend(markerscale=2)
    _save(fig, "07_spend_landscape.png")


def plot_contract_mix(df: pd.DataFrame) -> None:
    mix = (
        df.groupby(["Contract", config.TARGET], observed=True)
        .size().unstack(fill_value=0)
        .div(df.groupby("Contract", observed=True).size(), axis=0)
        .sort_values(1)
    )
    fig, ax = plt.subplots(figsize=(8.5, 4))
    mix.plot(kind="barh", stacked=True, ax=ax,
             color=[config.PALETTE[0], config.PALETTE[1]])
    ax.set_title("Churn share within each contract type")
    ax.set_xlabel("Share of customers")
    ax.set_ylabel("")
    ax.legend(["Retained", "Churned"], loc="lower right")
    ax.set_xlim(0, 1)
    _save(fig, "08_contract_mix.png")


# --------------------------------------------------------------------------- #
# Summary tables
# --------------------------------------------------------------------------- #
def compute_summary(df: pd.DataFrame) -> dict:
    engineered = add_features(df)
    target = config.TARGET

    numeric_summary = {}
    for column in ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
                   "num_services", "service_density"]:
        series = engineered[column].astype(float)
        numeric_summary[column] = {
            key: round(float(value), 3)
            for key, value in {
                "mean": series.mean(), "std": series.std(), "min": series.min(),
                "p25": series.quantile(0.25), "median": series.median(),
                "p75": series.quantile(0.75), "max": series.max(),
            }.items()
        }

    rates = {
        column: {
            str(level): {
                "churn_rate": round(float(value), 4), "customers": int(count)
            }
            for level, (value, count) in (
                engineered.groupby(column, observed=True)[target]
                .agg(["mean", "size"]).iterrows()
            )
        }
        for column in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
    }

    correlation = (
        _numeric_view(df).corr(numeric_only=True)[target]
        .drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
    )
    top_correlations = [
        {"feature": name, "correlation": round(float(value), 4)}
        for name, value in correlation.head(10).items()
    ]

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "churn_rate": round(float(engineered[target].mean()), 4),
        "class_counts": {
            str(k): int(v) for k, v in engineered[target].value_counts().sort_index().items()
        },
        "numeric_summary": numeric_summary,
        "categorical_churn_rates": rates,
        "top_correlations_with_target": top_correlations,
        "median_tenure_by_churn": {
            str(int(k)): float(v) for k, v in engineered.groupby(target)["tenure"].median().items()
        },
        "median_monthly_charges_by_churn": {
            str(int(k)): round(float(v), 2)
            for k, v in engineered.groupby(target)["MonthlyCharges"].median().items()
        },
        "median_services_by_churn": {
            str(int(k)): int(v)
            for k, v in engineered.groupby(target)["num_services"].median().items()
        },
    }


def run_eda(df: pd.DataFrame) -> dict:
    """Generate every EDA artefact and persist the summary JSON."""
    summary = compute_summary(df)

    plot_target_balance(df)
    plot_numeric_distributions(df)
    plot_categorical_churn_rates(
        df,
        ["Contract", "InternetService", "PaymentMethod", "TechSupport",
         "OnlineSecurity", "PaperlessBilling"],
        "Churn rate by service and billing attributes",
        "03_categorical_churn_rates.png",
    )
    plot_categorical_churn_rates(
        df,
        ["gender", "SeniorCitizen", "Partner", "Dependents", "MultipleLines",
         "StreamingTV"],
        "Churn rate by customer demographics",
        "04_demographic_churn_rates.png",
    )
    plot_correlation_heatmap(df)
    plot_tenure_cohort(df)
    plot_spend_landscape(df)
    plot_contract_mix(df)

    summary_path = config.REPORT_DIR / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eda] summary -> {summary_path.relative_to(config.ROOT)}")
    return summary
