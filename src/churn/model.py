"""Model training, hyper-parameter search, evaluation and persistence."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import loguniform, randint  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from . import config  # noqa: E402
from .features import add_features, build_feature_matrix, feature_columns  # noqa: E402

plt.rcParams["figure.autolayout"] = True


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                              sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=config.RANDOM_STATE,
        ),
    }


SEARCH_SPACES: dict[str, dict] = {
    "logistic_regression": {
        "model__C": loguniform(1e-2, 1e2),
        "model__solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "model__n_estimators": randint(300, 800),
        "model__max_depth": [None, 6, 10, 14],
        "model__min_samples_leaf": randint(1, 9),
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__class_weight": ["balanced", "balanced_subsample"],
    },
    "gradient_boosting": {
        "model__n_estimators": randint(200, 500),
        "model__learning_rate": loguniform(0.02, 0.2),
        "model__max_depth": randint(2, 5),
        "model__subsample": [0.7, 0.85, 1.0],
        "model__min_samples_leaf": randint(1, 6),
    },
}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def compare_models(X_train, y_train) -> tuple[object, list[dict]]:
    """Cross-validate every candidate model on the training split."""
    numeric, categorical = feature_columns()
    cross_validation = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    rows: list[dict] = []
    print("[model] cross-validating candidate models")
    for name, estimator in candidate_models().items():
        pipeline = Pipeline(
            [("preprocess", build_preprocessor(numeric, categorical)),
             ("model", estimator)]
        )
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cross_validation, scoring=scoring,
            n_jobs=-1, error_score="raise",
        )
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = round(float(np.mean(scores[f"test_{metric}"])), 4)
            row[f"{metric}_std"] = round(float(np.std(scores[f"test_{metric}"])), 4)
        row["fit_seconds"] = round(float(np.mean(scores["fit_time"])), 2)
        rows.append(row)
        print(f"[model]   {name:<20} roc_auc={row['roc_auc_mean']:.4f} "
              f"f1={row['f1_mean']:.4f} recall={row['recall_mean']:.4f}")

    rows.sort(key=lambda item: item["roc_auc_mean"], reverse=True)
    return rows[0]["model"], rows


def tune_model(name: str, X_train, y_train) -> tuple[Pipeline, dict, float]:
    """Randomised hyper-parameter search for the winning model family."""
    numeric, categorical = feature_columns()
    pipeline = Pipeline(
        [("preprocess", build_preprocessor(numeric, categorical)),
         ("model", candidate_models()[name])]
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=SEARCH_SPACES[name],
        n_iter=config.SEARCH_ITER,
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=config.SEARCH_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
        ),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    print(f"[model] tuned {name}: cv roc_auc={search.best_score_:.4f} "
          f"({elapsed:.1f}s, {config.SEARCH_ITER} configs)")
    best_params = {
        key.replace("model__", ""): (value.item() if hasattr(value, "item") else value)
        for key, value in search.best_params_.items()
    }
    return search.best_estimator_, best_params, elapsed


def select_threshold(pipeline: Pipeline, X_train, y_train) -> tuple[float, list[dict]]:
    """Pick the probability cut-off that maximises F1 on out-of-fold predictions."""
    folds = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    oof = np.zeros(len(y_train))
    for train_index, valid_index in folds.split(X_train, y_train):
        fold_model = clone(pipeline)
        fold_model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        oof[valid_index] = fold_model.predict_proba(X_train.iloc[valid_index])[:, 1]

    grid = np.round(np.arange(0.10, 0.91, 0.01), 2)
    scores = [
        {"threshold": float(t), "f1": float(f1_score(y_train, oof >= t)),
         "precision": float(precision_score(y_train, oof >= t, zero_division=0)),
         "recall": float(recall_score(y_train, oof >= t, zero_division=0))}
        for t in grid
    ]
    best = max(scores, key=lambda row: row["f1"])
    print(f"[model] decision threshold={best['threshold']:.2f} "
          f"(oof f1={best['f1']:.3f}, recall={best['recall']:.3f})")
    return best["threshold"], scores


def _metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def feature_importance(pipeline: Pipeline, top_n: int = 20) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:  # pragma: no cover - all candidates expose one of the two
        return []

    order = np.argsort(values)[::-1][:top_n]
    return [
        {"feature": names[index].split("__", 1)[-1], "importance": round(float(values[index]), 5)}
        for index in order
    ]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(config.FIGURE_DIR / filename, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(metrics: dict) -> None:
    matrix = metrics["test_metrics"]["confusion_matrix"]
    array = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(array, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ threshold {metrics['decision_threshold']:.2f}")
    _save(fig, "09_confusion_matrix.png")


def plot_roc_pr(y_test, probabilities, metrics: dict) -> None:
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color=config.PALETTE[1], linewidth=2,
                 label=f"AUC = {metrics['test_metrics']['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color=config.PALETTE[1], linewidth=2,
                 label=f"AP = {metrics['test_metrics']['average_precision']:.3f}")
    axes[1].axvline(metrics["test_metrics"]["recall"], linestyle=":", color="black",
                    linewidth=1, label="operating point")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend(loc="lower left")
    _save(fig, "10_roc_pr_curves.png")


def plot_feature_importance(importances: list[dict]) -> None:
    frame = pd.DataFrame(importances).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.barplot(data=frame, x="importance", y="feature", hue="feature",
                legend=False, palette="crest", ax=ax)
    ax.set_title("Top drivers of churn probability")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    _save(fig, "11_feature_importance.png")


def plot_calibration(y_test, probabilities) -> None:
    fraction_positive, mean_predicted = calibration_curve(
        y_test, probabilities, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(mean_predicted, fraction_positive, marker="o", linewidth=2,
            color=config.PALETTE[1], label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration (quantile bins)")
    ax.legend(loc="upper left")
    _save(fig, "12_calibration.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train(df, seed: int = config.RANDOM_STATE) -> dict:
    """Run the full modelling workflow and persist metrics, figures and model."""
    started = time.perf_counter()
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed
    )
    print(f"[model] train={len(X_train):,} rows | test={len(X_test):,} rows | "
          f"features={X.shape[1]} | churn rate={y.mean():.3f}")

    best_name, comparison = compare_models(X_train, y_train)
    pipeline, best_params, search_seconds = tune_model(best_name, X_train, y_train)
    threshold, threshold_curve = select_threshold(pipeline, X_train, y_train)

    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = _metrics(y_test, probabilities, threshold)

    interesting_thresholds = [0.30, 0.40, round(threshold, 2), 0.50, 0.60]
    operating_points = []
    for candidate in sorted(set(interesting_thresholds)):
        row = _metrics(y_test, probabilities, candidate)
        operating_points.append({
            "threshold": candidate,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
        })

    importances = feature_importance(pipeline)
    runtime = round(time.perf_counter() - started, 1)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "rows": int(len(df)),
            "features_raw": int(X.shape[1]),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "churn_rate": round(float(y.mean()), 4),
        },
        "cv_comparison": comparison,
        "selected_model": best_name,
        "best_params": best_params,
        "search_seconds": round(search_seconds, 1),
        "decision_threshold": round(float(threshold), 2),
        "threshold_selection": {
            "method": "maximise F1 on 5-fold out-of-fold probabilities",
            "curve": threshold_curve[::5],
        },
        "test_metrics": test_metrics,
        "operating_points": operating_points,
        "feature_importance": importances,
        "runtime_seconds": runtime,
    }

    # Figures ------------------------------------------------------------- #
    plot_confusion_matrix(metrics)
    plot_roc_pr(y_test, probabilities, metrics)
    plot_feature_importance(importances)
    plot_calibration(y_test, probabilities)

    # Persist ------------------------------------------------------------- #
    metrics_path = config.REPORT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bundle = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "features": feature_columns(),
        "metrics": metrics,
        "trained_at": metrics["generated_at"],
    }
    model_path = config.ARTIFACT_DIR / "churn_model.joblib"
    joblib.dump(bundle, model_path)

    print(f"[model] test roc_auc={test_metrics['roc_auc']:.4f} "
          f"f1={test_metrics['f1']:.4f} recall={test_metrics['recall']:.4f} "
          f"precision={test_metrics['precision']:.4f}")
    print(f"[model] metrics -> {metrics_path.relative_to(config.ROOT)}")
    print(f"[model] model   -> {model_path.relative_to(config.ROOT)}")
    return metrics


def load_model(path=None) -> dict:
    path = path or (config.ARTIFACT_DIR / "churn_model.joblib")
    return joblib.load(path)


def predict(df, bundle: dict | None = None) -> np.ndarray:
    """Score a cleaned dataframe, returning churn probabilities.

    Works with unlabelled data: only the trained feature columns are selected.
    """
    bundle = bundle or load_model()
    numeric, categorical = bundle["features"]
    engineered = add_features(df)
    return bundle["pipeline"].predict_proba(engineered[numeric + categorical])[:, 1]
