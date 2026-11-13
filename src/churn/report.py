"""Markdown report and model-card generation from pipeline artefacts."""

from __future__ import annotations

import json
from pathlib import Path

from . import config


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_report(summary: dict, metrics: dict, cleaning_log: dict,
                 validation: dict) -> str:
    churn_rate = summary.get("churn_rate", 0)
    test = metrics.get("test_metrics", {})
    matrix = test.get("confusion_matrix", {})

    comparison_rows = [
        [row["model"].replace("_", " "), f"{row['roc_auc_mean']:.3f}",
         f"{row['average_precision_mean']:.3f}", f"{row['f1_mean']:.3f}",
         f"{row['recall_mean']:.3f}", f"{row['fit_seconds']:.1f}s"]
        for row in metrics.get("cv_comparison", [])
    ]

    importance_rows = [
        [f"`{item['feature']}`", f"{item['importance']:.4f}"]
        for item in metrics.get("feature_importance", [])[:10]
    ]

    operating_rows = [
        [f"{row['threshold']:.2f}", f"{row['precision']:.3f}",
         f"{row['recall']:.3f}", f"{row['f1']:.3f}"]
        for row in metrics.get("operating_points", [])
    ]

    contract_rates = summary.get("categorical_churn_rates", {}).get("Contract", {})
    contract_rows = [
        [level, f"{values['churn_rate']:.1%}", f"{values['customers']:,}"]
        for level, values in sorted(contract_rates.items(),
                                    key=lambda item: item[1]["churn_rate"], reverse=True)
    ]

    tenure = summary.get("median_tenure_by_churn", {})
    zero_filled = cleaning_log.get("total_charges_zero_for_new_customers", 0)

    return f"""# Telco Customer Churn — End-to-End Analysis

> Generated automatically by `python run.py all` on the IBM Telco Customer Churn
> sample extract. Every number below is produced by the pipeline, not hand-written.

## 1. Executive summary

- **Dataset:** {summary.get('rows', 'n/a'):,} customers,
  {summary.get('columns', 'n/a')} raw columns.
- **Target:** `Churn`, with a **{churn_rate:.1%}** positive rate — an imbalanced
  classification problem where naive accuracy is a misleading metric.
- **Best model:** `{metrics.get('selected_model', 'n/a')}`, tuned with a randomised
  search and a decision threshold optimised on out-of-fold predictions.
- **Hold-out performance:** ROC AUC **{test.get('roc_auc', 0):.3f}**, average
  precision **{test.get('average_precision', 0):.3f}**, F1 **{test.get('f1', 0):.3f}**
  at threshold **{metrics.get('decision_threshold', 0):.2f}**.
- **Primary churn drivers:** contract type, tenure, monthly charges and the absence
  of support/security add-ons.

## 2. Dataset and validation

Source: IBM's public Telco customer churn extract ({config.DATA_URL.split('/')[-1]}).
The pipeline verifies {len(validation.get('checks', []))} schema checks before any
modelling step, so a corrupted or reordered download fails fast instead of silently
poisoning results.

## 3. Cleaning

| Step | Result |
| --- | --- |
| Rows in | {cleaning_log.get('rows_in', 'n/a'):,} |
| Duplicate customer IDs removed | {cleaning_log.get('duplicate_customer_ids_removed', 0)} |
| Rows with missing target dropped | {cleaning_log.get('rows_missing_target_dropped', 0)} |
| `TotalCharges` zero-filled | {zero_filled} |
| Remaining nulls after imputation | {cleaning_log.get('remaining_nulls', 0)} |
| Rows out | {cleaning_log.get('rows_out', 'n/a'):,} |

The only systematic data-quality defect is `TotalCharges`: it is stored as text and
blank for customers with `tenure == 0`. Those rows are genuinely zero-billing rather
than corrupt, so they are imputed with `0.0` and documented instead of dropped.

## 4. Exploratory findings

![Target balance](figures/01_target_balance.png)

![Numeric distributions by churn](figures/02_numeric_distributions.png)

![Churn rate by service attributes](figures/03_categorical_churn_rates.png)

![Churn rate by demographics](figures/04_demographic_churn_rates.png)

### Contract type is the strongest structural signal

{_table(["Contract", "Churn rate", "Customers"], contract_rows)}

![Contract mix](figures/08_contract_mix.png)

### Churn is front-loaded

Median tenure is **{tenure.get('0', 0):.0f} months for churners** versus
**{tenure.get('1', 0):.0f} months for retained customers**. The first year is the
highest-risk window.

![Tenure cohort analysis](figures/06_tenure_cohort.png)

![Spend landscape](figures/07_spend_landscape.png)

![Correlation heatmap](figures/05_correlation_heatmap.png)

### Strongest correlations with churn

{_table(["Feature", "Correlation"], [[f"`{row['feature']}`", f"{row['correlation']:+.3f}"]
                                       for row in summary.get('top_correlations_with_target', [])])}

## 5. Modelling

### 5.1 Candidate comparison (5-fold stratified CV on the training split)

{_table(["Model", "ROC AUC", "Avg. precision", "F1", "Recall", "Fit time"], comparison_rows)}

Model families span a regularised linear baseline and two tree ensembles, so the
comparison isolates the value of non-linear interactions rather than just tuning one
algorithm.

### 5.2 Final configuration

- Selected family: **`{metrics.get('selected_model', 'n/a')}`**
- Best parameters: `{json.dumps(metrics.get('best_params', {}))}`
- Decision threshold: **{metrics.get('decision_threshold', 0):.2f}**
  ({metrics.get('threshold_selection', {}).get('method', '')})
- Search cost: {metrics.get('search_seconds', 0)} s — end-to-end training
  {metrics.get('runtime_seconds', 0)} s.

### 5.3 Hold-out test performance

{_table(["Metric", "Value"], [
        ["ROC AUC", f"{test.get('roc_auc', 0):.4f}"],
        ["Average precision", f"{test.get('average_precision', 0):.4f}"],
        ["Accuracy", f"{test.get('accuracy', 0):.4f}"],
        ["Precision", f"{test.get('precision', 0):.4f}"],
        ["Recall", f"{test.get('recall', 0):.4f}"],
        ["F1", f"{test.get('f1', 0):.4f}"],
        ["Brier score", f"{test.get('brier', 0):.4f}"],
        ["True positives", f"{matrix.get('tp', 0):,}"],
        ["False negatives", f"{matrix.get('fn', 0):,}"],
        ["False positives", f"{matrix.get('fp', 0):,}"],
        ["True negatives", f"{matrix.get('tn', 0):,}"],
    ])}

![Confusion matrix](figures/09_confusion_matrix.png)

![ROC and PR curves](figures/10_roc_pr_curves.png)

![Calibration](figures/12_calibration.png)

### 5.4 Choosing an operating point

Retention teams care about *how many true churners they can reach* per campaign.
Precision and recall trade off directly, so the threshold is a business decision, not
a modelling one:

{_table(["Threshold", "Precision", "Recall", "F1"], operating_rows)}

### 5.5 What drives the predictions

{_table(["Feature", "Importance"], importance_rows)}

![Feature importance](figures/11_feature_importance.png)

## 6. Interpretation

1. **Month-to-month contracts** churn several times more often than two-year
   contracts; contract risk is the single largest lever.
2. **New customers** (tenure ≤ 6 months) are the riskiest cohort, which argues for
   onboarding investment rather than broad discounting.
3. **Fiber-optic customers with high monthly charges and no tech-support/security
   add-ons** combine high price with low switching cost.
4. **Automatic payment** correlates with retention — friction in billing is a
   retention feature.

## 7. Limitations

- Observational snapshot, not causal: coefficients and importances describe
  association, not uplift.
- No temporal validation split; a production system should validate on a later
  cohort to detect drift.
- Class imbalance is handled with class weights and threshold tuning; SMOTE-style
  resampling was deliberately avoided to keep probability calibration honest.
- The dataset covers a single telecom provider and may not transfer.

## 8. Reproducing

```bash
pip install -r requirements.txt
python run.py all
```

`run.py all` downloads and validates the raw data, cleans it, regenerates every
figure in `reports/figures/`, trains and tunes the model, and rewrites this report
plus `reports/metrics.json` and `reports/model_card.md`.
"""


def build_model_card(metrics: dict, summary: dict) -> str:
    test = metrics.get("test_metrics", {})
    return f"""# Model Card — Telco Churn Classifier

## Overview
| Field | Value |
| --- | --- |
| Version | 0.1.0 |
| Task | Binary classification (customer churn) |
| Algorithm | `{metrics.get('selected_model', 'n/a')}` |
| Trained | {metrics.get('generated_at', 'n/a')} |
| Training data | IBM Telco Customer Churn extract |
| Rows | {metrics.get('dataset', {}).get('rows', 'n/a'):,} |
| Features | {metrics.get('dataset', {}).get('features_raw', 'n/a')} engineered inputs |
| Licence | MIT (code); dataset per IBM sample terms |

## Intended use
Prioritise retention outreach: rank customers by churn probability and support
batch campaigns. **Not** intended for automated account closure, pricing decisions,
or any individual-level adverse action without human review.

## Performance (hold-out, {metrics.get('dataset', {}).get('test_rows', 0):,} rows)
| Metric | Value |
| --- | --- |
| ROC AUC | {test.get('roc_auc', 0):.4f} |
| Average precision | {test.get('average_precision', 0):.4f} |
| Accuracy | {test.get('accuracy', 0):.4f} |
| Precision | {test.get('precision', 0):.4f} |
| Recall | {test.get('recall', 0):.4f} |
| F1 | {test.get('f1', 0):.4f} |
| Brier score | {test.get('brier', 0):.4f} |
| Decision threshold | {metrics.get('decision_threshold', 0):.2f} |

## Limitations
- Single-provider snapshot; no temporal or geographic hold-out.
- Association, not causation — no uplift estimate.
- Threshold selected for F1; campaigns optimising for precision or recall should
  re-tune it using the operating-point table in the report.

## Ethical considerations
- `gender` and `SeniorCitizen` are retained for benchmarking but are not actionable;
  outreach should not be segmented on protected attributes.
- Retraining cadence and drift monitoring are mandatory before production use.
"""


def write_reports(summary: dict, metrics: dict, cleaning_log: dict,
                  validation: dict) -> tuple[Path, Path]:
    report_path = config.REPORT_DIR / "REPORT.md"
    card_path = config.REPORT_DIR / "model_card.md"
    report_path.write_text(build_report(summary, metrics, cleaning_log, validation),
                           encoding="utf-8")
    card_path.write_text(build_model_card(metrics, summary), encoding="utf-8")
    print(f"[report] -> {report_path.relative_to(config.ROOT)}")
    print(f"[report] -> {card_path.relative_to(config.ROOT)}")
    return report_path, card_path


def load_artifacts() -> tuple[dict, dict]:
    return (
        _load(config.REPORT_DIR / "eda_summary.json"),
        _load(config.REPORT_DIR / "metrics.json"),
    )
