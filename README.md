# Telco Customer Churn — End-to-End ML Pipeline

[![CI](https://github.com/Manish27-tech/telco-churn-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Manish27-tech/telco-churn-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A production-shaped, reproducible pipeline that predicts customer churn on the
[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset
(7,043 customers, 21 raw attributes).

The goal is not just a model score — it is the *whole* workflow: validated ingestion,
documented cleaning, exploratory analysis, leak-free feature engineering, model
comparison, tuning, threshold selection, reporting and automated tests.

---

## Results

Hold-out test set: **1,409 customers**, never seen during training or tuning.

| Metric | Value |
| --- | --- |
| ROC AUC | **0.845** |
| Average precision | **0.652** |
| F1 | **0.624** |
| Precision | 0.554 |
| Recall | 0.714 |
| Accuracy | 0.781 |
| Brier score | 0.160 |
| Decision threshold | 0.60 |

Cross-validation comparison (5-fold, training split):

| Model | ROC AUC | Avg. precision | F1 | Recall |
| --- | --- | --- | --- | --- |
| Logistic regression (tuned) | **0.848** | 0.660 | 0.632 | 0.801 |
| Gradient boosting | 0.846 | 0.661 | 0.581 | 0.518 |
| Random forest | 0.841 | 0.642 | 0.625 | 0.671 |

A regularised linear model wins here: after one-hot encoding, contract type, tenure and
support/security add-ons carry most of the signal, and the boosted trees mostly relearn
those monotone effects with more variance.

The full write-up with figures is in **[reports/REPORT.md](reports/REPORT.md)**.

---

## Pipeline

```
download  →  validate  →  clean  →  feature engineering  →  EDA
                                                              ↓
        report  ←  threshold selection  ←  tuning  ←  model comparison
                                                              ↓
                                                      evaluation + figures
```

Each stage is a module that can be imported on its own:

| Module | Responsibility |
| --- | --- |
| `src/churn/data.py` | Download with fallbacks, cache, schema validation (9 checks) |
| `src/churn/clean.py` | Type coercion, duplicate removal, documented imputation |
| `src/churn/features.py` | Revenue, service-footprint and contract-risk features |
| `src/churn/eda.py` | Summary statistics + 8 publication-ready figures |
| `src/churn/model.py` | Preprocessing, CV comparison, tuning, thresholding, evaluation |
| `src/churn/report.py` | Markdown report and model card generation |
| `src/churn/cli.py` | `download` / `eda` / `train` / `all` / `score` commands |

---

## Quickstart

```bash
git clone https://github.com/Manish27-tech/telco-churn-pipeline.git
cd telco-churn-pipeline
pip install -r requirements.txt

python run.py all        # download → EDA → train → reports (≈1 min)
python run.py eda        # analysis only
python run.py train      # modelling only
python run.py score --input data/raw/Telco-Customer-Churn.csv --output predictions.csv
```

`run.py all` regenerates everything in `reports/` — figures, metrics, report and model
card — from the raw CSV. Nothing is hand-edited.

Notebook (executed, outputs committed):

```bash
pip install -r requirements-dev.txt
python scripts/build_notebook.py
```

---

## Dataset and cleaning

The single data-quality defect is `TotalCharges`: it is stored as text and blank for the
11 customers with `tenure == 0`. Those accounts are genuinely zero-billing, so they are
imputed with `0.0` rather than dropped — dropping them would bias the model against the
highest-risk tenure cohort.

Everything else is validated before modelling:

```
[PASS] expected_columns          all source columns present
[PASS] unique_customer_id        7043 unique ids for 7043 rows
[PASS] target_domain             observed values: ['No', 'Yes']
[PASS] tenure_range              min=0, max=72
[PASS] monthly_charges_positive  min=18.25, max=118.75
```

---

## What the analysis finds

1. **Contract type dominates.** Month-to-month customers churn at **42.7%**, versus
   11.3% (one year) and 2.8% (two year).
2. **Churn is front-loaded.** 0–6 month customers churn at **53%**, falling to 7% after
   five years. First-year onboarding is the highest-leverage retention window.
3. **Fiber optic + no support add-ons** is the high-risk combination: 42% churn for
   fiber customers, with tech-support subscribers churning at only 15%.
4. **Billing friction matters.** Electronic-check payers churn at 45%; automatic payment
   customers churn at 15–17%.

---

## Design decisions

- **No leakage.** Imputation, scaling and one-hot encoding live inside a
  `ColumnTransformer` inside the `Pipeline`, so they are refit on every CV fold.
- **Class imbalance handled honestly.** `class_weight="balanced"` plus threshold tuning
  instead of SMOTE — resampling distorts predicted probabilities, which the retention
  team needs to rank customers.
- **Threshold chosen on out-of-fold predictions**, not on the test set, and reported
  alongside a precision/recall operating-point table so the business can move it.
- **Validation gate.** A corrupted or reordered download fails loudly instead of
  silently producing a plausible-looking report.
- **Deterministic.** Every split, search and model is seeded (`RANDOM_STATE = 42`).

---

## Project structure

```
telco-churn-pipeline/
├── src/churn/            # importable pipeline package
├── tests/                # 22 unit tests (pytest)
├── notebooks/            # executed EDA + modelling notebook
├── scripts/              # notebook builder
├── data/raw/             # committed source CSV for offline reproducibility
├── reports/              # REPORT.md, metrics.json, model_card.md, figures/
├── .github/workflows/    # CI: lint + tests + end-to-end pipeline
├── run.py                # zero-install entry point
└── pyproject.toml
```

---

## Testing

```bash
pip install -r requirements-dev.txt
ruff check src tests scripts
pytest -q
```

CI runs lint, the test suite on Python 3.11 and 3.12, then executes the full pipeline and
uploads the generated reports as build artefacts.

---

## Data & licence

Dataset: IBM Telco Customer Churn sample extract (public). Code: MIT — see
[LICENSE](LICENSE).
