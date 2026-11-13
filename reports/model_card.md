# Model Card — Telco Churn Classifier

## Overview
| Field | Value |
| --- | --- |
| Version | 0.1.0 |
| Task | Binary classification (customer churn) |
| Algorithm | `logistic_regression` |
| Trained | 2026-11-13T12:35:23+00:00 |
| Training data | IBM Telco Customer Churn extract |
| Rows | 7,043 |
| Features | 30 engineered inputs |
| Licence | MIT (code); dataset per IBM sample terms |

## Intended use
Prioritise retention outreach: rank customers by churn probability and support
batch campaigns. **Not** intended for automated account closure, pricing decisions,
or any individual-level adverse action without human review.

## Performance (hold-out, 1,409 rows)
| Metric | Value |
| --- | --- |
| ROC AUC | 0.8448 |
| Average precision | 0.6519 |
| Accuracy | 0.7715 |
| Precision | 0.5539 |
| Recall | 0.7139 |
| F1 | 0.6238 |
| Brier score | 0.1660 |
| Decision threshold | 0.60 |

## Limitations
- Single-provider snapshot; no temporal or geographic hold-out.
- Association, not causation — no uplift estimate.
- Threshold selected for F1; campaigns optimising for precision or recall should
  re-tune it using the operating-point table in the report.

## Ethical considerations
- `gender` and `SeniorCitizen` are retained for benchmarking but are not actionable;
  outreach should not be segmented on protected attributes.
- Retraining cadence and drift monitoring are mandatory before production use.
