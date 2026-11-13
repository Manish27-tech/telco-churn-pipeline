"""Command-line interface for the churn pipeline."""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from . import config, data, eda, model, report
from .clean import clean, save_processed


def _banner(step: str) -> None:
    print(f"\n{'=' * 72}\n {step}\n{'=' * 72}")


def _prepare(verbose: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    """Download, validate and clean in one call."""
    path = data.download(verbose=verbose)
    raw = data.load_raw(path)
    validation = data.validate(raw)
    if verbose:
        print(f"[data] validation passed ({len(validation['checks'])} checks, "
              f"{validation['rows']:,} rows)")
    cleaned, cleaning_log = clean(raw)
    if verbose:
        print(f"[clean] {cleaning_log['rows_in']:,} -> {cleaning_log['rows_out']:,} rows "
              f"| churn rate {cleaning_log['churn_rate']:.1%}")
    return cleaned, cleaning_log, validation


def cmd_download(args: argparse.Namespace) -> int:
    _banner("STEP 1/3 - download & validate")
    raw = data.load_raw(data.download(force=args.force))
    validation = data.validate(raw)
    for check in validation["checks"]:
        status = "PASS" if check["passed"] else ("FAIL" if check["critical"] else "warn")
        print(f"  [{status:>4}] {check['check']:<28} {check['detail']}")
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    _banner("STEP 2/3 - exploratory data analysis")
    cleaned, _, _ = _prepare()
    save_processed(cleaned)
    eda.run_eda(cleaned)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    _banner("STEP 3/3 - model training & evaluation")
    cleaned, cleaning_log, validation = _prepare()
    save_processed(cleaned)
    summary = eda.compute_summary(cleaned)
    metrics = model.train(cleaned, seed=config.RANDOM_STATE)
    report.write_reports(summary, metrics, cleaning_log, validation)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    cleaned, cleaning_log, validation = _prepare()
    save_processed(cleaned)

    _banner("STEP 2/3 - exploratory data analysis")
    summary = eda.run_eda(cleaned)

    _banner("STEP 3/3 - model training & evaluation")
    metrics = model.train(cleaned, seed=config.RANDOM_STATE)
    report.write_reports(summary, metrics, cleaning_log, validation)

    print(f"\n[done] end-to-end pipeline finished in {time.perf_counter() - started:.1f}s")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    bundle = model.load_model()
    frame = pd.read_csv(args.input)
    probabilities = model.predict(frame, bundle)
    output = pd.DataFrame({
        "row_id": range(len(frame)),
        "churn_probability": probabilities.round(4),
        "churn_prediction": (probabilities >= bundle["threshold"]).astype(int),
    })
    output.to_csv(args.output, index=False)
    print(f"[score] {len(output):,} rows -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churn", description="Telco customer churn end-to-end pipeline"
    )
    parser.add_argument("--verbose", action="store_true", help="extra logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download and validate raw data")
    download.add_argument("--force", action="store_true", help="re-download")
    download.set_defaults(func=cmd_download)

    eda_parser = subparsers.add_parser("eda", help="run exploratory analysis")
    eda_parser.set_defaults(func=cmd_eda)

    train_parser = subparsers.add_parser("train", help="train, tune and evaluate")
    train_parser.set_defaults(func=cmd_train)

    all_parser = subparsers.add_parser("all", help="run the full pipeline")
    all_parser.set_defaults(func=cmd_all)

    score_parser = subparsers.add_parser("score", help="score a CSV with the saved model")
    score_parser.add_argument("--input", required=True, help="input CSV path")
    score_parser.add_argument("--output", default="predictions.csv", help="output CSV path")
    score_parser.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI failure
        print(f"\n[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
