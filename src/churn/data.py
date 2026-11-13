"""Dataset acquisition, loading and schema validation."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from . import config


class DataValidationError(ValueError):
    """Raised when the raw dataset does not satisfy the expected schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False, verbose: bool = True) -> Path:
    """Download the raw CSV, trying the primary URL then any fallbacks.

    Returns the local path. The file is cached, so repeat runs are offline-safe.
    """
    destination = config.RAW_DIR / config.RAW_FILENAME
    if destination.exists() and not force:
        if verbose:
            print(f"[data] cached raw file: {destination.name} "
                  f"({destination.stat().st_size / 1024:.0f} KB)")
        return destination

    errors: list[str] = []
    for url in [config.DATA_URL, *config.DATA_FALLBACK_URLS]:
        try:
            if verbose:
                print(f"[data] downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if len(payload) < 10_000:
                raise ValueError(f"payload too small ({len(payload)} bytes)")
            destination.write_bytes(payload)
            if verbose:
                print(f"[data] saved {destination} "
                      f"(sha256={_sha256(destination)[:12]}...)")
            return destination
        except Exception as exc:  # noqa: BLE001 - we aggregate and report
            errors.append(f"{url} -> {exc}")
            if verbose:
                print(f"[data] failed: {exc}")

    raise RuntimeError("Could not download the dataset:\n  " + "\n  ".join(errors))


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV exactly as shipped (no type coercion)."""
    path = path or (config.RAW_DIR / config.RAW_FILENAME)
    if not path.exists():
        path = download(verbose=False)
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> dict:
    """Run schema and sanity checks, raising on critical failures.

    Returns a serialisable report describing every check performed.
    """
    checks: list[dict] = []
    critical_failures: list[str] = []

    def record(name: str, passed: bool, detail: str, critical: bool = False) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail,
                       "critical": critical})
        if not passed and critical:
            critical_failures.append(name)

    missing_cols = [c for c in config.SOURCE_COLUMNS if c not in df.columns]
    record("expected_columns", not missing_cols,
           "all source columns present" if not missing_cols else f"missing: {missing_cols}",
           critical=True)

    record("non_empty", len(df) > 0, f"{len(df)} rows", critical=True)
    record("unique_customer_id", df[config.ID_COLUMN].is_unique,
           f"{df[config.ID_COLUMN].nunique()} unique ids for {len(df)} rows",
           critical=True)

    target_values = set(df[config.TARGET].dropna().unique())
    record("target_domain", target_values <= {"Yes", "No"},
           f"observed values: {sorted(map(str, target_values))}")

    missing_target = int(df[config.TARGET].isna().sum())
    record("target_not_null", missing_target == 0, f"{missing_target} missing targets")

    age_domain = set(df["SeniorCitizen"].dropna().unique())
    record("senior_citizen_domain", age_domain <= {0, 1},
           f"observed values: {sorted(map(int, age_domain)) if age_domain else []}")

    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    record("tenure_range", tenure.between(0, 100).all(),
           f"min={tenure.min()}, max={tenure.max()}")

    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    record("monthly_charges_positive", (monthly > 0).all(),
           f"min={monthly.min():.2f}, max={monthly.max():.2f}")

    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    non_numeric_total = int(total.isna().sum() - df["TotalCharges"].isna().sum())
    record("total_charges_numericish", non_numeric_total >= 0,
           f"{non_numeric_total} non-numeric values coerced to NaN")

    if critical_failures:
        raise DataValidationError(
            "Dataset failed critical validation: " + ", ".join(critical_failures)
        )

    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "checks": checks,
        "sha256": _sha256(config.RAW_DIR / config.RAW_FILENAME)
        if (config.RAW_DIR / config.RAW_FILENAME).exists() else None,
    }
    df.attrs["validation"] = report
    return report
