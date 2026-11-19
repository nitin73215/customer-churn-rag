"""Create a simple, interview-defendable drift report from prediction logs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_model.csv"
configured_log_path = Path(
    os.getenv("CHURN_PREDICTION_LOG_PATH", str(PROJECT_ROOT / "logs" / "predictions.csv"))
).expanduser()
PREDICTION_LOG_PATH = (
    configured_log_path
    if configured_log_path.is_absolute()
    else (PROJECT_ROOT / configured_log_path).resolve()
)
REPORT_PATH = PROJECT_ROOT / "reports" / "monitoring" / "drift_report.json"

# Demo thresholds: useful for portfolio monitoring, not universal production limits.
STANDARDIZED_MEAN_DIFFERENCE_THRESHOLD = 0.20
TOTAL_VARIATION_DISTANCE_THRESHOLD = 0.10
MINIMUM_MONITORING_SAMPLE_SIZE = 30

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]
PREDICTION_COLUMNS = [
    "churn_probability",
    "analytical_predicted_churn",
    "business_risk_segment",
]


def _as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _numeric_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        reference_values = pd.to_numeric(reference[feature], errors="coerce")
        current_values = pd.to_numeric(current[feature], errors="coerce")
        reference_mean = float(reference_values.mean())
        current_mean = float(current_values.mean())
        reference_std = float(reference_values.std(ddof=1))
        standardized_difference = (
            (current_mean - reference_mean) / reference_std
            if reference_std > 0
            else 0.0
        )
        report[feature] = {
            "reference_mean": reference_mean,
            "current_mean": current_mean,
            "reference_std": reference_std,
            "standardized_mean_difference": float(standardized_difference),
            "demo_drift_flag": bool(
                abs(standardized_difference) >= STANDARDIZED_MEAN_DIFFERENCE_THRESHOLD
            ),
            "missing_current": int(current_values.isna().sum()),
            "impossible_negative_values": int((current_values < 0).sum()),
        }
    return report


def _categorical_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    categorical_features = [
        column
        for column in reference.columns
        if column not in NUMERIC_FEATURES and column != "churn_value"
    ]
    report: dict[str, Any] = {}

    for feature in categorical_features:
        reference_values = reference[feature].dropna().astype(str)
        current_values = current[feature].dropna().astype(str)
        reference_distribution = reference_values.value_counts(normalize=True)
        current_distribution = current_values.value_counts(normalize=True)
        categories = sorted(set(reference_distribution.index) | set(current_distribution.index))
        total_variation_distance = 0.5 * sum(
            abs(
                float(reference_distribution.get(category, 0.0))
                - float(current_distribution.get(category, 0.0))
            )
            for category in categories
        )
        unseen = sorted(set(current_distribution.index) - set(reference_distribution.index))
        report[feature] = {
            "reference_distribution": {
                key: float(value) for key, value in reference_distribution.items()
            },
            "current_distribution": {
                key: float(value) for key, value in current_distribution.items()
            },
            "total_variation_distance": float(total_variation_distance),
            "unseen_categories": unseen,
            "demo_drift_flag": bool(
                total_variation_distance >= TOTAL_VARIATION_DISTANCE_THRESHOLD or unseen
            ),
            "missing_current": int(current[feature].isna().sum()),
        }
    return report


def build_report() -> dict[str, Any]:
    if not REFERENCE_DATA_PATH.is_file():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_DATA_PATH}")

    reference = pd.read_csv(REFERENCE_DATA_PATH)
    base_report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_rows": int(len(reference)),
        "prediction_log": str(PREDICTION_LOG_PATH),
        "demo_thresholds": {
            "absolute_standardized_mean_difference": STANDARDIZED_MEAN_DIFFERENCE_THRESHOLD,
            "total_variation_distance": TOTAL_VARIATION_DISTANCE_THRESHOLD,
            "minimum_sample_size": MINIMUM_MONITORING_SAMPLE_SIZE,
        },
    }

    if not PREDICTION_LOG_PATH.is_file() or PREDICTION_LOG_PATH.stat().st_size == 0:
        return {
            **base_report,
            "status": "no_prediction_data",
            "message": "No prediction log exists yet; make API predictions and rerun monitoring.",
            "current_rows": 0,
        }

    current = pd.read_csv(PREDICTION_LOG_PATH)
    required_features = [column for column in reference.columns if column != "churn_value"]
    missing_columns = [
        column for column in required_features + PREDICTION_COLUMNS if column not in current
    ]
    if missing_columns:
        return {
            **base_report,
            "status": "invalid_prediction_log",
            "current_rows": int(len(current)),
            "missing_columns": missing_columns,
        }

    numeric = _numeric_report(reference, current)
    categorical = _categorical_report(reference, current)
    drift_flags = [
        details["demo_drift_flag"]
        for details in [*numeric.values(), *categorical.values()]
    ]
    sample_status = (
        "insufficient_prediction_data"
        if len(current) < MINIMUM_MONITORING_SAMPLE_SIZE
        else ("drift_detected" if any(drift_flags) else "ok")
    )

    risk_distribution = (
        current["business_risk_segment"]
        .value_counts(normalize=True, dropna=False)
        .to_dict()
    )
    traffic_label_distribution = (
        current["traffic_label"].value_counts(normalize=True, dropna=False).to_dict()
        if "traffic_label" in current
        else {"unlabeled": 1.0}
    )
    prediction_summary = {
        "traffic_label_distribution": {
            str(key): float(value) for key, value in traffic_label_distribution.items()
        },
        "average_churn_probability": _as_float(
            pd.to_numeric(current["churn_probability"], errors="coerce").mean()
        ),
        "analytical_predicted_churn_rate": _as_float(
            pd.to_numeric(current["analytical_predicted_churn"], errors="coerce").mean()
        ),
        "risk_segment_distribution": {
            str(key): float(value) for key, value in risk_distribution.items()
        },
    }
    data_quality = {
        "total_missing_feature_values": int(current[required_features].isna().sum().sum()),
        "rows_with_any_missing_feature": int(current[required_features].isna().any(axis=1).sum()),
        "total_impossible_negative_numeric_values": int(
            sum(details["impossible_negative_values"] for details in numeric.values())
        ),
    }

    return {
        **base_report,
        "status": sample_status,
        "current_rows": int(len(current)),
        "sample_size_warning": (
            f"Drift flags are unstable below {MINIMUM_MONITORING_SAMPLE_SIZE} rows."
            if len(current) < MINIMUM_MONITORING_SAMPLE_SIZE
            else None
        ),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "predictions": prediction_summary,
        "data_quality": data_quality,
    }


def write_report(report: dict[str, Any]) -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return REPORT_PATH


def main() -> int:
    report = build_report()
    output_path = write_report(report)
    print(f"Monitoring status: {report['status']}")
    print(f"Current prediction rows: {report.get('current_rows', 0)}")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
