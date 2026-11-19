"""FastAPI application for serving the calibrated churn model."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal
from uuid import uuid4

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
from fastapi import FastAPI, Header, HTTPException

from api.schemas import CustomerInput, HealthResponse, PredictionResponse


LOGGER = logging.getLogger("customer_churn_api")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_tuned_churn_model.joblib"
THRESHOLD_POLICY_PATH = PROJECT_ROOT / "models" / "business_threshold_policy.json"
MLFLOW_DATABASE_PATH = PROJECT_ROOT / "mlops" / "mlflow.db"
REGISTERED_MODEL_NAME = "telco_churn_calibrated_xgboost"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

MODEL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
]


def _runtime_log_path() -> Path:
    configured = os.getenv("CHURN_PREDICTION_LOG_PATH")
    if not configured:
        return PROJECT_ROOT / "logs" / "predictions.csv"

    configured_path = Path(configured).expanduser()
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path
    return configured_path.resolve()


PREDICTION_LOG_PATH = _runtime_log_path()
DEFAULT_TRAFFIC_LABEL = os.getenv("CHURN_TRAFFIC_LABEL", "production")
PREDICTION_LOG_LOCK = Lock()
LEGACY_TRAFFIC_LABEL = "legacy_unlabeled"


@dataclass(frozen=True)
class ThresholdPolicy:
    analytical: float
    campaign: float
    high_risk: float

    def risk_segment(self, probability: float) -> str:
        if probability >= self.high_risk:
            return "High Risk"
        if probability >= self.campaign:
            return "Medium Risk"
        if probability >= self.analytical:
            return "Watchlist"
        return "Low Risk"


@dataclass(frozen=True)
class ModelRuntime:
    model: Any
    source: Literal["mlflow_champion", "joblib_fallback"]
    alias: str | None
    version: str | None


def _load_threshold_policy() -> ThresholdPolicy:
    if not THRESHOLD_POLICY_PATH.is_file():
        raise RuntimeError(f"Threshold policy not found: {THRESHOLD_POLICY_PATH}")

    with THRESHOLD_POLICY_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    required = (
        "analytical_threshold",
        "campaign_threshold",
        "high_risk_threshold",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise RuntimeError(f"Threshold policy is missing keys: {missing}")

    policy = ThresholdPolicy(
        analytical=float(metadata["analytical_threshold"]),
        campaign=float(metadata["campaign_threshold"]),
        high_risk=float(metadata["high_risk_threshold"]),
    )
    if not 0 <= policy.analytical <= policy.campaign <= policy.high_risk <= 1:
        raise RuntimeError("Threshold policy values are invalid or out of order.")
    return policy


def _tracking_uri() -> str:
    return "sqlite:///" + MLFLOW_DATABASE_PATH.resolve().as_posix()


def _validate_model_features(model: Any) -> None:
    feature_names = list(getattr(model, "feature_names_in_", []))
    if feature_names and feature_names != MODEL_FEATURES:
        raise RuntimeError(
            "Loaded model feature order does not match the API contract. "
            f"Expected {MODEL_FEATURES}; received {feature_names}."
        )


def _load_model_runtime() -> ModelRuntime:
    """Prefer the champion registry alias and explicitly log any fallback."""

    try:
        if not MLFLOW_DATABASE_PATH.is_file():
            raise FileNotFoundError(
                f"MLflow tracking database not found: {MLFLOW_DATABASE_PATH}"
            )

        tracking_uri = _tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        model_version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            MODEL_ALIAS,
        )
        loaded_model = mlflow.sklearn.load_model(MODEL_URI)
        _validate_model_features(loaded_model)
        LOGGER.info(
            "Loaded model_source=mlflow_champion alias=%s version=%s",
            MODEL_ALIAS,
            model_version.version,
        )
        return ModelRuntime(
            model=loaded_model,
            source="mlflow_champion",
            alias=MODEL_ALIAS,
            version=str(model_version.version),
        )
    except Exception as registry_error:
        LOGGER.warning(
            "MLflow champion unavailable (%s: %s); attempting explicit joblib fallback.",
            type(registry_error).__name__,
            registry_error,
        )

        if not MODEL_PATH.is_file():
            raise RuntimeError(
                "Neither the MLflow champion nor the joblib fallback is available."
            ) from registry_error

        try:
            loaded_model = joblib.load(MODEL_PATH)
            _validate_model_features(loaded_model)
        except Exception as fallback_error:
            raise RuntimeError("The joblib fallback model could not be loaded.") from fallback_error

        LOGGER.warning("Loaded model_source=joblib_fallback path=%s", MODEL_PATH)
        return ModelRuntime(
            model=loaded_model,
            source="joblib_fallback",
            alias=None,
            version=None,
        )


THRESHOLDS = _load_threshold_policy()
MODEL_RUNTIME = _load_model_runtime()

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Serve calibrated churn probabilities and existing business-policy decisions.",
    version="2.0.0",
)


def _append_prediction_log(
    prediction_id: str,
    model_record: dict[str, object],
    probability: float,
    analytical_prediction: int,
    campaign_eligible: int,
    high_priority: int,
    risk_segment: str,
    traffic_label: str,
) -> None:
    log_row: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_id": prediction_id,
        **model_record,
        "churn_probability": probability,
        "analytical_predicted_churn": analytical_prediction,
        "campaign_eligible": campaign_eligible,
        "high_priority_customer": high_priority,
        "business_risk_segment": risk_segment,
        "traffic_label": traffic_label,
        "model_source": MODEL_RUNTIME.source,
        "model_alias": MODEL_RUNTIME.alias or "",
        "model_version": MODEL_RUNTIME.version or "",
    }

    with PREDICTION_LOG_LOCK:
        PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _migrate_prediction_log_schema(PREDICTION_LOG_PATH, list(log_row))
        write_header = not PREDICTION_LOG_PATH.exists() or PREDICTION_LOG_PATH.stat().st_size == 0
        with PREDICTION_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(log_row))
            if write_header:
                writer.writeheader()
            writer.writerow(log_row)


def _migrate_prediction_log_schema(log_path: Path, expected_fields: list[str]) -> None:
    """Upgrade the pre-traffic-label CSV schema while preserving every log row."""
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return

    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] == expected_fields:
        return

    legacy_fields = [field for field in expected_fields if field != "traffic_label"]
    if rows[0] != legacy_fields:
        raise RuntimeError(
            f"Unexpected prediction-log schema in {log_path}; refusing to append mixed columns."
        )

    traffic_index = expected_fields.index("traffic_label")
    migrated_rows: list[list[str]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) == len(legacy_fields):
            row = [*row[:traffic_index], LEGACY_TRAFFIC_LABEL, *row[traffic_index:]]
        elif len(row) != len(expected_fields):
            raise RuntimeError(
                f"Unexpected field count on prediction-log row {row_number}; refusing migration."
            )
        migrated_rows.append(row)

    backup_path = log_path.with_suffix(log_path.suffix + ".pre-traffic-label.bak")
    if not backup_path.exists():
        backup_path.write_bytes(log_path.read_bytes())

    temporary_path = log_path.with_suffix(log_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(expected_fields)
        writer.writerows(migrated_rows)
    temporary_path.replace(log_path)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL_RUNTIME.model is not None,
        model_source=MODEL_RUNTIME.source,
        model_alias=MODEL_RUNTIME.alias,
        model_version=MODEL_RUNTIME.version,
        feature_count=len(MODEL_FEATURES),
        analytical_threshold=THRESHOLDS.analytical,
        campaign_threshold=THRESHOLDS.campaign,
        high_risk_threshold=THRESHOLDS.high_risk,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    customer: CustomerInput,
    x_traffic_label: Annotated[
        str | None,
        Header(max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
    ] = None,
) -> PredictionResponse:
    prediction_id = str(uuid4())
    effective_traffic_label = x_traffic_label or DEFAULT_TRAFFIC_LABEL

    try:
        model_record = customer.to_model_record()
        customer_frame = pd.DataFrame([model_record], columns=MODEL_FEATURES)
        probability = float(MODEL_RUNTIME.model.predict_proba(customer_frame)[0, 1])
        if not 0 <= probability <= 1:
            raise RuntimeError("Model returned a probability outside [0, 1].")

        analytical_prediction = int(probability >= THRESHOLDS.analytical)
        campaign_eligible = int(probability >= THRESHOLDS.campaign)
        high_priority = int(probability >= THRESHOLDS.high_risk)
        risk_segment = THRESHOLDS.risk_segment(probability)

        _append_prediction_log(
            prediction_id=prediction_id,
            model_record=model_record,
            probability=probability,
            analytical_prediction=analytical_prediction,
            campaign_eligible=campaign_eligible,
            high_priority=high_priority,
            risk_segment=risk_segment,
            traffic_label=effective_traffic_label,
        )

        return PredictionResponse(
            prediction_id=prediction_id,
            churn_probability=probability,
            analytical_predicted_churn=analytical_prediction,
            campaign_eligible=campaign_eligible,
            high_priority_customer=high_priority,
            business_risk_segment=risk_segment,
            traffic_label=effective_traffic_label,
            model_source=MODEL_RUNTIME.source,
            model_alias=MODEL_RUNTIME.alias,
            model_version=MODEL_RUNTIME.version,
        )
    except HTTPException:
        raise
    except Exception:
        LOGGER.exception("Prediction failed for prediction_id=%s", prediction_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed. Check the server logs.",
        ) from None
