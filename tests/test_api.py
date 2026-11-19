from collections.abc import Iterator
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import joblib
import pandas as pd
import pytest

from api.main import LEGACY_TRAFFIC_LABEL, _migrate_prediction_log_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_tuned_churn_model.joblib"
POLICY_PATH = PROJECT_ROOT / "models" / "business_threshold_policy.json"
FEATURES = [
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


@dataclass(frozen=True)
class ServerRuntime:
    base_url: str
    prediction_log: Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


@pytest.fixture(scope="session")
def api_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ServerRuntime]:
    runtime_directory = tmp_path_factory.mktemp("api_runtime")
    prediction_log = runtime_directory / "predictions.csv"
    server_log = runtime_directory / "uvicorn.log"
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment["CHURN_PREDICTION_LOG_PATH"] = str(prediction_log)
    environment["CHURN_TRAFFIC_LABEL"] = "pytest_integration"

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with server_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )

        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    response = httpx.get(f"{base_url}/health", timeout=2)
                    if response.status_code == 200:
                        yield ServerRuntime(base_url=base_url, prediction_log=prediction_log)
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(1)

            log_handle.flush()
            startup_log = server_log.read_text(encoding="utf-8", errors="replace")
            pytest.fail(f"API server did not become healthy.\n{startup_log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


@pytest.fixture(scope="session")
def client(api_server: ServerRuntime) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=api_server.base_url, timeout=30) as test_client:
        yield test_client


def example_customer_payload() -> dict[str, object]:
    """Return a synthetic example profile for API contract tests."""
    return {
        "gender": "Female", "senior_citizen": 0, "partner": "Yes",
        "dependents": "No", "tenure_months": 12, "phone_service": "Yes",
        "multiple_lines": "No", "internet_service": "Fiber optic",
        "online_security": "No", "online_backup": "Yes",
        "device_protection": "No", "tech_support": "No",
        "streaming_tv": "Yes", "streaming_movies": "No",
        "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "monthly_charges": 75.0,
        "total_charges": 900.0,
    }


def _risk_segment(probability: float, policy: dict[str, object]) -> str:
    if probability >= float(policy["high_risk_threshold"]):
        return "High Risk"
    if probability >= float(policy["campaign_threshold"]):
        return "Medium Risk"
    if probability >= float(policy["analytical_threshold"]):
        return "Watchlist"
    return "Low Risk"


def test_prediction_log_schema_migration_preserves_legacy_and_new_rows(tmp_path: Path) -> None:
    log_path = tmp_path / "predictions.csv"
    expected_fields = ["prediction_id", "traffic_label", "model_source"]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prediction_id", "model_source"])
        writer.writerow(["legacy-id", "mlflow_champion"])
        writer.writerow(["new-id", "simulated_monitoring_validation", "mlflow_champion"])

    _migrate_prediction_log_schema(log_path, expected_fields)

    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "prediction_id": "legacy-id",
            "traffic_label": LEGACY_TRAFFIC_LABEL,
            "model_source": "mlflow_champion",
        },
        {
            "prediction_id": "new-id",
            "traffic_label": "simulated_monitoring_validation",
            "model_source": "mlflow_champion",
        },
    ]
    assert log_path.with_suffix(".csv.pre-traffic-label.bak").is_file()


def test_health_returns_200(client: httpx.Client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["feature_count"] == 19


def test_valid_customer_returns_probability_and_creates_log(
    client: httpx.Client,
    api_server: ServerRuntime,
) -> None:
    response = client.post(
        "/predict",
        json=example_customer_payload(),
        headers={"X-Traffic-Label": "pytest_simulated"},
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["model_source"] in {"mlflow_champion", "joblib_fallback"}
    assert body["traffic_label"] == "pytest_simulated"
    assert api_server.prediction_log.is_file()
    logged = pd.read_csv(api_server.prediction_log)
    assert set(logged["traffic_label"]) == {"pytest_simulated"}


def test_risk_segment_matches_threshold_policy(client: httpx.Client) -> None:
    response = client.post("/predict", json=example_customer_payload())
    assert response.status_code == 200
    body = response.json()
    probability = float(body["churn_probability"])
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    assert body["business_risk_segment"] == _risk_segment(probability, policy)
    assert body["analytical_predicted_churn"] == int(
        probability >= float(policy["analytical_threshold"])
    )
    assert body["campaign_eligible"] == int(
        probability >= float(policy["campaign_threshold"])
    )
    assert body["high_priority_customer"] == int(
        probability >= float(policy["high_risk_threshold"])
    )


def test_missing_required_feature_returns_422(client: httpx.Client) -> None:
    payload = example_customer_payload()
    payload.pop("contract")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("senior_citizen", 2),
        ("tenure_months", -1),
        ("monthly_charges", -0.01),
        ("total_charges", -1),
    ],
)
def test_invalid_numeric_input_returns_422(
    client: httpx.Client,
    field: str,
    invalid_value: object,
) -> None:
    payload = example_customer_payload()
    payload[field] = invalid_value
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_probability_matches_direct_model(client: httpx.Client) -> None:
    payload = example_customer_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    model_record = payload.copy()
    model_record["senior_citizen"] = "Yes" if payload["senior_citizen"] == 1 else "No"
    feature_frame = pd.DataFrame([model_record], columns=FEATURES)
    direct_model = joblib.load(MODEL_PATH)
    direct_probability = float(direct_model.predict_proba(feature_frame)[0, 1])

    assert response.json()["churn_probability"] == pytest.approx(
        direct_probability,
        abs=1e-12,
    )
