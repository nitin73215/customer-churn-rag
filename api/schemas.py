"""Pydantic request and response contracts for the churn API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


YesNo = Literal["Yes", "No"]
InternetDependent = Literal["Yes", "No", "No internet service"]


class CustomerInput(BaseModel):
    """The 19 raw features expected by the trained preprocessing pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    gender: Literal["Female", "Male"]
    senior_citizen: Literal[0, 1]
    partner: YesNo
    dependents: YesNo
    tenure_months: int = Field(ge=0)
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: InternetDependent
    online_backup: InternetDependent
    device_protection: InternetDependent
    tech_support: InternetDependent
    streaming_tv: InternetDependent
    streaming_movies: InternetDependent
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)

    def to_model_record(self) -> dict[str, object]:
        """Return values in the exact representation used during training."""

        record = self.model_dump()
        record["senior_citizen"] = "Yes" if self.senior_citizen == 1 else "No"
        return record


class PredictionResponse(BaseModel):
    prediction_id: str
    churn_probability: float
    analytical_predicted_churn: int
    campaign_eligible: int
    high_priority_customer: int
    business_risk_segment: str
    traffic_label: str
    model_source: Literal["mlflow_champion", "joblib_fallback"]
    model_alias: str | None = None
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_loaded: bool
    model_source: Literal["mlflow_champion", "joblib_fallback"]
    model_alias: str | None = None
    model_version: str | None = None
    feature_count: int
    analytical_threshold: float
    campaign_threshold: float
    high_risk_threshold: float
