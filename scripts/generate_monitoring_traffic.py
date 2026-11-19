"""Send existing customer rows through the API as labeled simulated traffic."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_model.csv"
TRAFFIC_LABEL = "simulated_monitoring_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.rows <= 1000:
        raise ValueError("--rows must be between 1 and 1000")

    customers = pd.read_csv(DATA_PATH).drop(columns=["churn_value"]).head(args.rows)
    headers = {"X-Traffic-Label": TRAFFIC_LABEL}

    with httpx.Client(base_url=args.url, headers=headers, timeout=30) as client:
        health = client.get("/health")
        health.raise_for_status()

        for index, row in customers.iterrows():
            payload = row.to_dict()
            payload["senior_citizen"] = 1 if payload["senior_citizen"] == "Yes" else 0
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            if response.json()["traffic_label"] != TRAFFIC_LABEL:
                raise RuntimeError(f"Traffic label was not preserved for row {index}.")

    print(f"Sent {len(customers)} existing customer rows.")
    print(f"Traffic label: {TRAFFIC_LABEL}")
    print("No synthetic feature values or production labels were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
