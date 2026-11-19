Model artifact policy
====================

The trained fallback model is intentionally not included in the public repository because it was fitted on a customer-level dataset whose redistribution rights and training-data privacy have not been independently cleared. After obtaining an authorized dataset copy and running the training workflow, place the resulting estimator at `models/final_tuned_churn_model.joblib` to run the API and Docker image. The `.joblib` file is ignored by Git.
