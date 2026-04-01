"""Generate predictions on the test set and output submission.csv."""

import json
import pickle

import pandas as pd
import numpy as np

from .config import OUTPUT_DIR, MODEL_DIR
from .data_loader import load_all
from .logger import get_logger

log = get_logger("predict")


def generate_submission():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "model.pkl", "rb") as f:
        payload = pickle.load(f)
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)

    threshold = config["threshold"]
    subsets = config["subsets"]
    models = payload["models"]
    scalers = payload.get("scalers", {})

    X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")
    _, _, test_df = load_all()

    # Average probabilities across ensemble
    ensemble_proba = np.zeros(len(X_test))
    n_models = 0

    for subset_name, cols in subsets.items():
        model = models[subset_name]
        X_subset = X_test[cols]
        if subset_name in scalers:
            X_subset = pd.DataFrame(
                scalers[subset_name].transform(X_subset),
                columns=cols, index=X_subset.index,
            )
        proba = model.predict_proba(X_subset)[:, 1]
        ensemble_proba += proba
        n_models += 1

    ensemble_proba /= n_models
    predictions = (ensemble_proba >= threshold).astype(bool)

    submission = pd.DataFrame({
        "call_id": test_df["call_id"].values,
        "predicted_ticket": predictions,
    })

    out_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)

    log.info(f"Submission saved to {out_path}")
    log.info(f"Model: ensemble ({n_models} models) | Threshold: {threshold:.2f}")
    log.info(f"Predicted tickets: {predictions.sum()} / {len(predictions)} ({predictions.mean():.1%})")

    return submission


def main():
    generate_submission()
