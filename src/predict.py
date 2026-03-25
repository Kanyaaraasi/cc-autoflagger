"""Generate predictions on the test set and output submission.csv."""

import json
import pickle

import pandas as pd

from .config import OUTPUT_DIR, MODEL_DIR
from .data_loader import load_all
from .logger import get_logger

log = get_logger("predict")


def generate_submission():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)

    threshold = config["threshold"]
    columns = config["columns"]
    best_model = config["best_model"]

    X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")
    X_test = X_test[columns]

    _, _, test_df = load_all()

    # Get probabilities
    if best_model == "ensemble":
        xgb_proba = model["xgb"].predict_proba(X_test)[:, 1]
        lgb_proba = model["lgb"].predict_proba(X_test)[:, 1]
        proba = 0.5 * xgb_proba + 0.5 * lgb_proba
    else:
        proba = model.predict_proba(X_test)[:, 1]

    predictions = (proba >= threshold).astype(bool)

    submission = pd.DataFrame({
        "call_id": test_df["call_id"].values,
        "predicted_ticket": predictions,
    })

    out_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)

    log.info(f"Submission saved to {out_path}")
    log.info(f"Model: {best_model} | Threshold: {threshold:.2f}")
    log.info(f"Predicted tickets: {predictions.sum()} / {len(predictions)} ({predictions.mean():.1%})")

    return submission


def main():
    generate_submission()
