"""Generate predictions on the test set and output submission CSVs."""

import json
import pickle

import pandas as pd

from .config import OUTPUT_DIR, MODEL_DIR, TARGET
from .data_loader import load_all
from .logger import get_logger

log = get_logger("predict")


def _load_model_and_config():
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)
    return model, config


def _get_proba(model, X, model_type):
    """Get probabilities from a model, handling ensemble case."""
    if model_type == "ensemble":
        parts = [model[k].predict_proba(X)[:, 1] for k in model if hasattr(model[k], "predict_proba")]
        return sum(parts) / len(parts)
    return model.predict_proba(X)[:, 1]


def _predict_stratified(model, config, X, df):
    """Predict using stratified model (route by outcome)."""
    import numpy as np
    columns = config["columns"]
    comp_mask = df["outcome"] == "completed"
    predictions = np.zeros(len(df), dtype=bool)
    probas = np.zeros(len(df))

    for group, mask_values in [("completed", comp_mask.values), ("non_completed", ~comp_mask.values)]:
        idx = np.where(mask_values)[0]
        if len(idx) == 0:
            continue
        group_model = model[group]
        group_config = config[group]
        proba = _get_proba(group_model, X.iloc[idx], group_config["best_model"])
        predictions[idx] = proba >= group_config["threshold"]
        probas[idx] = proba

    return probas, predictions


def _predict_single(model, config, X):
    """Predict using single model."""
    proba = _get_proba(model, X, config["best_model"])
    return proba, (proba >= config["threshold"]).astype(bool)


def generate_submission():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, config = _load_model_and_config()
    columns = config["columns"]
    is_stratified = config.get("stratified", False)

    train_df, val_df, test_df = load_all()

    X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")[columns]

    if is_stratified:
        proba, predictions = _predict_stratified(model, config, X_test, test_df)
        log.info("Using stratified model (completed + non-completed)")
    else:
        proba, predictions = _predict_single(model, config, X_test)
        log.info(f"Using single model: {config['best_model']} (t={config['threshold']:.2f})")

    submission = pd.DataFrame({
        "call_id": test_df["call_id"].values,
        "predicted_ticket": predictions,
    })
    out_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    log.info(f"Submission saved to {out_path}")
    log.info(f"Predicted tickets: {predictions.sum()} / {len(predictions)} ({predictions.mean():.1%})")

    return submission


def main():
    generate_submission()
