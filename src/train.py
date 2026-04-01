"""Train diverse ensemble, tune threshold via CV on combined train+val."""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from .config import OUTPUT_DIR, MODEL_DIR, TARGET
from .data_loader import load_all
from .features import FeaturePipeline, SUBSET_NUMERIC, SUBSET_TEXT_LIGHT, SUBSET_EMBEDDING
from .logger import get_logger

log = get_logger("train")


def find_best_threshold(y_true, y_proba):
    """Sweep thresholds to maximize F1."""
    best_thresh, best_score = 0.5, 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score


def _build_models(scale_pos_weight: float) -> dict:
    """Create the 3 diverse models for the ensemble."""
    return {
        SUBSET_NUMERIC: {
            "model": LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000, random_state=42,
            ),
            "scaler": StandardScaler(),
        },
        SUBSET_TEXT_LIGHT: {
            "model": LGBMClassifier(
                max_depth=2, num_leaves=4, n_estimators=150,
                learning_rate=0.05, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1,
            ),
            "scaler": None,
        },
        SUBSET_EMBEDDING: {
            "model": LGBMClassifier(
                max_depth=3, num_leaves=8, n_estimators=150,
                learning_rate=0.05, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1,
            ),
            "scaler": None,
        },
    }


def _fit_predict_model(entry, X_train, y_train, X_test):
    """Fit a single model and return test probabilities."""
    scaler = entry["scaler"]
    model = entry["model"]
    if scaler is not None:
        cols = X_train.columns
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def train_and_evaluate():
    """Full ensemble training pipeline with CV-averaged threshold."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load and combine train + val ---
    train_df, val_df, _ = load_all()
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    y_combined = combined_df[TARGET].astype(int).values

    neg, pos = (y_combined == 0).sum(), (y_combined == 1).sum()
    scale_pos_weight = neg / max(pos, 1)
    log.info(f"Combined: {len(combined_df)} rows, {pos} positive ({pos/len(y_combined):.1%})")

    # --- Extract features on combined data ---
    log.info("Fitting feature pipeline on combined train+val...")
    pipeline = FeaturePipeline()
    pipeline.fit(combined_df)

    X_combined = pipeline.transform(combined_df, split_name="combined")
    X_combined = X_combined.fillna(0)

    # Get feature subsets
    subsets = pipeline.get_subset_columns(list(X_combined.columns))
    for name, cols in subsets.items():
        log.info(f"  Subset '{name}': {len(cols)} features")

    # --- Also extract test features ---
    _, _, test_df = load_all()
    X_test_all = pipeline.transform(test_df, split_name="test")
    X_test_all = X_test_all.fillna(0)

    # Align columns
    for name in subsets:
        subsets[name] = [c for c in subsets[name] if c in X_combined.columns and c in X_test_all.columns]

    # --- 10-Fold CV for threshold estimation ---
    log.info("\n--- 10-Fold CV for threshold estimation ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_thresholds = []
    fold_f1s = []
    oof_probas = np.zeros(len(y_combined))

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_combined, y_combined)):
        models = _build_models(scale_pos_weight)
        fold_probas = np.zeros(len(te_idx))
        n_models = 0

        for subset_name, cols in subsets.items():
            X_tr = X_combined.iloc[tr_idx][cols]
            X_te = X_combined.iloc[te_idx][cols]
            y_tr = y_combined[tr_idx]

            proba = _fit_predict_model(models[subset_name], X_tr, y_tr, X_te)
            fold_probas += proba
            n_models += 1

        fold_probas /= n_models  # Average ensemble
        oof_probas[te_idx] = fold_probas

        thresh, f1 = find_best_threshold(y_combined[te_idx], fold_probas)
        fold_thresholds.append(thresh)
        fold_f1s.append(f1)
        log.info(f"  Fold {fold + 1}: F1={f1:.4f} threshold={thresh:.2f}")

    median_threshold = float(np.median(fold_thresholds))
    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))
    log.info(f"\n  CV F1: {mean_f1:.4f} ± {std_f1:.4f}")
    log.info(f"  Fold thresholds: {[f'{t:.2f}' for t in fold_thresholds]}")
    log.info(f"  Median threshold: {median_threshold:.2f}")

    # --- OOF evaluation at median threshold ---
    oof_pred = (oof_probas >= median_threshold).astype(int)
    oof_f1 = f1_score(y_combined, oof_pred, zero_division=0)
    oof_prec = precision_score(y_combined, oof_pred, zero_division=0)
    oof_rec = recall_score(y_combined, oof_pred, zero_division=0)
    log.info(f"\n  OOF @ median threshold: F1={oof_f1:.4f} Precision={oof_prec:.4f} Recall={oof_rec:.4f}")
    log.info(f"\n{classification_report(y_combined, oof_pred, target_names=['no_ticket', 'ticket'], zero_division=0)}")

    # --- Train final models on ALL combined data ---
    log.info("\n--- Training final models on all combined data ---")
    final_models = _build_models(scale_pos_weight)
    final_scalers = {}

    for subset_name, cols in subsets.items():
        X_all = X_combined[cols]
        entry = final_models[subset_name]
        scaler = entry["scaler"]
        model = entry["model"]
        if scaler is not None:
            X_all = pd.DataFrame(scaler.fit_transform(X_all), columns=cols, index=X_all.index)
            final_scalers[subset_name] = scaler
        model.fit(X_all, y_combined)
        log.info(f"  Trained {subset_name}: {type(model).__name__}")

    # --- Save ---
    save_payload = {
        "models": {name: entry["model"] for name, entry in final_models.items()},
        "scalers": final_scalers,
    }
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(save_payload, f)

    config = {
        "threshold": median_threshold,
        "subsets": {name: cols for name, cols in subsets.items()},
        "best_model": "ensemble",
        "cv_f1_mean": round(mean_f1, 4),
        "cv_f1_std": round(std_f1, 4),
        "oof_metrics": {
            "f1": round(oof_f1, 4),
            "precision": round(oof_prec, 4),
            "recall": round(oof_rec, 4),
        },
        "fold_thresholds": [round(t, 4) for t in fold_thresholds],
    }
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"\nEnsemble saved to {MODEL_DIR}")

    # --- Save features for predict.py ---
    X_combined.to_parquet(OUTPUT_DIR / "X_train.parquet")
    X_test_all.to_parquet(OUTPUT_DIR / "X_test.parquet")

    return "ensemble", median_threshold, subsets


def main():
    train_and_evaluate()
