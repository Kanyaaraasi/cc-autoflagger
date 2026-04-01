"""Stacking meta-learner: combines ML predictions + NLI contradiction scores."""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import OUTPUT_DIR, MODEL_DIR, TARGET
from .data_loader import load_all
from .logger import get_logger

log = get_logger("stack")


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


def _get_ml_oof_probas(X_train, y_train, ml_config):
    """Generate out-of-fold ML probabilities using 5-fold CV."""
    best_model_name = ml_config["best_model"]
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y_train))

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_train, y_train)):
        if best_model_name in ("lgb", "ensemble"):
            params = ml_config.get("lgb_params", {})
            model = LGBMClassifier(
                **params, scale_pos_weight=scale_pos_weight,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=5, random_state=42, verbose=-1,
            )
            model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        else:
            params = ml_config.get("xgb_params", {})
            model = XGBClassifier(
                **params, scale_pos_weight=scale_pos_weight,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, eval_metric="logloss", random_state=42,
            )
            model.fit(X_train.iloc[tr_idx], y_train[tr_idx], verbose=False)

        oof_proba[te_idx] = model.predict_proba(X_train.iloc[te_idx])[:, 1]
        log.info(f"  Fold {fold + 1}: OOF predictions generated")

    return oof_proba


def _get_ml_probas(ml_model, X, best_model_name):
    """Get ML probabilities from a trained model."""
    if best_model_name == "ensemble":
        return 0.5 * ml_model["xgb"].predict_proba(X)[:, 1] + \
               0.5 * ml_model["lgb"].predict_proba(X)[:, 1]
    return ml_model.predict_proba(X)[:, 1]


def stack_and_predict():
    """Two-level stacking: ML + NLI → LogReg meta-learner."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load base ML model ---
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        ml_model = pickle.load(f)
    with open(MODEL_DIR / "config.json") as f:
        ml_config = json.load(f)

    columns = ml_config["columns"]
    best_model_name = ml_config["best_model"]

    # --- Load data ---
    train_df, val_df, test_df = load_all()
    y_train = train_df[TARGET].astype(int).values
    y_val = val_df[TARGET].astype(int).values

    # --- Load ML features ---
    X_train_ml = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")[columns]
    X_val_ml = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")[columns]
    X_test_ml = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")[columns]

    # --- Load NLI features ---
    nli_train = pd.read_parquet(OUTPUT_DIR / "nli_train.parquet")
    nli_val = pd.read_parquet(OUTPUT_DIR / "nli_val.parquet")
    nli_test = pd.read_parquet(OUTPUT_DIR / "nli_test.parquet")
    nli_cols = list(nli_train.columns)
    log.info(f"NLI features: {nli_cols}")

    # --- Generate ML out-of-fold probabilities ---
    log.info("Generating ML out-of-fold predictions...")
    ml_oof_proba = _get_ml_oof_probas(X_train_ml, y_train, ml_config)

    # ML predictions on val and test from full model
    ml_val_proba = _get_ml_probas(ml_model, X_val_ml, best_model_name)
    ml_test_proba = _get_ml_probas(ml_model, X_test_ml, best_model_name)

    # --- Build meta-features ---
    def _build_meta(ml_proba, nli_df):
        return pd.DataFrame({
            "ml_proba": ml_proba,
            **{col: nli_df[col].values for col in nli_cols},
        })

    meta_train = _build_meta(ml_oof_proba, nli_train)
    meta_val = _build_meta(ml_val_proba, nli_val)
    meta_test = _build_meta(ml_test_proba, nli_test)

    log.info(f"Meta-features: {list(meta_train.columns)} ({meta_train.shape[1]} total)")

    # --- Train meta-learner ---
    log.info("Training meta-learner (LogisticRegression)...")
    meta_model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=42,
    )
    meta_model.fit(meta_train, y_train)

    for feat, coef in zip(meta_train.columns, meta_model.coef_[0]):
        log.info(f"  {feat}: coef={coef:.4f}")

    # --- Evaluate on val ---
    meta_val_proba = meta_model.predict_proba(meta_val)[:, 1]
    val_thresh, val_f1 = find_best_threshold(y_val, meta_val_proba)
    val_pred = (meta_val_proba >= val_thresh).astype(int)
    val_prec = precision_score(y_val, val_pred, zero_division=0)
    val_rec = recall_score(y_val, val_pred, zero_division=0)

    log.info(f"\nStacked Val: F1={val_f1:.4f} Precision={val_prec:.4f} Recall={val_rec:.4f} (threshold={val_thresh:.2f})")
    log.info(f"\n{classification_report(y_val, val_pred, target_names=['no_ticket', 'ticket'], zero_division=0)}")

    # Compare with base ML
    ml_thresh, ml_f1 = find_best_threshold(y_val, ml_val_proba)
    log.info(f"Base ML Val: F1={ml_f1:.4f} (threshold={ml_thresh:.2f})")
    log.info(f"Improvement: {val_f1 - ml_f1:+.4f}")

    # --- Generate test predictions ---
    meta_test_proba = meta_model.predict_proba(meta_test)[:, 1]
    predictions = (meta_test_proba >= val_thresh).astype(bool)

    submission = pd.DataFrame({
        "call_id": test_df["call_id"].values,
        "predicted_ticket": predictions,
    })

    out_path = OUTPUT_DIR / "submission_stacked.csv"
    submission.to_csv(out_path, index=False)
    log.info(f"\nStacked submission: {out_path}")
    log.info(f"Predicted tickets: {predictions.sum()} / {len(predictions)} ({predictions.mean():.1%})")

    # --- Save meta-model ---
    with open(MODEL_DIR / "meta_model.pkl", "wb") as f:
        pickle.dump(meta_model, f)

    with open(MODEL_DIR / "meta_config.json", "w") as f:
        json.dump({
            "threshold": val_thresh,
            "stacked_val_f1": round(val_f1, 4),
            "base_ml_val_f1": round(ml_f1, 4),
            "meta_features": list(meta_train.columns),
            "nli_cols": nli_cols,
        }, f, indent=2)

    log.info("Meta-model saved.")
    return val_f1
