"""Train models, tune thresholds, evaluate on validation set."""

import json
import pickle
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import OUTPUT_DIR, MODEL_DIR, TARGET
from .data_loader import load_all
from .logger import get_logger

log = get_logger("train")


def find_best_threshold(y_true, y_proba, metric="f1"):
    """Sweep thresholds to maximize F1."""
    best_thresh, best_score = 0.5, 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score


def find_precision_threshold(y_true, y_proba, min_recall=0.9):
    """Find threshold that maximizes precision while keeping recall >= min_recall."""
    best_thresh, best_prec = 0.5, 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        if rec >= min_recall and prec > best_prec:
            best_prec = prec
            best_thresh = thresh
    return best_thresh, best_prec


def analyze_errors(y_true, y_proba, threshold, X, call_ids):
    """Analyze false positives and false negatives."""
    y_pred = (y_proba >= threshold).astype(int)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    if fp_mask.sum() > 0:
        log.info(f"\n--- FALSE POSITIVES ({fp_mask.sum()}) ---")
        fp_indices = np.where(fp_mask)[0]
        for idx in fp_indices:
            cid = call_ids.iloc[idx] if hasattr(call_ids, 'iloc') else call_ids[idx]
            prob = y_proba[idx]
            log.info(f"  {cid[:12]}... proba={prob:.3f} (threshold={threshold:.2f})")
            # Show top features for this call
            row = X.iloc[idx]
            top_feats = row[row != 0].abs().sort_values(ascending=False).head(5)
            for feat, val in top_feats.items():
                log.info(f"    {feat} = {val:.4f}")

    if fn_mask.sum() > 0:
        log.info(f"\n--- FALSE NEGATIVES ({fn_mask.sum()}) ---")
        fn_indices = np.where(fn_mask)[0]
        for idx in fn_indices:
            cid = call_ids.iloc[idx] if hasattr(call_ids, 'iloc') else call_ids[idx]
            prob = y_proba[idx]
            log.info(f"  {cid[:12]}... proba={prob:.3f}")


def grid_search_cv(X_train, y_train, scale_pos_weight):
    """Grid search over XGBoost + LightGBM hyperparameters using CV."""
    log.info("\n--- Grid Search (CV on train) ---")

    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200, 300],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score, best_params, best_model_type = 0, {}, "xgb"

    for depth, lr, n_est in product(
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["n_estimators"],
    ):
        # XGBoost
        cv_scores = []
        for tr_idx, te_idx in skf.split(X_train, y_train):
            model = XGBClassifier(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                scale_pos_weight=scale_pos_weight, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=3,
                eval_metric="logloss", random_state=42,
            )
            model.fit(X_train.iloc[tr_idx], y_train[tr_idx], verbose=False)
            proba = model.predict_proba(X_train.iloc[te_idx])[:, 1]
            thresh, f1 = find_best_threshold(y_train[te_idx], proba)
            cv_scores.append(f1)

        mean_f1 = np.mean(cv_scores)
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = {"max_depth": depth, "learning_rate": lr, "n_estimators": n_est}
            best_model_type = "xgb"

    log.info(f"Best XGB params: {best_params} → CV F1={best_score:.4f}")

    # LightGBM search
    lgb_best_score, lgb_best_params = 0, {}
    for depth, lr, n_est in product([3, 4, 5], [0.05, 0.1], [100, 200, 300]):
        cv_scores = []
        for tr_idx, te_idx in skf.split(X_train, y_train):
            model = LGBMClassifier(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                scale_pos_weight=scale_pos_weight, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=5,
                random_state=42, verbose=-1,
            )
            model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            proba = model.predict_proba(X_train.iloc[te_idx])[:, 1]
            thresh, f1 = find_best_threshold(y_train[te_idx], proba)
            cv_scores.append(f1)

        mean_f1 = np.mean(cv_scores)
        if mean_f1 > lgb_best_score:
            lgb_best_score = mean_f1
            lgb_best_params = {"max_depth": depth, "learning_rate": lr, "n_estimators": n_est}

    log.info(f"Best LGB params: {lgb_best_params} → CV F1={lgb_best_score:.4f}")

    return best_params, lgb_best_params


def train_and_evaluate():
    """Full training pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
    X_val = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")

    train_df, val_df, _ = load_all()
    y_train = train_df[TARGET].astype(int).values
    y_val = val_df[TARGET].astype(int).values

    common_cols = sorted(set(X_train.columns) & set(X_val.columns))
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)
    log.info(f"Features: {len(common_cols)}")
    log.info(f"Train: {X_train.shape[0]} rows, {pos} positive ({pos/len(y_train):.1%})")
    log.info(f"Val: {X_val.shape[0]} rows, {y_val.sum()} positive ({y_val.mean():.1%})")

    # --- Grid Search ---
    xgb_params, lgb_params = grid_search_cv(X_train, y_train, scale_pos_weight)

    # --- Train final models with best params ---
    log.info("\n--- Training Final XGBoost ---")
    xgb = XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3,
        eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_proba = xgb.predict_proba(X_val)[:, 1]

    log.info("\n--- Training Final LightGBM ---")
    lgb = LGBMClassifier(
        **lgb_params,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.8, min_child_samples=5,
        random_state=42, verbose=-1,
    )
    lgb.fit(X_train, y_train)
    lgb_val_proba = lgb.predict_proba(X_val)[:, 1]

    # --- Ensemble (average probabilities) ---
    ensemble_proba = 0.5 * xgb_val_proba + 0.5 * lgb_val_proba

    # --- Evaluate all three ---
    for name, proba in [("XGBoost", xgb_val_proba), ("LightGBM", lgb_val_proba), ("Ensemble", ensemble_proba)]:
        thresh, f1 = find_best_threshold(y_val, proba)
        pred = (proba >= thresh).astype(int)
        rec = recall_score(y_val, pred, zero_division=0)
        prec = precision_score(y_val, pred, zero_division=0)
        log.info(f"\n{name} (threshold={thresh:.2f}): F1={f1:.4f} Precision={prec:.4f} Recall={rec:.4f} Predicted={pred.sum()}")

    # --- Pick best model ---
    results = {}
    for name, proba in [("xgb", xgb_val_proba), ("lgb", lgb_val_proba), ("ensemble", ensemble_proba)]:
        thresh, f1 = find_best_threshold(y_val, proba)
        results[name] = {"f1": f1, "threshold": thresh, "proba": proba}

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_proba = results[best_name]["proba"]
    best_thresh = results[best_name]["threshold"]
    best_f1 = results[best_name]["f1"]
    log.info(f"\n*** Best model: {best_name} with F1={best_f1:.4f} ***")

    # --- Detailed report ---
    val_pred = (best_proba >= best_thresh).astype(int)
    log.info(f"\n{classification_report(y_val, val_pred, target_names=['no_ticket', 'ticket'], zero_division=0)}")

    # --- Precision-aware threshold ---
    prec_thresh, prec_score_val = find_precision_threshold(y_val, best_proba, min_recall=0.9)
    prec_pred = (best_proba >= prec_thresh).astype(int)
    prec_f1 = f1_score(y_val, prec_pred, zero_division=0)
    prec_rec = recall_score(y_val, prec_pred, zero_division=0)
    log.info(f"Precision-optimized (recall>=0.9): threshold={prec_thresh:.2f} F1={prec_f1:.4f} Precision={prec_score_val:.4f} Recall={prec_rec:.4f}")

    # --- Error analysis ---
    analyze_errors(y_val, best_proba, best_thresh, X_val, val_df["call_id"])

    # --- CV stability check ---
    log.info("\n--- 5-Fold CV Stability ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1s = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_train, y_train)):
        xgb_cv = XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight,
                                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                eval_metric="logloss", random_state=42)
        xgb_cv.fit(X_train.iloc[tr_idx], y_train[tr_idx], verbose=False)
        fold_proba = xgb_cv.predict_proba(X_train.iloc[te_idx])[:, 1]
        fold_thresh, fold_f1 = find_best_threshold(y_train[te_idx], fold_proba)
        cv_f1s.append(fold_f1)
        log.info(f"  Fold {fold + 1}: F1={fold_f1:.4f} (threshold={fold_thresh:.2f})")
    log.info(f"  Mean CV F1: {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")

    # --- Feature importance ---
    importance = pd.Series(xgb.feature_importances_, index=common_cols).sort_values(ascending=False)
    log.info("\n--- Top 20 Features ---")
    for feat, imp in importance.head(20).items():
        log.info(f"  {feat}: {imp:.4f}")

    # --- Save best model ---
    model_to_save = xgb if best_name == "xgb" else (lgb if best_name == "lgb" else {"xgb": xgb, "lgb": lgb})
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model_to_save, f)
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump({
            "threshold": best_thresh,
            "columns": common_cols,
            "best_model": best_name,
            "xgb_params": xgb_params,
            "lgb_params": lgb_params,
        }, f)

    log.info(f"\nModel saved ({best_name}) to {MODEL_DIR}")
    return best_name, best_thresh, common_cols


def main():
    train_and_evaluate()
