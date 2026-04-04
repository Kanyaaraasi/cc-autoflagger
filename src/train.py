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
    best_val_pred = (best_proba >= best_thresh).astype(int)
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump({
            "threshold": best_thresh,
            "columns": common_cols,
            "best_model": best_name,
            "xgb_params": xgb_params,
            "lgb_params": lgb_params,
            "val_metrics": {
                "f1": round(f1_score(y_val, best_val_pred, zero_division=0), 4),
                "precision": round(precision_score(y_val, best_val_pred, zero_division=0), 4),
                "recall": round(recall_score(y_val, best_val_pred, zero_division=0), 4),
            },
            "cv_f1_mean": round(float(np.mean(cv_f1s)), 4),
            "cv_f1_std": round(float(np.std(cv_f1s)), 4),
        }, f)

    log.info(f"\nModel saved ({best_name}) to {MODEL_DIR}")
    return best_name, best_thresh, common_cols


def _train_group(X_train, y_train, X_val, y_val, group_name, scale_pos_weight):
    """Train and evaluate a single group (completed or non-completed)."""
    log.info(f"\n{'='*60}")
    log.info(f"  {group_name}: {len(X_train)} train ({y_train.sum()} tickets), {len(X_val)} val ({y_val.sum()} tickets)")
    log.info(f"{'='*60}")

    xgb_params, lgb_params = grid_search_cv(X_train, y_train, scale_pos_weight)

    xgb = XGBClassifier(
        **xgb_params, scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3,
        eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_proba = xgb.predict_proba(X_val)[:, 1]

    lgb = LGBMClassifier(
        **lgb_params, scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.8, min_child_samples=5,
        random_state=42, verbose=-1,
    )
    lgb.fit(X_train, y_train)
    lgb_proba = lgb.predict_proba(X_val)[:, 1]

    ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba

    results = {}
    for name, proba in [("xgb", xgb_proba), ("lgb", lgb_proba), ("ensemble", ensemble_proba)]:
        thresh, f1 = find_best_threshold(y_val, proba)
        pred = (proba >= thresh).astype(int)
        prec = precision_score(y_val, pred, zero_division=0)
        rec = recall_score(y_val, pred, zero_division=0)
        results[name] = {"f1": f1, "threshold": thresh, "proba": proba}
        log.info(f"  {name}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} (t={thresh:.2f}, flagged={pred.sum()})")

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = xgb if best_name == "xgb" else (lgb if best_name == "lgb" else {"xgb": xgb, "lgb": lgb})
    best_thresh = results[best_name]["threshold"]
    best_proba = results[best_name]["proba"]
    best_f1 = results[best_name]["f1"]

    log.info(f"  *** Best: {best_name} F1={best_f1:.4f} ***")

    return {
        "model": best_model,
        "model_type": best_name,
        "threshold": best_thresh,
        "proba": best_proba,
        "f1": best_f1,
        "xgb_params": xgb_params,
        "lgb_params": lgb_params,
    }


def train_stratified():
    """Train outcome-stratified models: separate for completed vs non-completed."""
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

    log.info(f"Features: {len(common_cols)}")
    log.info(f"Train: {len(X_train)} ({y_train.sum()} tickets)")
    log.info(f"Val: {len(X_val)} ({y_val.sum()} tickets)")

    # --- Split by outcome ---
    comp_train = train_df["outcome"] == "completed"
    comp_val = val_df["outcome"] == "completed"

    groups = {
        "completed": (comp_train.values, comp_val.values),
        "non_completed": (~comp_train.values, ~comp_val.values),
    }

    models = {}
    combined_pred = np.zeros(len(y_val), dtype=int)
    combined_proba = np.zeros(len(y_val))

    for group_name, (train_mask, val_mask) in groups.items():
        Xtr = X_train[train_mask]
        ytr = y_train[train_mask]
        Xv = X_val[val_mask]
        yv = y_val[val_mask]

        neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
        spw = neg / max(pos, 1)

        result = _train_group(Xtr, ytr, Xv, yv, group_name, spw)
        models[group_name] = result

        # Fill combined predictions
        val_idx = np.where(val_mask)[0]
        pred = (result["proba"] >= result["threshold"]).astype(int)
        combined_pred[val_idx] = pred
        combined_proba[val_idx] = result["proba"]

    # --- Combined evaluation ---
    log.info(f"\n{'='*60}")
    log.info("  COMBINED RESULTS")
    log.info(f"{'='*60}")

    f1 = f1_score(y_val, combined_pred, zero_division=0)
    prec = precision_score(y_val, combined_pred, zero_division=0)
    rec = recall_score(y_val, combined_pred, zero_division=0)
    caught = (combined_pred & y_val).sum()
    flagged = combined_pred.sum()
    wrong = (combined_pred & (y_val == 0)).sum()

    log.info(f"  F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
    log.info(f"  Caught: {caught}/{y_val.sum()}  Flagged: {flagged}  Wrong: {wrong}")
    log.info(f"\n{classification_report(y_val, combined_pred, target_names=['no_ticket', 'ticket'], zero_division=0)}")

    # Per-group breakdown
    for group_name, (_, val_mask) in groups.items():
        yv = y_val[val_mask]
        pv = combined_pred[val_mask]
        g_caught = (pv & yv).sum()
        g_flagged = pv.sum()
        g_wrong = (pv & (yv == 0)).sum()
        log.info(f"  {group_name:15s}: caught {g_caught}/{yv.sum()} flagged {g_flagged} wrong {g_wrong}")

    # --- Error analysis ---
    analyze_errors(y_val, combined_proba, 0.5, X_val, val_df["call_id"])

    # --- Save ---
    model_to_save = {
        "completed": models["completed"]["model"],
        "non_completed": models["non_completed"]["model"],
    }
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model_to_save, f)

    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump({
            "stratified": True,
            "columns": common_cols,
            "completed": {
                "best_model": models["completed"]["model_type"],
                "threshold": models["completed"]["threshold"],
                "f1": round(models["completed"]["f1"], 4),
                "xgb_params": models["completed"]["xgb_params"],
                "lgb_params": models["completed"]["lgb_params"],
            },
            "non_completed": {
                "best_model": models["non_completed"]["model_type"],
                "threshold": models["non_completed"]["threshold"],
                "f1": round(models["non_completed"]["f1"], 4),
                "xgb_params": models["non_completed"]["xgb_params"],
                "lgb_params": models["non_completed"]["lgb_params"],
            },
            "val_metrics": {
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            },
        }, f, indent=2)

    log.info(f"\nStratified models saved to {MODEL_DIR}")
    return f1


def main():
    train_stratified()
