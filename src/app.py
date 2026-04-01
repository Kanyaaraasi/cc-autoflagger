"""FastAPI dashboard for the Call Quality Auto-Flagger."""

import json
import pickle
import re

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import f1_score, precision_score, recall_score

from .config import OUTPUT_DIR, MODEL_DIR, TARGET
from .data_loader import load_all, parse_responses

app = FastAPI(title="CareCaller Auto-Flagger Dashboard")

# --- Global state loaded on startup ---
_state = {}


def _ensemble_predict_proba(models, scalers, subsets, X):
    """Average probabilities from ensemble models."""
    proba = np.zeros(len(X))
    n = 0
    for subset_name, cols in subsets.items():
        model = models[subset_name]
        X_sub = X[cols]
        if subset_name in scalers:
            X_sub = pd.DataFrame(
                scalers[subset_name].transform(X_sub),
                columns=cols, index=X_sub.index,
            )
        proba += model.predict_proba(X_sub)[:, 1]
        n += 1
    return proba / n


def _load_state():
    """Load model, data, and precompute predictions."""
    if _state:
        return

    # Load model + config
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        payload = pickle.load(f)
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)

    models = payload["models"]
    scalers = payload.get("scalers", {})
    subsets = config["subsets"]
    threshold = config["threshold"]

    # All columns needed
    all_columns = sorted(set(c for cols in subsets.values() for c in cols))

    # Load data
    train, val, test = load_all()

    # Load precomputed features
    X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")[all_columns]
    X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")[all_columns]

    # Note: combined train+val is stored as X_train.parquet now
    # For dashboard, we need per-split features. Re-extract from stored combined.
    # Train portion is first len(train) rows, val portion is next len(val) rows.
    X_train_split = X_train.iloc[:len(train)].reset_index(drop=True)
    X_val_split = X_train.iloc[len(train):len(train)+len(val)].reset_index(drop=True)

    # Compute predictions for all splits
    splits = {}
    for name, df, X in [("train", train, X_train_split), ("val", val, X_val_split), ("test", test, X_test)]:
        proba = _ensemble_predict_proba(models, scalers, subsets, X)
        pred = proba >= threshold

        records = []
        for idx, (_, row) in enumerate(df.iterrows()):
            rec = {
                "call_id": row["call_id"],
                "outcome": row["outcome"],
                "call_duration": int(row["call_duration"]),
                "response_completeness": float(row["response_completeness"]),
                "answered_count": int(row["answered_count"]),
                "whisper_mismatch_count": int(row["whisper_mismatch_count"]),
                "turn_count": int(row["turn_count"]),
                "probability": round(float(proba[idx]), 4),
                "predicted_ticket": bool(pred[idx]),
                "predicted_category": "",
                "split": name,
            }
            if name != "test":
                rec["actual_ticket"] = bool(row[TARGET])
            records.append(rec)

        splits[name] = records

    # Val metrics
    y_val = val[TARGET].astype(int).values
    val_proba = _ensemble_predict_proba(models, scalers, subsets, X_val_split)
    val_pred = (val_proba >= threshold).astype(int)

    _state.update({
        "models": models,
        "scalers": scalers,
        "subsets": subsets,
        "config": config,
        "columns": all_columns,
        "threshold": threshold,
        "splits": splits,
        "all_calls": splits["train"] + splits["val"] + splits["test"],
        "train_df": train,
        "val_df": val,
        "test_df": test,
        "X_train": X_train_split,
        "X_val": X_val_split,
        "X_test": X_test,
        "val_metrics": {
            "f1": round(f1_score(y_val, val_pred, zero_division=0), 4),
            "precision": round(precision_score(y_val, val_pred, zero_division=0), 4),
            "recall": round(recall_score(y_val, val_pred, zero_division=0), 4),
        },
    })


@app.on_event("startup")
def startup():
    _load_state()


# --- Static files ---
static_dir = str(OUTPUT_DIR.parent / "src" / "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def dashboard():
    return FileResponse(static_dir + "/index.html")


@app.get("/call/{call_id}")
def call_detail_page(call_id: str):
    return FileResponse(static_dir + "/call.html")


# --- API Endpoints ---

@app.get("/api/stats")
def get_stats():
    s = _state
    all_calls = s["all_calls"]
    flagged = [c for c in all_calls if c["predicted_ticket"]]

    # Outcome breakdown
    outcome_counts = {}
    for c in all_calls:
        o = c["outcome"]
        if o not in outcome_counts:
            outcome_counts[o] = {"total": 0, "flagged": 0}
        outcome_counts[o]["total"] += 1
        if c["predicted_ticket"]:
            outcome_counts[o]["flagged"] += 1

    return {
        "total_calls": len(all_calls),
        "flagged_calls": len(flagged),
        "flagged_pct": round(len(flagged) / len(all_calls) * 100, 1),
        "threshold": s["threshold"],
        "val_metrics": s["val_metrics"],
        "model": s["config"]["best_model"],
        "feature_count": len(s["columns"]),
        "outcome_breakdown": outcome_counts,
        "splits": {
            "train": len(s["splits"]["train"]),
            "val": len(s["splits"]["val"]),
            "test": len(s["splits"]["test"]),
        },
    }


@app.get("/api/calls")
def get_calls(split: str = Query(default="all"), flagged_only: bool = Query(default=False)):
    if split == "all":
        calls = _state["all_calls"]
    else:
        calls = _state["splits"].get(split, [])

    if flagged_only:
        calls = [c for c in calls if c["predicted_ticket"]]

    return sorted(calls, key=lambda c: c["probability"], reverse=True)


@app.get("/api/call/{call_id}")
def get_call_detail(call_id: str):
    s = _state

    for split_name, df, X in [
        ("train", s["train_df"], s["X_train"]),
        ("val", s["val_df"], s["X_val"]),
        ("test", s["test_df"], s["X_test"]),
    ]:
        match = df[df["call_id"] == call_id]
        if len(match) == 0:
            continue

        row = match.iloc[0]
        idx = match.index[0]
        # Map to feature index
        feat_idx = df.index.get_loc(idx)
        feat_row = X.iloc[feat_idx]

        # Probability
        X_single = X.iloc[[feat_idx]]
        proba = float(_ensemble_predict_proba(s["models"], s["scalers"], s["subsets"], X_single)[0])
        predicted = proba >= s["threshold"]

        # Parse transcript into turns
        transcript = str(row.get("transcript_text", ""))
        turns = []
        for m in re.finditer(r"\[(AGENT|USER)\]:\s*(.*?)(?=\[(?:AGENT|USER)\]:|$)", transcript, re.DOTALL):
            turns.append({"role": m.group(1).lower(), "text": m.group(2).strip()})

        # Parse Q&A
        responses = parse_responses(row.get("responses_json", ""))

        # Signal breakdown
        signals = {}

        # Transcript diff
        signals["transcript_diff"] = {
            "wer": round(float(feat_row.get("diff_wer", 0)), 4),
            "cer": round(float(feat_row.get("diff_cer", 0)), 4),
            "similarity": round(float(feat_row.get("diff_seq_similarity", 1)), 4),
        }

        # Number checker
        signals["number_checker"] = {
            "mismatches": int(feat_row.get("num_mismatches", 0)),
            "implausible": int(feat_row.get("num_implausible", 0)),
            "gaps": int(feat_row.get("num_response_transcript_gap", 0)),
        }

        # Flow checker
        signals["flow_checker"] = {
            "edit_distance": int(feat_row.get("flow_edit_distance", 0)),
            "missing_states": int(feat_row.get("flow_missing_states", 0)),
            "question_coverage": round(float(feat_row.get("flow_question_coverage", 0)), 2),
        }

        # Outcome predictor
        signals["outcome_predictor"] = {
            "disagreement": int(feat_row.get("outcome_disagreement", 0)),
            "confidence": round(float(feat_row.get("outcome_pred_confidence", 0)), 4),
            "entropy": round(float(feat_row.get("outcome_pred_entropy", 0)), 4),
        }

        # Embedding similarity
        signals["embeddings"] = {
            "vn_positive_similarity": round(float(feat_row.get("emb_vn_positive_similarity", 0)), 4),
        }

        # Top features for this call
        nonzero = feat_row[feat_row != 0].abs().sort_values(ascending=False)
        top_features = [{"name": k, "value": round(float(feat_row[k]), 4)} for k in nonzero.head(20).index]

        result = {
            "call_id": call_id,
            "split": split_name,
            "outcome": row["outcome"],
            "call_duration": int(row["call_duration"]),
            "response_completeness": float(row["response_completeness"]),
            "answered_count": int(row["answered_count"]),
            "whisper_mismatch_count": int(row["whisper_mismatch_count"]),
            "turn_count": int(row["turn_count"]),
            "probability": round(proba, 4),
            "predicted_ticket": bool(predicted),
            "predicted_category": "",
            "validation_notes": str(row.get("validation_notes", "")),
            "transcript_turns": turns,
            "responses": responses,
            "signals": signals,
            "top_features": top_features,
        }

        if split_name != "test":
            result["actual_ticket"] = bool(row[TARGET])

        return result

    return {"error": "Call not found"}


@app.get("/api/importance")
def get_importance(top: int = Query(default=30)):
    # Use first LightGBM model's feature importances
    s = _state
    for subset_name in ["text_light", "embedding"]:
        model = s["models"].get(subset_name)
        if hasattr(model, "feature_importances_"):
            cols = s["subsets"][subset_name]
            imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
            return [
                {"feature": name, "importance": round(float(val), 4)}
                for name, val in imp.head(top).items()
            ]
    return []


def main():
    """CLI entry point: uv run dashboard"""
    print("Starting CareCaller Dashboard at http://localhost:8000")
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
