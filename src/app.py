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

_state = {}


def compute_flag_explanation(signals, proba, predicted_ticket):
    """Generate plain-English explanation for why a call was flagged or not."""
    reasons = []

    nli = signals.get("nli", {})
    if nli.get("max_contradiction", 0) > 0.7:
        reasons.append(f"NLI detected contradiction between validation notes and call data (score: {nli['max_contradiction']*100:.0f}%)")
    if nli.get("answered_count_contradiction", 0) > 0.5:
        reasons.append(f"NLI contradiction on answered count ({nli['answered_count_contradiction']*100:.0f}%)")

    resp = signals.get("response_checker", {})
    if resp.get("not_in_transcript", 0) > 0.1:
        reasons.append(f"{resp['not_in_transcript']*100:.0f}% of responses not found in transcript")
    if resp.get("empty_count", 0) > 3:
        reasons.append(f"{resp['empty_count']} empty responses recorded")

    heur = signals.get("heuristics", {})
    fired = heur.get("fired", {})
    if fired:
        names = [k.replace("rule_", "").replace("_", " ") for k in fired]
        reasons.append(f"{len(fired)} heuristic rule(s) triggered: {', '.join(names)}")

    diff = signals.get("transcript_diff", {})
    if diff.get("wer", 0) > 0.1:
        reasons.append(f"Word error rate elevated ({diff['wer']*100:.0f}%)")

    out = signals.get("outcome_predictor", {})
    if out.get("disagreement", 0):
        reasons.append("Outcome predictor disagrees with recorded outcome")

    nc = signals.get("number_checker", {})
    if nc.get("mismatches", 0) > 0:
        reasons.append("Number mismatch detected between transcript and recorded answer")
    if nc.get("implausible", 0) > 0:
        reasons.append("Implausible health value recorded")

    if predicted_ticket:
        if reasons:
            parts = [f"({i+1}) {r}" for i, r in enumerate(reasons)]
            return "Flagged because: " + ", ".join(parts)
        return f"Flagged (probability: {proba*100:.1f}%). No single dominant signal — combined feature pattern triggered the model."
    else:
        if reasons:
            return f"Not flagged (probability: {proba*100:.1f}%). Minor signals detected but below threshold: {reasons[0].lower()}"
        return f"Not flagged. All signals within normal range. Probability: {proba*100:.1f}%"


def compute_signal_health(signals):
    """Compute green/yellow/red health status for each signal group."""
    health = {}

    heur = signals.get("heuristics", {})
    n = heur.get("total_fired", 0)
    health["heuristics"] = "red" if n >= 2 else ("yellow" if n == 1 else "green")

    diff = signals.get("transcript_diff", {})
    wer = diff.get("wer", 0)
    health["transcript_diff"] = "red" if wer > 0.15 else ("yellow" if wer > 0.05 else "green")

    nc = signals.get("number_checker", {})
    issues = nc.get("mismatches", 0) + nc.get("implausible", 0)
    health["number_checker"] = "red" if issues >= 2 else ("yellow" if issues == 1 else "green")

    fc = signals.get("flow_checker", {})
    cov = fc.get("question_coverage", 1)
    health["flow_checker"] = "red" if cov < 0.5 else ("yellow" if cov < 0.8 else "green")

    out = signals.get("outcome_predictor", {})
    health["outcome_predictor"] = "red" if out.get("disagreement", 0) else "green"

    kw = signals.get("text_features", {})
    n_kw = len(kw.get("keywords_found", {}))
    health["text_features"] = "red" if n_kw >= 3 else ("yellow" if n_kw >= 1 else "green")

    resp = signals.get("response_checker", {})
    nit = resp.get("not_in_transcript", 0)
    health["response_checker"] = "red" if nit > 0.15 else ("yellow" if nit > 0.05 else "green")

    nli = signals.get("nli", {})
    mc = nli.get("max_contradiction", 0)
    health["nli"] = "red" if mc > 0.7 else ("yellow" if mc > 0.3 else "green")

    return health


def compute_contributions(feat_row, importance):
    """Compute approximate feature contributions for a single call."""
    contribs = []
    for feat in feat_row.index:
        val = float(feat_row[feat])
        imp = float(importance.get(feat, 0))
        if val != 0 and imp > 0:
            contribs.append({"feature": feat, "value": round(imp * val, 4)})

    # Sort by absolute value, take top 10 positive and top 10 negative
    contribs.sort(key=lambda c: abs(c["value"]), reverse=True)
    pos = [c for c in contribs if c["value"] > 0][:10]
    neg = [c for c in contribs if c["value"] < 0][:10]
    return sorted(pos + neg, key=lambda c: c["value"], reverse=True)


def _load_state():
    """Load models, data, and precompute predictions."""
    if _state:
        return

    # Load base ML model + config
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)

    columns = config["columns"]
    threshold = config["threshold"]

    # Load meta-model if available (stacking)
    meta_model, meta_config = None, None
    meta_path = MODEL_DIR / "meta_model.pkl"
    meta_config_path = MODEL_DIR / "meta_config.json"
    if meta_path.exists() and meta_config_path.exists():
        with open(meta_path, "rb") as f:
            meta_model = pickle.load(f)
        with open(meta_config_path) as f:
            meta_config = json.load(f)

    # Load data
    train, val, test = load_all()

    # Load precomputed features
    X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")[columns]
    X_val = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")[columns]
    X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")[columns]

    # Load NLI features if available
    nli_splits = {}
    for name in ["train", "val", "test"]:
        nli_path = OUTPUT_DIR / f"nli_{name}.parquet"
        if nli_path.exists():
            nli_splits[name] = pd.read_parquet(nli_path)

    # Compute predictions for all splits
    splits = {}
    for name, df, X in [("train", train, X_train), ("val", val, X_val), ("test", test, X_test)]:
        proba = model.predict_proba(X)[:, 1]
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
            # Add NLI score if available
            if name in nli_splits:
                rec["nli_max_contradiction"] = round(float(nli_splits[name].iloc[idx]["nli_max_contradiction"]), 4)
            if name != "test":
                rec["actual_ticket"] = bool(row[TARGET])
            records.append(rec)

        splits[name] = records

    # Val metrics
    y_val = val[TARGET].astype(int).values
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= threshold).astype(int)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
    else:
        importance = pd.Series(dtype=float)

    _state.update({
        "model": model,
        "config": config,
        "meta_model": meta_model,
        "meta_config": meta_config,
        "columns": columns,
        "threshold": threshold,
        "splits": splits,
        "all_calls": splits["train"] + splits["val"] + splits["test"],
        "train_df": train,
        "val_df": val,
        "test_df": test,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "nli_splits": nli_splits,
        "importance": importance,
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

    # Model info
    model_name = "Stacked NLI + LightGBM" if s["meta_config"] else s["config"]["best_model"].upper()

    # NLI summary
    nli_summary = None
    nli_all = s["nli_splits"]
    if nli_all:
        import pandas as pd
        all_nli = pd.concat([nli_all.get("train", pd.DataFrame()), nli_all.get("val", pd.DataFrame()), nli_all.get("test", pd.DataFrame())], ignore_index=True)
        if "nli_max_contradiction" in all_nli.columns:
            scores = all_nli["nli_max_contradiction"]
            nli_summary = {
                "High (>0.7)": {"count": int((scores > 0.7).sum()), "color": "var(--destructive)"},
                "Medium (0.3-0.7)": {"count": int(((scores > 0.3) & (scores <= 0.7)).sum()), "color": "var(--chart-4)"},
                "Low (<0.3)": {"count": int((scores <= 0.3).sum()), "color": "var(--chart-1)"},
            }

    return {
        "total_calls": len(all_calls),
        "flagged_calls": len(flagged),
        "flagged_pct": round(len(flagged) / len(all_calls) * 100, 1),
        "threshold": s["threshold"],
        "val_metrics": s["val_metrics"],
        "model": model_name,
        "feature_count": len(s["columns"]),
        "has_nli": bool(s["nli_splits"]),
        "has_stacking": s["meta_config"] is not None,
        "stacked_val_f1": s["meta_config"].get("val_f1", s["meta_config"].get("stacked_val_f1")) if s["meta_config"] else None,
        "nli_summary": nli_summary,
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
        feat_row = X.loc[idx]

        # Probability
        proba = float(s["model"].predict_proba(X.loc[[idx]])[0, 1])
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

        # Heuristic rules
        rule_cols = [c for c in feat_row.index if c.startswith("rule_")]
        signals["heuristics"] = {
            "fired": {c: int(feat_row[c]) for c in rule_cols if feat_row[c] > 0},
            "total_fired": int(feat_row.get("rule_count_fired", 0)),
        }

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

        # Text features (keyword flags)
        kw_cols = [c for c in feat_row.index if c.startswith("vn_kw_")]
        signals["text_features"] = {
            "keywords_found": {c.replace("vn_kw_", ""): int(feat_row[c]) for c in kw_cols if feat_row[c] > 0},
            "validation_notes_words": int(feat_row.get("vn_word_count", 0)),
        }

        # Response checker
        signals["response_checker"] = {
            "not_in_transcript": round(float(feat_row.get("resp_not_in_transcript", 0)), 4),
            "empty_count": int(feat_row.get("resp_empty_count", 0)),
            "binary_ratio": round(float(feat_row.get("resp_binary_ratio", 0)), 4),
            "words_per_answered": round(float(feat_row.get("resp_words_per_answered", 0)), 1),
            "duration_per_answered": round(float(feat_row.get("resp_duration_per_answered", 0)), 1),
        }

        # NLI signals
        nli_data = s["nli_splits"].get(split_name)
        if nli_data is not None:
            feat_idx = df.index.get_loc(idx)
            nli_row = nli_data.iloc[feat_idx]
            signals["nli"] = {
                "max_contradiction": round(float(nli_row.get("nli_max_contradiction", 0)), 4),
                "answered_count_contradiction": round(float(nli_row.get("nli_answered_count_contradiction", 0)), 4),
                "outcome_contradiction": round(float(nli_row.get("nli_outcome_contradiction", 0)), 4),
                "completeness_contradiction": round(float(nli_row.get("nli_completeness_contradiction", 0)), 4),
                "num_contradictions": int(nli_row.get("nli_num_contradictions", 0)),
                "mean_entailment": round(float(nli_row.get("nli_mean_entailment", 1)), 4),
            }
        else:
            signals["nli"] = {
                "max_contradiction": 0, "answered_count_contradiction": 0,
                "outcome_contradiction": 0, "completeness_contradiction": 0,
                "num_contradictions": 0, "mean_entailment": 1,
            }

        # Top features for this call
        nonzero = feat_row[feat_row != 0].abs().sort_values(ascending=False)
        top_features = [{"name": k, "value": round(float(feat_row[k]), 4)} for k in nonzero.head(20).index]

        # Computed explanations
        flag_explanation = compute_flag_explanation(signals, proba, predicted)
        signal_health = compute_signal_health(signals)
        contributions = compute_contributions(feat_row, s["importance"])

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
            "signal_health": signal_health,
            "flag_explanation": flag_explanation,
            "contributions": contributions,
            "top_features": top_features,
        }

        if split_name != "test":
            result["actual_ticket"] = bool(row[TARGET])

        return result

    return {"error": "Call not found"}


@app.get("/api/importance")
def get_importance(top: int = Query(default=30)):
    imp = _state["importance"]
    features = [
        {"feature": name, "importance": round(float(val), 4)}
        for name, val in imp.head(top).items()
    ]

    # Meta-learner coefficients as structured data
    meta_learner = None
    meta = _state.get("meta_config")
    if meta and _state.get("meta_model"):
        meta_model = _state["meta_model"]
        meta_features = meta.get("meta_features", [])
        if hasattr(meta_model, "coef_") and len(meta_features) == len(meta_model.coef_[0]):
            meta_learner = [
                {"name": feat, "coefficient": round(float(coef), 4), "direction": "ticket" if coef > 0 else "clean"}
                for feat, coef in zip(meta_features, meta_model.coef_[0])
            ]

    return {"features": features, "meta_learner": meta_learner}


@app.get("/api/threshold-sweep")
def get_threshold_sweep():
    """Sweep thresholds and return metrics for interactive tuning."""
    s = _state
    y_val = s["val_df"][TARGET].astype(int).values
    val_proba = s["model"].predict_proba(s["X_val"])[:, 1]
    all_proba = np.concatenate([
        s["model"].predict_proba(s["X_train"])[:, 1],
        val_proba,
        s["model"].predict_proba(s["X_test"])[:, 1],
    ])

    sweep = []
    for t in np.arange(0.05, 0.96, 0.05):
        t = round(float(t), 2)
        val_pred = (val_proba >= t).astype(int)
        tp = int(((val_pred == 1) & (y_val == 1)).sum())
        fp = int(((val_pred == 1) & (y_val == 0)).sum())
        fn = int(((val_pred == 0) & (y_val == 1)).sum())
        tn = int(((val_pred == 0) & (y_val == 0)).sum())
        sweep.append({
            "threshold": t,
            "flagged_count": int((all_proba >= t).sum()),
            "val_f1": round(f1_score(y_val, val_pred, zero_division=0), 4),
            "val_precision": round(precision_score(y_val, val_pred, zero_division=0), 4),
            "val_recall": round(recall_score(y_val, val_pred, zero_division=0), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    meta = s.get("meta_config") or {}
    config = s.get("config", {})
    return {
        "sweep": sweep,
        "cv_folds": meta.get("cv_f1_scores", []),
        "cv_f1_mean": meta.get("cv_f1_mean"),
        "cv_f1_std": meta.get("cv_f1_std"),
        "base_model": {"val_f1": config.get("val_metrics", {}).get("f1"), "threshold": config.get("threshold")},
        "stacked_model": {"val_f1": meta.get("val_f1"), "threshold": meta.get("threshold")},
    }


def main():
    """CLI entry point: uv run dashboard"""
    print("Starting CareCaller Dashboard at http://localhost:8000")
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
